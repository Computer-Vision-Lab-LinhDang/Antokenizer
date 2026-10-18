"""v6 encoder: a ViT/16 (DINOv3 ViT-S or SigLIP2 ViT-B layout) over a 2-frame tube, run per camera, with
depth-hypothesis and camera-motion heads.

The network sees pixels, camera-frame patch rays, normalised intrinsics and the tube dt only: no rig pose, no
camera index, no measured depth. Measured depth is a tag computed outside the network (rays.patch_depth_stats).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .config import V6Config
from .contracts import EncoderOutput, Observation
from .rays import fourier_log_depth, patch_depth_stats, patch_rays

DINOV3_REPO = 'facebook/dinov3-vits16-pretrain-lvd1689m'
SIGLIP2_REPO = 'google/siglip2-base-patch16-256'
SIGLIP2_REPOS = {'siglip2_b16': SIGLIP2_REPO, 'siglip2_so400m16': 'google/siglip2-so400m-patch16-256'}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ROPE_THETA = 100.0
LAYER_NORM_EPS = 1e-5
LAYERSCALE_INIT = 1e-5
# Per-backbone layout: pixel norm, LayerNorm eps, GELU form, LayerScale init, positions (RoPE or learned table).
LAYOUTS = {
    'dinov3_vits16': dict(mean=IMAGENET_MEAN, std=IMAGENET_STD, eps=1e-5, gelu='none', ls_init=LAYERSCALE_INIT,
                          rope=True),
    'siglip2_b16': dict(mean=(.5, .5, .5), std=(.5, .5, .5), eps=1e-6, gelu='tanh', ls_init=1.0, rope=False),
}
LAYOUTS['siglip2_so400m16'] = LAYOUTS['siglip2_b16']
DUV_MAX = 8.0                                  # hypothesis offset within the 16 px patch
LOG_Z_RANGE = (math.log(0.5), math.log(250.))  # metres
LOG_SIGMA_MAX = 5.0


def rope_tables(hp: int, wp: int, head_dim: int, device=None) -> tuple[Tensor, Tensor]:
    """DINOv3 2D RoPE (cos, sin) [hp*wp, head_dim]: patch centres in [-1, 1] per axis, row-major, (y, x) bands."""
    inv_freq = ROPE_THETA ** -(torch.arange(head_dim // 4, device=device, dtype=torch.float32) * 4 / head_dim)
    ch = (torch.arange(hp, device=device, dtype=torch.float32) + 0.5) / hp
    cw = (torch.arange(wp, device=device, dtype=torch.float32) + 0.5) / wp
    coords = torch.stack(torch.meshgrid(ch, cw, indexing='ij'), -1).reshape(-1, 2) * 2 - 1
    angle = (2 * math.pi * coords[:, :, None] * inv_freq).flatten(1).tile(2)
    return angle.cos(), angle.sin()


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        m, t, e = x.shape
        q, k, v = self.qkv(x).reshape(m, t, 3, self.heads, e // self.heads).permute(2, 0, 3, 1, 4)
        if cos is not None:
            p = t - cos.shape[0]  # prefix (cls + registers) gets no RoPE
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q = torch.cat((q[:, :, :p], q[:, :, p:] * cos + _rotate_half(q[:, :, p:]) * sin), 2)
            k = torch.cat((k[:, :, :p], k[:, :, p:] * cos + _rotate_half(k[:, :, p:]) * sin), 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(m, t, e))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = LAYERSCALE_INIT):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, gelu: str = 'none'):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gelu = gelu

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate=self.gelu))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float = LAYER_NORM_EPS, gelu: str = 'none',
                 ls_init: float = LAYERSCALE_INIT, hidden: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim, ls_init)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, hidden or 4 * dim, gelu)
        self.ls2 = LayerScale(dim, ls_init)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), cos, sin))
        return x + self.ls2(self.mlp(self.norm2(x)))


class AttentionPool(nn.Module):
    """One learned query attending over a camera's tokens."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        m, t, e = x.shape
        h = self.heads
        k, v = self.kv(self.norm(x)).reshape(m, t, 2, h, e // h).permute(2, 0, 3, 1, 4)
        q = self.q(self.query).reshape(1, 1, h, e // h).transpose(1, 2).expand(m, h, 1, e // h).contiguous()
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))
        return self.proj(out.reshape(m, e))


class V6Encoder(nn.Module):
    def __init__(self, cfg: V6Config):
        super().__init__()
        self.cfg = cfg
        e, p, d = cfg.encoder_width, cfg.patch_size, cfg.depth_hypotheses
        self.layout = lay = LAYOUTS[cfg.backbone]
        self.prefix = 1 + cfg.encoder_registers
        self.patch_embed = nn.Conv3d(3, e, kernel_size=(cfg.tubelet, p, p), stride=(cfg.tubelet, p, p))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, e))
        self.register_tokens = nn.Parameter(torch.zeros(1, cfg.encoder_registers, e))
        # ray (3) + fx/W, fy/H (2) + Fourier(log(dt + 1e-3))
        self.geo_embed = nn.Sequential(nn.Linear(5 + 2 * cfg.fourier_bands, e), nn.GELU(), nn.Linear(e, e))
        self.blocks = nn.ModuleList(Block(e, cfg.encoder_heads, lay['eps'], lay['gelu'], lay['ls_init'],
                                          cfg.encoder_mlp_hidden) for _ in range(cfg.encoder_layers))
        self.norm = nn.LayerNorm(e, eps=lay['eps'])
        # learned absolute positions for layouts without RoPE (patch tokens only)
        self.pos_embed = None if lay['rope'] else nn.Parameter(torch.zeros(1, cfg.patches_per_camera, e))
        self.depth_head = nn.Sequential(nn.LayerNorm(e), nn.Linear(e, e), nn.GELU(), nn.Linear(e, d * 5))
        self.motion_pool = AttentionPool(e, cfg.encoder_heads)
        self.motion_head = nn.Sequential(nn.Linear(e, e), nn.GELU(), nn.Linear(e, 12))
        self.register_buffer('pixel_mean', torch.tensor(lay['mean']).view(3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(lay['std']).view(3, 1, 1), persistent=False)
        if lay['rope']:
            cos, sin = rope_tables(cfg.image_height // p, cfg.image_width // p, e // cfg.encoder_heads)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None
        self.grad_checkpointing = False  # per-block activation checkpointing while training
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, LayerScale):
                module.gamma.fill_(self.layout['ls_init'])
        self.patch_embed.reset_parameters()
        self.patch_embed.weight[:, :, :-1].zero_()  # earlier frames start at 0: a single-image model
        for t in (self.cls_token, self.register_tokens, self.motion_pool.query):
            nn.init.normal_(t, std=0.02)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # depth hypotheses start spread over 2..60 m
        d = self.cfg.depth_hypotheses
        lo, hi = LOG_Z_RANGE
        frac = (torch.linspace(math.log(2.), math.log(60.), d) - lo) / (hi - lo)
        bias = self.depth_head[-1].bias.view(d, 5)
        bias[:, 2] = torch.logit(frac)
        last = self.motion_head[-1]
        last.weight.zero_()
        last.bias[6:].fill_(-2.)  # log-std

    def forward(self, obs: Observation) -> EncoderOutput:
        cfg = self.cfg
        images = obs.images
        if images.dim() != 6:
            raise ValueError('images must be [B,V,T,3,H,W]')
        b, v, t, c, h, w = images.shape
        if not 1 <= v <= cfg.max_cameras:
            raise ValueError(f'camera count {v} outside 1..{cfg.max_cameras}')
        if c != 3 or (h, w) != (cfg.image_height, cfg.image_width):
            raise ValueError(f'images must be 3 x {cfg.image_height} x {cfg.image_width}, got {c} x {h} x {w}')
        if t == 1 and cfg.tubelet > 1:
            images = images.expand(b, v, cfg.tubelet, c, h, w)  # still image: repeat, dt = 0
        elif t != cfg.tubelet:
            raise ValueError(f'tube length {t} != cfg.tubelet {cfg.tubelet}')
        if not images.is_floating_point():
            images = images.float() / 255
        n, e, dev = cfg.patches_per_camera, cfg.encoder_width, images.device
        cam_ok = obs.camera_valid.to(dev).bool()

        # Dead cameras: zero pixels, nominal intrinsics. Cameras are independent batch rows from here on.
        x = torch.where(cam_ok[:, :, None, None, None, None], images, 0.)
        x = (x - self.pixel_mean) / self.pixel_std
        x = x.reshape(b * v, cfg.tubelet, 3, h, w).transpose(1, 2)
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)  # [BV,N,E]

        nominal = torch.tensor([[w, 0., w / 2], [0., w, h / 2], [0., 0., 1.]], device=dev)
        K = torch.where(cam_ok[..., None, None], obs.intrinsics.to(dev).float(), nominal)
        rays = patch_rays(K, cfg)  # [B,V,N,3] fp32
        dt = torch.as_tensor(obs.tube_dt, device=dev).float().reshape(b).clamp_min(0)
        dt_feat = fourier_log_depth(torch.log(dt + 1e-3), cfg.fourier_bands)  # [B,2F]
        focal = torch.stack((K[..., 0, 0] / w, K[..., 1, 1] / h), -1)  # [B,V,2]
        geo = torch.cat((rays, focal[:, :, None].expand(b, v, n, 2),
                         dt_feat[:, None, None].expand(b, v, n, dt_feat.shape[-1])), -1)
        tok = tok + self.geo_embed(geo.reshape(b * v, n, -1)).to(tok.dtype)
        if self.pos_embed is not None:
            tok = tok + self.pos_embed.to(tok.dtype)

        z = torch.cat((self.cls_token.expand(b * v, -1, -1).to(tok.dtype),
                       self.register_tokens.expand(b * v, -1, -1).to(tok.dtype), tok), 1)
        for block in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                z = checkpoint(block, z, self.rope_cos, self.rope_sin, use_reentrant=False)
            else:
                z = block(z, self.rope_cos, self.rope_sin)
        z = self.norm(z)
        patches = z[:, self.prefix:]

        raw = self.depth_head(patches).float().reshape(b, v, n, cfg.depth_hypotheses, 5)
        lo, hi = LOG_Z_RANGE
        depth_hyp = torch.cat((DUV_MAX * torch.tanh(raw[..., :2]),
                               lo + (hi - lo) * torch.sigmoid(raw[..., 2:3]),
                               LOG_SIGMA_MAX * torch.tanh(raw[..., 3:4] / LOG_SIGMA_MAX),
                               raw[..., 4:5]), -1)
        motion = self.motion_head(self.motion_pool(z)).float().reshape(b, v, 12)

        tokens = torch.where(cam_ok[:, :, None, None], patches.reshape(b, v, n, e), 0.)
        depth_hyp = torch.where(cam_ok[:, :, None, None, None], depth_hyp, 0.)
        motion = torch.where(cam_ok[:, :, None], motion, 0.)
        valid = cam_ok[:, :, None].expand(b, v, n)

        # Measured depth: a tag next to the tokens, never an input of the network.
        measured, measured_valid = patch_depth_stats(obs.depth.to(dev), obs.depth_valid.to(dev), cfg)
        measured_valid = measured_valid & valid
        measured = torch.where(measured_valid[..., None], measured, 0.)
        return EncoderOutput(tokens=tokens, valid=valid, depth_hyp=depth_hyp, ray_cam=rays, camera_motion=motion,
                             measured_depth=measured, measured_valid=measured_valid)


# ---- DINOv3 weights -------------------------------------------------------------------------------------------
# HF key names follow transformers' modeling_dinov3_vit.py
# (https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py):
# embeddings.{cls_token,register_tokens,patch_embeddings}, [model.]layer.i.{norm1,norm2,attention.{q,k,v,o}_proj,
# layer_scale{1,2}.lambda1,mlp.{up,down}_proj}, norm. Original Meta checkpoints
# (https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py) use this module's
# names, with storage_tokens, patch_embed.proj and attn.qkv.bias_mask.

_PREFIXES = ('module.', 'model.', 'backbone.', 'dinov3.', 'encoder.', 'teacher.')
_HF_LAYER = {'norm1.': 'norm1.', 'norm2.': 'norm2.', 'attention.o_proj.': 'attn.proj.',
             'layer_scale1.lambda1': 'ls1.gamma', 'layer_scale2.lambda1': 'ls2.gamma',
             'mlp.up_proj.': 'mlp.fc1.', 'mlp.down_proj.': 'mlp.fc2.'}
_PATCH_2D = 'patch_embed.weight@2d'
_SAFETENSORS_DTYPES = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
                       'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16, 'I8': torch.int8,
                       'U8': torch.uint8, 'BOOL': torch.bool}


def _read_safetensors(path: Path, keep=None) -> dict:
    """Read a .safetensors file; `keep(name)` filters tensors (only those are read from disk)."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework='pt') as f:
            return {k: f.get_tensor(k) for k in f.keys() if keep is None or keep(k)}
    except ImportError:
        pass
    # Minimal reader: u64 header length, JSON header, raw little-endian data.
    out = {}
    with open(path, 'rb') as f:
        size = int.from_bytes(f.read(8), 'little')
        header = json.loads(f.read(size))
        base = 8 + size
        for name, info in header.items():
            if name == '__metadata__' or (keep is not None and not keep(name)):
                continue
            dtype = _SAFETENSORS_DTYPES[info['dtype']]
            start, end = info['data_offsets']
            f.seek(base + start)
            data = bytearray(f.read(end - start))
            flat = torch.frombuffer(data, dtype=dtype) if data else torch.empty(0, dtype=dtype)
            out[name] = flat.reshape(info['shape']).clone()
    return out


def _read_file(path: Path) -> dict:
    if path.suffix == '.safetensors':
        return _read_safetensors(path)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('teacher', 'model', 'state_dict'):
        if isinstance(sd, dict) and isinstance(sd.get(key), dict):
            sd = sd[key]
    return sd if isinstance(sd, dict) else {}


def _read_hub(repo: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download
        return _read_file(Path(hf_hub_download(repo, 'model.safetensors')))
    except Exception:
        pass
    try:
        from transformers import AutoModel
        return AutoModel.from_pretrained(repo).state_dict()
    except Exception:
        return {}


def _read_state_dict(source) -> dict:
    source = DINOV3_REPO if source is None else str(source)
    path = Path(source).expanduser()
    if path.is_dir():
        files = sorted(path.glob('*.safetensors'))
        if not files:
            files = [f for pattern in ('pytorch_model.bin', '*.pth', '*.pt', '*.bin') for f in sorted(path.glob(pattern))][:1]
        sd = {}
        for f in files:
            sd.update(_read_file(f))
        return sd
    if path.is_file():
        return _read_file(path)
    if re.fullmatch(r'[\w][\w.-]*/[\w][\w.-]*', source):  # HF repo id, never a missing local path
        return _read_hub(source)
    return {}


def _to_local(sd: dict) -> dict:
    """Map an HF or Meta DINOv3 state dict onto V6Encoder names (patch weight as 2D under _PATCH_2D)."""
    clean = {}
    for key, value in sd.items():
        if not isinstance(value, Tensor):
            continue
        while key.startswith(_PREFIXES):
            key = key.split('.', 1)[1]
        clean[key] = value
    out, qkv = {}, {}
    for key, value in clean.items():
        if key in ('embeddings.cls_token', 'cls_token'):
            out['cls_token'] = value
        elif key in ('embeddings.register_tokens', 'storage_tokens', 'register_tokens'):
            out['register_tokens'] = value
        elif key in ('embeddings.patch_embeddings.weight', 'patch_embed.proj.weight'):
            out[_PATCH_2D] = value
        elif key in ('embeddings.patch_embeddings.bias', 'patch_embed.proj.bias'):
            out['patch_embed.bias'] = value
        elif key in ('norm.weight', 'norm.bias'):
            out[key] = value
        elif (m := re.fullmatch(r'layer\.(\d+)\.(.+)', key)):
            i, rest = m.groups()
            if (a := re.fullmatch(r'attention\.([qkv])_proj\.(weight|bias)', rest)):
                qkv.setdefault((i, a.group(2)), {})[a.group(1)] = value
                continue
            for src, dst in _HF_LAYER.items():
                if rest.startswith(src):
                    out[f'blocks.{i}.{dst}{rest[len(src):]}'] = value
                    break
        elif (m := re.fullmatch(r'blocks\.(\d+)\.(.+)', key)) and not key.endswith('bias_mask'):
            if key.endswith('attn.qkv.bias') and f'{key}_mask' in clean:
                value = value * torch.nan_to_num(clean[f'{key}_mask'].to(value.dtype))
            out[key] = value
    for (i, kind), parts in qkv.items():
        if 'q' in parts and 'v' in parts:
            k = parts.get('k', torch.zeros_like(parts['q']))  # HF ViT-S has no key bias
            out[f'blocks.{i}.attn.qkv.{kind}'] = torch.cat((parts['q'], k, parts['v']), 0)
    return out


def load_dinov3(encoder: V6Encoder, source=DINOV3_REPO) -> tuple[list[str], list[str]]:
    """Best-effort DINOv3 ViT-S/16 load: HF repo id, local dir, .safetensors or .pth/.bin file.

    The 2D patch weight goes into the last-frame slice of the tube Conv3d (earlier frames stay 0). Never raises:
    returns ([], all keys) when weights or libraries are unavailable, so callers keep the random init.
    """
    own = encoder.state_dict()
    loaded = []
    try:
        mapped = _to_local(_read_state_dict(source))
        with torch.no_grad():
            for key, value in mapped.items():
                if key == _PATCH_2D:
                    weight = encoder.patch_embed.weight
                    if value.shape == weight[:, :, -1].shape:
                        weight[:, :, -1].copy_(value)
                        weight[:, :, :-1].zero_()
                        loaded.append('patch_embed.weight')
                elif key in own and own[key].shape == value.shape:
                    own[key].copy_(value)
                    loaded.append(key)
    except Exception:
        pass
    loaded = sorted(set(loaded))
    return loaded, [k for k in own if k not in loaded]


# ---- SigLIP2 weights ------------------------------------------------------------------------------------------
# HF SiglipVisionModel names (https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/
# modeling_siglip.py): vision_model.embeddings.{patch_embedding,position_embedding}, vision_model.encoder.layers.i.
# {layer_norm1,self_attn.{q,k,v,out}_proj,layer_norm2,mlp.fc1,mlp.fc2}, vision_model.post_layernorm. Fixed-resolution
# SigLIP2 checkpoints (e.g. google/siglip2-base-patch16-256) use this layout. No cls/register tokens, no LayerScale.

_SIGLIP_LAYER = {'layer_norm1.': 'norm1.', 'layer_norm2.': 'norm2.', 'self_attn.out_proj.': 'attn.proj.',
                 'mlp.fc1.': 'mlp.fc1.', 'mlp.fc2.': 'mlp.fc2.'}


def _siglip_state_dict(source) -> dict:
    source = SIGLIP2_REPO if source is None else str(source)
    path = Path(source).expanduser()
    # vision tower only (full SiglipModel files prefix it with vision_model.; text tower and pooling head skipped)
    keep = lambda k: k.split('vision_model.', 1)[-1].startswith(('embeddings.', 'encoder.', 'post_layernorm.'))
    if path.is_dir():
        files = sorted(path.glob('*.safetensors'))
        if files:
            sd = {}
            for f in files:
                sd.update(_read_safetensors(f, keep))
            return sd
    if path.is_file():
        return _read_safetensors(path, keep) if path.suffix == '.safetensors' else _read_file(path)
    if re.fullmatch(r'[\w][\w.-]*/[\w][\w.-]*', source):
        try:
            from huggingface_hub import hf_hub_download
            return _read_safetensors(Path(hf_hub_download(source, 'model.safetensors')), keep)
        except Exception:
            return {}
    return {}


def _siglip_to_local(sd: dict, grid: tuple[int, int]) -> dict:
    """Map SigLIP vision weights onto V6Encoder names; the position table is resized to the patch grid."""
    out, qkv = {}, {}
    for key, value in sd.items():
        key = key.split('vision_model.', 1)[-1]
        if key == 'embeddings.patch_embedding.weight':
            out[_PATCH_2D] = value
        elif key == 'embeddings.patch_embedding.bias':
            out['patch_embed.bias'] = value
        elif key == 'embeddings.position_embedding.weight':
            n, e = value.shape
            side = int(round(math.sqrt(n)))
            table = value.float().reshape(1, side, side, e).permute(0, 3, 1, 2)
            table = F.interpolate(table, size=grid, mode='bicubic', align_corners=False)
            out['pos_embed'] = table.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], e)
        elif key.startswith('post_layernorm.'):
            out['norm.' + key.split('.', 1)[1]] = value
        elif (m := re.fullmatch(r'encoder\.layers\.(\d+)\.(.+)', key)):
            i, rest = m.groups()
            if (a := re.fullmatch(r'self_attn\.([qkv])_proj\.(weight|bias)', rest)):
                qkv.setdefault((i, a.group(2)), {})[a.group(1)] = value
                continue
            for src, dst in _SIGLIP_LAYER.items():
                if rest.startswith(src):
                    out[f'blocks.{i}.{dst}{rest[len(src):]}'] = value
                    break
    for (i, kind), parts in qkv.items():
        if len(parts) == 3:
            out[f'blocks.{i}.attn.qkv.{kind}'] = torch.cat((parts['q'], parts['k'], parts['v']), 0)
    return out


def _copy_mapped(encoder: V6Encoder, mapped: dict) -> list[str]:
    own = encoder.state_dict()
    loaded = []
    with torch.no_grad():
        for key, value in mapped.items():
            if key == _PATCH_2D:
                weight = encoder.patch_embed.weight
                if value.shape == weight[:, :, -1].shape:
                    weight[:, :, -1].copy_(value)
                    weight[:, :, :-1].zero_()
                    loaded.append('patch_embed.weight')
            elif key in own and own[key].shape == value.shape:
                own[key].copy_(value)
                loaded.append(key)
    return sorted(set(loaded))


def load_siglip2(encoder: V6Encoder, source=None) -> tuple[list[str], list[str]]:
    """Best-effort SigLIP2 /16 vision-tower load (HF repo id, local dir or .safetensors). Never raises."""
    if encoder.cfg.backbone not in SIGLIP2_REPOS:
        raise ValueError(f'encoder layout {encoder.cfg.backbone!r} is not a SigLIP2 layout')
    source = SIGLIP2_REPOS[encoder.cfg.backbone] if source is None else source
    cfg = encoder.cfg
    loaded = []
    try:
        grid = (cfg.image_height // cfg.patch_size, cfg.image_width // cfg.patch_size)
        loaded = _copy_mapped(encoder, _siglip_to_local(_siglip_state_dict(source), grid))
    except Exception:
        pass
    return loaded, [k for k in encoder.state_dict() if k not in loaded]


def load_backbone(encoder: V6Encoder, source=None) -> tuple[list[str], list[str]]:
    """Load pretrained weights for encoder.cfg.backbone."""
    if encoder.cfg.backbone in SIGLIP2_REPOS:
        return load_siglip2(encoder, source)
    return load_dinov3(encoder, DINOV3_REPO if source is None else source)
