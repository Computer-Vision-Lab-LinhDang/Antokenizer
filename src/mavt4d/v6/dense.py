"""Dense feature targets: a frozen DINOv3 ViT/16 on the target frame, sampled at a ray's pixel.

The teacher is a plain single-image DINOv3 (patch conv, cls + registers, RoPE blocks, final norm), loaded with the
same key mapping as the encoder (encoder._to_local). It is kept out of V6Model: no gradient, not in checkpoints.
Its patch features are the target of the rays the decoder renders (losses.render_loss, cfg.dense_dim) and of the
copy-last / warp baselines in evaluate.py. Patch (i, j) sits at pixel ((j + .5) p - .5, (i + .5) p - .5) (pixel
centres at integers), so a pixel reads the 4 nearest patches bilinearly.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import BACKBONES, V6Config
from .encoder import _PATCH_2D, DINOV3_REPOS, LAYOUTS, Block, _read_state_dict, _to_local, rope_tables


class DenseTeacher(nn.Module):
    """Images [N,3,H,W] (float in [0, 1] or uint8) -> final-norm patch features [N,D,H/p,W/p]."""

    def __init__(self, cfg: V6Config):
        super().__init__()
        name, p = cfg.dense_teacher, cfg.patch_size
        shape, lay = BACKBONES[name], LAYOUTS[name]
        e, heads = shape['encoder_width'], shape['encoder_heads']
        self.grid = (cfg.image_height // p, cfg.image_width // p)
        self.prefix = 1 + shape['encoder_registers']
        self.patch_embed = nn.Conv2d(3, e, kernel_size=p, stride=p)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, e))
        self.register_tokens = nn.Parameter(torch.zeros(1, shape['encoder_registers'], e))
        self.blocks = nn.ModuleList(Block(e, heads, lay['eps'], lay['gelu'], lay['ls_init'],
                                          shape.get('encoder_mlp_hidden', 0)) for _ in range(shape['encoder_layers']))
        self.norm = nn.LayerNorm(e, eps=lay['eps'])
        self.register_buffer('pixel_mean', torch.tensor(lay['mean']).view(3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(lay['std']).view(3, 1, 1), persistent=False)
        cos, sin = rope_tables(*self.grid, e // heads)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # frozen: always in eval mode

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        x = images.float() / 255 if images.dtype == torch.uint8 else images.float()
        tok = self.patch_embed((x - self.pixel_mean) / self.pixel_std).flatten(2).transpose(1, 2)
        n = tok.shape[0]
        z = torch.cat((self.cls_token.expand(n, -1, -1).to(tok.dtype),
                       self.register_tokens.expand(n, -1, -1).to(tok.dtype), tok), 1)
        for block in self.blocks:
            z = block(z, self.rope_cos, self.rope_sin)
        patches = self.norm(z)[:, self.prefix:]
        return patches.transpose(1, 2).unflatten(-1, self.grid)


def load_teacher(cfg: V6Config, source=None) -> tuple[DenseTeacher, int]:
    """DenseTeacher with every weight loaded from source (HF repo id, local dir or file; default: the teacher's HF
    repo). Raises when any is missing: a random teacher must not define the target. -> (teacher, tensors loaded)."""
    teacher = DenseTeacher(cfg)
    mapped = _to_local(_read_state_dict(DINOV3_REPOS[cfg.dense_teacher] if source is None else source))
    if _PATCH_2D in mapped:
        mapped['patch_embed.weight'] = mapped.pop(_PATCH_2D)
    own = teacher.state_dict()
    loaded = {k for k, v in mapped.items() if k in own and own[k].shape == v.shape}
    missing = sorted(set(own) - loaded)
    if missing:
        raise ValueError(f'dense teacher {cfg.dense_teacher} from {source!r}: {len(missing)} of {len(own)} tensors '
                         f'not found, e.g. {missing[:3]}')
    teacher.load_state_dict({k: mapped[k] for k in own})
    return teacher, len(loaded)


def teacher_features(teacher: DenseTeacher, images: Tensor, valid: Tensor) -> Tensor:
    """images [B,V,3,H,W] of cameras valid [B,V] -> features [B,V,D,h,w] (zeros for invalid cameras); only the
    valid cameras run through the teacher."""
    b, v = images.shape[:2]
    ok = valid.bool().flatten()
    out = images.new_zeros(b * v, teacher.norm.normalized_shape[0], *teacher.grid, dtype=torch.float32)
    if bool(ok.any()):
        out[ok] = teacher(images.flatten(0, 1)[ok]).float()
    return out.unflatten(0, (b, v))


def sample_features(features: Tensor, pixels: Tensor, height: int, width: int) -> Tensor:
    """features [B,V,D,h,w] at pixels [B,V,R,2] (u, v; centres at integers) of an height x width image -> [B,V,R,D],
    bilinear between patch centres (border patches past them)."""
    b, v, d = features.shape[:3]
    grid = (pixels.float() + .5) / pixels.new_tensor([width, height]).float() * 2 - 1
    out = F.grid_sample(features.flatten(0, 1).float(), grid.flatten(0, 1)[:, None], mode='bilinear',
                        padding_mode='border', align_corners=False)                   # [BV,D,1,R]
    return out[:, :, 0].transpose(1, 2).unflatten(0, (b, v))


def cosine(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity over the last axis, fp32."""
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-6)
