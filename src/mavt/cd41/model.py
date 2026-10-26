"""MAVT C-D 4.1: unified tokenizer with geometric transport and Δ-Tucker (design and module spec of 26/10).

forward(x, modality, cam) -> dict(loss, parts, diag[, outputs]), the interface of the v0 trainer:

  image   [B, 3, H, W]      anchor only: c, rho, kappa_0 (estimated, fills the schema; no target is made)
  video   [B, 3, T, H, W]   segments of K frames; frame 0 is the anchor; camera (xi, kappa) and motion M
                            estimated; median anchor depth 1; targets coded in order
  threed  [B, 3, V, H, W]   static_views: kappa and xi = log(E_i E_0^-1) copied from cam, M = 0 exactly; views
                            coded in parallel with time offset 0, so their order is not a signal
  world   [B, 3, T, H, W]   clips with known cameras (OmniWorld): coded as video; the given cameras and depth
                            only enter the loss (world_camera = "given": copied into the state instead)

Encoder side (may read the observations):
  FrameEncoder     frozen DINOv3 read at 4 depths (Mem_i, 4 x 256 read-only tokens) + pixel stem -> cells
  stream C         observation 0 and Mem_0 -> c, rho, kappa_0
  stream D         observation i, Mem_i and the context decoded from the public state S_{i-1} (video: H_0 and
                   the F^ of the earlier targets with their time offsets; static views: H_0 only, dt = 0);
                   274 tokens = 256 cells + camera + 16 motion + a -> level 1 (xi, kappa, M, a)
  refiner          level 2: alignment feedback F_i - sg(H^base(1)) on the cells -> increments; static views
                   (and given cameras) keep the geometry and refine a only
  Enc_R            [LN e^F ; LN PatchEmbed(e^RGB) ; sg H^base ; MetaEmbed(h, xi, kappa)] read by 80 ordered
                   queries with slot-causal self-attention -> r_i
No keys pass from stream C to stream D: a target sees the anchor only through the state.

Decoder side (reads only the state Z and the shared weights), the same functions in forward and decode(Z):
    H_0 = F_C(c),  U = F_U(c, grid),  beta = softmax(W_beta c)
    H_i^base = W_i H_0 + [(W_i U) B(a_i)] Q,   B(a)[p, q] = sum_b G[p, b, q] a[b],   W_i sparse (9 per row)
    F^_i = H_i^base + 1[L_i > 0] Dec_R(H_i^base, r_i[:L_i], MetaEmbed(h, xi_i, M_i, kappa_i, a_i)),  X^_i = Dec_RGB(F^_i)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (gradient checkpointing of the trunk)

from ..cd32 import geometry as geo
from ..cd32.depth import cell_log_inv_depth, silog
from ..cd32.encoder import FrameEncoder
from ..cd32.layers import MLP, StreamBlock, attention, merge_heads, split_heads
from ..cd32.losses import edge_aware_smooth, psnr, rgb_l1
from ..cd32.regs import _pearson
from ..cd32.vanilla import DecBlock
from ..dino_v0.camera import ZeroLinear
from ..dino_v0.losses import ignore_lpips_keys
from . import state as st
from . import transport as tr
from .config import CD41Config

MODALITIES = ("image", "video", "threed", "world")


def _blocks(dim: int, heads: int, n: int, mlp_ratio: float = 4.0, ctx: bool = False) -> nn.ModuleList:
    return nn.ModuleList([StreamBlock(dim, heads, mlp_ratio, ctx=ctx) for _ in range(n)])


def _run(blocks: nn.ModuleList, h: torch.Tensor, m: Optional[torch.Tensor] = None, checkpoint: bool = False):
    for blk in blocks:
        if checkpoint and torch.is_grad_enabled():
            h = torch.utils.checkpoint.checkpoint(blk, h, None, m, use_reentrant=False)
        else:
            h = blk(h, m=m)
    return h


def _param(*shape) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape) * 0.02)


def _log_rms(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log RMS over `dim`, kept as a size-1 axis: the scale that a LayerNorm of x throws away."""
    return 0.5 * torch.log(x.float().pow(2).mean(dim, keepdim=True) + 1e-8)


def time_features(dt: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Time offsets [...] (frame steps) -> [..., 2 n_freq], sin / cos at periods of 16, 8, 4, 2 ... steps."""
    w = math.pi / 8 * 2.0 ** torch.arange(n_freq, device=dt.device, dtype=torch.float32)
    a = dt.float().unsqueeze(-1) * w
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


def header_features(cfg: CD41Config, h: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """Header h [B', 16] and target index i [B'] (1 .. 3) -> [B', 5 + 2 time_freqs]: mode one-hot, time offset
    of target i, coordinate mode, camera mode. View ids and L_target are left out (order and length are not
    content; Enc_R must not know L, so that r[:L] is the same for every L)."""
    mode = F.one_hot(h[:, 0].round().long().clamp(0, 2), 3).float()
    dt = torch.gather(h[:, 2:6].float(), 1, i.long().view(-1, 1)).squeeze(1)
    return torch.cat([mode, time_features(dt, cfg.time_freqs), h[:, 13:15].float()], -1)


def header_dim(cfg: CD41Config) -> int:
    return 5 + 2 * cfg.time_freqs


class MetaEmbed(nn.Module):
    """Metadata vector (header features, camera, motion, coefficients) -> one embedding added to every cell."""

    def __init__(self, n_in: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.net(m)


# ---------------------------------------------------------------------------------------------- encoder side
class Stream(nn.Module):
    """One stream of the trunk: blocks over its own tokens, each also reading read-only context keys."""

    def __init__(self, cfg: CD41Config):
        super().__init__()
        self.blocks = _blocks(cfg.dim, cfg.heads, cfg.layers, cfg.mlp_ratio, ctx=True)
        self.norm = nn.LayerNorm(cfg.dim)
        self.use_checkpoint = False

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.norm(_run(self.blocks, h, m, self.use_checkpoint and self.training))


class TwoStreamTrunk(nn.Module):
    """Stream C (anchor) and stream D (targets), separate weights. Both read the DINO memory of their own
    observation; stream D also reads the context decoded from S_{i-1}."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        d, N = cfg.dim, cfg.n_cells
        self.cfg = cfg
        self.pos_emb = _param(1, N, d)
        self.kappa0_q = _param(1, 1, d)
        self.cam_q, self.motion_q, self.a_q = _param(1, 1, d), _param(1, cfg.n_motion, d), _param(1, 1, d)
        self.src_emb = _param(2, d)                    # read-only context: 0 DINO memory, 1 decoded from the state
        self.time = nn.Linear(2 * cfg.time_freqs, d)
        self.ctx_in = nn.Linear(feat_dim, d)
        self.cam_in = nn.Linear(10, d)                 # known camera (xi, kappa): static views, given cameras
        self.c, self.d = Stream(cfg), Stream(cfg)

    def memory(self, mem: torch.Tensor) -> torch.Tensor:
        """DINO memory [B', L, N, dim] -> [B', L N, dim]."""
        return (mem + self.pos_emb.unsqueeze(1) + self.src_emb[0]).flatten(1, 2)

    def context(self, feats: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Features decoded from S_{i-1} [B, m, N, d_feat] and their time offsets [B, m] -> [B, m N, dim]."""
        x = self.ctx_in(feats) + self.pos_emb.unsqueeze(1) + self.src_emb[1]
        return (x + self.time(time_features(dt, self.cfg.time_freqs)).unsqueeze(2)).flatten(1, 2)

    def anchor(self, x0: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """x0 [B, N, dim], mem [B, L, N, dim] -> [B, N + 1, dim] (cells, kappa_0)."""
        B = x0.shape[0]
        h = torch.cat([x0 + self.pos_emb, self.kappa0_q.expand(B, -1, -1).to(x0.dtype)], 1)
        return self.c(h, self.memory(mem).to(h.dtype))

    def target(self, x0, mem, ctx, dt, cam=None) -> torch.Tensor:
        """x0 [B', N, dim], mem [B', L, N, dim], ctx [B', Nc, dim], dt [B'], cam [B', 10] or None ->
        [B', N + 2 + n_motion, dim] (cells, camera, motion, a)."""
        B = x0.shape[0]
        cq = self.cam_q.expand(B, -1, -1)
        if cam is not None:
            cq = cq + self.cam_in(cam.float()).unsqueeze(1)
        h = torch.cat([x0 + self.pos_emb, cq.to(x0.dtype), self.motion_q.expand(B, -1, -1).to(x0.dtype),
                       self.a_q.expand(B, -1, -1).to(x0.dtype)], 1)
        h = h + self.time(time_features(dt, self.cfg.time_freqs)).unsqueeze(1).to(h.dtype)
        return self.d(h, torch.cat([self.memory(mem), ctx.to(mem.dtype)], 1).to(h.dtype))


class CameraHead(nn.Module):
    """Token -> kappa (bounded FOV, square pixels, principal point within pp_range of the centre), through
    three logits so that the refiner can add to them."""

    def __init__(self, cfg: CD41Config):
        super().__init__()
        self.cfg = cfg
        self.lin = ZeroLinear(cfg.dim, 3)
        span = cfg.fov_max - cfg.fov_min
        p0 = (cfg.fov_init - cfg.fov_min) / span if span > 1e-6 else 0.5
        p0 = min(max(p0, 1e-4), 1 - 1e-4)
        self.bias = math.log(p0 / (1 - p0))

    def logits(self, tok: torch.Tensor) -> torch.Tensor:
        return self.lin(tok).float()

    def kappa(self, o: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        fov = cfg.fov_min + (cfg.fov_max - cfg.fov_min) * torch.sigmoid(o[..., 0] + self.bias)
        return tr.kappa_from_fov(fov, cfg.pp_range * torch.tanh(o[..., 1:]))

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        return self.kappa(self.logits(tok))


class Refiner(nn.Module):
    """Level 2 of the GTA refinement. The cells of the stream D tokens get the alignment feedback
    delta = F_i - sg(H^base(1)) (LN + its log RMS) and the level-1 support; increments come from zero heads."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        d = cfg.dim
        self.norm_e, self.proj, self.mag = nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d), nn.Linear(1, d)
        self.sup_emb = _param(d)
        self.blocks = _blocks(d, cfg.heads, cfg.refine_layers, cfg.mlp_ratio)
        self.norm = nn.LayerNorm(d)
        self.d_pose, self.d_kappa, self.d_motion = ZeroLinear(d, 6), ZeroLinear(d, 3), ZeroLinear(d, 3)
        self.d_a = ZeroLinear(d, cfg.rank_a) if cfg.tucker else None

    def forward(self, h: torch.Tensor, delta: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        N = delta.shape[1]
        e = self.proj(self.norm_e(delta)) + self.mag(_log_rms(delta)) + support.unsqueeze(-1).float() * self.sup_emb
        h = torch.cat([h[:, :N] + e.to(h.dtype), h[:, N:]], 1)
        return self.norm(_run(self.blocks, h))


class SlotBlock(nn.Module):
    """Ordered slot queries: (slot-causal) self-attention among the slots, cross-attention to the fused residual
    cells, MLP."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.qkv, self.out1 = nn.LayerNorm(dim), nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out2 = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.norm3, self.mlp = nn.LayerNorm(dim), MLP(dim, 4.0)

    def forward(self, s: torch.Tensor, cells: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        H = self.heads
        q, k, v = (split_heads(t, H) for t in self.qkv(self.norm1(s)).chunk(3, -1))
        s = s + self.out1(merge_heads(attention(q, k, v, mask)))
        q = split_heads(self.q(self.norm_q(s)), H)
        k, v = (split_heads(t, H) for t in self.kv(self.norm_kv(cells)).chunk(2, -1))
        s = s + self.out2(merge_heads(attention(q, k, v)))
        return s + self.mlp(self.norm3(s))


class InnovationEncoder(nn.Module):
    """Enc_R: fused cells LN(e^F) W_F + LN(PatchEmbed(e^RGB)) W_RGB + LN(sg H^base) W_H + their log RMS +
    MetaEmbed(h, xi, kappa) + grid (+ support), read by n_slots ordered queries -> r [B', n_slots, d_r]."""

    def __init__(self, cfg: CD41Config, feat_dim: int, meta_dim: int):
        super().__init__()
        D, N, n = cfg.dec_dim, cfg.n_cells, cfg.n_slots
        self.patch_size = cfg.patch
        self.norm_f, self.wf = nn.LayerNorm(feat_dim), nn.Linear(feat_dim, D)
        self.patch, self.norm_p, self.wp = nn.Conv2d(3, D, cfg.patch, stride=cfg.patch), nn.LayerNorm(D), nn.Linear(D, D)
        self.norm_h, self.wh = nn.LayerNorm(feat_dim), nn.Linear(feat_dim, D)
        self.mag = nn.Linear(2, D)
        self.meta = MetaEmbed(meta_dim, D)
        self.pos_emb, self.sup_emb = _param(1, N, D), _param(D)
        self.queries = _param(1, n, D)
        self.blocks = nn.ModuleList([SlotBlock(D, cfg.dec_heads) for _ in range(cfg.enc_r_layers)])
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, cfg.d_r)
        self.register_buffer("causal", torch.ones(n, n, dtype=torch.bool).tril() if cfg.slot_causal else None,
                             persistent=False)

    def forward(self, eF, eRGB, Hb, support, meta) -> torch.Tensor:
        B = eF.shape[0]
        pe = self.patch(eRGB).flatten(2).transpose(1, 2)
        rgb_rms = 0.5 * torch.log(F.avg_pool2d(eRGB.float().pow(2).mean(1, keepdim=True), self.patch_size) + 1e-8)
        mag = torch.cat([_log_rms(eF), rgb_rms.flatten(2).transpose(1, 2)], -1)
        cells = (self.wf(self.norm_f(eF)) + self.wp(self.norm_p(pe)) + self.wh(self.norm_h(Hb)) + self.mag(mag)
                 + self.meta(meta).unsqueeze(1) + self.pos_emb + support.unsqueeze(-1).float() * self.sup_emb)
        s = self.queries.expand(B, -1, -1).to(cells.dtype)
        mask = None if self.causal is None else self.causal.view(1, 1, *self.causal.shape)
        for blk in self.blocks:
            s = blk(s, cells, mask)
        return self.out(self.norm(s))


# ---------------------------------------------------------------------------------------------- decoder side
class ConvSynth(nn.Module):
    """Module 2: H_0 = F_C(c) = Linear(d_c -> 256) -> depthwise-separable 3 x 3 -> GELU -> Linear(-> d);
    U = F_U(c, grid) = Linear(d_c + 2 -> 128) -> 3 x 3 conv -> GELU -> Linear(-> R_xy); beta = softmax(W_beta c)."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        G, Hc, Hu = cfg.grid, cfg.fc_hidden, cfg.fu_hidden
        self.G = G
        self.fc_in, self.fc_out = nn.Linear(cfg.d_c, Hc), nn.Linear(Hc, feat_dim)
        self.fc_dw, self.fc_pw = nn.Conv2d(Hc, Hc, 3, padding=1, groups=Hc), nn.Conv2d(Hc, Hc, 1)
        self.fu_in, self.fu_out = nn.Linear(cfg.d_c + 2, Hu), nn.Linear(Hu, cfg.rank_xy)
        self.fu_conv = nn.Conv2d(Hu, Hu, 3, padding=1)
        self.beta = nn.Linear(cfg.d_c, cfg.n_motion)
        self.register_buffer("grid_xy", geo.cell_centres(G) / G, persistent=False)      # u_j = (b + .5, a + .5) / G

    def _conv(self, x: torch.Tensor, *convs) -> torch.Tensor:
        B, N, C = x.shape
        y = x.transpose(1, 2).reshape(B, C, self.G, self.G)
        for conv in convs:
            y = conv(y)
        return y.flatten(2).transpose(1, 2)

    def forward(self, c: torch.Tensor) -> dict:
        H0 = self.fc_out(F.gelu(self._conv(self.fc_in(c), self.fc_dw, self.fc_pw)))
        g = self.grid_xy.to(c.dtype).expand(c.shape[0], -1, -1)
        U = self.fu_out(F.gelu(self._conv(self.fu_in(torch.cat([c, g], -1)), self.fu_conv)))
        return dict(H0=H0, U=U, beta=torch.softmax(self.beta(c).float(), -1))


class TransformerSynth(nn.Module):
    """synth = transformer: blocks over the cells of c, then H_0, U and beta heads (the first prototype)."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        D = cfg.dec_dim
        self.inp = nn.Linear(cfg.d_c, D)
        self.pos_emb = _param(1, cfg.n_cells, D)
        self.blocks = _blocks(D, cfg.dec_heads, cfg.syn_blocks)
        self.norm = nn.LayerNorm(D)
        self.h0, self.u, self.beta = nn.Linear(D, feat_dim), nn.Linear(D, cfg.rank_xy), nn.Linear(D, cfg.n_motion)

    def forward(self, c: torch.Tensor) -> dict:
        h = self.norm(_run(self.blocks, self.inp(c) + self.pos_emb))
        return dict(H0=self.h0(h), U=self.u(h), beta=torch.softmax(self.beta(h).float(), -1))


class RGBDecoder(nn.Module):
    """Dec_RGB: features [B', N, d] -> image [B', 3, H, W] (cells, then unpatchify + 3 x 3 conv)."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        D, p = cfg.dec_dim, cfg.patch
        self.cfg = cfg
        self.inp = nn.Linear(feat_dim, D)
        self.pos_emb = _param(1, cfg.n_cells, D)
        self.blocks = _blocks(D, cfg.dec_heads, cfg.rgb_blocks)
        self.norm = nn.LayerNorm(D)
        self.pix = nn.Linear(D, p * p * cfg.pix_out)
        self.pix_conv = nn.Conv2d(cfg.pix_out, 3, 3, padding=1)

    def forward(self, Fe: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        G, p, B = cfg.grid, cfg.patch, Fe.shape[0]
        h = self.norm(_run(self.blocks, self.inp(Fe) + self.pos_emb))
        x = self.pix(h).view(B, G, G, p, p, cfg.pix_out).permute(0, 5, 1, 3, 2, 4).reshape(B, cfg.pix_out, G * p, G * p)
        return self.pix_conv(F.gelu(x))


class ResidualDecoder(nn.Module):
    """Dec_R: the target cells, embedded from H^base, the support, the grid and MetaEmbed(h, xi, M, kappa, a),
    cross-attend to [null ; r[:L]] -> ΔF [B', N, d] float32; exactly 0 when L = 0 (skipped when every L is 0,
    masked otherwise, so no bias leaks in)."""

    def __init__(self, cfg: CD41Config, feat_dim: int, meta_dim: int):
        super().__init__()
        D, N = cfg.dec_dim, cfg.n_cells
        self.cfg = cfg
        self.inp = nn.Linear(feat_dim, D)
        self.meta = MetaEmbed(meta_dim, D)
        self.pos_emb, self.sup_emb = _param(1, N, D), _param(D)
        self.tok_in = nn.Linear(cfg.d_r, D)
        self.slot_emb = _param(1, cfg.n_slots, D)
        self.blocks = nn.ModuleList([DecBlock(D, cfg.dec_heads) for _ in range(cfg.dec_r_blocks)])
        self.norm = nn.LayerNorm(D)
        self.out = ZeroLinear(D, feat_dim)

    def forward(self, Hb, support, r, L, meta) -> torch.Tensor:
        if not bool((L > 0).any()):
            return torch.zeros(Hb.shape, device=Hb.device, dtype=torch.float32)
        x = self.inp(Hb) + self.pos_emb + support.unsqueeze(-1).float() * self.sup_emb + self.meta(meta).unsqueeze(1)
        tok = self.tok_in(r.to(x.dtype)) + self.slot_emb
        keep = torch.arange(self.cfg.n_slots, device=L.device).view(1, -1) < L.view(-1, 1)
        for blk in self.blocks:
            x = blk(x, tok, keep)
        out = self.out(self.norm(x)).float()
        return out * (L > 0).view(-1, 1, 1).float()


# ---------------------------------------------------------------------------------------------- losses
def feat_terms(pred, target) -> torch.Tensor:
    """(1 - cos) + MSE per cell [..., N]. Norms below 1 are clamped in the cosine: a cell without support has
    H^base = 0 exactly, and the cosine of a zero vector has an unbounded gradient (teacher features have
    norm ~ sqrt(d))."""
    p, t = pred.float(), target.float()
    cos = (p * t).sum(-1) / (p.norm(dim=-1).clamp_min(1.0) * t.norm(dim=-1).clamp_min(1.0))
    return 1 - cos + (p - t).pow(2).mean(-1)


def masked_feat_loss(pred, target, mask) -> torch.Tensor:
    """feat_terms averaged over the cells where mask is True. [B, N, d], mask [B, N]."""
    w = mask.float()
    return (feat_terms(pred, target) * w).sum() / w.sum().clamp_min(1.0)


def _split(v: torch.Tensor, B: int) -> torch.Tensor:
    """[B t, ...] (sample-major) -> [B, t, ...]."""
    return v.view(B, v.shape[0] // B, *v.shape[1:])


class MAVTCD41(nn.Module):
    COMPILE_REGIONS = ("StreamBlock", "DecBlock", "SlotBlock")
    MODALITIES = MODALITIES

    def __init__(self, cfg: CD41Config, use_checkpoint: bool = False):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg)
        d = self.feat_dim = self.encoder.dino.out_dim
        D, nm = cfg.dim, cfg.n_motion
        self.trunk = TwoStreamTrunk(cfg, d)
        self.trunk.c.use_checkpoint = self.trunk.d.use_checkpoint = use_checkpoint
        self.c_head = nn.Linear(D, cfg.d_c)
        self.rho_head = MLP(D, 1.0, out=1)
        nn.init.zeros_(self.rho_head.fc2.weight)
        nn.init.zeros_(self.rho_head.fc2.bias)
        self.cam_head = CameraHead(cfg)
        self.pose_head, self.motion_head = ZeroLinear(D, 6), ZeroLinear(D, 3)       # identity, no motion at init
        self.a_head = ZeroLinear(D, cfg.rank_a) if cfg.tucker else None
        self.refiner = Refiner(cfg, d) if cfg.refine else None
        self.synth = ConvSynth(cfg, d) if cfg.synth == "conv" else TransformerSynth(cfg, d)
        if cfg.tucker:
            R, Ra, Rc = cfg.rank_xy, cfg.rank_a, cfg.rank_c
            self.core = nn.Parameter(torch.randn(R, Ra, Rc) / math.sqrt(Ra))
            self.Q = nn.Linear(Rc, d, bias=False)
        meta_enc = header_dim(cfg) + 10
        self.enc_r = InnovationEncoder(cfg, d, meta_enc)
        self.dec_r = ResidualDecoder(cfg, d, meta_enc + 3 * nm + (cfg.rank_a if cfg.tucker else 0))
        self.dec_rgb = RGBDecoder(cfg, d)
        self.lpips = None
        if cfg.w_lpips > 0:
            from ..dino_v0.losses import LPIPSLoss
            self.lpips = LPIPSLoss()
        self.register_load_state_dict_post_hook(ignore_lpips_keys)

    # ------------------------------------------------------------------------------ decoder side
    def tucker_term(self, WU: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """[(W_i U) B(a_i)] Q. WU [B', N, R_xy], a [B', R_a] -> [B', N, d] float32."""
        Bm = torch.einsum("pkq,bk->bpq", self.core.float(), a.float())
        return self.Q(torch.bmm(WU.float(), Bm)).float()

    def target_op(self, syn: dict, rho: torch.Tensor, kappa0: torch.Tensor, z: dict) -> tuple:
        """W_i of B' = B t targets: anchor rho [B, N], kappa0 [B, 4], z = dict(xi [B', 6], M [B', nm, 3],
        kappa [B', 4]) -> (SparseTransportOp, per-cell displacement [B', N, 3])."""
        cfg = self.cfg
        t = z["xi"].shape[0] // rho.shape[0]
        rep = lambda v: v.repeat_interleave(t, 0)
        return tr.build_sparse_transport(rep(rho), rep(kappa0), rep(syn["beta"]), z["xi"], z["M"], z["kappa"],
                                         cfg.grid, cfg.r_cut, cfg.sigma_u, cfg.tau_z, cfg.knn, cfg.z_eps,
                                         cfg.use_smooth_cutoff, cfg.geometry)

    def target_base(self, syn: dict, z: dict, op: tr.SparseTransportOp) -> dict:
        """H^base = W H_0 + [(W U) B(a)] Q -> dict(Hb, warp (W H_0), delta or None), each [B', N, d] float32."""
        warp = op.apply(syn["H0"])
        delta = None
        if self.cfg.tucker and z.get("a") is not None:
            delta = self.tucker_term(op.apply(syn["U"]), z["a"])
        return dict(Hb=warp if delta is None else warp + delta, warp=warp, delta=delta)

    def meta_enc(self, h: torch.Tensor, i: torch.Tensor, z: dict) -> torch.Tensor:
        return torch.cat([header_features(self.cfg, h, i), z["xi"].float(), z["kappa"].float()], -1)

    def meta_dec(self, h: torch.Tensor, i: torch.Tensor, z: dict) -> torch.Tensor:
        parts = [self.meta_enc(h, i, z), z["M"].float().flatten(1)]
        if self.cfg.tucker:
            parts.append(z["a"].float())
        return torch.cat(parts, -1)

    def residual(self, Hb, support, z: dict, h: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.dec_r(Hb, support, z["r"], z["L"], self.meta_dec(h, i, z))

    @staticmethod
    def flat_targets(Z: dict) -> tuple:
        """State Z -> (z of the B T targets, each [B T, ...]; header [B T, 16]; target index [B T] in 1 .. T)."""
        B, T = Z["xi"].shape[:2]
        z = {k: Z[k].flatten(0, 1) for k in ("xi", "M", "kappa", "r", "L")}
        z["a"] = Z["a"].flatten(0, 1) if Z.get("a") is not None else None
        i = torch.arange(1, T + 1, device=Z["c"].device).repeat(B)
        return z, Z["h"].repeat_interleave(T, 0), i

    def decode(self, Z: dict) -> dict:
        """Decoder side alone: state Z (see state.py) -> dict(F_hat [B, n, N, d], rgb [B, n, 3, H, W],
        Hb [B, T, N, d] or None, support [B, T, N] or None). Each target reads only C, z_i and the header."""
        syn = self.synth(Z["c"])
        B = Z["c"].shape[0]
        T = Z["xi"].shape[1]
        feats, Hb, sup = [syn["H0"].float().unsqueeze(1)], None, None
        if T > 0:
            z, h, i = self.flat_targets(Z)
            op, _ = self.target_op(syn, Z["rho"], Z["kappa0"], z)
            Hb = self.target_base(syn, z, op)["Hb"]
            feats.append(_split(Hb + self.residual(Hb, op.support, z, h, i), B))
            Hb, sup = _split(Hb, B), _split(op.support, B)
        Fh = torch.cat(feats, 1)
        rgb = self.dec_rgb(Fh.flatten(0, 1))
        return dict(F_hat=Fh, rgb=rgb.view(B, Fh.shape[1], *rgb.shape[1:]), Hb=Hb, support=sup)

    # ------------------------------------------------------------------------------ helpers
    @staticmethod
    def to_frames(x: torch.Tensor, modality: str) -> torch.Tensor:
        return x.unsqueeze(1) if modality == "image" else x.permute(0, 2, 1, 3, 4)

    def sample_L(self, shape: tuple, device) -> torch.Tensor:
        cfg = self.cfg
        if not (self.training and cfg.nested):
            return torch.full(shape, cfg.n_slots, dtype=torch.long, device=device)
        choices = torch.tensor(cfg.nested_L, device=device)
        return choices[torch.randint(len(cfg.nested_L), shape, device=device)]

    def _rec(self, rgb, img, Fh, Fs, parts, diag) -> torch.Tensor:
        """Reconstruction of every observation: L1 + LPIPS (random subset) on pixels, cosine + MSE on features."""
        cfg = self.cfg
        rgb, im = rgb.flatten(0, 1), img.flatten(0, 1)
        l1, fl = rgb_l1(rgb, im), feat_terms(Fh, Fs).mean()
        parts["rec_rgb"], parts["rec_feat"] = l1.detach(), fl.detach()
        loss = cfg.w_rgb * l1 + cfg.w_feat * fl
        if self.lpips is not None:
            n = cfg.lpips_max_frames
            sel = torch.randperm(rgb.shape[0], device=rgb.device)[:n] if 0 < n < rgb.shape[0] else None
            lp = self.lpips(rgb, im) if sel is None else self.lpips(rgb[sel], im[sel])
            parts["lpips"] = lp.detach()
            loss = loss + cfg.w_lpips * lp
        return loss

    # ------------------------------------------------------------------------------ encoder side, targets
    def _encode_targets(self, x0, mem, Ft, Xt, ctx, syn, rho, kappa0, h, i, L, mode: str, cam) -> dict:
        """B' = B t targets that share one context. x0 [B', N, dim], mem [B', L, N, dim], Ft [B', N, d], Xt
        [B', 3, H, W], ctx [B', Nc, dim], h [B', 16], i [B'], L [B'], cam (xi [B', 6], kappa [B', 4]) when the
        camera is given -> dict(z, Hb, Hb1, support, support1, warp, delta, disp, rgb_b, F_hat)."""
        cfg = self.cfg
        N, nm, Bp = cfg.n_cells, cfg.n_motion, x0.shape[0]
        static, given = mode == "static_views", cam is not None
        dt = torch.gather(h[:, 2:6], 1, i.view(-1, 1)).squeeze(1)
        hD = self.trunk.target(x0, mem, ctx, dt, None if cam is None else torch.cat(cam, -1))
        o_cam = self.cam_head.logits(hD[:, N])
        mot = slice(N + 1, N + 1 + nm)
        if given:
            z = dict(xi=cam[0].float(), kappa=cam[1].float())
        else:
            z = dict(xi=self.pose_head(hD[:, N]).float(), kappa=self.cam_head.kappa(o_cam))
        z["M"] = torch.zeros(Bp, nm, 3, device=x0.device) if static else self.motion_head(hD[:, mot]).float()
        z["a"] = self.a_head(hD[:, N + 1 + nm]).float() if self.a_head is not None else None
        op, disp = self.target_op(syn, rho, kappa0, z)                                  # level 1, from the trunk
        bs1 = self.target_base(syn, z, op)
        out = dict(Hb1=bs1["Hb"], support1=op.support)
        if self.refiner is not None and (not given or not static or cfg.tucker):        # level 2
            rf = self.refiner
            hR = rf(hD, Ft.float() - bs1["Hb"].detach(), op.support)
            z2 = dict(z)
            if not given:
                z2["xi"] = z["xi"] + rf.d_pose(hR[:, N]).float()
                z2["kappa"] = self.cam_head.kappa(o_cam + rf.d_kappa(hR[:, N]).float())
            if not static:
                z2["M"] = z["M"] + rf.d_motion(hR[:, mot]).float()
            if rf.d_a is not None:
                z2["a"] = z["a"] + rf.d_a(hR[:, N + 1 + nm]).float()
            if not given or not static:                                                 # the geometry moved
                op, disp = self.target_op(syn, rho, kappa0, z2)
            z = z2
            bs = self.target_base(syn, z, op)
        else:
            bs = bs1
        Hb, sup = bs["Hb"], op.support
        with torch.no_grad():
            rgb_b = self.dec_rgb(Hb.detach())
        r = self.enc_r(Ft.float() - Hb.detach(), Xt - rgb_b, Hb.detach(), sup, self.meta_enc(h, i, z)).float()
        keep = torch.arange(cfg.n_slots, device=x0.device).view(1, -1) < L.view(-1, 1)
        z.update(r=r * keep.unsqueeze(-1), L=L)
        Fh = Hb + self.residual(Hb, sup, z, h, i)
        out.update(z=z, Hb=Hb, support=sup, warp=bs["warp"], delta=bs["delta"], disp=disp, rgb_b=rgb_b, F_hat=Fh)
        return out

    def _targets(self, x0, mem, Fs, frames, syn, rho, kappa0, h, mode, cam_t, L) -> dict:
        """The T = n - 1 targets of a segment. Static views: one parallel group whose context is H_0 alone.
        Video: one target at a time, the context holding H_0 and the F^ of the earlier targets, all decoded from
        the state (detached). -> dict of [B, T, ...] tensors (z merged the same way)."""
        B, T = x0.shape[:2]
        groups = [list(range(T))] if mode == "static_views" else [[j] for j in range(T)]
        ctx_f, ctx_t = [syn["H0"].detach().float()], [h[:, 2]]
        outs = []
        for g in groups:
            t = len(g)
            sel = lambda v: v[:, g].flatten(0, 1)
            ctx = self.trunk.context(torch.stack(ctx_f, 1), torch.stack(ctx_t, 1)).repeat_interleave(t, 0)
            cam = None if cam_t is None else (sel(cam_t[0]), sel(cam_t[1]))
            i = torch.tensor([j + 1 for j in g], device=x0.device).repeat(B)
            o = self._encode_targets(sel(x0), sel(mem), sel(Fs[:, 1:]), sel(frames[:, 1:]), ctx, syn, rho, kappa0,
                                     h.repeat_interleave(t, 0), i, sel(L), mode, cam)
            outs.append(o)
            if mode != "static_views":
                ctx_f.append(o["F_hat"].detach().float())
                ctx_t.append(h[:, 2 + g[0] + 1])
        merge = lambda vs: None if vs[0] is None else torch.cat([_split(v, B) for v in vs], 1)
        res = {k: merge([o[k] for o in outs]) for k in outs[0] if k != "z"}
        res["z"] = {k: merge([o["z"][k] for o in outs]) for k in outs[0]["z"]}
        return res

    # ------------------------------------------------------------------------------ one segment
    def _segment(self, frames, mode: str, cam: Optional[dict], parts, diag, L_given=None, gt_cam=False):
        """frames [B, n, 3, H, W] (observation 0 is the anchor); mode image | video | static_views;
        cam dict(c2w [B, n, 4, 4], K [B, n, 3, 3] [, depth, dyn]) or None; gt_cam: the cameras of a video
        segment enter the loss only (world clips)."""
        cfg = self.cfg
        B, n = frames.shape[:2]
        S, dev, N = cfg.image_size, frames.device, cfg.n_cells
        T = n - 1
        enc = self.encoder(frames.flatten(0, 1))
        x0 = enc["x0"].view(B, n, *enc["x0"].shape[1:])
        mem = enc["mem"].view(B, n, *enc["mem"].shape[1:])
        Fs = enc["feat_target"].view(B, n, *enc["feat_target"].shape[1:])
        hc = self.trunk.anchor(x0[:, 0], mem[:, 0])                                     # stream C: i = 0 only
        c = self.c_head(hc[:, :N]).float()
        rho_raw = self.rho_head(hc[:, :N]).squeeze(-1).float()
        kappa_est = self.cam_head(hc[:, N])
        given = mode == "static_views" or (gt_cam and cfg.world_camera == "given")
        kappa_cam = xi_cam = None
        if cam is not None:
            kappa_cam = tr.kappa_from_K(cam["K"].float(), S, S)                              # [B, n, 4]
            xi_cam = tr.se3_log(tr.relative_pose(cam["c2w"]))                                # [B, n, 6]
        kappa0 = kappa_cam[:, 0] if given else kappa_est
        rho = rho_raw if mode == "static_views" else geo.median_normalise(rho_raw)
        syn = self.synth(c)
        h = st.make_header(cfg, mode, n, torch.zeros(B, T, dtype=torch.long, device=dev), cam_given=given)
        Z = dict(h=h, c=c, rho=rho, kappa0=kappa0, xi=torch.zeros(B, 0, 6, device=dev))
        loss, tg = 0.0, None
        if T > 0:
            L = self.sample_L((B, T), dev) if L_given is None else L_given.to(dev).long().view(B, T)
            cam_t = (xi_cam[:, 1:], kappa_cam[:, 1:]) if given else None
            tg = self._targets(x0[:, 1:], mem[:, 1:], Fs, frames, syn, rho, kappa0, h, mode, cam_t, L)
            Z.update(tg["z"])
            if Z["a"] is None:
                del Z["a"]
            Z["h"][:, 10:10 + T] = L.float()
            Ft = Fs[:, 1:]
            bl = masked_feat_loss(tg["Hb"].flatten(0, 1), Ft.flatten(0, 1), tg["support"].flatten(0, 1))
            parts["base_feat"] = bl.detach()
            loss = loss + cfg.w_base * bl
            if self.refiner is not None and cfg.w_refine1 > 0:
                b1 = masked_feat_loss(tg["Hb1"].flatten(0, 1), Ft.flatten(0, 1), tg["support1"].flatten(0, 1))
                parts["base_feat1"] = b1.detach()
                loss = loss + cfg.w_refine1 * b1
            if mode != "static_views":
                fl = Z["M"].abs().mean()
                parts["flow_l1"] = fl.detach()
                loss = loss + cfg.w_flow * fl
            self._target_diag(Z, tg, Fs, frames, cam, diag)
        Fh = syn["H0"].float().unsqueeze(1)
        if tg is not None:
            Fh = torch.cat([Fh, tg["F_hat"]], 1)
        rgb_all = self.dec_rgb(Fh.flatten(0, 1))
        rgb_all = rgb_all.view(B, n, *rgb_all.shape[1:])
        loss = loss + self._rec(rgb_all, frames, Fh, Fs, parts, diag)
        if gt_cam and cam is not None and cfg.world_camera == "estimated" and T > 0:
            loss = loss + self._camera_loss(Z, kappa_cam, xi_cam, parts, diag)
        if cfg.w_smooth > 0 and mode != "image":
            s = edge_aware_smooth(rho, Fs[:, 0], cfg.grid)
            parts["smooth"] = s.detach()
            loss = loss + cfg.w_smooth * s
        if cam is not None and cam.get("depth") is not None:
            tgt, mask = cell_log_inv_depth(cam["depth"].float()[:, 0], cam["depth"][:, 0] > 0, cfg.grid)
            dg = silog(rho, tgt, mask, lam=cfg.depth_gt_lam)
            parts["depth_gt"] = dg.detach()
            diag["rho_gt_corr"] = _pearson(rho.detach(), tgt, mask).mean()
            if cfg.w_depth_gt > 0:
                loss = loss + cfg.w_depth_gt * dg
        if mode != "image":
            diag["rho_std"] = rho.detach().std(-1).mean()
        diag["fov_deg"] = torch.rad2deg(2 * torch.atan(0.5 / Z["kappa0"][:, 0].detach().clamp_min(1e-6))).mean()
        rgb = rgb_all.detach()
        diag["psnr"] = psnr(rgb.flatten(0, 1), frames.flatten(0, 1))
        diag["psnr_anchor"] = psnr(rgb[:, 0], frames[:, 0])
        diag["numbers"] = st.payload_numbers(cfg, Z).float().mean()
        if T > 0:
            diag["psnr_target"] = psnr(rgb[:, 1:].flatten(0, 1), frames[:, 1:].flatten(0, 1))
            diag["feat_cos"] = F.cosine_similarity(Fh[:, 1:].detach(), Fs[:, 1:].float(), -1).mean()
        return loss, dict(Z=Z, rgb=rgb_all, F_hat=Fh, Hb=None if tg is None else tg["Hb"],
                          support=None if tg is None else tg["support"], F=Fs)

    def _target_diag(self, Z, tg, Fs, frames, cam, diag) -> None:
        cfg = self.cfg
        Ft, Hb, sup = Fs[:, 1:].float(), tg["Hb"].detach().float(), tg["support"]
        cos = F.cosine_similarity(Hb, Ft, -1)
        diag["support_frac"] = sup.float().mean()
        if sup.any():
            diag["base_cos_sup"] = cos[sup].mean()
        if (~sup).any():
            diag["base_cos_new"] = cos[~sup].mean()
        sup1 = tg["support1"]
        if sup1.any():
            diag["base1_cos_sup"] = F.cosine_similarity(tg["Hb1"].detach().float(), Ft, -1)[sup1].mean()
        warp = tg["warp"].detach().float()
        diag["warp_cos_sup"] = F.cosine_similarity(warp, Ft, -1)[sup].mean() if sup.any() else torch.tensor(0.0)
        if tg["delta"] is not None:
            diag["tucker_rel"] = tg["delta"].detach().float().norm(dim=-1).mean() / warp.norm(dim=-1).mean().clamp_min(1e-6)
            diag["a_norm"] = Z["a"].detach().norm(dim=-1).mean()
        diag["psnr_base"] = psnr(tg["rgb_b"].flatten(0, 1), frames[:, 1:].flatten(0, 1))
        diag["psnr_copy"] = psnr(frames[:, :1].expand_as(frames[:, 1:]).flatten(0, 1), frames[:, 1:].flatten(0, 1))
        R, t = tr.se3_exp(Z["xi"].detach())
        diag["xi_rot_deg"] = geo.rotation_angle_deg(R).mean()
        diag["xi_trans"] = t.norm(dim=-1).mean()
        if cam is not None and cam.get("dyn") is not None:
            share = F.adaptive_avg_pool2d(cam["dyn"].float()[:, 0].unsqueeze(1), cfg.grid).flatten(1)   # [B, N]
            mov = tg["disp"].detach().norm(dim=-1).mean(1)
            dyn = share >= 0.5
            diag["dyn_frac"] = dyn.float().mean()
            if dyn.any():
                diag["flow_dyn"] = mov[dyn].mean()
            if (~dyn).any():
                diag["flow_static"] = mov[~dyn].mean()

    def _camera_loss(self, Z, kappa_cam, xi_cam, parts, diag) -> torch.Tensor:
        """Estimated cameras of a video segment against the given ones (the decoder never sees the latter)."""
        R, t = tr.se3_exp(Z["xi"])
        Rg, tg = tr.se3_exp(xi_cam[:, 1:])
        lr = (R - Rg).abs().mean()
        lt = (t - tg).abs().mean()
        lk = (Z["kappa"] - kappa_cam[:, 1:]).abs().mean() + (Z["kappa0"] - kappa_cam[:, 0]).abs().mean()
        cl = lr + lt + lk
        parts["cam_gt"] = cl.detach()
        with torch.no_grad():
            diag["cam_rot_err_deg"] = geo.rotation_angle_deg(R.transpose(-1, -2) @ Rg).mean()
            diag["cam_trans_err"] = (t - tg).norm(dim=-1).mean()
            diag["cam_trans_gt"] = tg.norm(dim=-1).mean()
        return self.cfg.w_cam_gt * cl

    # ------------------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, modality: str, cam: Optional[dict] = None,
                L: Optional[torch.Tensor] = None, return_outputs: bool = False) -> dict:
        """x in [-1, 1]; L (eval) innovation lengths [B * segments, K - 1] or None (n_slots)."""
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")
        cfg = self.cfg
        frames = self.to_frames(x, modality)
        B, n_all = frames.shape[:2]
        parts, diag = {}, {}
        if modality in ("threed", "world") and cam is None:
            raise ValueError(f"{modality} needs cam = dict(c2w, K)")
        if modality == "image":
            loss, outputs = self._segment(frames, "image", None, parts, diag)
        elif modality == "threed":
            if n_all > cfg.keyframe_every:
                raise ValueError(f"{n_all} views: a segment holds at most keyframe_every={cfg.keyframe_every}")
            loss, outputs = self._segment(frames, "static_views", cam, parts, diag, L)
        else:
            K = cfg.keyframe_every if n_all > cfg.keyframe_every else n_all
            if n_all % K:
                raise ValueError(f"clip length {n_all} must be a multiple of keyframe_every={cfg.keyframe_every}")
            S = n_all // K
            seg = lambda t: t.reshape(B * S, K, *t.shape[2:])
            cs = None if cam is None else {k: seg(v) for k, v in cam.items() if v is not None}
            loss, outputs = self._segment(seg(frames), "video", cs, parts, diag, L, gt_cam=modality == "world")
            outputs["rgb"] = outputs["rgb"].reshape(B, n_all, *outputs["rgb"].shape[2:])
        dev = x.device
        diag = {k: (v if torch.is_tensor(v) else torch.tensor(float(v))).to(dev).float().detach() for k, v in diag.items()}
        out = dict(loss=loss, parts=parts, diag=diag)
        if return_outputs:
            out["outputs"] = outputs
        return out


def load_cd41(ckpt_path: str, device="cpu", strict: bool = True, **overrides):
    """Checkpoint of scripts/train_cd41.py -> (model in eval mode, cfg, step)."""
    from .config import config_from_dict
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck.get("cfg", {}), **{"w_lpips": 0.0, **overrides})
    model = MAVTCD41(cfg)
    sd = ck["model"] if "model" in ck else ck
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.dino.model.")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return model.to(device).eval(), cfg, int(ck.get("step", -1))
