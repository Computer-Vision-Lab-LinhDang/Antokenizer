"""Tier E: query decoder. Reads ONLY the compressed latent z_r.

For a query at (t, u, v) of frame t in a video segment the decoder gets zc = z_r(C_k) (keyframe
content grid) and zd = z_r(D_t) (how frame t deviates from C_k). Per query:

  Δ      = offset in (u, v) units at which the query reads C, last layer zero-init -> Δ = 0 at init
           delta_mode "mean"  (v0): MLP([mean_M in_d(zd) ; fourier(u, v)]): one D summary for all queries
           delta_mode "query" (v1): q = Linear(fourier(u, v)) (+ ZeroLinear(ray) for posed views);
                                    readD_q = cross-attn(LN q -> LN in_d(zd)), no RoPE;
                                    Δ = MLP([readD_q ; fourier(u, v)]): each region its own motion
  readC  = cross-attn to the (2r+1)^2 C cells around the cell containing (u, v) + Δ; the query
           sits at the SHIFTED centre under AxialRoPE, so the gradient reaches Δ through RoPE.
           Computed as dense attention over all N cells with a window mask (same result as
           gathering the window, without (2r+1)^2 copies of k / v and their scatter-add backward)
  readD  = cross-attn to in_d(zd) tokens by content (no RoPE)
  o      = sigmoid(Linear([h ; readC]))                   bias init cfg.gate_init_bias
  h     += o * readC + (1 - o) * readD                     (images: h += readC, o = 1)
  h     += SelfAttn(h) (RoPE on the unshifted coords); h += MLP(h)       pre-LN everywhere

warp_only=True (video): every block does h += readC (o = 1, readD skipped) while Δ is still computed
from zd, so D reaches the output only through Δ: a decode that forces Δ to carry the motion.

Heads on the final LN(h): feat [B, Q, feat_dim] per query; rgb by cfg.rgb_head:
  "linear" (v0): per-query Linear -> rgb [B, Q, p*p*3], image None
  "conv"   (v1): h of all N queries (row-major grid order, Q == N) -> [B, dec_dim, G, G] -> conv in ->
                 ResBlocks -> log2(p) x (nearest x2, 3x3 conv, ResBlocks) at cfg.conv_channels ->
                 GN, SiLU, 3x3 conv -> image [B, 3, H, W]; rgb = image_to_patches(image), so per-patch
                 losses work unchanged. No patch grid: neighbouring patches share conv context.
Every row of the batch is independent (rows may be B * T flattened frames, each with its own t).
Coordinates: u = column, v = row, in [0, 1].
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import ZeroLinear
from .config import DinoV0Config
from .rope import AxialRoPE, grid_coords

UV_MASK = (0.0, 1.0, 1.0, 0.0)           # the C read uses only the (u, v) bits of q_mask


# ---- patch helpers -----------------------------------------------------------------------

def image_to_patches(img: torch.Tensor, p: int) -> torch.Tensor:
    """[B, 3, H, W] -> [B, N, p*p*3], patches row-major, each patch flattened as (py, px, c)."""
    B, C, H, W = img.shape
    gh, gw = H // p, W // p
    x = img.reshape(B, C, gh, p, gw, p).permute(0, 2, 4, 3, 5, 1)  # B, gh, gw, py, px, c
    return x.reshape(B, gh * gw, p * p * C)


def patches_to_image(patches: torch.Tensor, grid: int, p: int) -> torch.Tensor:
    """Exact inverse of image_to_patches for a square grid x grid layout."""
    B = patches.shape[0]
    x = patches.reshape(B, grid, grid, p, p, 3).permute(0, 5, 1, 3, 2, 4)  # B, c, gh, py, gw, px
    return x.reshape(B, 3, grid * p, grid * p)


def query_coords(grid: int, t: float, B: int, Q_idx: Optional[torch.Tensor] = None,
                 device=None) -> torch.Tensor:
    """[B, Q, 4] patch-centre coords (t, u, v, 0) of all N cells, or of Q_idx ([Q] or [B, Q])."""
    c = grid_coords(grid, t, device=device)                         # [N, 4]
    if Q_idx is None:
        return c.unsqueeze(0).expand(B, -1, -1).clone()
    Q_idx = Q_idx.to(c.device)
    if Q_idx.dim() == 1:
        return c[Q_idx].unsqueeze(0).expand(B, -1, -1).clone()
    return c[Q_idx]


def gather_window(centre: torch.Tensor, grid: int, r: int):
    """Cells of the (2r+1)^2 window around the cell containing each centre.

    centre: [B, Q, 2] (u, v) in [0, 1] (may lie outside; the centre cell is clamped to the grid).
    Returns idx [B, Q, W] (flat row*grid + col, clamped for gathering), valid [B, Q, W] (window
    cell lies inside the grid) and cell [B, Q, W, 2] (u, v) cell-centre coords of each window cell.
    """
    with torch.no_grad():
        col = (centre[..., 0] * grid).floor().long().clamp(0, grid - 1)   # [B, Q]
        row = (centre[..., 1] * grid).floor().long().clamp(0, grid - 1)
        off = torch.arange(-r, r + 1, device=centre.device)
        dr, dc = torch.meshgrid(off, off, indexing="ij")
        dr, dc = dr.reshape(-1), dc.reshape(-1)                           # [W]
        rows = row.unsqueeze(-1) + dr                                     # [B, Q, W]
        cols = col.unsqueeze(-1) + dc
        valid = (rows >= 0) & (rows < grid) & (cols >= 0) & (cols < grid)
        rows_c, cols_c = rows.clamp(0, grid - 1), cols.clamp(0, grid - 1)
        idx = rows_c * grid + cols_c
        cell = torch.stack([(cols_c.float() + 0.5) / grid, (rows_c.float() + 0.5) / grid], -1)
    return idx, valid, cell


def fourier_uv(uv: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """[..., 2] -> [..., 2 + 4 * n_freq]: raw (u, v) plus sin / cos at 2^k * pi."""
    freq = (2.0 ** torch.arange(n_freq, device=uv.device, dtype=torch.float32)) * math.pi
    a = uv.float().unsqueeze(-1) * freq                                   # [..., 2, F]
    return torch.cat([uv.float(), a.sin().flatten(-2), a.cos().flatten(-2)], -1).to(uv.dtype)


# ---- blocks ----------------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hid), nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def window_bias(centre: torch.Tensor, grid: int, r: int, dtype) -> torch.Tensor:
    """[B, Q, N] additive mask: 0 where cell n is in the (2r+1)^2 window around the (clamped) cell
    containing each centre [B, Q, 2], -inf elsewhere. Same window as gather_window."""
    with torch.no_grad():
        col = (centre[..., 0] * grid).floor().long().clamp(0, grid - 1)   # [B, Q]
        row = (centre[..., 1] * grid).floor().long().clamp(0, grid - 1)
        n = torch.arange(grid * grid, device=centre.device)
        inside = ((row.unsqueeze(-1) - n // grid).abs() <= r) & ((col.unsqueeze(-1) - n % grid).abs() <= r)
        bias = torch.zeros(inside.shape, device=centre.device, dtype=dtype)
        return bias.masked_fill(~inside, float("-inf"))


class _WindowCrossAttn(nn.Module):
    """Query -> the (2r+1)^2 C cells around its shifted centre, AxialRoPE on (shifted centre, cell centres).

    forward: dense attention over all N cells, each cell's key rotated once, window as an additive
    mask (float, not bool: bool masks gave zero gradients on some SDPA backends).
    forward_gather: the original gather-the-window version, kept as the reference in tests."""

    def __init__(self, dim: int, heads: int, rope: AxialRoPE):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.q, self.kv, self.o = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.rope = rope

    def read(self, x, c, win: tuple):
        """win from QueryDecoder.forward: (bias, q_pos, q_m, c_pos, c_m) -> dense masked read (window_read
        "dense"); (idx, valid, q_pos, k_pos, q_m, k_m) -> gathered window read (window_read "gather")."""
        return self.forward(x, c, *win) if len(win) == 5 else self.forward_gather(x, c, *win)

    def forward(self, x, c, bias, q_pos, q_m, c_pos, c_m):
        # x [B,Q,D]; c [B,N,D]; bias [B,Q,N]; q_pos/q_m [B,Q,4]; c_pos/c_m [B,N,4]
        B, Q, D = x.shape
        N = c.shape[1]
        q = self.rope(self.q(x).view(B, Q, self.h, self.hd).transpose(1, 2), q_pos, q_m)   # [B,H,Q,hd]
        k, v = self.kv(c).view(B, N, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)           # [B,H,N,hd]
        k = self.rope(k, c_pos, c_m)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.unsqueeze(1).to(q.dtype))
        return self.o(out.transpose(1, 2).reshape(B, Q, D))

    def forward_gather(self, x, c, idx, valid, q_pos, k_pos, q_m, k_m):
        # x [B,Q,D]; c [B,N,D]; idx/valid [B,Q,W]; q_pos/q_m [B,Q,4]; k_pos/k_m [B,Q,W,4]
        B, Q, D = x.shape
        W = idx.shape[-1]
        q = self.q(x).view(B, Q, self.h, self.hd).transpose(1, 2)               # [B,H,Q,hd]
        q = self.rope(q, q_pos, q_m)
        kv = self.kv(c)                                                         # [B,N,2D]
        kv = kv.gather(1, idx.reshape(B, Q * W, 1).expand(-1, -1, 2 * D))       # [B,QW,2D]
        k, v = kv.view(B, Q * W, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)     # [B,H,QW,hd]
        k = self.rope(k, k_pos.reshape(B, Q * W, 4), k_m.reshape(B, Q * W, 4))
        k = k.view(B, self.h, Q, W, self.hd)
        v = v.view(B, self.h, Q, W, self.hd)
        s = (q.unsqueeze(3) * k).sum(-1) / math.sqrt(self.hd)                   # [B,H,Q,W]
        s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
        a = s.softmax(-1)
        out = (a.unsqueeze(-1) * v).sum(3)                                      # [B,H,Q,hd]
        return self.o(out.transpose(1, 2).reshape(B, Q, D))


class _Attn(nn.Module):
    """Plain multi-head attention; RoPE on q / k when coords are given."""

    def __init__(self, dim: int, heads: int, rope: Optional[AxialRoPE] = None):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.q, self.kv, self.o = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x, ctx, q_pos=None, q_m=None, k_pos=None, k_m=None, k_valid=None):
        # k_valid [B, N] bool | None: the keys each row may attend to (a prefix of the D tokens)
        B, Q, D = x.shape
        N = ctx.shape[1]
        q = self.q(x).view(B, Q, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(ctx).view(B, N, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.rope is not None and q_pos is not None:
            q, k = self.rope(q, q_pos, q_m), self.rope(k, k_pos, k_m)
        mask = None if k_valid is None else k_valid[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, Q, D))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: DinoV0Config, rope: AxialRoPE):
        super().__init__()
        d = cfg.dec_dim
        self.n_q, self.n_c, self.read_c = nn.LayerNorm(d), nn.LayerNorm(d), _WindowCrossAttn(d, cfg.dec_heads, rope)
        self.n_qd, self.n_d, self.read_d = nn.LayerNorm(d), nn.LayerNorm(d), _Attn(d, cfg.dec_heads)
        self.gate = nn.Linear(2 * d, 1)
        nn.init.constant_(self.gate.bias, cfg.gate_init_bias)
        self.n_s, self.self_attn = nn.LayerNorm(d), _Attn(d, cfg.dec_heads, rope)
        self.n_m, self.mlp = nn.LayerNorm(d), _MLP(d, cfg.mlp_ratio)

    def forward(self, h, c, d, win, q_coords, q_mask, d_valid=None):
        hn = self.n_q(h)
        rc = self.read_c.read(hn, self.n_c(c), win)
        if d is None:
            o = torch.ones_like(h[..., :1])
            h = h + rc
        else:
            rd = self.read_d(self.n_qd(h), self.n_d(d), k_valid=d_valid)
            o = torch.sigmoid(self.gate(torch.cat([hn, rc], -1)))
            h = h + o * rc + (1 - o) * rd
        hs = self.n_s(h)
        h = h + self.self_attn(hs, hs, q_coords, q_mask, q_coords, q_mask)
        h = h + self.mlp(self.n_m(h))
        return h, o


# ---- conv rgb head ---------------------------------------------------------------------------

def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, ch), ch)


class _ResBlock(nn.Module):
    """x + conv(SiLU(GN(conv(SiLU(GN(x)))))), 3x3 convs, width preserved."""

    def __init__(self, ch: int):
        super().__init__()
        self.n1, self.c1 = _gn(ch), nn.Conv2d(ch, ch, 3, padding=1)
        self.n2, self.c2 = _gn(ch), nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class ConvUpsampler(nn.Module):
    """[B, G*G, dim] tokens in row-major grid order -> image [B, 3, G * 2^S, G * 2^S], S = len(channels) - 1.

    channels[0] is the width at grid resolution, then one width per nearest-x2 stage. No output
    activation (the v0 linear rgb head had none either; targets are in [-1, 1]).

    channels_last: the weights in channels_last and the token grid read as [B, G, G, D], which IS the NHWC
    layout of [B, D, G, G] (a view, no copy), so every conv runs on NHWC tensors and MIOpen / cuDNN skip the
    NCHW <-> NHWC transposes they insert around their NHWC kernels otherwise. The image comes back NCHW."""

    def __init__(self, dim: int, channels: tuple, res_blocks: int, channels_last: bool = False):
        super().__init__()
        ch = list(channels)
        self.channels_last = channels_last
        self.conv_in = nn.Conv2d(dim, ch[0], 3, padding=1)
        self.stages = nn.ModuleList([nn.Sequential(*[_ResBlock(ch[0]) for _ in range(res_blocks)])])
        for cin, cout in zip(ch[:-1], ch[1:]):
            self.stages.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(cin, cout, 3, padding=1),
                                             *[_ResBlock(cout) for _ in range(res_blocks)]))
        self.n_out, self.conv_out = _gn(ch[-1]), nn.Conv2d(ch[-1], 3, 3, padding=1)
        if channels_last:
            self.to(memory_format=torch.channels_last)

    def forward(self, h: torch.Tensor, grid: int) -> torch.Tensor:
        B, N, D = h.shape
        if self.channels_last:                                               # NHWC view of the token grid
            x = h.reshape(B, grid, grid, D).permute(0, 3, 1, 2)
        else:
            x = h.transpose(1, 2).reshape(B, D, grid, grid)                  # row n = r * G + c -> [r, c]
        x = self.conv_in(x)
        for st in self.stages:
            x = st(x)
        out = self.conv_out(F.silu(self.n_out(x)))
        return out.contiguous() if self.channels_last else out


class QueryDecoder(nn.Module):
    """z_r (C, D) + query coords -> per-query rgb patch, DINO feature, offset Δ and gate o.

    Parameters of the v1 options exist only when their flag is on (delta_* for delta_mode "query",
    rgb_conv for rgb_head "conv"), so with default flags the state_dict keys are the v0 ones."""

    def __init__(self, cfg: DinoV0Config, feat_dim: int):
        super().__init__()
        assert cfg.delta_mode in ("mean", "query"), f"unknown delta_mode {cfg.delta_mode!r}"
        assert cfg.rgb_head in ("linear", "conv"), f"unknown rgb_head {cfg.rgb_head!r}"
        assert cfg.window_read in ("dense", "gather"), f"unknown window_read {cfg.window_read!r}"
        self.ray_q = ZeroLinear(6, cfg.dec_dim)        # query rays for posed views; zero at init
        d = cfg.dec_dim
        self.cfg, self.grid, self.r = cfg, cfg.grid, cfg.read_window
        self.in_c_image = nn.Linear(cfg.zr_c_image, d)
        self.in_c_video = nn.Linear(cfg.zr_c_video, d)
        self.in_d = nn.Linear(cfg.zr_d, d)
        if cfg.d_nested:                               # always-visible null token: the empty prefix reads it alone
            self.d_null = nn.Parameter(torch.zeros(1, 1, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.n_fourier = 8
        nf = 2 + 4 * self.n_fourier
        if cfg.delta_mode == "mean":
            self.offset = nn.Sequential(nn.Linear(d + nf, d), nn.GELU(), nn.Linear(d, 2))
            last = self.offset[-1]
        else:                                          # per-query Δ: each query reads D for itself
            self.delta_q = nn.Linear(nf, d)
            self.delta_ray = ZeroLinear(6, d)          # target-view ray of the query (posed views)
            self.delta_n_q, self.delta_n_d = nn.LayerNorm(d), nn.LayerNorm(d)
            self.delta_read = _Attn(d, cfg.dec_heads)  # no RoPE: D tokens have no coordinates
            self.delta_mlp = nn.Sequential(nn.Linear(d + nf, d), nn.GELU(), nn.Linear(d, 2))
            last = self.delta_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.rope = AxialRoPE(d // cfg.dec_heads)
        self.blocks = nn.ModuleList([DecoderBlock(cfg, self.rope) for _ in range(cfg.dec_blocks)])
        self.norm = nn.LayerNorm(d)
        if cfg.rgb_head == "linear":
            self.rgb_head = nn.Linear(d, cfg.patch * cfg.patch * 3)
        else:
            stages = len(cfg.conv_channels) - 1
            assert self.grid * 2 ** stages == cfg.image_size, (
                f"conv_channels gives {stages} x2 stages: grid {self.grid} * 2^{stages} != image_size "
                f"{cfg.image_size} (need len(conv_channels) = log2(patch) + 1 = {int(math.log2(cfg.patch)) + 1})")
            self.rgb_conv = ConvUpsampler(d, tuple(cfg.conv_channels), cfg.conv_res_blocks, cfg.conv_channels_last)
        self.feat_head = nn.Linear(d, feat_dim)

    def d_tokens(self, zd: Optional[torch.Tensor], zd_len: Optional[torch.Tensor] = None) -> tuple:
        """Projected D tokens [B, M (+1), dim] and the keys each row may read, [B, M (+1)] bool | None.
        zd_len [B] long: the prefix of the D tokens visible to each row (nested dropout), None = all. With
        cfg.d_nested the learned null token is appended and always visible, so an empty prefix is a code."""
        if zd is None:
            return None, None
        d = self.in_d(zd)
        B, M = d.shape[:2]
        null = getattr(self, "d_null", None)
        if null is not None:
            d = torch.cat([d, null.expand(B, -1, -1).to(d.dtype)], 1)
        if zd_len is None:
            return d, None
        zd_len = zd_len.to(d.device)
        if null is None and bool((zd_len <= 0).any()):
            raise ValueError("an empty D prefix needs d_nested (the null token): without it a row has no key to read")
        valid = torch.arange(M, device=d.device)[None] < zd_len[:, None]   # [B,M]
        if null is not None:
            valid = torch.cat([valid, valid.new_ones(B, 1)], 1)
        return d, valid

    def compute_delta(self, d: Optional[torch.Tensor], q_coords: torch.Tensor,
                      q_rays: Optional[torch.Tensor] = None, d_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Δ [B, Q, 2] in (u, v) units from the projected D tokens; 0 when there is no D.
        q_rays [B, Q, 6] enter only the "query" mode; d_valid [B, M] bool: the D tokens each row may read."""
        if d is None:
            return torch.zeros_like(q_coords[..., 1:3])
        B, Q = q_coords.shape[:2]
        if self.cfg.delta_mode == "mean":
            if d_valid is None:
                dm = d.mean(1, keepdim=True)
            else:
                w = d_valid.to(d.dtype).unsqueeze(-1)
                dm = (d * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)
            dm = dm.expand(-1, Q, -1)
            return self.offset(torch.cat([dm, fourier_uv(q_coords[..., 1:3], self.n_fourier).to(dm.dtype)], -1))
        f = fourier_uv(q_coords[..., 1:3], self.n_fourier).to(d.dtype)       # [B,Q,nf]
        q = self.delta_q(f)
        if q_rays is not None:
            q = q + self.delta_ray(q_rays.to(q.dtype))
        rd = self.delta_read(self.delta_n_q(q), self.delta_n_d(d), k_valid=d_valid)   # [B,Q,D] own read of D
        return self.delta_mlp(torch.cat([rd, f], -1))

    def forward(self, zc: torch.Tensor, zd: Optional[torch.Tensor], q_coords: torch.Tensor,
                q_mask: torch.Tensor, modality: str, delta_override: Optional[torch.Tensor] = None,
                q_rays: Optional[torch.Tensor] = None, warp_only: bool = False,
                zd_len: Optional[torch.Tensor] = None) -> dict:
        """zc [B, N, zr_c], zd [B, M, zr_d] | None, q_coords / q_mask [B, Q, 4] (each row its own t).
        q_rays [B, Q, 6]: Plücker rays of the target view's queries (posed 3D views), else None.
        warp_only: with zd, blocks add readC only (gate 1, readD skipped); Δ still comes from zd.
        zd_len [B] long | None: each row reads only the first zd_len D tokens (nested dropout; see d_tokens).

        Returns rgb [B, Q, p*p*3], image [B, 3, H, W] (conv head) | None (linear head),
        feat [B, Q, feat_dim], delta [B, Q, 2], gate [B, Q, 1] (last block)."""
        B, Q = q_coords.shape[:2]
        if self.cfg.rgb_head == "conv":
            assert Q == self.cfg.n_patches, f"conv rgb head decodes the full grid: Q = {Q} != N = {self.cfg.n_patches}"
        if modality == "image":
            assert zc.shape[-1] == self.cfg.zr_c_image, "image zc must have zr_c_image channels"
            c = self.in_c_image(zc)
        else:
            assert zc.shape[-1] == self.cfg.zr_c_video, "video zc must have zr_c_video channels"
            c = self.in_c_video(zc)
        d, d_valid = self.d_tokens(zd, zd_len)
        delta = self.compute_delta(d, q_coords, q_rays, d_valid) if delta_override is None else delta_override

        # window read of C at the shifted centre (computed once, shared by all blocks)
        centre = q_coords[..., 1:3] + delta                                    # [B,Q,2]
        uv = torch.tensor(UV_MASK, device=q_mask.device, dtype=q_mask.dtype)
        q_m = q_mask * uv                                                      # (u, v) bits only
        q_pos = torch.cat([q_coords[..., :1], centre, q_coords[..., 3:]], -1)
        if self.cfg.window_read == "gather":                                   # the (2r+1)^2 cells per query
            idx, valid, cell = gather_window(centre, self.grid, self.r)
            W = idx.shape[-1]
            k_pos = torch.cat([q_coords[..., :1].unsqueeze(2).expand(-1, -1, W, -1), cell.to(q_coords.dtype),
                               torch.zeros(*cell.shape[:-1], 1, device=cell.device, dtype=q_coords.dtype)], -1)
            win = (idx, valid, q_pos, k_pos, q_m, q_m.unsqueeze(2).expand(-1, -1, W, -1))
        else:                                                                  # dense over all N cells, masked
            N = c.shape[1]
            c_pos = grid_coords(self.grid, 0.0, device=q_coords.device).to(q_coords.dtype)
            c_pos = torch.cat([q_coords[:, :1, :1].expand(-1, N, -1), c_pos[None, :, 1:].expand(B, -1, -1)], -1)
            c_m = q_m[:, :1].expand(-1, N, -1)                                 # cells share the queries' mask
            win = (window_bias(centre, self.grid, self.r, c.dtype), q_pos, q_m, c_pos, c_m)

        h = self.query.expand(B, Q, -1).to(c.dtype)
        if q_rays is not None:
            h = h + self.ray_q(q_rays.to(c.dtype))
        d_blk = None if warp_only else d                                       # warp-only: D acts via Δ alone
        gate = None
        for blk in self.blocks:
            h, gate = blk(h, c, d_blk, win, q_coords, q_mask, d_valid)
        h = self.norm(h)
        if self.cfg.rgb_head == "linear":
            rgb, image = self.rgb_head(h), None
        else:
            image = self.rgb_conv(h, self.grid)
            rgb = image_to_patches(image, self.cfg.patch)
        return dict(rgb=rgb, image=image, feat=self.feat_head(h), delta=delta, gate=gate)
