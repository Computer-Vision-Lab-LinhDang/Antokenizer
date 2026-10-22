"""Tier E: query decoder. Reads ONLY the compressed latent z_r.

For a query at (t, u, v) of frame t in a video segment the decoder gets zc = z_r(C_k) (keyframe
content grid) and zd = z_r(D_t) (how frame t deviates from C_k). Per query:

  Δ      = MLP([mean_M in_d(zd) ; fourier(u, v)])        last layer zero-init -> Δ = 0 at init
  readC  = cross-attn to the (2r+1)^2 C cells around the cell containing (u, v) + Δ; the query
           sits at the SHIFTED centre under AxialRoPE, so the gradient reaches Δ through RoPE
  readD  = cross-attn to in_d(zd) tokens by content (no RoPE)
  o      = sigmoid(Linear([h ; readC]))                   bias init cfg.gate_init_bias
  h     += o * readC + (1 - o) * readD                     (images: h += readC, o = 1)
  h     += SelfAttn(h) (RoPE on the unshifted coords); h += MLP(h)       pre-LN everywhere

Heads: rgb [B, Q, p*p*3] and feat [B, Q, feat_dim]. Coordinates: u = column, v = row, in [0, 1].
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


class _WindowCrossAttn(nn.Module):
    """Query -> its own gathered (2r+1)^2 C cells, AxialRoPE on (shifted centre, cell centres)."""

    def __init__(self, dim: int, heads: int, rope: AxialRoPE):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.q, self.kv, self.o = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x, c, idx, valid, q_pos, k_pos, q_m, k_m):
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

    def forward(self, x, ctx, q_pos=None, q_m=None, k_pos=None, k_m=None):
        B, Q, D = x.shape
        N = ctx.shape[1]
        q = self.q(x).view(B, Q, self.h, self.hd).transpose(1, 2)
        k, v = self.kv(ctx).view(B, N, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.rope is not None and q_pos is not None:
            q, k = self.rope(q, q_pos, q_m), self.rope(k, k_pos, k_m)
        out = F.scaled_dot_product_attention(q, k, v)
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

    def forward(self, h, c, d, win, q_coords, q_mask):
        idx, valid, q_pos, k_pos, q_m, k_m = win
        hn = self.n_q(h)
        rc = self.read_c(hn, self.n_c(c), idx, valid, q_pos, k_pos, q_m, k_m)
        if d is None:
            o = torch.ones_like(h[..., :1])
            h = h + rc
        else:
            rd = self.read_d(self.n_qd(h), self.n_d(d))
            o = torch.sigmoid(self.gate(torch.cat([hn, rc], -1)))
            h = h + o * rc + (1 - o) * rd
        hs = self.n_s(h)
        h = h + self.self_attn(hs, hs, q_coords, q_mask, q_coords, q_mask)
        h = h + self.mlp(self.n_m(h))
        return h, o


class QueryDecoder(nn.Module):
    """z_r (C, D) + query coords -> per-query rgb patch, DINO feature, offset Δ and gate o."""

    def __init__(self, cfg: DinoV0Config, feat_dim: int):
        super().__init__()
        self.ray_q = ZeroLinear(6, cfg.dec_dim)        # query rays for posed views; zero at init
        d = cfg.dec_dim
        self.cfg, self.grid, self.r = cfg, cfg.grid, cfg.read_window
        self.in_c_image = nn.Linear(cfg.zr_c_image, d)
        self.in_c_video = nn.Linear(cfg.zr_c_video, d)
        self.in_d = nn.Linear(cfg.zr_d, d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.n_fourier = 8
        self.offset = nn.Sequential(nn.Linear(d + 2 + 4 * self.n_fourier, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.rope = AxialRoPE(d // cfg.dec_heads)
        self.blocks = nn.ModuleList([DecoderBlock(cfg, self.rope) for _ in range(cfg.dec_blocks)])
        self.norm = nn.LayerNorm(d)
        self.rgb_head = nn.Linear(d, cfg.patch * cfg.patch * 3)
        self.feat_head = nn.Linear(d, feat_dim)

    def compute_delta(self, d: Optional[torch.Tensor], q_coords: torch.Tensor) -> torch.Tensor:
        """Δ [B, Q, 2] in (u, v) units from the projected D tokens; 0 when there is no D."""
        if d is None:
            return torch.zeros_like(q_coords[..., 1:3])
        B, Q = q_coords.shape[:2]
        dm = d.mean(1, keepdim=True).expand(-1, Q, -1)
        return self.offset(torch.cat([dm, fourier_uv(q_coords[..., 1:3], self.n_fourier).to(dm.dtype)], -1))

    def forward(self, zc: torch.Tensor, zd: Optional[torch.Tensor], q_coords: torch.Tensor,
                q_mask: torch.Tensor, modality: str, delta_override: Optional[torch.Tensor] = None,
                q_rays: Optional[torch.Tensor] = None) -> dict:
        """q_rays [B, Q, 6]: Plücker rays of the target view's queries (posed 3D views), else None."""
        B, Q = q_coords.shape[:2]
        if modality == "image":
            assert zc.shape[-1] == self.cfg.zr_c_image, "image zc must have zr_c_image channels"
            c = self.in_c_image(zc)
        else:
            assert zc.shape[-1] == self.cfg.zr_c_video, "video zc must have zr_c_video channels"
            c = self.in_c_video(zc)
        d = None if zd is None else self.in_d(zd)
        delta = self.compute_delta(d, q_coords) if delta_override is None else delta_override

        # window read of C at the shifted centre (computed once, shared by all blocks)
        centre = q_coords[..., 1:3] + delta                                    # [B,Q,2]
        idx, valid, cell = gather_window(centre, self.grid, self.r)            # [B,Q,W], [B,Q,W,2]
        uv = torch.tensor(UV_MASK, device=q_mask.device, dtype=q_mask.dtype)
        q_m = q_mask * uv                                                      # (u, v) bits only
        q_pos = torch.cat([q_coords[..., :1], centre, q_coords[..., 3:]], -1)
        W = idx.shape[-1]
        k_pos = torch.cat([q_coords[..., :1].unsqueeze(2).expand(-1, -1, W, -1), cell.to(q_coords.dtype),
                           torch.zeros_like(cell[..., :1]).to(q_coords.dtype)], -1)  # [B,Q,W,4]
        k_m = q_m.unsqueeze(2).expand(-1, -1, W, -1)
        win = (idx, valid, q_pos, k_pos, q_m, k_m)

        h = self.query.expand(B, Q, -1).to(c.dtype)
        if q_rays is not None:
            h = h + self.ray_q(q_rays.to(c.dtype))
        gate = None
        for blk in self.blocks:
            h, gate = blk(h, c, d, win, q_coords, q_mask)
        h = self.norm(h)
        return dict(rgb=self.rgb_head(h), feat=self.feat_head(h), delta=delta, gate=gate)
