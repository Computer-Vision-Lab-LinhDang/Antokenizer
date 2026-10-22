"""Tier B: plain R0 trunk (RGAT postponed).

Each layer, a token of frame t attends to all N tokens of frame t plus the tokens of frame t-1
in the (2r+1)^2 grid window around its own cell. Causal and parallel over T: keys are
[x_t ; x_{t-1}] (frame -1 = zeros, fully masked) with one boolean mask [N, 2N] built from the grid.
RoPE uses each key's own frame coords, so Delta t is visible. T = 1 is plain full attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import DinoV0Config
from .rope import AxialRoPE


def window_mask(grid: int, r: int, device=None) -> torch.Tensor:
    """[N, N] bool: query cell i may read previous-frame cell j (Chebyshev distance <= r)."""
    idx = torch.arange(grid * grid, device=device)
    row, col = idx // grid, idx % grid
    return ((row[:, None] - row[None]).abs() <= r) & ((col[:, None] - col[None]).abs() <= r)


def _shift(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Shift by one along dim (frame t gets frame t-1); frame 0 gets zeros."""
    pad = torch.zeros_like(x.narrow(dim, 0, 1))
    return torch.cat([pad, x.narrow(dim, 0, x.shape[dim] - 1)], dim)


class TrunkLayer(nn.Module):
    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        d, self.h = cfg.dim, cfg.heads
        self.hd = d // cfg.heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        hidden = int(d * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.rope = AxialRoPE(self.hd)

    def attn(self, x: torch.Tensor, ang: torch.Tensor, ang_prev: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        q, k, v = self.qkv(x).view(B, T, N, 3, self.h, self.hd).permute(3, 0, 1, 4, 2, 5)  # [B,T,H,N,hd]
        rot = lambda z, a: self.rope.rotate(z.reshape(B * T, self.h, N, self.hd),
                                            a.repeat(B, 1, 1)).view(B, T, self.h, N, self.hd)
        q, k_cur, k_prev = rot(q, ang), rot(k, ang), rot(_shift(k, 1), ang_prev)
        k = torch.cat([k_cur, k_prev], 3)
        v = torch.cat([v, _shift(v, 1)], 3)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)       # mask [T,1,N,2N]
        return self.out(o.transpose(2, 3).reshape(B, T, N, D))

    def forward(self, x, ang, ang_prev, mask):
        x = x + self.attn(self.norm1(x), ang, ang_prev, mask)
        return x + self.mlp(self.norm2(x))


class Trunk(nn.Module):
    """x [B, T, N, dim], coords [T, N, 4], mask [T, N, 4] -> [B, T, N, dim]."""

    def __init__(self, cfg: DinoV0Config, use_checkpoint: bool = False):
        super().__init__()
        assert cfg.dim % cfg.heads == 0
        self.r, self.use_checkpoint = cfg.temporal_window, use_checkpoint
        self.layers = nn.ModuleList(TrunkLayer(cfg) for _ in range(cfg.trunk_layers))
        self.norm = nn.LayerNorm(cfg.dim)

    def attn_mask(self, T: int, N: int, device=None) -> torch.Tensor:
        grid = int(round(N ** 0.5))
        assert grid * grid == N, "tokens must form a square grid"
        prev = window_mask(grid, self.r, device).expand(T, N, N).clone()
        prev[0] = False                                                    # frame 0 has no past
        cur = torch.ones(T, N, N, dtype=torch.bool, device=device)
        return torch.cat([cur, prev], -1).unsqueeze(1)                      # [T, 1, N, 2N]

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = x.shape
        ang = self.layers[0].rope.angles(coords, mask)                     # [T, N, hd/2], same for all layers
        ang_prev = _shift(ang, 0)
        am = self.attn_mask(T, N, x.device)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, ang, ang_prev, am, use_reentrant=False)
            else:
                x = layer(x, ang, ang_prev, am)
        return self.norm(x)
