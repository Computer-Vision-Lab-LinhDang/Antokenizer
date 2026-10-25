"""Residual encoder Enc_R: the second stack, run after the trial decode.

  e     = X_t - sg[X~_t]                         DINO residual of the predicted frame (geometry + previous frame)
  slots = L_max learned queries, read [e ; sg[X~_t] ; sg[c_k]] as read-only context
  D_innov = (mu, logvar) of L_max tokens x d_r, kept as a prefix of random length L (nested dropout)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import cell_coords
from .config import CD32Config
from .layers import Rope3, StreamBlock


class ResidualEncoder(nn.Module):
    def __init__(self, cfg: CD32Config, feat_dim: int):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.norm_e, self.norm_x = nn.LayerNorm(feat_dim), nn.LayerNorm(feat_dim)
        self.res_in = nn.Linear(2 * feat_dim, d)
        self.c_in = nn.Linear(cfg.d_c_video, d)
        self.ctx_type = nn.Parameter(torch.randn(2, d) * 0.02)
        self.slots = nn.Parameter(torch.randn(1, cfg.L_max, d) * 0.02)
        self.blocks = nn.ModuleList([StreamBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.enc_r_layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2 * cfg.d_r)
        self.rope = Rope3(cfg.rope_dims, cfg.rope_base)
        cc = cell_coords(cfg.grid, 0.0)
        mask = torch.tensor([0.0, 1.0, 1.0]).expand_as(cc)
        self.register_buffer("ctx_coords", torch.cat([cc, cc], 0), persistent=False)
        self.register_buffer("ctx_mask", torch.cat([mask, mask], 0).clone(), persistent=False)

    def forward(self, e: torch.Tensor, x_pred: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """e, x_pred [B, N, feat_dim] (x_pred detached), c [B, N, d_c] (detached) -> mu, logvar [B, L_max, d_r]."""
        B = e.shape[0]
        ctx = torch.cat([self.res_in(torch.cat([self.norm_e(e), self.norm_x(x_pred)], -1)) + self.ctx_type[0],
                         self.c_in(c) + self.ctx_type[1]], 1)
        rope = self.rope.tables(self.ctx_coords, self.ctx_mask, dtype=ctx.dtype)
        s = self.slots.expand(B, -1, -1)
        for blk in self.blocks:
            s = blk(s, None, ctx, rope)
        mu, logvar = self.out(self.norm(s)).float().chunk(2, -1)
        return mu, logvar.clamp(-30, 20)
