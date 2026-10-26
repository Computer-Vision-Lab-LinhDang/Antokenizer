"""Stage 5: dual-space residual and the ordered innovation slots (section 10).

    e^F = F_i - sg F^base,   e^RGB = X_i - sg X^base
    E^r = LN(e^F W_r + ResidualPixelStem(e^RGB) + PE_2D)
    52 learned slot identities conditioned on z_i, the pooled residual and the candidate geometry; four blocks:
    cross-attention to E^r, slot self-attention with the causal mask j' <= j, MLP. Each slot -> 24 scalars:
    u = sigmoid (2), rho (same convention and bounds as the grains), sigma = 0.4 + 1.1 sigmoid, e (20, linear).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import geometry as g
from .anchor import PixelStem
from .config import GrainPriorConfig
from .layers import Block, run

N_GEOM = 6 + 4 + 1


class InnovationEncoder(nn.Module):
    def __init__(self, cfg: GrainPriorConfig, teacher_dim: int):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.w_r = nn.Linear(teacher_dim, d)
        self.stem = PixelStem(cfg.stem_channels)
        self.w_pe = nn.Linear(g.pe_dim(2, cfg.pe_freq), d)
        self.norm_e = nn.LayerNorm(d)
        self.slots = nn.Parameter(torch.randn(1, cfg.n_slots, d) * 0.02)
        self.c_z, self.c_res, self.c_geo = nn.Linear(cfg.d_coarse, d), nn.Linear(d, d), nn.Linear(N_GEOM, d)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, 1, True, cfg.mlp_ratio) for _ in range(cfg.res_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.d_r)
        self.register_buffer("uq", g.cell_centres(cfg.grid), persistent=False)
        self.register_buffer("causal", torch.tril(torch.ones(cfg.n_slots, cfg.n_slots, dtype=torch.bool)),
                             persistent=False)

    def forward(self, eF: torch.Tensor, eRGB: torch.Tensor, z: torch.Tensor, xi, kappa, M) -> torch.Tensor:
        """eF [B, N, D_t], eRGB [B, 3, H, W], z_i [B, 16, dc], candidate geometry -> r [B, n_slots, 24]."""
        cfg = self.cfg
        B = eF.shape[0]
        P = self.stem(eRGB)
        E = self.norm_e(self.w_r(eF.to(P.dtype)) + P + self.w_pe(g.fourier(self.uq, cfg.pe_freq)).to(P.dtype))
        geo = torch.cat([xi, kappa, M.norm(dim=-1).mean(-1, keepdim=True)], -1).float()
        s = self.slots + (self.c_z(z.float().mean(1)) + self.c_res(E.float().mean(1)) + self.c_geo(geo)).to(P.dtype).unsqueeze(1)
        s = self.norm(run(self.blocks, s, [(E, None, None)], self.causal.unsqueeze(0).expand(B, -1, -1)))
        o = self.head(s).float()
        u = torch.sigmoid(o[..., :2])
        rho = cfg.rho_lo + (cfg.rho_hi - cfg.rho_lo) * torch.sigmoid(o[..., 2:3])
        sigma = cfg.sigma_min + cfg.sigma_span * torch.sigmoid(o[..., 3:4])
        return torch.cat([u, rho, sigma, o[..., 4:]], -1)
