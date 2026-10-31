"""3.13: per-grain feature update of a target record (cfg.grain_df_k = k > 0).

For each anchor grain transported to the target, a code dfc [B, G, k] is read from the target at the grain's
transported position u_t:
    in  = [LN(F_t(u_t)) ; LN(e^F(u_t)) ; RGB_t footprint (8 x 8 samples over the grain's stratum window, read at a
           detached u_t: grid_sample's backward is slow on ROCm and the feature reads already train u_t) ; f_t]
    dfc = MLP(in) * valid
and the decoder reads the grain with f_t + dfc W (W: k -> d_f). W starts at zero: until it moves, a record with a
code decodes exactly as one without (the channel is an identity at initialisation of a pretrained model).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as g
from .config import GrainPriorConfig

FOOT = 8                                          # footprint samples per side


class GrainDelta(nn.Module):
    def __init__(self, cfg: GrainPriorConfig, teacher_dim: int):
        super().__init__()
        d, k = cfg.dim, cfg.grain_df_k
        self.grid, self.win = cfg.grid, cfg.stratum / cfg.grid
        self.n_F, self.n_e = nn.LayerNorm(teacher_dim), nn.LayerNorm(teacher_dim)
        self.inp = nn.Linear(2 * teacher_dim + 3 * FOOT * FOOT + cfg.d_f, d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.out = nn.Linear(d, k)
        self.W = nn.Linear(k, cfg.d_f, bias=False)
        nn.init.zeros_(self.W.weight)
        o = (torch.arange(FOOT, dtype=torch.float32) + 0.5) / FOOT - 0.5
        oy, ox = torch.meshgrid(o, o, indexing="ij")
        self.register_buffer("off", torch.stack([ox, oy], -1).view(1, 1, -1, 2), persistent=False)

    def footprint(self, X: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """X [B, 3, H, W] in [0, 1], u [B, G, 2] -> [B, G, 3 * FOOT^2] (border padding outside the image)."""
        pts = (u.float().unsqueeze(2) + self.off * self.win) * 2 - 1                    # [B, G, P, 2]
        s = F.grid_sample(X.float(), pts, mode="bilinear", padding_mode="border", align_corners=False)
        return s.permute(0, 2, 1, 3).flatten(2)                                          # [B, G, 3 P]

    def encode(self, F_t: torch.Tensor, eF: torch.Tensor, X: torch.Tensor, f: torch.Tensor, u: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
        """F_t, eF [B, N, D_t] on the cell grid; X [B, 3, H, W] in [0, 1]; f [B, G, d_f]; u [B, G, 2];
        valid [B, G] -> dfc [B, G, k] (zero where the grain is not valid)."""
        uc = u.float().clamp(0, 1)
        x = torch.cat([self.n_F(g.read_coarse(F_t.float(), uc, self.grid)),
                       self.n_e(g.read_coarse(eF.float(), uc, self.grid)),
                       self.footprint(X, uc.detach()), f.float()], -1)   # no grid_sampler backward
        return self.out(self.mlp(self.inp(x))) * valid.float().unsqueeze(-1)

    def expand(self, dfc: torch.Tensor) -> torch.Tensor:
        """dfc [B, G, k] -> feature update [B, G, d_f]."""
        return self.W(dfc.float())
