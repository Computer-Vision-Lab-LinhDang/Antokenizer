"""Memory transition (section 9): a diagonal, input-conditioned recurrence on the 16 coarse slots that can be
replayed from the serialized state, and the memory-conditioned grain appearance (9.2).

    lambda_{i,m} = softplus(W_lambda [b_{i,m}; e_m] + a_lambda),   v_{i,m} = tanh(W_v [b_{i,m}; e_m] + a_v)
    alpha_{i,m}  = exp(-lambda_{i,m} delta_i)
    z_{i,m}      = alpha_{i,m} * z_{i-1,m} + (1 - alpha_{i,m}) * v_{i,m}               (float32)

U(z, b, delta) depends only on the stored b, the elapsed time, the predecessor and shared weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as g
from .config import GrainPriorConfig
from .layers import CrossAttn, mlp2

E_SLOT = 32          # width of the shared slot embedding e_m


class Recurrence(nn.Module):
    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        self.cfg = cfg
        self.e_slot = nn.Parameter(torch.randn(cfg.n_coarse, E_SLOT) * 0.02)
        self.lam = nn.Linear(cfg.d_b + E_SLOT, cfg.d_coarse)
        self.val = nn.Linear(cfg.d_b + E_SLOT, cfg.d_coarse)

    def forward(self, z: torch.Tensor, b: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """U(z [B, 16, C], b [B, 16, d_b] (or [B, 32]), delta [B] seconds) -> z' [B, 16, C], float32."""
        B = z.shape[0]
        b = b.reshape(B, self.cfg.n_coarse, self.cfg.d_b)
        x = torch.cat([b.float(), self.e_slot.float().expand(B, -1, -1)], -1)
        with g.fp32(z.device):
            lam = F.softplus(F.linear(x, self.lam.weight.float(), self.lam.bias.float()))
            v = torch.tanh(F.linear(x, self.val.weight.float(), self.val.bias.float()))
            a = torch.exp(-lam * delta.float().view(B, 1, 1))
            return a * z.float() + (1 - a) * v


class Appearance(nn.Module):
    """A(f, u, z, c_0): a grain query cross-attends the 16 memory tokens and projects to d_f channels;
    f~ = f + [A(f, u, z, c_0) - A(f, u, c_0, c_0)], exactly f when z = c_0."""

    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.q = mlp2(cfg.d_f + g.pe_dim(2, cfg.pe_freq) + cfg.d_coarse, d, cfg.mlp_hidden)
        self.mem = nn.Linear(cfg.d_coarse, d)
        self.slot = nn.Parameter(torch.randn(1, cfg.n_coarse, d) * 0.02)
        self.ca = CrossAttn(d, cfg.heads)
        self.out = nn.Linear(d, cfg.d_f)

    def A(self, f, u, z, c0) -> torch.Tensor:
        q = self.q(torch.cat([f, g.fourier(u, self.cfg.pe_freq).to(f.dtype),
                              g.read_coarse(c0, u, self.cfg.coarse_grid).to(f.dtype)], -1))
        m = self.mem(z.to(q.dtype)) + self.slot
        return self.out(q + self.ca(q, m))

    def forward(self, f: torch.Tensor, u: torch.Tensor, z: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        return f + (self.A(f, u, z, c0) - self.A(f, u, c0, c0))
