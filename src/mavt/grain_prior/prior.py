"""Stage 8: deterministic causal prior (section 13).

    D^_{t+dt} = P_psi(S_t, dt, q_allowed),   S^_{t+dt} = AppendAndFold(S_t, D^_{t+dt})

Six blocks, width 512, cross-attention to the bank rebuilt from S_t. Output queries: 1 camera, 16 motion,
1 memory and 52 innovation queries, conditioned on the pooled state and the time offset; self-attention is
causal along the innovation order with the control tokens first. Heads use the posterior parameterisation.
Static novel views: the query camera is copied, M = 0, only b and r are predicted.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import geometry as g
from .config import GrainPriorConfig
from .layers import Block, mlp2, run, zero_last
from .trunk import ControlQueries, time_pe


class CausalPrior(nn.Module):
    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        n_ctl = 2 + cfg.n_motion
        self.q = nn.Parameter(torch.randn(1, n_ctl + cfg.n_slots, d) * 0.02)
        self.t = nn.Linear(g.pe_dim(1, cfg.pe_freq), d)
        self.cam = nn.Linear(6 + 4, d)
        self.pool = nn.Linear(d, d)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, 1, True, cfg.mlp_ratio) for _ in range(cfg.prior_layers)])
        self.norm = nn.LayerNorm(d)
        self.ctl = ControlQueries(cfg)
        self.r = zero_last(mlp2(d, cfg.d_r, cfg.mlp_hidden))
        n = n_ctl + cfg.n_slots
        m = torch.zeros(n, n, dtype=torch.bool)
        m[:n_ctl, :n_ctl] = True
        m[n_ctl:, :n_ctl] = True
        m[n_ctl:, n_ctl:] = torch.tril(torch.ones(cfg.n_slots, cfg.n_slots, dtype=torch.bool))
        self.register_buffer("mask", m, persistent=False)

    def forward(self, bank: tuple, dt: torch.Tensor, cam: tuple = None) -> dict:
        """bank (tokens, valid) of S_t; dt [B] seconds; cam (xi, kappa) of a static novel view or None.
        -> raw record fields dict(xi, kz, M, b, r_raw)."""
        cfg = self.cfg
        tok, valid = bank
        B = tok.shape[0]
        pooled = (tok * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        c = self.t(time_pe(dt, cfg.pe_freq)) + self.pool(pooled.float())
        if cam is not None:
            c = c + self.cam(torch.cat(cam, -1).float())
        h = self.q.expand(B, -1, -1) + c.unsqueeze(1).to(tok.dtype)
        h = self.norm(run(self.blocks, h, [(tok, None, valid)], self.mask.unsqueeze(0).expand(B, -1, -1)))
        n_ctl = 2 + cfg.n_motion
        out = self.ctl(h[:, 0], h[:, 1:n_ctl - 1], h[:, n_ctl - 1])
        out["r_raw"] = self.r(h[:, n_ctl:]).float()
        return out
