"""Stream D of the conditional two-stream trunk (sections 6.3, 6.4).

The previous state is rehydrated into a finite token bank B_{i-1} = PackTokens(G_0, c_0, z_{i-1}, {D_j}): each
field projected to the trunk width with a type embedding, the innovation slots of a record read through
their valid prefix only. Static views use B_i^view = PackTokens(G_0, c_0, kappa_0, T_i, kappa_i) instead (no
earlier records, no temporal memory). Before reading observation i, the posterior initializer
P_init(S_{i-1}, dt_i, q_i) proposes the record (identity / zero at initialisation). Stream D holds 274 query
tokens: 256 content queries, 1 pose, 16 motion controls, 1 memory update query; each of its six blocks
cross-attends B_{i-1}, then Y_i, then self-attends.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from . import geometry as g
from .config import GrainPriorConfig
from .layers import Block, CrossAttn, mlp2, run, zero_last

TOK_GRAIN, TOK_COARSE, TOK_MEM, TOK_POSE, TOK_MOTION, TOK_MEMB, TOK_INNOV, TOK_CAM0, TOK_CAMT = range(9)


def time_pe(t: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Time (seconds relative to the anchor) [...] -> Fourier features [..., 2 n_freq]."""
    return g.fourier(t.float().unsqueeze(-1) / 4.0, n_freq)


class Bank(nn.Module):
    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        self.cfg = cfg
        d, pf = cfg.dim, cfg.pe_freq
        pe1, pe2 = g.pe_dim(1, pf), g.pe_dim(2, pf)
        self.typ = nn.Parameter(torch.randn(9, d) * 0.02)
        self.grain = nn.Linear(cfg.d_f + pe2 + 2, d)
        self.coarse = nn.Linear(cfg.d_coarse, d)
        self.mem = nn.Linear(cfg.d_coarse, d)
        self.slot = nn.Parameter(torch.randn(1, cfg.n_coarse, d) * 0.02)
        self.pose = nn.Linear(6 + 4 + pe1, d)
        self.motion = nn.Linear(3 + pe1, d)
        self.m_idx = nn.Parameter(torch.randn(1, cfg.n_motion, d) * 0.02)
        self.memb = nn.Linear(cfg.n_memb + pe1, d)
        self.innov = nn.Linear(cfg.d_e + pe2 + 2 + pe1, d)
        self.r_idx = nn.Parameter(torch.randn(1, cfg.n_slots, d) * 0.02)
        self.cam0 = nn.Linear(4, d)
        self.camt = nn.Linear(6 + 4, d)

    def anchor_tokens(self, S: dict) -> list:
        cfg = self.cfg
        gr = self.grain(torch.cat([S["f"], g.fourier(S["u"], cfg.pe_freq), S["rho"].unsqueeze(-1),
                                   S["sigma"].unsqueeze(-1)], -1).float()) + self.typ[TOK_GRAIN]
        co = self.coarse(S["c0"].float()) + self.slot + self.typ[TOK_COARSE]
        return [gr, co]

    def record_tokens(self, rec: dict, t: torch.Tensor) -> tuple:
        """One record (t [B] its time offset) -> (tokens [B, 18 + n_slots, d], valid [B, 18 + n_slots])."""
        cfg = self.cfg
        B = rec["xi"].shape[0]
        tp = time_pe(t, cfg.pe_freq)
        pose = self.pose(torch.cat([rec["xi"], rec["kappa"], tp], -1).float()).unsqueeze(1) + self.typ[TOK_POSE]
        mo = self.motion(torch.cat([rec["M"], tp.unsqueeze(1).expand(-1, cfg.n_motion, -1)], -1).float())
        mo = mo + self.m_idx + self.typ[TOK_MOTION]
        mb = self.memb(torch.cat([rec["b"].reshape(B, -1), tp], -1).float()).unsqueeze(1) + self.typ[TOK_MEMB]
        r = rec["r"].float()
        inn = self.innov(torch.cat([r[..., 4:], g.fourier(r[..., :2], cfg.pe_freq), r[..., 2:4],
                                    tp.unsqueeze(1).expand(-1, cfg.n_slots, -1)], -1))
        inn = inn + self.r_idx + self.typ[TOK_INNOV]
        ok = torch.arange(cfg.n_slots, device=r.device).view(1, -1) < rec["L"].view(-1, 1)
        tok = torch.cat([pose, mo, mb, inn], 1)
        valid = torch.cat([torch.ones(B, 2 + cfg.n_motion, dtype=torch.bool, device=r.device), ok], 1)
        return tok, valid

    def temporal(self, S: dict, z_prev: torch.Tensor, times: list) -> tuple:
        """B_{i-1} = PackTokens(G_0, c_0, z_{i-1}, {D_j}): (tokens [B, nb, d], valid [B, nb])."""
        toks = self.anchor_tokens(S)
        toks.append(self.mem(z_prev.float()) + self.slot + self.typ[TOK_MEM])
        B = toks[0].shape[0]
        valid = [torch.ones(B, sum(t.shape[1] for t in toks), dtype=torch.bool, device=toks[0].device)]
        for rec, t in zip(S["recs"], times):
            tk, v = self.record_tokens(rec, t)
            toks.append(tk)
            valid.append(v)
        return torch.cat(toks, 1), torch.cat(valid, 1)

    def view(self, S: dict, xi_t: torch.Tensor, kappa_t: torch.Tensor) -> tuple:
        """B_i^view = PackTokens(G_0, c_0, kappa_0, T_i, kappa_i)."""
        toks = self.anchor_tokens(S)
        toks.append(self.cam0(S["kappa0"].float()).unsqueeze(1) + self.typ[TOK_CAM0])
        toks.append(self.camt(torch.cat([xi_t, kappa_t], -1).float()).unsqueeze(1) + self.typ[TOK_CAMT])
        tok = torch.cat(toks, 1)
        return tok, torch.ones(tok.shape[:2], dtype=torch.bool, device=tok.device)


class ControlQueries(nn.Module):
    """1 pose + n_motion + 1 memory query tokens -> heads (zero-init: identity camera, no motion, no update)."""

    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        d, h = cfg.dim, cfg.mlp_hidden
        self.cfg = cfg
        self.xi = zero_last(mlp2(d, 6, h))
        self.kz = zero_last(mlp2(d, 4, h))
        self.M = zero_last(mlp2(d, 3, h))
        self.b = zero_last(mlp2(d, cfg.n_memb, h))

    def forward(self, pose: torch.Tensor, motion: torch.Tensor, memb: torch.Tensor) -> dict:
        return dict(xi=self.xi(pose).float(), kz=self.kz(pose).float(), M=self.M(motion).float(),
                    b=self.b(memb).float())


class PInit(nn.Module):
    """Posterior initializer P_init(S_{i-1}, dt_i, q_i): reads the bank with the control queries."""

    def __init__(self, cfg: GrainPriorConfig, n_blocks: int = 2):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.q = nn.Parameter(torch.randn(1, 2 + cfg.n_motion, d) * 0.02)
        self.t = nn.Linear(g.pe_dim(1, cfg.pe_freq), d)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, 1, True, cfg.mlp_ratio) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d)
        self.heads = ControlQueries(cfg)

    def forward(self, bank: tuple, dt: torch.Tensor) -> dict:
        B = bank[0].shape[0]
        h = self.q.expand(B, -1, -1) + self.t(time_pe(dt, self.cfg.pe_freq)).unsqueeze(1)
        h = self.norm(run(self.blocks, h, [(bank[0], None, bank[1])]))
        return self.heads(h[:, 0], h[:, 1:-1], h[:, -1])


class StreamD(nn.Module):
    """274 query tokens initialised by f_init(PE(q), Read(B, q), D~_past, g_i); six blocks over (B, Y_i)."""

    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        self.cfg = cfg
        d, dc, nm = cfg.dim, cfg.d_coarse, cfg.n_motion
        self.n_tok = cfg.n_cells + 2 + nm
        self.q_pos = nn.Linear(g.pe_dim(2, cfg.pe_freq), d)
        self.q_type = nn.Parameter(torch.randn(4, d) * 0.02)                # content, pose, motion, memory
        self.m_idx = nn.Parameter(torch.randn(1, nm, d) * 0.02)
        self.read = CrossAttn(d, cfg.heads)
        self.past_pose = nn.Linear(6 + 4, d)
        self.past_motion = nn.Linear(3, d)
        self.past_memb = nn.Linear(cfg.n_memb, d)
        self.gctx = nn.Linear(dc, d)
        self.f_init = mlp2(4 * d, d, cfg.mlp_hidden)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, 2, True, cfg.mlp_ratio) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(d)
        self.heads = ControlQueries(cfg)
        self.register_buffer("uq", g.cell_centres(cfg.grid), persistent=False)

    def forward(self, bank: tuple, Y: torch.Tensor, gi: torch.Tensor, past: dict) -> tuple:
        """-> (H [B, 274, d], first estimates dict(xi, kz, M, b))."""
        cfg = self.cfg
        B, N, d = Y.shape
        dt = Y.dtype
        base = torch.cat([self.q_pos(g.fourier(self.uq, cfg.pe_freq)).to(dt).unsqueeze(0).expand(B, -1, -1)
                          + self.q_type[0],
                          (self.q_type[1]).view(1, 1, d).expand(B, 1, d),
                          (self.q_type[2] + self.m_idx).expand(B, -1, -1),
                          (self.q_type[3]).view(1, 1, d).expand(B, 1, d)], 1).to(dt)
        rd = self.read(base, bank[0].to(dt), None, bank[1])
        zeros = torch.zeros(B, N, d, device=Y.device, dtype=dt)
        pst = torch.cat([zeros, self.past_pose(torch.cat([past["xi"], past["kz"]], -1)).unsqueeze(1).to(dt),
                         self.past_motion(past["M"]).to(dt), self.past_memb(past["b"]).unsqueeze(1).to(dt)], 1)
        gr = g.read_coarse(gi.float(), self.uq.expand(B, -1, -1), cfg.coarse_grid)
        gm = gi.float().mean(1, keepdim=True).expand(-1, 2 + cfg.n_motion, -1)
        gc = self.gctx(torch.cat([gr, gm], 1).to(dt))
        h = self.f_init(torch.cat([base, rd, pst, gc], -1))
        h = self.norm(run(self.blocks, h, [(bank[0].to(dt), None, bank[1]), (Y, None, None)]))
        est = self.heads(h[:, N], h[:, N + 1:N + 1 + cfg.n_motion], h[:, -1])
        return h, est
