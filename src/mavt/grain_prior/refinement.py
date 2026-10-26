"""Stage 4B: correspondence read with exploration and the two observed refinement passes (sections 8.2, 8.3).

    pi_kp^(l) = (1 - eps) softmax_p(-|16 (v_p - u^_ik)|^2 / (2 s_l^2)) + eps / 256      (uniform if u^ invalid)
    A_kp      = softmax_p(q_k . k_p / sqrt(d_h) + log pi_kp),  e_k = sum_p A_kp v_p,  v-bar_k = sum_p A_kp v_p(pos)
    (dxi, dM, db, deta)^(l) = f_upd^(l)(H_D^(6), {e_k, v-bar_k - u^_ik}, Pool(F_i - F^base,(l)))

The current image is read over all 256 patches (no hard 9-NN); s_0 = 3, s_1 = 1.5 patches, eps = 0.05.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import geometry as g
from .anchor import gaussian_prior, pool_coarse
from .config import GrainPriorConfig
from .layers import Block, run
from .trunk import ControlQueries


class CorrRead(nn.Module):
    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        d = cfg.dim
        self.cfg, self.heads = cfg, cfg.heads
        self.q_tok = nn.Linear(cfg.d_f + g.pe_dim(2, cfg.pe_freq), d)
        self.pass_emb = nn.Parameter(torch.randn(cfg.refine_passes, d) * 0.02)
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.register_buffer("uq", g.cell_centres(cfg.grid), persistent=False)

    def forward(self, f, u_k, u_hat, valid, Y, ell: int) -> tuple:
        """-> (e [B, G, d], v_bar [B, G, 2] expected matched position)."""
        cfg = self.cfg
        B, Gn, _ = f.shape
        H = self.heads
        q = self.q_tok(torch.cat([f.float(), g.fourier(u_k, cfg.pe_freq)], -1)).to(Y.dtype) + self.pass_emb[ell]
        sh = lambda t: t.reshape(B, t.shape[1], H, -1).transpose(1, 2)
        qh, kh, vh = sh(self.wq(self.nq(q))), sh(self.wk(self.nk(Y))), sh(self.wv(self.nk(Y)))
        lp = gaussian_prior(u_hat, self.uq, cfg.corr_s[ell], cfg.corr_eps, cfg.grid, valid)    # [B, G, N]
        with g.fp32(Y.device):
            logits = (qh.float() @ kh.float().transpose(-1, -2)) / math.sqrt(qh.shape[-1]) + lp.unsqueeze(1)
            A = torch.softmax(logits, -1)
            v_bar = A.mean(1) @ self.uq
        e = (A.to(vh.dtype) @ vh).transpose(1, 2).reshape(B, Gn, -1)
        return self.wo(e), v_bar


class UpdateHead(nn.Module):
    """f_upd^(l): the control tokens of H_D read the grain evidence, the pooled feature residual and the
    content tokens; zero-initialised heads give the increments."""

    def __init__(self, cfg: GrainPriorConfig, teacher_dim: int):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.ev = nn.Linear(d + 2 + g.pe_dim(2, cfg.pe_freq) + 1, d)
        self.res = nn.Linear(teacher_dim, d)
        self.typ = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, 1, True, cfg.mlp_ratio) for _ in range(cfg.upd_blocks)])
        self.norm = nn.LayerNorm(d)
        self.heads = ControlQueries(cfg)

    def forward(self, H_D, e, v_bar, u_hat, valid, resid) -> dict:
        cfg = self.cfg
        N = cfg.n_cells
        dt = H_D.dtype
        off = cfg.grid * (v_bar - u_hat.float()) * valid.unsqueeze(-1)
        ev = self.ev(torch.cat([e.float(), off, g.fourier(off / cfg.grid, cfg.pe_freq), valid.float().unsqueeze(-1)],
                               -1)).to(dt) + self.typ[0]
        rs = self.res(pool_coarse(resid.float(), cfg.grid, cfg.coarse_grid)).to(dt) + self.typ[1]
        E = torch.cat([ev, rs, H_D[:, :N] + self.typ[2]], 1)
        h = self.norm(run(self.blocks, H_D[:, N:], [(E, None, None)]))
        return self.heads(h[:, 0], h[:, 1:1 + cfg.n_motion], h[:, -1])
