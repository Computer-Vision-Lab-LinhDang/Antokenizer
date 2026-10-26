"""Stages 1-2 (semantic / detail features, first context), stream C of stage 3 and stage 4A (anchor C).

    F_i = sg DINOv3(X_i) [256, D_t]          P_i = PixelStem(X_i) [256, 512]
    Y_i = LN(F_i W_F + P_i + PE_2D)          g_i = tanh(AvgPool_4x4(Y_i) W_g) [16, 128]
    stream C: A_C,p = MLP_C[PE(u_p), ReadCoarse(g_0, u_p), Mean(g_0)], six blocks reading E = Y_0
    c_0 = tanh(AvgPool_4x4(H_C) W_c);  kappa_0 = f_kappa0(Mean H_C) (copied when the camera is known)
    grains: one per 2 x 2 stratum, soft location u_k inside the stratum, conditional grain query
    q_k = f_g[ReadCoarse(c_0, u_k), Mean(c_0), PE(u_k)], two read-refine blocks over (H_C, F_0, P_0) with a
    Gaussian spatial prior plus exploration -> f_k (60), rho_k, sigma_k = 0.4 + 1.1 sigmoid
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cd32.encoder import MultiLayerDINO
from . import geometry as g
from .config import GrainPriorConfig
from .layers import Block, mlp2, run, zero_last


class PixelStem(nn.Module):
    """Four Conv2d stride 2 (kernel 3, padding 1), GELU between: [B', 3, H, W] -> [B', N, C_last]."""

    def __init__(self, channels: tuple, c_in: int = 3):
        super().__init__()
        ch = [c_in] + list(channels)
        layers = []
        for i in range(4):
            layers.append(nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1))
            if i < 3:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(2).transpose(1, 2)


def pool_coarse(x: torch.Tensor, grid: int, side: int) -> torch.Tensor:
    """[B, G*G, C] -> AvgPool to [B, side^2, C]."""
    B, N, C = x.shape
    img = x.transpose(1, 2).reshape(B, C, grid, grid)
    return F.adaptive_avg_pool2d(img.float(), side).flatten(2).transpose(1, 2).to(x.dtype)


class Stems(nn.Module):
    """Frames [B', 3, H, W] in [-1, 1] -> dict(F [B', N, D_t] float32 teacher, P, Y [B', N, dim], g [B', 16, dc])."""

    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        self.cfg = cfg
        self.dino = MultiLayerDINO(cfg)
        assert cfg.stem_channels[-1] == cfg.dim, "the pixel stem ends at the trunk width"
        self.pixel = PixelStem(cfg.stem_channels)
        self.w_f = nn.Linear(self.dino.out_dim, cfg.dim)
        self.w_pe = nn.Linear(g.pe_dim(2, cfg.pe_freq), cfg.dim)
        self.norm = nn.LayerNorm(cfg.dim)
        self.w_g = nn.Linear(cfg.dim, cfg.d_coarse)
        self.register_buffer("uq", g.cell_centres(cfg.grid), persistent=False)

    @property
    def teacher_dim(self) -> int:
        return self.dino.out_dim

    def pe2d(self) -> torch.Tensor:
        return self.w_pe(g.fourier(self.uq, self.cfg.pe_freq))

    def forward(self, frames: torch.Tensor) -> dict:
        cfg = self.cfg
        Ft, _, _ = self.dino(frames)
        Ft = Ft.float().detach()
        P = self.pixel(frames)
        Y = self.norm(self.w_f(Ft.to(P.dtype)) + P + self.pe2d())
        gi = torch.tanh(self.w_g(pool_coarse(Y, cfg.grid, cfg.coarse_grid)))
        return dict(F=Ft, P=P, Y=Y, g=gi)


class StreamC(nn.Module):
    """Stream C: reads the anchor observation only."""

    def __init__(self, cfg: GrainPriorConfig):
        super().__init__()
        self.cfg = cfg
        dc = cfg.d_coarse
        self.init = mlp2(g.pe_dim(2, cfg.pe_freq) + 2 * dc, cfg.dim, cfg.mlp_hidden)
        self.blocks = nn.ModuleList([Block(cfg.dim, cfg.heads, 1, True, cfg.mlp_ratio) for _ in range(cfg.layers)])
        self.register_buffer("uq", g.cell_centres(cfg.grid), persistent=False)

    def forward(self, Y0: torch.Tensor, g0: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = Y0.shape[0]
        u = self.uq.expand(B, -1, -1)
        x = torch.cat([g.fourier(u, cfg.pe_freq), g.read_coarse(g0.float(), u, cfg.coarse_grid),
                       g0.float().mean(1, keepdim=True).expand(-1, cfg.n_cells, -1)], -1)
        h = self.init(x.to(Y0.dtype))
        return run(self.blocks, h, [(Y0, None, None)])


def gaussian_prior(u_q: torch.Tensor, v: torch.Tensor, s: float, eps: float, grid: int,
                   valid: torch.Tensor = None) -> torch.Tensor:
    """log[(1 - eps) softmax_p(-|G (v_p - u_k)|^2 / (2 s^2)) + eps / P]: u_q [B, M, 2], v [P, 2] -> [B, M, P].
    Rows with valid = False get the uniform prior (exploration only)."""
    with g.fp32(u_q.device):
        d2 = ((grid * (u_q.float().unsqueeze(2) - v.float().view(1, 1, -1, 2))) ** 2).sum(-1)
        p = torch.softmax(-d2 / (2 * s * s), -1)
        P = v.shape[0]
        if valid is not None:
            p = torch.where(valid.unsqueeze(-1), p, torch.full_like(p, 1.0 / P))
        return torch.log((1 - eps) * p + eps / P)


class AnchorHeads(nn.Module):
    """H_C^(6) (+ F_0, P_0) -> c_0, kappa_0 logits, grains (u, rho~, sigma, f)."""

    def __init__(self, cfg: GrainPriorConfig, teacher_dim: int):
        super().__init__()
        self.cfg = cfg
        d, dc, pe2 = cfg.dim, cfg.d_coarse, g.pe_dim(2, cfg.pe_freq)
        self.w_c = nn.Linear(d, dc)
        self.kappa0 = zero_last(mlp2(d, 4, cfg.mlp_hidden))
        self.score = mlp2(d + dc, 1, cfg.mlp_hidden)
        self.offset = zero_last(mlp2(d, 2, cfg.mlp_hidden))
        self.q_init = mlp2(2 * dc + pe2, d, cfg.mlp_hidden)
        self.src = nn.ModuleList([nn.Linear(d, d), nn.Linear(teacher_dim, d), nn.Linear(d, d)])
        self.src_type = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, 1, True, cfg.mlp_ratio) for _ in range(cfg.grain_blocks)])
        self.norm = nn.LayerNorm(d)
        self.w_f = nn.Linear(d, cfg.d_f)
        self.w_rho = mlp2(d, 1, cfg.mlp_hidden)
        self.w_sigma = mlp2(d, 1, cfg.mlp_hidden)
        nn.init.zeros_(self.w_rho[-1].weight)
        nn.init.constant_(self.w_rho[-1].bias, math.log(-cfg.rho_lo / cfg.rho_hi))     # rho~ = 0 at init: d = 1
        s, G = cfg.stratum, cfg.grid
        uq = g.cell_centres(G)
        rows, cols = torch.arange(G).repeat_interleave(G), torch.arange(G).repeat(G)
        stratum = (rows // s) * cfg.n_strata + cols // s                                 # [N] stratum of each cell
        order = torch.argsort(stratum * G * G + torch.arange(G * G))                    # cells grouped by stratum
        self.register_buffer("uq", uq, persistent=False)
        self.register_buffer("members", order.view(cfg.n_grains, s * s), persistent=False)
        lo = torch.stack([(torch.arange(cfg.n_grains) % cfg.n_strata) * s,
                          (torch.arange(cfg.n_grains) // cfg.n_strata) * s], -1).float() / G
        self.register_buffer("box_lo", lo, persistent=False)
        self.register_buffer("box_hi", lo + s / G, persistent=False)

    def forward(self, H: torch.Tensor, F0: torch.Tensor, P0: torch.Tensor) -> dict:
        cfg = self.cfg
        B, N, d = H.shape
        c0 = torch.tanh(self.w_c(pool_coarse(H, cfg.grid, cfg.coarse_grid))).float()
        kz = self.kappa0(H.mean(1)).float()
        kz = kz + torch.tensor([g.kappa_logit_init(cfg)] * 2 + [0.0, 0.0], device=H.device)
        # ---- soft location inside each stratum
        u_all = self.uq.expand(B, -1, -1)
        s_p = self.score(torch.cat([H, g.read_coarse(c0, u_all, cfg.coarse_grid).to(H.dtype)], -1)).squeeze(-1)
        idx = self.members                                                               # [G, s^2]
        alpha = torch.softmax(s_p.float()[:, idx] / cfg.temp_s, -1)                    # [B, G, s^2]
        u_bar = (alpha.unsqueeze(-1) * self.uq[idx].unsqueeze(0)).sum(2)                # [B, G, 2]
        o = self.offset(H[:, idx].mean(2)).float()
        u = torch.maximum(torch.minimum(u_bar + 0.5 * torch.tanh(o) / cfg.grid, self.box_hi), self.box_lo)
        # ---- conditional grain query, read-refine over (H_C, F_0, P_0)
        q = self.q_init(torch.cat([g.read_coarse(c0, u, cfg.coarse_grid), c0.mean(1, keepdim=True).expand(-1, cfg.n_grains, -1),
                                   g.fourier(u, cfg.pe_freq)], -1).to(H.dtype))
        E = torch.cat([self.src[0](H) + self.src_type[0], self.src[1](F0.to(H.dtype)) + self.src_type[1],
                       self.src[2](P0) + self.src_type[2]], 1)
        lp = gaussian_prior(u, self.uq, cfg.grain_read_s, cfg.corr_eps, cfg.grid)       # [B, G, N]
        bias = lp.repeat(1, 1, 3).unsqueeze(1)
        h = run(self.blocks, q, [(E, bias, None)])
        h = self.norm(h)
        rho_t = cfg.rho_lo + (cfg.rho_hi - cfg.rho_lo) * torch.sigmoid(self.w_rho(h).squeeze(-1).float())
        sigma = cfg.sigma_min + cfg.sigma_span * torch.sigmoid(self.w_sigma(h).squeeze(-1).float())
        return dict(c0=c0, kappa_logits=kz, u=u, rho_raw=rho_t, sigma=sigma, f=self.w_f(h).float(), h=h)
