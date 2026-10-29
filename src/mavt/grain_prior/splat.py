"""rgb_decoder = splat: the RGB-grid decoder state from the grains by context, frequency atoms and a geometric splat.

1. Context: self-attention over the grains (tokens of f, u^, log depth, log sigma) and the coarse slots [c_0; z].
   Grain-to-grain attention is biased by the read's geometry term, log nu of grain i's position against the
   grains j (geom_bias: Gaussian of radius sigma_j, depth d_j / tau_d), scaled per head (init 1); slots: no bias.
2. Per grain: a decoder feature e (splat_feat) and K frequency atoms (LTE-style): omega_k = wmax tanh(.) cycles
   per cell (2D), amplitudes (a_cos, a_sin) in R^splat_feat (zero-init: e alone at step 0).
3. Splat onto the RGB query grid: query q reads the grains of the (2 w + 1)^2 strata around its own with
   w_qj = softmax_j(-|grid (u_q - u^_j)|^2 / (2 sigma_j^2) - d_j / tau_d) (the read's log nu, restricted to the
   window), X_q = sum_j w_qj (e_j + sum_k a_cos_jk cos 2 pi <omega_jk, delta_qj> + a_sin_jk sin(.)),
   delta_qj = grid (u_q - u^_j). Queries are grouped by stratum so each group shares its window (two matmuls).
The output W_out X_q joins the read's initial state a0; the existing RGB head is the local refiner. The window is
indexed by the anchor stratum of each grain: exact on images and static anchors; transported grains (video) that
leave their stratum's neighbourhood are read only inside it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import geometry as g
from .layers import attend
from .readout import geom_bias


def window_index(n_side: int, w: int) -> tuple:
    """Grain ids [n_side^2, (2 w + 1)^2] of the strata around each stratum (row-major, x fastest; grain id =
    stratum id) and their validity (inside the image)."""
    ys, xs = torch.meshgrid(torch.arange(n_side), torch.arange(n_side), indexing="ij")
    d = torch.arange(-w, w + 1)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    ny, nx = ys.reshape(-1, 1) + dy.reshape(1, -1), xs.reshape(-1, 1) + dx.reshape(1, -1)
    ok = (ny >= 0) & (ny < n_side) & (nx >= 0) & (nx < n_side)
    return ny.clamp(0, n_side - 1) * n_side + nx.clamp(0, n_side - 1), ok


class _CtxBlock(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        self.n1, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Linear(int(d * mlp_ratio), d))

    def forward(self, h, bias, valid):
        q, k, v = self.qkv(self.n1(h)).chunk(3, -1)
        h = h + self.o(attend(q, k, v, self.heads, bias, valid))
        return h + self.mlp(self.n2(h))


class GrainSplat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, C, K = cfg.dim, cfg.splat_feat, cfg.splat_atoms
        self.tok = nn.Linear(cfg.d_f + g.pe_dim(2, cfg.pe_freq) + 2, d)
        self.ctok = nn.Linear(cfg.d_coarse, d)
        self.slot_pe = nn.Parameter(torch.randn(2 * cfg.n_coarse, d) * 0.02)
        self.bias_scale = nn.Parameter(torch.ones(cfg.heads))
        self.blocks = nn.ModuleList([_CtxBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.splat_layers)])
        self.norm = nn.LayerNorm(d)
        self.feat = nn.Linear(d, C)
        self.amp = nn.Linear(d, K * 2 * C)
        nn.init.zeros_(self.amp.weight)
        nn.init.zeros_(self.amp.bias)
        self.freq = nn.Linear(d, K * 2)
        self.out = nn.Linear(C, d)
        n = cfg.n_strata
        widx, wok = window_index(n, cfg.splat_window)
        self.register_buffer("widx", widx, persistent=False)
        self.register_buffer("wok", wok, persistent=False)
        qg = cfg.query_grid
        m = qg // n
        qy, qx = torch.arange(qg).repeat_interleave(qg), torch.arange(qg).repeat(qg)
        stratum = (qy // m) * n + qx // m
        perm = torch.argsort(stratum * qg * qg + torch.arange(qg * qg))
        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("inv", torch.argsort(perm), persistent=False)

    def _depth(self, z_cam, scale):
        return z_cam.float() / scale.float().view(-1, 1) if self.cfg.tau_d_mode == "median" else z_cam.float()

    def _logits(self, uq, u_w, s_w, d_w, ok_w):
        """uq [..., m, 2], window grains u_w [..., W, 2], sigma / depth / ok [..., W] -> (weights, delta)."""
        cfg = self.cfg
        delta = cfg.grid * (uq.float().unsqueeze(-2) - u_w.float().unsqueeze(-3))           # [..., m, W, 2]
        ln = -(delta ** 2).sum(-1) / (2 * s_w.float().unsqueeze(-2) ** 2) - d_w.unsqueeze(-2) / cfg.tau_d
        ln = ln.masked_fill(~ok_w.unsqueeze(-2), float("-inf"))
        has = ok_w.any(-1, keepdim=True).unsqueeze(-2)
        w = torch.where(has, ln, torch.zeros_like(ln)).softmax(-1) * has
        return w.masked_fill(~ok_w.unsqueeze(-2), 0.0), delta

    def weights(self, uq, u_hat, sigma, depth, valid):
        """Per-query window weights for arbitrary queries uq [B, Q, 2] (tests / diagnostics) -> (w, idx, ok)."""
        n = self.cfg.n_strata
        s = (uq.float() * n).floor().clamp(0, n - 1).long()
        s = s[..., 1] * n + s[..., 0]                                                    # [B, Q]
        idx, wok = self.widx[s], self.wok[s]                                             # [B, Q, W]
        bi = torch.arange(uq.shape[0], device=uq.device).view(-1, 1, 1)
        take = lambda t: t[bi, idx]
        ok = wok & take(valid)
        w, _ = self._logits(uq.unsqueeze(-2), take(u_hat), take(sigma), take(depth.float()), ok)
        return w.squeeze(-2), idx, ok

    def forward(self, f_t, u_hat, z_cam, sigma, valid, scale, c0, z, uq) -> torch.Tensor:
        """Grains at the observation (f~ [B, G, d_f], u^ [B, G, 2], z_cam, sigma, valid [B, G]), depth scale [B],
        c_0 / z_i [B, n_c, d_c], the RGB query grid uq [B, Q, 2] (cell centres of query_grid) -> [B, Q, dim]."""
        cfg = self.cfg
        B, G = f_t.shape[:2]
        dd = self._depth(z_cam, scale)
        x = self.tok(torch.cat([f_t.float(), g.fourier(u_hat.float(), cfg.pe_freq),
                                torch.log(dd.clamp_min(1e-4)).unsqueeze(-1),
                                torch.log(sigma.float().clamp_min(1e-4)).unsqueeze(-1)], -1))
        slots = self.ctok(torch.cat([c0.float(), z.float()], 1)) + self.slot_pe
        h = torch.cat([x.to(slots.dtype), slots], 1)
        T = h.shape[1]
        gb = geom_bias(u_hat, u_hat, sigma, dd, valid, cfg.grid, cfg.tau_d)                # [B, G, G]
        bias = h.new_zeros(B, cfg.heads, T, T, dtype=torch.float32)
        bias[:, :, :G, :G] = gb.unsqueeze(1) * self.bias_scale.view(1, -1, 1, 1)
        kv = torch.cat([valid, torch.ones(B, T - G, dtype=torch.bool, device=valid.device)], 1)
        for blk in self.blocks:
            h = blk(h, bias, kv)
        hg = self.norm(h[:, :G])
        K, C = cfg.splat_atoms, cfg.splat_feat
        e = self.feat(hg)                                                                # [B, G, C]
        A = self.amp(hg).view(B, G, K, 2, C)
        om = cfg.splat_wmax * torch.tanh(self.freq(hg).float()).view(B, G, K, 2)          # cycles per cell
        n2, W = self.widx.shape
        uq_g = uq[:, self.perm].view(B, n2, -1, 2)                                       # [B, S, m, 2]
        wi = self.widx.view(-1)
        pick = lambda t: t[:, wi].view(B, n2, W, *t.shape[2:])
        ok = self.wok.unsqueeze(0) & pick(valid)
        w, delta = self._logits(uq_g, pick(u_hat), pick(sigma), pick(dd), ok)              # [B, S, m, W], [.., 2]
        X = w.to(e.dtype) @ pick(e)                                                      # [B, S, m, C]
        if K:
            ph = 2 * math.pi * torch.einsum("bsmwx,bswkx->bsmwk", delta, pick(om))
            Aw = pick(A)                                                                 # [B, S, W, K, 2, C]
            cw = (w.unsqueeze(-1) * torch.cos(ph)).flatten(-2).to(e.dtype)               # [B, S, m, W K]
            sw = (w.unsqueeze(-1) * torch.sin(ph)).flatten(-2).to(e.dtype)
            X = X + cw @ Aw[..., 0, :].flatten(2, 3) + sw @ Aw[..., 1, :].flatten(2, 3)
        X = X.flatten(1, 2)[:, self.inv]                                                 # [B, Q, C] row-major
        return self.out(X)
