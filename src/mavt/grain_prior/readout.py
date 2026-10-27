"""Public decoder (section 12): coarse estimate, then grain evidence, then innovation; one operator for every
query type, with the same weights in encoder synthesis and in public decoding.

    a^(0)   = MLP_a[PE_type(q), CA(PE_type(q), [c_0; z_i]), PoolGeom(C, D_i)]            (query_init cond)
    k_ik    = W_K[f~_ik; PE(P^can_ik); PE(u_k)],  v_ik = W_V f~_ik
    Q^(l)   = W_Q LN(a^(l)) + W_PE PE_type(q),    e^(l) = Attn(Q^(l), K_i, V_i; B^geom)
    a^(l+1) = a^(l) + MLP_ref[a^(l), e^(l), PE_type(q)],     F^base = W_out a^(2),  F^init = W_out a^(0)
    B^geom  = log nu,  nu_qk ∝ exp(-|16 (u_q - u^_k)|^2 / (2 sigma_k^2)) exp(-d_k / tau_d) over grains in front
    dF      = 1[L > 0] f_out(CA(f_q[a^(2), PE(q)], Embed(r_i); B^r)),  B^r = -|16 (u_q - u_ij)|^2 / (2 sigma_ij^2)
    RGB     = sigmoid(conv(up x2 conv ... (W F)))  on the 16 x 16 x D_t feature grid
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as g
from .config import GrainPriorConfig
from .layers import CrossAttn, attend, mlp2

N_POOLGEOM = 18


def local_support(uq, uhat, sigma, valid, grid):
    """Diagnostic v2: on-screen, finite front-facing grain within 3 kernel radii.

    Evidence locality is not opacity or calibrated confidence, and never an eval mask.
    """
    on_screen = valid & torch.isfinite(uhat).all(-1) & ((uhat >= 0) & (uhat <= 1)).all(-1)
    d2 = ((grid * (uq.float().unsqueeze(2) - uhat.float().unsqueeze(1))) ** 2).sum(-1)
    return (on_screen.unsqueeze(1) & (d2 <= 9 * sigma.float().unsqueeze(1).square())).any(-1)


def pool_geom(xi, kappa, kappa0, rho, M, valid) -> torch.Tensor:
    """PoolGeom(C, D_i): pooled geometric metadata of the state at observation i [B, 18] (no target data)."""
    return torch.cat([xi, kappa, kappa0, rho.mean(-1, keepdim=True), rho.std(-1, keepdim=True),
                      M.norm(dim=-1).mean(-1, keepdim=True), valid.float().mean(-1, keepdim=True)], -1).float()


def geom_bias(u_q: torch.Tensor, u_hat: torch.Tensor, sigma: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor,
              grid: int, tau: float) -> torch.Tensor:
    """log nu [B, Q, G]: normalised over the valid grains of each query (rows without one: 0)."""
    with g.fp32(u_q.device):
        d2 = ((grid * (u_q.float().unsqueeze(2) - u_hat.float().unsqueeze(1))) ** 2).sum(-1)
        ln = -d2 / (2 * sigma.float().unsqueeze(1) ** 2) - depth.float().unsqueeze(1) / tau
        ln = ln.masked_fill(~valid.unsqueeze(1), float("-inf"))
        lse = torch.logsumexp(ln, -1, keepdim=True)
        ok = torch.isfinite(lse)
        return torch.where(ok & valid.unsqueeze(1), ln - torch.where(ok, lse, torch.zeros_like(lse)),
                           torch.zeros_like(ln))


class ReadLayer(nn.Module):
    def __init__(self, cfg: GrainPriorConfig, d_pe: int):
        super().__init__()
        d = cfg.dim
        self.heads = cfg.heads
        self.norm = nn.LayerNorm(d)
        self.wq, self.wpe = nn.Linear(d, d), nn.Linear(d_pe, d, bias=False)
        self.nk, self.nv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wk, self.wv, self.wo = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ref = mlp2(3 * d, d, cfg.mlp_hidden)
        self.strict = cfg.readout_version != "legacy"
        if self.strict:
            self.temperature = nn.Parameter(torch.zeros(()))
        elif cfg.read_qk_norm:
            self.qn, self.kn = nn.LayerNorm(d // cfg.heads), nn.LayerNorm(d // cfg.heads)

    def _head_norm(self, x, ln):
        return ln(x.unflatten(-1, (self.heads, -1))).flatten(-2) if hasattr(self, "qn") else x

    def forward(self, a, pe, ktok, vtok, bias, valid):
        Q = self._head_norm(self.wq(self.norm(a)) + self.wpe(pe), getattr(self, "qn", None))
        K = self._head_norm(self.wk(self.nk(ktok)), getattr(self, "kn", None))
        if self.strict:
            norm = lambda x: F.normalize(x.unflatten(-1, (self.heads, -1)), dim=-1).flatten(-2)
            Q = norm(Q) * (8 * self.temperature.sigmoid()) * (Q.shape[-1] // self.heads) ** .5
            K = norm(K)
        e = self.wo(attend(Q, K, self.wv(self.nv(vtok)), self.heads, bias, valid, strict=self.strict))
        if self.strict:
            e = e * valid.any(-1)[:, None, None].to(e.dtype)
        return a + self.ref(torch.cat([a, e, pe], -1))


class RGBHead(nn.Module):
    """Feature grid [B, G*G, D_t] -> image [B, 3, 16 G, 16 G] in [0, 1]."""

    def __init__(self, cfg: GrainPriorConfig, teacher_dim: int):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(teacher_dim, cfg.dim)
        ch = [cfg.dim] + list(cfg.rgb_channels)
        self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, padding=1) for i in range(4)])
        self.final = nn.Conv2d(ch[-1], 3, 3, padding=1)

    def forward(self, Fh: torch.Tensor) -> torch.Tensor:
        B, N, _ = Fh.shape
        G = self.cfg.grid
        x = self.proj(Fh).transpose(1, 2).reshape(B, -1, G, G)
        for conv in self.convs:
            x = F.gelu(conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)))
        return torch.sigmoid(self.final(x).float())


class Readout(nn.Module):
    def __init__(self, cfg: GrainPriorConfig, teacher_dim: int):
        super().__init__()
        self.cfg = cfg
        d, pf, dc = cfg.dim, cfg.pe_freq, cfg.d_coarse
        pe2, pe3 = g.pe_dim(2, pf), g.pe_dim(3, pf)
        self.q_pe = nn.Linear(pe2, d)
        self.q_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)                        # pixel query type
        # ---- a^(0)
        self.mem_tok = nn.Linear(dc, d)
        self.mem_type = nn.Parameter(torch.randn(2, d) * 0.02)                         # c_0, z_i
        self.mem_slot = nn.Parameter(torch.randn(1, cfg.n_coarse, d) * 0.02)
        self.mem_grid = nn.Linear(pe2, d)
        self.init_ca = CrossAttn(d, cfg.heads)
        self.poolgeom = mlp2(N_POOLGEOM, d, cfg.mlp_hidden)
        self.mlp_a = mlp2(3 * d, d, cfg.mlp_hidden)
        self.uncond = nn.Parameter(torch.randn(1, cfg.n_cells, d) * 0.02)
        # ---- grain reads
        self.ktok = nn.Linear(cfg.d_f + pe3 + pe2, d)
        self.vtok = nn.Linear(cfg.d_f, d)
        self.layers = nn.ModuleList([ReadLayer(cfg, d) for _ in range(cfg.read_layers)])
        self.w_out = nn.Linear(d, teacher_dim)
        self.depth = mlp2(d, 1, cfg.mlp_hidden)
        # ---- innovation read
        self.f_q = mlp2(2 * d, d, cfg.mlp_hidden)
        self.r_tok = nn.Linear(cfg.d_e + pe2 + 2, d)
        self.r_ca = CrossAttn(d, cfg.heads)
        self.f_out = nn.Linear(d, teacher_dim, bias=False)
        self.register_buffer("uq", g.cell_centres(cfg.grid), persistent=False)
        self.register_buffer("ucoarse", g.cell_centres(cfg.coarse_grid), persistent=False)

    def query_pe(self, B: int, uq=None) -> torch.Tensor:
        uq = self.uq.expand(B,-1,-1) if uq is None else uq
        return self.q_pe(g.fourier(uq, self.cfg.pe_freq)) + self.q_type

    def init(self, pe, c0, z, pg, uq=None) -> torch.Tensor:
        cfg = self.cfg
        B = pe.shape[0]
        if cfg.query_init == "uncond":
            uncond = self.uncond.expand(B, -1, -1)
            return (uncond if uq is None else g.read_coarse(uncond, uq, cfg.grid)) + pe
        grid = self.mem_grid(g.fourier(self.ucoarse, cfg.pe_freq)).unsqueeze(0)
        tok = torch.cat([self.mem_tok(c0.float()) + self.mem_type[0], self.mem_tok(z.float()) + self.mem_type[1]], 1)
        tok = tok + (self.mem_slot + grid).repeat(1, 2, 1)
        ca = self.init_ca(pe, tok.to(pe.dtype))
        return self.mlp_a(torch.cat([pe, ca, self.poolgeom(pg).to(pe.dtype).unsqueeze(1).expand(-1, pe.shape[1], -1)], -1))

    def base(self, f_t, P_can, u_k, u_hat, z_cam, sigma, valid, scale, c0, z, pg, uq=None) -> dict:
        """Grains at observation i (f~ [B, G, d_f], P^can, anchor u_k, projection u^ / z / valid, sigma), depth
        scale of the gauge [B], c_0, z_i, PoolGeom -> dict(a0, a, F_base, F_init, depth_grain, depth_dense,
        support [B, Q])."""
        cfg = self.cfg
        B = f_t.shape[0]
        pe = self.query_pe(B, uq)
        a0 = self.init(pe, c0, z, pg, uq)
        sc = scale.float().view(B, 1, 1)
        kt = self.ktok(torch.cat([f_t.float(), g.fourier(P_can.float() / sc, cfg.pe_freq),
                                  g.fourier(u_k.float(), cfg.pe_freq)], -1)).to(a0.dtype)
        vt = self.vtok(f_t.float()).to(a0.dtype)
        u_q = self.uq.expand(B, -1, -1) if uq is None else uq
        dd = z_cam.float() / sc.view(B, 1) if cfg.tau_d_mode == "median" else z_cam.float()
        lnu = geom_bias(u_q, u_hat, sigma, dd, valid, cfg.grid, cfg.tau_d)              # [B, Q, G]
        bias = lnu.unsqueeze(1) if cfg.geom_bias else None
        a = a0
        for layer in self.layers:
            a = layer(a, pe, kt, vt, bias, valid)
        with g.fp32(a.device):
            nu = lnu.exp() * valid.unsqueeze(1)
            support = valid.any(-1, keepdim=True).expand(-1, lnu.shape[1])
            far = z_cam.float().masked_fill(~valid, 0).amax(-1, keepdim=True).clamp_min(1.0)
            d_grain = torch.where(support, (nu * z_cam.float().unsqueeze(1)).sum(-1), far.expand_as(support).float())
            d_dense = torch.exp(self.depth(a.float()).squeeze(-1).float().clamp(-20, 20)) * scale.float().view(B, 1)
        support_v2 = local_support(u_q, u_hat, sigma, valid, cfg.grid)
        return dict(a0=a0, a=a, pe=pe, F_base=self.w_out(a).float(), F_init=self.w_out(a0).float(),
                    depth_grain=d_grain, depth_dense=d_dense,
                    support=support if cfg.measurement_version == 1 else support_v2,
                    support_v1=support, support_v2=support_v2)

    def innovation(self, a: torch.Tensor, pe: torch.Tensor, r: torch.Tensor, L: torch.Tensor,
                   return_hidden: bool = False, uq=None):
        """dF [B, Q, D_t] from the own-view innovation slots r [B, n_slots, d_r] (prefix L [B]); exactly 0 at L = 0.
        return_hidden: also the state increment dH [B, Q, dim] (the input of f_out), 0 at L = 0."""
        cfg = self.cfg
        B = a.shape[0]
        rf = r.float()
        tok = self.r_tok(torch.cat([rf[..., 4:], g.fourier(rf[..., :2], cfg.pe_freq), rf[..., 2:4]], -1)).to(a.dtype)
        ok = torch.arange(cfg.n_slots, device=r.device).view(1, -1) < L.view(-1, 1)
        sig = torch.where(ok, rf[..., 3], torch.ones_like(rf[..., 3])).clamp_min(cfg.sigma_min)   # slots past L: 0
        with g.fp32(a.device):
            u_q = self.uq.expand(B,-1,-1) if uq is None else uq
            d2 = ((cfg.grid * (u_q.unsqueeze(2) - rf[..., :2].unsqueeze(1))) ** 2).sum(-1)
            br = (-d2 / (2 * sig.unsqueeze(1) ** 2)).unsqueeze(1)
        q = self.f_q(torch.cat([a, pe], -1))
        on = (L > 0).float().view(B, 1, 1)
        dH = self.r_ca(q, tok, br, ok, strict=cfg.readout_version != "legacy")
        dF = self.f_out(dH).float() * on
        return (dF, dH * on.to(dH.dtype)) if return_hidden else dF
