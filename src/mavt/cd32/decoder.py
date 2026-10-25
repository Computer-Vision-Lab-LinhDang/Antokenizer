"""Decoder of bản 3.2: forward projection, local k-NN read, prediction, previous-frame read, innovation.

For each query cell u' of the target frame:

  1. neighbours   N(u') = the k projected keyframe cells nearest to u'; R = distance to the (k+1)-th
  2. geometric read
        h_geom(u') = Attn( q = gamma(u') + ray(u'),
                           kv = MLP([c_k(u_j) ; gamma_pos(u' - p_j) ; gamma(t - t_k)]),
                           logit_bias = -lam_d d_j^2 - lam_z Δlog z_j + w_cut(d_j) + cull_j )
        Δlog z enters the logit bias only, never the value path (the 3.2 leak fix: depth would otherwise
        carry content); then self-attention blocks over the 256 queries.
  3. previous frame   h_prev = WindowRead(h_geom, sg[h_{t-1}]) (3 x 3 window, zero-init): the reference
        of the innovation is the previous decoded frame (DCVC-style conditional coding)
  4. innovation   h_innov = CrossAttn(q = MLP(gamma(u'), sg[h_pred], d_min), kv = [null ; D_innov[:L]]),
        0 when L = 0 (the null token has value 0 and the branch output is masked)
  5. h = h_pred (L = 0, no stop-gradient) or sg[h_pred] + h_innov (L > 0); DINO feature head and conv rgb.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dino_v0.camera import ZeroLinear
from ..dino_v0.decoder import ConvUpsampler
from . import geometry as geo
from .backbone import cell_coords
from .config import CD32Config
from .layers import MLP, Rope3, StreamBlock, WindowRead, attention, merge_heads, split_heads, window_mask


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class GeomRead(nn.Module):
    def __init__(self, cfg: CD32Config):
        super().__init__()
        D, H, nf = cfg.dec_dim, cfg.dec_heads, cfg.n_freq
        self.cfg, self.heads = cfg, H
        self.c_in = nn.ModuleDict(dict(image=nn.Linear(cfg.d_c_image, D), video=nn.Linear(cfg.d_c_video, D)))
        self.e_mlp = nn.Sequential(nn.Linear(D + geo.fourier_dim(2, nf) + geo.fourier_dim(1, nf), D), nn.GELU(),
                                   nn.Linear(D, 2 * D))
        self.q_in = nn.Linear(geo.fourier_dim(2, nf), D)
        self.ray_q = ZeroLinear(6, D)
        self.q_norm = nn.LayerNorm(D)
        self.q_proj = nn.Linear(D, D)
        self.out = nn.Linear(D, D)
        self.lam_d = nn.Parameter(torch.full((H,), _inv_softplus(1.0)))
        self.lam_z = nn.Parameter(torch.full((H,), _inv_softplus(1.0)))
        self.register_buffer("uq", geo.cell_centres(cfg.grid), persistent=False)

    def query(self, B: int, rays: Optional[torch.Tensor]) -> torch.Tensor:
        q = self.q_in(geo.fourier(self.uq / self.cfg.grid, self.cfg.n_freq)).unsqueeze(0).expand(B, -1, -1)
        return q + self.ray_q(rays) if rays is not None else q

    def forward(self, c: torch.Tensor, kind: str, p, z, valid, cull, dt: float,
                rays: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """c [B, N, d_c]; p [B, N, 2], z, valid [B, N] projected keyframe cells; cull [B, N] or None;
        dt = (t - t_k) / K. -> (h [B, N, D], d_min [B, N])."""
        cfg = self.cfg
        B, N = p.shape[:2]
        idx, d2, R2, pj = geo.knn_read(self.uq, p, valid, cfg.knn)                 # [B, N, k]
        w_cut = geo.cut_window(d2, R2, cfg.cut_eps)
        logz = torch.log(z.clamp_min(cfg.z_eps))
        logz_j = torch.gather(logz.unsqueeze(1).expand(-1, N, -1), 2, idx)
        dlz = geo.soft_zbuffer(logz_j, d2, w_cut, cfg.softmin_beta)
        cv = self.c_in["image" if kind == "image" else "video"](c)                  # [B, N, D]
        cj = torch.gather(cv.unsqueeze(1).expand(-1, N, -1, -1), 2,
                          idx.unsqueeze(-1).expand(-1, -1, -1, cv.shape[-1]))      # [B, N, k, D]
        off = self.uq.view(1, N, 1, 2) - pj
        tt = torch.full_like(off[..., :1], float(dt))
        e = torch.cat([cj, geo.fourier(off, cfg.n_freq).to(cj.dtype), geo.fourier(tt, cfg.n_freq).to(cj.dtype)], -1)
        k, v = self.e_mlp(e).chunk(2, -1)
        H = self.heads
        k, v = k.unflatten(-1, (H, -1)), v.unflatten(-1, (H, -1))                  # [B, N, k, H, dh]
        q0 = self.query(B, rays)
        q = self.q_proj(self.q_norm(q0)).unflatten(-1, (H, -1))                     # [B, N, H, dh]
        with geo.fp32(q.device):                                                    # distances: float32
            logits = torch.einsum("bnhd,bnkhd->bnhk", q.float(), k.float()) / math.sqrt(q.shape[-1])
            bias = -F.softplus(self.lam_d.float()).view(1, 1, H, 1) * d2.unsqueeze(2) \
                - F.softplus(self.lam_z.float()).view(1, 1, H, 1) * dlz.unsqueeze(2) + w_cut.unsqueeze(2)
            if cull is not None:
                bias = bias + torch.gather(cull.unsqueeze(1).expand(-1, N, -1), 2, idx).unsqueeze(2)
            a = torch.softmax(logits + bias, -1).to(v.dtype)
        out = torch.einsum("bnhk,bnkhd->bnhd", a, v).flatten(-2)
        return q0 + self.out(out), d2[..., 0].clamp_min(0).sqrt()


class InnovBlock(nn.Module):
    """Cross-attention of the queries to [null ; innovation tokens] (null value fixed at 0) + MLP."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim)
        self.null_k = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.out = nn.Linear(dim, dim)
        self.norm2, self.mlp = nn.LayerNorm(dim), MLP(dim, 4.0)

    def forward(self, x: torch.Tensor, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        q = split_heads(self.q(self.norm_q(x)), self.heads)
        k, v = self.kv(self.norm_kv(tok)).chunk(2, -1)
        k = torch.cat([self.null_k.expand(B, -1, -1).to(k.dtype), k], 1)
        v = torch.cat([torch.zeros_like(v[:, :1]), v], 1)
        k, v = split_heads(k, self.heads), split_heads(v, self.heads)
        mask = torch.cat([torch.ones_like(valid[:, :1]), valid], 1).view(B, 1, 1, -1)
        x = x + self.out(merge_heads(attention(q, k, v, mask)))
        return x + self.mlp(self.norm2(x))


class InnovBranch(nn.Module):
    def __init__(self, cfg: CD32Config):
        super().__init__()
        D, nf = cfg.dec_dim, cfg.n_freq
        self.cfg = cfg
        self.tok_in = nn.Linear(cfg.d_r, D)
        self.slot_emb = nn.Parameter(torch.randn(1, cfg.L_max, D) * 0.02)
        self.norm_h = nn.LayerNorm(D)
        self.q_in = nn.Sequential(nn.Linear(geo.fourier_dim(2, nf) + D + geo.fourier_dim(1, nf), D), nn.GELU(),
                                  nn.Linear(D, D))
        self.blocks = nn.ModuleList([InnovBlock(D, cfg.dec_heads) for _ in range(cfg.dec_innov_blocks)])
        self.norm = nn.LayerNorm(D)
        self.out = ZeroLinear(D, D)
        self.register_buffer("uq", geo.cell_centres(cfg.grid), persistent=False)

    def forward(self, h_pred_sg: torch.Tensor, d_min: torch.Tensor, tokens: torch.Tensor,
                L: torch.Tensor) -> torch.Tensor:
        """h_pred_sg [B, N, D] (already detached), d_min [B, N], tokens [B, L_max, d_r], L [B] long."""
        cfg = self.cfg
        B, N, _ = h_pred_sg.shape
        g_u = geo.fourier(self.uq / cfg.grid, cfg.n_freq).unsqueeze(0).expand(B, -1, -1)
        g_d = geo.fourier(d_min.unsqueeze(-1) / 2, cfg.n_freq)
        x = self.q_in(torch.cat([g_u.to(h_pred_sg.dtype), self.norm_h(h_pred_sg), g_d.to(h_pred_sg.dtype)], -1))
        tok = self.tok_in(tokens.to(x.dtype)) + self.slot_emb
        valid = torch.arange(cfg.L_max, device=L.device).view(1, -1) < L.view(-1, 1)
        for blk in self.blocks:
            x = blk(x, tok, valid)
        return self.out(self.norm(x)) * (L > 0).view(B, 1, 1).to(x.dtype)


class Decoder(nn.Module):
    def __init__(self, cfg: CD32Config, feat_dim: int):
        super().__init__()
        D = cfg.dec_dim
        self.cfg = cfg
        self.read = GeomRead(cfg)
        self.rope = Rope3(cfg.rope_dims, cfg.rope_base)
        self.pred_blocks = nn.ModuleList([StreamBlock(D, cfg.dec_heads, 4.0, ctx=False)
                                          for _ in range(cfg.dec_pred_blocks)])
        self.prev = WindowRead(D, cfg.dec_heads) if cfg.prev_read else None
        self.innov = InnovBranch(cfg)
        self.feat_head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, feat_dim))
        self.rgb_norm = nn.LayerNorm(D)
        self.rgb_conv = ConvUpsampler(D, tuple(cfg.conv_channels), cfg.conv_res_blocks)
        cc = cell_coords(cfg.grid, 0.0)
        self.register_buffer("xy_mask", torch.tensor([0.0, 1.0, 1.0]).expand_as(cc).clone(), persistent=False)
        self.register_buffer("xy_coords", cc, persistent=False)
        self.register_buffer("win", window_mask(cfg.grid, 1), persistent=False)

    def _rope(self, dtype):
        return self.rope.tables(self.xy_coords, self.xy_mask, dtype=dtype)       # [1, N, hd] -> broadcast

    def project(self, rho, K_src, R, tau, m=None, K_dst=None, scale=None) -> tuple:
        """Keyframe cells -> (p, z, valid, cull) in the target camera (bản 3.2 forward projection)."""
        cfg = self.cfg
        f32 = lambda t: None if t is None else t.float()
        with geo.fp32(rho.device):
            rho, K_src, R, tau, m, K_dst, scale = map(f32, (rho, K_src, R, tau, m, K_dst, scale))
            P = geo.unproject(rho, K_src, self.read.uq)
            P1 = geo.transform(P if m is None else P + m, R, tau, scale)
            p, z, valid = geo.project(P1, K_src if K_dst is None else K_dst, cfg.z_eps)
            cull = None
            if cfg.backface_kappa > 0:
                cull = geo.backface_bias(geo.cell_normals(P, cfg.grid), R, P1, cfg.backface_kappa)
        return p, z, valid, cull

    def predict(self, c, kind: str, proj: tuple, dt: float, rays=None, prev: Optional[torch.Tensor] = None):
        """-> (h_geom, h_prev or None, d_min). h_geom: geometric read + self-attention blocks."""
        p, z, valid, cull = proj
        h, d_min = self.read(c, kind, p, z, valid, cull, dt, rays)
        rope = self._rope(h.dtype)
        for blk in self.pred_blocks:
            h = blk(h, rope)
        h_prev = None
        if self.prev is not None and prev is not None:
            h_prev = self.prev(h, prev, rope, self.win)
        return h, h_prev, d_min

    def feat(self, h: torch.Tensor) -> torch.Tensor:
        return self.feat_head(h)

    def rgb(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, N, D] -> image [B, 3, H, W]."""
        return self.rgb_conv(self.rgb_norm(h), self.cfg.grid)
