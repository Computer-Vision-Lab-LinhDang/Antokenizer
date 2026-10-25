"""Two-stream backbone (MM-DiT style, separate weights) on frozen DINOv3: C stream at keyframes, D stream
at the other frames, with an asymmetric mask.

Per layer l (B' = clips x segments, T' = K - 1 non-key frames):

  block C   q = H_C (256 cells + 1 focal token)
            keys = [K_C ; K_mem(keyframe)]                              C never reads D or other frames
  block D   q = H_D^t (256 cells + 1 pose token + g^2 flow tokens), t = 1 .. K-1
            keys = [K_C ; K_D^(t-1) ; K_D^t ; K_mem(t)]                 sparse-causal (Tune-A-Video), sink = C
                   (K_D^(t-1) is empty for t = 1: the frame before it is the keyframe, i.e. C)
  link      q = the link token of segment k (D stream weights)
            keys = [K_C(k) ; K_C(k-1) ; K_link]                         K_C(k-1) cached, empty for k = 0

K_C, V_C come from block C of the same layer (same-layer joint attention). RoPE (t, x, y) with t = the
frame index in the clip, so C(k-1), C(k) and D(t) see their relative offsets. Posed views add a Plücker
ray embedding to the cell tokens instead of a view axis (camera indices have no order).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from ..dino_v0.camera import ZeroLinear
from .config import CD32Config
from .layers import MLP, Rope3, StreamBlock, split_heads

# token types
T_C_CELL, T_F, T_D_CELL, T_POSE, T_FLOW, T_LINK, T_SLOT = range(7)


def cell_coords(grid: int, t, device=None) -> torch.Tensor:
    """[..., N, 3] coords (t, x, y) of the cells; t a float or a tensor [...]."""
    r = torch.arange(grid, device=device, dtype=torch.float32) + 0.5
    y, x = torch.meshgrid(r, r, indexing="ij")
    xy = torch.stack([x, y], -1).reshape(-1, 2)
    t = torch.as_tensor(t, dtype=torch.float32, device=device)
    tt = t.unsqueeze(-1).unsqueeze(-1).expand(*t.shape, xy.shape[0], 1)
    return torch.cat([tt, xy.expand(*t.shape, -1, -1)], -1)


class TwoStreamBackbone(nn.Module):
    def __init__(self, cfg: CD32Config):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        d, G, g = cfg.dim, cfg.grid, cfg.flow_grid
        self.rope = Rope3(cfg.rope_dims, cfg.rope_base)
        self.type_emb = nn.Parameter(torch.randn(7, d) * 0.02)
        self.f_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pose_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.flow_tokens = nn.Parameter(torch.randn(1, g * g, d) * 0.02)
        self.link_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.c_blocks = nn.ModuleList([StreamBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.layers)])
        self.d_blocks = self.c_blocks if cfg.shared_streams else \
            nn.ModuleList([StreamBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.layers)])
        self.norm_c, self.norm_d = nn.LayerNorm(d), nn.LayerNorm(d)
        r = (torch.arange(g, dtype=torch.float32) + 0.5) * (G / g)
        fy, fx = torch.meshgrid(r, r, indexing="ij")
        self.register_buffer("flow_xy", torch.stack([fx, fy], -1).reshape(-1, 2), persistent=False)
        self.use_checkpoint = False

    # ------------------------------------------------------------------------------ tokens / ropes
    def _c_rope(self, t_key: torch.Tensor, dtype) -> tuple:
        """C tokens: cells at (t_k, x, y), focal token unrotated. t_key [B'] -> tables [B', 1, N+1, hd]."""
        G = self.cfg.grid
        cc = cell_coords(G, t_key, t_key.device)                                  # [B', N, 3]
        f = torch.zeros(cc.shape[0], 1, 3, device=cc.device)
        return self.rope.tables(torch.cat([cc, f], 1), dtype=dtype)

    def _d_rope(self, t: torch.Tensor, dtype) -> tuple:
        """D tokens of frames t [B', T'] -> tables [B', T', 1, N+1+g^2, hd]: cells (t, x, y), pose (t, -, -),
        flow tokens (t, x, y) at the coarse cell centres."""
        G = self.cfg.grid
        cc = cell_coords(G, t, t.device)                                          # [B', T', N, 3]
        tt = t.unsqueeze(-1).unsqueeze(-1)
        pose = torch.cat([tt, torch.zeros(*t.shape, 1, 2, device=t.device)], -1)
        fl = torch.cat([tt.expand(*t.shape, self.flow_xy.shape[0], 1),
                        self.flow_xy.expand(*t.shape, -1, -1)], -1)
        return self.rope.tables(torch.cat([cc, pose, fl], -2), dtype=dtype)

    def _mem_rope(self, t: torch.Tensor, n_layers: int, dtype) -> tuple:
        """Memory tokens (L layers x N cells) of frames t [...] -> tables [..., 1, L*N, hd]."""
        cc = cell_coords(self.cfg.grid, t, t.device)                              # [..., N, 3]
        cc = cc.unsqueeze(-3).expand(*cc.shape[:-2], n_layers, -1, -1).flatten(-3, -2)
        return self.rope.tables(cc, dtype=dtype)

    # ------------------------------------------------------------------------------ forward
    def forward(self, x0_key: torch.Tensor, mem_key: torch.Tensor, t_key: torch.Tensor,
                x0_d: Optional[torch.Tensor] = None, mem_d: Optional[torch.Tensor] = None,
                t_d: Optional[torch.Tensor] = None, n_seg: int = 1) -> dict:
        """x0_key [B', N, dim], mem_key [B', L, N, dim], t_key [B'] (frame index of each keyframe);
        x0_d [B', T', N, dim], mem_d [B', T', L, N, dim], t_d [B', T'] for the D stream (None: C only);
        B' = B * n_seg with segments of a clip consecutive (the link reads segment k - 1 of the same clip).
        -> dict(hc [B', N+1, dim], hd [B', T', N+1+g^2, dim] or None, link [B', dim] or None)."""
        cfg = self.cfg
        Bp, N, d = x0_key.shape
        L = mem_key.shape[1]
        dt = x0_key.dtype
        te = self.type_emb
        hc = torch.cat([x0_key + te[T_C_CELL], (self.f_token + te[T_F]).expand(Bp, -1, -1)], 1)
        rope_c = self._c_rope(t_key, dt)
        mem_c = mem_key.flatten(1, 2)                                             # [B', L*N, dim]
        rope_mc = self._mem_rope(t_key, L, dt)
        run_d = x0_d is not None
        if run_d:
            Tp = x0_d.shape[1]
            g2 = cfg.n_flow
            hd = torch.cat([x0_d + te[T_D_CELL],
                            (self.pose_token + te[T_POSE]).expand(Bp * Tp, -1, -1).view(Bp, Tp, 1, d),
                            (self.flow_tokens + te[T_FLOW]).expand(Bp * Tp, -1, -1).view(Bp, Tp, g2, d)], 2)
            rope_d = self._d_rope(t_d, dt)                                        # [B', T', 1, ND, hd]
            mem_dd = mem_d.flatten(2, 3)                                          # [B', T', L*N, dim]
            rope_md = self._mem_rope(t_d, L, dt)
            hl = (self.link_token + te[T_LINK]).expand(Bp, -1, -1)
            rope_l = self.rope.tables(torch.stack([t_key, torch.zeros_like(t_key), torch.zeros_like(t_key)],
                                                  -1).unsqueeze(1), dtype=dt)
            has_prev = (torch.arange(Bp, device=x0_key.device) % n_seg) > 0       # link of segment k >= 1
        for l in range(cfg.layers):
            if run_d:
                args = (hc, hd, hl, rope_c, mem_c, rope_mc, rope_d, mem_dd, rope_md, rope_l, has_prev, n_seg)
                fn = lambda *a, l=l: self._layer(l, *a)
                if self.use_checkpoint and self.training:
                    hc, hd, hl = ckpt.checkpoint(fn, *args, use_reentrant=False)
                else:
                    hc, hd, hl = fn(*args)
            else:
                fn = lambda h, l=l: self._c_block(l, h, rope_c, mem_c, rope_mc)[0]
                hc = ckpt.checkpoint(fn, hc, use_reentrant=False) if self.use_checkpoint and self.training else fn(hc)
        out = dict(hc=self.norm_c(hc), hd=None, link=None)
        if run_d:
            out["hd"], out["link"] = self.norm_d(hd), self.norm_d(hl[:, 0])
        return out

    def _c_block(self, l, hc, rope_c, mem_c, rope_mc):
        blk = self.c_blocks[l]
        q, k, v = blk.own(hc, rope_c)
        km, vm = blk.ctx(mem_c, rope_mc)
        return blk.finish(hc, q, [k, km], [v, vm]), k, v

    def _layer(self, l, hc, hd, hl, rope_c, mem_c, rope_mc, rope_d, mem_dd, rope_md, rope_l, has_prev, n_seg):
        hc_new, kc, vc = self._c_block(l, hc, rope_c, mem_c, rope_mc)              # kc [B', H, NC, hd]
        blk = self.d_blocks[l]
        Bp, Tp, ND, d = hd.shape
        H = blk.heads
        q, k, v = blk.own(hd, rope_d)                                              # [B', T', H, ND, hd]
        km, vm = blk.ctx(mem_dd, rope_md)                                          # [B', T', H, L*N, hd]
        zk, zv = torch.zeros_like(k[:, :1]), torch.zeros_like(v[:, :1])
        kp, vp = torch.cat([zk, k[:, :-1]], 1), torch.cat([zv, v[:, :-1]], 1)      # frame t - 1 (none at t = 1)
        NC, NM = kc.shape[-2], km.shape[-2]
        kcx, vcx = kc.unsqueeze(1).expand(-1, Tp, -1, -1, -1), vc.unsqueeze(1).expand(-1, Tp, -1, -1, -1)
        mask = torch.ones(Tp, NC + 2 * ND + NM, dtype=torch.bool, device=hd.device)
        mask[0, NC:NC + ND] = False
        mask = mask.view(1, Tp, 1, 1, -1)
        hd_new = blk.finish(hd, q, [kcx, kp, k, km], [vcx, vp, v, vm], mask)
        # link token: reads C_k and the cached C_{k-1} of the same clip
        ql, kl, vl = blk.own(hl, rope_l)
        idx = torch.arange(Bp, device=hd.device)
        prev = torch.where(has_prev, idx - 1, idx)
        lmask = torch.ones(Bp, 1, 1, 2 * NC + 1, dtype=torch.bool, device=hd.device)
        lmask[:, 0, 0, NC:2 * NC] = has_prev.view(Bp, 1)
        hl_new = blk.finish(hl, ql, [kc, kc[prev], kl], [vc, vc[prev], vl], lmask)
        return hc_new, hd_new, hl_new


class CHeads(nn.Module):
    """Final C stream -> content c (mu, logvar), log inverse depth rho, focal length (FOV)."""

    def __init__(self, cfg: CD32Config):
        super().__init__()
        d, L = cfg.dim, len(cfg.dino_layers)
        self.cfg = cfg
        self.content_image = nn.Linear(d, 2 * cfg.d_c_image)
        self.content_video = nn.Linear(d, 2 * cfg.d_c_video)
        # rho reads the stream and the multi-layer DINOv3 memory of the keyframe (a DPT-style depth head)
        self.rho_in = nn.Linear(d * (1 + L), d)
        self.rho_mlp = MLP(d, 1.0, out=1)
        nn.init.zeros_(self.rho_mlp.fc2.weight)
        nn.init.zeros_(self.rho_mlp.fc2.bias)
        self.fov = ZeroLinear(d, 1)
        p0 = (cfg.fov_init - cfg.fov_min) / (cfg.fov_max - cfg.fov_min)
        self.fov_bias = math.log(p0 / (1 - p0))
        # posed views: rho_k += tanh(alpha) * CrossAttn(Q = cells of view 0, KV = cells of views >= 1)
        self.refine = nn.MultiheadAttention(d, cfg.rho_refine_heads, batch_first=True)
        self.refine_norm_q, self.refine_norm_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.refine_out = nn.Linear(d, 1)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, hc: torch.Tensor, mem_key: torch.Tensor, modality: str,
                other_views: Optional[torch.Tensor] = None) -> dict:
        cfg = self.cfg
        cells, ftok = hc[:, :-1], hc[:, -1]
        head = self.content_image if modality == "image" else self.content_video
        mu, logvar = head(cells).float().chunk(2, -1)
        logvar = logvar.clamp(-30, 20)
        mem = mem_key.permute(0, 2, 1, 3).flatten(2)                               # [B', N, L*dim]
        rho = self.rho_mlp(torch.relu(self.rho_in(torch.cat([cells, mem], -1)))).squeeze(-1).float()
        if other_views is not None:                                                # [B', V-1, N, dim]
            kv = self.refine_norm_kv(other_views.flatten(1, 2))
            a, _ = self.refine(self.refine_norm_q(cells), kv, kv, need_weights=False)
            rho = rho + torch.tanh(self.alpha) * self.refine_out(a).squeeze(-1).float()
        s = torch.sigmoid(self.fov(ftok).squeeze(-1).float() + self.fov_bias)
        fov = cfg.fov_min + (cfg.fov_max - cfg.fov_min) * s
        return dict(mu=mu, logvar=logvar, rho=rho, fov=fov)


class DHeads(nn.Module):
    """Final D stream -> pose xi_t (6), coarse 3D flow M_t (g x g x 3); link token -> 7 numbers.
    All zero-init: identity pose, no flow, identity link at the start."""

    def __init__(self, cfg: CD32Config):
        super().__init__()
        self.cfg = cfg
        self.pose = ZeroLinear(cfg.dim, 6)
        self.flow = ZeroLinear(cfg.dim, 3)
        self.link = ZeroLinear(cfg.dim, 7)

    def forward(self, hd: torch.Tensor, link: torch.Tensor) -> dict:
        N, g = self.cfg.n_cells, self.cfg.flow_grid
        xi = self.pose(hd[..., N, :]).float()                                       # [B', T', 6]
        M = self.flow(hd[..., N + 1:, :]).float()                                   # [B', T', g^2, 3]
        M = M.unflatten(-2, (g, g))
        return dict(xi=xi, M=M, link=self.link(link).float())
