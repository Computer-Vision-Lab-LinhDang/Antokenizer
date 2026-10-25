"""Vanilla C-D: the bản 3.2 latent and its geometric decoder on one plain transformer.

Same latent as the two-stream model (C_k = (c, rho, f) at the keyframe; D_t = (xi, M, r[:L]) at the other
frames; same budget and nested dropout) and the same forward projection + k-NN read in the decoder. What is
plain here:

  backbone   one pre-LN transformer (learned cell positions and frame embeddings, no RoPE, no per-layer DINO
             memory: the 4 DINOv3 layers are summed into the input tokens). The tokens of a segment are
                 [C: 256 cells + focal]  +  per non-key frame t  [256 cells + pose + g^2 flow + L_max slots]
             and the asymmetric mask of the design is an attention mask: C reads C only; D_t reads C, D_t
             and D_{t-1} (sparse-causal; `causal_d: false` lets D_t read every D frame of the segment).
  innovation the slot tokens of the same pass give D_innov (mu, logvar): no trial decode, no Enc_R.
  decoder    geometric read of c_k, then `dec_blocks` vanilla decoder blocks (self-attention over the cells,
             cross-attention to [null ; r[:L]], MLP), batched over all frames of a segment; a linear pixel
             head (unpatchify + one 3 x 3 conv) and a linear DINO feature head.
  dropped    link between keyframes, previous-frame read, L_pred, cross-view rho refinement.

`geometry: false` is the ablation: every frame is decoded through the identity projection, so D_innov has
to carry all change (same budget).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from ..dino_v0.camera import ZeroLinear
from ..dino_v0.decoder import ConvUpsampler
from .config import CD32Config, tiny_config
from .decoder import GeomRead, forward_project
from .layers import MLP, StreamBlock, attention, merge_heads, split_heads


@dataclass
class CDVConfig(CD32Config):
    """CD32Config plus the vanilla knobs; the two-stream fields it does not use are ignored."""
    dec_blocks: int = 4                   # vanilla decoder blocks after the geometric read
    rgb_head: str = "linear"              # linear: unpatchify + 3 x 3 conv | conv: the v0 conv upsampler
    pix_out: int = 8                      # channels per pixel before the 3 x 3 conv (linear head)
    causal_d: bool = True                 # D_t reads D_{t-1} and D_t; False: all D frames of the segment
    geometry: bool = True                 # False: identity projection everywhere (ablation)
    p_l0: float = 0.25                    # more L = 0 draws: they are the only geometry-only decode
    w_feat: float = 0.25                  # DINO feature loss is auxiliary; the pixels are the target
    w_lpips: float = 0.5

    def check(self) -> None:
        super().check()
        assert self.rgb_head in ("linear", "conv"), "rgb_head: linear | conv"
        assert self.dec_blocks >= 1


def tiny_cdv_config(**kw) -> CDVConfig:
    """CPU config for unit tests (stub DINO, 64^2 images); no LPIPS (would download VGG weights)."""
    base = dataclasses.asdict(tiny_config())
    base.update(dec_blocks=1, w_lpips=0.0)
    base.update(kw)
    return CDVConfig(**base)


# token types
T_C_CELL, T_F, T_D_CELL, T_POSE, T_FLOW, T_SLOT = range(6)


class VanillaBackbone(nn.Module):
    """One transformer over [C tokens ; D tokens of frames 1 .. T'] with the asymmetric mask."""

    def __init__(self, cfg: CDVConfig):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        d, N, g2, Lm, K = cfg.dim, cfg.n_cells, cfg.n_flow, cfg.L_max, cfg.keyframe_every
        p = lambda *s: nn.Parameter(torch.randn(*s) * 0.02)
        self.type_emb = p(6, d)
        self.pos_emb = p(1, N, d)                 # cell position, shared by C and D cells
        self.frame_emb = p(K, d)                  # 0 = keyframe, j = j-th frame (or view) after it
        self.f_token, self.pose_token = p(1, 1, d), p(1, 1, d)
        self.flow_tokens, self.slot_tokens = p(1, g2, d), p(1, Lm, d)
        self.blocks = nn.ModuleList([StreamBlock(d, cfg.heads, cfg.mlp_ratio, ctx=False) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(d)
        self.use_checkpoint = False
        self._masks = {}

    @property
    def n_c(self) -> int:
        return self.cfg.n_cells + 1

    @property
    def n_d(self) -> int:
        return self.cfg.n_cells + 1 + self.cfg.n_flow + self.cfg.L_max

    def mask(self, Tp: int, device) -> torch.Tensor:
        """Additive float mask [1, 1, n, n] (0 = may attend, -inf = may not) for T' non-key frames."""
        key = (Tp, str(device))
        if key not in self._masks:
            NC, ND = self.n_c, self.n_d
            n = NC + Tp * ND
            ok = torch.zeros(n, n, dtype=torch.bool)
            ok[:NC, :NC] = True                                                  # C reads C only
            for j in range(Tp):
                rows = slice(NC + j * ND, NC + (j + 1) * ND)
                ok[rows, :NC] = True                                             # D_t reads C
                ok[rows, rows] = True                                            # and itself
                if not self.cfg.causal_d:
                    ok[rows, NC:] = True                                         # every D frame
                elif j > 0:
                    ok[rows, NC + (j - 1) * ND:NC + j * ND] = True               # and D_{t-1}
            m = torch.zeros(n, n).masked_fill(~ok, float("-inf"))
            self._masks[key] = m.view(1, 1, n, n).to(device)
        return self._masks[key]

    def forward(self, x_key: torch.Tensor, x_d: Optional[torch.Tensor] = None) -> dict:
        """x_key [B', N, dim] keyframe cell tokens; x_d [B', T', N, dim] cell tokens of the other frames
        (None: C only). -> dict(c [B', N, dim], f [B', dim], d_cells [B', T', N, dim], pose [B', T', dim],
        flow [B', T', g^2, dim], slots [B', T', L_max, dim]); the D entries are None without x_d."""
        cfg = self.cfg
        Bp, N, d = x_key.shape
        g2, Lm = cfg.n_flow, cfg.L_max
        te, fe = self.type_emb, self.frame_emb
        hc = torch.cat([x_key + self.pos_emb + te[T_C_CELL] + fe[0],
                        (self.f_token + te[T_F] + fe[0]).expand(Bp, -1, -1)], 1)
        Tp = 0 if x_d is None else x_d.shape[1]
        if Tp:
            if Tp >= fe.shape[0]:
                raise ValueError(f"{Tp} frames after the keyframe need keyframe_every > {Tp}")
            cells = x_d + self.pos_emb.unsqueeze(1) + te[T_D_CELL]
            pose = (self.pose_token + te[T_POSE]).view(1, 1, 1, d).expand(Bp, Tp, 1, d)
            flow = (self.flow_tokens + te[T_FLOW]).view(1, 1, g2, d).expand(Bp, Tp, g2, d)
            slots = (self.slot_tokens + te[T_SLOT]).view(1, 1, Lm, d).expand(Bp, Tp, Lm, d)
            hd = torch.cat([cells, pose, flow, slots], 2) + fe[1:Tp + 1].view(1, Tp, 1, d)
            h = torch.cat([hc, hd.flatten(1, 2)], 1)
            mask = self.mask(Tp, h.device)
        else:
            h, mask = hc, None
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                h = ckpt.checkpoint(blk, h, None, None, None, mask, use_reentrant=False)
            else:
                h = blk(h, mask=mask)
        h = self.norm(h)
        out = dict(c=h[:, :N], f=h[:, N], d_cells=None, pose=None, flow=None, slots=None)
        if Tp:
            hd = h[:, self.n_c:].view(Bp, Tp, self.n_d, d)
            out.update(d_cells=hd[:, :, :N], pose=hd[:, :, N], flow=hd[:, :, N + 1:N + 1 + g2],
                       slots=hd[:, :, N + 1 + g2:])
        return out


class VanillaHeads(nn.Module):
    """Backbone output -> content (mu, logvar), rho, FOV, pose xi, flow M, innovation (mu, logvar)."""

    def __init__(self, cfg: CDVConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.content_image = nn.Linear(d, 2 * cfg.d_c_image)
        self.content_video = nn.Linear(d, 2 * cfg.d_c_video)
        self.rho = MLP(d, 1.0, out=1)
        nn.init.zeros_(self.rho.fc2.weight)
        nn.init.zeros_(self.rho.fc2.bias)
        self.fov = ZeroLinear(d, 1)
        p0 = (cfg.fov_init - cfg.fov_min) / (cfg.fov_max - cfg.fov_min)
        self.fov_bias = math.log(p0 / (1 - p0))
        self.pose, self.flow = ZeroLinear(d, 6), ZeroLinear(d, 3)      # identity pose, no flow at init
        self.slot_out = nn.Linear(d, 2 * cfg.d_r)

    def forward(self, bb: dict, modality: str) -> dict:
        cfg = self.cfg
        head = self.content_image if modality == "image" else self.content_video
        mu, logvar = head(bb["c"]).float().chunk(2, -1)
        s = torch.sigmoid(self.fov(bb["f"]).squeeze(-1).float() + self.fov_bias)
        out = dict(mu=mu, logvar=logvar.clamp(-30, 20), rho=self.rho(bb["c"]).squeeze(-1).float(),
                   fov=cfg.fov_min + (cfg.fov_max - cfg.fov_min) * s)
        if bb["slots"] is not None:
            g = cfg.flow_grid
            r_mu, r_logvar = self.slot_out(bb["slots"]).float().chunk(2, -1)
            out.update(xi=self.pose(bb["pose"]).float(), M=self.flow(bb["flow"]).float().unflatten(-2, (g, g)),
                       r_mu=r_mu, r_logvar=r_logvar.clamp(-30, 20))
        return out


class DecBlock(nn.Module):
    """Vanilla decoder block over the cells of one frame: self-attention, cross-attention to
    [null ; innovation tokens] (null value 0, output projection without bias: L = 0 adds nothing), MLP."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.qkv, self.out1 = nn.LayerNorm(dim), nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out2 = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim, bias=False)
        self.null_k = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm3, self.mlp = nn.LayerNorm(dim), MLP(dim, 4.0)

    def forward(self, x: torch.Tensor, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [B, N, D], tok [B, L_max, D], valid [B, L_max] bool (token j < L)."""
        B, H = x.shape[0], self.heads
        q, k, v = (split_heads(t, H) for t in self.qkv(self.norm1(x)).chunk(3, -1))
        x = x + self.out1(merge_heads(attention(q, k, v)))
        q = split_heads(self.q(self.norm_q(x)), H)
        k, v = self.kv(self.norm_kv(tok)).chunk(2, -1)
        k = torch.cat([self.null_k.expand(B, -1, -1).to(k.dtype), k], 1)
        v = torch.cat([torch.zeros_like(v[:, :1]), v], 1)
        mask = torch.cat([torch.ones_like(valid[:, :1]), valid], 1).view(B, 1, 1, -1)
        x = x + self.out2(merge_heads(attention(q, split_heads(k, H), split_heads(v, H), mask)))
        return x + self.mlp(self.norm3(x))


class VanillaDecoder(nn.Module):
    """Geometric read (the 3.2 operator) -> decoder blocks with the innovation tokens -> pixels and features."""

    def __init__(self, cfg: CDVConfig, feat_dim: int):
        super().__init__()
        D, N, Lm, p = cfg.dec_dim, cfg.n_cells, cfg.L_max, cfg.patch
        self.cfg = cfg
        self.read = GeomRead(cfg)
        self.pos_emb = nn.Parameter(torch.randn(1, N, D) * 0.02)
        self.tok_in = nn.Linear(cfg.d_r, D)
        self.slot_emb = nn.Parameter(torch.randn(1, Lm, D) * 0.02)
        self.blocks = nn.ModuleList([DecBlock(D, cfg.dec_heads) for _ in range(cfg.dec_blocks)])
        self.norm = nn.LayerNorm(D)
        self.feat_head = nn.Linear(D, feat_dim)
        if cfg.rgb_head == "linear":
            self.pix = nn.Linear(D, p * p * cfg.pix_out)
            self.pix_conv = nn.Conv2d(cfg.pix_out, 3, 3, padding=1)
        else:
            self.rgb_conv = ConvUpsampler(D, tuple(cfg.conv_channels), cfg.conv_res_blocks)

    def project(self, rho, K_src, R, tau, m=None, K_dst=None, scale=None) -> tuple:
        """Keyframe cells -> (p, z, valid, cull) in the target camera."""
        return forward_project(self.cfg, self.read.uq, rho, K_src, R, tau, m, K_dst, scale)

    def forward(self, c: torch.Tensor, kind: str, proj: tuple, dt, tokens: torch.Tensor, L: torch.Tensor,
                rays: Optional[torch.Tensor] = None) -> torch.Tensor:
        """c [B, N, d_c]; proj (p, z, valid, cull) of the frame; dt float or [B]; tokens [B, L_max, d_r];
        L [B] long; rays [B, N, 6] or None -> h [B, N, D] (normed)."""
        p, z, valid, cull = proj
        h, _ = self.read(c, kind, p, z, valid, cull, dt, rays)
        h = h + self.pos_emb
        tok = self.tok_in(tokens.to(h.dtype)) + self.slot_emb
        keep = torch.arange(self.cfg.L_max, device=L.device).view(1, -1) < L.view(-1, 1)
        for blk in self.blocks:
            h = blk(h, tok, keep)
        return self.norm(h)

    def feat(self, h: torch.Tensor) -> torch.Tensor:
        return self.feat_head(h)

    def rgb(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, N, D] -> image [B, 3, H, W] (no output activation; targets in [-1, 1])."""
        cfg = self.cfg
        G, p = cfg.grid, cfg.patch
        if cfg.rgb_head == "conv":
            return self.rgb_conv(h, G)
        B = h.shape[0]
        x = self.pix(h).view(B, G, G, p, p, cfg.pix_out).permute(0, 5, 1, 3, 2, 4).reshape(B, cfg.pix_out, G * p, G * p)
        return self.pix_conv(F.gelu(x))
