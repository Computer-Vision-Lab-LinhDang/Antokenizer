"""Latent heads of MAVT on DINOv3, v0.

z_r (reconstruction): Gaussian heads with KL (as in AToken) on C (per cell) and on D (per token).
    image: C -> 256 x zr_c_image; video: C -> 256 x zr_c_video per segment, D -> M x zr_d per frame.
z_s (understanding): attention pool of one learned query over Z_t = [C_k ; up(D_t)] -> zs_dim.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import DinoV0Config

LOGVAR_MIN, LOGVAR_MAX = -30.0, 20.0


def budget(cfg: DinoV0Config, modality: str) -> int:
    """Latent values per image (image) or per K-frame segment (video)."""
    return cfg.budget(modality)


class LatentHeads(nn.Module):
    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        self.cfg = cfg
        self.c_image = nn.Linear(cfg.dim, 2 * cfg.zr_c_image)
        self.c_video = nn.Linear(cfg.dim, 2 * cfg.zr_c_video)
        self.d = nn.Linear(cfg.d_dim, 2 * cfg.zr_d)

    def _gauss(self, h: torch.Tensor):
        mu, logvar = h.chunk(2, dim=-1)
        logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z, mu, logvar

    def encode_c(self, C: torch.Tensor, modality: str):
        """C [..., N, dim] -> (z, mu, logvar) each [..., N, zr_c_<modality>]."""
        head = self.c_image if modality == "image" else self.c_video
        return self._gauss(head(C))

    def encode_d(self, D: torch.Tensor):
        """D [..., M, d_dim] -> (z, mu, logvar) each [..., M, zr_d]."""
        return self._gauss(self.d(D))

    @staticmethod
    def kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mu, exp(logvar)) || N(0, 1)), mean over all elements."""
        return 0.5 * (mu.float().pow(2) + logvar.float().exp() - 1.0 - logvar.float()).mean()


class SemanticPool(nn.Module):
    """z_s = Linear(MHA(q, [C_k ; up(D_t)])): one learned query, cfg.heads heads, width dim -> zs_dim."""

    def __init__(self, cfg: DinoV0Config):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)
        self.norm = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads, batch_first=True)
        self.proj = nn.Linear(cfg.dim, cfg.zs_dim)

    def forward(self, C: torch.Tensor, D_up: Optional[torch.Tensor] = None) -> torch.Tensor:
        """C [B, N, dim], D_up [B, M, dim] | None -> z_s [B, zs_dim]."""
        Z = C if D_up is None else torch.cat([C, D_up], 1)
        kv = self.norm(Z)
        q = self.query.expand(Z.shape[0], -1, -1).to(Z.dtype)
        o, _ = self.attn(q, kv, kv, need_weights=False)
        return self.proj(o[:, 0])
