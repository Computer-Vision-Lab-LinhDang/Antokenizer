"""Continuous Latent Projection — reused from mavt/module6_latent.py.

Two paths:
  Generation:    encoded -> mu, sigma -> reparameterize -> z (B, N', 32)
  Understanding: encoded -> attention pool -> z_understand  (B, 768)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import LatentConfig
from .types import EncoderOutput, LatentOutput


class ContinuousLatentProjection(nn.Module):

    def __init__(self, cfg: Optional[LatentConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or LatentConfig()

        self.norm = nn.LayerNorm(self.cfg.d_encoder)
        self.mu_proj = nn.Linear(self.cfg.d_encoder, self.cfg.latent_dim)
        self.logvar_proj = nn.Linear(self.cfg.d_encoder, self.cfg.latent_dim)

        # Understanding path: CLS attention pooling
        self.cls_query = nn.Parameter(torch.randn(1, 1, self.cfg.d_encoder))
        nn.init.trunc_normal_(self.cls_query, std=0.02)
        self.understand_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.d_encoder, num_heads=8, batch_first=True,
        )
        self.understand_proj = nn.Linear(self.cfg.d_encoder, self.cfg.d_understand)
        self.understand_norm = nn.LayerNorm(self.cfg.d_understand)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z + torch.randn_like(z) * self.cfg.noise_std

    def forward(self, enc_out: EncoderOutput) -> LatentOutput:
        x = self.norm(enc_out.encoded)

        mu = self.mu_proj(x)
        log_var = self.logvar_proj(x)
        z = self.reparameterize(mu, log_var)

        B = x.size(0)
        cls = self.cls_query.expand(B, -1, -1)
        pooled, _ = self.understand_attn(query=cls, key=x, value=x)
        z_understand = self.understand_norm(self.understand_proj(pooled.squeeze(1)))

        return LatentOutput(z=z, z_understand=z_understand, mu=mu, log_var=log_var)

    def kl_loss(self, lat: LatentOutput) -> torch.Tensor:
        return (-0.5 * (1 + lat.log_var - lat.mu.pow(2) - lat.log_var.exp())).mean()


__all__ = ["ContinuousLatentProjection"]
