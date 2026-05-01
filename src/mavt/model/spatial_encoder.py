"""Spatial reconstruction encoder — backbone features → z_spatial."""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    """Linear VAE bottleneck applied per spatial token.

    Keeps the full spatial arrangement of backbone tokens intact.
    Used exclusively for pixel reconstruction (not for understanding).
    A very small KL weight (w_kl_spatial ≈ 1e-4) keeps the encoder
    near-deterministic and maximally informative.
    """

    def __init__(self, in_dim: int = 1152, latent_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, N, in_dim) → (z, mu, logvar, kl)  — z: (B, N, latent_dim)"""
        mu, logvar = self.proj(x).chunk(2, dim=-1)
        logvar = logvar.clamp(-30, 20)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        return z, mu, logvar, kl
