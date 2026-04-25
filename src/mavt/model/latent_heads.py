"""Stage 4: Latent projection heads.

VAEHead      → μ, logσ², z  (reparameterization, latent_dim per token)
SemanticHead → legacy pre-VAE semantic projection, currently unused by MAVT
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEHead(nn.Module):
    """Per-token VAE projection: [C;D] tokens -> latent_dim latent."""

    def __init__(self, in_dim: int = 1152, latent_dim: int = 32, kl_weight: float = 1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        # Outputs mu + logvar concatenated → 2 * latent_dim
        self.proj = nn.Linear(in_dim, 2 * latent_dim)

    def forward(
        self, compressed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        compressed : (B, N_c+N_d, D)

        Returns
        -------
        z      : (B, N_c+N_d, latent_dim)  — sampled latent
        mu     : (B, N_c+N_d, latent_dim)
        logvar : (B, N_c+N_d, latent_dim)
        loss_kl: scalar
        """
        stats = self.proj(compressed)                # (B, N, 2*L)
        mu, logvar = stats.chunk(2, dim=-1)          # each (B, N, L)
        logvar = logvar.clamp(-30, 20)               # numerical stability
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        loss_kl = self.kl_weight * kl.mean()

        return z, mu, logvar, loss_kl

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)


class SemanticHead(nn.Module):
    """Attention-pooled semantic projection: [C;D] tokens → 768-d sequence vector."""

    def __init__(self, in_dim: int = 1152, out_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, in_dim) * (in_dim ** -0.5))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads,
            batch_first=True, bias=True,
        )
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, compressed: torch.Tensor) -> torch.Tensor:
        """compressed : (B, N, D) → s : (B, out_dim)"""
        B = compressed.shape[0]
        q = self.query.expand(B, 1, -1)          # (B, 1, D)
        pooled, _ = self.pool_attn(q, compressed, compressed)  # (B, 1, D)
        pooled = self.norm(pooled.squeeze(1))     # (B, D)
        return self.proj(pooled)                  # (B, out_dim)
