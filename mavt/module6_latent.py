"""Module 6: Continuous Latent Projection.

Maps encoder output to a continuous latent space without quantization.
Two paths are computed:

  Generation path:    encoded → μ, σ → reparameterize → z  (B, N', 32)
  Understanding path: encoded → pool → z_understand         (B, 768)

The generation path uses VAE-style reparameterization + noise regularization.
z is used by the decoder (Module 7) for reconstruction.
z_understand is used for semantic tasks (classification, retrieval, etc.).

Design:
  - No codebook / VQ / FSQ — keeps continuous information.
  - KL weight is annealed during training (near zero → gradual).
  - Understanding path uses attention pooling (same as AToken's AttentionPooler).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import LatentConfig
from .types import EncoderOutput, LatentOutput


class VariationalProjection(nn.Module):
    """Projects encoder features to mean and log-variance for reparameterization."""

    def __init__(self, d_encoder: int, latent_dim: int) -> None:
        super().__init__()
        self.mu_proj = nn.Linear(d_encoder, latent_dim)
        self.logvar_proj = nn.Linear(d_encoder, latent_dim)

    def forward(
        self,
        x: torch.Tensor,  # (B, N', d_encoder)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var) each (B, N', latent_dim)."""
        return self.mu_proj(x), self.logvar_proj(x)


class ContinuousLatentProjection(nn.Module):
    """Continuous latent space projection (no quantization).

    Generation path:    z = μ + ε·σ  (reparameterization trick)
    Understanding path: z_understand = attention_pool(encoded)
    """

    def __init__(self, cfg: Optional[LatentConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or LatentConfig()

        self.norm = nn.LayerNorm(self.cfg.d_encoder)
        self.var_proj = VariationalProjection(
            d_encoder=self.cfg.d_encoder,
            latent_dim=self.cfg.latent_dim,
        )

        # Understanding path: attention pooler (CLS-style)
        self.cls_query = nn.Parameter(torch.randn(1, 1, self.cfg.d_encoder))
        nn.init.trunc_normal_(self.cls_query, std=0.02)
        self.understand_attn = nn.MultiheadAttention(
            embed_dim=self.cfg.d_encoder,
            num_heads=8,
            batch_first=True,
        )
        self.understand_proj = nn.Linear(self.cfg.d_encoder, self.cfg.d_understand)
        self.understand_norm = nn.LayerNorm(self.cfg.d_understand)

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """VAE reparameterization trick + optional noise regularization."""
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Noise regularization
        z = z + torch.randn_like(z) * self.cfg.noise_std
        return z

    def forward(self, encoder_out: EncoderOutput) -> LatentOutput:
        """Project encoder output to continuous latent.

        Args:
            encoder_out: MAVTEncoder output.

        Returns:
            LatentOutput with z, z_understand, mu, log_var.
        """
        x = self.norm(encoder_out.encoded)  # (B, N', d_encoder)

        # Generation path
        mu, log_var = self.var_proj(x)
        z = self.reparameterize(mu, log_var)

        # Understanding path
        B = x.size(0)
        cls = self.cls_query.expand(B, -1, -1)
        pooled, _ = self.understand_attn(query=cls, key=x, value=x)
        pooled = pooled.squeeze(1)  # (B, d_encoder)
        z_understand = self.understand_norm(self.understand_proj(pooled))

        return LatentOutput(z=z, z_understand=z_understand, mu=mu, log_var=log_var)

    def kl_loss(self, latent_out: LatentOutput) -> torch.Tensor:
        """KL divergence: KL(q(z|x) || N(0,I))."""
        mu, log_var = latent_out.mu, latent_out.log_var
        return (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean()


__all__ = ["VariationalProjection", "ContinuousLatentProjection"]
