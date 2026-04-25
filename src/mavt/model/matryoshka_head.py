"""Stage 4 — Matryoshka head.

Operates on the slot-pooled output ``compressed`` (B, N, D_max). For each
nested prefix ``d_k`` exposed by ``compressed[..., :d_k]`` we run two
parallel branches that share the same tokens but differ in channel width:

  • Reconstruction:   VAEHead_k(d_k -> 2*latent_dim) → (z_k, mu_k, logvar_k, kl_k)
                      z_k is always ``latent_dim``-d so a single shared
                      AsymmetricDecoder can decode every prefix.
  • Understanding:    learnable query q_k attends over the d_k-channel tokens
                      → g_k ∈ R^{d_k}, then per-prefix heads
                      (semantic distillation, optional classification).

All sub-modules are eagerly constructed in ``__init__`` so their parameters
are registered before ``configure_optimizers`` runs.
"""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _pick_num_heads(dim: int, max_heads: int = 8) -> int:
    """Largest divisor of ``dim`` not exceeding ``max_heads`` (≥1)."""
    for h in range(min(max_heads, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


class _PrefixVAEHead(nn.Module):
    """Linear(d_k -> 2·latent_dim) with reparameterised sampling and unweighted KL."""

    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.proj(x).chunk(2, dim=-1)
        logvar = logvar.clamp(-30, 20)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        return z, mu, logvar, kl


class MatryoshkaHead(nn.Module):
    """Per-prefix VAE + understanding pool over a Matryoshka of channel widths.

    Parameters
    ----------
    dims          : nested channel widths, e.g. (8, 64, 256, 512, 1152).
                    Must be sorted ascending; the last entry must equal the
                    width of ``compressed`` fed at forward time.
    latent_dim    : VAE latent width, shared across prefixes.
    semantic_dim  : output dim of the semantic distillation head.
    num_classes   : optional classification head; disabled when ``None``.
    max_pool_heads: cap on attention heads for the pooler (smaller prefixes
                    fall back to the largest divisor of ``d_k``).
    """

    def __init__(
        self,
        dims: Sequence[int] = (8, 64, 256, 512, 1152),
        latent_dim: int = 32,
        semantic_dim: int = 768,
        num_classes: Optional[int] = None,
        max_pool_heads: int = 8,
    ):
        super().__init__()
        self.dims: Tuple[int, ...] = tuple(int(d) for d in dims)
        self.latent_dim = int(latent_dim)
        self.semantic_dim = int(semantic_dim)
        if not self.dims or list(self.dims) != sorted(self.dims):
            raise ValueError(f"dims must be ascending, got {self.dims}")

        # Per-prefix reconstruction head
        self.vae_heads = nn.ModuleDict({
            str(d): _PrefixVAEHead(d, latent_dim) for d in self.dims
        })

        # Per-prefix understanding pool
        self.pool_queries = nn.ParameterDict({
            str(d): nn.Parameter(torch.randn(1, 1, d) * (d ** -0.5))
            for d in self.dims
        })
        self.pool_attn = nn.ModuleDict({
            str(d): nn.MultiheadAttention(
                embed_dim=d,
                num_heads=_pick_num_heads(d, max_pool_heads),
                batch_first=True, bias=True,
            )
            for d in self.dims
        })
        self.pool_norm = nn.ModuleDict({
            str(d): nn.LayerNorm(d) for d in self.dims
        })

        # Per-prefix downstream heads on the pooled global vector g_k
        self.sem_heads = nn.ModuleDict({
            str(d): nn.Linear(d, semantic_dim) for d in self.dims
        })
        self.cls_heads = (
            nn.ModuleDict({str(d): nn.Linear(d, num_classes) for d in self.dims})
            if num_classes else None
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self, compressed: torch.Tensor
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Run both branches for every prefix.

        Parameters
        ----------
        compressed : (B, N, D_max)

        Returns
        -------
        dict keyed by ``d_k`` containing per-prefix tensors:
            z, mu, logvar, kl     — reconstruction branch
            g, sem, [cls]         — understanding branch
        """
        if compressed.shape[-1] != self.dims[-1]:
            raise ValueError(
                f"MatryoshkaHead expected last dim {self.dims[-1]}, "
                f"got {compressed.shape[-1]}"
            )

        B = compressed.shape[0]
        out: Dict[int, Dict[str, torch.Tensor]] = {}
        for d in self.dims:
            tokens = (compressed if d == compressed.shape[-1]
                      else compressed[..., :d]).contiguous()

            # Reconstruction branch — VAE bottleneck
            z, mu, logvar, kl = self.vae_heads[str(d)](tokens)

            # Understanding branch — attention pool to a single global vector
            q = self.pool_queries[str(d)].expand(B, -1, -1)
            pooled, _ = self.pool_attn[str(d)](q, tokens, tokens)
            g = self.pool_norm[str(d)](pooled.squeeze(1))

            entry: Dict[str, torch.Tensor] = {
                'z': z, 'mu': mu, 'logvar': logvar, 'kl': kl,
                'g': g,
                'sem': self.sem_heads[str(d)](g),
            }
            if self.cls_heads is not None:
                entry['cls'] = self.cls_heads[str(d)](g)
            out[d] = entry
        return out
