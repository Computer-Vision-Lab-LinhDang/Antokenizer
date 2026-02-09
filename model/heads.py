from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    """Projects sequence tokens into latent reconstruction space."""

    def __init__(
        self,
        dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class SemanticHead(nn.Module):
    """Lightweight head for classification or distillation logits."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.norm(pooled)
        x = self.dropout(x)
        return self.out(x)


__all__ = ["ReconstructionHead", "SemanticHead"]
