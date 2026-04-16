from __future__ import annotations

import torch
from torch import nn


class SemanticHead(nn.Module):
    def __init__(self, embed_dim: int, text_embed_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Linear(embed_dim, text_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool(query, x, x, need_weights=False)
        return self.proj(pooled.squeeze(1))
