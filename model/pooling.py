from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AttentionPooler(nn.Module):
    """Attention pooling without CLS tokens, following the paper's head."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, dim = tokens.shape
        query = self.query.expand(b, -1, -1)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        pooled, _ = self.attn(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
        )
        pooled = pooled.squeeze(1)
        pooled = self.norm(pooled)
        return pooled


__all__ = ["AttentionPooler"]
