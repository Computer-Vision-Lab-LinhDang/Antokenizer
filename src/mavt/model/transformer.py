"""Standard pre-norm Transformer block (SigLIP2-compatible)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardTransformerBlock(nn.Module):
    """Pre-LN transformer block with multi-head self-attention.

    Designed to receive SigLIP2 weight initialization via `load_siglip2_block`.
    """

    def __init__(self, dim: int = 1152, num_heads: int = 16,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Fused QKV for efficiency
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, D)
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Self-attention
        xn = self.norm1(x)
        qkv = self.qkv(xn).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)  # each (B, H, N, d)

        attn = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(B, N, D)

        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x
