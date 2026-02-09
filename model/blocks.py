from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from atoken.core.rope4d import apply_rope_4d


def _mask_tensor(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x
    if mask.dim() != 2:
        raise ValueError("Mask must have shape (B, N)")
    return x * mask.unsqueeze(-1).to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        activation_layer = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }.get(activation, nn.GELU)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = activation_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # Each (B, N, C)

        q = q.view(b, n, self.num_heads, self.head_dim)
        k = k.view(b, n, self.num_heads, self.head_dim)
        v = v.view(b, n, self.num_heads, self.head_dim)

        q, k = apply_rope_4d(q, k, positions)

        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            key_mask = ~mask[:, None, None, :]  # (B, 1, 1, N)
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        context = torch.matmul(attn_weights, v)  # (B, H, N, D)
        context = context.transpose(1, 2).contiguous().view(b, n, c)
        context = self.proj(context)
        context = self.proj_drop(context)

        if mask is not None:
            context = _mask_tensor(context, mask)

        return context


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out = self.attn(x, positions=positions, mask=mask)
        x = residual + attn_out
        x = _mask_tensor(x, mask)

        residual = x
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out
        x = _mask_tensor(x, mask)
        return x


__all__ = ["TransformerBlock", "MultiHeadSelfAttention", "FeedForward"]
