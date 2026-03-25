"""Module 5: MAVT Encoder.

Pre-norm Transformer encoder with 4D RoPE.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.rope4d import apply_rope_4d

from .config import EncoderConfig
from .types import EncoderOutput, PatchifyOutput


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block with 4D RoPE attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        r = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if positions is not None:
            q, k = apply_rope_4d(q, k, positions)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        x = r + out
        x = x + self.ffn(self.norm2(x))
        return x


class MAVTEncoder(nn.Module):
    """Transformer encoder with 4D RoPE. No graph, no stubs."""

    def __init__(self, cfg: Optional[EncoderConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()

        n_heads = 16 if self.cfg.d_model >= 1024 else 12
        while self.cfg.d_model % n_heads != 0:
            n_heads -= 1

        n_total = self.cfg.n_blocks_stage3 + self.cfg.n_blocks_stage4

        self.blocks = nn.ModuleList([
            _TransformerBlock(self.cfg.d_model, n_heads)
            for _ in range(n_total)
        ])
        self.norm_out = nn.LayerNorm(self.cfg.d_model)

    def forward(self, patch_out: PatchifyOutput) -> EncoderOutput:
        x = patch_out.f_spatial
        positions = patch_out.positions

        for block in self.blocks:
            x = block(x, positions=positions)

        x = self.norm_out(x)
        return EncoderOutput(encoded=x, positions_out=positions)


__all__ = ["MAVTEncoder"]
