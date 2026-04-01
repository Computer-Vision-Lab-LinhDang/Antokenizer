"""Token expansion modules for decoding compressed representations.

UnifiedDetailExpander   Position-queried cross-attention: (Nc+Nd) → N_original.
                        Works for any modality — queries are generated from
                        the original 4D grid positions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UnifiedDetailExpander(nn.Module):
    """Expand compressed tokens back to original grid resolution.

    Each original grid position generates a query (via a position encoder),
    then cross-attends to the compressed content+detail tokens.

    This is modality-agnostic: the position encoder learns the meaning of
    (t, x, y, z) coordinates from data.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 2,
    ) -> None:
        super().__init__()

        # Convert 4D position → d_model query vector.
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    d_model, n_heads, batch_first=True,
                ),
                "norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                "norm_ffn": nn.LayerNorm(d_model),
            }))

    def forward(
        self,
        compressed_z: torch.Tensor,       # (B, Nc+Nd, d_model)
        original_positions: torch.Tensor,  # (B, N_orig, 4)
    ) -> torch.Tensor:
        """Expand compressed tokens to original grid.

        Returns: ``(B, N_orig, d_model)``
        """
        # Position-based queries: each grid point asks "what feature do I have?"
        Q = self.pos_encoder(original_positions)  # (B, N_orig, d_model)

        out = Q
        for layer in self.layers:
            residual = out
            attn_out, _ = layer["cross_attn"](out, compressed_z, compressed_z)
            out = layer["norm"](attn_out + residual)
            out = layer["norm_ffn"](layer["ffn"](out) + out)

        return out


__all__ = ["UnifiedDetailExpander"]
