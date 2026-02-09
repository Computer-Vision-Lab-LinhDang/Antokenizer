from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from atoken.core.patchify import PatchifyMetadata
from atoken.core.sparse_tensor import SparseTensor4D

from .blocks import TransformerBlock


class ATokenDecoder(nn.Module):
    """Symmetric Transformer decoder for reconstruction."""

    def __init__(
        self,
        *,
        out_channels: int = 3,
        patch_size: tuple[int, int, int] = (4, 16, 16),
        d_model: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        latent_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        latent_dim = latent_dim or d_model

        self.input_proj = nn.Linear(latent_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        kernel_t, kernel_h, kernel_w = patch_size
        self.output_proj = nn.Linear(
            d_model, out_channels * kernel_t * kernel_h * kernel_w
        )

    def forward(
        self,
        sparse: SparseTensor4D,
    ) -> Dict[str, torch.Tensor]:
        tokens = sparse.tokens
        positions = sparse.positions
        mask = sparse.mask

        x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x, positions=positions, mask=mask)
        x = self.norm(x)

        patch_tokens = self.output_proj(x)
        reconstruction = self._unpatchify(patch_tokens, sparse)
        return {"patch_tokens": patch_tokens, "reconstruction": reconstruction}

    def _unpatchify(
        self,
        patch_tokens: torch.Tensor,
        sparse: SparseTensor4D,
    ) -> torch.Tensor:
        metadata: Optional[PatchifyMetadata] = sparse.metadata.get("patchify")  # type: ignore[assignment]
        if metadata is None:
            raise ValueError("Patchify metadata missing; cannot unpatchify.")

        if metadata.stride != metadata.kernel:
            raise NotImplementedError(
                "Unpatchify currently supports only non-overlapping patches."
            )

        b = patch_tokens.size(0)
        n_total = (
            metadata.num_temporal * metadata.num_height * metadata.num_width
        )
        if patch_tokens.size(1) != n_total:
            raise ValueError(
                f"Token count mismatch: expected {n_total}, got {patch_tokens.size(1)}"
            )

        kernel_t, kernel_h, kernel_w = metadata.kernel
        n_t = metadata.num_temporal
        n_h = metadata.num_height
        n_w = metadata.num_width

        x = patch_tokens.view(
            b,
            n_t,
            n_h,
            n_w,
            self.out_channels,
            kernel_t,
            kernel_h,
            kernel_w,
        )
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(
            b,
            self.out_channels,
            n_t * kernel_t,
            n_h * kernel_h,
            n_w * kernel_w,
        )

        pad_t_front, pad_t_back = metadata.padding["t"]
        pad_h_pre, pad_h_post = metadata.padding["h"]
        pad_w_pre, pad_w_post = metadata.padding["w"]
        _, _, orig_t, orig_h, orig_w = metadata.original_shape

        t_end = x.size(2) - pad_t_back if pad_t_back > 0 else x.size(2)
        h_end = x.size(3) - pad_h_post if pad_h_post > 0 else x.size(3)
        w_end = x.size(4) - pad_w_post if pad_w_post > 0 else x.size(4)

        x = x[
            :,
            :,
            pad_t_front:t_end,
            pad_h_pre:h_end,
            pad_w_pre:w_end,
        ]

        if x.size(2) != orig_t or x.size(3) != orig_h or x.size(4) != orig_w:
            raise RuntimeError("Unpatchify produced mismatched spatial/temporal size.")
        return x


__all__ = ["ATokenDecoder"]
