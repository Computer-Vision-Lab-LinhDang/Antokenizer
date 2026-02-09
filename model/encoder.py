from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from atoken.core.patchify import SpaceTimePatchifier
from atoken.core.sparse_tensor import SparseTensor4D

from .blocks import TransformerBlock
from .pooling import AttentionPooler


class ATokenEncoder(nn.Module):
    """Transformer encoder that operates on sparse 4D tokens."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        patch_size: tuple[int, int, int] = (4, 16, 16),
        stride: Optional[tuple[int, int, int]] = None,
        d_model: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        patchifier: Optional[SpaceTimePatchifier] = None,
    ) -> None:
        super().__init__()
        self.patchifier = patchifier or SpaceTimePatchifier(
            patch_size=patch_size,
            stride=stride,
        )
        kernel_t, kernel_h, kernel_w = patch_size
        token_dim = in_channels * kernel_t * kernel_h * kernel_w

        self.token_proj = nn.Linear(token_dim, d_model)
        self.token_dropout = nn.Dropout(dropout)

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
        self.pooler = AttentionPooler(d_model, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        inputs: Union[torch.Tensor, SparseTensor4D],
        *,
        return_sparse: bool = False,
    ) -> Dict[str, Union[torch.Tensor, SparseTensor4D]]:
        if isinstance(inputs, SparseTensor4D):
            sparse = inputs
        else:
            sparse = self.patchifier(inputs)

        tokens = sparse.tokens
        positions = sparse.positions
        mask = sparse.mask

        x = self.token_proj(tokens)
        x = self.token_dropout(x)

        for block in self.blocks:
            x = block(x, positions=positions, mask=mask)

        x = self.norm(x)
        pooled = self.pooler(x, mask=mask)

        outputs: Dict[str, Union[torch.Tensor, SparseTensor4D]] = {
            "sequence": x,
            "pooled": pooled,
        }

        if return_sparse:
            outputs["sparse"] = SparseTensor4D(
                tokens=x,
                positions=positions,
                mask=mask,
                metadata=dict(sparse.metadata),
                weights=sparse.weights,
            )

        return outputs

    def load_siglip2_checkpoint(
        self, _state_dict: Dict[str, Any], *, strict: bool = False
    ) -> None:
        """Placeholder for SigLIP2 weight conversion."""
        raise NotImplementedError(
            "SigLIP2 checkpoint loading will be implemented in a follow-up stage."
        )


__all__ = ["ATokenEncoder"]
