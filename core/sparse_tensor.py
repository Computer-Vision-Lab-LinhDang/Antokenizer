from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class SparseTensor4D:
    """Container for sparse 4D tokens and their metadata.

    Attributes:
        tokens: Tensor of shape (B, N, D).
        positions: Tensor of shape (B, N, 4) with (t, x, y, z) indices.
        mask: Boolean tensor of shape (B, N) indicating valid tokens.
        weights: Optional float tensor weighting tokens (B, N, 1 or B, N).
        metadata: Arbitrary dictionary with provenance details.
    """

    tokens: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor
    weights: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens.dim() != 3:
            raise ValueError("tokens must be (B, N, D)")
        if self.positions.shape[:2] != self.tokens.shape[:2]:
            raise ValueError("positions must align with tokens batch/length")
        if self.positions.size(-1) != 4:
            raise ValueError("positions must contain 4 axes (t, x, y, z)")
        if self.mask.shape != self.tokens.shape[:2]:
            raise ValueError("mask must be (B, N)")
        if self.weights is not None and self.weights.shape[:2] != self.tokens.shape[:2]:
            raise ValueError("weights must align with tokens batch/length")
        if not self.mask.dtype == torch.bool:
            raise TypeError("mask must be boolean")

    @property
    def batch_size(self) -> int:
        return self.tokens.size(0)

    @property
    def num_tokens(self) -> int:
        return self.tokens.size(1)

    @property
    def token_dim(self) -> int:
        return self.tokens.size(2)

    def clone(self) -> "SparseTensor4D":
        return SparseTensor4D(
            tokens=self.tokens.clone(),
            positions=self.positions.clone(),
            mask=self.mask.clone(),
            weights=None if self.weights is None else self.weights.clone(),
            metadata=self.metadata.copy(),
        )

    def to(self, *args: Any, **kwargs: Any) -> "SparseTensor4D":
        weights = self.weights.to(*args, **kwargs) if self.weights is not None else None
        return SparseTensor4D(
            tokens=self.tokens.to(*args, **kwargs),
            positions=self.positions.to(*args, **kwargs),
            mask=self.mask.to(*args, **kwargs),
            weights=weights,
            metadata=self.metadata,
        )

    def pin_memory(self) -> "SparseTensor4D":
        weights = self.weights.pin_memory() if self.weights is not None else None
        return SparseTensor4D(
            tokens=self.tokens.pin_memory(),
            positions=self.positions.pin_memory(),
            mask=self.mask.pin_memory(),
            weights=weights,
            metadata=self.metadata,
        )

    def masked_tokens(self, fill: float = 0.0) -> torch.Tensor:
        out = self.tokens.clone()
        out[~self.mask] = fill
        return out

    def flatten_valid(self) -> torch.Tensor:
        """Return tokens filtered to valid entries only."""
        mask = self.mask.view(self.batch_size, self.num_tokens)
        return self.tokens[mask]

    def update_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
