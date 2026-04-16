from __future__ import annotations

import torch
from torch import nn


class DiscreteFSQHead(nn.Module):
    def __init__(self, embed_dim: int, levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)) -> None:
        super().__init__()
        self.levels = levels
        self.proj = nn.Linear(embed_dim, len(levels))

    def _quantize_dim(self, values: torch.Tensor, levels: int) -> tuple[torch.Tensor, torch.Tensor]:
        codebook = torch.linspace(-1.0, 1.0, steps=levels, device=values.device, dtype=values.dtype)
        distances = (values.unsqueeze(-1) - codebook).abs()
        indices = distances.argmin(dim=-1)
        quantized = codebook[indices]
        return quantized, indices

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = torch.tanh(self.proj(x))
        quantized_chunks = []
        index_chunks = []
        for dim, level in enumerate(self.levels):
            quantized, indices = self._quantize_dim(projected[..., dim], level)
            quantized_chunks.append(quantized)
            index_chunks.append(indices)
        quantized = torch.stack(quantized_chunks, dim=-1)
        indices = torch.stack(index_chunks, dim=-1)
        quantized = projected + (quantized - projected).detach()
        return {
            "quantized": quantized,
            "indices": indices,
            "projected": projected,
            "commitment_loss": projected.new_zeros(()),
        }
