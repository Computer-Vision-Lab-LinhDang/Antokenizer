from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiCodebookVQ(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_codebooks: int = 4,
        codebook_size: int = 1024,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if embed_dim % num_codebooks != 0:
            raise ValueError("embed_dim must be divisible by num_codebooks.")
        self.embed_dim = embed_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = embed_dim // num_codebooks
        self.decay = decay
        self.epsilon = epsilon
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.codebook = nn.Parameter(torch.randn(num_codebooks, codebook_size, self.codebook_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer(
            "ema_weight",
            torch.zeros(num_codebooks, codebook_size, self.codebook_dim),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=1e-6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.input_proj(x)
        grouped = projected.view(*projected.shape[:-1], self.num_codebooks, self.codebook_dim)
        grouped = grouped.permute(0, 1, 2, 3).contiguous()
        normalized_inputs = self._normalize(grouped)
        normalized_codebook = self._normalize(self.codebook)
        similarities = torch.einsum("btnc,nkc->btnk", normalized_inputs, normalized_codebook)
        indices = similarities.argmax(dim=-1)
        quantized = []
        for codebook_idx in range(self.num_codebooks):
            quantized.append(self.codebook[codebook_idx][indices[..., codebook_idx]])
        quantized = torch.stack(quantized, dim=2)
        quantized = quantized.reshape(*projected.shape)

        if self.training:
            self._ema_update(grouped.detach(), indices)

        commitment_loss = F.mse_loss(projected.detach(), quantized) + F.mse_loss(projected, quantized.detach())
        quantized = projected + (quantized - projected).detach()
        return {
            "quantized": quantized,
            "indices": indices,
            "projected": projected,
            "commitment_loss": commitment_loss,
        }

    def _ema_update(self, grouped: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
        cluster_size = one_hot.sum(dim=(0, 1))
        dw = torch.einsum("btnk,btnc->nkc", one_hot, grouped)
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
        self.ema_weight.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

        n = self.ema_cluster_size.sum(dim=-1, keepdim=True)
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )
        updated_codebook = self.ema_weight / cluster_size.unsqueeze(-1).clamp_min(self.epsilon)
        self.codebook.data.copy_(updated_codebook)
