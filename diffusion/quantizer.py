"""Vector quantization modules for discrete token representation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantizer for converting continuous embeddings to discrete tokens.

    Maps continuous vectors to nearest codebook entries, enabling
    discrete diffusion on the resulting tokens.

    Uses straight-through estimator for gradient flow during training.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        init_std: float = 1.0,
    ) -> None:
        """Initialize vector quantizer.

        Args:
            num_embeddings: Codebook size K.
            embedding_dim: Dimension of each codebook vector.
            commitment_cost: Weight for commitment loss.
            init_std: Standard deviation for codebook initialization.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -init_std / num_embeddings,
            init_std / num_embeddings
        )

    def forward(
        self,
        z: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.LongTensor, torch.Tensor, Optional[torch.Tensor]]:
        """Quantize continuous vectors.

        Args:
            z: Input embeddings (B, N, D) or (B, D).
            return_loss: Whether to compute VQ losses.

        Returns:
            tokens: Discrete token indices (B, N) or (B,).
            quantized: Quantized embeddings (same shape as z).
            loss: VQ loss if return_loss=True, else None.
        """
        # Handle both 2D and 3D inputs
        input_shape = z.shape
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, D)

        B, N, D = z.shape
        z_flat = z.reshape(-1, D)  # (B*N, D)

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e.T
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Get nearest codebook entries
        tokens_flat = distances.argmin(dim=-1)  # (B*N,)
        tokens = tokens_flat.view(B, N)

        # Quantize
        quantized_flat = self.embedding(tokens_flat)  # (B*N, D)
        quantized = quantized_flat.view(B, N, D)

        loss = None
        if return_loss:
            # Commitment loss: encourage encoder output to stay close to codebook
            commitment_loss = F.mse_loss(quantized.detach(), z)

            # Codebook loss: encourage codebook to stay close to encoder output
            codebook_loss = F.mse_loss(quantized, z.detach())

            loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Restore original shape
        if len(input_shape) == 2:
            tokens = tokens.squeeze(1)
            quantized = quantized.squeeze(1)

        return tokens, quantized, loss

    def embed(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Convert discrete tokens to embeddings.

        Args:
            tokens: Token indices (B, N) or (B,).

        Returns:
            Embeddings (B, N, D) or (B, D).
        """
        return self.embedding(tokens)

    def get_codebook(self) -> torch.Tensor:
        """Get codebook weights."""
        return self.embedding.weight.data


class EMAVectorQuantizer(nn.Module):
    """Vector Quantizer with Exponential Moving Average codebook updates.

    Uses EMA to update codebook entries instead of gradient descent,
    which is more stable and doesn't require the codebook loss.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        """Initialize EMA vector quantizer.

        Args:
            num_embeddings: Codebook size K.
            embedding_dim: Dimension of each codebook vector.
            commitment_cost: Weight for commitment loss.
            decay: EMA decay factor.
            eps: Small constant for numerical stability.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Codebook (not a parameter, updated via EMA)
        self.register_buffer("embedding", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))

        self.embedding.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.ema_w.data.copy_(self.embedding.data)

    def forward(
        self,
        z: torch.Tensor,
        return_loss: bool = True,
    ) -> Tuple[torch.LongTensor, torch.Tensor, Optional[torch.Tensor]]:
        """Quantize with EMA updates.

        Args:
            z: Input embeddings (B, N, D) or (B, D).
            return_loss: Whether to compute commitment loss.

        Returns:
            tokens: Discrete token indices.
            quantized: Quantized embeddings.
            loss: Commitment loss if return_loss=True.
        """
        input_shape = z.shape
        if z.dim() == 2:
            z = z.unsqueeze(1)

        B, N, D = z.shape
        z_flat = z.reshape(-1, D)

        # Compute distances
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.t())
        )

        # Get nearest entries
        tokens_flat = distances.argmin(dim=-1)
        tokens = tokens_flat.view(B, N)

        # One-hot encoding for EMA update
        encodings = F.one_hot(tokens_flat, self.num_embeddings).float()

        # EMA update (only during training)
        if self.training:
            # Update cluster sizes
            self.cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # Update embedding sums
            dw = torch.matmul(encodings.t(), z_flat)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Normalize embeddings
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            )
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # Quantize
        quantized_flat = self.embedding[tokens_flat]
        quantized = quantized_flat.view(B, N, D)

        loss = None
        if return_loss:
            # Only commitment loss needed with EMA
            loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        if len(input_shape) == 2:
            tokens = tokens.squeeze(1)
            quantized = quantized.squeeze(1)

        return tokens, quantized, loss

    def embed(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Convert tokens to embeddings."""
        return F.embedding(tokens, self.embedding)

    def get_codebook(self) -> torch.Tensor:
        """Get codebook."""
        return self.embedding


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (RVQ) for high-fidelity quantization.

    Applies multiple VQ stages where each stage quantizes the
    residual from previous stages. This allows for higher effective
    codebook size with smaller individual codebooks.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_quantizers: int = 4,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
    ) -> None:
        """Initialize RVQ.

        Args:
            num_embeddings: Codebook size per stage.
            embedding_dim: Embedding dimension.
            num_quantizers: Number of VQ stages.
            commitment_cost: Commitment loss weight.
            use_ema: Whether to use EMA updates.
        """
        super().__init__()

        self.num_quantizers = num_quantizers

        VQClass = EMAVectorQuantizer if use_ema else VectorQuantizer

        self.quantizers = nn.ModuleList([
            VQClass(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])

    def forward(
        self,
        z: torch.Tensor,
        return_loss: bool = True,
        num_stages: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, Optional[torch.Tensor]]:
        """Quantize with residual stages.

        Args:
            z: Input embeddings (B, N, D).
            return_loss: Whether to compute loss.
            num_stages: Number of stages to use (default: all).

        Returns:
            tokens: Token indices (B, N, num_stages).
            quantized: Sum of quantized residuals (B, N, D).
            loss: Sum of VQ losses.
        """
        num_stages = num_stages or self.num_quantizers

        all_tokens = []
        quantized = torch.zeros_like(z)
        residual = z
        total_loss = 0.0

        for i in range(num_stages):
            tokens_i, quantized_i, loss_i = self.quantizers[i](residual, return_loss)
            all_tokens.append(tokens_i)
            quantized = quantized + quantized_i
            residual = residual - quantized_i.detach()

            if loss_i is not None:
                total_loss = total_loss + loss_i

        # Stack tokens: (B, N, num_stages)
        tokens = torch.stack(all_tokens, dim=-1)

        loss = total_loss if return_loss else None

        return tokens, quantized, loss

    def embed(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Convert multi-stage tokens to embeddings.

        Args:
            tokens: Token indices (B, N, num_stages).

        Returns:
            Sum of embeddings (B, N, D).
        """
        quantized = torch.zeros(
            tokens.size(0), tokens.size(1), self.quantizers[0].embedding_dim,
            device=tokens.device, dtype=self.quantizers[0].embedding.dtype
        )

        for i in range(tokens.size(-1)):
            quantized = quantized + self.quantizers[i].embed(tokens[..., i])

        return quantized


__all__ = [
    "VectorQuantizer",
    "EMAVectorQuantizer",
    "ResidualVectorQuantizer",
]
