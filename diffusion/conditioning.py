"""Conditioning modules for diffusion models.

Provides various conditioning mechanisms:
- CrossAttentionConditioner: Attend to external context (zC, zA, y_features)
- FiLMConditioner: Feature-wise Linear Modulation for lightweight conditioning
- AdaLayerNorm: Adaptive LayerNorm with learned scale/shift
- TimestepEmbedding: Sinusoidal + MLP timestep encoding
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for timesteps.

    Similar to transformer positional encoding but applied to
    scalar timestep values.
    """

    def __init__(self, dim: int, max_timesteps: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_timesteps = max_timesteps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timestep indices (B,) or scalar.

        Returns:
            Embeddings (B, dim).
        """
        device = t.device
        half_dim = self.dim // 2

        embeddings = math.log(self.max_timesteps) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(-1).float() * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))

        return embeddings


class TimestepEmbedding(nn.Module):
    """Timestep embedding with sinusoidal encoding + MLP projection.

    Projects timestep to a conditioning vector that can modulate
    the denoiser network via FiLM or AdaLN.
    """

    def __init__(
        self,
        dim: int,
        max_timesteps: int = 1000,
        hidden_mult: int = 4,
    ) -> None:
        """Initialize timestep embedding.

        Args:
            dim: Output embedding dimension.
            max_timesteps: Maximum timestep value.
            hidden_mult: Hidden layer multiplier.
        """
        super().__init__()

        self.sinusoidal = SinusoidalPositionEmbedding(dim, max_timesteps)
        hidden_dim = dim * hidden_mult

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timestep indices (B,).

        Returns:
            Conditioning embeddings (B, dim).
        """
        emb = self.sinusoidal(t)
        return self.mlp(emb)


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning.

    Applies affine transformation: gamma * x + beta
    where gamma and beta are derived from conditioning signal.

    Efficient for scalar or low-dimensional conditions like
    timestep or degradation parameters.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """Initialize FiLM conditioner.

        Args:
            dim: Feature dimension to modulate.
            cond_dim: Conditioning input dimension.
            hidden_dim: Optional hidden layer dimension.
        """
        super().__init__()

        hidden_dim = hidden_dim or cond_dim

        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gamma_proj = nn.Linear(hidden_dim, dim)
        self.beta_proj = nn.Linear(hidden_dim, dim)

        # Initialize gamma to 1, beta to 0
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Input features (B, ..., dim).
            cond: Conditioning signal (B, cond_dim).

        Returns:
            Modulated features (B, ..., dim).
        """
        h = self.net(cond)
        gamma = self.gamma_proj(h)  # (B, dim)
        beta = self.beta_proj(h)  # (B, dim)

        # Expand for broadcasting
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x + beta


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization.

    Applies LayerNorm then modulates with learned scale and shift
    derived from conditioning signal. More stable than FiLM for
    deep networks.

    y = (1 + scale) * LayerNorm(x) + shift
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
    ) -> None:
        """Initialize AdaLN.

        Args:
            dim: Feature dimension.
            cond_dim: Conditioning dimension.
            eps: LayerNorm epsilon.
        """
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Project condition to scale and shift
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2),
        )

        # Initialize to identity transform
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input features (B, N, dim) or (B, dim).
            cond: Conditioning signal (B, cond_dim).

        Returns:
            Normalized and modulated features.
        """
        x = self.norm(x)

        # Get scale and shift from condition
        cond_out = self.cond_proj(cond)  # (B, dim * 2)
        scale, shift = cond_out.chunk(2, dim=-1)  # Each (B, dim)

        # Expand for sequence dimension if needed
        if x.dim() == 3 and scale.dim() == 2:
            scale = scale.unsqueeze(1)  # (B, 1, dim)
            shift = shift.unsqueeze(1)

        return x * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    """AdaLN-Zero: AdaLN with zero-initialized modulation.

    Used in DiT (Diffusion Transformers) for conditioning.
    Outputs scale, shift, and gate for both attention and MLP.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # 6 outputs: scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 6),
        )

        # Zero initialization for residual learning
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get modulation parameters.

        Args:
            x: Input for shape reference (B, N, dim).
            cond: Conditioning signal (B, cond_dim).

        Returns:
            Tuple of (scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp).
        """
        cond_out = self.cond_proj(cond)  # (B, dim * 6)
        scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = cond_out.chunk(6, dim=-1)

        # Expand for sequence dimension
        if x.dim() == 3:
            scale_attn = scale_attn.unsqueeze(1)
            shift_attn = shift_attn.unsqueeze(1)
            gate_attn = gate_attn.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
            shift_mlp = shift_mlp.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)

        return scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp

    def modulate(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Apply modulation to normalized input."""
        return self.norm(x) * (1 + scale) + shift


class CrossAttentionConditioner(nn.Module):
    """Cross-attention for conditioning on external context.

    Allows the model to attend to conditioning sequences like
    semantic tokens (zC), artifact tokens (zA), or image features.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        """Initialize cross-attention.

        Args:
            dim: Query/output dimension.
            num_heads: Number of attention heads.
            context_dim: Context (key/value) dimension. Defaults to dim.
            dropout: Attention dropout.
            qkv_bias: Whether to use bias in projections.
        """
        super().__init__()

        context_dim = context_dim or dim

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            x: Query tokens (B, N, dim).
            context: Context tokens (B, M, context_dim).
            context_mask: Optional mask for context (B, M).

        Returns:
            Attended features (B, N, dim).
        """
        B, N, C = x.shape
        M = context.size(1)

        # Compute Q, K, V
        q = self.to_q(x)  # (B, N, dim)
        kv = self.to_kv(context)  # (B, M, dim * 2)
        k, v = kv.chunk(2, dim=-1)  # Each (B, M, dim)

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, D)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, D)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, M)

        # Apply context mask
        if context_mask is not None:
            context_mask = context_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, M)
            attn = attn.masked_fill(~context_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class GatedCrossAttention(nn.Module):
    """Gated cross-attention with learnable gate.

    output = x + gate * cross_attention(x, context)

    Allows the model to learn how much to attend to context.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.cross_attn = CrossAttentionConditioner(
            dim=dim,
            num_heads=num_heads,
            context_dim=context_dim,
            dropout=dropout,
        )

        # Learnable gate, initialized to small value
        self.gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply gated cross-attention."""
        attn_out = self.cross_attn(x, context, context_mask)
        return x + torch.tanh(self.gate) * attn_out


__all__ = [
    "SinusoidalPositionEmbedding",
    "TimestepEmbedding",
    "FiLMConditioner",
    "AdaLayerNorm",
    "AdaLayerNormZero",
    "CrossAttentionConditioner",
    "GatedCrossAttention",
]
