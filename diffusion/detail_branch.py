"""Detail branch for D3PM discrete diffusion on detail tokens (zD).

Implements discrete diffusion denoising for high-frequency detail tokens,
conditioned on completed semantic tokens (zC), artifact tokens (zA),
and degradation parameters.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from atoken.core.rope4d import apply_rope_4d

from .conditioning import AdaLayerNorm, CrossAttentionConditioner, TimestepEmbedding
from .d3pm import D3PM


class DetailTransformerBlock(nn.Module):
    """Transformer block with AdaLN for timestep conditioning.

    Architecture:
    1. Self-attention with AdaLN modulation
    2. Cross-attention to conditioning context
    3. MLP with AdaLN modulation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ) -> None:
        """Initialize detail transformer block.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            cond_dim: Conditioning dimension (for timestep).
            mlp_ratio: MLP hidden dimension multiplier.
            dropout: Dropout rate.
            attention_dropout: Attention dropout rate.
            context_dim: Cross-attention context dimension.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # AdaLN for self-attention
        self.norm1 = AdaLayerNorm(dim, cond_dim)

        # Self-attention
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # AdaLN for cross-attention
        self.norm2 = AdaLayerNorm(dim, cond_dim)

        # Cross-attention
        self.cross_attn = CrossAttentionConditioner(
            dim=dim,
            num_heads=num_heads,
            context_dim=context_dim or dim,
            dropout=dropout,
        )

        # AdaLN for MLP
        self.norm3 = AdaLayerNorm(dim, cond_dim)

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Learnable gate for residual connections
        self.gate_attn = nn.Parameter(torch.ones(1))
        self.gate_cross = nn.Parameter(torch.ones(1))
        self.gate_mlp = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_emb: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tokens (B, N, dim).
            context: Cross-attention context (B, M, context_dim).
            time_emb: Timestep embedding (B, cond_dim).
            positions: 4D positions for RoPE (B, N, 4).
            mask: Token mask (B, N).
            context_mask: Context mask (B, M).

        Returns:
            Output tokens (B, N, dim).
        """
        # Self-attention with AdaLN
        residual = x
        x = self.norm1(x, time_emb)
        x = self._self_attention(x, positions, mask)
        x = residual + self.gate_attn * x

        # Cross-attention with AdaLN
        residual = x
        x = self.norm2(x, time_emb)
        x = self.cross_attn(x, context, context_mask)
        x = residual + self.gate_cross * x

        # MLP with AdaLN
        residual = x
        x = self.norm3(x, time_emb)
        x = self.mlp(x)
        x = residual + self.gate_mlp * x

        return x

    def _self_attention(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Self-attention with optional 4D RoPE."""
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if positions provided
        if positions is not None:
            q, k = apply_rope_4d(q, k, positions)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            key_mask = ~mask[:, None, None, :]
            attn = attn.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class DetailBranch(nn.Module):
    """Detail branch using D3PM discrete diffusion on zD tokens.

    Denoises detail tokens conditioned on:
    - zC_completed: Completed semantic tokens from semantic branch
    - zA_embeddings: Artifact tokens
    - y_features: Shallow image features
    - degradation_params: Encoded degradation parameters
    """

    def __init__(
        self,
        d_model: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        vocab_size: int = 16384,
        num_timesteps: int = 50,
        transition_type: str = "absorbing",
        schedule_type: str = "cosine",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ) -> None:
        """Initialize detail branch.

        Args:
            d_model: Model dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            vocab_size: Detail token vocabulary size.
            num_timesteps: Number of diffusion steps.
            transition_type: "uniform" or "absorbing".
            schedule_type: Noise schedule type.
            mlp_ratio: MLP hidden dimension multiplier.
            dropout: Dropout rate.
            attention_dropout: Attention dropout rate.
            context_dim: Context dimension.
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps

        # D3PM core
        self.d3pm = D3PM(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            transition_type=transition_type,
            schedule_type=schedule_type,
            loss_type="hybrid",
            parametrization="x0",
        )

        # Token embedding (+1 for [MASK] in absorbing)
        self.full_vocab_size = self.d3pm.full_vocab_size
        self.token_embed = nn.Embedding(self.full_vocab_size, d_model)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(d_model, num_timesteps)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DetailTransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                cond_dim=d_model,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                context_dim=context_dim or d_model,
            )
            for _ in range(depth)
        ])

        # Final normalization and projection
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.head = nn.Linear(d_model, vocab_size)  # Predict original vocab only

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        zD_tokens: torch.LongTensor,
        zC_completed: torch.Tensor,
        zA_embeddings: torch.Tensor,
        y_features: torch.Tensor,
        degradation_params: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            zD_tokens: Ground truth detail tokens (B, N_D).
            zC_completed: Completed semantic embeddings (B, N_C, D).
            zA_embeddings: Artifact embeddings (B, N_A, D).
            y_features: Image features (B, N_y, D).
            degradation_params: Degradation embedding (B, D).
            positions: 4D positions (B, N_D, 4).
            mask: Token mask (B, N_D).

        Returns:
            Dictionary with loss and logits.
        """
        B, N = zD_tokens.shape
        device = zD_tokens.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffusion: corrupt tokens
        zD_t = self.d3pm.q_sample(zD_tokens, t)

        # Get model predictions
        logits = self._denoise(
            zD_t, t, zC_completed, zA_embeddings, y_features,
            degradation_params, positions, mask
        )

        # Compute D3PM losses
        losses = self.d3pm.p_losses(zD_tokens, t, logits, zD_t)

        return {
            "loss": losses["loss"],
            "ce_loss": losses.get("ce_loss", torch.tensor(0.0)),
            "vb_loss": losses.get("vb_loss", torch.tensor(0.0)),
            "logits": logits,
        }

    def _denoise(
        self,
        zD_t: torch.LongTensor,
        t: torch.Tensor,
        zC_completed: torch.Tensor,
        zA_embeddings: torch.Tensor,
        y_features: torch.Tensor,
        degradation_params: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Denoising forward pass.

        Args:
            zD_t: Noisy detail tokens (B, N_D).
            t: Timesteps (B,).
            zC_completed: Semantic embeddings (B, N_C, D).
            zA_embeddings: Artifact embeddings (B, N_A, D).
            y_features: Image features (B, N_y, D).
            degradation_params: Degradation embedding (B, D).
            positions: 4D positions (B, N_D, 4).
            mask: Token mask (B, N_D).

        Returns:
            Logits predicting x_0 (B, N_D, vocab_size).
        """
        # Embed noisy tokens
        x = self.token_embed(zD_t)

        # Timestep embedding
        t_emb = self.time_embed(t)

        # Add degradation params to timestep embedding if provided
        if degradation_params is not None:
            t_emb = t_emb + degradation_params

        # Concatenate all context
        context = torch.cat([zC_completed, zA_embeddings, y_features], dim=1)

        # Forward through transformer
        for block in self.blocks:
            x = block(
                x, context=context, time_emb=t_emb,
                positions=positions, mask=mask
            )

        x = self.norm(x)

        # Predict logits (original vocab only, not [MASK])
        logits = self.head(x)

        return logits

    @torch.no_grad()
    def sample(
        self,
        zC_completed: torch.Tensor,
        zA_embeddings: torch.Tensor,
        y_features: torch.Tensor,
        num_tokens: int,
        degradation_params: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        progress: bool = False,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Generate detail tokens via reverse diffusion.

        Args:
            zC_completed: Completed semantic embeddings (B, N_C, D).
            zA_embeddings: Artifact embeddings (B, N_A, D).
            y_features: Image features (B, N_y, D).
            num_tokens: Number of detail tokens to generate.
            degradation_params: Degradation embedding (B, D).
            positions: 4D positions (B, num_tokens, 4).
            temperature: Sampling temperature.
            progress: Whether to show progress bar.

        Returns:
            tokens: Generated detail tokens (B, num_tokens).
            embeddings: Token embeddings (B, num_tokens, D).
        """
        B = zC_completed.size(0)
        device = zC_completed.device

        # Initialize with [MASK] tokens (for absorbing)
        mask_id = self.d3pm.mask_token_id
        x_t = torch.full((B, num_tokens), mask_id, device=device, dtype=torch.long)

        timesteps = range(self.num_timesteps - 1, -1, -1)
        if progress:
            try:
                from tqdm import tqdm
                timesteps = tqdm(timesteps, desc="D3PM Sampling")
            except ImportError:
                pass

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Get model prediction
            logits = self._denoise(
                x_t, t, zC_completed, zA_embeddings, y_features,
                degradation_params, positions, None
            )

            # Sample x_{t-1}
            if t_val > 0:
                x_t = self.d3pm.p_sample(logits, x_t, t, temperature=temperature)
            else:
                # At t=0, take argmax
                x_t = logits.argmax(dim=-1)

        # Get embeddings
        embeddings = self.token_embed(x_t)

        return x_t, embeddings

    @torch.no_grad()
    def sample_fast(
        self,
        zC_completed: torch.Tensor,
        zA_embeddings: torch.Tensor,
        y_features: torch.Tensor,
        num_tokens: int,
        degradation_params: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        temperature: float = 1.0,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Fast sampling with fewer steps using DDIM-like skipping.

        Args:
            zC_completed: Completed semantic embeddings.
            zA_embeddings: Artifact embeddings.
            y_features: Image features.
            num_tokens: Number of tokens to generate.
            degradation_params: Degradation embedding.
            positions: 4D positions.
            num_steps: Number of sampling steps (< num_timesteps).
            temperature: Sampling temperature.

        Returns:
            tokens: Generated tokens.
            embeddings: Token embeddings.
        """
        B = zC_completed.size(0)
        device = zC_completed.device

        # Compute timestep indices for skipping
        step_indices = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device
        )

        # Initialize
        mask_id = self.d3pm.mask_token_id
        x_t = torch.full((B, num_tokens), mask_id, device=device, dtype=torch.long)

        for i, t_val in enumerate(step_indices):
            t = torch.full((B,), t_val.item(), device=device, dtype=torch.long)

            logits = self._denoise(
                x_t, t, zC_completed, zA_embeddings, y_features,
                degradation_params, positions, None
            )

            if i < len(step_indices) - 1:
                # Sample using model prediction
                probs = F.softmax(logits / temperature, dim=-1)
                x_pred = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                x_pred = x_pred.view(B, num_tokens)

                # Re-noise to next timestep
                t_next = step_indices[i + 1]
                if t_next > 0:
                    x_t = self.d3pm.q_sample(x_pred, t_next.expand(B))
                else:
                    x_t = x_pred
            else:
                x_t = logits.argmax(dim=-1)

        embeddings = self.token_embed(x_t)

        return x_t, embeddings


__all__ = ["DetailBranch", "DetailTransformerBlock"]
