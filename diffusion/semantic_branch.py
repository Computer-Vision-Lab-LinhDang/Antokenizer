"""Semantic branch for masked modeling on semantic tokens (zC).

Implements MaskGIT-style parallel decoding for the semantic branch,
which predicts semantic tokens conditioned on artifact tokens (zA)
and shallow image features (y_features).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from atoken.core.rope4d import apply_rope_4d

from .conditioning import CrossAttentionConditioner


class SemanticTransformerBlock(nn.Module):
    """Transformer block with optional cross-attention for semantic modeling.

    Consists of:
    1. Self-attention with 4D RoPE
    2. Optional cross-attention to conditioning context
    3. Feed-forward network
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        cross_attn: bool = True,
        context_dim: Optional[int] = None,
    ) -> None:
        """Initialize semantic transformer block.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
            dropout: Dropout rate.
            attention_dropout: Attention dropout rate.
            cross_attn: Whether to include cross-attention.
            context_dim: Context dimension for cross-attention.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Cross-attention (optional)
        self.cross_attn = None
        if cross_attn:
            self.norm_cross = nn.LayerNorm(dim, eps=1e-6)
            self.cross_attn = CrossAttentionConditioner(
                dim=dim,
                num_heads=num_heads,
                context_dim=context_dim or dim,
                dropout=dropout,
            )

        # MLP
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tokens (B, N, dim).
            positions: 4D positions (B, N, 4) for RoPE.
            mask: Token validity mask (B, N).
            context: Cross-attention context (B, M, context_dim).
            context_mask: Context mask (B, M).

        Returns:
            Output tokens (B, N, dim).
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self._self_attention(x, positions, mask)
        x = residual + x

        # Cross-attention
        if self.cross_attn is not None and context is not None:
            residual = x
            x = self.norm_cross(x)
            x = residual + self.cross_attn(x, context, context_mask)

        # MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

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
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, B, N, H, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if positions provided
        if positions is not None:
            q, k = apply_rope_4d(q, k, positions)

        # Reshape for attention
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            key_mask = ~mask[:, None, None, :]  # (B, 1, 1, N)
            attn = attn.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class SemanticBranch(nn.Module):
    """Semantic branch using masked language modeling on zC tokens.

    Uses MaskGIT-style parallel decoding:
    1. Start with all [MASK] tokens
    2. Predict all positions simultaneously
    3. Keep top-k confident predictions, re-mask the rest
    4. Repeat until all positions filled

    Conditioned on:
    - zA: Artifact tokens from degraded input
    - y_features: Shallow CNN features from degraded image
    """

    def __init__(
        self,
        d_model: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        vocab_size: int = 8192,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        mask_ratio: float = 0.15,
        context_dim: Optional[int] = None,
    ) -> None:
        """Initialize semantic branch.

        Args:
            d_model: Model dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            vocab_size: Semantic token vocabulary size.
            mlp_ratio: MLP hidden dimension multiplier.
            dropout: Dropout rate.
            attention_dropout: Attention dropout rate.
            mask_ratio: Ratio of tokens to mask during training.
            context_dim: Context dimension for cross-attention.
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.mask_token_id = vocab_size  # [MASK] at index vocab_size

        # Token embedding (+1 for [MASK])
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)

        # Learnable position embedding for masked positions
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SemanticTransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                cross_attn=True,
                context_dim=context_dim or d_model,
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Prediction head
        self.head = nn.Linear(d_model, vocab_size)

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
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        zC_tokens: torch.LongTensor,
        zA_embeddings: torch.Tensor,
        y_features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass with random masking.

        Args:
            zC_tokens: Ground truth semantic tokens (B, N_C).
            zA_embeddings: Artifact token embeddings (B, N_A, D).
            y_features: Shallow image features (B, N_y, D).
            positions: 4D positions for RoPE (B, N_C, 4).
            mask: Token validity mask (B, N_C).

        Returns:
            loss: Cross-entropy loss on masked tokens.
            zC_completed: Predicted embeddings for all positions (B, N_C, D).
        """
        B, N = zC_tokens.shape
        device = zC_tokens.device

        # Random masking
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Sample mask positions
        mask_indices = self._sample_mask_indices(B, N, device)

        # Replace masked positions with [MASK] token
        zC_masked = zC_tokens.clone()
        zC_masked[mask_indices] = self.mask_token_id

        # Embed tokens
        x = self.token_embed(zC_masked)

        # Concatenate context: [zA, y_features]
        context = torch.cat([zA_embeddings, y_features], dim=1)

        # Forward through transformer
        for block in self.blocks:
            x = block(x, positions=positions, mask=mask, context=context)

        x = self.norm(x)

        # Predict logits
        logits = self.head(x)  # (B, N, vocab_size)

        # Compute loss only on masked positions
        masked_logits = logits[mask_indices]  # (num_masked, vocab_size)
        masked_targets = zC_tokens[mask_indices]  # (num_masked,)

        loss = F.cross_entropy(masked_logits, masked_targets)

        return loss, x

    def _sample_mask_indices(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.BoolTensor:
        """Sample random mask positions."""
        num_masked = max(1, int(seq_len * self.mask_ratio))
        noise = torch.rand(batch_size, seq_len, device=device)
        sorted_indices = noise.argsort(dim=1)
        mask_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        mask_indices.scatter_(1, sorted_indices[:, :num_masked], True)
        return mask_indices

    @torch.no_grad()
    def sample(
        self,
        zA_embeddings: torch.Tensor,
        y_features: torch.Tensor,
        num_tokens: int,
        positions: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        temperature: float = 1.0,
        confidence_threshold: float = 0.0,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """MaskGIT-style parallel decoding.

        Args:
            zA_embeddings: Artifact embeddings (B, N_A, D).
            y_features: Image features (B, N_y, D).
            num_tokens: Number of semantic tokens to generate.
            positions: 4D positions (B, num_tokens, 4).
            num_steps: Number of decoding iterations.
            temperature: Sampling temperature.
            confidence_threshold: Minimum confidence to keep prediction.

        Returns:
            tokens: Generated semantic tokens (B, num_tokens).
            embeddings: Token embeddings (B, num_tokens, D).
        """
        B = zA_embeddings.size(0)
        device = zA_embeddings.device

        # Initialize with all [MASK]
        tokens = torch.full((B, num_tokens), self.mask_token_id, device=device)

        context = torch.cat([zA_embeddings, y_features], dim=1)

        # Cosine schedule for number of tokens to unmask
        # More tokens unmasked in later steps
        mask_schedule = self._cosine_mask_schedule(num_steps, num_tokens)

        for step in range(num_steps):
            # Get current mask
            is_masked = (tokens == self.mask_token_id)

            if not is_masked.any():
                break

            # Forward pass
            x = self.token_embed(tokens)

            for block in self.blocks:
                x = block(x, positions=positions, context=context)

            x = self.norm(x)
            logits = self.head(x)  # (B, N, vocab_size)

            # Sample predictions
            probs = F.softmax(logits / temperature, dim=-1)
            pred_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
            pred_tokens = pred_tokens.view(B, num_tokens)

            # Compute confidence
            confidence = probs.max(dim=-1).values  # (B, N)

            # Determine how many tokens to unmask this step
            num_to_unmask = mask_schedule[step]

            # Unmask most confident predictions
            for b in range(B):
                masked_pos = is_masked[b].nonzero().squeeze(-1)
                if len(masked_pos) == 0:
                    continue

                # Get confidence at masked positions
                masked_conf = confidence[b, masked_pos]

                # Number to unmask for this sample
                n_unmask = min(num_to_unmask, len(masked_pos))

                # Get top confident positions
                _, top_indices = masked_conf.topk(n_unmask)
                unmask_pos = masked_pos[top_indices]

                # Only unmask if above threshold
                for pos in unmask_pos:
                    if confidence[b, pos] >= confidence_threshold:
                        tokens[b, pos] = pred_tokens[b, pos]

        # Final embedding
        embeddings = self.token_embed(tokens)

        return tokens, embeddings

    def _cosine_mask_schedule(self, num_steps: int, num_tokens: int) -> list:
        """Cosine schedule for progressive unmasking."""
        schedule = []
        for i in range(num_steps):
            # Cosine schedule: unmask more tokens in later steps
            ratio = math.cos(math.pi / 2 * i / num_steps)
            remaining = int(num_tokens * ratio)
            if i == 0:
                prev_remaining = num_tokens
            to_unmask = prev_remaining - remaining
            schedule.append(max(1, to_unmask))
            prev_remaining = remaining
        return schedule


__all__ = ["SemanticBranch", "SemanticTransformerBlock"]
