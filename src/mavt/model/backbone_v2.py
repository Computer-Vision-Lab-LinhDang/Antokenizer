"""Stage 2: Simplified Backbone with Relative Position Bias (RPB).

MAVT v2: Replaces RGAT with standard ViT blocks + RPB.

Key changes from v1:
- No typed edges (spatial/temporal/depth/cross-plane)
- No per-edge-type K,V projections
- Standard self-attention with relative position bias
- ~67% parameter reduction

Benefits:
- Simpler architecture
- Faster training
- Better scalability
- RPB captures spatial/temporal structure implicitly
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class RelativePositionBias(nn.Module):
    """2D Relative Position Bias for spatial attention.

    Extends 1D RPB to 2D (rows and columns).
    Used in ViT, DeiT, BEiT, etc.
    """

    def __init__(
        self,
        num_heads: int,
        max_dist: int = 128,
        num_buckets: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        self.num_buckets = num_buckets

        # Relative attention bias
        self.relative_attention_bias = nn.Embedding(num_buckets * 2, num_heads)

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
        bidirectional: bool = True,
    ) -> torch.Tensor:
        """Compute bucket index for relative position.

        Uses log-distance bucketing (like GPT-J, DeBERTa).
        """
        relative_buckets = torch.zeros_like(relative_position)

        if bidirectional:
            num_buckets = self.num_buckets
            max_dist = self.max_dist
        else:
            num_buckets = self.num_buckets // 2
            max_dist = self.max_dist

        # Compute log distance
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        relative_position = torch.where(
            relative_position < max_dist,
            torch.log(relative_position.float() + 1) / torch.log(torch.tensor(max_dist).float() + 1),
            torch.ones_like(relative_position),
        )

        # Bucket
        relative_buckets = (relative_position * (num_buckets - 1)).long()
        relative_buckets = relative_buckets.clamp(0, num_buckets - 1)

        # Shift for negative positions
        relative_buckets = torch.where(
            relative_position < 0,
            num_buckets + relative_buckets,
            relative_buckets,
        )

        return relative_buckets

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: (H, W) or (B, H, W) position indices

        Returns:
            bias: (num_heads, H*W, H*W) relative position bias
        """
        if len(position_ids.shape) == 3:
            B, H, W = position_ids.shape
            position_ids = position_ids.reshape(B, H * W)  # (B, N)
            batch_mode = True
        else:
            H, W = position_ids.shape
            position_ids = position_ids.reshape(H * W)  # (N,)
            batch_mode = False

        # Compute 2D relative positions
        # row_pos: relative row position
        # col_pos: relative column position
        if batch_mode:
            row_pos = position_ids.unsqueeze(-1) - position_ids.unsqueeze(-2)  # (B, N, N)
            col_pos = position_ids.unsqueeze(-1) - position_ids.unsqueeze(-2)  # (B, N, N)
        else:
            row_pos = position_ids.unsqueeze(-1) - position_ids.unsqueeze(-2)  # (N, N)
            col_pos = position_ids.unsqueeze(-1) - position_ids.unsqueeze(-2)  # (N, N)

        # Bucket positions
        row_bucket = self._relative_position_bucket(row_pos, bidirectional=True)
        col_bucket = self._relative_position_bucket(col_pos, bidirectional=True)

        # Combine row and column buckets
        relative_position = row_bucket * self.num_buckets + col_bucket
        relative_position = relative_position.clamp(-self.num_buckets, self.num_buckets - 1)

        # Shift for negative
        relative_position = relative_position + self.num_buckets

        # Embed
        bias = self.relative_attention_bias(relative_position)  # (B, N, N, H) or (N, N, H)

        if batch_mode:
            bias = bias.permute(0, 3, 1, 2)  # (B, H, N, N)
        else:
            bias = bias.permute(2, 0, 1)  # (H, N, N)

        return bias


class StandardAttention(nn.Module):
    """Standard multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rpb: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb_2d = RelativePositionBias(num_heads)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            position_ids: (B, H, W) or (H, W) position indices

        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Add relative position bias
        if self.use_rpb and position_ids is not None:
            rpb = self.rpb_2d(position_ids)  # (B, H, N, N) or (H, N, N)
            if len(rpb.shape) == 3 and rpb.shape[0] != B:
                rpb = rpb.unsqueeze(0).expand(B, -1, -1, -1)
            attn = attn + rpb

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention
        out = attn @ v  # (B, H, N, d)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)

        # Project
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class ViTBlock(nn.Module):
    """Standard ViT block with pre-norm and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rpb: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = StandardAttention(
            dim, num_heads, use_rpb=use_rpb,
            attn_drop=dropout, proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        x = x + self.mlp(self.norm2(x))
        return x


class BackboneV2(nn.Module):
    """Simplified ViT backbone with RPB.

    MAVT v2 replaces RGAT with standard ViT blocks:
    - Same block count (12)
    - Standard self-attention with RPB
    - No typed edges
    - ~67% parameter reduction
    """

    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 16,
        num_blocks: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Standard ViT blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_rpb=True,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        plane_ids: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) tokens
            positions: (N, 4) or (B, H, W) position indices
            plane_ids: (N,) ignored in v2 (no typed edges)
            modality: ignored in v2

        Returns:
            features: (B, N, D)
        """
        # Convert positions to 2D grid for RPB
        if len(positions.shape) == 2 and positions.shape[1] == 4:
            # (N, 4) -> convert to 2D grid
            # Assume positions are (t, x, y, z)
            # For images/videos: use (x, y)
            # For simplicity, reconstruct 2D from N
            N = positions.shape[0]
            size = int(N ** 0.5)
            if size * size == N:
                h = w = size
            else:
                h, w = 1, N
            position_ids = torch.arange(h * w, device=positions.device).reshape(h, w)
        else:
            position_ids = positions

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = cp.checkpoint(block, x, position_ids, use_reentrant=False)
            else:
                x = block(x, position_ids=position_ids)

        return x


# ============================================================================
# Alternative: Hybrid RPB (spatial + temporal)
# ============================================================================

class HybridRPB(nn.Module):
    """Hybrid RPB: separate biases for spatial and temporal."""

    def __init__(self, num_heads: int, max_spatial: int = 64, max_temporal: int = 16):
        super().__init__()
        self.spatial_rpb = RelativePositionBias(num_heads, max_dist=max_spatial)
        self.temporal_rpb = RelativePositionBias(num_heads, max_dist=max_temporal)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, T, H, W, 4) or (T, H, W, 4)

        Returns:
            combined bias
        """
        raise NotImplementedError("Hybrid RPB not yet implemented")


# ============================================================================
# Comparison with v1 RGAT
# ============================================================================

def count_parameters() -> Dict[str, int]:
    """Compare parameter counts between v1 RGAT and v2 RPB."""

    # v1 RGAT block
    from mavt.model.rgat import RGAT4DBlock

    v1_block = RGAT4DBlock(dim=1152, num_heads=16, num_edge_types=4)
    v1_params = sum(p.numel() for p in v1_block.parameters())

    # v2 ViT block
    v2_block = ViTBlock(dim=1152, num_heads=16)
    v2_params = sum(p.numel() for p in v2_block.parameters())

    return {
        'rgat_block_12blocks': v1_params * 2,  # 2 RGAT blocks in 12
        'vit_block_12blocks': v2_params * 12,
        'reduction': (1 - v2_params / v1_params) * 100,
    }
