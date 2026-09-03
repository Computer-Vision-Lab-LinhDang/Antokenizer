"""Unified 2D Position Embedding for (p, k) Coordinate System.

MAVT v2: Unified coordinate system across modalities.

(p, k) coordinate system:
- p = spatial patch index (x, y) [2D spatial position]
- k = temporal/scale/view position [1D axis]

This replaces:
- 4D (t, x, y, z) for RGAT in v1
- Enables simpler 2D attention with RPB
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unified2DPositionEmbedding(nn.Module):
    """Unified 2D position embedding for (p, k) coordinates.

    The (p, k) system represents:
    - p: 2D spatial position (x, y) → represented as 1D index
    - k: temporal/scale/view axis

    This is used with 2D relative position bias (RPB) instead of
    the full 4D typed edges in RGAT.
    """

    def __init__(
        self,
        dim: int,
        max_p: int = 4096,  # max spatial positions
        max_k: int = 256,   # max temporal/scale/view positions
    ):
        super().__init__()
        self.dim = dim
        self.max_p = max_p
        self.max_k = max_k

        # Separate embeddings for p and k
        self.embed_p = nn.Embedding(max_p, dim // 2)
        self.embed_k = nn.Embedding(max_k, dim // 2)

    def forward(
        self,
        p: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            p: (N,) or (B, N) spatial position indices
            k: (K,) or (B, K) temporal/scale/view indices

        Returns:
            pos_emb: (B, N, K, D) or (N, K, D) position embeddings
        """
        # Handle different shapes
        if len(p.shape) == 1:
            # (N,) -> unsqueeze for broadcasting
            p_emb = self.embed_p(p)  # (N, D//2)
            k_emb = self.embed_k(k)  # (K, D//2)
            # Combine: (N, D//2) + (K, D//2) -> (N, K, D)
            pos_emb = torch.cat([
                p_emb.unsqueeze(1).expand(-1, k.shape[0], -1),
                k_emb.unsqueeze(0).expand(p.shape[0], -1, -1),
            ], dim=-1)  # (N, K, D)
        else:
            # (B, N)
            B, N = p.shape
            p_emb = self.embed_p(p)  # (B, N, D//2)
            k_emb = self.embed_k(k)  # (K, D//2)
            # Combine: (B, N, D//2) + (K, D//2) -> (B, N, K, D)
            pos_emb = torch.cat([
                p_emb.unsqueeze(2).expand(-1, -1, k.shape[0], -1),
                k_emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
            ], dim=-1)  # (B, N, K, D)

        return pos_emb


class MultiScalePositionEmbedding(nn.Module):
    """Position embedding for multi-scale (image pyramid) features.

    For scale pyramid, positions are:
    - p: spatial position (same across scales)
    - k: scale level (0, 1, 2, ...)

    This handles the case where different scales have different
    effective spatial positions.
    """

    def __init__(
        self,
        dim: int,
        max_h: int = 64,
        max_w: int = 64,
        max_scales: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.max_scales = max_scales

        # 2D spatial embedding
        self.embed_h = nn.Embedding(max_h, dim // 4)
        self.embed_w = nn.Embedding(max_w, dim // 4)
        # Scale embedding
        self.embed_scale = nn.Embedding(max_scales, dim // 2)

    def forward(
        self,
        h_indices: torch.Tensor,
        w_indices: torch.Tensor,
        scale_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_indices: (N,) or (B, N) row indices
            w_indices: (N,) or (B, N) column indices
            scale_indices: (K,) or (B, K) scale level indices

        Returns:
            pos_emb: (B, N, K, D) position embeddings
        """
        # Clamp indices
        h = h_indices.clamp(0, self.max_h - 1)
        w = w_indices.clamp(0, self.max_w - 1)
        s = scale_indices.clamp(0, self.max_scales - 1)

        if len(h.shape) == 1:
            # (N,), (N,), (K,)
            h_emb = self.embed_h(h)  # (N, D//4)
            w_emb = self.embed_w(w)  # (N, D//4)
            s_emb = self.embed_scale(s)  # (K, D//2)

            # Combine spatial: (N, D//2)
            spatial = torch.cat([h_emb, w_emb], dim=-1)
            # Expand to (N, K, D//2)
            spatial = spatial.unsqueeze(1).expand(-1, s.shape[0], -1)
            # Expand scale: (N, K, D//2)
            scale = s_emb.unsqueeze(0).expand(h.shape[0], -1, -1)

            pos_emb = torch.cat([spatial, scale], dim=-1)  # (N, K, D)
        else:
            # (B, N), (B, N), (K,) or (B, K)
            B, N = h.shape
            h_emb = self.embed_h(h)  # (B, N, D//4)
            w_emb = self.embed_w(w)  # (B, N, D//4)

            if len(s.shape) == 1:
                s_emb = self.embed_scale(s)  # (K, D//2)
                spatial = torch.cat([h_emb, w_emb], dim=-1)  # (B, N, D//2)
                spatial = spatial.unsqueeze(2).expand(-1, -1, s.shape[0], -1)  # (B, N, K, D//2)
                scale = s_emb.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # (B, N, K, D//2)
            else:
                # (B, K)
                s_emb = self.embed_scale(s)  # (B, K, D//2)
                spatial = torch.cat([h_emb, w_emb], dim=-1)  # (B, N, D//2)
                spatial = spatial.unsqueeze(2).expand(-1, -1, s.shape[1], -1)  # (B, N, K, D//2)
                scale = s_emb.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, K, D//2)

            pos_emb = torch.cat([spatial, scale], dim=-1)  # (B, N, K, D)

        return pos_emb


# ============================================================================
# Utilities for (p, k) position construction
# ============================================================================

def construct_image_positions(
    H: int,
    W: int,
    num_scales: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct (p, k) positions for image with scale pyramid.

    Args:
        H: image height (patches)
        W: image width (patches)
        num_scales: number of scale levels
        device: torch device

    Returns:
        p: (N,) spatial indices (1D flattened)
        k: (K,) scale indices
    """
    N = H * W
    p = torch.arange(N, device=device)  # (N,)
    k = torch.arange(num_scales, device=device)  # (K,)
    return p, k


def construct_video_positions(
    T: int,
    H: int,
    W: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct (p, k) positions for video.

    For video, k = temporal frame index.

    Args:
        T: number of frames
        H: height (patches)
        W: width (patches)
        device: torch device

    Returns:
        p: (T*H*W,) spatial indices (1D flattened)
        k: (T,) frame indices
    """
    N_spatial = H * W
    N = T * N_spatial

    # p: spatial index (same for all frames)
    p_spatial = torch.arange(N_spatial, device=device)  # (N_spatial,)
    p = p_spatial.unsqueeze(0).expand(T, -1).reshape(-1)  # (T*N_spatial,)

    # k: frame index
    k = torch.arange(T, device=device).unsqueeze(1).expand(-1, N_spatial).reshape(-1)  # (T*N_spatial,)

    return p, k


def construct_threed_positions(
    S: int,
    num_views: int = 3,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct (p, k) positions for 3D triplane.

    For 3D, k = view index (0=XY, 1=XZ, 2=YZ).

    Args:
        S: plane size (patches)
        num_views: number of planes (default 3)
        device: torch device

    Returns:
        p: (N,) spatial indices
        k: (K,) view indices
    """
    N_plane = S * S
    N = num_views * N_plane

    p_spatial = torch.arange(N_plane, device=device)  # (N_plane,)
    p = p_spatial.unsqueeze(0).expand(num_views, -1).reshape(-1)  # (N,)

    k = torch.arange(num_views, device=device).unsqueeze(1).expand(-1, N_plane).reshape(-1)  # (N,)

    return p, k


# ============================================================================
# Position embedding for 4D (t, x, y, z) compatibility
# ============================================================================

class FourDPositionEmbedding(nn.Module):
    """Legacy 4D position embedding (kept for compatibility with v1).

    MAVT v2 uses Unified2DPositionEmbedding instead.
    """

    def __init__(
        self,
        dim: int,
        max_t: int = 16,
        max_x: int = 64,
        max_y: int = 64,
        max_z: int = 64,
    ):
        super().__init__()
        self.embed_t = nn.Embedding(max_t, dim // 4)
        self.embed_x = nn.Embedding(max_x, dim // 4)
        self.embed_y = nn.Embedding(max_y, dim // 4)
        self.embed_z = nn.Embedding(max_z, dim // 4)
        self.proj = nn.Linear(dim, dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (N, 4) or (B, N, 4)"""
        t = positions[..., 0].clamp(0, self.embed_t.num_embeddings - 1)
        x = positions[..., 1].clamp(0, self.embed_x.num_embeddings - 1)
        y = positions[..., 2].clamp(0, self.embed_y.num_embeddings - 1)
        z = positions[..., 3].clamp(0, self.embed_z.num_embeddings - 1)

        pe = torch.cat([
            self.embed_t(t),
            self.embed_x(x),
            self.embed_y(y),
            self.embed_z(z),
        ], dim=-1)

        return self.proj(pe)
