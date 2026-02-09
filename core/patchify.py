from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_tensor import SparseTensor4D


def _compute_required_padding(
    length: int, kernel: int, stride: int
) -> Tuple[int, int]:
    """Compute symmetric padding (pre, post) to make unfold cover the sequence.

    We only pad on the trailing side by default (front padding is zero) to keep
    positional indices aligned with the original coordinate system.
    """
    if length <= kernel:
        return 0, kernel - length
    remainder = (length - kernel) % stride
    if remainder == 0:
        return 0, 0
    return 0, stride - remainder


@dataclass
class PatchifyMetadata:
    """Book-keeping information describing the extracted patch grid."""

    num_temporal: int
    num_height: int
    num_width: int
    padding: Dict[str, Tuple[int, int]]
    stride: Tuple[int, int, int]
    kernel: Tuple[int, int, int]
    original_shape: Tuple[int, int, int, int, int]
    temporal_valid_ratio: torch.Tensor


class SpaceTimePatchifier(nn.Module):
    """Extracts spatio-temporal patches and produces a SparseTensor4D.

    Args:
        patch_size: Tuple of (t, h, w) kernel sizes.
        stride: Tuple of strides along (t, h, w). Defaults to non-overlapping patches.
        temporal_center_padding: If True, split temporal padding between front/back;
            otherwise pad only at the tail (default per paper description).
        use_spatial_padding: Whether to pad spatial dims to match kernel coverage.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        stride: Optional[Tuple[int, int, int]] = None,
        temporal_center_padding: bool = False,
        use_spatial_padding: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.temporal_center_padding = temporal_center_padding
        self.use_spatial_padding = use_spatial_padding

        if len(self.patch_size) != 3:
            raise ValueError("patch_size must be a 3-tuple (t, h, w)")
        if len(self.stride) != 3:
            raise ValueError("stride must be a 3-tuple (t, h, w)")
        if any(s <= 0 for s in self.stride):
            raise ValueError("stride values must be positive")
        if any(k <= 0 for k in self.patch_size):
            raise ValueError("patch_size values must be positive")

    def forward(self, x: torch.Tensor) -> SparseTensor4D:
        """Patchify an image/video batch.

        Args:
            x: Tensor of shape (B, C, H, W) for images or (B, C, T, H, W) for video.

        Returns:
            A SparseTensor4D with fields:
                tokens: [B, N, C * t * h * w]
                positions: [B, N, 4] with (t, x, y, z) coordinates
                mask: [B, N] boolean validity mask
        """
        if x.dim() not in (4, 5):
            raise ValueError("Expected input shape (B,C,H,W) or (B,C,T,H,W)")

        if x.dim() == 4:
            # Promote images to single-frame videos.
            x = x.unsqueeze(2)
        b, c, t, h, w = x.shape

        kernel_t, kernel_h, kernel_w = self.patch_size
        stride_t, stride_h, stride_w = self.stride

        pad_t_pre, pad_t_post = _compute_required_padding(t, kernel_t, stride_t)
        pad_h_pre, pad_h_post = (0, 0)
        pad_w_pre, pad_w_post = (0, 0)

        if self.use_spatial_padding:
            pad_h_pre, pad_h_post = _compute_required_padding(h, kernel_h, stride_h)
            pad_w_pre, pad_w_post = _compute_required_padding(w, kernel_w, stride_w)

        # Optionally distribute temporal padding.
        if self.temporal_center_padding:
            pad_t_front = pad_t_pre + pad_t_post // 2
            pad_t_back = pad_t_pre + pad_t_post - pad_t_front
        else:
            pad_t_front = pad_t_pre
            pad_t_back = pad_t_post

        padding = (
            pad_w_pre,
            pad_w_post,
            pad_h_pre,
            pad_h_post,
            pad_t_front,
            pad_t_back,
        )
        if any(padding):
            x = F.pad(x, padding, mode="constant", value=0.0)

        _, _, t_padded, h_padded, w_padded = x.shape

        # Build temporal windows using unfold (sliding blocks).
        temporal_windows = x.unfold(dimension=2, size=kernel_t, step=stride_t)
        # (B, C, N_t, kernel_t, H_p, W_p)
        n_t = temporal_windows.size(2)
        temporal_windows = temporal_windows.permute(0, 2, 3, 1, 4, 5)
        # (B, N_t, kernel_t, C, H_p, W_p)

        # Flatten B * N_t temporal groups for spatial unfolding.
        windows_flat = temporal_windows.reshape(
            b * n_t, kernel_t * c, h_padded, w_padded
        )

        patches = F.unfold(
            windows_flat,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
        )
        # (B * N_t, C * kernel_t * kernel_h * kernel_w, N_xy)
        patches = patches.transpose(1, 2)
        n_xy = patches.size(1)
        token_dim = patches.size(2)
        patches = patches.reshape(b, n_t * n_xy, token_dim)

        # Compute positional indices (t, x, y, z).
        grid_t = torch.arange(n_t, device=x.device) * stride_t
        n_h = (h_padded - kernel_h) // stride_h + 1
        n_w = (w_padded - kernel_w) // stride_w + 1
        grid_x = torch.arange(n_h, device=x.device) * stride_h
        grid_y = torch.arange(n_w, device=x.device) * stride_w
        mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        mesh = torch.stack((mesh_x, mesh_y), dim=-1).reshape(n_xy, 2)

        positions = torch.stack(
            [
                grid_t[:, None].expand(n_t, n_xy),
                mesh[None, :, 0].expand(n_t, n_xy),
                mesh[None, :, 1].expand(n_t, n_xy),
                torch.zeros((n_t, n_xy), device=x.device, dtype=torch.long),
            ],
            dim=-1,
        )
        # Broadcast batch dimension.
        positions = positions.unsqueeze(0).expand(b, n_t, n_xy, 4)
        positions = positions.reshape(b, n_t * n_xy, 4).contiguous()

        # Build validity mask considering original (un-padded) extents.
        temporal_mask = torch.zeros(
            t_padded, device=x.device, dtype=torch.bool
        )
        temporal_mask[pad_t_front : pad_t_front + t] = True
        temporal_mask = temporal_mask.unfold(dimension=0, size=kernel_t, step=stride_t)
        # (N_t, kernel_t)
        temporal_valid = temporal_mask.any(dim=-1)  # (N_t,)
        temporal_ratio = temporal_mask.float().mean(dim=-1)

        spatial_valid_x = (mesh[:, 0] + kernel_h) <= h
        spatial_valid_y = (mesh[:, 1] + kernel_w) <= w
        spatial_valid = spatial_valid_x & spatial_valid_y

        mask = temporal_valid[:, None] & spatial_valid[None, :]
        mask = mask.reshape(1, n_t * n_xy).expand(b, -1).clone()

        metadata = PatchifyMetadata(
            num_temporal=n_t,
            num_height=n_h,
            num_width=n_w,
            padding={
                "t": (pad_t_front, pad_t_back),
                "h": (pad_h_pre, pad_h_post),
                "w": (pad_w_pre, pad_w_post),
            },
            stride=self.stride,
            kernel=self.patch_size,
            original_shape=(b, c, t, h, w),
            temporal_valid_ratio=temporal_ratio,
        )

        return SparseTensor4D(
            tokens=patches,
            positions=positions,
            mask=mask,
            metadata={"patchify": metadata},
        )
