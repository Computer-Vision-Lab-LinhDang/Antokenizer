from __future__ import annotations

import torch


def inflate_conv2d_weight(weight_2d: torch.Tensor, temporal_patch_size: int) -> torch.Tensor:
    if weight_2d.ndim != 4:
        raise ValueError(f"Expected 4D Conv2d weight, got shape {tuple(weight_2d.shape)}.")
    if temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be >= 1.")

    out_channels, in_channels, height, width = weight_2d.shape
    weight_3d = weight_2d.new_zeros(
        out_channels,
        in_channels,
        temporal_patch_size,
        height,
        width,
    )
    weight_3d[:, :, -1] = weight_2d
    return weight_3d
