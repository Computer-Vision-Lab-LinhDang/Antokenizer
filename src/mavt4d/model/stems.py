"""Stems: raw sensor data -> tokens. Camera part of the Plan 4D stem; the LiDAR stem comes with hub tokens."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class CameraStem(nn.Module):
    """Non-overlapping patch embedding plus ``n_registers`` learned register tokens per frame."""

    def __init__(self, dim: int = 768, patch: int = 16, n_registers: int = 4) -> None:
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, patch, stride=patch)
        self.registers = nn.Parameter(torch.randn(n_registers, dim) * dim ** -0.5)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """images (F, 3, H, W) -> patch tokens (F, Hp*Wp, d), registers (F, R, d)."""
        H, W = images.shape[-2:]
        if H % self.patch or W % self.patch:
            raise ValueError(f"image size {H}x{W} is not a multiple of patch {self.patch}")
        tokens = self.proj(images).flatten(2).transpose(1, 2)
        return tokens, self.registers.expand(images.shape[0], -1, -1)
