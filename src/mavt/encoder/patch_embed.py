from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mavt.utils.inflate import inflate_conv2d_weight


class SpaceTimePatchEmbed(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=bias,
        )

    @classmethod
    def from_conv2d(
        cls,
        conv2d: nn.Conv2d,
        *,
        temporal_patch_size: int = 2,
    ) -> "SpaceTimePatchEmbed":
        patch = cls(
            embed_dim=conv2d.out_channels,
            patch_size=conv2d.kernel_size[0],
            temporal_patch_size=temporal_patch_size,
            bias=conv2d.bias is not None,
        )
        with torch.no_grad():
            patch.proj.weight.copy_(inflate_conv2d_weight(conv2d.weight, temporal_patch_size))
            if conv2d.bias is not None:
                patch.proj.bias.copy_(conv2d.bias)
        return patch

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward_image(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor (B, C, H, W), got {tuple(x.shape)}.")
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size.")
        x = x.unsqueeze(2)
        if self.temporal_patch_size > 1:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_patch_size - 1, 0))
        tokens = self.proj(x)
        _, _, _, h_blocks, w_blocks = tokens.shape
        return self._flatten(tokens), (h_blocks, w_blocks)

    def forward_video(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if x.ndim != 5:
            raise ValueError(f"Expected video tensor (B, C, T, H, W), got {tuple(x.shape)}.")
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError("Video height and width must be divisible by patch_size.")
        if x.shape[2] % self.temporal_patch_size != 0:
            raise ValueError("Video frames must be divisible by temporal_patch_size.")
        tokens = self.proj(x)
        _, _, t_blocks, h_blocks, w_blocks = tokens.shape
        return self._flatten(tokens), (t_blocks, h_blocks, w_blocks)
