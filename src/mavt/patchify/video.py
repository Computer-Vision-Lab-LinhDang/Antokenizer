from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from mavt.encoder.patch_embed import SpaceTimePatchEmbed
from mavt.patchify.coords import make_positions_video


class VideoPatchifier(nn.Module):
    def __init__(self, patch_embed: SpaceTimePatchEmbed) -> None:
        super().__init__()
        self.patch_embed = patch_embed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, (time_blocks, height_blocks, width_blocks) = self.patch_embed.forward_video(x)
        positions = make_positions_video(
            x.shape[0],
            time_blocks,
            height_blocks,
            width_blocks,
            device=x.device,
        )
        return tokens, positions

    @staticmethod
    def iter_tiles(
        x: torch.Tensor,
        *,
        tile_frames: int = 32,
        stride: int = 16,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        total_frames = x.shape[2]
        for start in range(0, max(total_frames - tile_frames, 0) + 1, stride):
            yield start, x[:, :, start : start + tile_frames]
        if total_frames <= tile_frames:
            return
        if (total_frames - tile_frames) % stride != 0:
            start = total_frames - tile_frames
            yield start, x[:, :, start : start + tile_frames]
