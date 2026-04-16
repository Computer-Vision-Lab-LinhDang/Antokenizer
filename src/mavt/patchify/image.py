from __future__ import annotations

import torch
from torch import nn

from mavt.encoder.patch_embed import SpaceTimePatchEmbed
from mavt.patchify.coords import make_positions_image


class ImagePatchifier(nn.Module):
    def __init__(self, patch_embed: SpaceTimePatchEmbed) -> None:
        super().__init__()
        self.patch_embed = patch_embed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, (height_blocks, width_blocks) = self.patch_embed.forward_image(x)
        positions = make_positions_image(
            x.shape[0],
            height_blocks,
            width_blocks,
            device=x.device,
        )
        return tokens, positions
