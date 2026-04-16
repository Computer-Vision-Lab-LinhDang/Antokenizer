from __future__ import annotations

import torch
from torch import nn

from mavt.decoder.base import DecoderViT
from mavt.patchify.coords import make_positions_image


class ImageDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        embed_dim: int = 256,
        patch_size: int = 16,
        depth: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.decoder = DecoderViT(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.to_pixels = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def expand_latent_dim(self, new_dim: int) -> None:
        self.decoder.expand_latent_dim(new_dim)

    def _unpatchify(self, patches: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch = patches.shape[0]
        h_blocks = height // self.patch_size
        w_blocks = width // self.patch_size
        patches = patches.view(batch, h_blocks, w_blocks, 3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(batch, 3, height, width)

    def forward(self, latents: torch.Tensor, batch: dict) -> torch.Tensor:
        target = batch["image"]
        height, width = target.shape[-2:]
        positions = make_positions_image(
            target.shape[0],
            height // self.patch_size,
            width // self.patch_size,
            device=target.device,
        )
        decoded = self.decoder(latents, positions)
        pixels = self.to_pixels(decoded)
        return self._unpatchify(pixels, height, width)
