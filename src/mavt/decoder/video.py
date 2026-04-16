from __future__ import annotations

import torch
from torch import nn

from mavt.decoder.base import DecoderViT
from mavt.patchify.coords import make_positions_video


class VideoDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        *,
        embed_dim: int = 256,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        depth: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.decoder = DecoderViT(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.to_pixels = nn.Linear(embed_dim, temporal_patch_size * patch_size * patch_size * 3)

    def expand_latent_dim(self, new_dim: int) -> None:
        self.decoder.expand_latent_dim(new_dim)

    def _unpatchify(self, patches: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
        batch = patches.shape[0]
        t_blocks = frames // self.temporal_patch_size
        h_blocks = height // self.patch_size
        w_blocks = width // self.patch_size
        patches = patches.view(
            batch,
            t_blocks,
            h_blocks,
            w_blocks,
            3,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return patches.view(batch, 3, frames, height, width)

    def forward(self, latents: torch.Tensor, batch: dict) -> torch.Tensor:
        target = batch["video"]
        frames, height, width = target.shape[-3:]
        positions = make_positions_video(
            target.shape[0],
            frames // self.temporal_patch_size,
            height // self.patch_size,
            width // self.patch_size,
            device=target.device,
        )
        query_time = positions[:, :, 0].unsqueeze(-1)
        key_time = positions[:, :, 0].unsqueeze(1)
        mask = key_time <= query_time
        decoded = self.decoder(latents, positions, query_mask=mask)
        pixels = self.to_pixels(decoded)
        return self._unpatchify(pixels, frames, height, width)
