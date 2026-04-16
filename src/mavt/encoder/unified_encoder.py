from __future__ import annotations

import torch
from torch import nn

from mavt.encoder.patch_embed import SpaceTimePatchEmbed
from mavt.encoder.siglip2_backbone import SigLIP2Backbone
from mavt.patchify.image import ImagePatchifier
from mavt.patchify.video import VideoPatchifier
from mavt.patchify.coords import MODALITY_TO_ID


class UnifiedEncoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.patch_embed = SpaceTimePatchEmbed(
            embed_dim=embed_dim,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        self.image_patchifier = ImagePatchifier(self.patch_embed)
        self.video_patchifier = VideoPatchifier(self.patch_embed)
        self.backbone = SigLIP2Backbone(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_path=checkpoint_path,
        )

    def _extract_modality(self, batch: dict) -> str:
        modality = batch["modality"]
        if isinstance(modality, (list, tuple)):
            return modality[0]
        return modality

    def build_attention_mask(self, positions: torch.Tensor, modality: str) -> torch.Tensor:
        batch, tokens, _ = positions.shape
        if modality != "video":
            return torch.ones(batch, tokens, tokens, device=positions.device, dtype=torch.bool)
        query_time = positions[:, :, 0].unsqueeze(-1)
        key_time = positions[:, :, 0].unsqueeze(1)
        return key_time <= query_time

    def forward(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, torch.Tensor]:
        modality = self._extract_modality(batch)
        if modality == "image":
            inputs = batch["image"]
            tokens, positions = self.image_patchifier(inputs)
        elif modality == "video":
            inputs = batch["video"]
            tokens, positions = self.video_patchifier(inputs)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        attn_mask = self.build_attention_mask(positions, modality)
        encoded = self.backbone(tokens, positions, attn_mask=attn_mask)
        modality_ids = torch.full(
            (encoded.shape[0], encoded.shape[1]),
            MODALITY_TO_ID[modality],
            device=encoded.device,
            dtype=torch.long,
        )
        return {
            "tokens": encoded,
            "positions": positions,
            "attn_mask": attn_mask,
            "modality_ids": modality_ids,
        }
