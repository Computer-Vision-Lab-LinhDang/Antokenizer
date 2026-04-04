"""Module 7: Asymmetric Decoder.

v1: Attention (global) → CNN upsampling → pixel output.
Full design: Attention → Mamba+GNN → CNN → modality heads.

For image: latent tokens (B, N, 32) → (B, 3, H, W)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import DecoderConfig
from .types import DecoderOutput, LatentOutput, Modality


class PixelShuffleUpsample(nn.Module):
    """Sub-pixel conv upsampling (PixelShuffle), scale_factor=2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.ps(self.conv(x))))
    
class FeedForward(nn.Module):
    """Standard Transformer FFN"""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AsymmetricDecoder(nn.Module):
    """Asymmetric decoder: Attention → CNN → pixel head."""

    def __init__(self, cfg: Optional[DecoderConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DecoderConfig()

        self.input_proj = nn.Linear(self.cfg.latent_dim, self.cfg.d_model)
        self.input_norm = nn.LayerNorm(self.cfg.d_model)

        # Stage A: self-attention blocks
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.cfg.d_model,
                num_heads=self.cfg.n_attn_heads,
                batch_first=True,
            )
            for _ in range(self.cfg.n_attn_blocks)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(self.cfg.d_model)
            for _ in range(self.cfg.n_attn_blocks)
        ])
        self.ffn_blocks = nn.ModuleList([
            FeedForward(
                d_model=self.cfg.d_model,
                hidden_dim=self.cfg.d_model * 4,
                dropout=self.cfg.dropout,
            )
            for _ in range(self.cfg.n_attn_blocks)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(self.cfg.d_model)
            for _ in range(self.cfg.n_attn_blocks)
        ])

        # Stage C: CNN upsampling (each PixelShuffle doubles resolution)
        cnn_layers = []
        in_ch = self.cfg.d_model
        for out_ch in self.cfg.cnn_channels:
            cnn_layers.append(PixelShuffleUpsample(in_ch, out_ch))
            in_ch = out_ch
        self.cnn_up = nn.Sequential(*cnn_layers)

        # Stage D: output head
        self.output_head = nn.Conv2d(in_ch, self.cfg.out_channels, kernel_size=1)
        self.norm_out = nn.LayerNorm(self.cfg.d_model)

    def forward(
        self,
        latent_out: LatentOutput,
        positions: torch.Tensor,
        target_shape: tuple[int, ...],
        modality: Modality = "image",
    ) -> DecoderOutput:
        if modality == "image":
            return self._decode_image(latent_out, positions, target_shape)
        if modality == "video":
            return self._decode_video(latent_out, positions, target_shape)
        if modality == "3d":
            return self._decode_3d(latent_out, positions, target_shape)
        raise ValueError(f"Unknown modality: {modality}")

    def _decode_image(
        self,
        latent_out: LatentOutput,
        positions: torch.Tensor,
        target_shape: tuple[int, ...],
    ) -> DecoderOutput:
        """Decode to (B, 3, H, W)."""
        B, N, _ = latent_out.z.shape
        p = self.cfg.patch_size
        H, W = target_shape[-2], target_shape[-1]
        n_h, n_w = H // p, W // p

        x = self.input_norm(self.input_proj(latent_out.z))  # (B, N, d_model)

        # Stage A: self-attention
        for attn, norm, ffn, ffn_norm in zip(
            self.attn_blocks, 
            self.attn_norms,
            self.ffn_blocks,
            self.ffn_norms
        ):
            res = x
            xn = norm(x)
            out, _ = attn(xn, xn, xn)
            x = res + out

            res = x
            xn = ffn_norm(x)
            x = res + ffn(xn)

        x = self.norm_out(x)  # (B, N, d_model)

        # Reshape to spatial feature map
        x = x.permute(0, 2, 1).reshape(B, self.cfg.d_model, n_h, n_w)  # (B, D, n_h, n_w)

        # Stage C: CNN upsample → (B, 64, H, W)
        x = self.cnn_up(x)

        # Stage D: pixel output
        reconstruction = self.output_head(x)  # (B, out_channels, H, W)
        return DecoderOutput(reconstruction=reconstruction)

    def _decode_video(self, latent_out, positions, target_shape) -> DecoderOutput:
        """v1: decode each temporal chunk as an image, stack back."""
        B, N, _ = latent_out.z.shape
        p = self.cfg.patch_size
        T, H, W = target_shape[-3], target_shape[-2], target_shape[-1]
        n_h, n_w = H // p, W // p
        n_t = T // 2  # tau=2
        N_spatial = n_h * n_w

        # Process each temporal group
        frames = []
        for ti in range(n_t):
            z_chunk = latent_out.z[:, ti * N_spatial:(ti + 1) * N_spatial]  # (B, N_sp, D)
            pos_chunk = positions[:, ti * N_spatial:(ti + 1) * N_spatial]
            lat = LatentOutput(
                z=z_chunk,
                z_understand=latent_out.z_understand,
                mu=latent_out.mu[:, ti * N_spatial:(ti + 1) * N_spatial],
                log_var=latent_out.log_var[:, ti * N_spatial:(ti + 1) * N_spatial],
            )
            out = self._decode_image(lat, pos_chunk, (B, 3, H, W))
            frames.append(out.reconstruction)  # (B, 3, H, W)

        video = torch.stack(frames, dim=2)  # (B, 3, n_t, H, W)
        return DecoderOutput(reconstruction=video)

    def _decode_3d(self, latent_out, positions, target_shape) -> DecoderOutput:
        """v1: decode as image (triplane reconstruction placeholder)."""
        if len(target_shape) == 5:
            S = target_shape[-1]
            img_shape = (target_shape[0], 3, S, S)
        else:
            img_shape = (target_shape[0], 3, 64, 64)
        N_per_plane = latent_out.z.shape[1] // 3
        lat = LatentOutput(
            z=latent_out.z[:, :N_per_plane],
            z_understand=latent_out.z_understand,
            mu=latent_out.mu[:, :N_per_plane],
            log_var=latent_out.log_var[:, :N_per_plane],
        )
        return self._decode_image(lat, positions[:, :N_per_plane], img_shape)


__all__ = ["PixelShuffleUpsample", "AsymmetricDecoder"]
