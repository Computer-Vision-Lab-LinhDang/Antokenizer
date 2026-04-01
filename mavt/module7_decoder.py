"""Module 7: Asymmetric Decoder.

Attention (global) → CNN upsampling → pixel output.

When *cd_cfg* is provided the decoder instantiates a
:class:`UnifiedDetailExpander` and routes through a unified
expand → spatial-decode path for **all** modalities.
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


class AsymmetricDecoder(nn.Module):
    """Asymmetric decoder: Attention → CNN → pixel head."""

    def __init__(
        self,
        cfg: Optional[DecoderConfig] = None,
        cd_cfg=None,
    ) -> None:
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

        # Stage C: CNN upsampling
        cnn_layers = []
        in_ch = self.cfg.d_model
        for out_ch in self.cfg.cnn_channels:
            cnn_layers.append(PixelShuffleUpsample(in_ch, out_ch))
            in_ch = out_ch
        self.cnn_up = nn.Sequential(*cnn_layers)

        # Stage D: output head
        self.output_head = nn.Conv2d(in_ch, self.cfg.out_channels, kernel_size=1)
        self.norm_out = nn.LayerNorm(self.cfg.d_model)

        # Unified detail expander (optional, for C-D decode)
        self.unified_expander = None
        if cd_cfg is not None:
            from .dynamics_expander import UnifiedDetailExpander

            self.unified_expander = UnifiedDetailExpander(
                d_model=self.cfg.d_model,
                n_heads=self.cfg.n_attn_heads,
                n_layers=cd_cfg.n_expander_layers,
            )

    def forward(
        self,
        latent_out: LatentOutput,
        positions: torch.Tensor,
        target_shape: tuple[int, ...],
        modality: Modality = "image",
        cd_metadata: Optional[dict] = None,
    ) -> DecoderOutput:
        if cd_metadata is not None and self.unified_expander is not None:
            return self._decode_unified_cd(
                latent_out, target_shape, modality, cd_metadata,
            )
        # Fallback: no C-D Split.
        if modality == "image":
            return self._decode_image(latent_out, positions, target_shape)
        if modality == "video":
            return self._decode_video(latent_out, positions, target_shape)
        if modality == "3d":
            return self._decode_3d(latent_out, positions, target_shape)
        raise ValueError(f"Unknown modality: {modality}")

    # ── unified C-D decode (all modalities) ──────────────────────────

    def _decode_unified_cd(self, latent_out, target_shape, modality, cd_metadata):
        """Decode from compressed content+detail tokens (any modality)."""
        # 1. Project + self-attention over compressed tokens.
        z = self.input_norm(self.input_proj(latent_out.z))

        for attn, norm in zip(self.attn_blocks, self.attn_norms):
            res = z
            zn = norm(z)
            out, _ = attn(zn, zn, zn)
            z = res + out
        z = self.norm_out(z)

        # 2. Expand to original grid positions.
        original_positions = cd_metadata["original_positions"]
        z_expanded = self.unified_expander(z, original_positions)

        # 3. Spatial decode (modality determines frame/plane grouping).
        p = self.cfg.patch_size
        H, W = target_shape[-2], target_shape[-1]
        n_h, n_w = H // p, W // p
        N_orig = cd_metadata["n_original"]

        if modality == "image":
            return DecoderOutput(
                reconstruction=self._spatial_decode(z_expanded, n_h, n_w),
            )

        if modality == "video":
            N_sp = n_h * n_w
            n_t = N_orig // N_sp
            frames = [
                self._spatial_decode(z_expanded[:, ti * N_sp:(ti + 1) * N_sp], n_h, n_w)
                for ti in range(n_t)
            ]
            return DecoderOutput(reconstruction=torch.stack(frames, dim=2))

        if modality == "3d":
            N_pp = N_orig // 3
            S = target_shape[-1] if len(target_shape) == 5 else 64
            n_s = S // p
            planes = [
                self._spatial_decode(z_expanded[:, pi * N_pp:(pi + 1) * N_pp], n_s, n_s)
                for pi in range(3)
            ]
            return DecoderOutput(reconstruction=torch.stack(planes, dim=2))

        raise ValueError(f"Unknown modality: {modality}")

    # ── per-modality fallbacks (no C-D Split) ────────────────────────

    def _decode_image(self, latent_out, positions, target_shape):
        B, N, _ = latent_out.z.shape
        p = self.cfg.patch_size
        H, W = target_shape[-2], target_shape[-1]
        n_h, n_w = H // p, W // p

        x = self.input_norm(self.input_proj(latent_out.z))
        for attn, norm in zip(self.attn_blocks, self.attn_norms):
            res = x
            xn = norm(x)
            out, _ = attn(xn, xn, xn)
            x = res + out
        x = self.norm_out(x)
        return DecoderOutput(reconstruction=self._spatial_decode(x, n_h, n_w))

    def _decode_video(self, latent_out, positions, target_shape):
        B, N, _ = latent_out.z.shape
        p = self.cfg.patch_size
        H, W = target_shape[-2], target_shape[-1]
        n_h, n_w = H // p, W // p
        N_spatial = n_h * n_w
        n_t = N // N_spatial

        frames = []
        for ti in range(n_t):
            s, e = ti * N_spatial, (ti + 1) * N_spatial
            lat = LatentOutput(
                z=latent_out.z[:, s:e],
                z_understand=latent_out.z_understand,
                mu=latent_out.mu[:, s:e],
                log_var=latent_out.log_var[:, s:e],
            )
            out = self._decode_image(lat, positions[:, s:e], (latent_out.z.shape[0], 3, H, W))
            frames.append(out.reconstruction)
        return DecoderOutput(reconstruction=torch.stack(frames, dim=2))

    def _decode_3d(self, latent_out, positions, target_shape):
        B = latent_out.z.shape[0]
        N_pp = latent_out.z.shape[1] // 3
        S = target_shape[-1] if len(target_shape) == 5 else 64
        img_shape = (B, 3, S, S)

        planes = []
        for pi in range(3):
            s, e = pi * N_pp, (pi + 1) * N_pp
            lat = LatentOutput(
                z=latent_out.z[:, s:e],
                z_understand=latent_out.z_understand,
                mu=latent_out.mu[:, s:e],
                log_var=latent_out.log_var[:, s:e],
            )
            out = self._decode_image(lat, positions[:, s:e], img_shape)
            planes.append(out.reconstruction)
        return DecoderOutput(reconstruction=torch.stack(planes, dim=2))

    # ── shared CNN decode ────────────────────────────────────────────

    def _spatial_decode(self, spatial_z, n_h, n_w):
        """(B, N_sp, d_model) → (B, 3, H, W) via CNN upsample."""
        B = spatial_z.shape[0]
        x = spatial_z.permute(0, 2, 1).reshape(B, self.cfg.d_model, n_h, n_w)
        x = self.cnn_up(x)
        return self.output_head(x)


__all__ = ["PixelShuffleUpsample", "AsymmetricDecoder"]
