"""Asymmetric Decoder for DDT.

Video/3D decode via batch scatter:
  1. Self-attention refine all tokens (content + dynamics)
  2. Build (B, n_frames, N_spatial, D) grid initialized from content
  3. Scatter ALL dynamics tokens at once via advanced indexing
  4. Batch CNN decode: (B*n_frames, N_sp, D) → (B*n_frames, 3, H, W)

No per-frame loop. One allocation, one scatter, one CNN call.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import DecoderConfig
from .types import DecoderOutput, LatentOutput, Modality


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.ps(self.conv(x))))


class AsymmetricDecoder(nn.Module):

    def __init__(self, cfg: Optional[DecoderConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DecoderConfig()

        self.input_proj = nn.Linear(self.cfg.latent_dim, self.cfg.d_model)
        self.input_norm = nn.LayerNorm(self.cfg.d_model)

        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(self.cfg.d_model, self.cfg.n_attn_heads, batch_first=True)
            for _ in range(self.cfg.n_attn_blocks)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(self.cfg.d_model)
            for _ in range(self.cfg.n_attn_blocks)
        ])
        self.norm_out = nn.LayerNorm(self.cfg.d_model)

        cnn_layers = []
        in_ch = self.cfg.d_model
        for out_ch in self.cfg.cnn_channels:
            cnn_layers.append(PixelShuffleUpsample(in_ch, out_ch))
            in_ch = out_ch
        self.cnn_up = nn.Sequential(*cnn_layers)
        self.output_head = nn.Conv2d(in_ch, self.cfg.out_channels, kernel_size=1)

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

    # ── Shared ──

    def _refine(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(self.input_proj(z))
        for attn, norm in zip(self.attn_blocks, self.attn_norms):
            res = x
            out, _ = attn(norm(x), norm(x), norm(x))
            x = res + out
        return self.norm_out(x)

    def _features_to_pixels(
        self, feat: torch.Tensor, n_h: int, n_w: int,
    ) -> torch.Tensor:
        """(B, N_spatial, d_model) -> (B, 3, H, W)."""
        B = feat.shape[0]
        fm = feat.permute(0, 2, 1).reshape(B, self.cfg.d_model, n_h, n_w)
        fm = self.cnn_up(fm)
        return self.output_head(fm)

    def _batch_scatter_decode(
        self,
        z: torch.Tensor,
        positions: torch.Tensor,
        n_frames: int,
        N_spatial: int,
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        """Batch scatter dynamics into content grid, decode all frames at once.

        Returns: (B, n_frames, C_out, H, W)
        """
        B = z.shape[0]
        device = z.device

        x = self._refine(z)                              # (B, N', D)
        content_feat = x[:, :N_spatial]                   # (B, N_sp, D)

        # Single allocation: all frames start from content
        grid = content_feat.unsqueeze(1).expand(B, n_frames, N_spatial, -1).contiguous()

        # Batch scatter all dynamics tokens at once
        n_dyn = x.shape[1] - N_spatial
        if n_dyn > 0:
            dyn_feat = x[:, N_spatial:]                   # (B, n_dyn, D)
            dyn_pos = positions[:, N_spatial:]             # (B, n_dyn, 4)

            t_idx = dyn_pos[:, :, 0].long().clamp(0, n_frames - 1)
            sp_idx = self._spatial_index(dyn_pos, n_w)    # (B, n_dyn)

            b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(t_idx)
            grid[b_idx, t_idx, sp_idx] = dyn_feat

        # Batch CNN: (B*n_frames, N_sp, D) → (B*n_frames, C_out, H, W)
        frames = self._features_to_pixels(
            grid.reshape(B * n_frames, N_spatial, -1), n_h, n_w,
        )
        C_out, H_out, W_out = frames.shape[1], frames.shape[2], frames.shape[3]
        return frames.reshape(B, n_frames, C_out, H_out, W_out)

    @staticmethod
    def _spatial_index(positions: torch.Tensor, n_w: int) -> torch.Tensor:
        """Compute intra-plane spatial index from 4D positions.

        Video (all planes share x,y grid):
            spatial_idx = y * n_w + x

        3D triplane (each plane uses different axes):
            XY (t=0): row=y, col=x  → y * n_w + x
            XZ (t=1): row=z, col=x  → z * n_w + x
            YZ (t=2): row=z, col=y  → z * n_w + y
        """
        t = positions[:, :, 0].long()
        x = positions[:, :, 1]
        y = positions[:, :, 2]
        z = positions[:, :, 3]

        # Default: video convention (row=y, col=x)
        row = y
        col = x

        # 3D planes: XZ uses (z, x), YZ uses (z, y)
        is_xz = (t == 1) & (z.abs() > 0.1)   # XZ plane: z varies, y=0
        is_yz = (t == 2) & (z.abs() > 0.1)   # YZ plane: z varies, x=0

        row = torch.where(is_xz | is_yz, z, row)
        col = torch.where(is_yz, y, col)

        return (row * n_w + col).long().clamp(0, n_w * n_w - 1)

    # ── Image ──

    def _decode_image(
        self, latent_out: LatentOutput, positions: torch.Tensor,
        target_shape: tuple[int, ...],
    ) -> DecoderOutput:
        p = self.cfg.patch_size
        H, W = target_shape[-2], target_shape[-1]
        n_h, n_w = H // p, W // p
        x = self._refine(latent_out.z)
        recon = self._features_to_pixels(x, n_h, n_w)
        return DecoderOutput(reconstruction=recon)

    # ── Video ──

    def _decode_video(
        self, latent_out: LatentOutput, positions: torch.Tensor,
        target_shape: tuple[int, ...],
    ) -> DecoderOutput:
        p = self.cfg.patch_size
        T, H, W = target_shape[-3], target_shape[-2], target_shape[-1]
        n_h, n_w = H // p, W // p
        N_spatial = n_h * n_w
        n_t = T // 2

        result = self._batch_scatter_decode(
            latent_out.z, positions, n_t, N_spatial, n_h, n_w,
        )
        # (B, n_t, C, H, W) → (B, C, n_t, H, W)
        return DecoderOutput(reconstruction=result.permute(0, 2, 1, 3, 4))

    # ── 3D triplane ──

    def _decode_3d(
        self, latent_out: LatentOutput, positions: torch.Tensor,
        target_shape: tuple[int, ...],
    ) -> DecoderOutput:
        p = self.cfg.patch_size
        S = target_shape[-1] if len(target_shape) == 5 else 64
        n_s = S // p
        N_per_plane = n_s * n_s

        result = self._batch_scatter_decode(
            latent_out.z, positions, 3, N_per_plane, n_s, n_s,
        )
        # result: (B, 3, C_out, S, S) — XY, XZ, YZ planes
        # Return XY plane as primary reconstruction
        return DecoderOutput(
            reconstruction=result[:, 0],
            aux={"all_planes": result},
        )


__all__ = ["AsymmetricDecoder"]
