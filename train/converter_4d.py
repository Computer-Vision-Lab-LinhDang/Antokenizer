"""Unified 4D Converter - Module 1+2 implementation.

Converts any modality (image/video/3d) to unified 4D token representation:
    tokens:    (N, 1152)  — SigLIP2 features
    positions: (N, 4)     — (t, x, y, z) coordinates
    freq_raw:  (N, 15)    — [spatial(7), temporal(4), depth(4)]
    freq_embed:(N, 128)   — frequency embeddings

This conversion happens BEFORE batching, on CPU side in dataset.__getitem__.
After conversion, all modalities have the same format and can be packed together.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Unified4DConverter:
    """Convert any modality to unified 4D token set.

    Runs in Dataset.__getitem__ (CPU side) to convert raw data
    to token format before NaViT packing.
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch: int = 2,
        siglip2_model: Optional[torch.nn.Module] = None,
        stf_module: Optional[torch.nn.Module] = None,
    ):
        """Initialize converter.

        Args:
            patch_size: Spatial patch size (default: 16)
            temporal_patch: Temporal patch size τ (default: 2)
            siglip2_model: Pre-loaded SigLIP2 model (frozen)
            stf_module: Space-Time-Frequency transform module
        """
        self.patch_size = patch_size
        self.temporal_patch = temporal_patch

        # SigLIP2 (frozen, shared across all conversions)
        # For now, use simple patch embedding as placeholder
        self.siglip2 = siglip2_model
        self.stf = stf_module

    @torch.no_grad()
    def convert(self, sample: dict) -> dict:
        """Main conversion entry point.

        Input: raw sample dict from any Dataset
        Output: unified 4D token dict

        Args:
            sample: Dict with keys like "image", "video", "triplane", "modality"

        Returns:
            Dict with keys:
                - tokens: (N, 1152)
                - positions: (N, 4)
                - freq_raw: (N, 15)
                - freq_embed: (N, 128)
                - n_tokens: int
                - modality: str
                - caption: str
        """
        modality = sample["modality"]

        if modality == "image":
            return self._convert_image(sample)
        elif modality == "video":
            return self._convert_video(sample)
        elif modality == "3d":
            return self._convert_3d(sample)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _convert_image(self, sample: dict) -> dict:
        """Convert image (3, H, W) → N tokens in 4D space.

        Example: image 256×256, patch_size=16
            N = (256/16)² = 256 tokens
            Each token: pos = (t=0, x=col, y=row, z=0)
        """
        img = sample["image"]  # (3, H, W)
        C, H, W = img.shape
        p = self.patch_size
        n_h, n_w = H // p, W // p
        N = n_h * n_w

        # ── Patch embedding (SigLIP2) ──
        if self.siglip2 is not None:
            features = self._siglip2_embed(img.unsqueeze(0)).squeeze(0)  # (N, 1152)
        else:
            # Simple patch extraction + linear projection
            features = self._simple_patch_embed(img, p, n_h, n_w)

        # ── 4D positions ──
        # Image: t=0, z=0, varying x and y
        positions = self._create_2d_positions(n_h, n_w)  # (N, 4)

        # ── Extract raw patches for frequency analysis ──
        raw_patches = self._extract_patches(img, p, n_h, n_w)  # (N, 3, p, p)

        # ── STF: spatial frequency only (image) ──
        if self.stf is not None:
            freq_embed, freq_raw = self._stf_forward(
                raw_patches.unsqueeze(0),
                temporal_signal=None,
                depth_signal=None
            )
            freq_embed = freq_embed.squeeze(0)  # (N, 128)
            freq_raw = freq_raw.squeeze(0)      # (N, 15)
        else:
            # Placeholder
            freq_embed = torch.zeros(N, 128)
            freq_raw = torch.zeros(N, 15)

        return {
            "tokens": features,
            "positions": positions,
            "freq_raw": freq_raw,
            "freq_embed": freq_embed,
            "n_tokens": N,
            "modality": "image",
            "caption": sample.get("caption", ""),
            "resolution": (H, W),
        }

    def _convert_video(self, sample: dict) -> dict:
        """Convert video (3, T, H, W) → N tokens in 4D space.

        Example: video 16 frames × 256×256, temporal_patch=2, spatial_patch=16
            n_t = 16/2 = 8 temporal chunks
            n_h × n_w = (256/16)² = 256 spatial patches per chunk
            N = 8 × 256 = 2048 tokens

        Each token: pos = (t=chunk_idx, x=col, y=row, z=0)
        """
        video = sample["video"]  # (3, T, H, W)
        C, T, H, W = video.shape
        p = self.patch_size
        τ = self.temporal_patch
        n_t = T // τ
        n_h, n_w = H // p, W // p
        N_spatial = n_h * n_w
        N_total = n_t * N_spatial

        all_features = []
        all_positions = []
        all_raw_patches = []

        for ti in range(n_t):
            # Average τ frames for SigLIP2 (image encoder)
            chunk = video[:, ti*τ:(ti+1)*τ].mean(dim=1)  # (3, H, W)

            # Embed chunk
            if self.siglip2 is not None:
                feat = self._siglip2_embed(chunk.unsqueeze(0)).squeeze(0)  # (N_spatial, 1152)
            else:
                feat = self._simple_patch_embed(chunk, p, n_h, n_w)

            all_features.append(feat)

            # Positions: t = temporal chunk index
            pos = self._create_2d_positions(n_h, n_w)
            pos[:, 0] = float(ti)  # Set t dimension
            all_positions.append(pos)

            # Raw patches (center frame of chunk)
            center_frame = video[:, ti*τ + τ//2]  # (3, H, W)
            rp = self._extract_patches(center_frame, p, n_h, n_w)
            all_raw_patches.append(rp)

        tokens = torch.cat(all_features, dim=0)       # (N_total, 1152)
        positions = torch.cat(all_positions, dim=0)    # (N_total, 4)
        raw_patches = torch.cat(all_raw_patches, dim=0)  # (N_total, 3, p, p)

        # ── STF: spatial + temporal frequency ──
        # For temporal freq, we'd need full temporal signal
        # Simplified version here
        if self.stf is not None:
            freq_embed, freq_raw = self._stf_forward(
                raw_patches.unsqueeze(0),
                temporal_signal=None,  # Would pass full video signal
                depth_signal=None
            )
            freq_embed = freq_embed.squeeze(0)[:N_total]
            freq_raw = freq_raw.squeeze(0)[:N_total]
        else:
            freq_embed = torch.zeros(N_total, 128)
            freq_raw = torch.zeros(N_total, 15)

        return {
            "tokens": tokens,
            "positions": positions,
            "freq_raw": freq_raw,
            "freq_embed": freq_embed,
            "n_tokens": N_total,
            "modality": "video",
            "caption": sample.get("caption", ""),
            "resolution": (T, H, W),
        }

    def _convert_3d(self, sample: dict) -> dict:
        """Convert 3D Triplane (3, 3, S, S) → N tokens in 4D space.

        3 planes mapped to 4D:
            XY plane (top-down):  pos = (0, x, y, 0)
            XZ plane (front):     pos = (0, x, 0, z)
            YZ plane (side):      pos = (0, 0, y, z)

        Example: triplane S=32, patch_size=4
            N_per_plane = (32/4)² = 64 tokens
            N_total = 3 × 64 = 192 tokens
        """
        triplane = sample["triplane"]  # (3, 3, S, S)
        n_planes, C, S, _ = triplane.shape

        # 3D patch size smaller for detail
        p3d = min(self.patch_size, S // 4) if S >= 16 else S // 2
        n_s = S // p3d
        N_per_plane = n_s * n_s
        N_total = 3 * N_per_plane

        all_features = []
        all_positions = []

        # Plane configurations
        plane_configs = [
            # XY plane: t=0, x=col, y=row, z=0
            {"t": 0, "x_dim": 1, "y_dim": 2, "z": 0},
            # XZ plane: t=0, x=col, y=0, z=row
            {"t": 0, "x_dim": 1, "y_dim": 3, "z_dim": 2},
            # YZ plane: t=0, x=0, y=col, z=row
            {"t": 0, "x_dim": 2, "y_dim": 3, "z_dim": 1},
        ]

        for pi in range(3):
            plane = triplane[pi]  # (3, S, S) — RGB image of this plane

            # Embed plane
            if self.siglip2 is not None:
                # May need to resize plane to match SigLIP2 input
                plane_resized = F.interpolate(
                    plane.unsqueeze(0), size=(224, 224), mode='bilinear'
                ) if S < 224 else plane.unsqueeze(0)
                feat = self._siglip2_embed(plane_resized).squeeze(0)
                # Subsample to match actual resolution
                if feat.shape[0] != N_per_plane:
                    feat = feat[:N_per_plane]
            else:
                feat = self._simple_patch_embed(plane, p3d, n_s, n_s)

            all_features.append(feat)

            # 4D positions for this plane
            pos = torch.zeros(N_per_plane, 4)
            pos[:, 0] = 0.0  # t = 0 always for 3D

            # Map grid coordinates to 4D based on plane type
            ga, gb = torch.meshgrid(
                torch.arange(n_s, dtype=torch.float32),
                torch.arange(n_s, dtype=torch.float32),
                indexing='ij'
            )

            if pi == 0:  # XY plane
                pos[:, 1] = gb.flatten()  # x
                pos[:, 2] = ga.flatten()  # y
                pos[:, 3] = 0.0           # z = 0
            elif pi == 1:  # XZ plane
                pos[:, 1] = gb.flatten()  # x
                pos[:, 2] = 0.0           # y = 0
                pos[:, 3] = ga.flatten()  # z
            else:  # YZ plane
                pos[:, 1] = 0.0           # x = 0
                pos[:, 2] = gb.flatten()  # y
                pos[:, 3] = ga.flatten()  # z

            all_positions.append(pos)

        tokens = torch.cat(all_features, dim=0)
        positions = torch.cat(all_positions, dim=0)

        # ── STF: spatial + depth frequency ──
        if self.stf is not None:
            raw_patches = torch.zeros(N_total, 3, p3d, p3d)  # Placeholder
            freq_embed, freq_raw = self._stf_forward(
                raw_patches.unsqueeze(0),
                temporal_signal=None,
                depth_signal=None  # Would extract depth signal from XZ/YZ planes
            )
            freq_embed = freq_embed.squeeze(0)[:N_total]
            freq_raw = freq_raw.squeeze(0)[:N_total]
        else:
            freq_embed = torch.zeros(N_total, 128)
            freq_raw = torch.zeros(N_total, 15)

        return {
            "tokens": tokens,
            "positions": positions,
            "freq_raw": freq_raw,
            "freq_embed": freq_embed,
            "n_tokens": N_total,
            "modality": "3d",
            "caption": sample.get("caption", ""),
            "views": sample.get("views"),
            "cameras": sample.get("cameras"),
        }

    # ── Helper methods ──

    def _siglip2_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get SigLIP2 patch embeddings.

        Args:
            x: (B, 3, H, W) or (1, 3, H, W)

        Returns:
            (N, 1152) features
        """
        # Placeholder - actual implementation would call SigLIP2
        B, C, H, W = x.shape
        p = self.patch_size
        N = (H // p) * (W // p)
        return torch.randn(N, 1152)

    def _simple_patch_embed(
        self, img: torch.Tensor, p: int, n_h: int, n_w: int
    ) -> torch.Tensor:
        """Simple patch embedding via unfold + linear projection.

        Args:
            img: (3, H, W)
            p: patch size
            n_h, n_w: number of patches in height and width

        Returns:
            (N, 1152) features
        """
        # Unfold to patches
        patches = img.unfold(1, p, p).unfold(2, p, p)  # (3, n_h, n_w, p, p)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(n_h * n_w, 3 * p * p)

        # Simple linear projection (placeholder)
        # In real implementation, would use learned projection
        feat_dim = 1152
        proj = torch.randn(3 * p * p, feat_dim)
        features = patches @ proj

        return features

    def _create_2d_positions(self, n_h: int, n_w: int) -> torch.Tensor:
        """Create 4D positions for 2D grid (image or video frame).

        Returns:
            (N, 4) positions with t=0, x=col, y=row, z=0
        """
        gy, gx = torch.meshgrid(
            torch.arange(n_h, dtype=torch.float32),
            torch.arange(n_w, dtype=torch.float32),
            indexing='ij'
        )
        N = n_h * n_w
        positions = torch.zeros(N, 4)
        positions[:, 0] = 0.0          # t = 0
        positions[:, 1] = gx.flatten()  # x
        positions[:, 2] = gy.flatten()  # y
        positions[:, 3] = 0.0          # z = 0
        return positions

    def _extract_patches(
        self, img: torch.Tensor, p: int, n_h: int, n_w: int
    ) -> torch.Tensor:
        """Extract raw patches from image.

        Args:
            img: (3, H, W)
            p: patch size
            n_h, n_w: number of patches

        Returns:
            (N, 3, p, p) patches
        """
        patches = img.unfold(1, p, p).unfold(2, p, p)  # (3, n_h, n_w, p, p)
        patches = patches.permute(1, 2, 0, 3, 4).reshape(n_h * n_w, 3, p, p)
        return patches

    def _stf_forward(
        self,
        patches: torch.Tensor,
        temporal_signal: Optional[torch.Tensor],
        depth_signal: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Space-Time-Frequency transform.

        Args:
            patches: (B, N, C, p, p)
            temporal_signal: Optional temporal signal for STFT
            depth_signal: Optional depth signal for STFT

        Returns:
            freq_embed: (B, N, 128)
            freq_raw: (B, N, 15)
        """
        # Placeholder - actual implementation would call STF module
        B, N = patches.shape[0], patches.shape[1]
        freq_embed = torch.zeros(B, N, 128)
        freq_raw = torch.zeros(B, N, 15)
        return freq_embed, freq_raw


__all__ = ["Unified4DConverter"]
