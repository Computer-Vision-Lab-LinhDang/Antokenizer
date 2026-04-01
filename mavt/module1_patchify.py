"""Module 1: Unified 4D Patchify + SigLIP2 Embedding.

Converts raw input (image / video / 3D triplane) into:
  - f_spatial:  (B, N, 1152)      frozen SigLIP2 patch projections
  - positions:  (B, N, 4)         4D coordinates (t, x, y, z)
  - raw_patches: (B, N, C, p, p)  pixel patches for Module 2 DWT

4D coordinate convention:
  image:     t=0,    x=col_idx, y=row_idx, z=0
  video:     t=chunk_idx, x=col_idx, y=row_idx, z=0
  3D XY:     t=0,    x=col_idx, y=row_idx, z=0
  3D XZ:     t=0,    x=col_idx, y=0,       z=row_idx
  3D YZ:     t=0,    x=0,       y=col_idx, z=row_idx

SigLIP2 note:
  Only the Conv2d patch projection is used — not SigLIP2's own positional
  encoding — because MAVT applies 4D RoPE as its positional signal.
  The backbone is frozen; MAVT learns a modality embedding on top.

Fallback:
  If SigLIP2 weights cannot be loaded, a randomly-initialized Conv2d with
  the same spec is used so the rest of the pipeline can run immediately.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .config import PatchifyConfig
from .types import Modality, PatchifyOutput

logger = logging.getLogger(__name__)


# ─── SigLIP2 patch backbone ───────────────────────────────────────────────────

class SigLIP2PatchEmbed(nn.Module):
    """Frozen SigLIP2-SO400M patch projection.

    Wraps the Conv2d patch projection from the SigLIP2 backbone.
    Falls back to a learnable Conv2d if weights are unavailable.
    """

    def __init__(self, model_name: str, embed_dim: int = 1152) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.siglip2_loaded = False
        self._init_patch_proj(model_name, embed_dim)

    def _init_patch_proj(self, model_name: str, embed_dim: int) -> None:
        try:
            from transformers import SiglipVisionModel  # type: ignore
            model = SiglipVisionModel.from_pretrained(
                model_name, ignore_mismatched_sizes=True
            )
            # Extract only the patch projection Conv2d; skip SigLIP2 pos encoding.
            self.patch_proj: nn.Module = model.vision_model.embeddings.patch_embedding
            self.siglip2_loaded = True
            logger.info("SigLIP2 patch projection loaded from %s", model_name)
        except Exception as exc:
            logger.warning(
                "Cannot load SigLIP2 (%s); using learnable Conv2d fallback.", exc
            )
            self.patch_proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
            nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
            nn.init.zeros_(self.patch_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, N, embed_dim)  where N = (H/16) * (W/16)."""
        out = self.patch_proj(x)                      # (B, D, Nh, Nw)
        return out.flatten(2).transpose(1, 2)          # (B, N, D)


# ─── Main module ─────────────────────────────────────────────────────────────

class Unified4DPatchify(nn.Module):
    """Unified patchifier for image, video, and 3D triplane inputs.

    All three modalities produce the same output format (PatchifyOutput),
    enabling the rest of the pipeline to be modality-agnostic.
    """

    def __init__(self, cfg: Optional[PatchifyConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PatchifyConfig()
        self.p = self.cfg.patch_size_spatial    # 16
        self.tau = self.cfg.patch_size_temporal  # 2

        self.siglip2 = SigLIP2PatchEmbed(
            model_name=self.cfg.siglip2_model,
            embed_dim=self.cfg.embed_dim,
        )
        if self.cfg.freeze_siglip2:
            self.siglip2.requires_grad_(False)

        # Learnable modality offset: 0=image, 1=video, 2=3D
        self.mod_embed = nn.Embedding(3, self.cfg.embed_dim)
        nn.init.normal_(self.mod_embed.weight, std=0.02)

        # For non-3-channel triplanes: linear patch projection (lazy init)
        self._triplane_proj: Optional[nn.Linear] = None

    # ─── Public API ───────────────────────────────────────────────────────

    @staticmethod
    def detect_modality(x: torch.Tensor) -> Modality:
        """Infer modality from tensor shape.

        (B, C, H, W)         → image
        (B, C, T, H, W)      → video  (call with modality='3d' for triplanes)
        """
        if x.dim() == 4:
            return "image"
        if x.dim() == 5:
            return "video"
        raise ValueError(f"Unsupported input rank {x.dim()} — expected 4 or 5.")

    def forward(
        self,
        x: torch.Tensor,
        modality: Optional[Modality] = None,
    ) -> PatchifyOutput:
        """Patchify input into 4D token representation.

        Args:
            x:        (B, C, H, W) image, (B, C, T, H, W) video,
                      or (B, 3, C_tri, S, S) 3D triplane.
            modality: Override auto-detection (required for 3D triplanes).

        Returns:
            PatchifyOutput — unified 4D token representation.
        """
        if modality is None:
            modality = self.detect_modality(x)

        if modality == "image":
            return self._forward_image(x)
        if modality == "video":
            return self._forward_video(x)
        if modality == "3d":
            return self._forward_3d(x)
        raise ValueError(f"Unknown modality: '{modality}'")

    # ─── Per-modality forward ─────────────────────────────────────────────

    def _forward_image(self, img: torch.Tensor) -> PatchifyOutput:
        """img: (B, C, H, W) → PatchifyOutput."""
        B, C, H, W = img.shape
        n_h, n_w = H // self.p, W // self.p

        f_spatial = self.siglip2(img) + self.mod_embed.weight[0]  # (B, N, D)
        positions = self._build_positions(B, n_t=1, n_h=n_h, n_w=n_w, device=img.device)

        return PatchifyOutput(
            f_spatial=f_spatial,
            positions=positions,
            modality="image",
        )

    def _forward_video(self, video: torch.Tensor) -> PatchifyOutput:
        """video: (B, C, T, H, W) → PatchifyOutput with per-frame tokens."""
        B, C, T, H, W = video.shape
        n_h, n_w = H // self.p, W // self.p
        N_spatial = n_h * n_w

        # Per-frame SigLIP2 encoding — no temporal chunking (tau removed).
        frames_flat = (
            video.permute(0, 2, 1, 3, 4)       # (B, T, C, H, W)
            .reshape(B * T, C, H, W)
        )
        f = self.siglip2(frames_flat).contiguous()  # (B*T, N_spatial, D)
        f_spatial = (
            f.reshape(B, T * N_spatial, self.cfg.embed_dim)
            + self.mod_embed.weight[1]
        )

        positions = self._build_positions(
            B, n_t=T, n_h=n_h, n_w=n_w, device=video.device,
        )

        return PatchifyOutput(
            f_spatial=f_spatial,
            positions=positions,
            modality="video",
        )

    def _forward_3d(self, triplane: torch.Tensor) -> PatchifyOutput:
        """triplane: (B, 3, C_tri, S, S) → PatchifyOutput with depth signal.

        Three orthogonal planes (XY, XZ, YZ) are patchified independently
        and concatenated into a single token sequence.
        """
        B, n_planes, C_tri, S, _ = triplane.shape
        if n_planes != 3:
            raise ValueError(f"Expected 3 triplane channels, got {n_planes}")
        n_s = S // self.p
        N_per_plane = n_s * n_s

        # ── Per-plane configs: (plane_idx, x_col_or_val, y_col_or_val, z_col_or_val) ─
        # 'col' → meshgrid col dim (gb), 'row' → meshgrid row dim (ga), int → constant
        plane_configs = [
            (0, "col", "row",  0),    # XY: x=col, y=row, z=0
            (1, "col",  0,  "row"),   # XZ: x=col, y=0,   z=row
            (2,  0,   "col", "row"),  # YZ: x=0,   y=col, z=row
        ]

        ga, gb = torch.meshgrid(
            torch.arange(n_s, device=triplane.device, dtype=torch.float32),
            torch.arange(n_s, device=triplane.device, dtype=torch.float32),
            indexing="ij",
        )
        ga_flat = ga.flatten()   # row indices
        gb_flat = gb.flatten()   # col indices

        f_list, pos_list = [], []

        for plane_idx, x_src, y_src, z_src in plane_configs:
            plane = triplane[:, plane_idx]  # (B, C_tri, S, S)

            if C_tri == 3:
                f = self.siglip2(plane)
            else:
                proj = self._get_triplane_proj(C_tri, triplane.device)
                raw = self._extract_raw_patches(plane, n_s, n_s)
                raw_flat = raw.view(B, N_per_plane, C_tri * self.p * self.p)
                f = proj(raw_flat)
            f_list.append(f)

            pos = torch.zeros(B, N_per_plane, 4, device=triplane.device)
            pos[:, :, 1] = (gb_flat if x_src == "col" else ga_flat if x_src == "row" else float(x_src))
            pos[:, :, 2] = (gb_flat if y_src == "col" else ga_flat if y_src == "row" else float(y_src))
            pos[:, :, 3] = (gb_flat if z_src == "col" else ga_flat if z_src == "row" else float(z_src))
            pos_list.append(pos)

        f_spatial = torch.cat(f_list, dim=1) + self.mod_embed.weight[2]
        positions = torch.cat(pos_list, dim=1)

        return PatchifyOutput(
            f_spatial=f_spatial,
            positions=positions,
            modality="3d",
        )

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _build_positions(
        self,
        B: int,
        n_t: int,
        n_h: int,
        n_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build (B, N_total, 4) position tensor on a regular (t, x, y, z=0) grid.

        t ∈ [0, n_t),  x ∈ [0, n_w),  y ∈ [0, n_h),  z = 0.
        """
        t_idx = torch.arange(n_t, device=device, dtype=torch.float32)
        y_idx = torch.arange(n_h, device=device, dtype=torch.float32)
        x_idx = torch.arange(n_w, device=device, dtype=torch.float32)

        grid_y, grid_x = torch.meshgrid(y_idx, x_idx, indexing="ij")  # (n_h, n_w) each
        grid_x = grid_x.flatten()   # (N_spatial,)
        grid_y = grid_y.flatten()   # (N_spatial,)
        N_spatial = n_h * n_w

        t_exp = t_idx[:, None].expand(n_t, N_spatial)     # (n_t, N_spatial)
        x_exp = grid_x[None].expand(n_t, N_spatial)
        y_exp = grid_y[None].expand(n_t, N_spatial)
        z_exp = torch.zeros(n_t, N_spatial, device=device)

        positions = torch.stack([t_exp, x_exp, y_exp, z_exp], dim=-1)  # (n_t, N_spatial, 4)
        positions = positions.reshape(1, n_t * N_spatial, 4).expand(B, -1, -1)
        return positions.contiguous()

    def _extract_raw_patches(
        self, x: torch.Tensor, n_h: int, n_w: int
    ) -> torch.Tensor:
        """x: (B, C, H, W) → (B, N, C, p, p)  where N = n_h * n_w."""
        B, C = x.shape[:2]
        p = self.p
        patches = x.unfold(2, p, p).unfold(3, p, p)   # (B, C, n_h, n_w, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, n_h, n_w, C, p, p)
        return patches.view(B, n_h * n_w, C, p, p)

    def _extract_depth_signal(
        self, triplane: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract patch-level depth features along z-axis from XZ and YZ planes.

        Returns:
            depth_xz: (B, n_s, S_z, C)  per x-patch, all z features
            depth_yz: (B, n_s, S_z, C)  per y-patch, all z features
        """
        B, _, C, S, _ = triplane.shape
        n_s = S // self.p
        p = self.p

        xz_plane = triplane[:, 1]  # (B, C, S_x, S_z)
        yz_plane = triplane[:, 2]  # (B, C, S_y, S_z)

        # Pool x-axis pixels into patches: (B, C, S_x, S_z) → (B, C, n_s, p, S_z) → mean → (B, C, n_s, S_z)
        depth_xz = xz_plane.view(B, C, n_s, p, S).mean(3)   # (B, C, n_s, S_z)
        depth_xz = depth_xz.permute(0, 2, 3, 1).contiguous()  # (B, n_s, S_z, C)

        depth_yz = yz_plane.view(B, C, n_s, p, S).mean(3)   # (B, C, n_s, S_z)
        depth_yz = depth_yz.permute(0, 2, 3, 1).contiguous()  # (B, n_s, S_z, C)

        return depth_xz, depth_yz

    def _get_triplane_proj(self, C_tri: int, device: torch.device) -> nn.Linear:
        """Lazy-init a linear projection for non-3-channel triplanes."""
        if self._triplane_proj is None:
            in_dim = C_tri * self.p * self.p
            self._triplane_proj = nn.Linear(in_dim, self.cfg.embed_dim).to(device)
            nn.init.trunc_normal_(self._triplane_proj.weight, std=0.02)
            nn.init.zeros_(self._triplane_proj.bias)
        return self._triplane_proj


__all__ = ["SigLIP2PatchEmbed", "Unified4DPatchify"]
