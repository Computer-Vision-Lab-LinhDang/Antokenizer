"""Unified 4D Patchify + SigLIP2 — stripped for DDT.

Changes from mavt/module1_patchify.py:
  - Removed _extract_depth_signal (no STFT depth)
  - Removed raw_patches_temporal output (no STFT temporal)
  - Simplified _forward_video to extract one raw patch per temporal chunk
  - Simplified _forward_3d (no depth_signal)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .config import PatchifyConfig
from .types import Modality, PatchifyOutput

logger = logging.getLogger(__name__)


class SigLIP2PatchEmbed(nn.Module):
    """Frozen SigLIP2-SO400M patch projection with Conv2d fallback."""

    def __init__(self, model_name: str, embed_dim: int = 1152) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.siglip2_loaded = False
        self._init_patch_proj(model_name, embed_dim)

    def _init_patch_proj(self, model_name: str, embed_dim: int) -> None:
        try:
            from transformers import SiglipVisionModel
            model = SiglipVisionModel.from_pretrained(
                model_name, ignore_mismatched_sizes=True
            )
            self.patch_proj: nn.Module = model.vision_model.embeddings.patch_embedding
            self.siglip2_loaded = True
            logger.info("SigLIP2 patch projection loaded from %s", model_name)
        except Exception as exc:
            logger.warning("Cannot load SigLIP2 (%s); using Conv2d fallback.", exc)
            self.patch_proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
            nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
            nn.init.zeros_(self.patch_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, N, embed_dim)."""
        return self.patch_proj(x).flatten(2).transpose(1, 2)


class Unified4DPatchify(nn.Module):
    """Patchifier for image, video, and 3D triplane inputs."""

    def __init__(self, cfg: Optional[PatchifyConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PatchifyConfig()
        self.p = self.cfg.patch_size_spatial
        self.tau = self.cfg.patch_size_temporal

        self.siglip2 = SigLIP2PatchEmbed(
            model_name=self.cfg.siglip2_model,
            embed_dim=self.cfg.embed_dim,
        )
        if self.cfg.freeze_siglip2:
            self.siglip2.requires_grad_(False)

        self.mod_embed = nn.Embedding(3, self.cfg.embed_dim)
        nn.init.normal_(self.mod_embed.weight, std=0.02)

        self._triplane_proj: Optional[nn.Linear] = None

    @staticmethod
    def detect_modality(x: torch.Tensor) -> Modality:
        if x.dim() == 4:
            return "image"
        if x.dim() == 5:
            return "video"
        raise ValueError(f"Unsupported input rank {x.dim()}")

    def forward(
        self, x: torch.Tensor, modality: Optional[Modality] = None,
    ) -> PatchifyOutput:
        if modality is None:
            modality = self.detect_modality(x)
        if modality == "image":
            return self._forward_image(x)
        if modality == "video":
            return self._forward_video(x)
        if modality == "3d":
            return self._forward_3d(x)
        raise ValueError(f"Unknown modality: '{modality}'")

    # ── Image ──

    def _forward_image(self, img: torch.Tensor) -> PatchifyOutput:
        B, C, H, W = img.shape
        n_h, n_w = H // self.p, W // self.p

        f_spatial = self.siglip2(img) + self.mod_embed.weight[0]
        positions = self._build_positions(B, 1, n_h, n_w, img.device)
        raw_patches = self._extract_raw_patches(img, n_h, n_w)

        return PatchifyOutput(
            f_spatial=f_spatial, positions=positions,
            raw_patches=raw_patches, modality="image",
        )

    # ── Video ──

    def _forward_video(self, video: torch.Tensor) -> PatchifyOutput:
        B, C, T, H, W = video.shape
        n_t = T // self.tau
        n_h, n_w = H // self.p, W // self.p
        N_spatial = n_h * n_w
        T_used = n_t * self.tau

        # SigLIP2 on temporal-chunk averages
        chunks = (
            video[:, :, :T_used]
            .view(B, C, n_t, self.tau, H, W)
            .mean(dim=3)
            .permute(0, 2, 1, 3, 4)
            .reshape(B * n_t, C, H, W)
        )
        f = self.siglip2(chunks).contiguous()
        f_spatial = (
            f.reshape(B, n_t * N_spatial, self.cfg.embed_dim)
            + self.mod_embed.weight[1]
        )

        positions = self._build_positions(B, n_t, n_h, n_w, video.device)

        # Raw patches: first frame of each temporal chunk
        first_frames = (
            video[:, :, :T_used]
            .view(B, C, n_t, self.tau, H, W)[:, :, :, 0]
            .permute(0, 2, 1, 3, 4)
            .reshape(B * n_t, C, H, W)
        )
        raw_patches = self._extract_raw_patches(first_frames, n_h, n_w)
        raw_patches = raw_patches.view(B, n_t * N_spatial, C, self.p, self.p)

        return PatchifyOutput(
            f_spatial=f_spatial, positions=positions,
            raw_patches=raw_patches, modality="video",
        )

    # ── 3D triplane ──

    def _forward_3d(self, triplane: torch.Tensor) -> PatchifyOutput:
        B, n_planes, C_tri, S, _ = triplane.shape
        n_s = S // self.p
        N_per_plane = n_s * n_s

        plane_configs = [
            (0, "col", "row", 0),
            (1, "col", 0, "row"),
            (2, 0, "col", "row"),
        ]
        ga, gb = torch.meshgrid(
            torch.arange(n_s, device=triplane.device, dtype=torch.float32),
            torch.arange(n_s, device=triplane.device, dtype=torch.float32),
            indexing="ij",
        )
        ga_flat, gb_flat = ga.flatten(), gb.flatten()

        f_list, pos_list, patch_list = [], [], []
        for plane_idx, x_src, y_src, z_src in plane_configs:
            plane = triplane[:, plane_idx]
            if C_tri == 3:
                f = self.siglip2(plane)
            else:
                proj = self._get_triplane_proj(C_tri, triplane.device)
                raw = self._extract_raw_patches(plane, n_s, n_s)
                f = proj(raw.view(B, N_per_plane, C_tri * self.p * self.p))
            f_list.append(f)

            pos = torch.zeros(B, N_per_plane, 4, device=triplane.device)
            pos[:, :, 0] = float(plane_idx)  # t=plane_idx: XY=0 (content), XZ=1, YZ=2
            pos[:, :, 1] = gb_flat if x_src == "col" else (ga_flat if x_src == "row" else float(x_src))
            pos[:, :, 2] = gb_flat if y_src == "col" else (ga_flat if y_src == "row" else float(y_src))
            pos[:, :, 3] = gb_flat if z_src == "col" else (ga_flat if z_src == "row" else float(z_src))
            pos_list.append(pos)
            patch_list.append(self._extract_raw_patches(plane, n_s, n_s))

        f_spatial = torch.cat(f_list, dim=1) + self.mod_embed.weight[2]
        positions = torch.cat(pos_list, dim=1)
        raw_patches = torch.cat(patch_list, dim=1)

        return PatchifyOutput(
            f_spatial=f_spatial, positions=positions,
            raw_patches=raw_patches, modality="3d",
        )

    # ── Helpers ──

    def _build_positions(
        self, B: int, n_t: int, n_h: int, n_w: int, device: torch.device,
    ) -> torch.Tensor:
        t_idx = torch.arange(n_t, device=device, dtype=torch.float32)
        y_idx = torch.arange(n_h, device=device, dtype=torch.float32)
        x_idx = torch.arange(n_w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_idx, x_idx, indexing="ij")
        grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
        N_sp = n_h * n_w

        t_exp = t_idx[:, None].expand(n_t, N_sp)
        x_exp = grid_x[None].expand(n_t, N_sp)
        y_exp = grid_y[None].expand(n_t, N_sp)
        z_exp = torch.zeros(n_t, N_sp, device=device)

        positions = torch.stack([t_exp, x_exp, y_exp, z_exp], dim=-1)
        return positions.reshape(1, n_t * N_sp, 4).expand(B, -1, -1).contiguous()

    def _extract_raw_patches(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        B, C = x.shape[:2]
        p = self.p
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(B, n_h * n_w, C, p, p)

    def _get_triplane_proj(self, C_tri: int, device: torch.device) -> nn.Linear:
        if self._triplane_proj is None:
            in_dim = C_tri * self.p * self.p
            self._triplane_proj = nn.Linear(in_dim, self.cfg.embed_dim).to(device)
            nn.init.trunc_normal_(self._triplane_proj.weight, std=0.02)
            nn.init.zeros_(self._triplane_proj.bias)
        return self._triplane_proj


__all__ = ["SigLIP2PatchEmbed", "Unified4DPatchify"]
