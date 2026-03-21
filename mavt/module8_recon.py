# mavt/module8_recon.py

from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.recon import ReconstructionLoss


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


# Self-Attention Block (lightweight)
class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_norm = self.norm(x_flat)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        out = out + x_flat
        return out.permute(0, 2, 1).view(B, C, H, W)


# Multi-scale refinement
class MultiScaleRefiner(nn.Module):
    def __init__(self, in_channels: int = 3, base_dim: int = 64):
        super().__init__()

        self.entry = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        self.stage1 = nn.Sequential(
            ResBlock(base_dim),
            ResBlock(base_dim),
        )

        self.down = nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1)

        self.stage2 = nn.Sequential(
            ResBlock(base_dim * 2),
            AttentionBlock(base_dim * 2),
            ResBlock(base_dim * 2),
        )

        self.up = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)

        self.stage3 = nn.Sequential(
            ResBlock(base_dim),
            AttentionBlock(base_dim),
        )

        self.out = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def forward(self, x):
        x0 = self.entry(x)

        x1 = self.stage1(x0)

        x2 = self.down(x1)
        x2 = self.stage2(x2)

        x3 = self.up(x2) + x1
        x3 = self.stage3(x3)

        return x + self.out(x3)


# Feature Pyramid Extractor
class FeaturePyramid(nn.Module):
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1)

    def forward(self, x) -> List[torch.Tensor]:
        f1 = F.gelu(self.conv1(x))
        f2 = F.gelu(self.conv2(f1))
        f3 = F.gelu(self.conv3(f2))
        return [f1, f2, f3]


# Main Reconstruction Head
class ReconstructionHead(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.refiner = MultiScaleRefiner(in_channels)
        self.pyramid = FeaturePyramid(in_channels)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ):
        if reconstruction.dim() == 5:
            B, C, T, H, W = reconstruction.shape
            x = reconstruction.view(B * T, C, H, W)
            x = self.refiner(x)
            recon = x.view(B, C, T, H, W)
        else:
            recon = self.refiner(reconstruction)

        recon_feats = None
        target_feats = None

        if target is not None:
            if recon.dim() == 5:
                recon_feats = self.pyramid(recon[:, :, 0])
                target_feats = self.pyramid(target[:, :, 0])
            else:
                recon_feats = self.pyramid(recon)
                target_feats = self.pyramid(target)

        return recon, recon_feats, target_feats


# Criterion
class ReconstructionCriterion(nn.Module):
    def __init__(
        self,
        *,
        lambda_l1: float = 1.0,
        lambda_lpips: float = 10.0,
        lambda_gram: float = 1e3,
        lambda_clip: float = 1.0,
    ):
        super().__init__()

        self.loss_fn = ReconstructionLoss()

        # weights (even if loss_fn handles internally → keep for control/logging)
        self.lambda_l1 = lambda_l1
        self.lambda_lpips = lambda_lpips
        self.lambda_gram = lambda_gram
        self.lambda_clip = lambda_clip

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        *,
        recon_features=None,
        target_features=None,
        num_patches=None,
    ):
        # 🔹 Normalize feature input
        if isinstance(recon_features, list):
            recon_features = recon_features[-1]
            target_features = target_features[-1]

        # 🔹 Compute base loss
        loss, logs = self.loss_fn(
            recon,
            target,
            recon_features=recon_features,
            target_features=target_features,
            num_patches=num_patches,
        )

        # 🔹 Ensure logs exist
        if logs is None:
            logs = {}

        # 🔹 (Optional) reweight if loss_fn doesn't handle it
        # Detect by keys (robust design)
        total_loss = 0.0

        if "l1" in logs:
            total_loss += self.lambda_l1 * logs["l1"]

        if "lpips" in logs:
            total_loss += self.lambda_lpips * logs["lpips"]

        if "gram" in logs:
            total_loss += self.lambda_gram * logs["gram"]

        if "clip" in logs:
            total_loss += self.lambda_clip * logs["clip"]

        # fallback if loss_fn already returned weighted loss
        if total_loss == 0:
            total_loss = loss

        logs["total"] = total_loss

        return total_loss, logs


__all__ = [
    "ReconstructionHead",
    "ReconstructionCriterion",
]