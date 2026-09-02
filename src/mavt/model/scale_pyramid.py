"""Scale Pyramid Encoder for Images.

MAVT v2: Use multi-resolution scales as synthetic "temporal axis" for images.

This enables:
1. Unified two-axis architecture across modalities
2. Semantic extraction from scale-invariant representation
3. Detail preservation from scale-variant representation
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalePyramidEncoder(nn.Module):
    """Encode images at multiple scales to create synthetic temporal axis.

    For video: temporal axis = frame index
    For image: scale pyramid = multi-resolution "time"

    Output shape: (B, N, K, D) where:
    - N = number of spatial patches (at finest scale)
    - K = number of scale levels
    - D = embedding dimension

    Scale levels: {1x, 2x, 4x, 8x} → K=4 default
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_scales: int = 4,
        scale_base: int = 2,
        patch_size: int = 16,
        align_mode: str = 'upsample',  # 'upsample' | 'pool' | 'interpolate'
    ):
        """
        Args:
            encoder: Base encoder that processes single-scale inputs
            num_scales: Number of scale levels (K)
            scale_base: Base for scale progression (2 = doubling)
            patch_size: Patch size for the encoder
            align_mode: How to align different scales to common resolution
        """
        super().__init__()
        self.encoder = encoder
        self.num_scales = num_scales
        self.scale_base = scale_base
        self.patch_size = patch_size
        self.align_mode = align_mode

    def get_scale_factors(self) -> List[int]:
        """Get scale factors for each pyramid level."""
        return [self.scale_base ** k for k in range(self.num_scales)]

    def align_scales(
        self,
        features: List[torch.Tensor],
        target_h: int,
        target_w: int,
    ) -> List[torch.Tensor]:
        """Align all scale features to a common (H, W) resolution.

        Args:
            features: List of (B, D, H_k, W_k) tensors at different scales
            target_h: Target height (number of patches)
            target_w: Target width (number of patches)

        Returns:
            List of (B, D, target_h, target_w) aligned tensors
        """
        aligned = []
        for feat in features:
            if feat.shape[2:] == (target_h, target_w):
                aligned.append(feat)
            elif self.align_mode == 'upsample':
                # Upsample coarser scales to match finest
                aligned.append(F.interpolate(feat, size=(target_h, target_w),
                                             mode='bilinear', align_corners=False))
            elif self.align_mode == 'pool':
                # Pool finer scales down
                scale = feat.shape[2] // target_h
                if scale > 1:
                    feat = F.avg_pool2d(feat, kernel_size=scale, stride=scale)
                aligned.append(feat)
            else:
                # Direct interpolate
                aligned.append(F.interpolate(feat, size=(target_h, target_w),
                                             mode='bilinear', align_corners=False))
        return aligned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            feat: (B, N, K, D) multi-scale features
                N = H//patch_size * W//patch_size (finest scale patches)
                K = num_scales
        """
        B, C, H, W = x.shape
        device = x.device

        # Finest scale patches
        N_h = H // self.patch_size
        N_w = W // self.patch_size
        N = N_h * N_w

        # Get target dimensions (finest scale)
        target_h, target_w = N_h, N_w

        # Encode at each scale
        features = []
        scale_factors = self.get_scale_factors()

        for k, scale in enumerate(scale_factors):
            if scale > 1:
                # Downsample image
                down_h = max(1, H // scale)
                down_w = max(1, W // scale)
                x_scaled = F.interpolate(x, size=(down_h, down_w),
                                         mode='bilinear', align_corners=False)
            else:
                x_scaled = x

            # Encode at this scale
            # Expected encoder output: (B, D, H', W')
            feat = self.encoder(x_scaled)

            # Handle different encoder output formats
            if len(feat.shape) == 2:  # (B, D) global features
                # Expand to spatial
                feat = feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_h, target_w)
            elif len(feat.shape) == 3:  # (B, N', D)
                feat = feat.transpose(1, 2).reshape(B, -1, target_h, target_w)
            # else: assume (B, D, H', W')

            # Ensure correct dimensions via interpolation
            feat = F.interpolate(feat, size=(target_h, target_w),
                                mode='bilinear', align_corners=False)

            features.append(feat)

        # Stack along scale dimension
        # features[k]: (B, D, target_h, target_w)
        # result: (B, K, D, target_h, target_w)
        stacked = torch.stack(features, dim=1)  # (B, K, D, H, W)

        # Reshape to (B, N, K, D)
        stacked = stacked.permute(0, 3, 4, 1, 2)  # (B, H, W, K, D)
        result = stacked.reshape(B, N, self.num_scales, -1)  # (B, N, K, D)

        return result


class LightweightScaleEncoder(nn.Module):
    """Lightweight multi-scale encoder without shared weights.

    Uses separate small encoders per scale for efficiency.
    """

    def __init__(
        self,
        dim: int = 256,
        num_scales: int = 4,
        scale_base: int = 2,
        patch_size: int = 16,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.scale_base = scale_base

        # Shared encoder with multi-scale input
        # In practice, use a single encoder processing scaled inputs
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(3, dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
        )

        # Scale-specific projections
        self.scale_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_scales)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            feat: (B, N, K, D)
        """
        B, C, H, W = x.shape
        scale_factors = [self.scale_base ** k for k in range(self.num_scales)]

        features = []
        for k, scale in enumerate(scale_factors):
            # Scale input
            if scale > 1:
                x_scaled = F.interpolate(x, scale_factor=1/scale,
                                         mode='bilinear', align_corners=False)
            else:
                x_scaled = x

            # Encode
            feat = self.shared_encoder(x_scaled)  # (B, D, H', W')

            # Project and align
            B, D, H_k, W_k = feat.shape
            feat_flat = feat.flatten(2).transpose(1, 2)  # (B, N_k, D)
            feat_proj = self.scale_proj[k](feat_flat)  # (B, N_k, D)

            # Interpolate to common resolution
            N_target = (H // 16) * (W // 16)
            H_target = H // 16
            W_target = W // 16
            feat_aligned = F.interpolate(
                feat_proj.transpose(1, 2).reshape(B, D, H_k, W_k),
                size=(H_target, W_target),
                mode='bilinear', align_corners=False
            )
            features.append(feat_aligned.flatten(2).transpose(1, 2))

        # Stack: (B, N, K, D)
        return torch.stack(features, dim=2)


# ============================================================================
# Testing Utilities
# ============================================================================

def measure_scale_statistics(
    encoder: nn.Module,
    images: torch.Tensor,
    num_scales: int = 4,
) -> dict:
    """Measure scale pyramid statistics for validation.

    Args:
        encoder: Scale pyramid encoder
        images: (B, 3, H, W) batch of images
        num_scales: Number of scales

    Returns:
        dict with statistics for validation
    """
    device = images.device

    # Encode images
    scale_encoder = ScalePyramidEncoder(encoder, num_scales=num_scales)
    scale_encoder.eval()

    with torch.no_grad():
        feat = scale_encoder(images)  # (B, N, K, D)

    B, N, K, D = feat.shape

    # Compute statistics
    total_energy = (feat ** 2).sum()

    # Invariant = mean across scales
    z_inv = feat.mean(dim=2)  # (B, N, D)
    inv_energy = (z_inv ** 2).sum()
    var_energy = ((feat - z_inv.unsqueeze(2)) ** 2).sum()

    energy_ratio = (inv_energy / (total_energy + 1e-8)).item()
    var_ratio = (var_energy / (total_energy + 1e-8)).item()

    # Per-scale energy
    per_scale_energy = []
    for k in range(K):
        scale_energy = (feat[:, :, k, :] ** 2).sum()
        per_scale_energy.append((scale_energy / (total_energy + 1e-8)).item())

    # Gini coefficient
    var_per_patch = (feat - z_inv.unsqueeze(2)).var(dim=[2, 3])  # (B, N)
    var_flat = var_per_patch.mean(dim=0)
    var_sorted = torch.sort(var_flat).values
    n = len(var_sorted)
    index = torch.arange(1, n + 1, device=device)
    gini = ((2 * index - n - 1) * var_sorted).sum() / (n * var_sorted.sum() + 1e-8)
    gini = gini.item()

    return {
        'energy_in_invariant': energy_ratio,
        'energy_in_variant': var_ratio,
        'per_scale_energy': per_scale_energy,
        'gini_coefficient': gini,
        'num_scales': K,
        'num_patches': N,
        'embed_dim': D,
    }


def compare_to_video_statistics(scale_stats: dict, video_stats: dict) -> dict:
    """Compare scale pyramid statistics to video temporal statistics.

    Returns validation result.
    """
    # Energy in variant should be similar
    var_diff = abs(scale_stats['energy_in_variant'] - video_stats['energy_in_variant'])

    # Gini should be in similar range
    gini_diff = abs(scale_stats['gini_coefficient'] - video_stats['gini_coefficient'])

    # Per-scale energy should be decreasing (coarser = more energy)
    per_scale = scale_stats['per_scale_energy']
    decreasing = all(per_scale[i] >= per_scale[i+1] for i in range(len(per_scale)-1))

    return {
        'variant_energy_match': var_diff < 0.02,  # Within 2%
        'gini_match': gini_diff < 0.1,  # Within 0.1
        'energy_decreasing': decreasing,
        'variant_energy_diff': var_diff,
        'gini_diff': gini_diff,
        'validation_passed': (
            var_diff < 0.02 and
            gini_diff < 0.1 and
            decreasing
        ),
    }
