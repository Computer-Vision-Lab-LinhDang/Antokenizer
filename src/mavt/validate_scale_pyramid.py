"""Validation script for Scale Pyramid Statistics.

This script measures scale pyramid statistics on images and compares
them to video temporal statistics to validate the approach.

Expected (from video audit):
- z_inv energy: 97.6-98.8%
- z_var energy: 1.2-2.4%
- Gini coefficient: 0.62-0.76

If scale pyramid matches, we can use it as synthetic "temporal axis" for images.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Scale Pyramid Encoder (Same as in mavt_v2.py)
# ============================================================================

class ScalePyramidEncoder(nn.Module):
    """Multi-scale encoder for images."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_scales: int = 4,
        patch_size: int = 16,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.patch_size = patch_size

        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

        # Scale-specific projections
        self.scale_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_scales)
        ])

    def forward(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, N, K, D)"""
        B, C, H, W = x.shape
        features = []

        for k in range(self.num_scales):
            scale = 2 ** k
            if scale > 1:
                h_s = max(1, H // scale)
                w_s = max(1, W // scale)
                x_scaled = F.interpolate(x, size=(h_s, w_s), mode='bilinear', align_corners=False)
            else:
                x_scaled = x

            feat = self.encoder(x_scaled)  # (B, D, H', W')
            feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            B, D, H_t, W_t = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(B, H_t * W_t, D)
            feat = self.scale_proj[k](feat)
            features.append(feat)

        return torch.stack(features, dim=2)  # (B, N, K, D)


# ============================================================================
# Two-Axis Decomposition (Same as in mavt_v2.py)
# ============================================================================

class TwoAxisDecomposition(nn.Module):
    """Split into invariant and variant."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim

    def forward(self, features: torch.Tensor, k_dim: int = 2):
        z_inv = features.mean(dim=k_dim)
        z_var = features - z_inv.unsqueeze(k_dim)

        with torch.no_grad():
            total_energy = (features ** 2).sum()
            inv_energy = (z_inv ** 2).sum()
            energy_ratio = (inv_energy / (total_energy + 1e-8)).item()
            var_per_patch = z_var.var(dim=[k_dim, -1])
            gini = self._compute_gini(var_per_patch.mean(dim=0))

        return {
            'z_inv': z_inv,
            'z_var': z_var,
            'energy_ratio': energy_ratio,
            'gini': gini,
            'var_energy_ratio': 1 - energy_ratio,
        }

    @staticmethod
    def _compute_gini(x: torch.Tensor) -> float:
        x = x.flatten()
        x = torch.sort(x).values
        n = len(x)
        index = torch.arange(1, n + 1, device=x.device)
        return ((2 * index - n - 1) * x).sum() / (n * x.sum() + 1e-8)


# ============================================================================
# Synthetic Dataset
# ============================================================================

class SyntheticImageDataset(Dataset):
    """Generate synthetic images for testing."""

    def __init__(
        self,
        num_images: int = 100,
        image_size: int = 256,
        include_patterns: bool = True,
    ):
        self.num_images = num_images
        self.image_size = image_size
        self.include_patterns = include_patterns

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a synthetic image with various patterns."""
        torch.manual_seed(idx)

        # Base image: random noise
        x = torch.randn(3, self.image_size, self.image_size)

        # Add patterns based on idx
        if self.include_patterns:
            pattern = idx % 5

            if pattern == 0:
                # Checkerboard
                x = self._checkerboard(x)
            elif pattern == 1:
                # Gradient
                x = self._gradient(x)
            elif pattern == 2:
                # Stripes
                x = self._stripes(x)
            elif pattern == 3:
                # Circles
                x = self._circles(x)
            else:
                # Texture
                x = self._texture(x)

        # Normalize to [-1, 1]
        x = torch.tanh(x)

        return x

    def _checkerboard(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[1:]
        checker = torch.zeros(H, W)
        for i in range(H):
            for j in range(W):
                checker[i, j] = (i // 16 + j // 16) % 2
        for c in range(3):
            x[c] = x[c] * 0.5 + checker * 0.5
        return x

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[1:]
        grad = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
        for c in range(3):
            x[c] = x[c] * 0.5 + grad * 0.5
        return x

    def _stripes(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[1:]
        stripes = (torch.arange(W) // 16) % 2
        for c in range(3):
            x[c] = x[c] * 0.5 + stripes * 0.5
        return x

    def _circles(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[1:]
        y, z = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        center = torch.tensor([H // 2, W // 2])
        dist = ((torch.stack([y, z]) - center.unsqueeze(-1).float()) ** 2).sum(dim=0).sqrt()
        circles = ((dist % 32) < 16).float()
        for c in range(3):
            x[c] = x[c] * 0.5 + circles * 0.5
        return x

    def _texture(self, x: torch.Tensor) -> torch.Tensor:
        # Add random high-frequency texture
        texture = torch.randn_like(x) * 0.3
        return x * 0.7 + texture


# ============================================================================
# Statistics Measurement
# ============================================================================

def measure_scale_statistics(
    encoder: ScalePyramidEncoder,
    decomposition: TwoAxisDecomposition,
    images: torch.Tensor,
    num_scales: int = 4,
) -> Dict[str, float]:
    """Measure scale pyramid statistics on a batch of images."""

    B, C, H, W = images.shape
    N_h = H // 16
    N_w = W // 16

    with torch.no_grad():
        # Encode with scale pyramid
        feat = encoder(images, N_h, N_w)  # (B, N, K, D)

        # Decompose
        result = decomposition(feat, k_dim=2)

        # Per-scale energy
        total_energy = (feat ** 2).sum()
        per_scale_energy = []
        for k in range(num_scales):
            scale_energy = (feat[:, :, k, :] ** 2).sum()
            per_scale_energy.append((scale_energy / (total_energy + 1e-8)).item())

    return {
        'energy_in_z_inv': result['energy_ratio'],
        'energy_in_z_var': result['var_energy_ratio'],
        'gini_coefficient': result['gini'],
        'per_scale_energy': per_scale_energy,
        'num_images': B,
        'image_size': H,
        'num_scales': num_scales,
    }


def compare_to_video_statistics(scale_stats: Dict) -> Dict:
    """Compare scale statistics to video temporal statistics.

    Video statistics (from audit6):
    - z_inv energy: 97.6-98.8%
    - z_var energy: 1.2-2.4%
    - Gini: 0.62-0.76
    """
    video_stats = {
        'energy_in_z_inv': (0.976, 0.988),  # min, max
        'energy_in_z_var': (0.012, 0.024),
        'gini_coefficient': (0.62, 0.76),
    }

    results = {}

    # Check energy in z_inv
    inv_min, inv_max = video_stats['energy_in_z_inv']
    inv_ok = inv_min <= scale_stats['energy_in_z_inv'] <= inv_max
    results['z_inv_energy_match'] = inv_ok
    results['z_inv_energy_diff'] = abs(
        scale_stats['energy_in_z_inv'] - (inv_min + inv_max) / 2
    )

    # Check energy in z_var
    var_min, var_max = video_stats['energy_in_z_var']
    var_ok = var_min <= scale_stats['energy_in_z_var'] <= var_max
    results['z_var_energy_match'] = var_ok
    results['z_var_energy_diff'] = abs(
        scale_stats['energy_in_z_var'] - (var_min + var_max) / 2
    )

    # Check Gini
    gini_min, gini_max = video_stats['gini_coefficient']
    gini_ok = gini_min <= scale_stats['gini_coefficient'] <= gini_max
    results['gini_match'] = gini_ok
    results['gini_diff'] = abs(
        scale_stats['gini_coefficient'] - (gini_min + gini_max) / 2
    )

    # Check energy decreasing (coarser = more energy)
    per_scale = scale_stats['per_scale_energy']
    energy_decreasing = all(
        per_scale[i] >= per_scale[i + 1] * 0.8  # Allow 20% tolerance
        for i in range(len(per_scale) - 1)
    )
    results['energy_decreasing'] = energy_decreasing

    # Overall
    results['validation_passed'] = (
        inv_ok and var_ok and gini_ok and energy_decreasing
    )

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate Scale Pyramid Statistics')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to test')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_scales', type=int, default=4, help='Number of scale levels')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()

    print("=" * 60)
    print("SCALE PYRAMID VALIDATION")
    print("=" * 60)
    print(f"Num images: {args.num_images}")
    print(f"Image size: {args.image_size}")
    print(f"Num scales: {args.num_scales}")
    print(f"Embed dim: {args.embed_dim}")
    print()

    # Create model
    encoder = ScalePyramidEncoder(
        embed_dim=args.embed_dim,
        num_scales=args.num_scales,
    )
    decomposition = TwoAxisDecomposition(dim=args.embed_dim)

    # Create dataset
    dataset = SyntheticImageDataset(
        num_images=args.num_images,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Measure statistics
    all_stats = []
    print("Measuring statistics...")

    for i, images in enumerate(loader):
        stats = measure_scale_statistics(
            encoder, decomposition, images, args.num_scales
        )
        all_stats.append(stats)
        if (i + 1) % 5 == 0:
            print(f"  Processed {(i + 1) * args.batch_size} images")

    # Average statistics
    avg_stats = {
        'energy_in_z_inv': sum(s['energy_in_z_inv'] for s in all_stats) / len(all_stats),
        'energy_in_z_var': sum(s['energy_in_z_var'] for s in all_stats) / len(all_stats),
        'gini_coefficient': sum(s['gini_coefficient'] for s in all_stats) / len(all_stats),
        'per_scale_energy': [
            sum(s['per_scale_energy'][k] for s in all_stats) / len(all_stats)
            for k in range(args.num_scales)
        ],
        'num_images': sum(s['num_images'] for s in all_stats),
    }

    # Compare to video
    comparison = compare_to_video_statistics(avg_stats)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n📊 Scale Pyramid Statistics:")
    print(f"  Energy in z_inv (invariant): {avg_stats['energy_in_z_inv']:.4f} ({avg_stats['energy_in_z_inv']*100:.2f}%)")
    print(f"  Energy in z_var (variant):   {avg_stats['energy_in_z_var']:.4f} ({avg_stats['energy_in_z_var']*100:.2f}%)")
    print(f"  Gini coefficient:            {avg_stats['gini_coefficient']:.4f}")

    print("\n📈 Per-Scale Energy:")
    for k, e in enumerate(avg_stats['per_scale_energy']):
        scale_size = args.image_size // (2 ** k)
        print(f"  Scale {k} ({scale_size}x{scale_size}): {e:.4f} ({e*100:.2f}%)")

    print("\n📊 Comparison to Video Temporal Axis:")
    print(f"  z_inv energy match:    {'✅' if comparison['z_inv_energy_match'] else '❌'} "
          f"(diff: {comparison['z_inv_energy_diff']*100:.2f}%)")
    print(f"  z_var energy match:    {'✅' if comparison['z_var_energy_match'] else '❌'} "
          f"(diff: {comparison['z_var_energy_diff']*100:.2f}%)")
    print(f"  Gini match:            {'✅' if comparison['gini_match'] else '❌'} "
          f"(diff: {comparison['gini_diff']:.4f})")
    print(f"  Energy decreasing:     {'✅' if comparison['energy_decreasing'] else '❌'}")

    print("\n" + "=" * 60)
    if comparison['validation_passed']:
        print("✅ VALIDATION PASSED")
        print("Scale pyramid statistics match video temporal axis!")
        print("Scale pyramid can be used as synthetic temporal axis for images.")
    else:
        print("❌ VALIDATION FAILED")
        print("Scale pyramid statistics do not match video temporal axis.")
        print("Consider adjusting scale levels or using alternative approach.")
    print("=" * 60)

    # Save results
    if args.output:
        results = {
            'scale_stats': {k: v for k, v in avg_stats.items() if k != 'per_scale_energy'},
            'per_scale_energy': avg_stats['per_scale_energy'],
            'comparison': comparison,
            'config': {
                'num_images': args.num_images,
                'image_size': args.image_size,
                'num_scales': args.num_scales,
                'embed_dim': args.embed_dim,
            },
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return comparison['validation_passed']


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
