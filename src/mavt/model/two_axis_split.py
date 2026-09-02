"""Stage 3: Two-Axis Decomposition (Invariant-Variant Split)

MAVT v2 Core Innovation:

Split features into:
- z_inv (Invariant): mean across temporal/scale/view axis
- z_var (Variant): deviation from invariant

Key insight from UniJEPA:
- "Photometric prediction learns invariant structure"
- "Temporal prediction learns equivariant dynamics"

Benefits:
- Semantic loss optimizes z_inv ONLY
- Reconstruction loss optimizes z_var ONLY
- No gradient conflict → semantic quality preserved
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TwoAxisOutput:
    """Output of TwoAxisDecomposition module."""
    z_inv: torch.Tensor      # (B, N, D) - semantic, invariant
    z_var: torch.Tensor      # (B, N, K, D) - detail, variant
    energy_ratio: float       # energy_in_z_inv / total_energy
    gini: float             # Gini coefficient of z_var variance


class TwoAxisDecomposition(nn.Module):
    """Two-axis decomposition module.

    Splits features along the temporal/scale/view axis (k) into:
    - z_inv: invariant component (mean across k)
    - z_var: variant component (deviation from mean)

    This separation allows:
    - Semantic loss on z_inv only (gradient isolation)
    - Reconstruction loss on z_var only (gradient isolation)
    - No gradient conflict between semantic and recon objectives
    """

    def __init__(
        self,
        dim: int = 1152,
        axis: str = 'k',  # 'k' for temporal/scale/view axis
    ):
        super().__init__()
        self.dim = dim
        self.axis = axis

    def forward(
        self,
        feat: torch.Tensor,
        k_dim: int = 2,
        return_metrics: bool = True,
    ) -> TwoAxisOutput | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: (B, N, K, D) tensor where K is the axis to split on
                  - Video: K = temporal frames (after patchify)
                  - Image: K = scale pyramid levels (after scale pyramid encoder)
                  - 3D: K = view planes (after patchify)
            k_dim: which dimension to split on (default: 2)
            return_metrics: whether to compute and return metrics

        Returns:
            TwoAxisOutput with:
                - z_inv: (B, N, D) - invariant (semantic) component
                - z_var: (B, N, K, D) - variant (detail) component
            OR tuple (z_inv, z_var) if return_metrics=False
        """
        # Invariant: mean across k dimension
        z_inv = feat.mean(dim=k_dim)  # (B, N, D)

        # Variant: deviation from invariant
        z_var = feat - z_inv.unsqueeze(k_dim)  # (B, N, K, D)

        if not return_metrics:
            return z_inv, z_var

        # Compute metrics
        with torch.no_grad():
            total_energy = (feat ** 2).sum()
            inv_energy = (z_inv ** 2).sum()
            energy_ratio = (inv_energy / (total_energy + 1e-8)).item()

            # Gini coefficient of variance per patch
            var_per_patch = z_var.var(dim=[k_dim, -1])  # (B, N)
            gini = self._compute_gini(var_per_patch.mean(dim=0))  # (N,)

        return TwoAxisOutput(
            z_inv=z_inv,
            z_var=z_var,
            energy_ratio=energy_ratio,
            gini=gini,
        )

    @staticmethod
    def _compute_gini(x: torch.Tensor) -> float:
        """Compute Gini coefficient of a 1D tensor.

        Gini = 0: perfect equality (all values equal)
        Gini = 1: perfect inequality (one value dominates)
        """
        x = x.flatten()
        x = torch.sort(x).values
        n = len(x)
        index = torch.arange(1, n + 1, device=x.device)
        return ((2 * index - n - 1) * x).sum() / (n * x.sum() + 1e-8)


class TwoAxisWithCompression(nn.Module):
    """Two-axis decomposition with adaptive token budget.

    Allocates tokens based on variance in z_var:
    - High variance regions get more tokens
    - Low variance regions get fewer tokens
    """

    def __init__(
        self,
        dim: int = 1152,
        base_budget: int = 64,
        min_budget: int = 8,
        max_budget: int = 128,
    ):
        super().__init__()
        self.base_budget = base_budget
        self.min_budget = min_budget
        self.max_budget = max_budget

        self.two_axis = TwoAxisDecomposition(dim=dim)

        # Compression layers for z_var
        self.var_compress = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )

    def forward(
        self,
        feat: torch.Tensor,
        k_dim: int = 2,
        adaptive_budget: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feat: (B, N, K, D)
            k_dim: axis to split on
            adaptive_budget: whether to use adaptive token allocation

        Returns:
            z_inv: (B, N, D)
            z_var_compressed: (B, N_budget, D) or (B, N, K, D) if not adaptive
            metrics: dict with variance stats
        """
        out = self.two_axis(feat, k_dim)
        z_inv, z_var = out.z_inv, out.z_var

        metrics = {
            'energy_ratio': torch.tensor(out.energy_ratio, device=feat.device),
            'gini': torch.tensor(out.gini, device=feat.device),
        }

        if not adaptive_budget:
            return z_inv, z_var, metrics

        # Compute variance per patch for token allocation
        var_per_patch = z_var.var(dim=[k_dim, -1])  # (B, N)

        # Normalize to budget
        total_var = var_per_patch.sum(dim=1, keepdim=True) + 1e-8  # (B, 1)
        budget_float = (var_per_patch / total_var) * self.base_budget  # (B, N)
        budget = budget_float.long().clamp(self.min_budget, self.max_budget)  # (B, N)

        metrics['avg_budget'] = budget.float().mean()

        # Compress z_var based on budget
        # For now, return pooled version based on variance
        # In full impl, would do importance sampling per patch
        z_var_pooled = self._pool_by_variance(z_var, budget, k_dim)

        return z_inv, z_var_pooled, metrics

    def _pool_by_variance(
        self,
        z_var: torch.Tensor,
        budget: torch.Tensor,
        k_dim: int,
    ) -> torch.Tensor:
        """Pool z_var tokens based on variance budget.

        Simplified: just return mean across k for now.
        Full impl would do importance sampling.
        """
        # Simplified: return variance-weighted mean
        var_per_k = z_var.var(dim=k_dim, keepdim=True)  # (B, N, 1, D)
        weights = F.softmax(var_per_k / (var_per_k.mean() + 1e-8), dim=k_dim)
        pooled = (z_var * weights).sum(dim=k_dim)  # (B, N, D)
        return pooled


# ============================================================================
# Alternative: Spatial-Temporal Split for Video
# ============================================================================

class SpatialTemporalSplit(nn.Module):
    """Alternative: Split along spatial (p) and temporal (k) separately.

    For video, this gives:
    - z_spatial: spatial structure (mean across frames)
    - z_temporal: temporal dynamics (deviation from spatial mean)
    """

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feat: (B, T, H, W, D)
        Returns:
            z_spatial: (B, H, W, D) - spatial mean
            z_temporal: (B, T, H, W, D) - temporal deviation
        """
        B, T, H, W, D = feat.shape

        # Spatial: mean across time
        z_spatial = feat.mean(dim=1)  # (B, H, W, D)

        # Temporal: deviation from spatial mean
        z_temporal = feat - z_spatial.unsqueeze(1)  # (B, T, H, W, D)

        return z_spatial, z_temporal


# ============================================================================
# Utilities
# ============================================================================

def compute_axis_energy(feat: torch.Tensor, axis: int = 2) -> Dict[str, float]:
    """Compute energy distribution along an axis.

    Useful for analyzing temporal vs scale distributions.

    Args:
        feat: (B, N, K, D)
        axis: which axis to analyze

    Returns:
        dict with per-level energy fractions
    """
    total_energy = (feat ** 2).sum()

    energies = {}
    for k in range(feat.shape[axis]):
        level_slice = [slice(None)] * len(feat.shape)
        level_slice[axis] = k
        level_energy = (feat[tuple(level_slice)] ** 2).sum()
        energies[f'level_{k}'] = (level_energy / (total_energy + 1e-8)).item()

    return energies


def verify_two_axis_statistics(
    feat: torch.Tensor,
    k_dim: int = 2,
    expected_var_ratio: float = 0.02,  # 2% for video
    tolerance: float = 0.5,  # ±50% tolerance
) -> bool:
    """Verify that two-axis decomposition statistics match expected values.

    Use this to validate scale pyramid for images against video statistics.
    """
    two_axis = TwoAxisDecomposition()
    out = two_axis(feat, k_dim)

    # Check energy ratio
    expected_inv = 1.0 - expected_var_ratio
    inv_ratio = out.energy_ratio

    # Check Gini is in reasonable range (0.4-0.8)
    gini_ok = 0.4 <= out.gini <= 0.8

    # Check energy is close to expected
    inv_ok = abs(inv_ratio - expected_inv) <= tolerance

    return gini_ok and inv_ok
