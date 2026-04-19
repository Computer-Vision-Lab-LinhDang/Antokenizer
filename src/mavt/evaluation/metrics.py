"""Evaluation metrics for MAVT.

Image  : rFID (requires torchmetrics-image), PSNR, SSIM
Video  : temporal PSNR (per-frame then averaged)
3D     : per-plane PSNR, SSIM
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR in dB. Inputs expected in [0, 1]."""
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(max_val ** 2 / (mse + 1e-8))


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Simplified SSIM via torchmetrics if available, else MSE proxy."""
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure as _ssim
        return _ssim(pred, target, data_range=1.0)
    except ImportError:
        return 1.0 - F.mse_loss(pred, target)


def compute_image_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """pred, target: (B, 3, H, W) in [0, 1]."""
    return {
        'psnr':  psnr(pred, target).item(),
        'ssim':  ssim(pred, target).item(),
    }


def compute_video_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """pred, target: (B, 3, T, H, W) in [0, 1]."""
    T = pred.shape[2]
    psnr_vals = [psnr(pred[:, :, t], target[:, :, t]).item() for t in range(T)]
    return {
        'temporal_psnr': sum(psnr_vals) / T,
        'temporal_psnr_min': min(psnr_vals),
    }


def compute_threed_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """pred, target: (B, 3, 3, H, W) — 3 planes each (B, 3, H, W)."""
    plane_names = ['xy', 'xz', 'yz']
    metrics: Dict[str, float] = {}
    for i, name in enumerate(plane_names):
        metrics[f'psnr_{name}'] = psnr(pred[:, i], target[:, i]).item()
        metrics[f'ssim_{name}'] = ssim(pred[:, i], target[:, i]).item()
    metrics['psnr_mean'] = sum(metrics[f'psnr_{n}'] for n in plane_names) / 3
    return metrics


class FIDTracker:
    """Accumulates features for FID computation using torchmetrics."""

    def __init__(self, feature: int = 2048):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self._fid = FrechetInceptionDistance(feature=feature, normalize=True)
            self._available = True
        except ImportError:
            self._available = False

    def update_real(self, imgs: torch.Tensor) -> None:
        if self._available:
            self._fid.update(imgs.clamp(0, 1), real=True)

    def update_fake(self, imgs: torch.Tensor) -> None:
        if self._available:
            self._fid.update(imgs.clamp(0, 1), real=False)

    def compute(self) -> Optional[float]:
        if self._available:
            return float(self._fid.compute())
        return None

    def reset(self) -> None:
        if self._available:
            self._fid.reset()
