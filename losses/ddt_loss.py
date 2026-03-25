"""Subband-aware reconstruction loss for DDT.

Three frequency-domain components:
  1. Pixel L1 (standard baseline)
  2. Structure L1: DWT LL2 subband of pred vs target
  3. Detail L1: DWT detail subbands (LH/HL/HH at both levels)
     w_detail > w_structure to prevent high-frequency suppression (FA-VAE insight)
  4. KL divergence

HaarDWT2D operates on full images (not patches), computing subbands for loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HaarDWT2DImage(nn.Module):
    """2-level Haar DWT on full images (any resolution, must be divisible by 4)."""

    @staticmethod
    def _haar_level(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        B, C, H, W = x.shape
        x_rs = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x00 = x_rs[:, :, :, 0, :, 0]
        x01 = x_rs[:, :, :, 0, :, 1]
        x10 = x_rs[:, :, :, 1, :, 0]
        x11 = x_rs[:, :, :, 1, :, 1]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, C, H, W) -> [LL2, LH2, HL2, HH2, LH1, HL1, HH1]"""
        ll1, lh1, hl1, hh1 = self._haar_level(x)
        ll2, lh2, hl2, hh2 = self._haar_level(ll1)
        return [ll2, lh2, hl2, hh2, lh1, hl1, hh1]


@dataclass
class LossWeights:
    w_pixel: float = 1.0
    w_structure: float = 1.0
    w_detail: float = 2.0
    w_kl: float = 1e-4
    w_lpips: float = 0.1
    use_lpips: bool = False


class SubbandAwareLoss(nn.Module):
    """Frequency-aware reconstruction + KL loss."""

    def __init__(self, weights: Optional[LossWeights] = None) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.dwt = _HaarDWT2DImage()
        self._lpips: Optional[nn.Module] = None
        if self.w.use_lpips:
            self._init_lpips()

    def _init_lpips(self) -> None:
        try:
            import lpips
            self._lpips = lpips.LPIPS(net="vgg")
            self._lpips.requires_grad_(False)
        except ImportError:
            pass

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Align video shapes
        if target.dim() == 5 and recon.dim() == 5:
            T = min(target.shape[2], recon.shape[2])
            target, recon = target[:, :, :T], recon[:, :, :T]
        elif target.dim() == 5 and recon.dim() == 4:
            target = target[:, :, 0]
        elif target.dim() == 4 and recon.dim() == 5:
            recon = recon[:, :, 0]

        # Pixel L1
        l_pixel = F.l1_loss(recon, target)

        # Subband L1 (flatten video to 4D if needed)
        recon_4d = self._to_4d(recon)
        target_4d = self._to_4d(target)

        # Ensure dimensions divisible by 4 for 2-level DWT
        H, W = recon_4d.shape[-2], recon_4d.shape[-1]
        if H >= 4 and W >= 4 and H % 4 == 0 and W % 4 == 0:
            pred_bands = self.dwt(recon_4d)
            tgt_bands = self.dwt(target_4d)
            l_structure = F.l1_loss(pred_bands[0], tgt_bands[0])
            l_detail = sum(
                F.l1_loss(p, t) for p, t in zip(pred_bands[1:], tgt_bands[1:])
            ) / 6.0
        else:
            l_structure = torch.zeros(1, device=recon.device).squeeze()
            l_detail = torch.zeros(1, device=recon.device).squeeze()

        # KL
        l_kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean()

        total = (
            self.w.w_pixel * l_pixel
            + self.w.w_structure * l_structure
            + self.w.w_detail * l_detail
            + self.w.w_kl * l_kl
        )

        # Optional LPIPS
        if self.w.use_lpips and self._lpips is not None and recon_4d.shape[1] == 3:
            lp = self._lpips(recon_4d.clamp(-1, 1), target_4d.clamp(-1, 1)).mean()
            total = total + self.w.w_lpips * lp

        logs = {
            "pixel_loss": l_pixel.detach(),
            "structure_loss": l_structure.detach(),
            "detail_loss": l_detail.detach(),
            "kl_loss": l_kl.detach(),
            "total_loss": total.detach(),
        }
        return {"loss": total, "logs": logs}

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        """Flatten video (B,C,T,H,W) -> (B*T,C,H,W) for DWT."""
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        return x


__all__ = ["LossWeights", "SubbandAwareLoss"]
