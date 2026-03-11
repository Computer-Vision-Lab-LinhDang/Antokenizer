"""Combined MAVT training loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    recon: float = 1.0
    kl: float = 1e-4
    lpips: float = 0.1
    use_lpips: bool = False


class MAVTLoss(nn.Module):
    """Combined loss: reconstruction (L1) + KL divergence + optional LPIPS."""

    def __init__(self, weights: Optional[LossWeights] = None) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self._lpips: Optional[nn.Module] = None
        if self.w.use_lpips:
            self._init_lpips()

    def _init_lpips(self) -> None:
        try:
            import lpips  # type: ignore
            self._lpips = lpips.LPIPS(net="vgg")
            self._lpips.requires_grad_(False)
        except ImportError:
            pass

    def forward(
        self,
        recon: torch.Tensor,       # (B, C, H, W) or (B, C, T, H, W)
        target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Align shapes
        if target.dim() == 5 and recon.dim() == 5:
            T = min(target.shape[2], recon.shape[2])
            target, recon = target[:, :, :T], recon[:, :, :T]
        elif target.dim() == 5 and recon.dim() == 4:
            target = target[:, :, 0]
        elif target.dim() == 4 and recon.dim() == 5:
            recon = recon[:, :, 0]

        recon_loss = F.l1_loss(recon, target)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()

        total = self.w.recon * recon_loss + self.w.kl * kl

        logs: dict[str, torch.Tensor] = {
            "recon_loss": recon_loss.detach(),
            "kl_loss":    kl.detach(),
        }

        if self.w.use_lpips and self._lpips is not None:
            # LPIPS requires (B, C, H, W) in [-1, 1]
            if recon.dim() == 4 and recon.shape[1] == 3:
                lp = self._lpips(recon.clamp(-1, 1), target.clamp(-1, 1)).mean()
                total = total + self.w.lpips * lp
                logs["lpips"] = lp.detach()

        logs["total_loss"] = total.detach()
        return {"loss": total, "logs": logs}


__all__ = ["LossWeights", "MAVTLoss"]
