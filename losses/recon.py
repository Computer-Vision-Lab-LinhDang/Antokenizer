from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_perc import CLIPPerceptualLoss
from .gram import GramLoss
from .lpips import LPIPSLoss


class ReconstructionLoss(nn.Module):
    """Aggregates perceptual and pixel reconstruction losses."""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_lpips: float = 10.0,
        lambda_gram: float = 1e3,
        lambda_clip: float = 1.0,
        normalise_by_patches: bool = True,
        lpips_loss: Optional[LPIPSLoss] = None,
        gram_loss: Optional[GramLoss] = None,
        clip_loss: Optional[CLIPPerceptualLoss] = None,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_lpips = lambda_lpips
        self.lambda_gram = lambda_gram
        self.lambda_clip = lambda_clip
        self.normalise_by_patches = normalise_by_patches

        self._lpips = lpips_loss
        self._gram = gram_loss or GramLoss()
        self._clip = clip_loss

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        *,
        target_lpips: Optional[torch.Tensor] = None,
        recon_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = 0.0
        logs: Dict[str, torch.Tensor] = {}

        if recon.dim() == 5 and target.dim() == 4:
            target = target.unsqueeze(2)

        if self.lambda_l1 > 0.0:
            l1 = F.l1_loss(recon, target, reduction="mean")
            if self.normalise_by_patches and num_patches:
                l1 = l1 / num_patches
            total_loss = total_loss + self.lambda_l1 * l1
            logs["l1_loss"] = l1.detach()

        if self.lambda_lpips > 0.0:
            if self._lpips is None:
                self._lpips = LPIPSLoss()
            lpips_loss, lpips_logs = self._lpips(
                recon.flatten(0, 1) if recon.dim() == 5 else recon,
                target.flatten(0, 1) if target.dim() == 5 else target,
            )
            if self.normalise_by_patches and num_patches:
                lpips_loss = lpips_loss / num_patches
            total_loss = total_loss + self.lambda_lpips * lpips_loss
            logs.update(lpips_logs)

        if self.lambda_gram > 0.0 and recon_features is not None and target_features is not None:
            gram_loss, gram_logs = self._gram(recon_features, target_features)
            if self.normalise_by_patches and num_patches:
                gram_loss = gram_loss / num_patches
            total_loss = total_loss + self.lambda_gram * gram_loss
            logs.update(gram_logs)

        if self.lambda_clip > 0.0:
            if self._clip is None:
                self._clip = CLIPPerceptualLoss()
            clip_loss, clip_logs = self._clip(
                recon.flatten(0, 1) if recon.dim() == 5 else recon,
                target.flatten(0, 1) if target.dim() == 5 else target,
            )
            if self.normalise_by_patches and num_patches:
                clip_loss = clip_loss / num_patches
            total_loss = total_loss + self.lambda_clip * clip_loss
            logs.update(clip_logs)

        logs["recon_total"] = torch.as_tensor(total_loss).detach()
        return torch.as_tensor(total_loss), logs


__all__ = ["ReconstructionLoss"]
