from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


try:
    import lpips as _lpips
except (ImportError, OSError):
    _lpips = None


class LPIPSLoss(nn.Module):
    """Wrapper around the LPIPS perceptual metric."""

    def __init__(
        self,
        net_type: str = "vgg",
        spatial: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.net_type = net_type
        self.spatial = spatial
        self.reduction = reduction
        self._lpips_fn: nn.Module | None = None

    def _lazy_init(self, device: torch.device) -> None:
        if self._lpips_fn is None:
            if _lpips is None:
                raise ImportError(
                    "lpips package is required for LPIPSLoss. "
                    "Install via `pip install lpips`."
                )
            self._lpips_fn = _lpips.LPIPS(
                net=self.net_type, spatial=self.spatial
            ).to(device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._lazy_init(pred.device)
        assert self._lpips_fn is not None
        loss = self._lpips_fn(pred, target)
        if loss.dim() > 1:
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()
        if self.reduction == "mean" and loss.dim() == 0:
            loss_value = loss
        else:
            loss_value = loss.mean()
        return loss_value, {"lpips_loss": loss_value.detach()}


__all__ = ["LPIPSLoss"]
