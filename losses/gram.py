from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _gram_matrix(features: torch.Tensor) -> torch.Tensor:
    b, c, h, w = features.shape
    features = features.view(b, c, h * w)
    gram = torch.matmul(features, features.transpose(1, 2))
    gram = gram / (c * h * w)
    return gram


class GramLoss(nn.Module):
    """Computes Gram matrix covariance matching loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gram_pred = _gram_matrix(pred_features)
        gram_target = _gram_matrix(target_features)
        loss = torch.mean((gram_pred - gram_target) ** 2, dim=(1, 2))
        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss, {"gram_loss": loss.detach()}


__all__ = ["GramLoss"]
