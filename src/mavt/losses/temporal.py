from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class OpticalFlowConsistencyLoss(nn.Module):
    """
    Lightweight temporal consistency proxy.

    A real RAFT-based implementation can replace this module without changing the
    training loop contract.
    """

    def __init__(self, weight: float = 0.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0 or target.ndim != 5:
            return target.new_zeros(())
        target_delta = target[:, :, 1:] - target[:, :, :-1]
        rec_delta = reconstruction[:, :, 1:] - reconstruction[:, :, :-1]
        return F.l1_loss(rec_delta, target_delta) * self.weight
