from __future__ import annotations

import torch
from torch import nn


class KLRegLoss(nn.Module):
    def __init__(self, weight: float = 1e-5) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean() * self.weight
