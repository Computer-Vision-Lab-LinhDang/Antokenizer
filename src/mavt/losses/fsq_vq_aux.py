from __future__ import annotations

import torch
from torch import nn


class DiscreteAuxLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, discrete_out: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not discrete_out:
            zero = torch.tensor(0.0)
            return zero, zero
        indices = discrete_out["indices"].reshape(-1)
        commitment = discrete_out.get("commitment_loss", indices.new_zeros(()).float())
        if indices.numel() == 0:
            usage = commitment.new_zeros(())
        else:
            unique = torch.unique(indices).numel()
            usage = commitment.new_tensor(unique / max(indices.numel(), 1))
        total = commitment * self.commitment_weight
        return total, usage
