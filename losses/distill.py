from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """KL-based distillation for semantic alignment."""

    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "batchmean",
        target_type: str = "softmax",
    ) -> None:
        super().__init__()
        if reduction not in {"batchmean", "mean", "sum"}:
            raise ValueError("Unsupported reduction")
        if target_type not in {"softmax", "sigmoid"}:
            raise ValueError("target_type must be 'softmax' or 'sigmoid'")
        self.temperature = temperature
        self.reduction = reduction
        self.target_type = target_type

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        t = self.temperature
        if self.target_type == "softmax":
            student_log_probs = F.log_softmax(student_logits / t, dim=-1)
            teacher_probs = F.softmax(teacher_logits / t, dim=-1)
        else:
            student_log_probs = torch.logsigmoid(student_logits / t)
            teacher_probs = torch.sigmoid(teacher_logits / t)
        if mask is not None:
            student_log_probs = student_log_probs[mask]
            teacher_probs = teacher_probs[mask]

        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction=self.reduction,
        ) * (t * t)
        return loss, {"distill_loss": loss.detach()}


__all__ = ["DistillationLoss"]
