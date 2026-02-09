from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from atoken.losses.distill import DistillationLoss
from atoken.model.encoder import ATokenEncoder
from atoken.model.heads import SemanticHead


class ImageSemanticTask(nn.Module):
    """Semantic understanding task with distillation support."""

    def __init__(
        self,
        encoder: ATokenEncoder,
        head: SemanticHead,
        loss_fn: DistillationLoss,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.loss_fn = loss_fn
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)
        targets = batch["target"].to(self.device)

        encoder_out = self.encoder(images, return_sparse=False)
        pooled = encoder_out["pooled"]
        logits = self.head(pooled)

        loss, logs = self.loss_fn(logits, targets)
        logs["logits_norm"] = logits.norm(dim=-1).mean().detach()
        return {"loss": loss, "logs": logs, "logits": logits, "pooled": pooled}
