from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from atoken.core.sparse_tensor import SparseTensor4D
from atoken.model.encoder import ATokenEncoder
from atoken.model.heads import SemanticHead


def _compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1, 5),
) -> Dict[str, torch.Tensor]:
    num_classes = logits.size(-1)
    # Filter topk to not exceed number of classes
    topk = tuple(k for k in topk if k <= num_classes)
    if not topk:
        topk = (1,)
    
    max_k = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(max_k, dim=-1, largest=True, sorted=True)
    correct = pred.eq(targets.unsqueeze(1))

    accuracies: Dict[str, torch.Tensor] = {}
    for k in topk:
        correct_k = correct[:, :k].float().sum(dim=1)
        accuracies[f"top{k:02d}"] = correct_k.mean()
    return accuracies


class ObjectClassificationTask(nn.Module):
    """Single-label object classification task built on ATOKEN encoder."""

    def __init__(
        self,
        encoder: ATokenEncoder,
        head: SemanticHead,
        *,
        label_smoothing: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.label_smoothing = label_smoothing
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)
        targets = batch["target"].to(self.device)

        encoder_out = self.encoder(images, return_sparse=True)
        pooled = encoder_out["pooled"]
        logits = self.head(pooled)

        loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
        )

        sparse: SparseTensor4D = encoder_out["sparse"]  # type: ignore[assignment]
        avg_tokens = sparse.mask.sum(dim=1).float().mean()

        metrics = _compute_accuracy(logits, targets)
        metrics.update(
            {
                "loss": loss.detach(),
                "avg_tokens": avg_tokens.detach(),
                "logits_norm": logits.norm(dim=-1).mean().detach(),
            }
        )

        return {
            "loss": loss,
            "logs": metrics,
            "logits": logits,
            "pooled": pooled,
        }
