from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from transformers import CLIPImageProcessor, CLIPVisionModel
except (ImportError, OSError):
    CLIPImageProcessor = None  # type: ignore
    CLIPVisionModel = None  # type: ignore


class CLIPPerceptualLoss(nn.Module):
    """Computes perceptual alignment using CLIP vision embeddings."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        reduction: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.reduction = reduction
        self.normalize = normalize
        self._model: nn.Module | None = None
        self._processor = None

    def _lazy_init(self, device: torch.device) -> None:
        if self._model is None:
            if CLIPVisionModel is None or CLIPImageProcessor is None:
                raise ImportError(
                    "transformers is required for CLIPPerceptualLoss. "
                    "Install via `pip install transformers`."
                )
            self._model = CLIPVisionModel.from_pretrained(
                self.model_name
            ).to(device)
            self._model.eval()
            self._processor = CLIPImageProcessor.from_pretrained(
                self.model_name
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._lazy_init(pred.device)
        assert self._model is not None and self._processor is not None

        with torch.no_grad():
            target_inputs = self._processor(
                images=target, return_tensors="pt"
            ).to(pred.device)
            target_features = self._model(
                **target_inputs
            ).pooler_output  # (B, D)

        pred_inputs = self._processor(images=pred, return_tensors="pt").to(
            pred.device
        )
        pred_features = self._model(**pred_inputs).pooler_output

        if self.normalize:
            pred_features = torch.nn.functional.normalize(pred_features, dim=-1)
            target_features = torch.nn.functional.normalize(
                target_features, dim=-1
            )

        loss = 1 - (pred_features * target_features).sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss, {"clip_perc_loss": loss.detach()}


__all__ = ["CLIPPerceptualLoss"]
