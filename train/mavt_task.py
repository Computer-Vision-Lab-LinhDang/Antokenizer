"""MAVT training task — wraps MAVTokenizer + MAVTLoss for the Trainer.

Supports two batch formats:
  1. Legacy tensor format:  {"image": (B,C,H,W)} or {"video": ...}
  2. NaViT packed format:   {"sequences": List[PackedSequence], "n_sequences": int}

The packed format processes each sample independently through the full pipeline
and averages losses across all samples in the batch.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from atoken.losses.mavt_loss import LossWeights, MAVTLoss
from atoken.mavt.config import MAVTConfig
from atoken.mavt.tokenizer import MAVTokenizer
from atoken.mavt.types import Modality

logger = logging.getLogger(__name__)


class MAVTTask(nn.Module):
    """Training task: takes a batch dict, returns {"loss", "logs"}.

    Legacy batch keys:
        "image":    (B, C, H, W)
        "video":    (B, C, T, H, W)
        "triplane": (B, 3, C, S, S)

    NaViT packed batch keys:
        "sequences":   List[PackedSequence]   — from NaViTCollator
        "n_sequences": int
    """

    def __init__(
        self,
        model_cfg: Optional[MAVTConfig] = None,
        loss_weights: Optional[LossWeights] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = MAVTokenizer(model_cfg)
        self.loss_fn = MAVTLoss(loss_weights)

    # ─── forward ─────────────────────────────────────────────────────────────

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        if "sequences" in batch:
            return self._forward_packed(batch["sequences"])
        # Legacy: single-tensor batch
        x, modality = self._extract_input(batch)
        dec, lat = self.tokenizer(x, modality)
        return self.loss_fn(
            recon=dec.reconstruction,
            target=x,
            mu=lat.mu,
            log_var=lat.log_var,
        )

    # ─── NaViT packed forward ─────────────────────────────────────────────────

    def _forward_packed(self, sequences: list) -> Dict[str, torch.Tensor]:
        """Process a list of PackedSequences.

        Each PackedSequence contains multiple samples packed together.
        Samples are processed independently; losses are averaged.

        Args:
            sequences: List[PackedSequence] from NaViTCollator.

        Returns:
            {"loss": Tensor, "logs": dict}
        """
        device = next(self.parameters()).device
        all_losses: list[torch.Tensor] = []
        all_logs: dict[str, list[torch.Tensor]] = defaultdict(list)

        for seq in sequences:
            samples = seq.samples if hasattr(seq, "samples") else seq
            for sample in samples:
                x = sample["data"]
                if x.dim() == 3:
                    x = x.unsqueeze(0)        # → (1, C, H, W)
                x = x.to(device)
                modality: Modality = sample.get("modality", "image")

                try:
                    dec, lat = self.tokenizer(x, modality)
                    result = self.loss_fn(
                        recon=dec.reconstruction,
                        target=x,
                        mu=lat.mu,
                        log_var=lat.log_var,
                    )
                    all_losses.append(result["loss"])
                    for k, v in result["logs"].items():
                        all_logs[k].append(v)
                except Exception as e:
                    logger.warning("Skipping sample (modality=%s, shape=%s): %s",
                                   modality, tuple(x.shape), e)

        if not all_losses:
            dummy = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss": dummy, "logs": {"total_loss": dummy.detach()}}

        loss = torch.stack(all_losses).mean()
        logs = {k: torch.stack(v).mean() for k, v in all_logs.items()}
        logs["total_loss"] = loss.detach()
        logs["n_samples"] = torch.tensor(float(len(all_losses)), device=device)
        return {"loss": loss, "logs": logs}

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _extract_input(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[Modality]]:
        if "image" in batch:
            return batch["image"], "image"
        if "video" in batch:
            return batch["video"], "video"
        if "triplane" in batch:
            return batch["triplane"], "3d"
        raise KeyError(
            f"Batch must contain 'image', 'video', or 'triplane'. Got: {list(batch.keys())}"
        )


__all__ = ["MAVTTask"]
