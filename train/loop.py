from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .ema import EMA


class Trainer:
    """Minimal training loop supporting grad accumulation and EMA."""

    def __init__(
        self,
        task: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        scaler: Optional[GradScaler] = None,
        ema: Optional[EMA] = None,
        grad_accum_steps: int = 1,
        max_norm: Optional[float] = None,
    ) -> None:
        self.task = task
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.ema = ema
        self.grad_accum_steps = grad_accum_steps
        self.max_norm = max_norm

    def train_epoch(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        epoch: int,
    ) -> Dict[str, float]:
        self.task.train()
        metrics = defaultdict(float)
        step = 0

        for batch_idx, batch in enumerate(dataloader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device, non_blocking=True)

            with autocast(enabled=self.scaler is not None):
                outputs = self.task(batch)
                loss = outputs["loss"] / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    if self.max_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.task.parameters(), self.max_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.task.parameters(), self.max_norm
                        )
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update(self.task.parameters())
                step += 1

            for key, value in outputs["logs"].items():
                metrics[key] += value.item()
            metrics["loss"] += outputs["loss"].item()

        num_steps = max(step, 1)
        return {k: v / num_steps for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        self.task.eval()
        metrics = defaultdict(float)
        count = 0

        for batch in dataloader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device, non_blocking=True)
            outputs = self.task(batch)
            for key, value in outputs["logs"].items():
                metrics[key] += value.item()
            metrics["loss"] += outputs["loss"].item()
            count += 1

        count = max(count, 1)
        return {k: v / count for k, v in metrics.items()}

    def save_checkpoint(self, path: str) -> None:
        state = {
            "task": self.task.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.task.load_state_dict(checkpoint["task"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
