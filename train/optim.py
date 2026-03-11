"""Optimizer and scheduler utilities for MAVT training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    total_steps: int = 100_000
    min_lr_ratio: float = 0.1


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> AdamW:
    """AdamW with weight decay on non-bias/norm params."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW(
        [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimConfig):
    """Linear warmup → cosine decay."""
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=cfg.warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.total_steps - cfg.warmup_steps,
        eta_min=cfg.lr * cfg.min_lr_ratio,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_steps])


__all__ = ["OptimConfig", "build_optimizer", "build_scheduler"]
