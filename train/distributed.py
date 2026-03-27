"""Distributed training utilities for MAVT.

Provides:
  - DDP setup/teardown (supports both torchrun and mp.spawn)
  - Adaptive token-budget batch sizing
  - Gradient accumulation to maintain consistent global token budget
  - DistributedSampler-aware dataloader factory

Usage:
    python train_stage1_ddp.py --gpus 8 --steps 200000
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .navit_dataset import (
    MultiResImageDataset,
    NaViTCollator,
    SyntheticDataset,
    VideoClipDataset,
)

logger = logging.getLogger(__name__)


# ─── DDP setup ───────────────────────────────────────────────────────────────

def setup_ddp(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> tuple[int, int, torch.device]:
    """Initialize distributed process group. Returns (rank, world_size, device).

    Supports two modes:
      1. torchrun: rank/world_size from env vars (pass None)
      2. mp.spawn: rank/world_size passed explicitly
    """
    from datetime import timedelta
    timeout = timedelta(minutes=10)

    if rank is not None and world_size is not None:
        # mp.spawn mode
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, timeout=timeout,
        )
        local_rank = rank
    else:
        # torchrun mode
        dist.init_process_group(backend="nccl", timeout=timeout)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if rank == 0:
        logger.info("DDP initialized: world_size=%d", world_size)
    return rank, world_size, device


def teardown_ddp():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ─── Adaptive token-budget batch config ──────────────────────────────────────

# Token counts per resolution (patch_size=16):
#   64px  ->   16 tokens
#   128px ->   64 tokens
#   256px ->  256 tokens
#   512px -> 1024 tokens
#
# With NaViT packing (max_seq_len=4096), one packed sequence can hold:
#   256x 64px images, or 64x 128px, or 16x 256px, or 4x 512px, or a mix.
#
# Adaptive strategy: instead of fixed batch_size (number of raw samples),
# use a TOKEN BUDGET per GPU per step. The collator packs samples into
# sequences fitting max_seq_len; we control how many raw samples to feed
# the collator so the total tokens across all packed sequences approximates
# the per-GPU token budget.

# Default per-GPU token budget (can be overridden).
# 4096 * 4 = 16384 tokens/GPU -> ~4 packed sequences of 4096 each.
# With 8 GPUs: 16384 * 8 = 131072 global tokens/step.
DEFAULT_TOKEN_BUDGET_PER_GPU = 16384

# When actual batch token count varies, use gradient accumulation to
# reach a target global budget before optimizer.step().
DEFAULT_GLOBAL_TOKEN_BUDGET = 131072  # 128K tokens across all GPUs


def compute_adaptive_batch_size(
    token_budget: int,
    resolutions: list[int],
    patch_size: int = 16,
    resolution_weights: Optional[list[float]] = None,
) -> int:
    """Estimate number of raw samples to draw for a given token budget.

    Computes expected tokens per sample based on resolution distribution,
    then returns ceil(budget / expected_tokens).
    """
    if resolution_weights is None:
        resolution_weights = [1.0 / len(resolutions)] * len(resolutions)

    expected_tokens = sum(
        w * (r // patch_size) ** 2
        for r, w in zip(resolutions, resolution_weights)
    )
    return max(1, int(token_budget / max(expected_tokens, 1)))


def compute_grad_accum_steps(
    global_token_budget: int,
    per_gpu_token_budget: int,
    world_size: int,
) -> int:
    """Number of micro-batches to accumulate before optimizer.step()."""
    tokens_per_step = per_gpu_token_budget * world_size
    return max(1, global_token_budget // tokens_per_step)


# ─── Distributed dataloader factory ──────────────────────────────────────────

def build_distributed_image_dataloader(
    root: Optional[str],
    resolutions: list[int],
    raw_batch_size: int,
    max_seq_len: int = 4096,
    patch_size: int = 16,
    num_workers: int = 4,
    split: float = 0.95,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    """Build DDP-aware dataloaders for Stage 1.

    Returns (train_dl, val_dl, train_sampler).
    The caller must call train_sampler.set_epoch(epoch) each epoch.
    """
    from torch.utils.data import random_split

    if root is None:
        full_ds = SyntheticDataset(500, resolutions, "image", patch_size)
        val_ds = SyntheticDataset(50, resolutions, "image", patch_size)
    else:
        try:
            full_ds = MultiResImageDataset(root, resolutions, patch_size)
        except FileNotFoundError:
            logger.warning("No images in %r, using synthetic data.", root)
            full_ds = SyntheticDataset(500, resolutions, "image", patch_size)

        n_val = max(1, int(len(full_ds) * (1 - split)))
        full_ds, val_ds = random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(seed),
        )

    # Distributed sampler for training
    train_sampler = DistributedSampler(
        full_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=seed, drop_last=True,
    )

    collator = NaViTCollator(max_seq_len=max_seq_len)

    train_dl = DataLoader(
        full_ds,
        batch_size=raw_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=max(4, raw_batch_size // 4),
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )
    return train_dl, val_dl, train_sampler


def count_batch_tokens(batch: dict) -> int:
    """Count total tokens across all packed sequences in a batch."""
    total = 0
    for seq in batch.get("sequences", []):
        total += seq.total_tokens
    return total


def all_reduce_scalar(value: float, device: torch.device) -> float:
    """Average a scalar across all ranks."""
    if not dist.is_initialized():
        return value
    t = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


__all__ = [
    "setup_ddp",
    "teardown_ddp",
    "is_main_process",
    "DEFAULT_TOKEN_BUDGET_PER_GPU",
    "DEFAULT_GLOBAL_TOKEN_BUDGET",
    "compute_adaptive_batch_size",
    "compute_grad_accum_steps",
    "build_distributed_image_dataloader",
    "count_batch_tokens",
    "all_reduce_scalar",
]
