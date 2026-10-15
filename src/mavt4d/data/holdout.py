"""Held-out LiDAR points: a sweep's mask depends only on f"{source}/{lidar_key}", so a sweep shared by
several samples (overlapping M2 windows) always hides the same points and cannot leak into input."""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace

import numpy as np
import torch
from torch import Tensor

from mavt4d.data.sample import SceneSample


def heldout_mask(n_points: int, frac: float, key: str) -> Tensor:
    """(n_points,) bool with exactly round(frac * n_points) True, seeded by sha1(key) only.

    Uses ``random.Random(seed).random()``, the one stream Python documents as stable across versions.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac: expected in [0, 1], got {frac}")
    rng = random.Random(int.from_bytes(hashlib.sha1(key.encode("utf-8")).digest(), "big"))
    scores = np.array([rng.random() for _ in range(n_points)])  # numpy sort: immune to torch thread contention
    mask = torch.zeros(n_points, dtype=torch.bool)
    mask[torch.from_numpy(np.argsort(scores, kind="stable")[: round(frac * n_points)])] = True
    return mask


def apply_holdout(sample: SceneSample, frac: float = 0.3) -> SceneSample:
    """Copy of ``sample`` with ``heldout`` set for every sweep; the input sample is left untouched."""
    if sample.lidar_points is None:
        raise ValueError("lidar_points: sample has no LiDAR to hold out")
    if sample.lidar_keys is None:
        raise ValueError("lidar_keys: required to key the held-out mask")
    masks = [heldout_mask(p.shape[0], frac, f"{sample.source}/{k}").to(p.device)
             for p, k in zip(sample.lidar_points, sample.lidar_keys, strict=True)]
    return replace(sample, heldout=masks)
