"""NaViT-style sequence packing utilities.

Core idea: instead of padding all images to the same size, pack multiple
images of different resolutions into a single sequence up to max_seq_len tokens.
A block-diagonal attention mask prevents cross-sample attention in the encoder.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch


def compute_token_count(H: int, W: int, patch_size: int = 16) -> int:
    """Number of tokens for an image of shape (H, W)."""
    return (H // patch_size) * (W // patch_size)


def compute_video_token_count(
    H: int, W: int, T: int, patch_size: int = 16, tau: int = 2
) -> int:
    """Number of tokens for a video clip of shape (T, H, W)."""
    return (T // tau) * (H // patch_size) * (W // patch_size)


@dataclass
class PackedSequence:
    """One packed sequence: multiple samples whose token counts sum ≤ max_seq_len."""

    samples: list[dict[str, Any]] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)

    def add(self, sample: dict[str, Any], n_tokens: int) -> None:
        self.samples.append(sample)
        self.token_counts.append(n_tokens)

    def __len__(self) -> int:
        return len(self.samples)


def greedy_pack(
    samples: list[dict[str, Any]],
    max_seq_len: int,
    shuffle: bool = True,
) -> list[PackedSequence]:
    """Pack samples into sequences using greedy first-fit decreasing.

    Args:
        samples:     Sample dicts, each with an "n_tokens" key.
        max_seq_len: Maximum token budget per packed sequence.
        shuffle:     Shuffle before packing to improve diversity.

    Returns:
        List of PackedSequence objects.
    """
    if shuffle:
        samples = samples.copy()
        random.shuffle(samples)

    sequences: list[PackedSequence] = []
    current = PackedSequence()

    for s in samples:
        n = s.get("n_tokens", 0)
        if n <= 0 or n > max_seq_len:
            continue  # skip empty or oversized samples
        if current.total_tokens + n > max_seq_len and current.samples:
            sequences.append(current)
            current = PackedSequence()
        current.add(s, n)

    if current.samples:
        sequences.append(current)

    return sequences


def build_block_attn_mask(
    token_counts: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Build block-diagonal boolean attention mask for a packed sequence.

    Token groups defined by token_counts are only allowed to attend within
    their own group. Used in the encoder to prevent cross-sample attention.

    Args:
        token_counts: Number of tokens per sample in the packed sequence.
        device:       Target device.

    Returns:
        (L, L) bool tensor — True means the position CAN attend to the key.
    """
    L = sum(token_counts)
    mask = torch.zeros(L, L, dtype=torch.bool, device=device)
    start = 0
    for c in token_counts:
        mask[start : start + c, start : start + c] = True
        start += c
    return mask


__all__ = [
    "PackedSequence",
    "compute_token_count",
    "compute_video_token_count",
    "greedy_pack",
    "build_block_attn_mask",
]
