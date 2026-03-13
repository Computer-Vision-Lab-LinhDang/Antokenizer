"""Enhanced NaViT-style packing with mask generation.

Extends basic packing with:
- Block-diagonal attention masks
- Per-sample graph isolation masks
- Mamba scan boundaries with state resets

Masks ensure tokens from different samples never interact during encoding.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PackedBatch:
    """A batch of packed sequences ready for model forward.

    Each packed sequence contains multiple 4D token samples.
    Masks ensure cross-sample isolation.
    """

    # Token data (B, L, ...)
    tokens: torch.Tensor        # (B, L, 1152)
    positions: torch.Tensor     # (B, L, 4)
    freq_raw: torch.Tensor      # (B, L, 15)
    freq_embed: torch.Tensor    # (B, L, 128)
    sample_ids: torch.Tensor    # (B, L) — sample assignment, -1 = padding

    # Masks
    attn_mask: torch.Tensor      # (B, L, L) — block-diagonal attention
    graph_mask: torch.Tensor     # (B, L, L) — graph construction mask
    mamba_boundaries: list       # List[List[Tuple]] per batch

    # Metadata
    sample_metadata: list        # List of sample info per batch
    n_samples: list[int]         # Number of samples per packed sequence
    packing_efficiency: list[float]  # Real tokens / total tokens per sequence


class EnhancedNaViTPacker:
    """Pack 4D token samples into sequences with full mask generation.

    Extends basic packing with:
    - Greedy bin packing for efficiency
    - Block-diagonal attention mask generation
    - Graph isolation mask generation
    - Mamba scan boundary detection
    """

    def __init__(
        self,
        max_seq_len: int = 4096,
        max_samples: int = 32,
        target_efficiency: float = 0.85,
    ):
        """Initialize packer.

        Args:
            max_seq_len: Maximum tokens per packed sequence (L)
            max_samples: Maximum samples per pack (prevent too many tiny samples)
            target_efficiency: Target packing efficiency (>85% real tokens)
        """
        self.L = max_seq_len
        self.max_samples = max_samples
        self.target_efficiency = target_efficiency

    def pack(self, samples: list[dict]) -> dict:
        """Pack converted 4D samples into a single sequence.

        Args:
            samples: List of converted 4D sample dicts, each with:
                - tokens: (N_i, 1152)
                - positions: (N_i, 4)
                - freq_raw: (N_i, 15)
                - freq_embed: (N_i, 128)
                - n_tokens: int
                - modality: str
                - caption: str

        Returns:
            Dict with:
                - tokens: (L, 1152)
                - positions: (L, 4)
                - freq_raw: (L, 15)
                - freq_embed: (L, 128)
                - sample_ids: (L,)
                - n_samples: int
                - n_real_tokens: int
                - packing_efficiency: float
                - sample_metadata: list[dict]
        """
        # Sort by token count (descending) for better packing
        samples_sorted = sorted(samples, key=lambda s: s["n_tokens"], reverse=True)

        packed_tokens = []
        packed_positions = []
        packed_freq_raw = []
        packed_freq_embed = []
        packed_sample_ids = []
        sample_metadata = []

        current_len = 0
        sample_id = 0

        for sample in samples_sorted:
            N = sample["n_tokens"]

            # Check: still have room?
            if current_len + N > self.L:
                continue  # Skip this sample

            # Check: not too many samples?
            if sample_id >= self.max_samples:
                break

            # Add sample
            packed_tokens.append(sample["tokens"])
            packed_positions.append(sample["positions"])
            packed_freq_raw.append(sample["freq_raw"])
            packed_freq_embed.append(sample["freq_embed"])
            packed_sample_ids.append(
                torch.full((N,), sample_id, dtype=torch.long)
            )

            sample_metadata.append({
                "sample_id": sample_id,
                "modality": sample["modality"],
                "caption": sample.get("caption", ""),
                "n_tokens": N,
                "start_idx": current_len,
                "end_idx": current_len + N,
            })

            current_len += N
            sample_id += 1

        # ── Pad remaining ──
        remaining = self.L - current_len
        if remaining > 0:
            d_token = packed_tokens[0].shape[-1] if packed_tokens else 1152
            d_freq = packed_freq_raw[0].shape[-1] if packed_freq_raw else 15
            d_femb = packed_freq_embed[0].shape[-1] if packed_freq_embed else 128

            packed_tokens.append(torch.zeros(remaining, d_token))
            packed_positions.append(torch.zeros(remaining, 4))
            packed_freq_raw.append(torch.zeros(remaining, d_freq))
            packed_freq_embed.append(torch.zeros(remaining, d_femb))
            packed_sample_ids.append(
                torch.full((remaining,), -1, dtype=torch.long)  # -1 = padding
            )

        # ── Concatenate ──
        result = {
            "tokens":      torch.cat(packed_tokens, dim=0),      # (L, 1152)
            "positions":   torch.cat(packed_positions, dim=0),    # (L, 4)
            "freq_raw":    torch.cat(packed_freq_raw, dim=0),     # (L, 15)
            "freq_embed":  torch.cat(packed_freq_embed, dim=0),   # (L, 128)
            "sample_ids":  torch.cat(packed_sample_ids, dim=0),   # (L,)
            "n_samples":   sample_id,
            "n_real_tokens": current_len,
            "packing_efficiency": current_len / self.L,
            "sample_metadata": sample_metadata,
        }

        return result

    def pack_batch(self, all_samples: list[dict], batch_size: int) -> list[dict]:
        """Pack multiple sequences for a batch.

        Args:
            all_samples: Many converted samples
            batch_size: Number of packed sequences (B)

        Returns:
            List of B packed dicts
        """
        # Shuffle samples
        samples = all_samples.copy()
        random.shuffle(samples)

        packs = []
        remaining = samples

        for _ in range(batch_size):
            if not remaining:
                break

            # Greedily select samples for this pack
            pack_samples = []
            pack_tokens = 0
            still_remaining = []

            for sample in remaining:
                n_tok = sample.get("n_tokens", 0)
                if pack_tokens + n_tok <= self.L and len(pack_samples) < self.max_samples:
                    pack_samples.append(sample)
                    pack_tokens += n_tok
                else:
                    still_remaining.append(sample)

            remaining = still_remaining

            if pack_samples:
                packs.append(self.pack(pack_samples))

        return packs

    def create_batch_tensors(self, packs: list[dict], device: torch.device) -> PackedBatch:
        """Convert packed dicts to batch tensors with masks.

        Args:
            packs: List of packed sequence dicts from pack_batch()
            device: Target device (cuda/cpu)

        Returns:
            PackedBatch with all tensors and masks ready for model
        """
        B = len(packs)
        L = packs[0]["tokens"].shape[0]  # Should all be same (max_seq_len)

        # Stack into batch tensors
        tokens = torch.stack([p["tokens"] for p in packs]).to(device)
        positions = torch.stack([p["positions"] for p in packs]).to(device)
        freq_raw = torch.stack([p["freq_raw"] for p in packs]).to(device)
        freq_embed = torch.stack([p["freq_embed"] for p in packs]).to(device)
        sample_ids = torch.stack([p["sample_ids"] for p in packs]).to(device)

        # Build masks
        attn_mask = torch.stack([
            build_attention_mask(sample_ids[b])
            for b in range(B)
        ]).to(device)  # (B, L, L)

        graph_mask = torch.stack([
            build_graph_mask(sample_ids[b])
            for b in range(B)
        ]).to(device)  # (B, L, L)

        mamba_boundaries = [
            build_mamba_boundaries(sample_ids[b])
            for b in range(B)
        ]

        # Gather metadata
        sample_metadata = [p["sample_metadata"] for p in packs]
        n_samples = [p["n_samples"] for p in packs]
        packing_efficiency = [p["packing_efficiency"] for p in packs]

        return PackedBatch(
            tokens=tokens,
            positions=positions,
            freq_raw=freq_raw,
            freq_embed=freq_embed,
            sample_ids=sample_ids,
            attn_mask=attn_mask,
            graph_mask=graph_mask,
            mamba_boundaries=mamba_boundaries,
            sample_metadata=sample_metadata,
            n_samples=n_samples,
            packing_efficiency=packing_efficiency,
        )


# ============================================================================
# Mask Generation Functions
# ============================================================================

def build_attention_mask(sample_ids: torch.Tensor) -> torch.Tensor:
    """Build block-diagonal attention mask.

    Tokens can ONLY attend within their own sample.

    Args:
        sample_ids: (L,) — sample index, -1 = padding

    Returns:
        (L, L) bool — True = can attend

    Example:
        sample_ids = [0,0,0, 1,1,1,1, 2,2, -1,-1]

              0 0 0  1 1 1 1  2 2  - -
        0   [ 1 1 1  0 0 0 0  0 0  0 0 ]
        0   [ 1 1 1  0 0 0 0  0 0  0 0 ]
        0   [ 1 1 1  0 0 0 0  0 0  0 0 ]
        1   [ 0 0 0  1 1 1 1  0 0  0 0 ]
        1   [ 0 0 0  1 1 1 1  0 0  0 0 ]
        1   [ 0 0 0  1 1 1 1  0 0  0 0 ]
        1   [ 0 0 0  1 1 1 1  0 0  0 0 ]
        2   [ 0 0 0  0 0 0 0  1 1  0 0 ]
        2   [ 0 0 0  0 0 0 0  1 1  0 0 ]
        -   [ 0 0 0  0 0 0 0  0 0  0 0 ]
        -   [ 0 0 0  0 0 0 0  0 0  0 0 ]
    """
    valid = (sample_ids >= 0)  # (L,)
    same_sample = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1))  # (L, L)

    # Can attend if: same sample AND both valid
    mask = same_sample & valid.unsqueeze(0) & valid.unsqueeze(1)

    return mask


def build_graph_mask(sample_ids: torch.Tensor) -> torch.Tensor:
    """Build graph construction mask.

    Affinity between tokens from different samples = -inf → never neighbors.

    Args:
        sample_ids: (L,) — sample index, -1 = padding

    Returns:
        (L, L) float — 0 = same sample, -inf = different/padding
    """
    valid = (sample_ids >= 0)
    same_sample = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1))
    both_valid = valid.unsqueeze(0) & valid.unsqueeze(1)

    # Same sample + both valid → 0, else -inf
    mask = torch.where(
        same_sample & both_valid,
        torch.tensor(0.0),
        torch.tensor(float('-inf'))
    )

    return mask


def build_mamba_boundaries(sample_ids: torch.Tensor) -> list[tuple]:
    """Build Mamba scan boundaries.

    Mamba needs to scan PER SAMPLE, resetting state between samples.

    Args:
        sample_ids: (L,) — sample index, -1 = padding

    Returns:
        List of (start_idx, end_idx, sample_id) tuples

    Example:
        sample_ids = [0,0,0, 1,1,1,1, 2,2, -1,-1]
        → [(0, 3, 0), (3, 7, 1), (7, 9, 2)]
    """
    boundaries = []
    current_id = -1
    start = 0

    for i in range(len(sample_ids)):
        sid = sample_ids[i].item()

        if sid < 0:  # padding
            if current_id >= 0:
                boundaries.append((start, i, current_id))
                current_id = -1
            continue

        if sid != current_id:
            if current_id >= 0:
                boundaries.append((start, i, current_id))
            start = i
            current_id = sid

    if current_id >= 0:
        boundaries.append((start, len(sample_ids), current_id))

    return boundaries


# ============================================================================
# Collate Function
# ============================================================================

def collate_4d_batch(batch_list: list[dict]) -> PackedBatch:
    """Collate function for DataLoader with 4D converted samples.

    Takes a list of converted 4D samples from DataLoader and packs them
    into a batch with masks.

    Args:
        batch_list: List of converted sample dicts from Unified4DConverter

    Returns:
        PackedBatch ready for model forward
    """
    packer = EnhancedNaViTPacker()

    # Pack into sequences (one pack per batch in this simple version)
    packs = packer.pack_batch(batch_list, batch_size=1)

    # Convert to batch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = packer.create_batch_tensors(packs, device)

    return batch


__all__ = [
    "PackedBatch",
    "EnhancedNaViTPacker",
    "build_attention_mask",
    "build_graph_mask",
    "build_mamba_boundaries",
    "collate_4d_batch",
]
