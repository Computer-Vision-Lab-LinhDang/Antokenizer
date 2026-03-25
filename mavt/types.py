"""Shared data structures (I/O contracts) for MAVT modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch

Modality = Literal["image", "video", "3d"]


@dataclass
class PatchifyOutput:
    """Output of Module 1: Unified 4D Patchify + SigLIP2."""

    f_spatial: torch.Tensor       # (B, N, 1152)
    positions: torch.Tensor       # (B, N, 4)
    raw_patches: torch.Tensor     # (B, N, C, p, p)
    modality: Modality
    raw_patches_temporal: Optional[torch.Tensor] = None
    depth_signal: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.f_spatial.size(0)

    @property
    def num_tokens(self) -> int:
        return self.f_spatial.size(1)


@dataclass
class EncoderOutput:
    """Output of Module 5: Transformer Encoder."""

    encoded: torch.Tensor         # (B, N', d_enc)
    positions_out: torch.Tensor   # (B, N', 4)


@dataclass
class LatentOutput:
    """Output of Module 6: Continuous Latent Projection."""

    z: torch.Tensor               # (B, N', latent_dim)
    z_understand: torch.Tensor    # (B, d_understand)
    mu: torch.Tensor
    log_var: torch.Tensor


@dataclass
class DecoderOutput:
    """Output of Module 7: Asymmetric Decoder."""

    reconstruction: torch.Tensor
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


__all__ = [
    "Modality",
    "PatchifyOutput",
    "EncoderOutput",
    "LatentOutput",
    "DecoderOutput",
]
