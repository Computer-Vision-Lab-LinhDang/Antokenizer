"""I/O contracts for DDT modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch

Modality = Literal["image", "video", "3d"]


@dataclass
class PatchifyOutput:
    f_spatial: torch.Tensor      # (B, N_total, 1152) SigLIP2 features
    positions: torch.Tensor      # (B, N_total, 4) (t, x, y, z)
    raw_patches: torch.Tensor    # (B, N_total, C, p, p) raw pixels for DWT
    modality: Modality


@dataclass
class WaveletOutput:
    features: torch.Tensor       # (B, N', d_model) enriched: f_sem + gated f_freq
    positions: torch.Tensor      # (B, N', 4) selected 4D positions
    subband_energy: torch.Tensor # (B, N_total, 7) all patches (for loss)
    content_mask: torch.Tensor   # (B, N') bool — True=content, False=dynamics
    n_content: int
    n_dynamics: int


@dataclass
class EncoderOutput:
    encoded: torch.Tensor        # (B, N', d_model)
    positions_out: torch.Tensor  # (B, N', 4)


@dataclass
class LatentOutput:
    z: torch.Tensor              # (B, N', latent_dim)
    z_understand: torch.Tensor   # (B, d_understand)
    mu: torch.Tensor             # (B, N', latent_dim)
    log_var: torch.Tensor        # (B, N', latent_dim)


@dataclass
class DecoderOutput:
    reconstruction: torch.Tensor  # (B, 3, H, W) or (B, 3, T, H, W)
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


__all__ = [
    "Modality",
    "PatchifyOutput",
    "WaveletOutput",
    "EncoderOutput",
    "LatentOutput",
    "DecoderOutput",
]
