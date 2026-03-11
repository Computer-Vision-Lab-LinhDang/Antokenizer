"""Shared data structures (I/O contracts) for MAVT modules.

Each dataclass represents the output of one module and the expected
input to the next. This makes the data flow explicit and type-safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch


Modality = Literal["image", "video", "3d"]


@dataclass
class PatchifyOutput:
    """Output of Module 1: Unified 4D Patchify + SigLIP2.

    Shapes:
        f_spatial:            (B, N, 1152)  SigLIP2 patch embeddings
        positions:            (B, N, 4)     4D coordinates (t, x, y, z)
        raw_patches:          (B, N, C, p, p)  raw pixel patches for DWT
        raw_patches_temporal: (B, N_spatial, T, C, p, p)  per-frame (video)
        depth_signal:         (depth_xz, depth_yz) each (B, n_s, S_z, C) (3D)
        modality:             detected input modality
    """

    f_spatial: torch.Tensor
    positions: torch.Tensor
    raw_patches: torch.Tensor
    modality: Modality
    raw_patches_temporal: Optional[torch.Tensor] = None  # video only
    depth_signal: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # 3D only
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.f_spatial.size(0)

    @property
    def num_tokens(self) -> int:
        return self.f_spatial.size(1)


@dataclass
class STFOutput:
    """Output of Module 2: Space-Time-Frequency Transform.

    Shapes:
        freq_embed: (B, N, 128)  learnable freq embedding (add to tokens)
        freq_raw:   (B, N, 15)   raw freq profile for graph construction
    """

    freq_embed: torch.Tensor  # (B, N, d_freq_embed)
    freq_raw: torch.Tensor    # (B, N, 15)


@dataclass
class GraphOutput:
    """Output of Module 3: Frequency-Informed 4D Graph Construction.

    Shapes:
        adj:          (B, N, N)   sparse adjacency matrix (symmetrized)
        edge_weights: (B, N, K)   weights of top-K kept edges
        neighbor_idx: (B, N, K)   indices of K nearest neighbors
    """

    adj: torch.Tensor           # (B, N, N)
    edge_weights: torch.Tensor  # (B, N, K)
    neighbor_idx: torch.Tensor  # (B, N, K)


@dataclass
class PosEncOutput:
    """Output of Module 4: Spectral Position Encoding.

    pos_encoding is added to token features before the encoder.
    spectral_coords are the raw Laplacian eigenvectors (for debugging).

    Shapes:
        pos_encoding:    (B, N, d_model)  combined positional features
        spectral_coords: (B, N, n_eigvecs)  raw spectral coordinates
    """

    pos_encoding: torch.Tensor    # (B, N, d_model)
    spectral_coords: torch.Tensor # (B, N, n_eigvecs)


@dataclass
class EncoderOutput:
    """Output of Module 5: MambaVision Encoder.

    Shapes:
        encoded:       (B, N', d_enc)  encoded token sequence (downsampled)
        positions_out: (B, N', 4)      4D positions after downsampling
        memory_state:  Titans memory state (opaque, passed to next step)
    """

    encoded: torch.Tensor
    positions_out: torch.Tensor
    memory_state: Optional[Any] = None  # Titans memory state (Stage 6)


@dataclass
class LatentOutput:
    """Output of Module 6: Continuous Latent Projection.

    Shapes:
        z:           (B, N', latent_dim)   reparameterized latent
        z_understand: (B, d_understand)    pooled understanding token (CLS)
        mu:          (B, N', latent_dim)   mean of posterior
        log_var:     (B, N', latent_dim)   log variance of posterior
    """

    z: torch.Tensor
    z_understand: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor


@dataclass
class DecoderOutput:
    """Output of Module 7: Asymmetric Decoder.

    reconstruction is the primary output; its meaning depends on modality:
        image → (B, 3, H, W)
        video → (B, 3, T, H, W)
        3D    → Gaussian splat parameters or triplane

    Shapes:
        reconstruction: primary output tensor
        aux:            optional auxiliary outputs (e.g. for multi-task loss)
    """

    reconstruction: torch.Tensor
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


__all__ = [
    "Modality",
    "PatchifyOutput",
    "STFOutput",
    "GraphOutput",
    "PosEncOutput",
    "EncoderOutput",
    "LatentOutput",
    "DecoderOutput",
]
