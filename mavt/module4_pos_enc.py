"""Module 4: Spectral Position Encoding.

v1 simplification: uses freq_embed as structural positional signal.
Projects freq_embed (128 dim) to d_model and adds to tokens as positional encoding.

Full design: Laplacian eigenvectors via Lanczos + SignNet sign-invariance.
The interface is preserved so the full implementation can replace this later.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import PosEncConfig
from .types import GraphOutput, PosEncOutput, STFOutput


class SignNet(nn.Module):
    """Sign-invariant eigenvector encoder: φ(u) = [f(+u) + f(-u)] / 2."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        hidden = out_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, eigvecs: torch.Tensor) -> torch.Tensor:
        """eigvecs: (B, N, k) → (B, N, out_dim)"""
        return (self.mlp(eigvecs) + self.mlp(-eigvecs)) * 0.5


class SpectralPositionEncoding(nn.Module):
    """Combined position encoding: structural (freq/spectral) + geometric (4D RoPE).

    v1: structural signal = freq_embed projected to d_model.
    4D RoPE is applied inside attention layers (rope4d.py), not here.
    """

    def __init__(
        self,
        cfg: Optional[PosEncConfig] = None,
        d_model: int = 1152,
        d_freq: int = 128,
    ) -> None:
        super().__init__()
        self.cfg = cfg or PosEncConfig()
        self.d_model = d_model

        # v1: project freq_embed (d_freq dim) → d_model
        self.freq_to_pos = nn.Sequential(
            nn.Linear(d_freq, d_model),
            nn.LayerNorm(d_model),
        )

        # SignNet: available for future full implementation
        self.sign_net = SignNet(in_dim=self.cfg.n_eigvecs, out_dim=self.cfg.d_spectral)

    def forward(
        self,
        graph_out: GraphOutput,
        stf_out: STFOutput,
    ) -> PosEncOutput:
        """Compute position encoding.

        v1: pos_encoding = Linear(freq_embed).
        spectral_coords is a placeholder (zeros) until Laplacian eigenvectors are implemented.

        Returns:
            PosEncOutput with pos_encoding (B, N, d_model) and spectral_coords (B, N, n_eigvecs).
        """
        # v1 simplification: use freq_embed as structural positional signal
        pos_encoding = self.freq_to_pos(stf_out.freq_embed)   # (B, N, d_model)

        B, N, _ = stf_out.freq_embed.shape
        spectral_coords = torch.zeros(
            B, N, self.cfg.n_eigvecs,
            device=stf_out.freq_embed.device,
            dtype=stf_out.freq_embed.dtype,
        )

        return PosEncOutput(pos_encoding=pos_encoding, spectral_coords=spectral_coords)

    def compute_laplacian_eigvecs(self, adj: torch.Tensor) -> torch.Tensor:
        """Placeholder: Laplacian eigenvectors via Lanczos (Stage 5 TODO).

        Args:
            adj: (B, N, N) normalized adjacency

        Returns:
            (B, N, n_eigvecs) top Laplacian eigenvectors
        """
        raise NotImplementedError(
            "Stage 5: implement Lanczos Laplacian eigenvector computation"
        )


__all__ = ["SignNet", "SpectralPositionEncoding"]
