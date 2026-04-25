"""Stage 5: Two decoder heads on the shared latent z.

  AsymmetricDecoder     — z → pixel reconstruction (recon head)
  UnderstandingDecoder  — z / MRL prefixes of z → semantic vector aligned
                          with vision teacher (understanding head)

Both heads sit downstream of z so the latent must preserve enough information
for pixel-faithful reconstruction AND semantic alignment simultaneously.

UnifiedDetailExpander    — cross-attends from target positions into z
PixelShuffleCNNDecoder   — 4-stage PixelShuffle CNN (16× spatial upsample)
"""

from __future__ import annotations
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mavt.model.transformer import StandardTransformerBlock


# --------------------------------------------------------------------------- #
#  Position-encoded query generator                                             #
# --------------------------------------------------------------------------- #

class FourDQueryEncoding(nn.Module):
    """Learnable 4D position encoding for decoder queries."""

    def __init__(self, dim: int, max_t: int = 16, max_x: int = 64,
                 max_y: int = 64, max_z: int = 64):
        super().__init__()
        self.embed_t = nn.Embedding(max_t, dim // 4)
        self.embed_x = nn.Embedding(max_x, dim // 4)
        self.embed_y = nn.Embedding(max_y, dim // 4)
        self.embed_z = nn.Embedding(max_z, dim // 4)
        self.proj = nn.Linear(dim, dim)

    def forward(self, positions: torch.Tensor, B: int) -> torch.Tensor:
        """positions: (N, 4) → queries (B, N, dim)"""
        t = positions[:, 0].clamp(0, self.embed_t.num_embeddings - 1)
        x = positions[:, 1].clamp(0, self.embed_x.num_embeddings - 1)
        y = positions[:, 2].clamp(0, self.embed_y.num_embeddings - 1)
        z = positions[:, 3].clamp(0, self.embed_z.num_embeddings - 1)
        pe = torch.cat([self.embed_t(t), self.embed_x(x),
                        self.embed_y(y), self.embed_z(z)], dim=-1)  # (N, dim)
        pe = self.proj(pe).unsqueeze(0).expand(B, -1, -1)           # (B, N, dim)
        return pe


# --------------------------------------------------------------------------- #
#  UnifiedDetailExpander                                                        #
# --------------------------------------------------------------------------- #

class UnifiedDetailExpander(nn.Module):
    """Inverts C-D Split: cross-attends from target grid to compressed z.

    Uses 2 cross-attention layers from position-encoded queries into the
    compressed VAE latent representation.
    """

    def __init__(self, latent_dim: int = 32, dec_dim: int = 768,
                 num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.query_enc = FourDQueryEncoding(dec_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm_q':  nn.LayerNorm(dec_dim),
                'norm_ff': nn.LayerNorm(dec_dim),
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=dec_dim, num_heads=num_heads,
                    kdim=latent_dim, vdim=latent_dim,
                    batch_first=True, bias=True,
                ),
                'ff': nn.Sequential(
                    nn.Linear(dec_dim, dec_dim * 4),
                    nn.GELU(),
                    nn.Linear(dec_dim * 4, dec_dim),
                ),
            })
            for _ in range(num_layers)
        ])

    def forward(self, z: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        """
        z               : (B, N_c+N_d, latent_dim)
        target_positions: (N_target, 4)

        Returns expanded : (B, N_target, dec_dim)
        """
        B = z.shape[0]
        q = self.query_enc(target_positions, B)   # (B, N_target, dec_dim)
        kv = self.norm_kv(z)                       # (B, N_z, latent_dim)

        for layer in self.layers:
            q_n = layer['norm_q'](q)
            out, _ = layer['cross_attn'](q_n, kv, kv)
            q = q + out
            q = q + layer['ff'](layer['norm_ff'](q))

        return q   # (B, N_target, dec_dim)


# --------------------------------------------------------------------------- #
#  PixelShuffle CNN decoder                                                     #
# --------------------------------------------------------------------------- #

class PixelShuffleCNNDecoder(nn.Module):
    """4-stage PixelShuffle CNN: 16× spatial upsampling.

    Input : (B, in_channels, H_grid, W_grid)   e.g. (B, 768, 16, 16)
    Output: (B, 3, H_out, W_out)               e.g. (B, 3, 256, 256)
    """

    def __init__(self, in_channels: int = 768):
        super().__init__()
        # Each block: Conv2d → PixelShuffle(2) → GELU  ⇒ 2× spatial
        self.up = nn.Sequential(
            # Block 1: 768 → 512·4, ÷4 ch after PS → 512, ×2 spatial
            nn.Conv2d(in_channels, 512 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            # Block 2: 512 → 256·4, → 256, ×2
            nn.Conv2d(512, 256 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            # Block 3: 256 → 128·4, → 128, ×2
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            # Block 4: 128 → 3·4, → 3, ×2  (output layer, no GELU)
            nn.Conv2d(128, 3 * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# --------------------------------------------------------------------------- #
#  AsymmetricDecoder                                                            #
# --------------------------------------------------------------------------- #

class AsymmetricDecoder(nn.Module):
    """Full asymmetric decoder: expander → self-attention blocks → CNN upsample.

    Handles all three modalities (image, video, 3D) with shared weights.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        dec_dim: int = 768,
        num_attn_blocks: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.expander = UnifiedDetailExpander(latent_dim, dec_dim, num_heads=num_heads)
        self.self_attn_blocks = nn.ModuleList([
            StandardTransformerBlock(dec_dim, num_heads, mlp_ratio)
            for _ in range(num_attn_blocks)
        ])
        self.cnn = PixelShuffleCNNDecoder(in_channels=dec_dim)

    def _decode_grid(
        self,
        z: torch.Tensor,           # (B, Nz, latent_dim)
        positions: torch.Tensor,   # (N_grid, 4)
        H_grid: int,
        W_grid: int,
    ) -> torch.Tensor:
        """Decode a single 2D grid → (B, 3, H_out, W_out)."""
        B = z.shape[0]
        expanded = self.expander(z, positions)  # (B, N_grid, dec_dim)
        for blk in self.self_attn_blocks:
            expanded = blk(expanded)

        # Reshape to 2D spatial grid
        feat = expanded.transpose(1, 2).reshape(B, -1, H_grid, W_grid)
        return self.cnn(feat)   # (B, 3, 16*H_grid, 16*W_grid)

    def forward(
        self,
        z: torch.Tensor,                # (B, Nz, latent_dim)
        target_positions: torch.Tensor, # (N_target, 4)
        modality: str,
        grid_shape: tuple,              # (H_grid, W_grid) for image/3D; (Tp, H, W) for video
    ) -> torch.Tensor:
        """Decode z → reconstructed pixel-space tensor."""
        if modality == 'image':
            H, W = grid_shape
            out = self._decode_grid(z, target_positions, H, W)
            return out  # (B, 3, H_out, W_out)

        elif modality == 'video':
            Tp, Hg, Wg = grid_shape
            N_frame = Hg * Wg
            # Decode per-frame with shared weights
            frames = []
            for t in range(Tp):
                pos_t = target_positions[t * N_frame:(t + 1) * N_frame]
                frame = self._decode_grid(z, pos_t, Hg, Wg)  # (B, 3, H, W)
                frames.append(frame)
            return torch.stack(frames, dim=2)   # (B, 3, Tp, H, W)

        elif modality == 'threed':
            N_plane = target_positions.shape[0] // 3
            Hg, Wg = grid_shape
            planes_out = []
            for p in range(3):
                pos_p = target_positions[p * N_plane:(p + 1) * N_plane]
                plane = self._decode_grid(z, pos_p, Hg, Wg)  # (B, 3, H, W)
                planes_out.append(plane)
            return torch.stack(planes_out, dim=1)  # (B, 3, 3, H, W)

        else:
            raise ValueError(f"Unknown modality: {modality}")


# --------------------------------------------------------------------------- #
#  UnderstandingDecoder                                                         #
# --------------------------------------------------------------------------- #

class UnderstandingDecoder(nn.Module):
    """Decode latent z → global semantic vector aligned with vision teacher.

    Mirror of AsymmetricDecoder but for the understanding output. Operating on
    z (the bottleneck) — not on the encoder's pre-VAE features — forces the
    latent to preserve enough semantic information to recover a SigLIP-aligned
    representation. This is the "understanding head" of the unified tokenizer.

    Architecture:
        z (B, Nz, latent_dim)
          → Linear(latent_dim → dec_dim) + LayerNorm
          → N self-attention blocks (refine token interactions)
          → attention pool with single learnable query → (B, dec_dim)
          → LayerNorm + Linear(dec_dim → semantic_dim)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        dec_dim: int = 768,
        semantic_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, dec_dim)
        self.norm_in = nn.LayerNorm(dec_dim)

        self.self_attn_blocks = nn.ModuleList([
            StandardTransformerBlock(dec_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.query = nn.Parameter(torch.randn(1, 1, dec_dim) * (dec_dim ** -0.5))
        self.pool = nn.MultiheadAttention(
            embed_dim=dec_dim, num_heads=num_heads,
            batch_first=True, bias=True,
        )
        self.norm_out = nn.LayerNorm(dec_dim)
        self.proj = nn.Linear(dec_dim, semantic_dim)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, Nz, latent_dim) -> semantic: (B, semantic_dim)."""
        x = self.norm_in(self.in_proj(z))
        for blk in self.self_attn_blocks:
            x = blk(x)
        B = x.shape[0]
        q = self.query.expand(B, 1, -1)
        pooled, _ = self.pool(q, x, x)
        return self.proj(self.norm_out(pooled.squeeze(1)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the full latent token sequence."""
        return self._decode(z)

    def forward_mrl(self, z: torch.Tensor, prefixes: Sequence[int]) -> Dict[int, torch.Tensor]:
        """Decode Matryoshka channel prefixes of the same latent sequence.

        Each prefix keeps the first ``p`` latent channels and zero-pads the
        rest, so every output is read through the same projection, transformer
        blocks, and learned pooling query.
        """
        latent_dim = z.shape[-1]
        outputs: Dict[int, torch.Tensor] = {}
        for prefix in prefixes:
            z_prefix = z if prefix == latent_dim else F.pad(z[..., :prefix], (0, latent_dim - prefix))
            outputs[prefix] = self._decode(z_prefix)
        return outputs
