"""Stage 1: Unified Conv3d patchification for image, video, and 3D triplane inputs."""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourDPositionEmbedding(nn.Module):
    """Learned 4D position embedding for (t, x, y, z) coordinates."""

    def __init__(self, dim: int, max_t: int = 16, max_x: int = 64,
                 max_y: int = 64, max_z: int = 64):
        super().__init__()
        self.embed_t = nn.Embedding(max_t, dim // 4)
        self.embed_x = nn.Embedding(max_x, dim // 4)
        self.embed_y = nn.Embedding(max_y, dim // 4)
        self.embed_z = nn.Embedding(max_z, dim // 4)
        self.proj = nn.Linear(dim, dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (N, 4) or (B, N, 4) long tensor."""
        t = positions[..., 0].clamp(0, self.embed_t.num_embeddings - 1)
        x = positions[..., 1].clamp(0, self.embed_x.num_embeddings - 1)
        y = positions[..., 2].clamp(0, self.embed_y.num_embeddings - 1)
        z = positions[..., 3].clamp(0, self.embed_z.num_embeddings - 1)
        pe = torch.cat([self.embed_t(t), self.embed_x(x),
                        self.embed_y(y), self.embed_z(z)], dim=-1)
        return self.proj(pe)


class PatchifyEncoder(nn.Module):
    """Unified Conv3d patchification.

    A single Conv3d handles all modalities. Images are temporally padded so the
    Conv3d output is numerically equivalent to a Conv2d (causal zero-pad).
    """

    def __init__(self, embed_dim: int = 1152, patch_size: int = 16, t_patch: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.t_patch = t_patch

        self.proj = nn.Conv3d(
            3, embed_dim,
            kernel_size=(t_patch, patch_size, patch_size),
            stride=(t_patch, patch_size, patch_size),
            bias=True,
        )
        nn.init.xavier_uniform_(self.proj.weight.reshape(embed_dim, -1).T
                                 .reshape(self.proj.weight.shape))
        nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------ #
    #  Position grid helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _image_positions(Hp: int, Wp: int, device) -> torch.Tensor:
        i = torch.arange(Hp, device=device)
        j = torch.arange(Wp, device=device)
        gi, gj = torch.meshgrid(i, j, indexing='ij')
        pos = torch.zeros(Hp * Wp, 4, dtype=torch.long, device=device)
        pos[:, 1] = gi.reshape(-1)
        pos[:, 2] = gj.reshape(-1)
        return pos  # (N, 4)

    @staticmethod
    def _video_positions(Tp: int, Hp: int, Wp: int, device) -> torch.Tensor:
        k = torch.arange(Tp, device=device)
        i = torch.arange(Hp, device=device)
        j = torch.arange(Wp, device=device)
        gk, gi, gj = torch.meshgrid(k, i, j, indexing='ij')
        pos = torch.zeros(Tp * Hp * Wp, 4, dtype=torch.long, device=device)
        pos[:, 0] = gk.reshape(-1)
        pos[:, 1] = gi.reshape(-1)
        pos[:, 2] = gj.reshape(-1)
        return pos  # (N, 4)

    # ------------------------------------------------------------------ #
    #  Modality-specific forward passes                                   #
    # ------------------------------------------------------------------ #

    def _conv3d_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Apply Conv3d to a single-frame input with causal padding.

        x: (B, 3, H, W)  →  tokens (B, D, 1, Hp, Wp)
        """
        x = x.unsqueeze(2)                      # (B, 3, 1, H, W)
        x = F.pad(x, (0, 0, 0, 0, 1, 0))       # zero-prepend 1 temporal frame
        out = self.proj(x)                       # (B, D, 1, Hp, Wp)
        return out, out.shape[3], out.shape[4]

    def forward_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 3, H, W) → tokens (B, N, D), positions (N, 4), plane_ids (N,)"""
        out, Hp, Wp = self._conv3d_image(x)
        B, D = out.shape[0], out.shape[1]
        tokens = out.permute(0, 2, 3, 4, 1).reshape(B, Hp * Wp, D)
        pos = self._image_positions(Hp, Wp, x.device)
        plane_ids = torch.full((Hp * Wp,), -1, dtype=torch.long, device=x.device)
        return tokens, pos, plane_ids

    def forward_video(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 3, T, H, W) → tokens (B, N, D), positions (N, 4), plane_ids (N,)"""
        out = self.proj(x)                       # (B, D, Tp, Hp, Wp)
        B, D, Tp, Hp, Wp = out.shape
        tokens = out.permute(0, 2, 3, 4, 1).reshape(B, Tp * Hp * Wp, D)
        pos = self._video_positions(Tp, Hp, Wp, x.device)
        plane_ids = torch.full((Tp * Hp * Wp,), -1, dtype=torch.long, device=x.device)
        return tokens, pos, plane_ids

    def forward_threed(self, planes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """planes: (B, 3, 3, S, S) — 3 planes each with 3 channels.

        Returns tokens (B, 3*Np, D), positions (3*Np, 4), plane_ids (3*Np,).
        """
        B = planes.shape[0]
        xy = planes[:, 0]   # (B, 3, S, S)
        xz = planes[:, 1]
        yz = planes[:, 2]

        def _patchify(p):
            out, Hp, Wp = self._conv3d_image(p)
            return out.permute(0, 2, 3, 4, 1).reshape(B, Hp * Wp, self.embed_dim), Hp, Wp

        tok_xy, Hp, Wp = _patchify(xy)
        tok_xz, _, _  = _patchify(xz)
        tok_yz, _, _  = _patchify(yz)
        N_plane = Hp * Wp

        device = planes.device
        i = torch.arange(Hp, device=device)
        j = torch.arange(Wp, device=device)
        gi, gj = torch.meshgrid(i, j, indexing='ij')
        fi = gi.reshape(-1)
        fj = gj.reshape(-1)
        z0 = torch.zeros(N_plane, dtype=torch.long, device=device)

        # XY: (0, x, y, 0), XZ: (0, x, 0, z), YZ: (0, 0, y, z)
        pos_xy = torch.stack([z0, fi, fj, z0], dim=-1)
        pos_xz = torch.stack([z0, fi, z0, fj], dim=-1)
        pos_yz = torch.stack([z0, z0, fi, fj], dim=-1)
        positions = torch.cat([pos_xy, pos_xz, pos_yz], dim=0)   # (3*Np, 4)

        plane_ids = torch.cat([
            torch.zeros(N_plane, dtype=torch.long, device=device),
            torch.ones(N_plane, dtype=torch.long, device=device),
            torch.full((N_plane,), 2, dtype=torch.long, device=device),
        ])  # (3*Np,)

        tokens = torch.cat([tok_xy, tok_xz, tok_yz], dim=1)  # (B, 3*Np, D)
        return tokens, positions, plane_ids

    def forward(self, x: torch.Tensor, modality: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if modality == 'image':
            return self.forward_image(x)
        elif modality == 'video':
            return self.forward_video(x)
        elif modality == 'threed':
            return self.forward_threed(x)
        else:
            raise ValueError(f"Unknown modality: {modality}")
