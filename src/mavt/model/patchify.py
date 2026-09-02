"""Stage 1: Unified Conv3d patchification for image, video, and 3D triplane inputs."""

from __future__ import annotations
from typing import Optional, Tuple

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

    A learned 4D position embedding is added to every token before it leaves
    this module — without it the downstream self-attention blocks are
    permutation-invariant and cannot recover spatial / temporal layout.
    """

    def __init__(
        self,
        embed_dim: int = 1152,
        patch_size: int = 16,
        t_patch: int = 2,
        max_t: int = 16,
        max_x: int = 64,
        max_y: int = 64,
        max_z: int = 64,
    ):
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

        self.pos_embed = FourDPositionEmbedding(
            embed_dim, max_t=max_t, max_x=max_x, max_y=max_y, max_z=max_z,
        )
        # Dense 2-D spatial table (row, col) inherited from SigLIP2 (24x24 @384px),
        # bicubic-resized to the working grid at forward time. None until
        # init_from_siglip2() is called.
        self.pos2d: Optional[nn.Parameter] = None
        self.siglip2_inherited = False

    def init_from_siglip2(self, model_name: str = "google/siglip2-so400m-patch16-384") -> None:
        """Reproduce SigLIP2's embedding layer exactly for images.

        * Conv3d kernel: zeros on every temporal slot except the LAST one, which gets
          SigLIP2's Conv2d(16x16) weights. The image path zero-pads one frame BEFORE
          the image, so the image lands on that last slot and the output equals
          SigLIP2's patch embedding. Video tubelets start by reading their last frame.
        * pos2d: SigLIP2's learned 24x24 position table.
        * FourDPositionEmbedding: output projection zeroed -> (t,x,y,z) start at 0.
        """
        from transformers import AutoModel
        emb = AutoModel.from_pretrained(model_name).vision_model.embeddings
        conv2d = emb.patch_embedding
        D, C, tp, p, _ = self.proj.weight.shape
        if conv2d.weight.shape != (D, C, p, p):
            raise RuntimeError(f"SigLIP2 patch kernel {tuple(conv2d.weight.shape)} does not match "
                               f"Conv3d slot {(D, C, p, p)} — check embed_dim / patch_size")
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, :, tp - 1] = conv2d.weight
            self.proj.bias.copy_(conv2d.bias)
            table = emb.position_embedding.weight                      # (G*G, D)
            G = int(round(table.shape[0] ** 0.5))
            self.pos2d = nn.Parameter(table.view(1, G, G, D).permute(0, 3, 1, 2).contiguous().clone())
            nn.init.zeros_(self.pos_embed.proj.weight)
            nn.init.zeros_(self.pos_embed.proj.bias)
        self.siglip2_inherited = True

    def _pos2d_flat(self, Hp: int, Wp: int, device) -> Optional[torch.Tensor]:
        if self.pos2d is None:
            return None
        t = self.pos2d.to(device)
        if t.shape[-2:] != (Hp, Wp):
            t = F.interpolate(t, size=(Hp, Wp), mode="bicubic", align_corners=False)
        return t[0].permute(1, 2, 0).reshape(Hp * Wp, -1)

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
        tokens = tokens + self.pos_embed(pos).unsqueeze(0)
        p2 = self._pos2d_flat(Hp, Wp, x.device)
        if p2 is not None:
            tokens = tokens + p2.unsqueeze(0)
        plane_ids = torch.full((Hp * Wp,), -1, dtype=torch.long, device=x.device)
        return tokens, pos, plane_ids

    def forward_video(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, 3, T, H, W) → tokens (B, N, D), positions (N, 4), plane_ids (N,)"""
        out = self.proj(x)                       # (B, D, Tp, Hp, Wp)
        B, D, Tp, Hp, Wp = out.shape
        tokens = out.permute(0, 2, 3, 4, 1).reshape(B, Tp * Hp * Wp, D)
        pos = self._video_positions(Tp, Hp, Wp, x.device)
        tokens = tokens + self.pos_embed(pos).unsqueeze(0)
        p2 = self._pos2d_flat(Hp, Wp, x.device)
        if p2 is not None:
            tokens = tokens + p2.repeat(Tp, 1).unsqueeze(0)
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
        tokens = tokens + self.pos_embed(positions).unsqueeze(0)
        p2 = self._pos2d_flat(Hp, Wp, device)
        if p2 is not None:
            tokens = tokens + p2.repeat(3, 1).unsqueeze(0)
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
