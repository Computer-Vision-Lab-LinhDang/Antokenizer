"""Masked axial RoPE on continuous coordinates c = (t, u, v, z).

Each head's dim is split evenly over the 4 axes (each axis gets an even number of dims).
An axis whose mask bit is 0 contributes angle 0, so it rotates as identity; tokens with
m = 0000 (the D queries) are not rotated at all. Angles are computed in float32 so that
t in seconds does not lose precision under bf16 autocast.

coords: [..., N, 4] float (t in seconds or frame index, u and v in [0, 1] as patch centres, z unused in v0)
mask:   [..., N, 4] {0, 1}
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AxialRoPE(nn.Module):
    def __init__(self, head_dim: int, n_axes: int = 4, base: float = 100.0,
                 axis_scale: tuple = (1.0, 16.0, 16.0, 16.0)):
        super().__init__()
        assert head_dim % (2 * n_axes) == 0, "head_dim must split into an even number of dims per axis"
        self.head_dim, self.n_axes = head_dim, n_axes
        self.per_axis = head_dim // n_axes                       # dims per axis (even)
        inv = base ** (-torch.arange(0, self.per_axis, 2, dtype=torch.float32) / self.per_axis)
        self.register_buffer("inv_freq", inv, persistent=False)  # [per_axis/2]
        # u, v in [0, 1]; scaling by the grid size makes one patch step ~ one radian at the top band
        self.register_buffer("axis_scale", torch.tensor(axis_scale, dtype=torch.float32), persistent=False)

    def angles(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[..., N, 4] -> [..., N, head_dim/2] angles (float32)."""
        c = coords.float() * self.axis_scale * mask.float()
        ang = c.unsqueeze(-1) * self.inv_freq                    # [..., N, 4, per_axis/2]
        return ang.flatten(-2)

    def rotate(self, x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
        """x: [B, H, N, head_dim]; ang: [B, N, head_dim/2] or [N, head_dim/2]."""
        if ang.dim() == 2:
            ang = ang.unsqueeze(0)
        cos, sin = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)  # [B, 1, N, hd/2]
        xf = x.float()
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
        return out.to(x.dtype)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.rotate(x, self.angles(coords, mask))


def grid_coords(grid: int, t: float = 0.0, device=None) -> torch.Tensor:
    """[grid*grid, 4] patch-centre coords (t, u, v, 0), u = column, v = row, both in [0, 1]."""
    r = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid
    v, u = torch.meshgrid(r, r, indexing="ij")
    tt = torch.full_like(u, float(t))
    return torch.stack([tt, u, v, torch.zeros_like(u)], -1).reshape(-1, 4)


MASK_IMAGE = (0, 1, 1, 0)
MASK_VIDEO = (1, 1, 1, 0)
MASK_NONE = (0, 0, 0, 0)


def modality_mask(modality: str, n: int, device=None) -> torch.Tensor:
    bits = {"image": MASK_IMAGE, "video": MASK_VIDEO, "none": MASK_NONE}[modality]
    return torch.tensor(bits, dtype=torch.float32, device=device).expand(n, 4).clone()
