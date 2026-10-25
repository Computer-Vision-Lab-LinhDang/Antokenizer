"""Masked axial RoPE on continuous coordinates c = (t, u, v, z).

Each head's dim is split evenly over the 4 axes (each axis gets an even number of dims).
An axis whose mask bit is 0 contributes angle 0, so it rotates as identity; tokens with
m = 0000 (the D queries) are not rotated at all. Angles are computed in float32 so that
t in seconds does not lose precision under bf16 autocast.

Rotation: dims (2i, 2i + 1) of a head form a pair rotated by angle i (interleaved pairing, v0).
rope_tables turns the angles into cos / sin tables once (per forward, shared by all layers) and
apply_rope computes x * cos + pair_swap(x) * sin: the v0 function without its stack + flatten copies.

coords: [..., N, 4] float (t in seconds or frame index, u and v in [0, 1] as patch centres, z unused in v0)
mask:   [..., N, 4] {0, 1}
"""
from __future__ import annotations

import torch
import torch.nn as nn


def compute_dtype(x: torch.Tensor) -> torch.dtype:
    """dtype the attention inputs (Linear outputs) will have: the autocast dtype when autocast is on
    for x's device, else x.dtype. Used to build rope tables and additive masks once per forward."""
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return x.dtype


def rope_tables(ang: torch.Tensor, dtype=None) -> tuple:
    """Angles [..., N, hd/2] (float32) -> (cos, sin) [..., N, hd]: cos_i at dims 2i and 2i + 1, sin as
    (-sin_i, +sin_i). cos / sin are taken in float32, then cast to dtype (the rotation's dtype) if given."""
    c, s = ang.cos(), ang.sin()
    cos = torch.stack((c, c), -1).flatten(-2)
    sin = torch.stack((-s, s), -1).flatten(-2)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [..., hd] with tables broadcastable to it: out_2i = x_2i cos_i - x_2i+1 sin_i, out_2i+1 = x_2i sin_i + x_2i+1 cos_i.

    3 kernels (flip, mul, addcmul). The swapped copy comes first so that the output is contiguous in x's
    logical layout even when x is a strided view (the trunk relies on that to get [B, H, T, N, hd] q / k)."""
    xs = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)       # pair swap (x_2i+1, x_2i)
    return torch.addcmul(xs * sin, x, cos)


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

    def rotate(self, x: torch.Tensor, ang) -> torch.Tensor:
        """x: [B, H, N, head_dim]; ang: angles [B, N, head_dim/2] or [N, head_dim/2], rotated in float32
        (v0 numerics; the decoder's angles carry the gradient of Δ), or a (cos, sin) pair from rope_tables
        broadcastable to x, rotated in the tables' dtype (trunk / C: tables built once per forward)."""
        if isinstance(ang, tuple):
            return apply_rope(x, *ang).to(x.dtype)
        if ang.dim() == 2:
            ang = ang.unsqueeze(0)
        cos, sin = rope_tables(ang.unsqueeze(1))                 # [B, 1, N, hd]
        return apply_rope(x.float(), cos, sin).to(x.dtype)

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
