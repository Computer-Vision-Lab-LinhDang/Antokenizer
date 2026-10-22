"""Camera helpers for posed views (3D objects): Plücker rays per patch and relative poses.

Convention: c2w is a 4x4 camera-to-world matrix in OpenCV axes (x right, y down, z forward);
K is the 3x3 pixel intrinsic matrix of the image as fed to the model (after resize/crop).
Loaders convert other conventions (e.g. Blender/OpenGL: flip y and z) before calling these.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def patch_rays(c2w: torch.Tensor, K: torch.Tensor, image_size: int, grid: int) -> torch.Tensor:
    """c2w [..., 4, 4], K [..., 3, 3] -> Plücker rays (d, o x d) at patch centres [..., grid*grid, 6]."""
    dev, dt = c2w.device, torch.float32
    r = (torch.arange(grid, device=dev, dtype=dt) + 0.5) * (image_size / grid)
    v, u = torch.meshgrid(r, r, indexing="ij")
    pix = torch.stack([u, v, torch.ones_like(u)], -1).reshape(-1, 3)                 # [N, 3]
    Kinv = torch.linalg.inv(K.float())
    d_cam = torch.einsum("...ij,nj->...ni", Kinv, pix)                               # [..., N, 3]
    R, t = c2w[..., :3, :3].float(), c2w[..., :3, 3].float()
    d = torch.einsum("...ij,...nj->...ni", R, d_cam)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    o = t.unsqueeze(-2).expand_as(d)
    return torch.cat([d, torch.cross(o, d, dim=-1)], -1)


def relative_pose(c2w_ref: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Pose of each view in the reference view's frame: inv(c2w_ref) @ c2w, flattened [..., 12]."""
    rel = torch.linalg.inv(c2w_ref.float()) @ c2w.float()
    return rel[..., :3, :4].flatten(-2)


class ZeroLinear(nn.Linear):
    """Linear layer that starts at zero, so a new conditioning input changes nothing at init."""

    def __init__(self, i: int, o: int):
        super().__init__(i, o)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
