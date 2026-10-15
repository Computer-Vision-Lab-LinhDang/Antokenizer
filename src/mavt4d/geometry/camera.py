"""Pinhole camera geometry.

Pixel coordinates are continuous (u = x right, v = y down) with pixel i covering [i, i + 1), so a
patch (r, c) covers u in [c·patch, (c+1)·patch) and its centre is ((c + 0.5)·patch, (r + 0.5)·patch).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def transform_points(T: Tensor, p: Tensor) -> Tensor:
    """Apply rigid transforms ``T`` (..., 4, 4) to points ``p`` (..., N, 3)."""
    return p @ T[..., :3, :3].transpose(-1, -2) + T[..., None, :3, 3]


def invert_se3(T: Tensor) -> Tensor:
    """Inverse of rigid transforms (..., 4, 4) via R^T, -R^T t."""
    R_t = T[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = R_t
    out[..., :3, 3] = -(R_t @ T[..., :3, 3:]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def project(p_world: Tensor, K: Tensor, cam_to_world: Tensor) -> tuple[Tensor, Tensor]:
    """World points (N, 3) -> pixels (N, 2) and depth (N,) = camera-frame z.

    ``uv`` is meaningless where ``depth <= 0``; callers must mask on depth.
    """
    p = transform_points(invert_se3(cam_to_world), p_world)
    depth = p[:, 2]
    return (p @ K.T)[:, :2] / depth[:, None], depth


def unproject(uv: Tensor, depth: Tensor, K: Tensor, cam_to_world: Tensor) -> Tensor:
    """Pixels (N, 2) at camera-frame depth (N,) -> world points (N, 3)."""
    rays = torch.cat([uv, torch.ones_like(uv[:, :1])], 1) @ torch.linalg.inv(K).T
    return transform_points(cam_to_world, rays * depth[:, None])


def pixel_to_patch(uv: Tensor, patch: int, Hp: int, Wp: int) -> tuple[Tensor, Tensor]:
    """Pixels (N, 2) -> patch (row, col) (N, 2) int64 and inside-the-grid mask (N,).

    ``rc`` is only meaningful where ``inside``; NaN/inf pixels are never inside.
    """
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u < Wp * patch) & (v >= 0) & (v < Hp * patch)
    return torch.floor(uv.flip(-1) / patch).long(), inside


def zbuffer_visibility(uv: Tensor, depth: Tensor, ref_uv: Tensor, ref_depth: Tensor, H: int, W: int,
                       cell_px: int = 4, tol_m: float = 0.5, tol_rel: float = 0.0) -> Tensor:
    """(N,) bool: point is in the image, in front of the camera and within ``tol_m + tol_rel * depth`` of the
    nearest reference depth in its ``cell_px`` x ``cell_px`` cell. A cell with no reference counts as visible."""
    nr, nc = -(-H // cell_px), -(-W // cell_px)

    def cell(uv: Tensor, d: Tensor) -> tuple[Tensor, Tensor]:
        rc, inside = pixel_to_patch(uv, cell_px, nr, nc)
        ok = inside & (uv[:, 0] < W) & (uv[:, 1] < H) & (d > 0)
        return ok, torch.where(ok, rc[:, 0] * nc + rc[:, 1], 0)

    ref_ok, ref_idx = cell(ref_uv, ref_depth)
    zmin = torch.full((nr * nc,), float("inf"), dtype=ref_depth.dtype, device=ref_depth.device)
    zmin.scatter_reduce_(0, ref_idx[ref_ok], ref_depth[ref_ok], "amin")
    ok, idx = cell(uv, depth)
    return ok & (depth <= zmin[idx] + tol_m + tol_rel * depth)


def patch_ray_dirs(K: Tensor, cam_to_world: Tensor, Hp: int, Wp: int, patch: int) -> Tensor:
    """Unit world-frame ray directions (Hp, Wp, 3) through each patch centre."""
    r, c = torch.meshgrid(torch.arange(Hp, device=K.device), torch.arange(Wp, device=K.device), indexing="ij")
    uv1 = torch.stack([(c + 0.5) * patch, (r + 0.5) * patch, torch.ones_like(r, dtype=torch.float)], -1)
    rays = uv1.to(K.dtype) @ torch.linalg.inv(K).T @ cam_to_world[:3, :3].T
    return F.normalize(rays, dim=-1)


def project_visible(p_world: Tensor, ref_world: Tensor, K: Tensor, cam_to_world: Tensor, H: int, W: int,
                    cell_px: int = 4, tol_m: float = 0.5, tol_rel: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
    """Project points (N, 3) into one camera and z-buffer them against reference points (R, 3), e.g. all
    LiDAR train points of the paired sweep. Returns uv (N, 2), depth (N,), visible (N,) bool."""
    uv, depth = project(p_world, K, cam_to_world)
    ref_uv, ref_depth = project(ref_world, K, cam_to_world)
    return uv, depth, zbuffer_visibility(uv, depth, ref_uv, ref_depth, H, W, cell_px, tol_m, tol_rel)
