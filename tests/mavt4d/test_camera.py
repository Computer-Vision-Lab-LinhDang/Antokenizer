"""Camera geometry: SE(3), pinhole projection, pixel→patch binning, z-buffer visibility, patch rays."""
from __future__ import annotations

import torch

from mavt4d.geometry.camera import (invert_se3, patch_ray_dirs, pixel_to_patch, project, transform_points,
                                    unproject, zbuffer_visibility)

K = torch.tensor([[100.0, 0.0, 40.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])  # W=80, H=48


def _pose(seed: int, t: tuple[float, float, float] = (1.0, -2.0, 0.5)) -> torch.Tensor:
    """Random proper rigid transform (4, 4) float32."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3], T[:3, 3] = q, torch.tensor(t, dtype=torch.float64)
    return T.float()


def _cam_points(n: int, seed: int, z_lo: float = 2.0, z_hi: float = 20.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = z_lo + (z_hi - z_lo) * torch.rand(n, generator=g)
    xy = (torch.rand(n, 2, generator=g) * 2 - 1) * z[:, None] * 0.4
    return torch.cat([xy, z[:, None]], 1)


def test_project_unproject_roundtrip():
    c2w = _pose(0)
    p_cam = _cam_points(200, seed=1)
    p_world = transform_points(c2w, p_cam)
    uv, depth = project(p_world, K, c2w)
    assert torch.allclose(depth, p_cam[:, 2], atol=1e-4)
    assert torch.allclose(unproject(uv, depth, K, c2w), p_world, atol=1e-4)


def test_points_behind_camera_have_negative_depth():
    c2w = _pose(2)
    p_cam = _cam_points(50, seed=3)
    p_cam[25:, 2] *= -1
    _, depth = project(transform_points(c2w, p_cam), K, c2w)
    assert (depth[:25] > 0).all() and (depth[25:] < 0).all()


def test_invert_se3_composes_to_identity():
    T = torch.stack([torch.stack([_pose(3 * i + j, (i, -j, 2.0)) for j in range(3)]) for i in range(2)])
    inv = invert_se3(T)
    eye = torch.eye(4).expand(2, 3, 4, 4)
    assert torch.allclose(T @ inv, eye, atol=1e-5) and torch.allclose(inv @ T, eye, atol=1e-5)
    p = torch.randn(2, 3, 7, 3)
    assert torch.allclose(transform_points(inv, transform_points(T, p)), p, atol=1e-5)


def test_pixel_to_patch_boundaries():
    Hp, Wp, patch = 3, 5, 16
    uv = torch.tensor([[15.99, 0.0], [16.0, 0.0], [0.0, 15.99], [0.0, 16.0], [79.9, 47.9],
                       [-0.01, 5.0], [80.0, 5.0], [5.0, 48.0], [float("nan"), 5.0]])
    rc, inside = pixel_to_patch(uv, patch, Hp, Wp)
    assert rc.dtype == torch.int64
    assert rc[:5].tolist() == [[0, 0], [0, 1], [0, 0], [1, 0], [2, 4]]
    assert inside.tolist() == [True] * 5 + [False] * 4


def test_zbuffer_far_point_hidden_by_near_point_same_cell():
    ref_uv = torch.tensor([[10.5, 10.5], [30.0, 30.0]])
    ref_depth = torch.tensor([5.0, 8.0])
    uv = torch.tensor([[11.5, 9.2],  # same 4 px cell as the near ref, far behind it
                       [10.5, 10.5],  # the near ref itself
                       [50.0, 10.0],  # empty cell: no evidence of occlusion
                       [90.0, 10.0],  # outside the image
                       [11.5, 9.2]])  # behind the camera
    depth = torch.tensor([20.0, 5.0, 50.0, 1.0, -1.0])
    vis = zbuffer_visibility(uv, depth, ref_uv, ref_depth, H=48, W=80, cell_px=4, tol_m=0.5)
    assert vis.tolist() == [False, True, True, False, False]


def test_zbuffer_tolerance():
    ref_uv = torch.tensor([[20.2, 20.2], [21.0, 22.0]])
    ref_depth = torch.tensor([10.0, 12.0])  # nearest surface in the cell is at 10 m
    uv = torch.tensor([[22.0, 21.0], [23.0, 23.0]])
    vis = zbuffer_visibility(uv, torch.tensor([10.4, 10.6]), ref_uv, ref_depth, H=48, W=80, tol_m=0.5)
    assert vis.tolist() == [True, False]


def test_zbuffer_depth_relative_tolerance():
    """tol = tol_m + tol_rel * depth of the tested point: 5% of 10.4 m is 0.52 m, of 10.6 m 0.53 m."""
    ref_uv = torch.tensor([[20.2, 20.2], [40.5, 10.5]])
    ref_depth = torch.tensor([10.0, 30.0])  # nearest surfaces of two cells
    uv = torch.tensor([[22.0, 21.0], [22.0, 21.0], [41.0, 11.0]])
    depth = torch.tensor([10.4, 10.6, 31.4])
    rel = zbuffer_visibility(uv, depth, ref_uv, ref_depth, H=48, W=80, tol_m=0.0, tol_rel=0.05)
    assert rel.tolist() == [True, False, True]  # 31.4 <= 30 + 1.57
    fixed = zbuffer_visibility(uv, depth, ref_uv, ref_depth, H=48, W=80, tol_m=0.5)
    assert fixed.tolist() == [True, False, False]

def test_patch_ray_dirs_unit_and_center_is_optical_axis():
    c2w = _pose(4)
    Hp, Wp, patch = 3, 5, 16
    dirs = patch_ray_dirs(K, c2w, Hp, Wp, patch)
    assert dirs.shape == (Hp, Wp, 3)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(Hp, Wp), atol=1e-6)
    assert torch.allclose(dirs[1, 2], c2w[:3, 2], atol=1e-6)
    # every ray, walked 10 m from the camera centre, projects back onto its patch centre
    uv, _ = project(c2w[:3, 3] + 10 * dirs.reshape(-1, 3), K, c2w)
    r, c = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
    centers = torch.stack([(c + 0.5) * patch, (r + 0.5) * patch], -1).reshape(-1, 2).float()
    assert torch.allclose(uv, centers, atol=1e-3)
