"""LiDAR hubs: farthest-point centres, kNN members, built only from input (non-held-out) points."""
from __future__ import annotations

import pytest
import torch

from mavt4d.data.sample import SceneSample
from mavt4d.geometry.camera import transform_points
from mavt4d.geometry.hubs import build_hubs, farthest_point_sample, knn


def _cloud(seed: int, n: int = 400) -> torch.Tensor:
    return torch.randn(n, 3, generator=torch.Generator().manual_seed(seed)) * torch.tensor([10.0, 10.0, 1.0])


def _sample(points: list[torch.Tensor], heldout: list[torch.Tensor] | None = None) -> SceneSample:
    S = len(points)
    l2w = torch.eye(4).repeat(S, 1, 1)
    l2w[:, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg mount yaw
    l2w[:, :3, 3] = torch.tensor([3.0, -1.0, 1.8])
    return SceneSample(scene_id="s", source="synthetic", images=torch.zeros(1, 1, 3, 16, 16),
                       frame_times=torch.zeros(1, 1, dtype=torch.float64), intrinsics=torch.eye(3)[None],
                       cam_to_world=torch.eye(4)[None, None],
                       lidar_points=[torch.cat([p, torch.rand(len(p), 1)], 1) for p in points],
                       lidar_to_world=l2w, lidar_times=torch.zeros(S, dtype=torch.float64),
                       lidar_keys=[f"k{s}" for s in range(S)], heldout=heldout)


def test_fps_deterministic():
    xyz = _cloud(0)
    a, b = farthest_point_sample(xyz, 32), farthest_point_sample(xyz, 32)
    assert a.dtype == torch.int64 and a.shape == (32,) and torch.equal(a, b) and a[0] == 0
    assert farthest_point_sample(xyz, 32, start_idx=7)[0] == 7


def test_fps_coverage_beats_random():
    for seed in range(5):
        xyz = _cloud(seed)
        fps = xyz[farthest_point_sample(xyz, 32)]
        rnd = xyz[torch.randperm(len(xyz), generator=torch.Generator().manual_seed(seed))[:32]]
        min_pair = lambda p: torch.pdist(p).min()  # noqa: E731
        assert min_pair(fps) >= min_pair(rnd)


def test_fps_unique_indices():
    idx = farthest_point_sample(_cloud(1), 64)
    assert len(idx.unique()) == 64


def test_knn_matches_bruteforce():
    ref, query = _cloud(2, 300), _cloud(3, 50)
    idx = knn(query, ref, 8)
    d = ((query[:, None] - ref[None]) ** 2).sum(-1)
    assert idx.shape == (50, 8) and idx.dtype == torch.int64
    assert torch.equal(idx, d.argsort(dim=1)[:, :8])


def test_build_hubs_uses_only_input_points():
    near = _cloud(4, 300)
    far = torch.randn(50, 3, generator=torch.Generator().manual_seed(5)) + torch.tensor([100.0, 0.0, 0.0])
    held = torch.cat([torch.zeros(300, dtype=torch.bool), torch.ones(50, dtype=torch.bool)])
    s = _sample([torch.cat([near, far])], [held])
    hubs = build_hubs(s, K=16, M=8)
    assert hubs.centers_sensor.shape == (1, 16, 3) and hubs.rel_xyz.shape == (1, 16, 8, 3)
    assert (hubs.centers_sensor[0] - torch.tensor([100.0, 0.0, 0.0])).norm(dim=-1).min() > 50
    pts = s.input_points(0)
    assert hubs.member_idx.max() < len(pts)
    members = pts[hubs.member_idx[0], :3]
    assert torch.allclose(hubs.rel_xyz[0], members - hubs.centers_sensor[0, :, None], atol=1e-6)
    assert torch.allclose(hubs.intensity[0, ..., 0], pts[hubs.member_idx[0], 3])
    assert torch.allclose(hubs.centers_world[0], transform_points(s.lidar_to_world[0], hubs.centers_sensor[0]))


def test_build_hubs_pads_when_fewer_than_K():
    s = _sample([_cloud(6, 10), _cloud(7, 40)])
    hubs = build_hubs(s, K=16, M=32)
    assert hubs.centers_world.shape == (2, 16, 3) and hubs.member_idx.shape == (2, 16, 32)
    assert hubs.member_idx[0].max() < 10 and len(hubs.member_idx[0].unique()) == 10
    assert len(hubs.centers_sensor[1].unique(dim=0)) == 16  # enough points: no repeats


def test_build_hubs_none_without_lidar():
    s = SceneSample(scene_id="s", source="openimages", images=torch.zeros(1, 1, 3, 16, 16),
                    frame_times=torch.zeros(1, 1, dtype=torch.float64))
    assert build_hubs(s) is None
