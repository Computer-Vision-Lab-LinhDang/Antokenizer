"""SceneSample: tier rules, shape validation and held-out filtering of LiDAR input points."""
from __future__ import annotations

import re
from dataclasses import replace

import pytest
import torch

from mavt4d.data.sample import Boxes3D, SceneSample


def _boxes(m: int = 2) -> Boxes3D:
    return Boxes3D(center=torch.zeros(m, 3), size=torch.ones(m, 3), yaw=torch.zeros(m),
                   instance_id=torch.arange(m))


def _sample(V: int = 2, T: int = 1, lidar: bool = True, calib: bool = True, S: int = 2,
            n: int = 10) -> SceneSample:
    """Tiny sample; point ``i`` of every sweep has x = i so filtering is checkable by identity."""
    kw = {}
    if calib:
        kw.update(intrinsics=torch.eye(3).repeat(V, 1, 1), cam_to_world=torch.eye(4).repeat(V, T, 1, 1))
    if lidar:
        pts = torch.zeros(n, 4)
        pts[:, 0] = torch.arange(n)
        kw.update(lidar_points=[pts.clone() for _ in range(S)], lidar_to_world=torch.eye(4).repeat(S, 1, 1),
                  lidar_times=torch.zeros(S, dtype=torch.float64), lidar_keys=[f"sd{s}" for s in range(S)])
    return SceneSample(scene_id="scene-0", source="synthetic", images=torch.zeros(V, T, 3, 32, 48),
                       frame_times=torch.zeros(V, T, dtype=torch.float64), **kw)


def test_tier_A_requires_lidar_and_calibration():
    assert _sample().tier == "A"
    assert _sample(lidar=False).tier == "C"
    assert replace(_sample(), intrinsics=None).tier == "C"
    assert replace(_sample(), cam_to_world=None).tier == "C"
    assert _sample(T=3, calib=False).tier == "B"  # LiDAR without calibration is not geometry


def test_tier_B_video_without_geometry():
    assert _sample(V=1, T=4, lidar=False, calib=False).tier == "B"


def test_tier_C_single_image():
    assert _sample(V=1, T=1, lidar=False, calib=False).tier == "C"


def test_validate_rejects_bad_shapes():
    V, T, S, n = 2, 2, 2, 10
    good = replace(_sample(V=V, T=T, S=S, n=n),
                   heldout=[torch.zeros(n, dtype=torch.bool)] * S,
                   boxes=[_boxes(), _boxes(3)],
                   patch_flow=torch.zeros(V, T - 1, 2, 3, 3))
    good.validate()

    pts = good.lidar_points
    bad = {
        "images": torch.zeros(V, T, 4, 32, 48),
        "frame_times": torch.zeros(V, T + 1, dtype=torch.float64),
        "intrinsics": torch.eye(4).repeat(V, 1, 1),
        "cam_to_world": torch.eye(4).repeat(V, 1, 1),
        "lidar_points": [pts[0], torch.zeros(n, 3)],
        "lidar_to_world": torch.eye(4).repeat(S + 1, 1, 1),
        "lidar_times": torch.zeros(S, 1, dtype=torch.float64),
        "lidar_keys": None,  # required whenever lidar_points is set
        "heldout": [torch.zeros(n, dtype=torch.uint8)] * S,  # a non-bool mask would gather, not filter
        "boxes": [_boxes(), replace(_boxes(), yaw=torch.zeros(5))],
        "patch_flow": torch.zeros(V, T, 2, 3, 3),
    }
    for field, value in bad.items():
        with pytest.raises(ValueError, match=re.escape(field)):
            replace(good, **{field: value}).validate()


def test_input_points_excludes_heldout():
    s = _sample(S=1)
    held = torch.zeros(10, dtype=torch.bool)
    held[[1, 4, 7]] = True
    s = replace(s, heldout=[held])
    assert s.input_points(0)[:, 0].tolist() == [0, 2, 3, 5, 6, 8, 9]


def test_input_points_without_heldout_returns_all():
    s = _sample(S=2)
    assert s.heldout is None
    assert torch.equal(s.input_points(1), s.lidar_points[1])
