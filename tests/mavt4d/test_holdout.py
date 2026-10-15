"""Hold-out split of LiDAR points: a pure function of (source, sweep key), never of global RNG state."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from mavt4d.data.holdout import apply_holdout, heldout_mask
from mavt4d.data.sample import SceneSample


def _sample(keys: list[str], ns: list[int] | None = None, scene_id: str = "scene-0") -> SceneSample:
    """Tier-A sample whose point ``i`` of every sweep has x = i, so membership is checkable by id."""
    ns = ns or [200] * len(keys)
    pts = []
    for n in ns:
        p = torch.zeros(n, 4)
        p[:, 0] = torch.arange(n)
        pts.append(p)
    S = len(keys)
    return SceneSample(scene_id=scene_id, source="nuscenes", images=torch.zeros(1, 1, 3, 16, 16),
                       frame_times=torch.zeros(1, 1, dtype=torch.float64), intrinsics=torch.eye(3)[None],
                       cam_to_world=torch.eye(4)[None, None], lidar_points=pts,
                       lidar_to_world=torch.eye(4).repeat(S, 1, 1),
                       lidar_times=torch.arange(S, dtype=torch.float64), lidar_keys=list(keys))


def test_fraction_within_one_percent():
    m = heldout_mask(34_000, 0.3, "nuscenes/lidar-token-0")
    assert m.shape == (34_000,) and m.dtype == torch.bool
    assert abs(m.float().mean().item() - 0.3) <= 0.01
    with pytest.raises(ValueError, match="frac"):
        heldout_mask(10, 1.5, "nuscenes/lidar-token-0")


def test_deterministic_regardless_of_global_seed():
    torch.manual_seed(0), random.seed(0), np.random.seed(0)
    a = heldout_mask(5_000, 0.3, "nuscenes/abc")
    torch.manual_seed(123), random.seed(123), np.random.seed(123)
    torch.rand(10), random.random(), np.random.rand()
    b = heldout_mask(5_000, 0.3, "nuscenes/abc")
    assert torch.equal(a, b)


def test_different_keys_differ():
    assert not torch.equal(heldout_mask(5_000, 0.3, "nuscenes/abc"), heldout_mask(5_000, 0.3, "nuscenes/abd"))


def test_apply_holdout_sets_mask_for_every_sweep():
    s = _sample(["k0", "k1", "k2"], ns=[100, 150, 230])
    out = apply_holdout(s, frac=0.3)
    assert s.heldout is None  # input untouched
    assert len(out.heldout) == 3
    for i, (h, n) in enumerate(zip(out.heldout, [100, 150, 230])):
        assert h.shape == (n,) and h.dtype == torch.bool
        assert torch.equal(h, heldout_mask(n, 0.3, f"nuscenes/k{i}"))
    out.validate()


def test_input_points_never_contain_heldout():
    out = apply_holdout(_sample(["k0", "k1"]), frac=0.3)
    for s in range(2):
        seen = set(out.input_points(s)[:, 0].long().tolist())
        held = set(out.heldout[s].nonzero().flatten().tolist())
        assert held and not (seen & held)
        assert seen | held == set(range(200))


def test_same_sweep_same_mask_across_samples():
    # two overlapping M2 windows of one scene share sweeps k2, k3 at different positions
    a = apply_holdout(_sample(["k0", "k1", "k2", "k3"], scene_id="scene-0/w0"))
    b = apply_holdout(_sample(["k2", "k3", "k4", "k5"], scene_id="scene-0/w2"))
    assert torch.equal(a.heldout[2], b.heldout[0])
    assert torch.equal(a.heldout[3], b.heldout[1])
    assert not torch.equal(a.heldout[2], a.heldout[3])
