"""check_omniworld.py: decoders and the depth/pose reprojection on a synthetic scene with known motion."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_omniworld", ROOT / "scripts" / "data" / "check_omniworld.py")
co = importlib.util.module_from_spec(spec); spec.loader.exec_module(co)  # type: ignore[union-attr]


def test_flow_decode_range_is_symmetric():
    u = np.array([[0, 32767, 65535]], np.uint16)
    f = co.decode_flow(u, u)[..., 0]
    assert np.allclose(f[0, [0, 2]], [-300.0, 300.0]) and abs(f[0, 1]) < 0.01


def test_depth_decode_masks_and_is_monotonic():
    raw = np.array([[50, 1000, 20000, 65520]], np.uint16)
    d, valid = co.decode_depth(raw)
    assert valid.tolist() == [[False, True, True, False]]
    assert d[0, 1] < d[0, 2]


def test_quaternion_matches_axis_rotation():
    R = co.quat_wxyz_to_R([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])   # 90° about z
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_reprojection_of_a_wall_under_sideways_motion():
    H, W = 40, 60
    K = np.array([[50.0, 0, 30], [0, 50.0, 20], [0, 0, 1]])
    depth = np.full((H, W), 10.0)                     # fronto-parallel wall 10 units away
    w2c_a = np.eye(4)
    w2c_b = np.eye(4); w2c_b[0, 3] = -1.0             # camera b moved +1 along x
    static = np.ones((H, W), bool)
    e, (fu, fv), ok = co.pair_errors(depth, depth, K, w2c_a, w2c_b, static)
    assert e["absrel_median"] < 1e-9 and e["within_5pct"] == 1.0
    assert np.allclose(fu[ok], -5.0) and np.allclose(fv[ok], 0.0)   # flow = -f * tx / z
