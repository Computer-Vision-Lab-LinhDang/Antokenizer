"""Depth manifest builders (scripts/make_depth_manifest.py): LiDAR projection, Argoverse 2 poses, OmniWorld rows."""
import importlib.util
import json
import os

import numpy as np
import pytest
from PIL import Image

from mavt.cd32.depth_data import DepthManifest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location("make_depth_manifest",
                                              os.path.join(ROOT, "scripts", "make_depth_manifest.py"))
mdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdm)

K0 = dict(fx=100.0, fy=100.0, cx=40.0, cy=30.0, k1=0.0, k2=0.0, k3=0.0, w=80, h=60)


def _mat_to_quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-3:
        return w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
    x = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
    return (R[2, 1] - R[1, 2]) / (4 * x), x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x)


def test_quat_matches_rotation():
    assert np.allclose(mdm._quat(np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    R = mdm._quat(*q)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(mdm._quat(*_mat_to_quat(R)), R)


def test_project_lidar_pinhole_distortion_and_zbuffer():
    pts = np.array([[0.0, 0.0, 5.0],           # principal point
                    [0.1, -0.05, 2.0],         # xn = 0.05, yn = -0.025
                    [0.0, 0.0, 9.0],           # behind the first point, same pixel
                    [0.0, 0.0, -3.0]])         # behind the camera
    d = mdm.project_lidar(pts, K0, 1.0)
    assert d.shape == (60, 80) and (d > 0).sum() == 2
    assert d[30, 40] == 5.0 and d[27, 45] == 2.0
    K = dict(K0, k1=-0.2)
    d = mdm.project_lidar(np.array([[0.5, 0.0, 2.0]]), K, 0.5)              # xn = 0.25, r^2 = 1/16
    u = (100 * 0.25 * (1 - 0.2 / 16) + 40) * 0.5
    assert d.shape == (30, 40) and d[15, int(u)] == 2.0


def test_project_lidar_drops_points_past_the_distortion_fold():
    K = dict(K0, k1=-0.5)                       # r (1 - 0.5 r^2) peaks at r = 0.816, then folds back
    assert mdm._r_limit(-0.5, 0, 0, 10.0) == pytest.approx(np.sqrt(2 / 3), abs=1e-3)
    d = mdm.project_lidar(np.array([[1.2, 0.0, 1.0]]), K, 1.0)              # r = 1.2 would land at u = 74: dropped
    assert (d > 0).sum() == 0


def test_project_lidar_drops_far_points_next_to_near_ones():
    pts = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 10.0], [3.0, 0.0, 10.0]])   # u = 40, 41 (beside it), 70
    d = mdm.project_lidar(pts, K0, 1.0)
    assert d[30, 40] == 2.0 and d[30, 41] == 0 and d[30, 70] == 10.0


def _fake_av2_log(tmp_path, monkeypatch):
    """One log: a wall 10 m ahead of the ego at the sweep; the ego drives 0.5 m forward until the image."""
    tables = {}
    R_s = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float64)        # camera (x right, y down, z fwd) -> ego
    q = _mat_to_quat(R_s)
    log = tmp_path / "av2" / "val" / "log0"
    cam_dir = log / "sensors" / "cameras" / "ring_front_center"
    lid_dir = log / "sensors" / "lidar"
    cam_dir.mkdir(parents=True)
    lid_dir.mkdir(parents=True)
    (log / "calibration").mkdir()
    tables[str(log / "calibration" / "egovehicle_SE3_sensor.feather")] = dict(
        sensor_name=np.array(["ring_front_center"]), qw=np.array([q[0]]), qx=np.array([q[1]]),
        qy=np.array([q[2]]), qz=np.array([q[3]]), tx_m=np.array([1.0]), ty_m=np.array([0.0]),
        tz_m=np.array([1.5]))
    tables[str(log / "calibration" / "intrinsics.feather")] = dict(
        sensor_name=np.array(["ring_front_center"]), fx_px=np.array([100.0]), fy_px=np.array([100.0]),
        cx_px=np.array([40.0]), cy_px=np.array([30.0]), k1=np.array([0.0]), k2=np.array([0.0]),
        k3=np.array([0.0]), width_px=np.array([80]), height_px=np.array([60]))
    tl, tc = 1_000_000_000, 1_010_000_000                                  # image 10 ms after the sweep
    yaw = np.pi / 2                                                        # ego faces +y of the city
    qe = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    tables[str(log / "city_SE3_egovehicle.feather")] = dict(
        timestamp_ns=np.array([tl, tc], np.int64), qw=np.array([qe[0]] * 2), qx=np.array([0.0] * 2),
        qy=np.array([0.0] * 2), qz=np.array([qe[3]] * 2), tx_m=np.array([3.0, 3.0]),
        ty_m=np.array([2.0, 2.5]), tz_m=np.array([0.0, 0.0]))
    yy, zz = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-1, 4, 51))
    tables[str(lid_dir / f"{tl}.feather")] = dict(x=np.full(yy.size, 10.0), y=yy.ravel(), z=zz.ravel())
    for p in tables:
        open(p, "w").close()
    Image.new("RGB", (80, 60), (90, 120, 150)).save(cam_dir / f"{tc}.jpg")
    monkeypatch.setattr(mdm, "_feather", lambda p: tables[p])
    return tmp_path / "av2"


def test_argoverse2_moves_the_sweep_into_the_camera(tmp_path, monkeypatch):
    root = _fake_av2_log(tmp_path, monkeypatch)
    rows = mdm._av2_log((str(root / "val" / "log0"), str(tmp_path / "out"), ["ring_front_center"], 1, 1.0, 30.0))
    assert len(rows) == 1
    r = rows[0]
    d = np.array(Image.open(tmp_path / "out" / r["depth"])).astype(np.float32) * r["scale"]
    # wall at 10 m ahead of the ego at the sweep; at the image the ego is 0.5 m nearer, the camera 1 m ahead
    assert np.allclose(d[d > 0], 8.5, atol=1 / 256)
    assert (d > 0).mean() > 0.1 and os.path.exists(r["image"])


def test_argoverse2_command_writes_a_loadable_manifest(tmp_path, monkeypatch):
    root = _fake_av2_log(tmp_path, monkeypatch)
    monkeypatch.setattr(mdm, "ProcessPoolExecutor", _SerialPool)
    assert mdm.main(["argoverse2", "--root", str(root), "--split", "val", "--cams", "ring_front_center",
                     "--per-log", "1", "--scale", "0.5", "--out-dir", str(tmp_path / "out")]) == 0
    ds = DepthManifest(str(tmp_path / "out" / "val" / "manifest.jsonl"), size=32, train=False)
    item = ds[0]
    v = item["depth"][item["depth"] > 0]
    assert len(ds) == 1 and item["min_cell_frac"] == 0.02 and v.numel() > 0
    assert np.allclose(v.numpy(), 8.5, atol=0.02)


class _SerialPool:
    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, jobs):
        return map(fn, jobs)


def test_omniworld_rows_index_the_packed_depth(tmp_path):
    root = tmp_path / "pack"
    sd = root / "train" / "uid0"
    (sd / "rgb").mkdir(parents=True)
    (sd / "split0_0").mkdir()
    frames = np.arange(0, 40, 2)
    with open(sd / "scene.json", "w") as f:
        json.dump(dict(uid="uid0", segments=[dict(name="split0_0", frames=frames.tolist())]), f)
    np.savez(sd / "split0_0" / "meta.npz", frames=frames)
    np.save(sd / "split0_0" / "depth.npy", np.stack([np.full((16, 28), j + 1.0, np.float16) for j in range(20)]))
    for f in frames:
        Image.new("RGB", (112, 64), (int(f), 0, 0)).save(sd / "rgb" / f"{f}.jpg")
    out = tmp_path / "m" / "train.jsonl"
    assert mdm.main(["omniworld", "--root", str(root), "--split", "train", "--every", "8", "--out", str(out)]) == 0
    rows = [json.loads(l) for l in open(out)]
    assert [r["index"] for r in rows] == [0, 8, 16]
    assert rows[1]["image"].endswith(f"rgb/{frames[8]}.jpg")
    ds = DepthManifest(str(out), size=32, train=False)
    assert torch_allclose(ds[2]["depth"], 17.0)


def torch_allclose(t, v):
    return bool(np.allclose(t.numpy(), v))
