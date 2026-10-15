"""nuScenes reader on a fake on-disk dataset (tests/mavt4d/fixtures/fake_nuscenes.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fixtures import fake_nuscenes as fk
from mavt4d.data import nuscenes as nu

S = 256 / 900  # resize factor 1600x900 -> 455x256
CROP = 3  # (455 - 448) // 2 columns dropped on the left


@pytest.fixture(scope="module")
def gt(tmp_path_factory):
    return fk.write_fake_nuscenes(tmp_path_factory.mktemp("nusc"))


@pytest.fixture
def make_ds(gt, monkeypatch):
    monkeypatch.setattr(nu, "load_official_split", lambda version, split: list(fk.SCENE_NAMES))

    def make(mode="m3", **kw):
        return nu.NuScenesSceneDataset(gt["root"], version=fk.VERSION, split="mini_train", mode=mode,
                                       cameras=fk.CAMERAS, **kw)
    return make


def _close(a, b, atol=1e-4):
    return np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=atol)


def test_tables_load_and_index(gt):
    tb = nu.NuScenesTables(gt["root"], fk.VERSION)
    assert len(tb.scene) == 2 and len(tb.sample) == 4 and len(tb.instance) == 4
    assert len(tb.sample_data) == len(gt["time_us"])
    for name in fk.SCENE_NAMES:
        scene = tb.scene_by_name[name]
        assert tb.scene_samples(scene) == gt["samples"][name]
        for ch in (*fk.CAMERAS, "LIDAR_TOP"):
            chain = tb.chain(scene, ch)
            assert chain == gt["chain"][(name, ch)]  # linked across scenes on disk, cut by the reader
            assert all(tb.channel(t) == ch for t in chain)
    for sample, by_ch in gt["keyframes"].items():
        assert tb.keyframe_data[sample] == by_ch
        assert sorted(tb.annotations_of[sample]) == sorted(
            a["token"] for a in tb.sample_annotation.values() if a["sample_token"] == sample)


def test_cam_to_world_chain_and_recentering(gt, make_ds):
    tb_lidar = fk.sensor_matrix("LIDAR_TOP")
    # m3: world = ego frame of LIDAR_TOP at the keyframe
    s = make_ds("m3")[0]
    s.validate()
    sample = gt["samples"][fk.SCENE_NAMES[0]][0]
    ref = np.linalg.inv(gt["ego"][gt["keyframes"][sample]["LIDAR_TOP"]])
    assert _close(s.lidar_to_world[0], tb_lidar)  # ego(ref) @ inv(ego(ref)) = identity
    assert _close(s.lidar_to_world[0] @ torch.linalg.inv(torch.from_numpy(tb_lidar).float()), np.eye(4))
    for v, cam in enumerate(fk.CAMERAS):
        sd = gt["keyframes"][sample][cam]
        assert _close(s.cam_to_world[v, 0], ref @ gt["ego"][sd] @ fk.sensor_matrix(cam))
        assert s.frame_times[v, 0].item() == pytest.approx((gt["time_us"][sd] - gt["time_us"][
            gt["keyframes"][sample]["LIDAR_TOP"]]) / 1e6)
    assert s.cam_to_world[:, :, :3, 3].abs().max() < 10  # recentred: global coords are ~1000 m
    # m2: world = ego frame of the LiDAR sweep nearest to the first frame
    m2 = make_ds("m2", clip_len=4)[1]
    m2.validate()
    assert _close(m2.lidar_to_world[0], tb_lidar)
    ref2 = np.linalg.inv(gt["ego"][m2.lidar_keys[0]])
    cam_sd = gt["chain"][(fk.SCENE_NAMES[0], "CAM_FRONT")][1:5]
    for t, sd in enumerate(cam_sd):
        assert _close(m2.cam_to_world[0, t], ref2 @ gt["ego"][sd] @ fk.sensor_matrix("CAM_FRONT"))


def test_intrinsics_resize_crop_formula(make_ds):
    s = make_ds("m3")[0]
    assert s.images.shape == (2, 1, 3, 256, 448) and s.images.dtype == torch.float32
    assert s.images.min() >= -1 and s.images.max() <= 1
    (fx, _, cx), (_, fy, cy), _ = fk.K_RAW
    want = [[fx * S, 0, cx * S - CROP], [0, fy * S, cy * S], [0, 0, 1]]
    for v in range(2):
        assert _close(s.intrinsics[v], want, atol=1e-3)
    # the image content moved exactly like the intrinsics: stripe centre u -> u*s - 3
    col = s.images[0, 0].mean(dim=(0, 1))
    w = col - col.min()
    centroid = (w * (torch.arange(448) + 0.5)).sum() / w.sum()  # pixel i covers [i, i + 1)
    assert abs(centroid.item() - ((sum(fk.STRIPE_U) / 2) * S - CROP)) < 0.2


def test_nearest_lidar_sweep_per_frame(gt, make_ds):
    ds = make_ds("m2", clip_len=4)
    name = fk.SCENE_NAMES[0]
    lidar = gt["chain"][(name, "LIDAR_TOP")]
    lidar_t = np.array([gt["time_us"][t] for t in lidar])
    for start in range(4):
        s = ds[start]
        assert len(s.lidar_points) == 4 and s.lidar_times.shape == (4,)
        for t, sd in enumerate(gt["chain"][(name, "CAM_FRONT")][start:start + 4]):
            want = lidar[int(np.argmin(np.abs(lidar_t - gt["time_us"][sd])))]
            assert s.lidar_keys[t] == want
            dt = s.lidar_times[t] - s.frame_times[0, t]
            assert dt.item() == pytest.approx((gt["time_us"][want] - gt["time_us"][sd]) / 1e6, abs=1e-6)


def test_box_interpolation_between_keyframes(gt, make_ds):
    name = fk.SCENE_NAMES[0]
    inst_a, inst_b = gt["instance_index"]["inst-A-0"], gt["instance_index"]["inst-B-0"]
    # m3 keyframe: boxes are the annotations, recentred
    s3 = make_ds("m3")[0]
    sample = gt["samples"][name][0]
    ref3 = np.linalg.inv(gt["ego"][gt["keyframes"][sample]["LIDAR_TOP"]])
    b = s3.boxes[0]
    assert sorted(b.instance_id.tolist()) == sorted([inst_a, inst_b])
    c, size, yaw = gt["boxes"][sample]["inst-A-0"]
    i = b.instance_id.tolist().index(inst_a)
    assert _close(b.center[i], (ref3 @ np.append(c, 1))[:3]) and _close(b.size[i], size)
    assert b.yaw[i].item() == pytest.approx(yaw + math.atan2(ref3[1, 0], ref3[0, 0]), abs=1e-5)
    # m2 window over the whole scene: frames 0 and 6 are keyframes, 1..5 interpolate on camera time
    s = make_ds("m2", clip_len=7)[0]
    s.validate()
    ref = np.linalg.inv(gt["ego"][s.lidar_keys[0]])
    yaw_ref = math.atan2(ref[1, 0], ref[0, 0])
    (c0, sz, y0), (c1, _, y1) = fk.BOX_A
    cam = gt["chain"][(name, "CAM_FRONT")]
    t0, t1 = gt["time_us"][cam[0]], gt["time_us"][cam[6]]
    for t in range(7):
        bt = s.boxes[t]
        ids = bt.instance_id.tolist()
        assert (inst_b in ids) == (t == 0)  # B is only annotated in keyframe 0: dropped when interpolating
        w = (gt["time_us"][cam[t]] - t0) / (t1 - t0)
        i = ids.index(inst_a)
        want_c = (ref @ np.append((1 - w) * np.array(c0) + w * np.array(c1), 1))[:3]
        assert _close(bt.center[i], want_c) and _close(bt.size[i], sz)
        assert bt.yaw[i].item() == pytest.approx((1 - w) * y0 + w * y1 + yaw_ref, abs=1e-5)


def test_dev_split_deterministic_about_10pct():
    names = [f"scene-{i:04d}" for i in range(1000)]
    dev = nu.dev_split_scenes(names)
    assert dev == nu.dev_split_scenes(list(reversed(names)))
    assert dev <= set(names) and 0.07 <= len(dev) / len(names) <= 0.13
    assert dev == {n for n in names if nu.dev_split_scenes([n])}  # membership depends on the name alone


def test_m2_windows_within_scene(gt, make_ds):
    ds = make_ds("m2", clip_len=4)
    assert len(ds) == 2 * (7 - 4 + 1)  # 7 CAM_FRONT frames per scene; the on-disk chain is linked across
    for i in range(len(ds)):
        s = ds[i]
        own = set(gt["chain"][(s.scene_id, "LIDAR_TOP")])
        assert s.images.shape[:2] == (1, 4) and set(s.lidar_keys) <= own
        assert (s.frame_times[0, -1] - s.frame_times[0, 0]).item() == pytest.approx(3 * fk.CAM_DT, abs=1e-5)
    assert [ds[i].scene_id for i in range(len(ds))] == [fk.SCENE_NAMES[0]] * 4 + [fk.SCENE_NAMES[1]] * 4


def test_lidar_keys_are_sample_data_tokens(gt, make_ds):
    tb = nu.NuScenesTables(gt["root"], fk.VERSION)
    s3 = make_ds("m3")[0]
    assert s3.lidar_keys == [gt["keyframes"][gt["samples"][fk.SCENE_NAMES[0]][0]]["LIDAR_TOP"]]
    s2 = make_ds("m2", clip_len=4)[2]
    for s in (s3, s2):
        assert len(s.lidar_keys) == len(s.lidar_points)
        for key, pts in zip(s.lidar_keys, s.lidar_points):
            assert key in tb.sample_data and tb.channel(key) == "LIDAR_TOP"
            assert pts.shape == (gt["n_points"][key], 4)  # the file of that token was read
