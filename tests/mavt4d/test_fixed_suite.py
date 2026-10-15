"""E1.3 fixed test suite v1: streaming JSON reader, window mining, selection, build script, box projection."""
from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from mavt4d.data.jsonstream import iter_json_array
from mavt4d.eval import fixed_suite as fs
from mavt4d.geometry.camera import project

ROOT = Path(__file__).resolve().parents[2]
K = 8


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- jsonstream

TRICKY = [
    {"token": "a", "s": "x]y[z}{,", "q": "say \"hi\" \\ end", "nested": [1, {"d": "]"}, []], "e": -1.5e-3,
     "u": "é ✓ 中"},
    {"n": None, "t": True, "f": False, "empty": {}, "list": [], "num": 1234567890123},
    {"k": "\\\"", "slash": "a/b", "ctl": "tab\tnl\n", "brace": "{\"x\": [1]}"},
]


def _variants() -> list[str]:
    return [
        json.dumps(TRICKY, indent=2, ensure_ascii=False),
        json.dumps(TRICKY, separators=(",", ":")),
        json.dumps(TRICKY, ensure_ascii=True),
        "  [\n\n" + ",\n\n  ".join(json.dumps(o, ensure_ascii=False) for o in TRICKY) + "\n ]\n  ",
    ]


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64, 1 << 20])
def test_stream_matches_json_load(tmp_path, chunk):
    for i, text in enumerate(_variants()):
        p = tmp_path / f"t{i}.json"
        p.write_text(text, encoding="utf-8")
        assert list(iter_json_array(p, chunk_size=chunk)) == json.loads(text)


@pytest.mark.parametrize("text", ["[]", " [ \n ] ", "\n[\n]\n"])
def test_stream_empty_array(tmp_path, text):
    p = tmp_path / "e.json"
    p.write_text(text)
    assert list(iter_json_array(p, chunk_size=2)) == []


@pytest.mark.parametrize("text", ['{"a": 1}', '[{"a": 1}', '[{"a": 1} {"b": 2}]', "[1, 2]",
                                  '[{"a": "unterminated', '[{"a": 1},]', ""])
def test_stream_rejects_malformed(tmp_path, text):
    p = tmp_path / "bad.json"
    p.write_text(text)
    with pytest.raises(ValueError):
        list(iter_json_array(p, chunk_size=3))


def test_bucket_matches_splits():
    splits = _load(ROOT / "src" / "mavt" / "data" / "splits.py", "_splits_for_test")
    for key in ["short_event:scene-0003:abc", "", "é", "pair:x|y"]:
        assert fs.bucket(key) == splits.bucket(key)


# ----------------------------------------------------------------------------------------------- mining

def _quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _ann(inst, x, y=0.0, attrs=(), vis=4, pts=10, cat="vehicle.car"):
    return fs.Ann(inst, cat, (x, y, 0.5), (2.0, 4.5, 1.6), (1.0, 0.0, 0.0, 0.0), vis, tuple(attrs), pts)


def _scene(name, ego_xy, frames, ego_yaw=0.0):
    yaws = ego_yaw if isinstance(ego_yaw, (list, tuple)) else [ego_yaw] * len(ego_xy)
    kfs = tuple(fs.Keyframe(f"{name}-s{t}", 1_000_000 + 500_000 * t, (ego_xy[t][0], ego_xy[t][1], 0.0),
                            _quat(yaws[t]), f"{name}-l{t}", {c: f"{name}-{c}-{t}" for c in fs.CAMERAS},
                            tuple(frames[t]))
                for t in range(len(ego_xy)))
    return fs.Scene(name, f"tok-{name}", kfs)


def _still(n=K):
    return [(0.05 * (t % 2), 0.0) for t in range(n)]


def _driving(speed, n=K):
    return [(speed * t, 0.0) for t in range(n)]


def _cats(scene, rare=None):
    return fs.classify_window(scene, 0, K, rare or {})


def test_static_scene_moving_camera():
    frames = [[_ann("parked", 20 + 0.05 * (t % 2), 5, ["vehicle.parked"]),
               _ann("cone", 15, -3, cat="movable_object.trafficcone")] for t in range(K)]
    cats = _cats(_scene("a", _driving(1.0), frames))
    assert set(cats) == {"static_scene_moving_camera"}
    inst, crit = cats["static_scene_moving_camera"]
    assert set(inst) == {"parked", "cone"} and crit["ego_disp_m"] == pytest.approx(7.0)
    # an ambiguous mover (moving attribute, 2 m displacement) breaks the static scene
    amb = [f + [_ann("ped", 10 + 2 * t / (K - 1), 2, ["pedestrian.moving"], cat="human.pedestrian.adult")]
           for t, f in enumerate(frames)]
    assert "static_scene_moving_camera" not in _cats(_scene("a", _driving(1.0), amb))


def test_fixed_camera_moving_object():
    frames = [[_ann("car", 10 + 2 * t, 3, ["vehicle.moving"])] for t in range(K)]
    cats = _cats(_scene("b", _still(), frames))
    assert set(cats) == {"fixed_camera_moving_object"}
    assert cats["fixed_camera_moving_object"][0] == ["car"]
    turning = [math.radians(2.0 * t) for t in range(K)]  # 14 deg over the window
    assert "fixed_camera_moving_object" not in _cats(_scene("b", _still(), frames, turning))
    far = [[_ann("car", 60 + 2 * t, 3, ["vehicle.moving"])] for t in range(K)]
    assert "fixed_camera_moving_object" not in _cats(_scene("b", _still(), far))


def test_moving_camera_moving_object():
    two = [[_ann("c1", 10 + 2 * t, 3, ["vehicle.moving"]), _ann("c2", 12 + 2 * t, -3, ["vehicle.moving"])]
           for t in range(K)]
    cats = _cats(_scene("c", _driving(1.5), two))
    assert "moving_camera_moving_object" in cats and "static_scene_moving_camera" not in cats
    assert set(cats["moving_camera_moving_object"][0]) == {"c1", "c2"}
    one = [[f[0]] for f in two]
    assert "moving_camera_moving_object" not in _cats(_scene("c", _driving(1.5), one))


def test_occlusion_reappearance():
    vis = [4, 4, 1, 1, 3, 4, 4, 4]
    frames = [[_ann("ped", 10, 2, ["pedestrian.standing"], vis=v, pts=0 if v == 1 else 5,
                    cat="human.pedestrian.adult")] for v in vis]
    cats = _cats(_scene("d", _still(), frames))
    assert "occlusion_reappearance" in cats
    inst, crit = cats["occlusion_reappearance"]
    assert inst == ["ped"] and crit["instances"][0]["frames"] == [1, 2, 4]
    gone = [[_ann("ped", 10, 2, vis=v)] for v in [4, 4, 4, 1, 1, 1, 1, 1]]
    assert "occlusion_reappearance" not in _cats(_scene("d", _still(), gone))
    lidar_only = [[_ann("ped", 10, 2, vis=4, pts=0 if t == 3 else 5)] for t in range(K)]
    assert "occlusion_reappearance" in _cats(_scene("d", _still(), lidar_only))


def test_short_event():
    attrs = ["vehicle.moving"] * 4 + ["vehicle.stopped"] * 4
    frames = [[_ann("car", 12 + (0.5 * t if t < 4 else 2.0), 3, [a])] for t, a in enumerate(attrs)]
    cats = _cats(_scene("e", _still(), frames))
    assert "short_event" in cats
    ev = cats["short_event"][1]["instances"][0]
    assert (ev["frame"], ev["from"], ev["to"]) == (4, "vehicle.moving", "vehicle.stopped")
    calm = [[_ann("car", 12, 3, ["vehicle.stopped" if t < 4 else "vehicle.parked"])] for t in range(K)]
    assert "short_event" not in _cats(_scene("e", _still(), calm))


def test_short_event_requires_observable_event():
    """Event frame (4) and the frame before (3): visibility bin >= 3, >= 5 LiDAR points, <= 30 m from ego."""
    attrs = ["vehicle.moving"] * 4 + ["vehicle.stopped"] * 4

    def frames(x=12.0, vis_before=4, pts_event=10):
        return [[_ann("car", x, 3, [a], vis=vis_before if t == 3 else 4, pts=pts_event if t == 4 else 10)]
                for t, a in enumerate(attrs)]

    ev = _cats(_scene("e", _still(), frames()))["short_event"][1]["instances"][0]
    assert (ev["frame"], ev["vis"], ev["lidar_pts"]) == (4, 4, 10)
    assert ev["dist_m"] == pytest.approx(math.hypot(12, 3), abs=1e-3) and ev["prev_vis"] == 4
    for bad in (dict(x=45.0), dict(vis_before=2), dict(pts_event=4)):
        assert "short_event" not in _cats(_scene("e", _still(), frames(**bad))), bad


def test_rare_category():
    frames = [[_ann("dog", 8, 1, cat="animal")] for _ in range(K)]
    rare = fs.rare_classes({"animal": 3, "vehicle.car": 997})
    assert set(rare) == {"animal"} and rare["animal"]["train_share"] == pytest.approx(0.003)
    assert _cats(_scene("f", _still(), frames), rare)["rare_category"][0] == ["dog"]
    assert "rare_category" not in _cats(_scene("f", _still(), frames), {})


def _pair_scenes(fx=10.0, b_last_vis=4, b_last_pts=10):
    """A, B end at ego-frame (fx, 3) in opposite directions; C is like A (same direction), 0.5 m away."""
    # A: ego at origin facing +x, car ends at ego-frame (fx, 3) heading +x.
    a = _scene("scene-a", _still(), [[_ann("ca", fx - 10 + 10 * t / (K - 1), 3, ["vehicle.moving"])]
                                     for t in range(K)])
    # B: ego at (100, 50) facing world +y; car moves world -y and ends at ego-frame (fx, 3) heading -x.
    b = _scene("scene-b", [(100.0, 50.0)] * K,
               [[_ann("cb", 97, 50 + fx + 10 - 10 * t / (K - 1), ["vehicle.moving"],
                      vis=b_last_vis if t == K - 1 else 4, pts=b_last_pts if t == K - 1 else 10)] for t in range(K)],
               ego_yaw=math.pi / 2)
    c = _scene("scene-c", _still(), [[_ann("cc", fx - 10 + 10 * t / (K - 1), 3 + 0.5 * t / (K - 1),
                                           ["vehicle.moving"])] for t in range(K)])
    return a, b, c


def test_same_final_position_pairs():
    a, b, c = _pair_scenes()
    rows, _ = fs.mine([a, b, c], K, 4, 2, {}, "v1.0-trainval")
    pair = [r for r in rows if r["category"] == fs.PAIR]
    assert len(pair) == 2
    assert pair[0]["pair_id"] == pair[1]["pair_id"] is not None
    assert len({r["scene_name"] for r in pair}) == 2 and "scene-b" in {r["scene_name"] for r in pair}
    crit = pair[0]["criteria"]
    assert crit["heading_diff_deg"] >= 90 and crit["final_pos_diff_m"] <= 2
    assert {crit["partner_id"], pair[0]["id"]} == {r["id"] for r in pair}
    assert crit["final_vis"] == 4 and crit["final_lidar_pts"] == 10
    rows_ac, _ = fs.mine([a, c], K, 4, 2, {}, "v1.0-trainval")
    assert not [r for r in rows_ac if r["category"] == fs.PAIR]


def test_pairs_require_visible_final_position():
    """Last annotation of the instance in each window: visibility bin >= 3, >= 5 LiDAR points, <= 30 m."""
    for bad in (dict(b_last_vis=2), dict(b_last_pts=4), dict(fx=35.0)):
        rows, _ = fs.mine(list(_pair_scenes(**bad)), K, 4, 2, {}, "v1.0-trainval")
        assert not [r for r in rows if r["category"] == fs.PAIR], bad


def test_select_deterministic_and_capped():
    scenes = [_scene(f"scene-{i}", _still(40), [[] for _ in range(40)]) for i in range(3)]
    cands = [fs.Cand("short_event", s, st, (), {}) for s in scenes for st in fs.window_starts(40, K, 4)]
    out = fs.select(cands, 5, K)
    assert len(out) == 5
    for s in scenes:
        starts = [c.start for c in out if c.scene.name == s.name]
        assert len(starts) <= 2 and all(abs(x - y) >= K for x in starts for y in starts if x != y)
    shuffled = cands[:]
    random.Random(0).shuffle(shuffled)
    assert [(c.scene.name, c.start) for c in fs.select(shuffled, 5, K)] == [(c.scene.name, c.start) for c in out]
    keys = [fs.bucket(f"short_event:{c.scene.name}:{c.scene.keyframes[c.start].sample_token}") for c in out]
    assert keys == sorted(keys)


def test_missing_modality_rows():
    sc = _scene("scene-m", _still(), [[] for _ in range(K)])
    rows = [fs.make_row(fs.Cand(cat, sc, 0, ("x",), {}), K, "v1.0-trainval")
            for cat in ("fixed_camera_moving_object", "short_event")]
    miss = fs.missing_modality_rows(rows, 40, K)
    assert len(miss) == 2 * len(fs.MISSING_VARIANTS)
    by_variant = {r["criteria"]["variant"]: r for r in miss if r["criteria"]["base_category"] == "short_event"}
    lid = by_variant["lidar_dropped"]["mask"]
    assert lid["lidar"] == [False] * K and all(v == [True] * K for v in lid["cameras"].values())
    half = by_variant["half_cameras_dropped"]["mask"]["cameras"]
    assert sum(not v[0] for v in half.values()) == 3
    front = by_variant["cam_front_only"]["mask"]
    assert [c for c, v in front["cameras"].items() if v[0]] == ["CAM_FRONT"] and front["lidar"] == [False] * K
    for r in miss:  # missing = mask False; the referenced tokens are kept, never blanked
        assert r["lidar_sd_tokens"] == rows[0]["lidar_sd_tokens"] and r["camera_sd_tokens"] == rows[0]["camera_sd_tokens"]
        assert r["category"] == "missing_modality" and r["mask"] is not None
    assert len(fs.missing_modality_rows(rows, 4, K)) == len(fs.MISSING_VARIANTS)  # whole clips under the quota


# ----------------------------------------------------------------------------------------------- geometry

def test_box_corners_axes():
    c = fs.box_corners((1.0, 2.0, 3.0), (2.0, 4.0, 1.0), (1.0, 0.0, 0.0, 0.0))  # w=2 (y), l=4 (x), h=1 (z)
    ext = c.max(0) - c.min(0)
    assert np.allclose(ext, [4.0, 2.0, 1.0]) and np.allclose(c.mean(0), [1.0, 2.0, 3.0])
    assert np.allclose(c[:4, 0], 3.0)  # corners 0-3 are the front face (+x, heading)
    c90 = fs.box_corners((0.0, 0.0, 0.0), (2.0, 4.0, 1.0), _quat(math.pi / 2))
    assert np.allclose(c90.max(0) - c90.min(0), [2.0, 4.0, 1.0]) and np.allclose(c90[:4, 1], 2.0)


def test_box_projection_known_camera():
    # Ego at (100, 50) facing world +y; camera 1.5 m up looking along ego +x (nuScenes CAM_FRONT axes).
    ego = fs.pose_matrix((100.0, 50.0, 0.0), _quat(math.pi / 2))
    cam = fs.pose_matrix((0.0, 0.0, 1.5), (0.5, -0.5, 0.5, -0.5))
    Kc = torch.tensor([[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    c2w = torch.from_numpy(ego @ cam)
    centre = (100.0, 60.0, 1.5)  # 10 m ahead of the ego, at camera height
    uv, depth = project(torch.tensor([centre], dtype=torch.float64), Kc, c2w)
    assert torch.allclose(uv[0], torch.tensor([800.0, 450.0], dtype=torch.float64), atol=1e-6)
    assert depth[0] == pytest.approx(10.0)
    corners = torch.from_numpy(fs.box_corners(centre, (2.0, 4.0, 1.0), _quat(math.pi / 2)))
    uv, depth = project(corners, Kc, c2w)
    assert (depth > 0).all()
    # box heading = world +y = away from the camera: the front face (0-3, +l/2) is the far face
    assert depth[:4].min() > depth[4:].max()
    assert torch.allclose(depth[:4], torch.full((4,), 12.0, dtype=torch.float64))
    # corners on the box's left (+y in box frame = world -x = ego left) project left of centre
    assert (uv[[0, 3, 4, 7], 0] < 800).all() and (uv[[1, 2, 5, 6], 0] > 800).all()
    # top corners (+z) project above the centre (smaller v)
    assert (uv[[0, 1, 4, 5], 1] < 450).all()


# ----------------------------------------------------------------------------------------------- build script

def _write_nuscenes(root: Path, version="v1.0-trainval") -> list[Path]:
    d = root / version
    d.mkdir(parents=True)
    chans = list(fs.CAMERAS) + ["LIDAR_TOP", "RADAR_FRONT"]
    intr = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]
    t = {
        "sensor": [{"token": f"sen-{c}", "channel": c, "modality": "camera" if c.startswith("CAM") else "lidar"}
                   for c in chans],
        "calibrated_sensor": [{"token": f"cs-{c}", "sensor_token": f"sen-{c}", "translation": [0.0, 0.0, 1.5],
                               "rotation": [1.0, 0.0, 0.0, 0.0], "camera_intrinsic": intr if c.startswith("CAM") else []}
                              for c in chans],
        "category": [{"token": "cat-car", "name": "vehicle.car", "description": ""},
                     {"token": "cat-animal", "name": "animal", "description": ""}],
        "attribute": [{"token": "att-mov", "name": "vehicle.moving", "description": ""},
                      {"token": "att-park", "name": "vehicle.parked", "description": ""}],
        "visibility": [{"token": str(i), "level": f"v{i}", "description": ""} for i in range(1, 5)],
        "instance": [{"token": "i-val", "category_token": "cat-car", "nbr_annotations": 9},
                     {"token": "i-tr-car", "category_token": "cat-car", "nbr_annotations": 9},
                     {"token": "i-tr-dog", "category_token": "cat-animal", "nbr_annotations": 9}],
        "scene": [], "sample": [], "sample_data": [], "ego_pose": [], "sample_annotation": [],
    }
    files = []
    for name in ("scene-0003", "scene-0001"):  # official val, official train
        toks = [f"{name}-s{i}" for i in range(9)]
        t["scene"].append({"token": f"tok-{name}", "log_token": "log", "nbr_samples": 9, "name": name,
                           "first_sample_token": toks[0], "last_sample_token": toks[-1], "description": ""})
        for i, st in enumerate(toks):
            ts = 1_000_000 + 500_000 * i
            t["sample"].append({"token": st, "timestamp": ts, "scene_token": f"tok-{name}",
                                "prev": toks[i - 1] if i else "", "next": toks[i + 1] if i < 8 else ""})
            for c in chans:
                sd = f"{st}-{c}"
                fn = f"samples/{c}/{sd}.{'jpg' if c.startswith('CAM') else 'pcd.bin'}"
                t["sample_data"].append({"token": sd, "sample_token": st, "ego_pose_token": sd,
                                         "calibrated_sensor_token": f"cs-{c}", "timestamp": ts, "fileformat": "",
                                         "is_key_frame": True, "height": 900, "width": 1600, "filename": fn,
                                         "prev": "", "next": ""})
                t["ego_pose"].append({"token": sd, "timestamp": ts, "rotation": [1.0, 0.0, 0.0, 0.0],
                                      "translation": [0.02 * (i % 2), 0.0, 0.0]})
                files.append(root / fn)
            sweep = f"{st}-sweep"  # non-keyframe rows must be ignored
            t["sample_data"].append({"token": sweep, "sample_token": st, "ego_pose_token": sweep,
                                     "calibrated_sensor_token": "cs-CAM_FRONT", "timestamp": ts + 1,
                                     "fileformat": "", "is_key_frame": False, "height": 900, "width": 1600,
                                     "filename": f"sweeps/CAM_FRONT/{sweep}.jpg", "prev": "", "next": ""})
            t["ego_pose"].append({"token": sweep, "timestamp": ts + 1, "rotation": [1.0, 0.0, 0.0, 0.0],
                                  "translation": [0.0, 0.0, 0.0]})
            anns = [("i-val", [10 + 1.5 * i, 3.0, 0.5], ["att-mov"])] if name == "scene-0003" else \
                [("i-tr-car", [20.0, 3.0, 0.5], ["att-park"]), ("i-tr-dog", [8.0, 1.0, 0.3], [])]
            for inst, xyz, att in anns:
                t["sample_annotation"].append({
                    "token": f"{st}-{inst}", "sample_token": st, "instance_token": inst, "visibility_token": "4",
                    "attribute_tokens": att, "translation": xyz, "size": [2.0, 4.5, 1.6],
                    "rotation": [1.0, 0.0, 0.0, 0.0], "prev": "", "next": "", "num_lidar_pts": 20,
                    "num_radar_pts": 0})
    for name, rows in t.items():
        (d / f"{name}.json").write_text(json.dumps(rows, indent=2))
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    return files


def test_build_end_to_end(tmp_path):
    bf = _load(ROOT / "scripts" / "mavt4d" / "build_fixed_suite.py", "_build_fixed_suite")
    files = _write_nuscenes(tmp_path / "nus")
    out = tmp_path / "eval" / "suite.jsonl"
    summary = bf.build(tmp_path / "nus", "v1.0-trainval", out, 8, 4, 40, val_names=["scene-0003"],
                       train_names=["scene-0001"], git_commit="test")
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["category"] for r in rows] == ["fixed_camera_moving_object"] + ["missing_modality"] * 3
    r = rows[0]
    for key in ("suite", "id", "category", "source", "version", "split", "scene_name", "scene_token", "fps",
                "keyframe_only", "sample_tokens", "timestamps_us", "cameras", "camera_sd_tokens",
                "lidar_sd_tokens", "instance_tokens", "criteria", "thresholds", "provenance", "mask", "pair_id",
                "notes"):
        assert key in r, key
    assert r["suite"] == fs.SUITE == "fixed_suite_v2" and r["split"] == "val" and r["scene_name"] == "scene-0003"
    assert len(r["sample_tokens"]) == 8 and r["instance_tokens"] == ["i-val"]
    assert r["lidar_sd_tokens"][0] == "scene-0003-s0-LIDAR_TOP"
    assert summary["counts"]["fixed_camera_moving_object"] == 1 and summary["counts"]["static_scene_moving_camera"] == 0
    assert summary["train_instances_by_category"] == {"animal": 1, "vehicle.car": 1}
    assert summary["val_scene_count"] == 1 and summary["git_commit"] == "test"
    assert out.with_suffix(".summary.json").exists() and out.with_suffix(".cache.json").exists()
    cache = json.loads(out.with_suffix(".cache.json").read_text())
    assert "scene-0003-s4-CAM_FRONT" in cache["sample_data"] and "scene-0003-s4" in cache["annotations"]

    with pytest.raises(ValueError):  # val and train must be disjoint
        bf.build(tmp_path / "nus", "v1.0-trainval", out, 8, 4, 40, val_names=["scene-0003"],
                 train_names=["scene-0003"], git_commit="test")
    missing = next(f for f in files if "s2-CAM_BACK" in f.name)
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="CAM_BACK"):
        bf.build(tmp_path / "nus", "v1.0-trainval", out, 8, 4, 40, val_names=["scene-0003"],
                 train_names=["scene-0001"], git_commit="test")


# ----------------------------------------------------------------------------------------------- renderer

def _two_camera_cache_and_row(category: str):
    """CAM_FRONT looks along ego +x, CAM_BACK along ego -x. The key car is ahead at every frame except 6,
    where it is behind the ego."""
    intr = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]
    rot = {"CAM_FRONT": [0.5, -0.5, 0.5, -0.5], "CAM_BACK": [0.5, -0.5, -0.5, 0.5]}
    samples = [f"s{t}" for t in range(K)]
    cache = {"sample_data": {}, "ego_pose": {"ego": {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}},
             "calibrated_sensor": {f"cs-{c}": {"translation": [0.0, 0.0, 1.5], "rotation": rot[c], "camera_intrinsic": intr}
                                   for c in rot},
             "annotations": {}}
    for t, s in enumerate(samples):
        for c in rot:
            cache["sample_data"][f"{c}-{t}"] = {"ego_pose": "ego", "cs": f"cs-{c}", "width": 1600, "height": 900,
                                                "filename": ""}
        cache["annotations"][s] = [{"instance": "car", "xyz": [-10.0 if t == 6 else 10.0, 0.0, 1.5],
                                    "size": [2.0, 4.5, 1.6], "rotation": [1.0, 0.0, 0.0, 0.0]}]
    crit = {"instances": [{"instance": "car", "frame": 6}]} if category == fs.EVENT else {}
    row = {"id": f"{category}:x:00", "category": category, "cameras": list(rot), "mask": None,
           "sample_tokens": samples, "camera_sd_tokens": {c: [f"{c}-{t}" for t in range(K)] for c in rot},
           "instance_tokens": ["car"], "criteria": crit}
    return cache, row


def test_renderer_reference_frame():
    rf = _load(ROOT / "scripts" / "mavt4d" / "render_fixed_suite.py", "_render_fixed_suite")
    _, ev = _two_camera_cache_and_row(fs.EVENT)
    _, pair = _two_camera_cache_and_row(fs.PAIR)
    _, fixed = _two_camera_cache_and_row(fs.FIXED)
    miss = {**ev, "id": "missing_modality:lidar_dropped:" + ev["id"], "category": fs.MISSING,
            "criteria": {"base_id": ev["id"], "base_category": fs.EVENT, "variant": "lidar_dropped"}}
    by_id = {ev["id"]: ev}
    assert rf.reference_frame(ev, by_id) == 6 and rf.reference_frame(pair, by_id) == K - 1
    assert rf.reference_frame(miss, by_id) == 6 and rf.reference_frame(fixed, by_id) is None


def test_renderer_camera_at_event_frame():
    rf = _load(ROOT / "scripts" / "mavt4d" / "render_fixed_suite.py", "_render_fixed_suite")
    cache, ev = _two_camera_cache_and_row(fs.EVENT)
    cam, ref, _ = rf.choose_camera(ev, cache, rf.reference_frame(ev, {}))
    assert (cam, ref) == ("CAM_BACK", 6)  # the event frame, not the middle frame
    cache, fixed = _two_camera_cache_and_row(fs.FIXED)
    cam, ref, _ = rf.choose_camera(fixed, cache, rf.reference_frame(fixed, {}))
    assert (cam, ref) == ("CAM_FRONT", K // 2)
