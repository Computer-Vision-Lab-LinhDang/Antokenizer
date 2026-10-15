"""A tiny nuScenes on disk: 2 scenes x 2 keyframes, 12 Hz cameras, 20 Hz LIDAR_TOP sweeps.

Ground truth is kept as yaw/translation and built into matrices with explicit numpy formulas, so tests
check the reader's quaternion handling and transform chain against an independent path. The two scenes'
``prev``/``next`` chains are linked across the boundary (real nuScenes cuts them) so the reader's
scene guard is actually exercised.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "v1.0-mini"
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT")
SCENE_NAMES = ("scene-0001", "scene-0002")
IMG_W, IMG_H = 1600, 900
STRIPE_U = (1000, 1016)  # bright columns [1000, 1016) in every image
K_RAW = np.array([[1266.4, 0.0, 816.3], [0.0, 1266.4, 491.5], [0.0, 0.0, 1.0]])
CAM_YAW = {"CAM_FRONT": 0.0, "CAM_FRONT_LEFT": math.radians(55.0)}
CAM_OFFSET_S = {"CAM_FRONT": -0.0355, "CAM_FRONT_LEFT": -0.045}  # camera keyframe vs LiDAR keyframe
CAM_DT, LIDAR_DT, KEY_DT = 1 / 12, 0.05, 0.5
T_START_US = 1_532_402_927_000_000
SCENE_GAP_S = 0.6  # scene 1 starts this long after scene 0
EGO_P0, EGO_YAW0, EGO_SPEED, EGO_YAW_RATE = np.array([400.0, 1100.0, 0.0]), 0.3, 5.0, 0.2
# instance "A" moves and is in both keyframes; "B" is only in keyframe 0.
BOX_A = [((420.0, 1105.0, 1.0), (1.9, 4.5, 1.6), 0.1), ((423.0, 1106.0, 1.0), (1.9, 4.5, 1.6), 0.3)]
BOX_B = ((410.0, 1110.0, 0.9), (0.6, 0.7, 1.7), -2.0)

# camera frame (x right, y down, z forward) -> sensor frame (x forward, y left, z up)
R_CAM_AXES = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
Q_CAM_AXES = (0.5, -0.5, 0.5, -0.5)


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def se3(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _q_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _q_mul(a, b) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def ego_at(t_s: float) -> tuple[np.ndarray, float]:
    """Ego translation and yaw at ``t_s`` seconds after T_START_US."""
    return EGO_P0 + EGO_SPEED * t_s * np.array([math.cos(EGO_YAW0), math.sin(EGO_YAW0), 0.0]), \
        EGO_YAW0 + EGO_YAW_RATE * t_s


SENSOR_T = {"CAM_FRONT": (1.7, 0.0, 1.5), "CAM_FRONT_LEFT": (1.5, 0.5, 1.5), "LIDAR_TOP": (0.9, 0.0, 1.8)}
LIDAR_YAW = -math.pi / 2


def sensor_matrix(channel: str) -> np.ndarray:
    """sensor -> ego, from the yaw/axes ground truth."""
    if channel == "LIDAR_TOP":
        return se3(rot_z(LIDAR_YAW), SENSOR_T[channel])
    return se3(rot_z(CAM_YAW[channel]) @ R_CAM_AXES, SENSOR_T[channel])


def write_fake_nuscenes(root: Path) -> dict:
    """Write the dataset under ``root`` and return ground truth for the tests.

    Returns keys: ``root``, ``version``, ``ego`` {sd_token: ego->global 4x4}, ``time_us`` {sd_token: int},
    ``scene_of`` {sd_token: scene name}, ``chain`` {(scene name, channel): [sd tokens]},
    ``keyframes`` {sample_token: {channel: sd_token}}, ``samples`` {scene name: [sample tokens]},
    ``boxes`` {sample_token: {instance_token: (center, size, yaw)}} (global), ``n_points`` {sd_token: int},
    ``instance_index`` {instance_token: int}.
    """
    root = Path(root)
    meta = root / VERSION
    meta.mkdir(parents=True, exist_ok=True)
    channels = (*CAMERAS, "LIDAR_TOP")
    tab = {k: [] for k in ("scene", "sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor",
                           "sample_annotation", "instance")}
    gt = {"root": root, "version": VERSION, "ego": {}, "time_us": {}, "scene_of": {}, "chain": {},
          "keyframes": {}, "samples": {}, "boxes": {}, "n_points": {}, "instance_index": {}}
    for ch in channels:
        tab["sensor"].append({"token": f"sensor-{ch}", "channel": ch,
                              "modality": "lidar" if ch == "LIDAR_TOP" else "camera"})
        q = _q_yaw(LIDAR_YAW) if ch == "LIDAR_TOP" else _q_mul(_q_yaw(CAM_YAW[ch]), Q_CAM_AXES)
        tab["calibrated_sensor"].append({
            "token": f"cs-{ch}", "sensor_token": f"sensor-{ch}", "translation": list(SENSOR_T[ch]),
            "rotation": list(q), "camera_intrinsic": [] if ch == "LIDAR_TOP" else K_RAW.tolist()})

    img = np.full((IMG_H, IMG_W, 3), 60, np.uint8)
    img[:, STRIPE_U[0]:STRIPE_U[1]] = 250
    jpeg = Image.fromarray(img)
    last_sd = {ch: None for ch in channels}  # previous sd record per channel, linked across scenes
    n_kf = 2
    for si, name in enumerate(SCENE_NAMES):
        t_scene = si * (KEY_DT * (n_kf - 1) + SCENE_GAP_S)
        sample_toks = [f"sample-{si}-{k}" for k in range(n_kf)]
        gt["samples"][name] = sample_toks
        tab["scene"].append({"token": f"scene-tok-{si}", "log_token": "log-0", "nbr_samples": n_kf,
                             "first_sample_token": sample_toks[0], "last_sample_token": sample_toks[-1],
                             "name": name, "description": "fake"})
        for k, tok in enumerate(sample_toks):
            tab["sample"].append({"token": tok, "timestamp": _us(t_scene + k * KEY_DT),
                                  "prev": sample_toks[k - 1] if k else "",
                                  "next": sample_toks[k + 1] if k + 1 < n_kf else "",
                                  "scene_token": f"scene-tok-{si}"})
            gt["keyframes"][tok] = {}
        for ch in channels:
            if ch == "LIDAR_TOP":
                times = [t_scene + j * LIDAR_DT for j in range(round(KEY_DT * (n_kf - 1) / LIDAR_DT) + 1)]
                per_key = round(KEY_DT / LIDAR_DT)
            else:
                times = [t_scene + CAM_OFFSET_S[ch] + j * CAM_DT for j in range(round(KEY_DT * (n_kf - 1) / CAM_DT) + 1)]
                per_key = round(KEY_DT / CAM_DT)
            chain = []
            for j, t in enumerate(times):
                key = j % per_key == 0
                k = j // per_key if key else min(j // per_key + 1, n_kf - 1)  # sweeps point at next keyframe
                tok = f"sd-{si}-{ch}-{j}"
                folder = "samples" if key else "sweeps"
                ext = "pcd.bin" if ch == "LIDAR_TOP" else "jpg"
                fn = f"{folder}/{ch}/fake__{ch}__{_us(t)}.{ext}"
                (root / fn).parent.mkdir(parents=True, exist_ok=True)
                if ch == "LIDAR_TOP":
                    n = 100 + 10 * si + j  # unique count per sweep identifies the file
                    pts = np.zeros((n, 5), np.float32)
                    pts[:, 0] = np.linspace(-30.0, 30.0, n)
                    pts[:, 1] = np.linspace(20.0, -20.0, n)
                    pts[:, 3] = np.arange(n) % 255
                    pts.tofile(root / fn)
                    gt["n_points"][tok] = n
                else:
                    jpeg.save(root / fn, quality=90)
                rec = {"token": tok, "sample_token": sample_toks[k], "ego_pose_token": tok,
                       "calibrated_sensor_token": f"cs-{ch}", "timestamp": _us(t),
                       "fileformat": "pcd" if ch == "LIDAR_TOP" else "jpg", "is_key_frame": key,
                       "height": 0 if ch == "LIDAR_TOP" else IMG_H, "width": 0 if ch == "LIDAR_TOP" else IMG_W,
                       "filename": fn, "prev": last_sd[ch]["token"] if last_sd[ch] else "", "next": ""}
                if last_sd[ch]:
                    last_sd[ch]["next"] = tok
                last_sd[ch] = rec
                tab["sample_data"].append(rec)
                p, yaw = ego_at(t)
                q = _q_yaw(yaw)
                tab["ego_pose"].append({"token": tok, "timestamp": _us(t), "rotation": list(q),
                                        "translation": p.tolist()})
                gt["ego"][tok] = se3(rot_z(yaw), p)
                gt["time_us"][tok] = _us(t)
                gt["scene_of"][tok] = name
                if key:
                    gt["keyframes"][sample_toks[k]][ch] = tok
                chain.append(tok)
            gt["chain"][(name, ch)] = chain
        _write_boxes(tab, gt, si, sample_toks)
    for name, rows in tab.items():
        (meta / f"{name}.json").write_text(json.dumps(rows))
    return gt


def _write_boxes(tab: dict, gt: dict, si: int, sample_toks: list[str]) -> None:
    inst_a, inst_b = f"inst-A-{si}", f"inst-B-{si}"
    anns_a = [f"ann-A-{si}-{k}" for k in range(2)]
    for inst, anns in ((inst_a, anns_a), (inst_b, [f"ann-B-{si}-0"])):
        gt["instance_index"][inst] = len(tab["instance"])
        tab["instance"].append({"token": inst, "category_token": "cat-0", "nbr_annotations": len(anns),
                                "first_annotation_token": anns[0], "last_annotation_token": anns[-1]})
    rows = [(anns_a[0], inst_a, sample_toks[0], BOX_A[0], "", anns_a[1]),
            (anns_a[1], inst_a, sample_toks[1], BOX_A[1], anns_a[0], ""),
            (f"ann-B-{si}-0", inst_b, sample_toks[0], BOX_B, "", "")]
    for tok, inst, sample, (c, size, yaw), prev, nxt in rows:
        tab["sample_annotation"].append({
            "token": tok, "sample_token": sample, "instance_token": inst, "visibility_token": "4",
            "attribute_tokens": [], "translation": list(c), "size": list(size), "rotation": list(_q_yaw(yaw)),
            "prev": prev, "next": nxt, "num_lidar_pts": 10, "num_radar_pts": 0})
        gt["boxes"].setdefault(sample, {})[inst] = (np.array(c), np.array(size), yaw)


def _us(t_s: float) -> int:
    return T_START_US + round(t_s * 1e6)
