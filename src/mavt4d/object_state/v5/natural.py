"""Versioned natural-source adapters. Targets and observations remain separate.

Web split keys reuse mavt-split-v1. nuScenes reads keyframes only; no temporal
interpolation, pseudo masks, or zero-valued supervision for absent geometry.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from mavt.data.splits import assign, bucket, group_key, video_source

SUPPORTED_SOURCES = {"openimages", "openvid", "nuscenes", "dl3dv", "arkitscenes", "co3d", "omniworld_epic"}
MIXTURE = {"openimages": .20, "openvid": .25, "nuscenes": .55}


def mixture_counts(total, weights=None):
    weights = MIXTURE if weights is None else weights
    if not weights or set(weights) - SUPPORTED_SOURCES or any(not math.isfinite(w) or w <= 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1., abs_tol=1e-8):
        raise ValueError("positive supported source weights must sum to one")
    if total < len(weights):
        raise ValueError("effective batch must contain all configured sources")
    counts = {s: int(total * w) for s, w in weights.items()}
    for source in sorted(weights, key=lambda s: total * weights[s] - counts[s], reverse=True)[:total - sum(counts.values())]:
        counts[source] += 1
    if not all(counts.values()):
        raise ValueError("source weight too small for effective batch")
    return counts


def nuscenes_metadata_fingerprint(root, version, official_names):
    """Pin every consumed table and the resolved official scene-name split."""
    from mavt4d.data.nuscenes import _TABLES
    from .adapters.manifest import sha256
    payload = {"version": version,
               "tables": {name: sha256(Path(root) / version / (name + ".json")) for name in sorted(_TABLES)},
               "official_scene_names": sorted(set(official_names))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observation_schedule(profile, seed=17):
    rng = random.Random(seed)
    if profile == "S":
        return [True] * 3
    if profile == "L":
        return [True] * 6
    if profile != "G":
        raise ValueError("unknown observation profile")
    observed, gap = rng.randint(4, 6), rng.randint(2, 6)
    left = rng.randint(2, observed - 1)
    return [True] * left + [False] * gap + [True] * (observed - left)


def sample_plan(times, profile, *, seed=17, horizon=6):
    """Tubelet endpoints; shorter sources have masked horizons, never duplicates."""
    if len(times) == 1:
        return {"prefix": [0], "observed": [True], "future": [], "profile": "image"}
    if len(times) < 3:
        raise ValueError("video needs observation and a real held-out frame")
    schedule = observation_schedule(profile, seed)
    endpoints = list(range(1, len(times), 2))
    # Reserve at least one future endpoint. G must retain a genuine revisit;
    # if footage is too short use S and report the realized profile.
    if len(endpoints) <= len(schedule):
        schedule = [True] * min(3, max(1, len(endpoints) - 1))
        profile = "S-short"
    prefix = endpoints[:len(schedule)]
    future = []
    for h in range(1, horizon + 1):
        desired = times[prefix[-1]] + .5 * h
        candidates = torch.arange(len(times))[torch.arange(len(times)) > prefix[-1]]
        if not len(candidates):
            break
        candidate = int(candidates[(times[candidates] - desired).abs().argmin()])
        if abs(float(times[candidate] - desired)) <= .05 and candidate not in future:
            future.append(candidate)
    return {"prefix": prefix, "observed": schedule, "future": future, "profile": profile}


def validate_clip(clip):
    rgb, times, valid = clip["rgb"], clip["times"], clip["rgb_valid"]
    if rgb.ndim != 5 or rgb.shape[2] != 3 or valid.shape != (*rgb.shape[:2], 1, *rgb.shape[-2:]):
        raise ValueError("invalid clip tensors")
    if len(times) != len(rgb) or not torch.isfinite(times).all() or (len(times) > 1 and not (times[1:] > times[:-1]).all()):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB must be finite in [0,1]")
    for name in ("camera_valid", "metric_valid"):
        if name in clip and (clip[name].shape != rgb.shape[:2] or clip[name].dtype != torch.bool):
            raise ValueError(f"invalid {name} tensor")
    for name, shape in (("intrinsics", (3, 3)), ("camera_c2w", (4, 4))):
        if name in clip and (clip[name].shape != (*rgb.shape[:2], *shape) or not torch.isfinite(clip[name]).all()):
            raise ValueError(f"invalid {name} tensor")
    for name in ("depth", "input_depth"):
        if name in clip:
            value = clip[name]
            if value.shape != valid.shape or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"invalid {name} tensor")
            mask = clip.get(name + "_valid")
            if mask is None or mask.shape != value.shape or mask.dtype != torch.bool:
                raise ValueError(f"missing {name} validity")
    return clip


def _resize_image(image, height, width):
    # Exact anisotropic resize; calibrated K applies the same x/y factors.
    return torch.from_numpy(np.array(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), copy=True)).permute(2, 0, 1).float() / 255


def _media_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError("manifest media missing or outside approved root")
    return path


class NaturalSources:
    """Group-first sampler with bounded per-draw decoding; no FP32 corpus cache."""
    def __init__(self, config, *, split="train"):
        if split not in ("train", "dev", "val"):
            raise ValueError("unknown split")
        self.config, self.split = config, split
        self.root = Path(config["root"]).resolve()
        self.height, self.width = config.get("height", 192), config.get("width", 256)
        self.groups, self.manifest_hashes = {}, {}
        configured = {s for s in SUPPORTED_SOURCES if s + "_manifest" in config or (s == "nuscenes" and "nuscenes_root" in config)}
        if configured != set(MIXTURE) and "mixture" not in config:
            raise ValueError("non-default inventory requires an explicit frozen mixture")
        self.mixture = config.get("mixture", MIXTURE)
        if set(self.mixture) != configured:
            raise ValueError("mixture must exactly match configured sources; missing data is never reassigned")
        mixture_counts(max(256, len(configured)), self.mixture)
        for source, modality, fraction in (("openimages", "image", .005), ("openvid", "video", .02)):
            if source + "_manifest" not in config:
                continue
            manifest = _media_path(self.root, config[source + "_manifest"])
            self.manifest_hashes[source] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            groups = defaultdict(list)
            with manifest.open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = group_key(modality, row["path"])
                    official = assign(key, fraction)
                    internal_dev = bucket(key, "mavt-v5-internal-dev-v1") < .02
                    keep = official == "val" if split == "val" else official == "train" and internal_dev == (split == "dev")
                    if keep:
                        groups[key].append(row)
            if not groups:
                raise ValueError(f"empty {source} {split} split")
            self.groups[source] = dict(sorted(groups.items()))
        self.keys = {s: list(g) for s, g in self.groups.items()}
        self.video_strata = {tag: [k for k, rows in self.groups.get("openvid", {}).items()
                                   if (video_source(rows[0]["path"])[1] == "celebv") == (tag == "celebv")]
                             for tag in ("celebv", "other")}
        if "openvid" in self.groups and not all(self.video_strata.values()):
            raise ValueError("OpenVid cap requires both CelebV and non-CelebV groups")
        if "nuscenes_root" in config:
            self._init_nuscenes()
        from .adapters.manifest import SCHEMA, SOURCES, safe_path
        for source in SOURCES:
            if source + "_manifest" not in config: continue
            manifest = _media_path(self.root, config[source + "_manifest"])
            self.manifest_hashes[source] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            groups = defaultdict(list)
            with manifest.open() as stream:
                for line in stream:
                    if not line.strip(): continue
                    row = json.loads(line)
                    if row.get("schema") != SCHEMA or row.get("source") != source:
                        raise ValueError("normalized manifest source/schema mismatch")
                    if row.get("split") != split: continue
                    archive = safe_path(self.root, row["archive"])
                    if archive.stat().st_size != row["archive_size"] or archive.stat().st_mtime_ns != row.get("archive_mtime_ns", archive.stat().st_mtime_ns): raise ValueError("frozen archive changed")
                    if config.get("verify_archive_sha256", False):
                        from .adapters.manifest import sha256
                        if sha256(archive) != row.get("archive_sha256"): raise ValueError("archive checksum mismatch")
                    groups[row["group"]].append(row)
            if not groups: raise ValueError(f"empty {source} {split} split")
            self.groups[source] = dict(sorted(groups.items()))
            self.keys[source] = list(self.groups[source])

    def _init_nuscenes(self):
        from mavt4d.data.nuscenes import NuScenesTables, load_official_split
        root = (self.root / self.config["nuscenes_root"]).resolve()
        if not root.is_relative_to(self.root):
            raise ValueError("nuScenes root outside approved root")
        metadata_root = root if (root / "v1.0-trainval").is_dir() else root / "metadata"
        self.nusc = NuScenesTables(metadata_root, "v1.0-trainval")
        self.nusc.dataroot = root
        official = set(load_official_split("v1.0-trainval", "val" if self.split == "val" else "train"))
        groups = {}
        for token, scene in self.nusc.scene.items():
            if scene["name"] not in official:
                continue
            dev = bucket(scene["log_token"], "mavt-v5-nusc-log-dev-v1") < .10
            if self.split != "val" and dev != (self.split == "dev"):
                continue
            samples = self.nusc.scene_samples(token)
            if len(samples) >= 4:
                groups[token] = samples
        if not groups:
            raise ValueError("empty nuScenes split")
        self.groups["nuscenes"] = dict(sorted(groups.items()))
        self.keys["nuscenes"] = list(self.groups["nuscenes"])
        # Metadata provenance pins all table files without exposing paths to W&B.
        self.manifest_hashes["nuscenes"] = nuscenes_metadata_fingerprint(metadata_root, "v1.0-trainval", official)

    def inventory(self):
        return {"schema": "natural-v5-seven-source-v1", "split": self.split,
                "groups": {s: len(g) for s, g in self.groups.items()},
                "records": {s: sum(map(len, g.values())) for s, g in self.groups.items()},
                "manifest_sha256": self.manifest_hashes,
                "mixture": self.mixture, "synthetic_main": False, "input_views": 1,
                "nuscenes_keyframes_only": True, "celebv_draw_cap": .5}

    def draw(self, source, seed, *, profile="S", horizon=6, plan_seed=17, group=None):
        rng = random.Random(seed)
        keys = self.keys[source]
        if source == "openvid":
            keys = self.video_strata["celebv" if seed % 2 == 0 else "other"]
        if source == "co3d" and group is None:
            categories = sorted({rows[0].get("category", "unknown") for rows in self.groups[source].values()})
            category = rng.choice(categories)
            keys = [k for k in keys if self.groups[source][k][0].get("category", "unknown") == category]
        key = group if group is not None else keys[rng.randrange(len(keys))]
        if key not in self.groups[source]: raise ValueError("unknown frozen source group")
        if source == "nuscenes":
            clip = self._nuscenes(key, rng, profile, horizon)
        elif source not in ("openimages", "openvid"):
            from .adapters.decode import decode_record
            row = rng.choice(self.groups[source][key])
            clip = decode_record(self.root, row, rng, self.height, self.width,
                                 max_frames=2 * (len(observation_schedule(profile, plan_seed)) + horizon))
        else:
            row = rng.choice(self.groups[source][key])
            path = _media_path(self.root, row["path"])
            if source == "openimages":
                with Image.open(path) as im:
                    rgb = _resize_image(im, self.height, self.width)[None, None]
                clip = {"rgb": rgb, "times": torch.zeros(1, dtype=torch.float64)}
            else:
                clip = self._video(path, rng, profile, horizon, plan_seed=plan_seed)
        clip.setdefault("rgb_valid", torch.ones(*clip["rgb"].shape[:2], 1, self.height, self.width, dtype=torch.bool))
        clip.update(source=source, group=key, seed=seed)
        if clip.get("time_valid", True) and len(clip["times"]) != 2:
            clip["plan"] = sample_plan(clip["times"], profile, seed=plan_seed, horizon=horizon)
        else:
            clip["plan"] = {"prefix": list(range(min(len(clip["times"]), 6))),
                            "observed": [True] * min(len(clip["times"]), 6),
                            "future": [], "profile": "unordered_views", "temporal_supervision": False}
        if not clip.get("time_valid", True):
            clip["plan"]["future"] = []
            clip["plan"]["temporal_supervision"] = False
        clip.setdefault("camera_valid", torch.zeros(*clip["rgb"].shape[:2], dtype=torch.bool))
        clip.setdefault("metric_valid", torch.zeros(*clip["rgb"].shape[:2], dtype=torch.bool))
        clip.setdefault("depth_provenance", "missing")
        clip.setdefault("time_valid", source != "openimages")
        clip.setdefault("timestamp_provenance", "single_image")
        clip.setdefault("scene_static", False)
        return validate_clip(clip)

    def _video(self, path, rng, profile, horizon, *, plan_seed=17):
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or count < 4:
                raise ValueError("video lacks usable frame count/rate")
            wanted = 2 * (len(observation_schedule(profile, plan_seed)) + horizon)
            # Round each target time independently: rounding a fixed stride at
            # 30fps drifts beyond the 50ms target tolerance after a few steps.
            offsets = sorted(set(round(index * .25 * fps) for index in range(wanted)))
            start = rng.randint(0, max(0, count - 1 - offsets[-1]))
            indices = [start + offset for offset in offsets if start + offset < count]
            images, timestamps = [], []
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("video frame decode failed")
                images.append(_resize_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), self.height, self.width))
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            pts = np.asarray(timestamps)
            actual_pts = np.isfinite(pts).all() and (np.diff(pts) > 0).all()
            if not actual_pts:
                pts = np.asarray(indices, dtype=np.float64) / fps
            return {"rgb": torch.stack(images)[:, None], "times": torch.tensor(pts - pts[0], dtype=torch.float64),
                    "timestamp_provenance": "decoder_pts" if actual_pts else "frame_index_fps"}
        finally:
            cap.release()

    def _nuscenes(self, key, rng, profile, horizon):
        from mavt4d.data.nuscenes import CAM6, resize_crop
        tb, samples = self.nusc, self.groups["nuscenes"][key]
        wanted = 2 * (len(observation_schedule(profile, 17)) + horizon)
        start = rng.randint(0, max(0, len(samples) - wanted))
        chosen = samples[start:start + wanted]
        first_view = rng.randrange(6)
        channels = (CAM6[first_view], CAM6[(first_view + rng.randrange(1, 6)) % 6])
        images, intrinsics, cameras, times, depths, input_depths = [], [], [], [], [], []
        first_sd = tb.keyframe_data[chosen[0]][channels[0]]
        gauge = np.linalg.inv(tb.ego_to_global(first_sd) @ tb.sensor_to_ego(first_sd))
        for sample in chosen:
            im_t, k_t, pose_t, depth_t, input_t = [], [], [], [], []
            lidar_key = tb.keyframe_data[sample]["LIDAR_TOP"]
            lidar_row = tb.sample_data[lidar_key]
            lidar = np.fromfile(tb.dataroot / lidar_row["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
            lidar_world = tb.ego_to_global(lidar_key) @ tb.sensor_to_ego(lidar_key)
            points = lidar @ lidar_world[:3, :3].T + lidar_world[:3, 3]
            # Stable per-return holdout. No target return is an encoder input.
            salt = int(hashlib.sha1(("nuscenes/" + lidar_key).encode()).hexdigest()[:8], 16)
            holdout = np.random.default_rng(salt).random(len(points)) < .2
            for channel in channels:
                sd = tb.keyframe_data[sample][channel]
                row = tb.sample_data[sd]
                with Image.open(tb.dataroot / row["filename"]) as im:
                    original_w, original_h = im.size
                    im_t.append(_resize_image(im, self.height, self.width))
                k = tb.raw_intrinsics(sd).copy()
                k[0] *= self.width / original_w
                k[1] *= self.height / original_h
                camera_world = tb.ego_to_global(sd) @ tb.sensor_to_ego(sd)
                k_t.append(torch.tensor(k, dtype=torch.float32))
                pose_t.append(torch.tensor(gauge @ camera_world, dtype=torch.float32))
                camera_points = (points - camera_world[:3, 3]) @ camera_world[:3, :3]
                pixel = camera_points @ k.T
                z = camera_points[:, 2]
                uv = pixel[:, :2] / np.maximum(pixel[:, 2:], 1e-8)
                x, y = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
                eligible = (z > .1) & (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
                # Sparse z-buffer separately on disjoint input/target return sets.
                for select, output in ((eligible & holdout, depth_t), (eligible & ~holdout, input_t)):
                    flat = np.full(self.height * self.width, np.inf, np.float32)
                    np.minimum.at(flat, y[select] * self.width + x[select], z[select])
                    flat[~np.isfinite(flat)] = 0
                    output.append(torch.from_numpy(flat.reshape(1, self.height, self.width)))
            images.append(torch.stack(im_t)); intrinsics.append(torch.stack(k_t)); cameras.append(torch.stack(pose_t))
            depths.append(torch.stack(depth_t)); input_depths.append(torch.stack(input_t))
            times.append(tb.sample_data[tb.keyframe_data[sample][channels[0]]]["timestamp"] / 1e6)
        rgb = torch.stack(images)
        return {"rgb": rgb, "times": torch.tensor(np.asarray(times) - times[0], dtype=torch.float64),
                "intrinsics": torch.stack(intrinsics), "camera_c2w": torch.stack(cameras),
                "depth": torch.stack(depths), "input_depth": torch.stack(input_depths),
                "geometry_valid": torch.ones(len(rgb), 2, dtype=torch.bool),
                "camera_valid": torch.ones(len(rgb), 2, dtype=torch.bool),
                "metric_valid": torch.ones(len(rgb), 2, dtype=torch.bool),
                "depth_valid": torch.stack(depths) > 0,
                "input_depth_valid": torch.stack(input_depths) > 0,
                "timestamp_provenance": "sensor_keyframe", "camera_channels": channels,
                "depth_provenance": "measured", "depth_detail": "sparse_lidar_no_dynamic_transport"}
