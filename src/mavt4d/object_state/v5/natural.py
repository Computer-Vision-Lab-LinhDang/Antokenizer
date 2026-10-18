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
        raise ValueError("no eligible temporal profile: video needs observation and a real held-out frame")
    endpoints = list(range(1, len(times), 2))
    schedules = {name: observation_schedule(name, seed) for name in ('S','L','G')}
    eligible = [name for name, flags in schedules.items() if len(endpoints) >= len(flags)
                and float(times[-1] - times[endpoints[len(flags)-1]]) >= .5 * horizon - .05]
    if not eligible:
        raise ValueError('no eligible temporal profile for requested horizon')
    if profile not in eligible:
        # Re-sample only among feasible profiles; do not shorten the observation contract.
        profile = random.Random(seed ^ 0x50524F46).choices(eligible,
                    weights=[{'S':.4,'L':.3,'G':.3}[name] for name in eligible])[0]
    schedule = schedules[profile]
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


def static_view_plan(count, profile, *, seed=17):
    """Camera-index schedule; held-out views never become tubelet inputs."""
    if count < 3:
        raise ValueError("static multiview requires two observations and a held-out view")
    rng = random.Random(seed)
    eligible = ["S"] + (["L"] if count >= 4 else []) + (["G"] if count >= 5 else [])
    if profile not in eligible:
        profile = rng.choice(eligible)
    observed = 2 if profile == "S" else 3
    gap = rng.randint(1, min(3, count - observed - 1)) if profile == "G" else 0
    flags = [True, True] + [False] * gap + [True] if profile == "G" else [True] * observed
    return {"prefix": list(range(len(flags))), "observed": flags,
            "heldout": [len(flags)], "future": [], "profile": profile + "-views",
            "temporal_supervision": False}


def normalize_clip_gauge(clip):
    """Anchor all input/target geometry once, using only an admitted camera."""
    if "camera_c2w" not in clip:
        return clip
    from .gauge import normalize_camera_sequence
    valid = clip["camera_valid"]
    admitted = torch.zeros_like(valid)
    for frame, observed in zip(clip["plan"]["prefix"], clip["plan"]["observed"]):
        if observed:
            # The encoder receives camera metadata only for tubelet endpoints.
            admitted[frame, 0] = True
    if not bool((valid & admitted).any()):
        # Future-only camera labels cannot choose an inference origin.
        clip['camera_valid'] = torch.zeros_like(valid)
        clip['metric_valid'] = torch.zeros_like(valid)
        clip['gauge_provenance'] = 'unanchored_camera_metadata_unavailable'
        return clip
    poses, mapping = normalize_camera_sequence(clip["camera_c2w"], valid, admitted)
    clip["camera_c2w"] = poses
    clip["scene_to_external"] = clip.get("scene_to_external", torch.eye(4, dtype=mapping.dtype)) @ mapping
    if "object_to_world" in clip:
        clip["object_to_world"] = torch.linalg.inv(mapping) @ clip["object_to_world"]
    clip["gauge_provenance"] = "first_admitted_camera_se3_v1"
    return clip


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
        from .adapters.decode import static_capture_evidence
        static_records = {source: sum(static_capture_evidence(row) for rows in groups.values()
                                     for row in rows if isinstance(row,dict))
                          for source,groups in self.groups.items()}
        return {"schema": "natural-v5-seven-source-v1", "split": self.split,
                "groups": {s: len(g) for s, g in self.groups.items()},
                "records": {s: sum(map(len, g.values())) for s, g in self.groups.items()},
                "manifest_sha256": self.manifest_hashes,
                "mixture": self.mixture, "synthetic_main": False, "input_views": 1,
                "nuscenes_keyframes_only": True, "celebv_draw_cap": .5,
                "static_capture_protocol_records": static_records,
                "data_contract": "per_draw_endpoint_gauge_multiview_v2",
                "status_normalization": "per_example_present_classes_v1",
                "ray_bounds": "domain_gauge_v1"}

    def draw(self, source, seed, *, profile="S", horizon=6, plan_seed=17, group=None):
        for attempt in range(32):
            try:
                clip = self._draw_once(source, seed + attempt * 2000066, profile=profile,
                                       horizon=horizon, plan_seed=plan_seed, group=group)
            except ValueError as error:
                if 'no eligible temporal profile' not in str(error):
                    raise
                continue
            clip['draw_seed'] = seed
            clip['bridge_selected'] = random.Random(seed ^ 0x42524944).random() < .25
            clip['eligibility_resamples'] = attempt
            return clip
        raise ValueError(f'{source}: no eligible temporal profile in 32 group draws')

    def _draw_once(self, source, seed, *, profile="S", horizon=6, plan_seed=17, group=None):
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
            clip = self._nuscenes(key, rng, profile, horizon, plan_seed=plan_seed)
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
        clip.update(source=source, group=key, seed=seed, draw_seed=seed)
        clip["bridge_selected"] = random.Random(seed ^ 0x42524944).random() < .25
        if clip.get("time_valid", True):
            clip["plan"] = sample_plan(clip["times"], profile, seed=plan_seed, horizon=horizon)
        else:
            clip["plan"] = static_view_plan(len(clip["times"]), profile, seed=plan_seed)
        clip.setdefault("camera_valid", torch.zeros(*clip["rgb"].shape[:2], dtype=torch.bool))
        clip.setdefault("metric_valid", torch.zeros(*clip["rgb"].shape[:2], dtype=torch.bool))
        clip.setdefault("depth_provenance", "missing")
        clip.setdefault("time_valid", source != "openimages")
        clip.setdefault("timestamp_provenance", "single_image")
        clip.setdefault("scene_static", False)
        normalize_clip_gauge(clip)
        return validate_clip(clip)

    def capacity_draw(self, source, seed, *, horizon=6, profile="G", attempts=96):
        """Find a real maximum-profile draw, or fail with explicit missing coverage."""
        for attempt in range(attempts):
            candidate_seed = seed + attempt * 104729
            # G max schedule uses six observations and six skipped endpoints.
            plan_seed = (next(s for s in range(candidate_seed, candidate_seed + 2000)
                             if len(observation_schedule("G", s)) == 12) if profile == "G" else candidate_seed)
            clip = self.draw(source, candidate_seed, profile=profile, horizon=horizon, plan_seed=plan_seed)
            plan = clip["plan"]
            if source == "openimages":
                return clip
            if not clip.get("time_valid", True):
                if plan["profile"] == profile+"-views" and (profile != "G" or plan["observed"].count(False) == 3):
                    return clip
            elif plan["profile"] == profile and len(plan["future"]) == horizon and len(plan["prefix"]) == (12 if profile=="G" else 3):
                return clip
        raise ValueError(f"{source}: no proven maximum {profile}/H{horizon} capacity draw in {attempts} attempts")

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

    def _nuscenes(self, key, rng, profile, horizon, *, plan_seed=17):
        from mavt4d.data.nuscenes import CAM6, resize_crop
        tb, samples = self.nusc, self.groups["nuscenes"][key]
        wanted = 2 * (len(observation_schedule(profile, plan_seed)) + horizon)
        start = rng.randint(0, max(0, len(samples) - wanted))
        chosen = samples[start:start + wanted]
        first_view = rng.randrange(6)
        channels = (CAM6[first_view], CAM6[(first_view + rng.randrange(1, 6)) % 6])
        images, intrinsics, cameras, times, depths, input_depths = [], [], [], [], [], []
        mask_boxes = {}
        first_sd = tb.keyframe_data[chosen[0]][channels[0]]
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
                if channel == channels[0]:
                    from mavt4d.data.nuscenes import _se3
                    boxes = []
                    for token in tb.annotations_of.get(sample, ()):
                        annotation = tb.sample_annotation[token]
                        transform = _se3(annotation['rotation'], annotation['translation'])
                        width, length, height = annotation['size']
                        corners = np.array([[x*length/2,y*width/2,z*height/2]
                                            for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                        world = corners @ transform[:3,:3].T + transform[:3,3]
                        cam = (world-camera_world[:3,3]) @ camera_world[:3,:3]
                        if not (cam[:,2] > .1).all():
                            continue  # avoid unbounded boxes crossing the near plane
                        projected = cam @ k.T
                        uv_box = projected[:,:2]/projected[:,2:]
                        left, top = np.maximum(uv_box.min(0), [0,0])
                        right, bottom = np.minimum(uv_box.max(0), [self.width,self.height])
                        if right>left and bottom>top:
                            boxes.append([float(left),float(top),float(right),float(bottom)])
                    mask_boxes[len(images)] = boxes
                k_t.append(torch.tensor(k, dtype=torch.float32))
                pose_t.append(torch.tensor(camera_world, dtype=torch.float32))
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
                "depth_provenance": "measured", "depth_detail": "sparse_lidar_no_dynamic_transport",
                "mask_boxes_xyxy": mask_boxes, "mask_boxes_provenance": "projected_gt_boxes_not_surface_correspondence"}
