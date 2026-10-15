"""nuScenes as SceneSample: m3 (one keyframe, 6 cameras, 1 LiDAR sweep) and m2 (8-frame CAM_FRONT clips).

Conventions:
- world = ego frame of the reference LIDAR_TOP sweep (m3: the keyframe's; m2: the sweep nearest frame 0),
  and t0 = that sweep's timestamp, so the space and time origins are the same sensor reading;
- each image is resized isotropically by s = H / 900 and centre-cropped to W in one PIL call, so
  K' = [[fx s, 0, cx s - left], [0, fy s, cy s], [0, 0, 1]] holds exactly (left = 3 for 1600x900 -> 448);
- m2 boxes: a keyframe camera frame takes its sample's annotations; frames in between interpolate the
  neighbouring keyframes of the same chain per instance on camera time (instances in only one are dropped).
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from mavt4d.data.sample import Boxes3D, SceneSample

CAM6 = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")  # ring
LIDAR = "LIDAR_TOP"
_TABLES = ("scene", "sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor", "sample_annotation",
           "instance")
_OFFICIAL = {"v1.0-mini": ("mini_train", "mini_val"), "v1.0-trainval": ("train", "val")}


def load_official_split(version: str, split: str) -> list[str]:
    """Scene names of an official devkit split; mini splits only for v1.0-mini, train/val for trainval."""
    if split not in _OFFICIAL.get(version, ()):
        raise ValueError(f"split: {split!r} is not an official split of {version!r}")
    from nuscenes.utils.splits import create_splits_scenes  # lazy: the devkit pulls in matplotlib/cv2
    return list(create_splits_scenes()[split])


def dev_split_scenes(train_scene_names: list[str]) -> set[str]:
    """About 10% of train scenes, chosen by sha1(name) mod 10 == 0 (independent of order and RNG)."""
    return {n for n in train_scene_names if int(hashlib.sha1(n.encode("utf-8")).hexdigest(), 16) % 10 == 0}


def _split_scene_names(version: str, split: str) -> set[str]:
    if split in ("train", "dev"):
        train = load_official_split(version, "train")
        dev = dev_split_scenes(train)
        return dev if split == "dev" else set(train) - dev
    return set(load_official_split(version, split))


def _se3(rotation: list[float], translation: list[float]) -> np.ndarray:
    """4x4 float64 from a nuScenes (w, x, y, z) quaternion and a translation."""
    w, x, y, z = np.asarray(rotation, dtype=np.float64) / np.linalg.norm(rotation)
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = translation
    return T


class NuScenesTables:
    """The eight JSON tables keyed by token, plus the indexes the reader needs."""

    def __init__(self, dataroot: str | Path, version: str):
        self.dataroot = Path(dataroot)
        for name in _TABLES:
            with open(self.dataroot / version / f"{name}.json") as f:
                setattr(self, name, {r["token"]: r for r in json.load(f)})
        self.scene_by_name = {r["name"]: tok for tok, r in self.scene.items()}
        self.instance_index = {tok: i for i, tok in enumerate(self.instance)}
        channel_of_cs = {tok: self.sensor[r["sensor_token"]]["channel"] for tok, r in self.calibrated_sensor.items()}
        self._channel = {tok: channel_of_cs[r["calibrated_sensor_token"]] for tok, r in self.sample_data.items()}
        self.keyframe_data: dict[str, dict[str, str]] = defaultdict(dict)
        for tok, r in self.sample_data.items():
            if r["is_key_frame"]:
                self.keyframe_data[r["sample_token"]][self._channel[tok]] = tok
        self.annotations_of: dict[str, list[str]] = defaultdict(list)
        for tok, r in self.sample_annotation.items():
            self.annotations_of[r["sample_token"]].append(tok)

    def channel(self, sd_token: str) -> str:
        return self._channel[sd_token]

    def scene_of(self, sd_token: str) -> str:
        return self.sample[self.sample_data[sd_token]["sample_token"]]["scene_token"]

    def scene_samples(self, scene_token: str) -> list[str]:
        """Keyframe sample tokens of a scene in time order."""
        out, tok = [], self.scene[scene_token]["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self.sample[tok]["next"]
        return out

    def chain(self, scene_token: str, channel: str) -> list[str]:
        """sample_data tokens of one channel inside one scene in time order (stops at the scene boundary
        even where prev/next link into the neighbouring scene)."""
        first = self.keyframe_data[self.scene[scene_token]["first_sample_token"]][channel]
        head = first
        while (p := self.sample_data[head]["prev"]) and self.scene_of(p) == scene_token:
            head = p
        out = [head]
        while (n := self.sample_data[out[-1]]["next"]) and self.scene_of(n) == scene_token:
            out.append(n)
        return out

    def ego_to_global(self, sd_token: str) -> np.ndarray:
        r = self.ego_pose[self.sample_data[sd_token]["ego_pose_token"]]
        return _se3(r["rotation"], r["translation"])

    def sensor_to_ego(self, sd_token: str) -> np.ndarray:
        r = self.calibrated_sensor[self.sample_data[sd_token]["calibrated_sensor_token"]]
        return _se3(r["rotation"], r["translation"])

    def raw_intrinsics(self, sd_token: str) -> np.ndarray:
        return np.array(self.calibrated_sensor[self.sample_data[sd_token]["calibrated_sensor_token"]]
                        ["camera_intrinsic"], dtype=np.float64)


def resize_crop(width: int, height: int, H: int, W: int) -> tuple[float, int]:
    """Scale and left crop for an isotropic resize to height H then a centre crop to width W."""
    s = H / height
    left = (round(width * s) - W) // 2
    if left < 0:
        raise ValueError(f"resolution: {width}x{height} resized to height {H} is narrower than {W}")
    return s, left


class NuScenesSceneDataset(Dataset):
    """m3: one item per keyframe sample. m2: one item per window of ``clip_len`` consecutive frames of
    ``m2_camera`` (every start, overlapping) inside a scene. ``heldout`` is left unset (T15 applies it)."""

    def __init__(self, dataroot: str | Path, version: str = "v1.0-trainval",
                 split: Literal["train", "dev", "val", "mini_train", "mini_val"] = "train",
                 mode: Literal["m3", "m2"] = "m3", resolution: tuple[int, int] = (256, 448),
                 cameras: tuple[str, ...] = CAM6, m2_camera: str = "CAM_FRONT", clip_len: int = 8):
        if mode not in ("m3", "m2"):
            raise ValueError(f"mode: expected 'm3' or 'm2', got {mode!r}")
        self.tables = tb = NuScenesTables(dataroot, version)
        self.mode, self.resolution, self.cameras = mode, tuple(resolution), tuple(cameras)
        self.m2_camera, self.clip_len = m2_camera, clip_len
        names = _split_scene_names(version, split)
        scenes = sorted((tb.scene[tok]["name"], tok) for tok in tb.scene if tb.scene[tok]["name"] in names)
        self._chains: dict[tuple[str, str], list[str]] = {}
        self.items: list[tuple[str, str | int]] = []
        for _, sc in scenes:
            if mode == "m3":
                self.items += [(sc, s) for s in tb.scene_samples(sc)]
            else:
                n = len(self._chain(sc, m2_camera))
                self.items += [(sc, i) for i in range(n - clip_len + 1)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> SceneSample:
        scene, key = self.items[i]
        s = self._m3(scene, key) if self.mode == "m3" else self._m2(scene, key)
        s.validate()
        return s

    def _chain(self, scene: str, channel: str) -> list[str]:
        if (scene, channel) not in self._chains:
            self._chains[scene, channel] = self.tables.chain(scene, channel)
        return self._chains[scene, channel]

    def _m3(self, scene: str, sample: str) -> SceneSample:
        kf = self.tables.keyframe_data[sample]
        cams = [kf[c] for c in self.cameras]
        lidar = kf[LIDAR]
        ref, t0 = self._reference(lidar)
        images, K, c2w = self._cameras([[sd] for sd in cams], ref)
        return SceneSample(
            scene_id=self.tables.scene[scene]["name"], source="nuscenes", images=images,
            frame_times=self._times([[sd] for sd in cams], t0), intrinsics=K, cam_to_world=c2w,
            **self._lidar([lidar], ref, t0), boxes=[_boxes(self._annotations(sample, ref))])

    def _m2(self, scene: str, start: int) -> SceneSample:
        chain = self._chain(scene, self.m2_camera)
        frames = chain[start:start + self.clip_len]
        lidar_chain = self._chain(scene, LIDAR)
        lidar_t = [self.tables.sample_data[t]["timestamp"] for t in lidar_chain]
        sweeps = [lidar_chain[_nearest(lidar_t, self.tables.sample_data[f]["timestamp"])] for f in frames]
        ref, t0 = self._reference(sweeps[0])
        images, K, c2w = self._cameras([frames], ref)
        return SceneSample(
            scene_id=self.tables.scene[scene]["name"], source="nuscenes", images=images,
            frame_times=self._times([frames], t0), intrinsics=K, cam_to_world=c2w,
            **self._lidar(sweeps, ref, t0),
            boxes=[_boxes(self._chain_boxes(chain, start + j, ref)) for j in range(len(frames))])

    def _reference(self, lidar_sd: str) -> tuple[np.ndarray, int]:
        """global -> world (inverse ego pose of the reference sweep) and t0 in microseconds."""
        return np.linalg.inv(self.tables.ego_to_global(lidar_sd)), self.tables.sample_data[lidar_sd]["timestamp"]

    def _times(self, sds: list[list[str]], t0: int) -> Tensor:
        return torch.tensor([[(self.tables.sample_data[t]["timestamp"] - t0) / 1e6 for t in row] for row in sds],
                            dtype=torch.float64)

    def _cameras(self, sds: list[list[str]], ref: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        """images (V, T, 3, H, W), intrinsics (V, 3, 3) from each camera's first frame, cam_to_world (V, T, 4, 4)."""
        tb, (H, W) = self.tables, self.resolution
        images, Ks, c2w = [], [], []
        for row in sds:
            images.append(torch.stack([self._image(t) for t in row]))
            rec = tb.sample_data[row[0]]
            s, left = resize_crop(rec["width"], rec["height"], H, W)
            K = tb.raw_intrinsics(row[0])
            Ks.append([[K[0, 0] * s, 0.0, K[0, 2] * s - left], [0.0, K[1, 1] * s, K[1, 2] * s], [0.0, 0.0, 1.0]])
            c2w.append(np.stack([ref @ tb.ego_to_global(t) @ tb.sensor_to_ego(t) for t in row]))
        return (torch.stack(images), torch.tensor(Ks, dtype=torch.float32),
                torch.from_numpy(np.stack(c2w)).float())

    def _image(self, sd_token: str) -> Tensor:
        """(3, H, W) in [-1, 1]: isotropic resize by s and crop of W columns starting at ``left`` in one call."""
        H, W = self.resolution
        with Image.open(self.tables.dataroot / self.tables.sample_data[sd_token]["filename"]) as im:
            im = im.convert("RGB")
            s, left = resize_crop(im.width, im.height, H, W)
            im = im.resize((W, H), Image.BILINEAR, box=(left / s, 0, (left + W) / s, im.height))
            x = torch.from_numpy(np.asarray(im, dtype=np.float32))
        return (x / 127.5 - 1.0).permute(2, 0, 1)

    def _lidar(self, sweeps: list[str], ref: np.ndarray, t0: int) -> dict:
        tb = self.tables
        pts = [torch.from_numpy(np.fromfile(tb.dataroot / tb.sample_data[t]["filename"], dtype=np.float32)
                                .reshape(-1, 5)[:, :4].copy()) for t in sweeps]
        l2w = np.stack([ref @ tb.ego_to_global(t) @ tb.sensor_to_ego(t) for t in sweeps])
        return {"lidar_points": pts, "lidar_to_world": torch.from_numpy(l2w).float(),
                "lidar_times": self._times([sweeps], t0)[0], "lidar_keys": list(sweeps)}

    def _annotations(self, sample: str, ref: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
        """instance id -> (centre, size, yaw) of a keyframe's annotations in the world frame."""
        tb, out = self.tables, {}
        for tok in tb.annotations_of.get(sample, ()):
            a = tb.sample_annotation[tok]
            T = ref @ _se3(a["rotation"], a["translation"])
            out[tb.instance_index[a["instance_token"]]] = (
                T[:3, 3], np.asarray(a["size"], dtype=np.float64), math.atan2(T[1, 0], T[0, 0]))
        return out

    def _chain_boxes(self, chain: list[str], g: int, ref: np.ndarray) -> dict:
        """Boxes at chain position ``g``: exact on keyframes, interpolated between the neighbouring keyframes
        of the chain on camera time (clamped to the nearest keyframe outside the first/last one)."""
        sd = self.tables.sample_data
        keys = [j for j, t in enumerate(chain) if sd[t]["is_key_frame"]]
        k = bisect.bisect_right(keys, g) - 1
        if k < 0 or keys[k] == g or k == len(keys) - 1:
            return self._annotations(sd[chain[keys[max(k, 0)]]]["sample_token"], ref)
        g0, g1 = keys[k], keys[k + 1]
        t, t_0, t_1 = (sd[chain[j]]["timestamp"] for j in (g, g0, g1))
        w = (t - t_0) / (t_1 - t_0)
        a0 = self._annotations(sd[chain[g0]]["sample_token"], ref)
        a1 = self._annotations(sd[chain[g1]]["sample_token"], ref)
        out = {}
        for inst in a0.keys() & a1.keys():
            (c0, s0, y0), (c1, s1, y1) = a0[inst], a1[inst]
            dy = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
            out[inst] = ((1 - w) * c0 + w * c1, (1 - w) * s0 + w * s1, y0 + w * dy)
        return out


def _nearest(times: list[int], t: int) -> int:
    """Index of the timestamp in sorted ``times`` closest to ``t`` (earlier one on ties)."""
    i = bisect.bisect_left(times, t)
    if i == 0:
        return 0
    if i == len(times):
        return len(times) - 1
    return i if times[i] - t < t - times[i - 1] else i - 1


def _boxes(by_instance: dict[int, tuple[np.ndarray, np.ndarray, float]]) -> Boxes3D:
    ids = sorted(by_instance)
    rows = [by_instance[i] for i in ids]
    return Boxes3D(center=torch.tensor(np.array([r[0] for r in rows]).reshape(-1, 3), dtype=torch.float32),
                   size=torch.tensor(np.array([r[1] for r in rows]).reshape(-1, 3), dtype=torch.float32),
                   yaw=torch.tensor([r[2] for r in rows], dtype=torch.float32).reshape(-1),
                   instance_id=torch.tensor(ids, dtype=torch.int64).reshape(-1))
