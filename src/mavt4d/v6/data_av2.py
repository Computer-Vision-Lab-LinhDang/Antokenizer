"""Argoverse 2 Sensor clips for v6 (no av2 devkit dependency).

Layout and fields follow the public AV2 Sensor docs
(https://argoverse.github.io/user-guide/datasets/sensor.html) and av2-api
(https://github.com/argoverse/av2-api: src/av2/utils/io.py, src/av2/datasets/sensor/av2_sensor_dataloader.py,
src/av2/geometry/camera/pinhole_camera.py):

  <root>/<split>/<log_id>/
    sensors/cameras/<ring_*>/<timestamp_ns>.jpg     20 Hz, undistorted; ring_front_center is portrait
    sensors/lidar/<timestamp_ns>.feather            10 Hz; x, y, z (+ intensity, laser_number, offset_ns), ego frame
    calibration/intrinsics.feather                  sensor_name, fx_px, fy_px, cx_px, cy_px, k1.., height_px, width_px
    calibration/egovehicle_SE3_sensor.feather       sensor_name, qw, qx, qy, qz, tx_m, ty_m, tz_m (sensor -> ego)
    city_SE3_egovehicle.feather                     timestamp_ns, qw..qz, tx_m..tz_m (ego -> city)
    annotations.feather                             timestamp_ns, track_uuid, category, length_m, width_m, height_m,
                                                    qw..qz, tx_m..tz_m, num_interior_pts; cuboids in the ego frame
                                                    at lidar timestamps (absent for the test split)

Camera frame (pinhole_camera.py): +z out of the camera, +y down, +x across the imager (right).
Ego frame: +x forward, +y left, +z up. Pixel centres at integer coordinates.

Clip: `steps` lidar sweeps ~`step_dt` apart. World = ego frame at the clip's first sweep. Each camera's tube is
`tubelet` frames ~`tube_dt` apart; a camera with no frame near one of those times is invalid at that step, and a log's
first clip starts at the first sweep where every camera has its whole tube.
Every random decision has its own stream: Generator(PCG64(SeedSequence([seed, index, PURPOSE, ...]))).
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .contracts import Observation

RING_CAMERAS = ('ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left',
                'ring_rear_right', 'ring_side_left', 'ring_side_right')
CAMERA_KEY = {name: i for i, name in enumerate(RING_CAMERAS)}
# Order of the AV2 Sensor docs category list.
CATEGORIES = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BOLLARD', 'CONSTRUCTION_CONE', 'CONSTRUCTION_BARREL', 'STOP_SIGN',
              'BICYCLE', 'LARGE_VEHICLE', 'WHEELED_DEVICE', 'BUS', 'BOX_TRUCK', 'SIGN', 'TRUCK', 'MOTORCYCLE',
              'BICYCLIST', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'MOTORCYCLIST', 'DOG', 'SCHOOL_BUS', 'WHEELED_RIDER',
              'STROLLER', 'ARTICULATED_BUS', 'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_SIGN', 'WHEELCHAIR',
              'RAILED_VEHICLE', 'OFFICIAL_SIGNALER', 'TRAFFIC_LIGHT_TRAILER', 'ANIMAL')
CATEGORY_ID = {name: i for i, name in enumerate(CATEGORIES)}

# RNG purpose ids: one independent stream per decision.
PURPOSE_CAMERA_COUNT = 1
PURPOSE_CAMERA_PICK = 2
PURPOSE_DEPTH_SPLIT = 3

INPUT_DEPTH_FRACTION = 0.8
PACK_INFO = 'pack.json'  # written by pack_av2.py at the root of a packed split
MIN_DEPTH = 0.1          # m, camera z
MAX_SYNC_DT = 0.06       # s, frame-t image vs lidar sweep
MAX_TUBE_ERR = 0.25      # x tube_dt, an earlier tube frame vs its target time
_POSE_COLS = ['qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m']
_BOX_COLS = ['timestamp_ns', 'track_uuid', 'category', 'length_m', 'width_m', 'height_m', *_POSE_COLS,
             'num_interior_pts']


def rng_for(seed, index, purpose, *extra):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(index), int(purpose),
                                                                        *map(int, extra)])))


# numpy geometry (float64), quaternions (w, x, y, z)

def _qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


def _qconj(q):
    return q * np.array([1., -1., -1., -1.])


def _qnorm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)


def _qmat(q):
    w, x, y, z = np.moveaxis(_qnorm(q), -1, 0)
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
                    -1).reshape(*q.shape[:-1], 3, 3)


def _se3(q, t):
    m = np.zeros((*q.shape[:-1], 4, 4))
    m[..., :3, :3] = _qmat(q)
    m[..., :3, 3] = t
    m[..., 3, 3] = 1
    return m


def _se3_inv(m):
    r = np.swapaxes(m[..., :3, :3], -1, -2)
    out = np.zeros_like(m)
    out[..., :3, :3] = r
    out[..., :3, 3] = -(r @ m[..., :3, 3:])[..., 0]
    out[..., 3, 3] = 1
    return out


def _nearest(sorted_ts, query):
    i = np.searchsorted(sorted_ts, query).clip(1, len(sorted_ts) - 1)
    left, right = sorted_ts[i - 1], sorted_ts[i]
    return np.where(np.abs(query - left) <= np.abs(right - query), i - 1, i)


def crop_params(w0, h0, height, width):
    """Aspect-preserving scale s so (w0, h0) covers (width, height); centred crop box (x0, y0) in source px edges."""
    s = max(height / h0, width / w0)
    return s, (w0 - width / s) / 2, (h0 - height / s) / 2


def adjust_intrinsics(k, s, x0, y0):
    """Exact K after scaling by s and cropping at source edge offset (x0, y0); pixel centres at integers."""
    k = np.asarray(k, dtype=np.float64).copy()
    k[0, 0] *= s
    k[1, 1] *= s
    k[0, 1] *= s
    k[0, 2] = (k[0, 2] + 0.5 - x0) * s - 0.5
    k[1, 2] = (k[1, 2] + 0.5 - y0) * s - 0.5
    return k


def load_image(path, height, width):
    """Decode a JPEG with DCT draft, resize to cover and centre crop. Returns (uint8 [H,W,3], (s, x0, y0))."""
    from PIL import Image
    with Image.open(path) as im:
        w0, h0 = im.size
        s, x0, y0 = crop_params(w0, h0, height, width)
        drafted = im.draft('RGB', (max(1, math.ceil(w0 * s - 1e-6)), max(1, math.ceil(h0 * s - 1e-6))))
        k = w0 / drafted[1][2] if drafted else 1.0   # DCT scale actually applied
        box = (x0 / k, y0 / k, (x0 + width / s) / k, (y0 + height / s) / k)
        out = im.convert('RGB').resize((width, height), Image.BILINEAR, box=box)
        return np.asarray(out), (s, x0, y0)


def zbuffer(points_cam, k, height, width):
    """Nearest camera-z per pixel. Returns (flat pixel ids ascending, z)."""
    z = points_cam[:, 2]
    keep = z > MIN_DEPTH
    p, z = points_cam[keep], z[keep]
    u = np.floor(k[0, 0] * p[:, 0] / z + k[0, 1] * p[:, 1] / z + k[0, 2] + 0.5)
    v = np.floor(k[1, 1] * p[:, 1] / z + k[1, 2] + 0.5)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    flat = (v[inside] * width + u[inside]).astype(np.int64)
    z = z[inside]
    order = np.lexsort((z, flat))
    flat, z = flat[order], z[order]
    first = np.ones(len(flat), dtype=bool)
    first[1:] = flat[1:] != flat[:-1]
    return flat[first], z[first]


def choose_cameras(available, cameras, max_cameras, min_multi_cam, seed, index):
    """Camera subset for one clip, in RING_CAMERAS order."""
    available = [c for c in RING_CAMERAS if c in available]
    if cameras == 'all':
        return available[:max_cameras]
    if isinstance(cameras, int):
        return available[:min(cameras, max_cameras)]
    if cameras != 'random':
        raise ValueError(f'cameras must be all, an int or random, got {cameras!r}')
    n = min(len(available), max_cameras)
    if n == 0:
        return []
    count_rng = rng_for(seed, index, PURPOSE_CAMERA_COUNT)
    # P(count >= 2) = m + (1 - m)(n - 1)/n >= m
    low = 2 if n >= 2 and count_rng.random() < min_multi_cam else 1
    count = int(count_rng.integers(low, n + 1))
    pick = rng_for(seed, index, PURPOSE_CAMERA_PICK).choice(len(available), size=count, replace=False)
    return [available[i] for i in sorted(pick)]


def _read(path, columns=None):
    import pyarrow as pa              # lazy: importing this module must not need pyarrow
    import pyarrow.feather as feather
    table = feather.read_table(path, columns=columns, memory_map=True)
    out = {}
    for name in table.column_names:
        col = table.column(name)
        if pa.types.is_dictionary(col.type):
            col = col.cast(col.type.value_type)
        out[name] = col.to_numpy(zero_copy_only=False)
    return out


def _list_ts(directory, suffix):
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.array([int(n[:-len(suffix)]) for n in names if n.endswith(suffix)], dtype=np.int64))


def keep_share(log_ids, fraction):
    """The first ceil(fraction * N) of the sorted log ids under a fixed permutation, in sorted order."""
    log_ids = sorted(log_ids)
    if fraction >= 1:
        return log_ids
    order = np.random.default_rng(20261019).permutation(len(log_ids))
    keep = set(order[:math.ceil(fraction * len(log_ids))].tolist())
    return [n for i, n in enumerate(log_ids) if i in keep]


class Av2ClipDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, cfg, steps=6, step_dt=0.5, tube_dt=0.1, cameras='all', min_multi_cam=0.2,
                 seed=0, max_boxes=64, clip_stride=None, uint8_images=False, log_fraction=1.0):
        """log_fraction < 1 keeps that share of the logs, chosen by a fixed permutation of the sorted log ids, so
        a smaller share is always a subset of a larger one (for data-scaling runs)."""
        self.dir = Path(root) / split if split else Path(root)
        self.cfg, self.steps, self.step_dt, self.tube_dt = cfg, int(steps), float(step_dt), float(tube_dt)
        self.cameras, self.min_multi_cam = cameras, float(min_multi_cam)
        self.seed, self.max_boxes = int(seed), int(max_boxes)
        self.uint8_images = uint8_images
        self.frames = cfg.tubelet
        self.log_ids, self.lidar_ts, self.index = [], [], []
        self._meta, self._boxes = {}, OrderedDict()
        for log_id in keep_share(self._list_logs(), log_fraction):
            ts = self._lidar_times(self.dir / log_id)
            if len(ts) < max(self.steps, 2):
                continue
            li = len(self.log_ids)
            self.log_ids.append(log_id)
            self.lidar_ts.append(ts)
            period = float(np.median(np.diff(ts)))
            stride = clip_stride or max(1, round(self.step_dt * 1e9 / period))
            first_ok = ts[0] + (self.frames - 1) * self.tube_dt * 1e9   # tube frames exist before the clip
            start = int(np.searchsorted(ts, first_ok - 0.25 * period))
            cams = self._log_cam_ts(self.dir / log_id)
            while start < len(ts) and not all(self._camera_frames(cams, c, ts[start]) for c in cams['cam_ts']):
                start += 1                                                # a camera starts late: no history yet
            for i in range(start, len(ts), stride):
                target = ts[i] + np.arange(self.steps) * self.step_dt * 1e9
                if target[-1] > ts[-1] + 0.5 * period:
                    break
                j = _nearest(ts, target)
                if np.all(np.abs(ts[j] - target) <= 0.5 * period) and np.all(np.diff(j) > 0):
                    self.index.append((li, j))

    def __len__(self):
        return len(self.index)

    # file access: everything read from disk goes through these (PackedAv2ClipDataset reads a packed copy)

    def _list_logs(self):
        return sorted(os.listdir(self.dir))

    def _lidar_times(self, log):
        return _list_ts(log / 'sensors' / 'lidar', '.feather')

    def _camera_times(self, log, cam):
        return _list_ts(log / 'sensors' / 'cameras' / cam, '.jpg')

    def _has_camera(self, log, cam):
        return (log / 'sensors' / 'cameras' / cam).is_dir()

    def _log_cam_ts(self, log):
        """{'cam_ts': {cam: timestamps}} for the ring cameras that have frames (enough meta for _camera_frames)."""
        cam_ts = {c: self._camera_times(log, c) for c in RING_CAMERAS if self._has_camera(log, c)}
        return {'cam_ts': {c: t for c, t in cam_ts.items() if len(t)}}

    def _table(self, path, columns=None):
        """A .feather table as {column: numpy array}, or None when the file is absent."""
        return _read(path, columns) if path.exists() else None

    def _image(self, log, cam, ts):
        """uint8 [H,W,3] and the (s, x0, y0) crop that maps source pixels to it."""
        return load_image(log / 'sensors' / 'cameras' / cam / f'{ts}.jpg', self.cfg.image_height, self.cfg.image_width)

    def _lidar_in_camera(self, log, lidar_t, cam, cam_T_lidar, k, cache):
        """Sweep lidar_t in camera cam: (flat pixel index, z) of the nearest point per pixel. cache: per sweep."""
        if 'pts' not in cache:
            p = _read(log / 'sensors' / 'lidar' / f'{lidar_t}.feather', ['x', 'y', 'z'])
            cache['pts'] = np.stack([p['x'], p['y'], p['z']], -1).astype(np.float64)   # stored as float16
        pts = cache['pts']
        return zbuffer(pts @ cam_T_lidar[:3, :3].T + cam_T_lidar[:3, 3], k, self.cfg.image_height,
                       self.cfg.image_width)

    # frame choice, shared with pack_av2.py

    def _camera_frames(self, meta, cam, lidar_t):
        """Camera timestamps of the tube for sweep lidar_t, oldest first (the last one is nearest the sweep),
        or None when no frame is within MAX_SYNC_DT of the sweep, or no strictly older frame is within
        MAX_TUBE_ERR * tube_dt of an earlier tube time."""
        cts = meta['cam_ts'][cam]
        if not len(cts):
            return None
        cur = int(_nearest(cts, np.array([lidar_t]))[0]) if len(cts) > 1 else 0
        if abs(cts[cur] - lidar_t) > MAX_SYNC_DT * 1e9:
            return None
        idx = [cur]
        for f in range(1, self.frames):                               # earlier tube frames, strictly older
            want = cts[cur] - f * self.tube_dt * 1e9
            j = min(int(_nearest(cts, np.array([want]))[0]), idx[-1] - 1)
            if j < 0 or abs(cts[j] - want) > MAX_TUBE_ERR * self.tube_dt * 1e9:
                return None
            idx.append(j)
        return [int(cts[j]) for j in idx[::-1]]

    def _cam_T_lidar(self, meta, cam, cam_t, city_T_ego):
        """Camera (at its capture time cam_t) <- ego at the sweep (city_T_ego): ego-motion compensated."""
        ego_cam_T_city = _se3_inv(_se3(*self._pose(meta, np.array([cam_t], dtype=np.int64)))[0])
        return _se3_inv(meta['ego_T_cam'][cam]) @ ego_cam_T_city @ city_T_ego

    # per-log caches

    def _log_meta(self, li):
        meta = self._meta.get(li)
        if meta is not None:
            return meta
        log = self.dir / self.log_ids[li]
        intr = self._table(log / 'calibration' / 'intrinsics.feather')
        extr = self._table(log / 'calibration' / 'egovehicle_SE3_sensor.feather')
        ego_T_cam = {str(n): _se3(np.array([extr[c][i] for c in _POSE_COLS[:4]], dtype=np.float64),
                                  np.array([extr[c][i] for c in _POSE_COLS[4:]], dtype=np.float64))
                     for i, n in enumerate(extr['sensor_name'])}
        k = {}
        for i, n in enumerate(intr['sensor_name']):
            k[str(n)] = np.array([[intr['fx_px'][i], 0, intr['cx_px'][i]], [0, intr['fy_px'][i], intr['cy_px'][i]],
                                  [0, 0, 1]], dtype=np.float64)
        cams = [c for c in RING_CAMERAS if c in k and c in ego_T_cam and self._has_camera(log, c)]
        pose = self._table(log / 'city_SE3_egovehicle.feather')
        order = np.argsort(pose['timestamp_ns'])
        meta = dict(cams=cams, k=k, ego_T_cam=ego_T_cam,
                    cam_ts={c: self._camera_times(log, c) for c in cams},
                    pose_ts=pose['timestamp_ns'][order].astype(np.int64),
                    pose_q=np.stack([pose[c] for c in _POSE_COLS[:4]], -1)[order].astype(np.float64),
                    pose_t=np.stack([pose[c] for c in _POSE_COLS[4:]], -1)[order].astype(np.float64))
        self._meta[li] = meta
        return meta

    def _log_boxes(self, li):
        if li in self._boxes:
            self._boxes.move_to_end(li)
            return self._boxes[li]
        a = self._table(self.dir / self.log_ids[li] / 'annotations.feather', _BOX_COLS)
        boxes = None
        if a is not None:
            ts = a['timestamp_ns'].astype(np.int64)
            order = np.argsort(ts, kind='stable')
            uniq, start = np.unique(ts[order], return_index=True)
            rows = dict(zip(uniq.tolist(), np.split(order, start[1:])))
            boxes = dict(rows=rows, uuid=a['track_uuid'].astype(str),
                         category=np.array([CATEGORY_ID.get(str(c), -1) for c in a['category']], dtype=np.int64),
                         size=np.stack([a['length_m'], a['width_m'], a['height_m']], -1).astype(np.float64),
                         q=np.stack([a[c] for c in _POSE_COLS[:4]], -1).astype(np.float64),
                         t=np.stack([a[c] for c in _POSE_COLS[4:]], -1).astype(np.float64),
                         pts=a['num_interior_pts'].astype(np.int64))
        self._boxes[li] = boxes
        while len(self._boxes) > 4:
            self._boxes.popitem(last=False)
        return boxes

    def _pose(self, meta, ts):
        """city <- ego at ts (int64 ns array) as (q [N,4], t [N,3]): lerp / nlerp, clamped at the ends."""
        pts = meta['pose_ts']
        ts = np.asarray(ts, dtype=np.int64)
        if len(pts) == 1:
            return np.repeat(meta['pose_q'], len(ts), 0), np.repeat(meta['pose_t'], len(ts), 0)
        i = np.searchsorted(pts, ts).clip(1, len(pts) - 1)
        t0, t1 = pts[i - 1], pts[i]
        a = ((ts - t0) / np.maximum(t1 - t0, 1)).clip(0, 1)[:, None]
        q0, q1 = meta['pose_q'][i - 1], meta['pose_q'][i]
        q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
        return _qnorm((1 - a) * q0 + a * q1), (1 - a) * meta['pose_t'][i - 1] + a * meta['pose_t'][i]

    # sample

    def __getitem__(self, index):
        H, W, V = self.cfg.image_height, self.cfg.image_width, self.cfg.max_cameras
        F, T = self.frames, self.steps
        li, lidar_idx = self.index[index]
        meta = self._log_meta(li)
        log = self.dir / self.log_ids[li]
        lidar_ts = self.lidar_ts[li][lidar_idx]
        cams = choose_cameras(meta['cams'], self.cameras, V, self.min_multi_cam, self.seed, index)

        city_q, city_t = self._pose(meta, lidar_ts)                   # ego(t) -> city, [T,4] [T,3]
        city_T_ego = _se3(city_q, city_t)
        world_T_ego = _se3_inv(city_T_ego[0]) @ city_T_ego
        world_q = _qmul(_qconj(city_q[:1]), city_q)

        images = np.zeros((T, V, F, H, W, 3), dtype=np.uint8)
        f = max(H, W)   # placeholder pinhole for padded cameras
        intrinsics = np.tile(np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]]), (T, V, 1, 1))
        cam_to_rig = np.tile(np.eye(4), (T, V, 1, 1))
        camera_key = np.full((T, V), -1, dtype=np.int64)
        camera_valid = np.zeros((T, V), dtype=bool)
        depth = np.zeros((T, V, H * W), dtype=np.float32)
        depth_valid = np.zeros((T, V, H * W), dtype=bool)
        target = np.zeros((T, V, H * W), dtype=np.float32)
        target_valid = np.zeros((T, V, H * W), dtype=bool)
        tube_dt = np.zeros(T)

        for s in range(T):
            sweep, dts = {}, []
            for v, cam in enumerate(cams):
                key = CAMERA_KEY[cam]
                camera_key[s, v] = key
                cam_to_rig[s, v] = meta['ego_T_cam'][cam]
                frames = self._camera_frames(meta, cam, lidar_ts[s])
                if frames is None:
                    continue
                for f, ts in enumerate(frames):
                    images[s, v, f], crop = self._image(log, cam, ts)
                k = adjust_intrinsics(meta['k'][cam], *crop)
                intrinsics[s, v] = k
                camera_valid[s, v] = True
                dts.append((frames[-1] - frames[0]) * 1e-9 / max(F - 1, 1))
                # lidar -> camera at its capture time (ego-motion compensated)
                cam_T_lidar = self._cam_T_lidar(meta, cam, frames[-1], city_T_ego[s])
                flat, z = self._lidar_in_camera(log, int(lidar_ts[s]), cam, cam_T_lidar, k, sweep)
                inp = rng_for(self.seed, index, PURPOSE_DEPTH_SPLIT, s, key).random(len(flat)) < INPUT_DEPTH_FRACTION
                depth[s, v, flat[inp]] = z[inp]
                depth_valid[s, v, flat[inp]] = True
                target[s, v, flat[~inp]] = z[~inp]
                target_valid[s, v, flat[~inp]] = True
            tube_dt[s] = float(np.mean(dts)) if dts else 0.

        img = torch.from_numpy(images).permute(0, 1, 2, 5, 3, 4).contiguous()
        if not self.uint8_images:
            img = img.float().div_(255)
        hw = (T, V, H, W)
        return dict(
            images=img, tube_dt=torch.from_numpy(tube_dt).float(),
            intrinsics=torch.from_numpy(intrinsics).float(), cam_to_rig=torch.from_numpy(cam_to_rig).float(),
            camera_key=torch.from_numpy(camera_key), camera_valid=torch.from_numpy(camera_valid),
            depth=torch.from_numpy(depth).view(hw), depth_valid=torch.from_numpy(depth_valid).view(hw),
            target_depth=torch.from_numpy(target).view(hw), target_valid=torch.from_numpy(target_valid).view(hw),
            rig_pose=torch.from_numpy(world_T_ego).float(), rig_pose_valid=torch.ones(T, dtype=torch.bool),
            has_metric_depth=torch.ones(T, dtype=torch.bool),
            t=torch.from_numpy((lidar_ts - lidar_ts[0]) * 1e-9).double(),
            timestamp_ns=torch.from_numpy(lidar_ts.copy()), log_id=self.log_ids[li],
            boxes=self._clip_boxes(li, lidar_ts, world_T_ego, world_q))

    def _clip_boxes(self, li, lidar_ts, world_T_ego, world_q):
        T, M = len(lidar_ts), self.max_boxes
        out = dict(center=np.zeros((T, M, 3)), size=np.zeros((T, M, 3)), quaternion=np.tile([1., 0, 0, 0], (T, M, 1)),
                   category=np.full((T, M), -1, dtype=np.int64), track=np.full((T, M), -1, dtype=np.int64),
                   num_interior_pts=np.zeros((T, M), dtype=np.int64), mask=np.zeros((T, M), dtype=bool))
        boxes = self._log_boxes(li)
        if boxes is not None:
            rows = [boxes['rows'].get(int(ts), np.zeros(0, dtype=np.int64)) for ts in lidar_ts]
            # nearest max_boxes to the ego per step
            rows = [r[np.argsort(np.linalg.norm(boxes['t'][r, :2], axis=-1), kind='stable')[:M]] for r in rows]
            uuids = sorted({u for r in rows for u in boxes['uuid'][r]})
            track = {u: i for i, u in enumerate(uuids)}
            for s, r in enumerate(rows):
                n = len(r)
                rw = world_T_ego[s]
                out['center'][s, :n] = boxes['t'][r] @ rw[:3, :3].T + rw[:3, 3]
                q = _qnorm(_qmul(np.broadcast_to(world_q[s], (n, 4)), boxes['q'][r]))
                out['quaternion'][s, :n] = np.where(q[:, :1] < 0, -q, q)
                out['size'][s, :n] = boxes['size'][r]
                out['category'][s, :n] = boxes['category'][r]
                out['track'][s, :n] = [track[u] for u in boxes['uuid'][r]]
                out['num_interior_pts'][s, :n] = boxes['pts'][r]
                out['mask'][s, :n] = True
        return {k: torch.from_numpy(v).float() if v.dtype == np.float64 else torch.from_numpy(v)
                for k, v in out.items()}


class PackedAv2ClipDataset(Av2ClipDataset):
    """A split written by pack_av2.py. It gives the same clips and tensors as Av2ClipDataset on the source split,
    except that images were re-encoded at the target size. LiDAR is already projected into each camera, tables are
    .npz, and nothing needs pyarrow. Only the sweeps and frames the source clips use were written."""

    def __init__(self, root, split, cfg, steps=6, step_dt=0.5, tube_dt=0.1, **kw):
        d = Path(root) / split if split else Path(root)
        info = json.loads((d / PACK_INFO).read_text())
        want = dict(image_height=cfg.image_height, image_width=cfg.image_width, tubelet=cfg.tubelet,
                    steps=int(steps), step_dt=float(step_dt), tube_dt=float(tube_dt))
        bad = {key: (info.get(key), val) for key, val in want.items() if info.get(key) != val}
        if bad:
            raise ValueError(f'{d} was packed for other settings (packed, wanted): {bad}')
        self.image_ext = info['image_ext']
        super().__init__(root, split, cfg, steps=steps, step_dt=step_dt, tube_dt=tube_dt, **kw)

    def _list_logs(self):
        return sorted(n for n in os.listdir(self.dir) if (self.dir / n / 'lidar_ts.npy').exists())

    def _lidar_times(self, log):
        return np.load(log / 'lidar_ts.npy')

    def _camera_times(self, log, cam):
        return np.load(log / 'sensors' / 'cameras' / cam / 'ts.npy')

    def _has_camera(self, log, cam):
        return (log / 'sensors' / 'cameras' / cam / 'ts.npy').exists()

    def _table(self, path, columns=None):
        path = path.with_suffix('.npz')
        if not path.exists():
            return None
        with np.load(path) as z:
            return {c: z[c] for c in (columns or z.files)}

    def _image(self, log, cam, ts):
        from PIL import Image
        d = log / 'sensors' / 'cameras' / cam
        with Image.open(d / f'{ts}.{self.image_ext}') as im:
            img = np.asarray(im.convert('RGB'))
        return img, tuple(np.load(d / 'crop.npy').tolist())

    def _lidar_in_camera(self, log, lidar_t, cam, cam_T_lidar, k, cache):
        if 'packed' not in cache:
            with np.load(log / 'sensors' / 'lidar' / f'{lidar_t}.npz') as z:
                cache['packed'] = dict(z)
        return cache['packed'][f'{cam}_flat'].astype(np.int64), cache['packed'][f'{cam}_z']


def open_av2(root, split, cfg, **kw):
    """PackedAv2ClipDataset when root/split was written by pack_av2.py, else Av2ClipDataset."""
    d = Path(root) / split if split else Path(root)
    return (PackedAv2ClipDataset if (d / PACK_INFO).exists() else Av2ClipDataset)(root, split, cfg, **kw)


def collate_clips(samples):
    """Stack per-clip dicts on a new batch dim 0 (nested dicts recurse, non-tensors become lists)."""
    out = {}
    for key, value in samples[0].items():
        if isinstance(value, torch.Tensor):
            out[key] = torch.stack([s[key] for s in samples])
        elif isinstance(value, dict):
            out[key] = collate_clips([s[key] for s in samples])
        else:
            out[key] = [s[key] for s in samples]
    return out


def to_observation(batch, step):
    """One step of a collated batch as a contracts.Observation."""
    images = batch['images'][:, step]
    if images.dtype == torch.uint8:
        images = images.float().div_(255)
    return Observation(images=images, tube_dt=batch['tube_dt'][:, step], intrinsics=batch['intrinsics'][:, step],
                       cam_to_rig=batch['cam_to_rig'][:, step], camera_key=batch['camera_key'][:, step],
                       camera_valid=batch['camera_valid'][:, step], depth=batch['depth'][:, step],
                       depth_valid=batch['depth_valid'][:, step], rig_pose=batch['rig_pose'][:, step],
                       rig_pose_valid=batch['rig_pose_valid'][:, step],
                       has_metric_depth=batch['has_metric_depth'][:, step], t=batch['t'][:, step])
