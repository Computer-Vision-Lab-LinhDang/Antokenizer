"""OmniWorld-Game clips for v6, read from a split written by pack_omniworld.py.

Source conventions (dataset README at revision 4e34897; scripts/data/check_omniworld.py geom found depth/pose
reprojection AbsRel ~0.3 % with them): camera/split_i.json holds per-frame focals, cx, cy at 1280 x 720 (pixel
centres at +0.5) and world -> camera OpenCV poses (quats w, x, y, z, not normalised; trans); depth PNGs are uint16
mapped through near 1 / far 1000; the scene's metric scale multiplies depth and camera -> world translation.

Pack (pack_omniworld.py), per split directory:
  pack.json                         settings (SOURCE, image size, tubelet, step_dt, tube_dt, image_ext)
  <uid>/scene.json                  uid, fps, metric_scale
  <uid>/split_<i>.npz               frames [n] original indices, origin (first frame of the split), K [n,3,3] at the
                                    target size (pixel centres at integers), c2w [n,4,4] camera -> world, metres;
                                    pair_from / pair_absrel: depth_consistency of each anchor with the next one
  <uid>/rgb/<frame>.<image_ext>     target size, centre crop as data_av2.load_image
  <uid>/depth/<frame>.npy           float16 camera z in metres, 0 where invalid (sky, too close, > MAX_DEPTH)

v6 view (this module), shaped like Av2ClipDataset so batches collate with AV2: one camera, key 0, in slot 0 of
max_cameras. The rig is the camera with its axes renamed to the ego convention (+x = camera +z forward, +y = camera
-x, +z = camera -y up), so cam_to_rig is a fixed rotation. World = rig at the clip's first step.
ponytail: the rig follows the camera's pitch and roll, so world z (up for the BEV scene map) is the first frame's
camera up, not gravity; a tilted first frame tilts the map. Upgrade: a gravity estimate per clip.
Dense depth: its valid pixels split 80/20 into input and target, like the LiDAR points. No 3D boxes: an empty table
of max_boxes rows (mask False), so the box teacher and the slot losses stay off.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from . import data_av2
from .data_av2 import INPUT_DEPTH_FRACTION, PACK_INFO, PURPOSE_DEPTH_SPLIT, _se3_inv, keep_share, rng_for

SOURCE = 'omniworld-game'
NEAR, FAR = 1.0, 1000.0      # README section 2.3 depth encoding
MAX_DEPTH = 250.0            # m; beyond it depth is invalid (top of the encoder's depth range, fits float16)
# Some OmniWorld splits have broken cameras (reproj_error_refine_filtered 124-273 vs 13-84, focal jumping by
# thousands of px within a split). On 7 pilot scenes (110 step pairs, 0.5 s apart) consistent pairs stay at 0.1-3.9 %
# median depth AbsRel (median 0.8 %) and broken ones reach 9-30 %, so a clip needs every step pair at most this.
MAX_PAIR_ABSREL = 0.05
CAMERA_KEY = 0
EGO_FROM_CAM = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def decode_depth(raw, metric_scale):
    """uint16 depth PNG -> (camera z in metres float32 [H,W], valid [H,W])."""
    d = raw.astype(np.float64) / 65535.0
    z = d / (FAR - d * (FAR - NEAR)) / 0.004 * float(metric_scale)
    valid = (d >= 0.0015) & (d <= 65500.0 / 65535.0) & (z <= MAX_DEPTH)
    return np.where(valid, z, 0.).astype(np.float32), valid


def frame_steps(fps, step_dt, tube_dt):
    """Frames per clip step and per tube gap at fps."""
    return max(1, round(step_dt * fps)), max(1, round(tube_dt * fps))


def anchors(origin, last, fps, step_dt, tube_dt, tubelet):
    """[(anchor frame, tube frames oldest first)] on the grid origin + tube (tubelet - 1) + k step, up to last."""
    step, tube = frame_steps(fps, step_dt, tube_dt)
    first = origin + tube * (tubelet - 1)
    return [(a, [a - tube * (tubelet - 1 - j) for j in range(tubelet)]) for a in range(first, last + 1, step)]


def depth_consistency(z_a, k_a, c2w_a, z_b, k_b, c2w_b):
    """Lift frame a's depth (camera z [H,W], 0 = invalid) with K and camera -> world, project into frame b, compare
    with b's depth at the landing pixel (centres at integers). -> (median |z - z_b| / z_b, count); nan when none."""
    z_a, z_b = np.asarray(z_a, np.float64), np.asarray(z_b, np.float64)
    h, w = z_a.shape
    v, u = np.nonzero(z_a > 0)
    p = np.stack((u, v, np.ones_like(u)), -1).astype(np.float64) @ np.linalg.inv(k_a).T * z_a[v, u][:, None]
    rel_t = np.linalg.inv(np.asarray(c2w_b, np.float64)) @ np.asarray(c2w_a, np.float64)
    q = p @ rel_t[:3, :3].T + rel_t[:3, 3]
    zq = q[:, 2]
    front = zq > 1e-6
    uv = q[front] @ np.asarray(k_b, np.float64).T
    iu, iv = np.round(uv[:, 0] / zq[front]).astype(np.int64), np.round(uv[:, 1] / zq[front]).astype(np.int64)
    inside = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    zq, iu, iv = zq[front][inside], iu[inside], iv[inside]
    zb = z_b[iv, iu]
    hit = zb > 0
    rel = np.abs(zq[hit] - zb[hit]) / zb[hit]
    return (float(np.median(rel)) if rel.size else math.nan), int(rel.size)


def reprojection_absrel(sample, s, s2):
    """depth_consistency between steps s and s2 of a sample, from its own intrinsics, cam_to_rig, rig pose and
    depth (input and target pixels): the check that the whole chain of conventions agrees."""
    def view(t):
        z = torch.where(sample['depth_valid'][t, 0], sample['depth'][t, 0], sample['target_depth'][t, 0])
        z = torch.where(sample['depth_valid'][t, 0] | sample['target_valid'][t, 0], z, 0.)
        c2w = sample['rig_pose'][t].double() @ sample['cam_to_rig'][t, 0].double()
        return z.numpy(), sample['intrinsics'][t, 0].double().numpy(), c2w.numpy()
    return depth_consistency(*view(s), *view(s2))


class OmniWorldClipDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, cfg, steps=6, step_dt=0.5, tube_dt=0.1, cameras='all', seed=0, max_boxes=64,
                 uint8_images=False, log_fraction=1.0, max_absrel=MAX_PAIR_ABSREL):
        """cameras is accepted for the Av2ClipDataset signature; OmniWorld-Game has one camera.
        log_fraction keeps that share of the scenes (nested, as data_av2.keep_share). A clip is kept only when every
        consecutive step pair's depth_consistency (stored by the packer) is at most max_absrel."""
        self.dir = Path(root).expanduser() / split if split else Path(root).expanduser()
        info = json.loads((self.dir / PACK_INFO).read_text())
        want = dict(image_height=cfg.image_height, image_width=cfg.image_width, tubelet=cfg.tubelet,
                    step_dt=float(step_dt), tube_dt=float(tube_dt))
        bad = {key: (info.get(key), val) for key, val in want.items() if info.get(key) != val}
        if info.get('source') != SOURCE or bad:
            raise ValueError(f'{self.dir} was packed for other settings (packed, wanted): {bad or info.get("source")}')
        self.cfg, self.steps, self.step_dt, self.tube_dt = cfg, int(steps), float(step_dt), float(tube_dt)
        self.seed, self.max_boxes, self.uint8_images = int(seed), int(max_boxes), uint8_images
        self.image_ext = info['image_ext']
        self.scenes, self.index = [], []
        uids = sorted(p.parent.name for p in self.dir.glob('*/scene.json'))
        for uid in keep_share(uids, log_fraction):
            scene = json.loads((self.dir / uid / 'scene.json').read_text())
            si = len(self.scenes)
            self.scenes.append(dict(scene, splits={}))
            for path in sorted((self.dir / uid).glob('split_*.npz')):
                with np.load(path) as z:
                    cam = {key: z[key] for key in z.files}
                name = path.stem
                self.scenes[si]['splits'][name] = cam
                row = {int(f): i for i, f in enumerate(cam['frames'])}
                pair = dict(zip(cam['pair_from'].tolist(), cam['pair_absrel'].tolist()))
                grid = anchors(int(cam['origin']), int(cam['frames'].max()), scene['fps'], step_dt, tube_dt,
                               cfg.tubelet)
                ok = [all(f in row for f in tube) for _, tube in grid]
                steady = [pair.get(a, math.inf) <= max_absrel for a, _ in grid]   # step a -> next step
                for k in range(len(grid) - self.steps + 1):
                    if all(ok[k:k + self.steps]) and all(steady[k:k + self.steps - 1]):
                        self.index.append((si, name, [[row[f] for f in tube] for _, tube in grid[k:k + self.steps]]))

    def __len__(self):
        return len(self.index)

    def _image(self, uid, frame):
        from PIL import Image
        with Image.open(self.dir / uid / 'rgb' / f'{frame}.{self.image_ext}') as im:
            return np.asarray(im.convert('RGB'))

    def __getitem__(self, index):
        H, W, V, F, T = self.cfg.image_height, self.cfg.image_width, self.cfg.max_cameras, self.cfg.tubelet, self.steps
        si, name, rows = self.index[index]
        scene = self.scenes[si]
        uid, fps, cam = scene['uid'], float(scene['fps']), scene['splits'][name]
        frames = cam['frames']
        anchor = np.array([frames[r[-1]] for r in rows], dtype=np.int64)
        rig_T_cam = np.eye(4)
        rig_T_cam[:3, :3] = EGO_FROM_CAM
        world_T_rig = cam['c2w'][[r[-1] for r in rows]].astype(np.float64) @ _se3_inv(rig_T_cam)
        world_T_rig = _se3_inv(world_T_rig[0]) @ world_T_rig

        f = max(H, W)   # placeholder pinhole for padded cameras, as data_av2
        intrinsics = np.tile(np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]]), (T, V, 1, 1))
        cam_to_rig = np.tile(np.eye(4), (T, V, 1, 1))
        camera_key = np.full((T, V), -1, dtype=np.int64)
        camera_valid = np.zeros((T, V), dtype=bool)
        images = np.zeros((T, V, F, H, W, 3), dtype=np.uint8)
        depth = np.zeros((T, V, H * W), dtype=np.float32)
        depth_valid = np.zeros((T, V, H * W), dtype=bool)
        target = np.zeros((T, V, H * W), dtype=np.float32)
        target_valid = np.zeros((T, V, H * W), dtype=bool)
        for s, r in enumerate(rows):
            camera_key[s, 0], camera_valid[s, 0], cam_to_rig[s, 0] = CAMERA_KEY, True, rig_T_cam
            intrinsics[s, 0] = cam['K'][r[-1]]
            for j, row in enumerate(r):
                images[s, 0, j] = self._image(uid, int(frames[row]))
            z = np.load(self.dir / uid / 'depth' / f'{int(frames[r[-1]])}.npy').astype(np.float32).reshape(-1)
            flat = np.flatnonzero(z > 0)
            inp = rng_for(self.seed, index, PURPOSE_DEPTH_SPLIT, s, CAMERA_KEY).random(len(flat)) < INPUT_DEPTH_FRACTION
            depth[s, 0, flat[inp]], depth_valid[s, 0, flat[inp]] = z[flat[inp]], True
            target[s, 0, flat[~inp]], target_valid[s, 0, flat[~inp]] = z[flat[~inp]], True
        tube_dt = np.array([(frames[r[-1]] - frames[r[0]]) / fps / max(F - 1, 1) for r in rows])

        img = torch.from_numpy(images).permute(0, 1, 2, 5, 3, 4).contiguous()
        if not self.uint8_images:
            img = img.float().div_(255)
        hw, M = (T, V, H, W), self.max_boxes
        return dict(
            images=img, tube_dt=torch.from_numpy(tube_dt).float(),
            intrinsics=torch.from_numpy(intrinsics).float(), cam_to_rig=torch.from_numpy(cam_to_rig).float(),
            camera_key=torch.from_numpy(camera_key), camera_valid=torch.from_numpy(camera_valid),
            depth=torch.from_numpy(depth).view(hw), depth_valid=torch.from_numpy(depth_valid).view(hw),
            target_depth=torch.from_numpy(target).view(hw), target_valid=torch.from_numpy(target_valid).view(hw),
            rig_pose=torch.from_numpy(world_T_rig).float(), rig_pose_valid=torch.ones(T, dtype=torch.bool),
            has_metric_depth=torch.ones(T, dtype=torch.bool),
            t=torch.from_numpy((anchor - anchor[0]) / fps).double(),
            timestamp_ns=torch.from_numpy(np.round(anchor / fps * 1e9).astype(np.int64)),
            log_id=f'{SOURCE}/{uid}/{name}',
            boxes=dict(center=torch.zeros(T, M, 3), size=torch.zeros(T, M, 3),
                       quaternion=torch.tensor([1., 0., 0., 0.]).repeat(T, M, 1),
                       category=torch.full((T, M), -1, dtype=torch.int64), track=torch.full((T, M), -1, dtype=torch.int64),
                       num_interior_pts=torch.zeros(T, M, dtype=torch.int64), mask=torch.zeros(T, M, dtype=torch.bool)))


def _open_one(root, split, cfg, **kw):
    d = Path(root).expanduser() / split if split else Path(root).expanduser()
    info = d / PACK_INFO
    if info.exists() and json.loads(info.read_text()).get('source') == SOURCE:
        return OmniWorldClipDataset(root, split, cfg, **kw)
    return data_av2.open_av2(Path(root).expanduser(), split, cfg, **kw)


def open_clips(roots, split, cfg, **kw):
    """Clips of one or more data roots (comma-separated): an OmniWorld pack or an AV2 split (raw or packed).
    ponytail: several roots are concatenated, so a batch draws sources in proportion to their clip counts; set a
    mixture with a weighted sampler when that matters."""
    parts = [_open_one(r.strip(), split, cfg, **kw) for r in str(roots).split(',') if r.strip()]
    return parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
