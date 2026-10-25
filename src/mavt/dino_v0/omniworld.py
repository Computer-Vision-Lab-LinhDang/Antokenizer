"""OmniWorld-Game for MAVT on DINOv3: packed format, clips with geometry targets, and the targets' maths.

Packed by scripts/pack_omniworld_dino.py; one directory per split (train / val, by uid hash, the v6 rule):
  <split>/pack.json                    settings (image H x W, base stride, ...)
  <split>/<uid>/scene.json             uid, fps, metric_scale, segments [{name, frames}]
  <split>/<uid>/rgb/<frame>.jpg        H x W (default 256 x 448): cover-resize of the 1280 x 720 frame, centre crop
  <split>/<uid>/<seg>/meta.npz         a camera-valid run of one OmniWorld split, every `base_stride`-th raw frame:
                                       frames [F] raw indices, K [F,3,3] at H x W (pixel centres at integers),
                                       c2w [F,4,4] OpenCV camera -> world (m), pair_absrel [F-1] depth reprojection
                                       AbsRel to the next frame (camera check), pair_epe [F-1] median |flow - rigid
                                       flow| on static pixels (px of H x W; NaN without flow)
  <split>/<uid>/<seg>/depth.npy        [F, h, w] float16 camera z (m), 0 = invalid; h, w = H / AUX, W / AUX
  <split>/<uid>/<seg>/dyn.npy          [F, h, w] uint8 share of dynamic-object pixels (gdino mask) x 255
  <split>/<uid>/<seg>/flow.npy         [F, h, w, 2] float16 forward displacement to the next packed frame (px of H x W),
                                       NaN where unknown (last frame, no flow in the scene, landing off the frame)

A clip (OmniWorldClips) is n_frames packed frames at a packed stride, cropped to size x size with one horizontal
offset (multiple of the cell size) for the whole clip. The camera goes into the model relative to the clip's first
frame, translations divided by that frame's median depth (Plücker moments and pose inputs stay O(1), like Objaverse).

Geometry targets (corr_targets): for every keyframe k of the clip and frame t >= k, the decoder's shift
Δ*(cell at t) = (x_k - x_t) / size in (u, v) units, i.e. where frame t's content sits in keyframe k:
  static cells (dynamic share < 0.5): frame t's depth lifted with its camera, projected into k; a pixel counts
    when visible in k (depth test against k's depth) and inside the crop; a cell needs >= MIN_STATIC of its pixels.
  dynamic cells: frame k's dynamic pixels moved by the packed flow chained from k to t; a cell needs >= MIN_DYN
    landed samples that agree within MAX_SPREAD px.
  t == k: Δ* = 0 everywhere.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
import math
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

SOURCE = "omniworld-game"
NEAR, FAR = 1.0, 1000.0          # OmniWorld README 2.3 depth encoding
MAX_DEPTH = 250.0                # m; beyond it depth is invalid (as v6)
FLOW_RANGE = 300.0               # README 2.6: uint16 -> [-300, 300] px (its snippet's default flips the minimum)
AUX = 4                          # image px per depth / dyn / flow pixel
MAX_PAIR_ABSREL = 0.05           # v6: broken cameras show 9-30 % median depth AbsRel between steps, good ones < 4 %
MIN_STATIC = 8                   # of the AUX x AUX = 16 pixels of a cell (at cell = 16 px)
MIN_DYN = 4
MAX_SPREAD = 16.0                # px: landed samples of one dynamic cell must agree within one cell
MOVE_PX = 2.0                    # a pixel whose flow differs from the rigid flow by more than this moves (moving_mask)
EDGE_LOG_DEPTH = 0.1             # |log z| jump to a 4-neighbour that marks a depth edge (flow sampling is unreliable there)


# ---------------------------------------------------------------------------- source decoding (packer)
def split_bucket(uid: str, salt: str = "mavt-split-v1") -> float:
    """v6 scene split: sha1('mavt-split-v1\\0omniworld-game:<uid>') in [0, 1)."""
    return int(hashlib.sha1(f"{salt}\0{SOURCE}:{uid}".encode()).hexdigest()[:15], 16) / 16 ** 15


def decode_depth(raw: np.ndarray, metric_scale: float) -> np.ndarray:
    """uint16 depth PNG -> camera z in metres float32, 0 where invalid (too close, sky, > MAX_DEPTH)."""
    d = raw.astype(np.float64) / 65535.0
    z = d / (FAR - d * (FAR - NEAR)) / 0.004 * float(metric_scale)
    valid = (d >= 0.0015) & (d <= 65500.0 / 65535.0) & (z <= MAX_DEPTH)
    return np.where(valid, z, 0.0).astype(np.float32)


def decode_flow(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """flow_u_16 / flow_v_16 uint16 PNGs -> forward flow to the next raw frame [H, W, 2] float32 (source px)."""
    return np.stack([u, v], -1).astype(np.float32) / 65535.0 * (2 * FLOW_RANGE) - FLOW_RANGE


def quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (np.asarray(q, np.float64) / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1).reshape(-1, 3, 3)


def split_cameras(cam: dict, metric_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """camera/split_<i>.json -> (K [n,3,3] source px with pixel centres at integers, c2w [n,4,4] metres), as v6."""
    q = np.asarray(cam["quats"], np.float64)
    n = q.shape[0]
    focal = np.broadcast_to(np.asarray(cam["focals"], np.float64).ravel(), (n,))
    cx, cy = (float(np.asarray(cam[c], np.float64).ravel()[0]) - 0.5 for c in ("cx", "cy"))
    K = np.zeros((n, 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = focal, focal, cx, cy, 1.0
    w2c = np.tile(np.eye(4), (n, 1, 1))
    w2c[:, :3, :3] = quat_matrix(q)
    w2c[:, :3, 3] = np.asarray(cam["trans"], np.float64)
    c2w = np.linalg.inv(w2c)
    c2w[:, :3, 3] *= float(metric_scale)
    return K, c2w


def cover_crop(src_w: int, src_h: int, height: int, width: int) -> tuple[float, float, float]:
    """Scale s and source offsets (x0, y0) of a cover-resize to height x width with a centre crop."""
    s = max(height / src_h, width / src_w)
    return s, (src_w - width / s) / 2, (src_h - height / s) / 2


def adjust_intrinsics(K: np.ndarray, s: float, x0: float, y0: float) -> np.ndarray:
    """K after scaling by s and cropping at source offset (x0, y0); pixel centres at integers (v6)."""
    K = np.asarray(K, np.float64).copy()
    K[..., 0, 0] *= s
    K[..., 1, 1] *= s
    K[..., 0, 2] = (K[..., 0, 2] + 0.5 - x0) * s - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5 - y0) * s - 0.5
    return K


def aux_centres(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Image-px coordinates (x [h,w], y [h,w]) of the AUX x AUX pixel blocks' centres."""
    xs = np.arange(w) * AUX + (AUX - 1) / 2
    ys = np.arange(h) * AUX + (AUX - 1) / 2
    return np.meshgrid(xs, ys)


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample img [H, W, C] at float pixel coordinates (centres at integers), edge-clamped -> [..., C]."""
    H, W = img.shape[:2]
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    ax, ay = (x - x0)[..., None], (y - y0)[..., None]
    return ((img[y0, x0] * (1 - ax) + img[y0, x1] * ax) * (1 - ay)
            + (img[y1, x0] * (1 - ax) + img[y1, x1] * ax) * ay)


def project(x: np.ndarray, y: np.ndarray, z: np.ndarray, K_a: np.ndarray, c2w_a: np.ndarray,
            K_b: np.ndarray, c2w_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixels (x, y) of view a at camera depth z -> (x_b, y_b, z_b) in view b (float64, any shape)."""
    p = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K_a).T * z[..., None]
    rel = np.linalg.inv(c2w_b) @ c2w_a
    q = p @ rel[:3, :3].T + rel[:3, 3]
    zb = q[..., 2]
    uv = q @ np.asarray(K_b, np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return uv[..., 0] / zb, uv[..., 1] / zb, zb


def rigid_flow_epe(depth_a: np.ndarray, dyn_a: np.ndarray, flow_a: np.ndarray, K_a, c2w_a, K_b, c2w_b) -> float:
    """Median |flow - rigid flow| (px) over static valid pixels of view a (aux grid, image px); NaN when none."""
    h, w = depth_a.shape
    xs, ys = aux_centres(h, w)
    xb, yb, zb = project(xs, ys, depth_a.astype(np.float64), K_a, c2w_a, K_b, c2w_b)
    rigid = np.stack([xb - xs, yb - ys], -1)
    ok = (depth_a > 0) & (dyn_a < 128) & (zb > 1e-6) & np.isfinite(flow_a).all(-1)
    if not ok.any():
        return math.nan
    return float(np.median(np.linalg.norm(flow_a[ok] - rigid[ok], axis=-1)))


def depth_absrel(depth_a: np.ndarray, K_a, c2w_a, depth_b: np.ndarray, K_b, c2w_b) -> float:
    """Median |z - z_b| / z_b of view a's depth reprojected into view b (aux grids); NaN when nothing lands."""
    h, w = depth_a.shape
    xs, ys = aux_centres(h, w)
    xb, yb, zb = project(xs, ys, depth_a.astype(np.float64), K_a, c2w_a, K_b, c2w_b)
    ok = (depth_a > 0) & (zb > 1e-6)
    iu, iv = np.floor(xb[ok] / AUX).astype(np.int64), np.floor(yb[ok] / AUX).astype(np.int64)
    inside = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    ref = depth_b[iv[inside], iu[inside]].astype(np.float64)
    hit = ref > 0
    if not hit.any():
        return math.nan
    return float(np.median(np.abs(zb[ok][inside][hit] - ref[hit]) / ref[hit]))


def moving_mask(depth: np.ndarray, flow: np.ndarray, K_a, c2w_a, K_b, c2w_b, offset: int = 0) -> np.ndarray:
    """[h, w] bool: pixels of view a whose flow to view b differs from the rigid (camera-only) flow by > MOVE_PX,
    away from depth edges. gdino masks miss some movers: on 8 scenes 5 % of its 'static' pixels disagree, 67 % of
    them on depth edges (aux-grid sampling) and 25 % in clusters off edges (movers). depth [h, w] on a crop starting
    offset px into the frame of flow [h, W', 2]; K for the crop."""
    h, w = depth.shape
    xs, ys = aux_centres(h, w)
    xb, yb, zb = project(xs, ys, depth.astype(np.float64), K_a, c2w_a, K_b, c2w_b)
    a0 = offset // AUX
    f = np.asarray(flow[:, a0:a0 + w], np.float32)
    ok = (depth > 0) & (zb > 1e-6) & np.isfinite(f).all(-1)
    err = np.linalg.norm(np.nan_to_num(f) - np.stack([xb - xs, yb - ys], -1), axis=-1)
    lz = np.log(np.maximum(depth.astype(np.float64), 1e-3))
    g = np.zeros_like(lz)
    dx, dy = np.abs(np.diff(lz, axis=1)), np.abs(np.diff(lz, axis=0))
    for sl_a, sl_b, d in (((slice(None), slice(1, None)), (slice(None), slice(None, -1)), dx),
                          ((slice(1, None),), (slice(None, -1),), dy)):
        np.maximum(g[sl_a], d, out=g[sl_a])                         # both pixels of each jump are on the edge
        np.maximum(g[sl_b], d, out=g[sl_b])
    return ok & (err > MOVE_PX) & (g <= EDGE_LOG_DEPTH)


# ---------------------------------------------------------------------------- geometry targets (dataset)
def chain_flow(flows: Sequence[np.ndarray], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move image-px points (x, y) through consecutive forward flows (each [h, w, 2] on the aux grid, image px).
    -> (x', y', ok); ok is False once a point leaves the frame or meets an unknown (NaN) flow."""
    ok = np.ones(x.shape, bool)
    for f in flows:
        h, w = f.shape[:2]
        d = bilinear(np.asarray(f, np.float32), (x - (AUX - 1) / 2) / AUX, (y - (AUX - 1) / 2) / AUX)
        ok &= np.isfinite(d).all(-1)
        x, y = x + np.nan_to_num(d[..., 0]), y + np.nan_to_num(d[..., 1])
        ok &= (x >= -0.5) & (x <= w * AUX - 0.5) & (y >= -0.5) & (y <= h * AUX - 0.5)
    return x, y, ok


def _cells(a: np.ndarray, cell_aux: int) -> np.ndarray:
    """[h, w, ...] aux-grid array -> [h/c, w/c, c*c, ...] blocks of each cell."""
    h, w = a.shape[:2]
    g, gw = h // cell_aux, w // cell_aux
    b = a.reshape(g, cell_aux, gw, cell_aux, *a.shape[2:]).swapaxes(1, 2)
    return b.reshape(g, gw, cell_aux * cell_aux, *a.shape[2:])


def corr_targets(depth: np.ndarray, dyn: np.ndarray, K: np.ndarray, c2w: np.ndarray, chains: dict,
                 keys: Sequence[int], size: int, cell: int, offset: int) -> dict:
    """Decoder-shift targets of one clip (see the module docstring).

    depth / dyn [T, h, h] float on the crop's aux grid (dyn in [0, 1]); K [T,3,3] for the crop, c2w [T,4,4];
    chains {(k, t): list of full-width flow arrays from frame k to frame t} (missing: no dynamic targets);
    keys: keyframe indices; offset: the crop's left edge in packed px (aux grid columns offset / AUX).
    -> corr [nK, T, G, G, 2] (u, v units), valid [nK, T, G, G] bool, dyncell [T, G, G] bool."""
    T, h, _ = depth.shape
    c = cell // AUX
    G = h // c
    corr = np.zeros((len(keys), T, G, G, 2), np.float32)
    valid = np.zeros((len(keys), T, G, G), bool)
    dyncell = _cells(dyn.transpose(1, 2, 0), c).mean(2).transpose(2, 0, 1) >= 0.5          # [T, G, G]
    xs, ys = aux_centres(h, h)
    for ki, k in enumerate(keys):
        for t in range(k, T):
            if t == k:
                valid[ki, t] = True
                continue
            # static cells: rigid backward projection t -> k with a visibility test in k
            xk, yk, zk = project(xs, ys, depth[t].astype(np.float64), K[t], c2w[t], K[k], c2w[k])
            inside = (xk >= -0.5) & (xk < size - 0.5) & (yk >= -0.5) & (yk < size - 0.5) & (zk > 1e-6)
            iu = np.clip(np.floor((xk + 0.5) / AUX).astype(np.int64), 0, h - 1)
            iv = np.clip(np.floor((yk + 0.5) / AUX).astype(np.int64), 0, h - 1)
            zref = depth[k][iv, iu].astype(np.float64)
            seen = inside & (zref > 0) & (zk <= zref * 1.03 + 1e-3)
            ok = (depth[t] > 0) & (dyn[t] < 0.5) & seen
            disp = np.stack([xk - xs, yk - ys], -1)
            okc, dc = _cells(ok, c), _cells(np.where(ok[..., None], disp, 0.0), c)
            n = okc.sum(2)
            st = (n >= MIN_STATIC) & ~dyncell[t]
            corr[ki, t][st] = (dc.sum(2)[st] / n[st][:, None] / size).astype(np.float32)
            valid[ki, t] |= st
            # dynamic cells: frame k's dynamic pixels moved by the chained flow
            flows = chains.get((k, t))
            if flows is None or not dyncell[t].any():
                continue
            src = dyn[k] >= 0.5
            x0, y0 = xs[src] + offset, ys[src]
            x1, y1, okf = chain_flow(flows, x0, y0)
            x1 = x1 - offset
            land = okf & (x1 >= -0.5) & (x1 < size - 0.5) & (y1 >= -0.5) & (y1 < size - 0.5)
            ci = np.clip(np.floor((y1[land] + 0.5) / cell).astype(np.int64), 0, G - 1)
            cj = np.clip(np.floor((x1[land] + 0.5) / cell).astype(np.int64), 0, G - 1)
            d = np.stack([x0[land] - offset - x1[land], y0[land] - y1[land]], -1)
            for (i, j) in set(zip(ci.tolist(), cj.tolist())):
                if not dyncell[t, i, j]:
                    continue
                m = (ci == i) & (cj == j)
                dd = d[m]
                if dd.shape[0] >= MIN_DYN and float(dd.std(0).max()) <= MAX_SPREAD:
                    corr[ki, t, i, j] = dd.mean(0) / size
                    valid[ki, t, i, j] = True
    return dict(corr=corr, valid=valid, dyncell=dyncell)


# ---------------------------------------------------------------------------- dataset
def _scenes(root: str) -> list:
    out = []
    for uid in sorted(os.listdir(root)):
        f = os.path.join(root, uid, "scene.json")
        if os.path.exists(f):
            with open(f) as fh:
                out.append(json.load(fh))
    return out


class OmniWorldClips(Dataset):
    """Clips of n_frames packed frames (raw stride `frame_stride`, a multiple of the pack's base stride) with
    camera and geometry targets. Returns {'data': [3,T,S,S] in [-1, 1], 'modality': 'world', 'c2w' [T,4,4]
    (clip-relative, depth-normalised), 'K' [T,3,3], 'stride', 'geo_depth' [T,h,h] (m), 'geo_dyn' [T,h,h],
    'geo_corr' [nK,T,G,G,2], 'geo_valid' [nK,T,G,G], 'geo_dyncell' [T,G,G]}. Train: a random stride among those
    that fit, random start, random crop offset (multiple of cell); eval: the first stride, centre crop.
    A clip is redrawn (up to max_retries) when a camera pair in it fails the depth AbsRel check."""

    def __init__(self, root: str, split: str = "train", n_frames: int = 8, frame_stride=2, size: int = 256,
                 cell: int = 16, keyframe_every: int = 4, train: bool = True, chunk: int = 16, max_retries: int = 20,
                 eval_stride: Optional[int] = None, cache_segments: int = 64, geo_targets: bool = True):
        from .data import parse_strides
        self.root = os.path.join(root, split)
        with open(os.path.join(self.root, "pack.json")) as f:
            self.info = json.load(f)
        self.base = int(self.info["base_stride"])
        strides = parse_strides(frame_stride)
        self.pstrides = sorted({s // self.base for s in strides if s % self.base == 0 and s >= self.base})
        if not self.pstrides:
            raise ValueError(f"no frame stride in {strides} is a multiple of the pack's base stride {self.base}")
        self.eval_ps = (int(eval_stride) // self.base) if eval_stride else self.pstrides[0]
        self.n_frames, self.size, self.cell, self.K, self.train = n_frames, size, cell, keyframe_every, train
        self.max_retries, self.chunk, self.geo_targets = max_retries, chunk, geo_targets
        if int(self.info["image_height"]) != size or (int(self.info["image_width"]) - size) % cell:
            raise ValueError(f"pack is {self.info['image_height']} x {self.info['image_width']}: needs height {size} "
                             f"and a width that crops to {size} in steps of {cell}")
        self.segs, self.items = [], []
        min_span = (n_frames - 1) * min(self.pstrides) + 1
        for sc in _scenes(self.root):
            for seg in sc["segments"]:
                F = int(seg["frames"])
                if F < min_span:
                    continue
                si = len(self.segs)
                self.segs.append((sc["uid"], seg["name"]))
                self.items += [(si, a) for a in range(0, F - min_span + 1, chunk)]
        if not self.items:
            raise RuntimeError(f"no clips under {self.root}")
        self._rng = random.Random()
        self._cache: OrderedDict = OrderedDict()                  # LRU: each segment holds 3 open memory maps
        self.cache_segments = cache_segments

    def __len__(self) -> int:
        return len(self.items)

    def _seg(self, si: int) -> dict:
        """A segment's meta and memory-mapped arrays (only the clip's frames are read). The cache is bounded:
        3 open maps per segment, and the 3,609 train segments would pass a 1024 open-file limit."""
        if si in self._cache:
            self._cache.move_to_end(si)
            return self._cache[si]
        uid, name = self.segs[si]
        d = os.path.join(self.root, uid, name)
        with np.load(os.path.join(d, "meta.npz")) as m:
            meta = {k: m[k] for k in m.files}
        seg = dict(uid=uid, meta=meta, **{k: np.load(os.path.join(d, f"{k}.npy"), mmap_mode="r")
                                          for k in ("depth", "dyn", "flow")})
        self._cache[si] = seg
        while len(self._cache) > self.cache_segments:
            self._cache.popitem(last=False)
        return seg

    def _draw(self, i: int, rng: random.Random) -> Optional[tuple]:
        si, a = self.items[i]
        s = self._seg(si)
        F = len(s["meta"]["frames"])
        fits = [p for p in self.pstrides if a + (self.n_frames - 1) * p < F]
        ps = rng.choice(fits) if self.train else (self.eval_ps if self.eval_ps in fits else fits[0])
        span = (self.n_frames - 1) * ps + 1
        j0 = a + (rng.randint(0, min(self.chunk - 1, F - span - a)) if self.train else 0)
        bad = s["meta"]["pair_absrel"][j0:j0 + span - 1]
        if not np.all(np.nan_to_num(bad, nan=1.0) <= MAX_PAIR_ABSREL):
            return None
        return si, j0, ps

    def __getitem__(self, i: int) -> dict:
        rng = self._rng if self.train else random.Random(i)
        for _ in range(self.max_retries):
            got = self._draw(i, rng)
            if got is not None:
                return self._load(*got, rng)
            i = rng.randrange(len(self.items)) if self.train else (i + 1) % len(self.items)
        raise RuntimeError("OmniWorldClips: no clip with valid cameras after max_retries draws")

    def _load(self, si: int, j0: int, ps: int, rng: random.Random) -> dict:
        from PIL import Image
        s = self._seg(si)
        meta = s["meta"]
        idx = j0 + ps * np.arange(self.n_frames)
        H, W = int(self.info["image_height"]), int(self.info["image_width"])
        S, cell = self.size, self.cell
        n_off = (W - S) // cell
        off = cell * (rng.randint(0, n_off) if self.train else n_off // 2)
        imgs = []
        for f in meta["frames"][idx]:
            im = Image.open(os.path.join(self.root, s["uid"], "rgb", f"{int(f)}.jpg")).convert("RGB")
            imgs.append(np.asarray(im)[:S, off:off + S])
        data = torch.from_numpy(np.stack(imgs)).permute(3, 0, 1, 2).float() / 127.5 - 1.0     # [3,T,S,S]
        K = meta["K"][idx].copy()
        K[:, 0, 2] -= off
        c2w = meta["c2w"][idx]
        a0, a1 = off // AUX, (off + S) // AUX
        depth = np.asarray(s["depth"][idx][:, :S // AUX, a0:a1], np.float32)
        z0 = depth[0][depth[0] > 0]
        scale = float(np.median(z0)) if z0.size else 1.0
        rel = np.linalg.inv(c2w[0]) @ c2w
        rel[:, :3, 3] /= scale
        out = dict(data=data, modality="world", c2w=torch.from_numpy(rel).float(), K=torch.from_numpy(K).float(),
                   stride=ps * self.base, depth=torch.from_numpy(depth))
        if not self.geo_targets:                                    # cameras + depth only (the cd32 models)
            return out
        dyn = np.asarray(s["dyn"][idx][:, :S // AUX, a0:a1], np.float32) / 255.0
        Kf, F = meta["K"].copy(), len(meta["frames"])
        Kf[:, 0, 2] -= off
        for t, p in enumerate(idx):                                 # movers gdino missed: flow vs rigid flow
            if p + 1 < F and np.isfinite(meta["pair_epe"][p]):
                mv = moving_mask(depth[t], s["flow"][p], Kf[p], meta["c2w"][p], Kf[p + 1], meta["c2w"][p + 1], off)
                dyn[t] = np.maximum(dyn[t], mv)
        keys = list(range(0, self.n_frames, self.K))
        chains = {}
        if np.isfinite(meta["pair_epe"][j0:idx[-1]]).all():         # flow known for every step of the clip
            for k in keys:
                for t in range(k + 1, self.n_frames):
                    chains[(k, t)] = [s["flow"][p] for p in range(int(idx[k]), int(idx[t]))]
        tg = corr_targets(depth, dyn, K, c2w, chains, keys, S, cell, off)
        out.update(geo_depth=out["depth"], geo_dyn=torch.from_numpy(dyn), geo_corr=torch.from_numpy(tg["corr"]),
                   geo_valid=torch.from_numpy(tg["valid"]), geo_dyncell=torch.from_numpy(tg["dyncell"]))
        return out
