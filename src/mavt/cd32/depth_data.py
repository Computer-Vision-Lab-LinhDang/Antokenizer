"""Depth-labelled images for the depth probe (scripts/train_depth_probe.py) and the depth eval.

A manifest is a jsonl file, one row per image:

    {"image": "rgb/0001.jpg", "depth": "depth/0001.png", "scale": 0.001, "max_depth": 10.0}

- image, depth: paths, relative to the manifest's folder unless absolute.
- scale: metres (or the dataset's unit) per stored value, e.g. 0.001 for uint16 millimetres. Default 1.
- min_depth / max_depth (optional): values outside are invalid. 0, NaN and inf are always invalid.
- index (optional): frame of a stacked [F, H, W] array (.npy is memory-mapped, so only that frame is read).
- key (optional): array name inside .npz / .h5 files (default "depth").
- min_cell_frac (optional): share of valid pixels a cell needs to count (default: the trainer's value);
  sparse labels such as projected LiDAR need a low one.

The depth map may have a lower or higher resolution than the image, with the same aspect ratio.
Depth files: .png (8 or 16 bit), .npy, .npz, .h5 / .hdf5, .pfm, .exr (needs OpenCV with OpenEXR, or
imageio). Items are {'data' [3, S, S] in [-1, 1], 'depth' [S, S] float32 (0 = invalid), 'min_cell_frac'}:
training draws a random square crop (35 to 100 % of the largest square, by area) and a horizontal flip,
eval takes the centre crop. Depth is resampled without mixing valid and invalid pixels: nearest neighbour
when the map is coarser than the output, else the geometric mean of the valid pixels under each output pixel.
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"{path}: not a PFM file")
        w, h = map(int, f.readline().decode("latin-1").split())
        scale = float(f.readline().decode("latin-1").strip())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
    ch = 3 if header == "PF" else 1
    return np.flipud(data.reshape(h, w, ch)[..., 0]).astype(np.float32)


def read_depth(path: str, key: str = "depth", index: Optional[int] = None) -> np.ndarray:
    """Depth file -> float32 [H, W] of stored values (not yet scaled)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        from PIL import Image
        with Image.open(path) as im:
            a = np.array(im)
    elif ext == ".npy":
        a = np.load(path, mmap_mode="r")
        a = a[index] if index is not None else a
    elif ext == ".npz":
        with np.load(path) as z:
            a = z[key] if key in z else z[z.files[0]]
            a = a[index] if index is not None else a
    elif ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            a = np.array(f[key][index] if index is not None else f[key])
    elif ext == ".pfm":
        a = read_pfm(path)
    elif ext == ".exr":
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        try:
            import cv2
            a = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if a is None:
                raise OSError("cv2 could not read it")
        except (ImportError, OSError):
            import imageio.v3 as iio
            a = iio.imread(path)
    else:
        raise ValueError(f"unknown depth format: {path}")
    a = np.array(a, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def clean_depth(raw: np.ndarray, scale: float = 1.0, min_depth: float = 0.0,
                max_depth: float = float("inf")) -> np.ndarray:
    d = raw.astype(np.float32) * scale
    bad = ~np.isfinite(d) | (d <= max(0.0, min_depth)) | (d > max_depth)
    d[bad] = 0.0
    return d


def _crop_box(w: int, h: int, size: int, rng: Optional[random.Random], scale=(0.35, 1.0)) -> tuple:
    """Square crop (left, top, side) in image pixels: random for training, centre for eval."""
    s = min(w, h)
    if rng is None:
        return (w - s) // 2, (h - s) // 2, s
    side = min(s, max(size // 2, int(round(s * rng.uniform(*scale) ** 0.5))))
    return rng.randint(0, w - side), rng.randint(0, h - side), side


def resample_depth(depth: np.ndarray, x0: float, y0: float, side_x: float, side_y: float, size: int) -> np.ndarray:
    """Depth map region [x0, x0 + side_x) x [y0, y0 + side_y) (depth-map pixels) -> [size, size], 0 = invalid.
    Coarser map: nearest neighbour. Finer map: geometric mean of the valid pixels under each output pixel."""
    dh, dw = depth.shape
    if side_x <= size and side_y <= size:
        xs = np.clip(np.floor(x0 + (np.arange(size) + 0.5) * side_x / size), 0, dw - 1).astype(np.int64)
        ys = np.clip(np.floor(y0 + (np.arange(size) + 0.5) * side_y / size), 0, dh - 1).astype(np.int64)
        return depth[ys][:, xs]
    r0, r1 = max(0, int(np.floor(y0))), min(dh, int(np.ceil(y0 + side_y)))
    c0, c1 = max(0, int(np.floor(x0))), min(dw, int(np.ceil(x0 + side_x)))
    sub = depth[r0:r1, c0:c1]
    yy, xx = np.nonzero(sub > 0)
    out = np.zeros((size, size), np.float32)
    if yy.size == 0:
        return out
    oy = np.floor((yy + r0 + 0.5 - y0) * size / side_y).astype(np.int64)
    ox = np.floor((xx + c0 + 0.5 - x0) * size / side_x).astype(np.int64)
    ok = (oy >= 0) & (oy < size) & (ox >= 0) & (ox < size)
    idx = oy[ok] * size + ox[ok]
    logd = np.log(sub[yy[ok], xx[ok]].astype(np.float64))
    s = np.bincount(idx, weights=logd, minlength=size * size)
    n = np.bincount(idx, minlength=size * size)
    has = n > 0
    out.reshape(-1)[has] = np.exp(s[has] / n[has]).astype(np.float32)
    return out


def crop_pair(img, depth: np.ndarray, size: int, rng: Optional[random.Random]):
    """PIL image + depth [h, w] (any resolution with the same aspect) -> uint8 [S, S, 3], float32 [S, S]."""
    from PIL import Image
    w, h = img.size
    left, top, side = _crop_box(w, h, size, rng)
    im = img.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)
    dh, dw = depth.shape
    sx, sy = dw / w, dh / h
    d = resample_depth(depth, left * sx, top * sy, side * sx, side * sy, size)
    a = np.array(im.convert("RGB"))
    if rng is not None and rng.random() < 0.5:
        a, d = a[:, ::-1], d[:, ::-1]
    return np.ascontiguousarray(a), np.ascontiguousarray(d)


class DepthManifest(Dataset):
    """Rows of a depth manifest (see the module doc) -> {'data', 'depth', 'min_cell_frac', 'modality'}."""

    def __init__(self, manifest: str, size: int = 256, train: bool = True, limit: int = 0,
                 min_cell_frac: float = 0.5):
        self.root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        if limit > 0:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise RuntimeError(f"empty depth manifest: {manifest}")
        self.size, self.train, self.min_cell_frac = size, train, min_cell_frac
        self._rng = random.Random()

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image
        r = self.rows[i]
        with Image.open(self._path(r["image"])) as im:
            img = im.convert("RGB")
        raw = read_depth(self._path(r["depth"]), r.get("key", "depth"), r.get("index"))
        d = clean_depth(raw, float(r.get("scale", 1.0)), float(r.get("min_depth", 0.0)),
                        float(r.get("max_depth", float("inf"))))
        a, d = crop_pair(img, d, self.size, self._rng if self.train else None)
        x = torch.from_numpy(a).float().div_(127.5).sub_(1.0).permute(2, 0, 1)
        return dict(data=x, depth=torch.from_numpy(d), modality="image",
                    min_cell_frac=float(r.get("min_cell_frac", self.min_cell_frac)))


def depth_datasets(manifests: Sequence[str], size: int, train: bool, limit: int = 0,
                   min_cell_frac: float = 0.5) -> Dataset:
    sets = [DepthManifest(m, size, train, limit, min_cell_frac) for m in manifests]
    return sets[0] if len(sets) == 1 else ConcatDataset(sets)


def worker_init(wid: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    seed = (torch.initial_seed() + wid) % 2 ** 32
    for ds in getattr(info.dataset, "datasets", [info.dataset]):
        if hasattr(ds, "_rng"):
            ds._rng = random.Random(seed)
