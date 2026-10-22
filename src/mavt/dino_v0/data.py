"""Data for MAVT on DINOv3 v0: ImageNet-1k (HF parquet) images and OpenVid-1M clips.

All datasets return dicts with 'data' in [-1, 1] and 'modality'; layouts follow the model:
image [3, H, W], video and posed 3D views [3, T, H, W] (threed adds 'c2w' [T,4,4] and 'K' [T,3,3]).

ImageNet parquet shards hold ~100 rows per row group, so random row access would re-read a row
group per sample. `ParquetGroupBatchSampler` instead gives each DataLoader worker its own stream
of row groups (shuffled in windows of a few groups) so every row group is read once per epoch.
"""
from __future__ import annotations

import bisect
import glob
import io
import json
import logging
import math
import os
import random
from collections import OrderedDict
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- transforms
def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 [H, W, 3] (or [T, H, W, 3]) -> float in [-1, 1], channels first ([3,H,W] / [3,T,H,W])."""
    t = torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True)).float().div_(127.5).sub_(1.0)
    return t.permute(2, 0, 1) if t.dim() == 3 else t.permute(3, 0, 1, 2)


def _resize_short(img, size: int):
    from PIL import Image
    w, h = img.size
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    return img.resize((nw, nh), Image.BICUBIC)


def _center_crop(img, size: int):
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def _random_resized_crop(img, size: int, rng: random.Random, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3)):
    from PIL import Image
    w, h = img.size
    area = w * h
    for _ in range(10):
        a = area * rng.uniform(*scale)
        r = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
        if 0 < cw <= w and 0 < ch <= h:
            l, t = rng.randint(0, w - cw), rng.randint(0, h - ch)
            return img.resize((size, size), Image.BICUBIC, box=(l, t, l + cw, t + ch))
    return _center_crop(_resize_short(img, size), size)


# ---------------------------------------------------------------------------- ImageNet parquet
class ImageNetParquet(Dataset):
    """ImageNet-1k in the HF parquet layout: <root>/data/<split>-*.parquet (or <root>/<split>-*.parquet).

    Index = row groups from parquet footers only (cheap). Each worker keeps a small LRU cache of
    decoded row groups (image bytes + labels)."""

    def __init__(self, root: str, split: str = "train", size: int = 256, train: Optional[bool] = None,
                 cache_groups: int = 8):
        import pyarrow.parquet as pq
        files = sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))
        if not files:
            files = sorted(glob.glob(os.path.join(root, f"{split}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no {split}-*.parquet under {root} or {root}/data")
        self.files, self.size = files, size
        self.train = (split == "train") if train is None else train
        self.cache_groups = cache_groups
        self.groups: list[tuple[int, int, int, int]] = []           # (file, row_group, start, n)
        n = 0
        for fi, f in enumerate(files):
            md = pq.ParquetFile(f).metadata
            for g in range(md.num_row_groups):
                r = md.row_group(g).num_rows
                if r:
                    self.groups.append((fi, g, n, r))
                    n += r
        self._starts = [g[2] for g in self.groups]
        self._len = n
        self._pf: dict = {}
        self._cache: OrderedDict = OrderedDict()
        self._rng = random.Random()

    def __len__(self) -> int:
        return self._len

    def __getstate__(self):                                       # do not ship file handles to workers
        d = self.__dict__.copy()
        d["_pf"], d["_cache"] = {}, OrderedDict()
        return d

    def _group(self, gi: int):
        if gi in self._cache:
            self._cache.move_to_end(gi)
            return self._cache[gi]
        import pyarrow.parquet as pq
        fi, g, _, _ = self.groups[gi]
        pf = self._pf.get(fi)
        if pf is None:
            pf = self._pf[fi] = pq.ParquetFile(self.files[fi])
        cols = [c for c in ("image", "label") if c in pf.schema_arrow.names]
        tab = pf.read_row_group(g, columns=cols)
        img = tab.column("image").combine_chunks()
        imgs = img.field("bytes").to_pylist() if hasattr(img, "field") else img.to_pylist()
        labels = tab.column("label").to_pylist() if "label" in cols else [-1] * len(imgs)
        self._cache[gi] = (imgs, labels)
        while len(self._cache) > self.cache_groups:
            self._cache.popitem(last=False)
        return self._cache[gi]

    def __getitem__(self, i: int) -> dict:
        from PIL import Image
        gi = bisect.bisect_right(self._starts, i) - 1
        imgs, labels = self._group(gi)
        j = i - self.groups[gi][2]
        img = Image.open(io.BytesIO(imgs[j])).convert("RGB")
        if self.train:
            img = _random_resized_crop(img, self.size, self._rng)
            if self._rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = _center_crop(_resize_short(img, self.size), self.size)
        lab = labels[j]
        return dict(data=_to_tensor(np.asarray(img)), modality="image", label=-1 if lab is None else int(lab))


class ParquetGroupBatchSampler(Sampler):
    """Batch sampler for ImageNetParquet that keeps each worker on its own row groups.

    Row groups are shuffled per epoch, split across ranks, then across DataLoader workers. Each
    worker stream takes `window` groups at a time, shuffles their rows and cuts them into batches.
    Batches are emitted round-robin over workers, matching DataLoader's dispatch order, so a worker
    only ever touches its own groups (cache_groups >= window keeps every read to once per epoch)."""

    def __init__(self, ds: ImageNetParquet, batch_size: int, num_workers: int, rank: int = 0,
                 world: int = 1, seed: int = 0, window: int = 4, shuffle: bool = True):
        self.ds, self.bs, self.W = ds, batch_size, max(1, num_workers)
        self.rank, self.world, self.seed, self.window, self.shuffle = rank, world, seed, window, shuffle
        self.epoch = 0
        ds.cache_groups = max(ds.cache_groups, window + 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_stream(self, groups, rng) -> Iterator[list]:
        for k in range(0, len(groups), self.window):
            rows = [i for gi in groups[k:k + self.window]
                    for i in range(self.ds.groups[gi][2], self.ds.groups[gi][2] + self.ds.groups[gi][3])]
            if self.shuffle:
                rng.shuffle(rows)
            for b in range(0, len(rows) - self.bs + 1, self.bs):
                yield rows[b:b + self.bs]

    def __iter__(self):
        rng = random.Random(self.seed * 1000003 + self.epoch)
        order = list(range(len(self.ds.groups)))
        if self.shuffle:
            rng.shuffle(order)
        mine = order[self.rank::self.world]
        streams = [self._worker_stream(mine[w::self.W], random.Random(rng.random())) for w in range(self.W)]
        while True:
            for s in streams:
                b = next(s, None)
                if b is None:
                    return
                yield b

    def __len__(self) -> int:
        return max(1, len(self.ds) // (self.world * self.bs))


# ---------------------------------------------------------------------------- OpenVid clips
def read_manifest(manifest: str, path_key: Optional[str] = "path") -> list[dict]:
    """Read a (possibly still growing) jsonl manifest; relative paths resolve against its dir.
    A directory means its manifest.jsonl, or else all its *.jsonl files in name order."""
    if os.path.isdir(manifest):
        cand = os.path.join(manifest, "manifest.jsonl")
        files = [cand] if os.path.exists(cand) else sorted(glob.glob(os.path.join(manifest, "*.jsonl")))
        if not files:
            raise FileNotFoundError(f"no *.jsonl manifest in {manifest}")
    else:
        files = [manifest]
    out = []
    for mf in files:
        out.extend(_read_jsonl(mf, path_key))
    return out


def _read_jsonl(manifest: str, path_key: Optional[str]) -> list[dict]:
    base = os.path.dirname(os.path.abspath(manifest))
    out = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:                           # partial last line while writing
                continue
            if path_key is not None:
                p = r.get(path_key)
                if not p:
                    continue
                r[path_key] = p if os.path.isabs(p) else os.path.join(base, p)
            out.append(r)
    return out


def _have_av() -> bool:
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def _out_hw(w: int, h: int, size: int) -> tuple[int, int]:
    s = size / min(w, h)
    return max(size, round(w * s)), max(size, round(h * s))


def decode_clip_av(path: str, n_frames: int, stride: int, size: int, rng: random.Random,
                   total_hint: Optional[int] = None, fps_hint: Optional[float] = None) -> np.ndarray:
    """PyAV: random start, seek to the keyframe before it, decode sequentially, keep every
    `stride`-th frame. Frames are rescaled (shorter side = size) inside libswscale.
    Returns uint8 [T, h, w, 3] (not yet cropped). Short videos repeat their last frame."""
    import av
    span = (n_frames - 1) * stride + 1
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        fps = float(st.average_rate or st.guessed_rate or fps_hint or 30.0)
        total = int(st.frames or 0) or int(total_hint or 0)
        if not total and st.duration is not None and st.time_base is not None:
            total = int(float(st.duration * st.time_base) * fps)
        start = rng.randint(0, max(0, total - span)) if total > span else 0
        want = [start + k * stride for k in range(n_frames)]
        tb = float(st.time_base) if st.time_base is not None else 1.0 / fps
        t0 = float(st.start_time or 0) * tb
        if start > 0:
            c.seek(int((t0 + start / fps) / tb), stream=st, backward=True, any_frame=False)
        W, H = st.codec_context.width, st.codec_context.height
        ow, oh = _out_hw(W, H, size)
        got, wi, idx = [], 0, None
        for fr in c.decode(st):
            if fr.pts is not None:
                idx = int(round((fr.pts * tb - t0) * fps))
            else:
                idx = 0 if idx is None else idx + 1
            if wi < len(want) and want[wi] <= idx:                # (< idx: dropped frame / seek overshoot)
                arr = fr.reformat(width=ow, height=oh, format="rgb24").to_ndarray()
                while wi < len(want) and want[wi] <= idx:
                    got.append(arr)
                    wi += 1
            if wi == len(want):
                break
    if not got:
        raise RuntimeError(f"no frames decoded from {path}")
    while len(got) < n_frames:
        got.append(got[-1])
    return np.stack(got[:n_frames])


def decode_clip_torchvision(path: str, n_frames: int, stride: int, size: int, rng: random.Random,
                            total_hint: Optional[int] = None, fps_hint: Optional[float] = None) -> np.ndarray:
    """Fallback via torchvision.io.read_video (slow: decodes the time window at full resolution)."""
    from torchvision.io import read_video
    import torch.nn.functional as F
    fps = float(fps_hint or 30.0)
    total = int(total_hint or 0)
    span = (n_frames - 1) * stride + 1
    start = rng.randint(0, max(0, total - span)) if total > span else 0
    v, _, info = read_video(path, start_pts=start / fps, end_pts=(start + span + 1) / fps,
                            pts_unit="sec", output_format="TCHW")
    if v.shape[0] == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    v = v[::stride][:n_frames]
    if v.shape[0] < n_frames:
        v = torch.cat([v, v[-1:].expand(n_frames - v.shape[0], -1, -1, -1)])
    H, W = v.shape[-2:]
    ow, oh = _out_hw(W, H, size)
    v = F.interpolate(v.float(), size=(oh, ow), mode="bilinear", antialias=True, align_corners=False)
    return v.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def _pick_decoder():
    if _have_av():
        return decode_clip_av
    try:
        import torchvision.io as tio
        if hasattr(tio, "read_video"):
            return decode_clip_torchvision
    except Exception:
        pass
    raise ImportError("OpenVidClips needs a video decoder: `pip install av` (PyAV); "
                      "torchvision.io.read_video is used only as a fallback when it is available.")


def split_tail(items: list, tail: int, split: str) -> list:
    """split 'train' drops the last `tail` items (held out for eval), 'heldout' keeps only them."""
    if split == "all" or tail <= 0:
        return items
    if split == "heldout":
        return items[-tail:]
    if split == "train":
        return items[:-tail] if len(items) > tail else []
    raise ValueError(f"split must be train | heldout | all, got {split}")


class OpenVidClips(Dataset):
    """Random n_frames clips (every `frame_stride`-th frame) from OpenVid-1M at original resolution.

    Returns {'data': [3, T, size, size] in [-1, 1], 'modality': 'video'}. Broken files are replaced
    by another random index (each failing path is logged once). The last `tail` manifest lines (as
    read at init) are the eval hold-out: split='train' excludes them, split='heldout' is only them."""

    def __init__(self, manifest: str, n_frames: int = 8, frame_stride: int = 2, size: int = 256,
                 train: bool = True, max_retries: int = 20, decoder=None, tail: int = 0,
                 split: Optional[str] = None):
        split = split or ("train" if train else "heldout")
        self.items = split_tail(read_manifest(manifest), tail, split)
        if not self.items:
            raise RuntimeError(f"empty manifest {manifest}")
        self.n_frames, self.stride, self.size, self.train = n_frames, frame_stride, size, train
        self.max_retries = max_retries
        self._decode = decoder or _pick_decoder()
        self._bad: set = set()
        self._rng = random.Random()

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, i: int) -> torch.Tensor:
        r = self.items[i]
        clip = self._decode(r["path"], self.n_frames, self.stride, self.size, self._rng,
                            total_hint=r.get("n_frames"), fps_hint=r.get("fps"))
        T, h, w, _ = clip.shape
        s = self.size
        if self.train:
            t, l = self._rng.randint(0, h - s), self._rng.randint(0, w - s)
        else:
            t, l = (h - s) // 2, (w - s) // 2
        return _to_tensor(clip[:, t:t + s, l:l + s])

    def __getitem__(self, i: int) -> dict:
        for _ in range(self.max_retries):
            try:
                return dict(data=self._load(i), modality="video")
            except Exception as ex:                                # broken / truncated / still copying
                p = self.items[i]["path"]
                if p not in self._bad:
                    self._bad.add(p)
                    log.warning("OpenVidClips: skipping %s (%s: %s)", p, type(ex).__name__, ex)
                i = self._rng.randrange(len(self.items))
        raise RuntimeError(f"OpenVidClips: {self.max_retries} consecutive broken files")


# ---------------------------------------------------------------------------- Objaverse posed views
def objaverse_camera(cam: dict, size: int, render_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Render json -> (c2w [4,4] OpenCV, K [3,3] at `size`). The json axes are already OpenCV:
    c2w columns = x, y, z, origin. fx = (W/2) / tan(x_fov/2) at render resolution, scaled to `size`."""
    c2w = torch.eye(4)
    for j, k in enumerate(("x", "y", "z", "origin")):
        c2w[:3, j] = torch.tensor(cam[k], dtype=torch.float32)
    half = render_size / 2
    fx = half / math.tan(cam["x_fov"] / 2)
    fy = half / math.tan(cam.get("y_fov", cam["x_fov"]) / 2)
    s = size / render_size
    K = torch.tensor([[fx * s, 0.0, half * s], [0.0, fy * s, half * s], [0.0, 0.0, 1.0]])
    return c2w, K


class ObjaverseViews(Dataset):
    """n_views random distinct posed renders of one Objaverse object (view 0 = keyframe).

    root/manifest.lvis_full.jsonl rows (uid, category, blank, ...); blank objects are skipped.
    root/renders/<uid>/NNNNN.png (RGBA, transparent bg) + NNNNN.json camera. RGBA is composited
    over a background (train: one random gray level per object, eval: white) and resized to `size`
    (no crop). The last `holdout` non-blank uids are the eval split ('heldout'); 'train' excludes them.
    Returns {'data': [3,T,H,W], 'modality': 'threed', 'c2w': [T,4,4], 'K': [T,3,3], 'label': category}."""

    MANIFEST = "manifest.lvis_full.jsonl"

    def __init__(self, root: str, n_views: int = 4, size: int = 256, train: bool = True, bg: str = "random",
                 holdout: int = 500, split: Optional[str] = None, manifest: Optional[str] = None,
                 max_retries: int = 20):
        self.root, self.n_views, self.size, self.train = root, n_views, size, train
        self.bg = bg if train else ("white" if bg == "random" else bg)
        split = split or ("train" if train else "heldout")
        rows = read_manifest(manifest or os.path.join(root, self.MANIFEST), path_key=None)
        rows = [r for r in rows if r.get("uid") and not r.get("blank", False)]
        self.items = split_tail(rows, holdout, split)
        if not self.items:
            raise RuntimeError(f"no objects for split={split} under {root}")
        self.max_retries = max_retries
        self._bad: set = set()
        self._rng = random.Random()

    @staticmethod
    def heldout_uids(root: str, holdout: int = 500, manifest: Optional[str] = None) -> list[str]:
        rows = read_manifest(manifest or os.path.join(root, ObjaverseViews.MANIFEST), path_key=None)
        return [r["uid"] for r in rows if r.get("uid") and not r.get("blank", False)][-holdout:] if holdout > 0 else []

    def __len__(self) -> int:
        return len(self.items)

    def _views(self, d: str) -> list[str]:
        return sorted(f[:-4] for f in os.listdir(d)
                      if f.endswith(".png") and not f.endswith("_depth.png")
                      and os.path.exists(os.path.join(d, f[:-4] + ".json")))

    def _load(self, i: int, rng: random.Random) -> dict:
        from PIL import Image
        r = self.items[i]
        d = os.path.join(self.root, "renders", r["uid"])
        names = self._views(d)
        if not names:
            raise RuntimeError("no views")
        if self.train:
            pick = rng.sample(names, self.n_views) if len(names) >= self.n_views else \
                [rng.choice(names) for _ in range(self.n_views)]
        else:                                                       # eval: fixed, spread-out views
            pick = [names[(k * len(names)) // self.n_views] for k in range(self.n_views)]
        level = rng.randint(0, 255) if self.bg == "random" else (0 if self.bg == "black" else 255)
        frames, c2ws, Ks = [], [], []
        for n in pick:
            im = Image.open(os.path.join(d, n + ".png")).convert("RGBA")
            rw = im.size[0]
            bgim = Image.new("RGBA", im.size, (level, level, level, 255))
            im = Image.alpha_composite(bgim, im).convert("RGB")
            if im.size != (self.size, self.size):
                im = im.resize((self.size, self.size), Image.BICUBIC)
            frames.append(np.asarray(im))
            with open(os.path.join(d, n + ".json")) as f:
                c2w, K = objaverse_camera(json.load(f), self.size, render_size=rw)
            c2ws.append(c2w)
            Ks.append(K)
        return dict(data=_to_tensor(np.stack(frames)), modality="threed", c2w=torch.stack(c2ws),
                    K=torch.stack(Ks), label=r.get("category", ""))

    def __getitem__(self, i: int) -> dict:
        rng = self._rng if self.train else random.Random(i)
        for _ in range(self.max_retries):
            try:
                return self._load(i, rng)
            except Exception as ex:
                uid = self.items[i]["uid"]
                if uid not in self._bad:
                    self._bad.add(uid)
                    log.warning("ObjaverseViews: skipping %s (%s: %s)", uid, type(ex).__name__, ex)
                i = self._rng.randrange(len(self.items))
        raise RuntimeError(f"ObjaverseViews: {self.max_retries} consecutive broken objects")


# ---------------------------------------------------------------------------- loaders
def collate(batch: list[dict]) -> dict:
    out = dict(data=torch.stack([b["data"] for b in batch]), modality=batch[0]["modality"])
    for k in ("c2w", "K"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    if "label" in batch[0]:
        labs = [b["label"] for b in batch]
        out["label"] = torch.tensor(labs) if all(isinstance(l, int) for l in labs) else labs
    return out


def _worker_init(wid: int) -> None:
    import signal
    # SIGTERM/SIGINT go to the whole process group (torchrun, tmux Ctrl-C); workers ignore them so the
    # trainer can finish its step and save. They still exit when the main process goes away.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    info = torch.utils.data.get_worker_info()
    seed = info.seed % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    ds = info.dataset
    if hasattr(ds, "_rng"):
        ds._rng = random.Random(seed)


def mixed_loaders(imagenet: Optional[str] = None, openvid: Optional[str] = None,
                  objaverse: Optional[str] = None, *, batch_image: int = 32, batch_video: int = 4,
                  batch_3d: int = 4, size: int = 256, n_frames: int = 8, frame_stride: int = 2,
                  n_views: int = 4, video_tail: int = 1000, objaverse_holdout: int = 500,
                  num_workers: int = 8, rank: int = 0, world: int = 1, seed: int = 0,
                  split: str = "train") -> dict:
    """{'image' | 'video' | 'threed': (loader, sampler)} for whichever sources are given.
    Image batches use ParquetGroupBatchSampler (row-group aware, rank-sharded); video and threed use
    DistributedSampler. With split='train' the OpenVid manifest tail and the Objaverse hold-out uids
    are excluded."""
    train = split == "train"
    out = {}
    common = dict(num_workers=num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available(),
                  worker_init_fn=_worker_init if num_workers > 0 else None,
                  persistent_workers=num_workers > 0)
    if num_workers > 0:
        common["prefetch_factor"] = 4
    if imagenet:
        ds = ImageNetParquet(imagenet, split, size=size, train=train)
        bs = ParquetGroupBatchSampler(ds, batch_image, num_workers, rank, world, seed, shuffle=train)
        out["image"] = (DataLoader(ds, batch_sampler=bs, **common), bs)
    if openvid:
        ds = OpenVidClips(openvid, n_frames=n_frames, frame_stride=frame_stride, size=size, train=train,
                          tail=video_tail)
        smp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=train, seed=seed, drop_last=True)
        out["video"] = (DataLoader(ds, batch_size=batch_video, sampler=smp, drop_last=True, **common), smp)
    if objaverse:
        ds = ObjaverseViews(objaverse, n_views=n_views, size=size, train=train, holdout=objaverse_holdout)
        smp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=train, seed=seed, drop_last=True)
        out["threed"] = (DataLoader(ds, batch_size=batch_3d, sampler=smp, drop_last=True, **common), smp)
    return out


def infinite(loader: DataLoader, sampler) -> Iterator[dict]:
    """Endless iterator over a loader, bumping the sampler epoch each pass."""
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        n = 0
        for b in loader:
            n += 1
            yield b
        if n == 0:
            raise RuntimeError("data loader produced no batches (dataset smaller than one batch per rank?)")
        epoch += 1
