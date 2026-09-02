"""Datasets for MAVT training.

SyntheticMultiModalDataset — smoke-test stub (no I/O).

Universal datasets (universal data root layout):
  UniversalImageDataset   — images/ + captions/images.json
  UniversalVideoDataset   — videos/ + captions/videos.json
  UniversalThreeDDataset  — 3d_objects/renders/<id>/{oxoy,oxoz,oyoz}.png + captions/3d.json

WebDataset-backed datasets (for large-scale training):
  WDSImageDataset         — reads directly from WDS .tar shards
  ShardVideoDataset       — reads from video2dataset shard dirs (NNNNN/*.mp4)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# --------------------------------------------------------------------------- #
#  Synthetic                                                                    #
# --------------------------------------------------------------------------- #

class SyntheticMultiModalDataset(Dataset):
    """Synthetic data for smoke-testing a single modality."""

    def __init__(self, n: int, modality: str, resolution: int = 128,
                 n_frames: int = 8, triplane_res: int = 64):
        self.n = n
        self.modality = modality
        self.resolution = resolution
        self.n_frames = n_frames
        self.triplane_res = triplane_res

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict:
        if self.modality == 'image':
            data = torch.randn(3, self.resolution, self.resolution)
        elif self.modality == 'video':
            data = torch.randn(3, self.n_frames, self.resolution, self.resolution)
        else:  # threed
            data = torch.randn(3, 3, self.triplane_res, self.triplane_res)
        return {'data': data, 'modality': self.modality}


# --------------------------------------------------------------------------- #
#  Universal datasets  (data/sample100-1 layout)                               #
# --------------------------------------------------------------------------- #

PLANE_NAMES = ('oxoy', 'oxoz', 'oyoz')  # front (XY), top (XZ), side (YZ)


class UniversalImageDataset(Dataset):
    """Images from data/sample100-1/images/ with captions.

    Returns:
        data     : (3, resolution, resolution) float tensor in [-1, 1]
        modality : 'image'
        caption  : str
        id       : filename stem
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, root: str, resolution: int = 256):
        root = Path(root)
        captions_path = root / 'captions' / 'images.json'
        self.captions: Dict[str, str] = (
            json.loads(captions_path.read_text()) if captions_path.exists() else {}
        )
        img_dir = root / 'images'
        if img_dir.exists():
            self.paths: List[Path] = sorted(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in self.EXTENSIONS
            )
        else:
            self.paths = []
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        from PIL import Image as PILImage
        path = self.paths[idx]
        obj_id = path.stem
        img = PILImage.open(path).convert('RGB')
        return {
            'data': self.transform(img),
            'modality': 'image',
            'caption': self.captions.get(obj_id, ''),
            'id': obj_id,
        }


class UniversalVideoDataset(Dataset):
    """Videos from data/sample100-1/videos/ with captions.

    Returns:
        data     : (3, n_frames, resolution, resolution) float tensor in [-1, 1]
        modality : 'video'
        caption  : str
        id       : filename stem
    """

    EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

    def __init__(self, root: str, n_frames: int = 16, resolution: int = 256):
        root = Path(root)
        captions_path = root / 'captions' / 'videos.json'
        self.captions: Dict[str, str] = (
            json.loads(captions_path.read_text()) if captions_path.exists() else {}
        )
        vid_dir = root / 'videos'
        if vid_dir.exists():
            self.paths: List[Path] = sorted(
                p for p in vid_dir.iterdir()
                if p.suffix.lower() in self.EXTENSIONS
            )
        else:
            self.paths = []
        self.n_frames = n_frames
        self.resolution = resolution
        self.frame_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        import torchvision.io as tvio
        path = self.paths[idx]
        obj_id = path.stem
        try:
            vframes, _, _ = tvio.read_video(str(path), pts_unit='sec', output_format='TCHW')
            T = vframes.shape[0]
            if T == 0:
                raise ValueError('empty video')
            if T < self.n_frames:
                reps = (self.n_frames // T) + 1
                vframes = vframes.repeat(reps, 1, 1, 1)
            step = max(1, T // self.n_frames)
            idxs = torch.arange(0, vframes.shape[0], step)[: self.n_frames]
            clip = vframes[idxs].float() / 255.0          # (T, 3, H, W) in [0,1]
            clip = torch.stack([self.frame_transform(clip[t]) for t in range(len(clip))])
            data = clip.permute(1, 0, 2, 3)               # (3, T, H, W)
        except Exception:  # noqa: BLE001
            data = torch.zeros(3, self.n_frames, self.resolution, self.resolution)
        return {
            'data': data,
            'modality': 'video',
            'caption': self.captions.get(obj_id, ''),
            'id': obj_id,
        }


class UniversalThreeDDataset(Dataset):
    """Triplane-PNG 3-D dataset from data/sample100-1/3d_objects/renders/.

    Each object folder contains oxoy.png, oxoz.png, oyoz.png (RGB, 256x256).
    They are stacked into a (3, 3, resolution, resolution) tensor:
      dim-0 -> plane index (oxoy=0, oxoz=1, oyoz=2)
      dim-1 -> RGB channel

    Returns:
        data     : (3, 3, resolution, resolution) float tensor in [-1, 1]
        modality : 'threed'
        caption  : str
        id       : object folder name
    """

    def __init__(self, root: str, resolution: int = 256):
        root = Path(root)
        captions_path = root / 'captions' / '3d.json'
        self.captions: Dict[str, str] = (
            json.loads(captions_path.read_text()) if captions_path.exists() else {}
        )
        renders_dir = root / '3d_objects' / 'renders'
        if renders_dir.exists():
            self.obj_dirs: List[Path] = sorted(
                d for d in renders_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
                and all((d / f'{p}.png').exists() for p in PLANE_NAMES)
            )
        else:
            self.obj_dirs = []
        self.plane_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),                         # [0,1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),   # [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.obj_dirs)

    def __getitem__(self, idx: int) -> Dict:
        from PIL import Image as PILImage
        obj_dir = self.obj_dirs[idx]
        obj_id = obj_dir.name
        planes = []
        for plane in PLANE_NAMES:
            img = PILImage.open(obj_dir / f'{plane}.png').convert('RGB')
            planes.append(self.plane_transform(img))       # (3, H, W)
        data = torch.stack(planes)                         # (3, 3, H, W)
        return {
            'data': data,
            'modality': 'threed',
            'caption': self.captions.get(obj_id, ''),
            'id': obj_id,
        }


# --------------------------------------------------------------------------- #
#  Shard-based datasets (direct reading without extraction)                    #
# --------------------------------------------------------------------------- #

class WDSImageDataset(Dataset):
    """Read images directly from WebDataset .tar shards.

    More efficient than extracting files — reads tar sequentially and
    builds an in-memory index on first access.
    """

    def __init__(self, shards_dir: str, resolution: int = 256):
        import tarfile
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        # Build index: list of (shard_path, member_name, label)
        self.index: List[Tuple[str, str, str]] = []
        shard_paths = sorted(Path(shards_dir).glob("*.tar"))

        for sp in shard_paths:
            try:
                with tarfile.open(sp, "r") as tar:
                    members = {m.name: m for m in tar.getmembers()}
                    keys = set()
                    for name in members:
                        key = name.rsplit(".", 1)[0]
                        keys.add(key)
                    for key in sorted(keys):
                        jpg_name = f"{key}.jpg"
                        if jpg_name in members:
                            # Try to read label
                            label = ""
                            txt_name = f"{key}.txt"
                            if txt_name in members:
                                f = tar.extractfile(members[txt_name])
                                if f:
                                    label = f.read().decode("utf-8", errors="replace").strip()
                            self.index.append((str(sp), jpg_name, label))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        import tarfile
        from PIL import Image as PILImage
        shard_path, member_name, label = self.index[idx]
        key = member_name.rsplit(".", 1)[0]

        try:
            with tarfile.open(shard_path, "r") as tar:
                f = tar.extractfile(member_name)
                if f is None:
                    raise ValueError(f"Cannot extract {member_name}")
                import io
                img = PILImage.open(io.BytesIO(f.read())).convert("RGB")
        except Exception:
            # Fallback: black image
            img_tensor = torch.zeros(3, self.resolution, self.resolution)
            return {'data': img_tensor, 'modality': 'image', 'caption': label, 'id': key}

        return {
            'data': self.transform(img),
            'modality': 'image',
            'caption': label,
            'id': key,
        }


class ShardVideoDataset(Dataset):
    """Read videos from video2dataset shard directories.

    Expected layout: shards_dir/NNNNN/{NNNNNNNN.mp4, NNNNNNNN.txt, ...}
    Also supports parquet metadata: shards_dir/NNNNN.parquet

    Uses parquet metadata for fast indexing when available (seconds vs minutes).
    Captions are loaded lazily at __getitem__ time to avoid reading 600K+ files.
    """

    def __init__(self, shards_dir: str, n_frames: int = 16,
                 resolution: int = 256, max_shards: int = None):
        self.n_frames = n_frames
        self.resolution = resolution
        self.shards_dir = Path(shards_dir)
        self.frame_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        # Build index: list of (shard_id, key) tuples — fast, no per-file I/O
        self.samples: List[Tuple[str, str]] = []  # (shard_id, file_key)

        shard_dirs = sorted(
            d for d in self.shards_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if max_shards:
            shard_dirs = shard_dirs[:max_shards]

        # Try parquet-based indexing first (much faster)
        used_parquet = False
        try:
            import pyarrow.parquet as pq
            for shard_dir in shard_dirs:
                pq_file = self.shards_dir / f"{shard_dir.name}.parquet"
                if pq_file.exists():
                    table = pq.read_table(pq_file, columns=["key"])
                    keys = table["key"].to_pylist()
                    for k in keys:
                        self.samples.append((shard_dir.name, k))
                    used_parquet = True
                else:
                    # Fallback for this shard: count mp4 files
                    for mp4 in sorted(shard_dir.glob("*.mp4")):
                        self.samples.append((shard_dir.name, mp4.stem))
        except ImportError:
            pass

        if not used_parquet:
            # Full directory scan fallback (slow but works)
            self.samples = []
            for shard_dir in shard_dirs:
                for mp4 in sorted(shard_dir.glob("*.mp4")):
                    self.samples.append((shard_dir.name, mp4.stem))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        import torchvision.io as tvio
        shard_id, file_key = self.samples[idx]
        shard_dir = self.shards_dir / shard_id
        mp4_path = shard_dir / f"{file_key}.mp4"
        obj_id = f"{shard_id}_{file_key}"

        # Lazy caption loading
        caption = ""
        txt_path = shard_dir / f"{file_key}.txt"
        if txt_path.exists():
            try:
                caption = txt_path.read_text(errors="replace").strip()
            except Exception:
                pass

        try:
            vframes, _, _ = tvio.read_video(
                str(mp4_path), pts_unit='sec', output_format='TCHW'
            )
            T = vframes.shape[0]
            if T == 0:
                raise ValueError('empty video')
            if T < self.n_frames:
                reps = (self.n_frames // T) + 1
                vframes = vframes.repeat(reps, 1, 1, 1)
                T = vframes.shape[0]
            step = max(1, T // self.n_frames)
            idxs = torch.arange(0, T, step)[:self.n_frames]
            clip = vframes[idxs].float() / 255.0
            clip = torch.stack([self.frame_transform(clip[t]) for t in range(len(clip))])
            data = clip.permute(1, 0, 2, 3)  # (3, T, H, W)
        except Exception:
            data = torch.zeros(3, self.n_frames, self.resolution, self.resolution)

        return {
            'data': data,
            'modality': 'video',
            'caption': caption,
            'id': obj_id,
        }


# --------------------------------------------------------------------------- #
#  Manifest-backed datasets (v3)                                               #
# --------------------------------------------------------------------------- #

def _load_manifest(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _record_caption(rec: Dict) -> str:
    if rec.get("caption"):
        return str(rec["caption"])
    cp = rec.get("caption_path")
    if cp:
        try:
            with open(cp) as f:
                return f.read().strip()
        except OSError:
            return ""
    return ""


class ManifestImageDataset(Dataset):
    """Images listed in a .jsonl manifest ({"path", "caption"?, "caption_path"?}).

    No filesystem scan at init; fail-loud: a decode failure retries a substitute
    sample up to 3 times, counts errors, then raises. Never yields zero tensors.
    """

    _MAX_RETRIES = 3

    def __init__(self, manifest_path: str, resolution: int = 256):
        self.records = _load_manifest(manifest_path)
        if not self.records:
            raise FileNotFoundError(f"Empty manifest: {manifest_path}")
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.n_errors = 0

    def __len__(self) -> int:
        return len(self.records)

    def _load_one(self, idx: int) -> Dict:
        from PIL import Image as PILImage
        rec = self.records[idx]
        img = PILImage.open(rec["path"]).convert("RGB")
        return {"data": self.transform(img), "modality": "image",
                "caption": _record_caption(rec), "id": Path(rec["path"]).stem}

    def __getitem__(self, idx: int) -> Dict:
        import random
        for _ in range(self._MAX_RETRIES):
            try:
                return self._load_one(idx)
            except Exception as exc:  # noqa: BLE001
                self.n_errors += 1
                if self.n_errors <= 3 or self.n_errors % 100 == 0:
                    print(f"[ManifestImageDataset] error #{self.n_errors}: {exc!r}")
                idx = random.randrange(len(self.records))
        raise RuntimeError(f"Image load failed {self._MAX_RETRIES}x (total errors={self.n_errors})")


class ManifestVideoDataset(Dataset):
    """Video clips from a .jsonl manifest ({"path", "caption"?}).

    v2 (2026-09): streaming PyAV decode — frames are resized *inside the decoder*
    and only the sampled ones are kept, so a 720p/500-frame clip costs ~0.1 s and a
    few MB instead of read_video's full-clip tensor (2.7 s, 1.2 GB). Frames are a
    contiguous window at ``frame_stride`` (default 2 → 16 frames ≈ 1.3 s @ 24 fps),
    never spread over the whole clip: the temporal patchify (t_patch=2) needs
    neighbouring frames to be neighbours in time. Spatial: short side → ``resolution``
    then centre crop (same geometry as ManifestImageDataset; no anisotropic squash).
    Fail-loud: a broken file raises, it never becomes a static repeated frame.
    """

    _MAX_RETRIES = 3

    def __init__(self, manifest_path: str, n_frames: int = 16, resolution: int = 256,
                 frame_stride: int = 2):
        self.records = _load_manifest(manifest_path)
        if not self.records:
            raise FileNotFoundError(f"Empty manifest: {manifest_path}")
        if n_frames < 1 or frame_stride < 1:
            raise ValueError("n_frames and frame_stride must be >= 1")
        self.n_frames = n_frames
        self.resolution = resolution
        self.frame_stride = frame_stride
        self.n_errors = 0

    def __len__(self) -> int:
        return len(self.records)

    # -- sampling ---------------------------------------------------------------
    def _sample_indices(self, n_total: int, rng: "torch.Generator") -> List[int]:
        """Contiguous window of n_frames at frame_stride; halve the stride until it
        fits; a clip shorter than n_frames returns every frame (padded by caller)."""
        T = self.n_frames
        if n_total <= T:
            return list(range(n_total))
        stride = self.frame_stride
        while stride > 1 and (T - 1) * stride + 1 > n_total:
            stride //= 2
        span = (T - 1) * stride + 1
        start = int(torch.randint(0, n_total - span + 1, (1,), generator=rng))
        return [start + i * stride for i in range(T)]

    # -- decoding ---------------------------------------------------------------
    def _target_size(self, w: int, h: int) -> Tuple[int, int]:
        r = self.resolution
        if w >= h:
            return max(r, round(w * r / h)), r
        return r, max(r, round(h * r / w))

    def _decode(self, path: str, rng: "torch.Generator") -> "torch.Tensor":
        import av
        import numpy as np
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            n_total = int(stream.frames or 0)
            frames: List[np.ndarray] = []
            if n_total > 0:
                idxs = self._sample_indices(n_total, rng)
                want, last = set(idxs), idxs[-1]
                tw = th = None
                for i, fr in enumerate(container.decode(stream)):
                    if i > last:
                        break
                    if i in want:
                        if tw is None:
                            tw, th = self._target_size(fr.width, fr.height)
                        frames.append(fr.to_ndarray(format="rgb24", width=tw, height=th))
            else:  # container without a frame count: decode everything (already small) then sample
                tw = th = None
                for fr in container.decode(stream):
                    if tw is None:
                        tw, th = self._target_size(fr.width, fr.height)
                    frames.append(fr.to_ndarray(format="rgb24", width=tw, height=th))
                if frames:
                    frames = [frames[i] for i in self._sample_indices(len(frames), rng)]
        if not frames:
            raise ValueError("no decodable frames")
        clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0  # (t, 3, H, W)
        r = self.resolution
        H, W = clip.shape[-2:]
        top, left = (H - r) // 2, (W - r) // 2
        clip = clip[:, :, top: top + r, left: left + r]
        if clip.shape[0] < self.n_frames:  # short clip: repeat last frame
            pad = clip[-1:].expand(self.n_frames - clip.shape[0], -1, -1, -1)
            clip = torch.cat([clip, pad], 0)
        return (clip - 0.5) / 0.5

    def _load_one(self, idx: int) -> Dict:
        rec = self.records[idx]
        rng = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,))))
        clip = self._decode(rec["path"], rng)
        return {"data": clip.permute(1, 0, 2, 3).contiguous(), "modality": "video",
                "caption": _record_caption(rec), "id": Path(rec["path"]).stem}

    def __getitem__(self, idx: int) -> Dict:
        import random
        for _ in range(self._MAX_RETRIES):
            try:
                return self._load_one(idx)
            except Exception as exc:  # noqa: BLE001
                self.n_errors += 1
                if self.n_errors <= 3 or self.n_errors % 100 == 0:
                    print(f"[ManifestVideoDataset] error #{self.n_errors}: {exc!r}")
                idx = random.randrange(len(self.records))
        raise RuntimeError(f"Video load failed {self._MAX_RETRIES}x (total errors={self.n_errors})")
