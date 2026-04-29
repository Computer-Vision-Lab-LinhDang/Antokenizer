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


def _read_video_pyav(path: str, n_frames: int) -> torch.Tensor:
    """Decode a video into ``(T, 3, H, W)`` uint8 frames using PyAV.

    Used in place of the removed ``torchvision.io.read_video`` (gone in
    torchvision >= 0.24). Loads at most ``n_frames * 4`` raw frames to bound
    memory; the caller does the final temporal subsample.
    """
    import av
    import numpy as np

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        frames: List[np.ndarray] = []
        # Cap raw decode budget: a 16-frame clip rarely benefits from > 64 raw frames.
        cap = max(n_frames * 4, n_frames + 8)
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) >= cap:
                break
    finally:
        container.close()

    if not frames:
        raise ValueError(f'no frames decoded from {path}')
    arr = np.stack(frames)                         # (T, H, W, 3) uint8
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)


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
        path = self.paths[idx]
        obj_id = path.stem
        try:
            vframes = _read_video_pyav(str(path), self.n_frames)
            T = vframes.shape[0]
            if T < self.n_frames:
                reps = (self.n_frames // T) + 1
                vframes = vframes.repeat(reps, 1, 1, 1)
            step = max(1, vframes.shape[0] // self.n_frames)
            idxs = torch.arange(0, vframes.shape[0], step)[: self.n_frames]
            clip = vframes[idxs].float() / 255.0          # (T, 3, H, W) in [0,1]
            clip = torch.stack([self.frame_transform(clip[t]) for t in range(len(clip))])
            data = clip.permute(1, 0, 2, 3)               # (3, T, H, W)
        except Exception as exc:  # noqa: BLE001
            print(f"[UniversalVideoDataset] decode failed for {path}: "
                  f"{type(exc).__name__}: {exc} — returning zeros")
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

    def __init__(self, root: str, resolution: int = 256,
                 renders_dir: Optional[str] = None):
        root_path = Path(root) if root else None
        captions_path = (root_path / 'captions' / '3d.json') if root_path else None
        self.captions: Dict[str, str] = (
            json.loads(captions_path.read_text())
            if captions_path is not None and captions_path.exists() else {}
        )
        renders_dir = (
            Path(renders_dir) if renders_dir
            else (root_path / '3d_objects' / 'renders' if root_path else None)
        )
        if renders_dir is not None and renders_dir.exists():
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

# Per-worker shard cache: {path: (image_bytes_list, labels_list)}
_hf_shard_cache: Dict[str, Tuple[List[bytes], List[str]]] = {}


class HFParquetImageDataset(Dataset):
    """Load images from HuggingFace-format parquet shards.

    Expects files named ``{split}-NNNNN-of-MMMMM.parquet`` with columns
    ``image`` (struct with 'bytes' key) and ``label`` (int).
    """

    def __init__(self, shards_dir: str, resolution: int = 256,
                 split: str = 'train', cache_shards: int = 4):
        import pyarrow.parquet as pq

        self.resolution = resolution
        self.cache_shards = cache_shards
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        # Build (shard_path, row_idx) index using only metadata — no data loaded.
        self._index: List[Tuple[str, int]] = []
        pq_files = sorted(Path(shards_dir).glob(f"{split}-*.parquet"))
        for pq_file in pq_files:
            if pq_file.stat().st_size == 0:
                continue
            meta = pq.read_metadata(str(pq_file))
            n = meta.num_rows
            path_str = str(pq_file)
            for i in range(n):
                self._index.append((path_str, i))

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, path: str) -> Tuple[List[bytes], List[str]]:
        global _hf_shard_cache
        if path not in _hf_shard_cache:
            import pyarrow.parquet as pq
            table = pq.read_table(path, columns=['image', 'label'])
            images = [r.as_py()['bytes'] for r in table['image']]
            labels = [str(v.as_py()) for v in table['label']]
            if len(_hf_shard_cache) >= self.cache_shards:
                _hf_shard_cache.pop(next(iter(_hf_shard_cache)))
            _hf_shard_cache[path] = (images, labels)
        return _hf_shard_cache[path]

    def __getitem__(self, idx: int) -> Dict:
        from PIL import Image as PILImage
        import io

        path, row_idx = self._index[idx]
        images, labels = self._load_shard(path)

        try:
            img = PILImage.open(io.BytesIO(images[row_idx])).convert('RGB')
            data = self.transform(img)
        except Exception:
            data = torch.zeros(3, self.resolution, self.resolution)

        return {
            'data': data,
            'modality': 'image',
            'caption': labels[row_idx],
            'id': f"{Path(path).stem}_{row_idx}",
        }


class WDSImageDataset(Dataset):
    """Read images directly from WebDataset .tar shards.

    Reads tar headers sequentially and builds an in-memory index on first
    access. The index (shard_path, member, label) is cached to disk per shard
    so repeated runs skip the expensive header walk — at 609 shards × ~108MB
    a cold scan takes ~10 minutes, while a warm load is sub-second.
    """

    def __init__(self, shards_dir: str, resolution: int = 256,
                 max_shards: Optional[int] = None,
                 cache_dir: Optional[str] = None):
        import tarfile
        import pickle
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

        shard_paths = sorted(Path(shards_dir).glob("*.tar"))
        if max_shards:
            shard_paths = shard_paths[:max_shards]

        cache_root = Path(cache_dir) if cache_dir else (Path(shards_dir) / '.wds_index_cache')
        cache_root.mkdir(parents=True, exist_ok=True)

        self.index: List[Tuple[str, str, str]] = []
        for sp in shard_paths:
            cache_file = cache_root / f"{sp.stem}.pkl"
            shard_index: Optional[List[Tuple[str, str, str]]] = None
            if cache_file.exists() and cache_file.stat().st_mtime >= sp.stat().st_mtime:
                try:
                    with open(cache_file, 'rb') as fh:
                        shard_index = pickle.load(fh)
                except Exception:  # noqa: BLE001
                    shard_index = None
            if shard_index is None:
                shard_index = []
                try:
                    with tarfile.open(sp, "r") as tar:
                        members = {m.name: m for m in tar.getmembers()}
                        keys = set()
                        for name in members:
                            key = name.rsplit(".", 1)[0]
                            keys.add(key)
                        for key in sorted(keys):
                            jpg_name = f"{key}.jpg"
                            if jpg_name not in members:
                                continue
                            label = ""
                            txt_name = f"{key}.txt"
                            if txt_name in members:
                                f = tar.extractfile(members[txt_name])
                                if f:
                                    label = f.read().decode("utf-8", errors="replace").strip()
                            shard_index.append((str(sp), jpg_name, label))
                except Exception:  # noqa: BLE001
                    continue
                try:
                    with open(cache_file, 'wb') as fh:
                        pickle.dump(shard_index, fh, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:  # noqa: BLE001
                    pass
            self.index.extend(shard_index)

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
                if pq_file.exists() and pq_file.stat().st_size > 0:
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
            vframes = _read_video_pyav(str(mp4_path), self.n_frames)
            T = vframes.shape[0]
            if T < self.n_frames:
                reps = (self.n_frames // T) + 1
                vframes = vframes.repeat(reps, 1, 1, 1)
                T = vframes.shape[0]
            step = max(1, T // self.n_frames)
            idxs = torch.arange(0, T, step)[:self.n_frames]
            clip = vframes[idxs].float() / 255.0
            clip = torch.stack([self.frame_transform(clip[t]) for t in range(len(clip))])
            data = clip.permute(1, 0, 2, 3)  # (3, T, H, W)
        except Exception as exc:
            print(f"[ShardVideoDataset] decode failed for {mp4_path}: "
                  f"{type(exc).__name__}: {exc} — returning zeros")
            data = torch.zeros(3, self.n_frames, self.resolution, self.resolution)

        return {
            'data': data,
            'modality': 'video',
            'caption': caption,
            'id': obj_id,
        }
