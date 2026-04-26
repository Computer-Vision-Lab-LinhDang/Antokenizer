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
