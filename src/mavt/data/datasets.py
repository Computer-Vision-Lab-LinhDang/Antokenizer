"""Datasets for MAVT training.

SyntheticMultiModalDataset — smoke-test stub (no I/O).

Universal datasets (universal data root layout):
  UniversalImageDataset   — images/ + captions/images.json
  UniversalVideoDataset   — videos/ + captions/videos.json
  UniversalThreeDDataset  — 3d_objects/renders/<id>/{oxoy,oxoz,oyoz}.png + captions/3d.json
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
        self.paths: List[Path] = sorted(
            p for p in (root / 'images').iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        )
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
        self.paths: List[Path] = sorted(
            p for p in (root / 'videos').iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        )
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

    Each object folder contains oxoy.png, oxoz.png, oyoz.png (RGB, 256×256).
    They are stacked into a (3, 3, resolution, resolution) tensor:
      dim-0 → plane index (oxoy=0, oxoz=1, oyoz=2)
      dim-1 → RGB channel

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
        self.obj_dirs: List[Path] = sorted(
            d for d in renders_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
            and all((d / f'{p}.png').exists() for p in PLANE_NAMES)
        )
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
