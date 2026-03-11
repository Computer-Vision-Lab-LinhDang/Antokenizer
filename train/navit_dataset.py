"""Multi-resolution datasets and NaViT-style collator.

Supports:
  - MultiResImageDataset   : images at random resolutions from a set
  - VideoClipDataset       : video clips at random resolutions
  - SyntheticDataset       : random tensors for smoke testing
  - NaViTCollator          : packs a flat sample list into packed sequences
  - build_image_dataloader : factory for Stage 1
  - build_mixed_dataloader : factory for Stage 2 / Stage 3
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .packing import (
    PackedSequence,
    compute_token_count,
    compute_video_token_count,
    greedy_pack,
)

logger = logging.getLogger(__name__)

_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD  = [0.5, 0.5, 0.5]


# ─── NaViT collator ───────────────────────────────────────────────────────────

class NaViTCollator:
    """Packs a flat list of samples (from DataLoader) into packed sequences.

    DataLoader batch_size controls how many samples arrive per call.
    The collator packs them greedily into sequences ≤ max_seq_len tokens.

    Output batch:
        {
            "sequences":   List[PackedSequence],
            "n_sequences": int,
        }
    """

    def __init__(self, max_seq_len: int = 4096, shuffle: bool = True) -> None:
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle

    def __call__(self, samples: list[dict]) -> dict:
        sequences = greedy_pack(samples, self.max_seq_len, shuffle=self.shuffle)
        return {"sequences": sequences, "n_sequences": len(sequences)}


# ─── Image dataset ────────────────────────────────────────────────────────────

class MultiResImageDataset(Dataset):
    """Images loaded from disk at randomly-selected resolutions.

    Each __getitem__ picks a random H=W from `resolutions`, resizes the image,
    and returns it together with modality metadata and token count.

    Resolution must be a multiple of patch_size (default 16).
    """

    _EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def __init__(
        self,
        root: str,
        resolutions: Optional[list[int]] = None,
        patch_size: int = 16,
        task: str = "recon",
    ) -> None:
        self.paths = [
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in self._EXTS
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root!r}")
        self.resolutions = resolutions or [64, 128, 256, 512]
        self.patch_size = patch_size
        self.task = task

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        from torchvision.transforms import functional as TF

        H = W = random.choice(self.resolutions)
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = img.resize((W, H), Image.BICUBIC)
            data = TF.normalize(TF.to_tensor(img), _NORM_MEAN, _NORM_STD)
        except Exception:
            data = torch.zeros(3, H, W)

        return {
            "data":      data,
            "modality":  "image",
            "task":      self.task,
            "resolution": (H, W),
            "n_tokens":  compute_token_count(H, W, self.patch_size),
        }


# ─── Video dataset ────────────────────────────────────────────────────────────

class VideoClipDataset(Dataset):
    """Video clips loaded at random resolutions.

    Falls back to repeating a single image as multiple frames when a video
    reader is unavailable or the file is not a true video.
    """

    _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        root: str,
        resolutions: Optional[list[int]] = None,
        clip_frames: int = 8,
        patch_size: int = 16,
        tau: int = 2,
        task: str = "recon",
    ) -> None:
        all_exts = self._VIDEO_EXTS | self._IMAGE_EXTS
        self.paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in all_exts]
        if not self.paths:
            raise FileNotFoundError(f"No video/image files found under {root!r}")
        self.resolutions = resolutions or [64, 128, 256, 512]
        self.clip_frames = clip_frames
        self.patch_size = patch_size
        self.tau = tau
        self.task = task

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        H = W = random.choice(self.resolutions)
        T = self.clip_frames
        data = self._load(self.paths[idx], T, H, W)
        return {
            "data":      data,
            "modality":  "video",
            "task":      self.task,
            "resolution": (H, W, T),
            "n_tokens":  compute_video_token_count(H, W, T, self.patch_size, self.tau),
        }

    def _load(self, path: Path, T: int, H: int, W: int) -> torch.Tensor:
        if path.suffix.lower() in self._VIDEO_EXTS:
            clip = self._load_video(path, T, H, W)
            if clip is not None:
                return clip
        return self._image_as_clip(path, T, H, W)

    def _load_video(self, path: Path, T: int, H: int, W: int) -> Optional[torch.Tensor]:
        try:
            import torchvision
            reader = torchvision.io.VideoReader(str(path), "video")
            frames = []
            for f in reader:
                frames.append(f["data"])
                if len(frames) >= T:
                    break
            if not frames:
                return None
            while len(frames) < T:
                frames.append(frames[-1])
            video = torch.stack(frames[:T], dim=1).float() / 255.0  # (3, T, H', W')
            import torch.nn.functional as F
            video = F.interpolate(
                video.unsqueeze(0), size=(T, H, W),
                mode="trilinear", align_corners=False,
            ).squeeze(0)
            return (video - 0.5) / 0.5
        except Exception:
            return None

    def _image_as_clip(self, path: Path, T: int, H: int, W: int) -> torch.Tensor:
        from PIL import Image
        from torchvision.transforms import functional as TF
        try:
            img = Image.open(path).convert("RGB").resize((W, H), Image.BICUBIC)
            frame = TF.normalize(TF.to_tensor(img), _NORM_MEAN, _NORM_STD)
        except Exception:
            frame = torch.zeros(3, H, W)
        return frame.unsqueeze(1).expand(3, T, H, W).clone()  # (3, T, H, W)


# ─── Synthetic dataset ────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """Generates random tensors — for smoke testing without real data."""

    def __init__(
        self,
        n_samples: int = 500,
        resolutions: Optional[list[int]] = None,
        modality: str = "image",
        patch_size: int = 16,
        tau: int = 2,
        task: str = "recon",
    ) -> None:
        self.n = n_samples
        self.resolutions = resolutions or [64, 128, 256]
        self.modality = modality
        self.patch_size = patch_size
        self.tau = tau
        self.task = task

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        H = W = random.choice(self.resolutions)
        if self.modality == "image":
            data = torch.randn(3, H, W)
            return {
                "data": data, "modality": "image", "task": self.task,
                "resolution": (H, W),
                "n_tokens": compute_token_count(H, W, self.patch_size),
            }
        if self.modality == "video":
            T = random.choice([4, 8])
            data = torch.randn(3, T, H, W)
            return {
                "data": data, "modality": "video", "task": self.task,
                "resolution": (H, W, T),
                "n_tokens": compute_video_token_count(H, W, T, self.patch_size, self.tau),
            }
        # 3d
        S = random.choice([32, 64]) if not self.resolutions else random.choice(self.resolutions)
        data = torch.randn(3, 3, S, S)
        return {
            "data": data, "modality": "3d", "task": self.task,
            "resolution": (S, S),
            "n_tokens": 3 * compute_token_count(S, S, self.patch_size),
        }


# ─── Dataloader factories ─────────────────────────────────────────────────────

def build_image_dataloader(
    root: Optional[str],
    resolutions: list[int],
    batch_size: int,
    max_seq_len: int = 4096,
    patch_size: int = 16,
    num_workers: int = 4,
    split: float = 0.95,
    synthetic_n: int = 500,
) -> tuple[DataLoader, DataLoader]:
    """NaViT dataloader for Stage 1 (image only).

    Returns (train_loader, val_loader).
    batch_size here = number of raw samples to pack into sequences per step.
    """
    from torch.utils.data import random_split

    if root is None:
        train_ds = SyntheticDataset(synthetic_n, resolutions, "image", patch_size)
        val_ds   = SyntheticDataset(max(32, synthetic_n // 10), resolutions, "image", patch_size)
    else:
        try:
            full_ds = MultiResImageDataset(root, resolutions, patch_size)
            n_val = max(1, int(len(full_ds) * (1 - split)))
            train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
        except FileNotFoundError:
            logger.warning("No images found in %r — using synthetic data.", root)
            train_ds = SyntheticDataset(synthetic_n, resolutions, "image", patch_size)
            val_ds   = SyntheticDataset(max(32, synthetic_n // 10), resolutions, "image", patch_size)

    collator = NaViTCollator(max_seq_len=max_seq_len)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=False, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=max(4, batch_size // 4), shuffle=False,
        num_workers=max(1, num_workers // 2), collate_fn=collator,
        pin_memory=False, drop_last=False,
    )
    return train_dl, val_dl


def build_mixed_dataloader(
    roots: dict[str, Optional[str]],
    stage_cfg,  # StageConfig
    batch_size: int,
    patch_size: int = 16,
    num_workers: int = 4,
    synthetic_n: int = 300,
    split: float = 0.95,
) -> tuple[DataLoader, DataLoader]:
    """NaViT dataloader for Stage 2 / Stage 3 (mixed modalities).

    roots: {"image": path_or_None, "video": path_or_None, "3d": path_or_None}
    Samples are drawn with WeightedRandomSampler to match task_ratios.
    """
    from torch.utils.data import random_split

    datasets: list[Dataset] = []
    sample_weights: list[float] = []

    for task, ratio in stage_cfg.task_ratios.items():
        modality = ("image" if "image" in task
                    else "video" if "video" in task
                    else "3d")
        root = roots.get(modality)

        if modality == "image":
            res = stage_cfg.image_resolutions
        elif modality == "video":
            res = stage_cfg.video_resolutions or [64, 128, 256]
        else:
            res = stage_cfg.triplane_sizes or [32, 64]

        try:
            if modality == "image" and root:
                ds = MultiResImageDataset(root, res, patch_size, task=task)
            elif modality == "video" and root:
                ds = VideoClipDataset(root, res, 8, patch_size, task=task)
            else:
                raise ValueError("no root")
        except Exception:
            ds = SyntheticDataset(synthetic_n, res, modality, patch_size, task=task)

        datasets.append(ds)
        # Weight each sample so that this dataset contributes `ratio` of draws
        per_sample_weight = ratio / len(ds)
        sample_weights.extend([per_sample_weight] * len(ds))

    combined = ConcatDataset(datasets)
    total_draws = min(len(combined), 100_000)
    sampler = WeightedRandomSampler(sample_weights, num_samples=total_draws, replacement=True)

    # Build a small deterministic val split from each dataset
    val_datasets = [SyntheticDataset(max(16, synthetic_n // 10), r, m, patch_size)
                    for m, r in [("image", stage_cfg.image_resolutions[:2]),
                                 ("video", stage_cfg.video_resolutions[:2] or [64, 128])]]
    val_ds = ConcatDataset(val_datasets)

    collator = NaViTCollator(max_seq_len=stage_cfg.max_seq_len)
    train_dl = DataLoader(
        combined, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=False, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=max(4, batch_size // 4), shuffle=False,
        num_workers=max(1, num_workers // 2), collate_fn=collator,
        pin_memory=False, drop_last=False,
    )
    return train_dl, val_dl


__all__ = [
    "NaViTCollator",
    "MultiResImageDataset",
    "VideoClipDataset",
    "SyntheticDataset",
    "build_image_dataloader",
    "build_mixed_dataloader",
]
