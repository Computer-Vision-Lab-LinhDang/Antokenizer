"""DataLoader implementations for diffusion generator training.

Supports multiple data formats:
1. Single folder with clean images (degradation applied on-the-fly)
2. Paired folders (input/groundtruth)
3. CSV/JSON annotation files
4. Video datasets (frame sequences)
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from .degradation import DegradationConfig, DegradationPipeline


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Paths
    data_dir: str = ""
    groundtruth_dir: Optional[str] = None  # If None, use same as data_dir

    # Image settings
    image_size: Tuple[int, int] = (256, 256)
    in_channels: int = 3

    # Data format
    data_format: Literal["single", "paired", "csv", "json"] = "single"
    annotation_file: Optional[str] = None

    # Augmentation
    use_augmentation: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = False
    random_crop: bool = True
    crop_scale: Tuple[float, float] = (0.8, 1.0)
    color_jitter: bool = False

    # Degradation
    apply_degradation: bool = True
    degradation_config: Optional[DegradationConfig] = None

    # Video settings
    is_video: bool = False
    num_frames: int = 16
    frame_stride: int = 1

    # Extensions
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".webm")


# ============================================================================
# Transform Utilities
# ============================================================================

class PairedTransform:
    """Apply same random transform to both input and groundtruth."""

    def __init__(
        self,
        size: Tuple[int, int],
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        random_crop: bool = True,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
    ) -> None:
        self.size = size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_crop = random_crop
        self.crop_scale = crop_scale

    def __call__(
        self,
        img: torch.Tensor,
        gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply paired transforms.

        Args:
            img: Input image tensor (C, H, W).
            gt: Groundtruth tensor (C, H, W).

        Returns:
            Transformed (img, gt) pair.
        """
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            img = torch.flip(img, dims=[-1])
            gt = torch.flip(gt, dims=[-1])

        # Random vertical flip
        if self.vertical_flip and random.random() > 0.5:
            img = torch.flip(img, dims=[-2])
            gt = torch.flip(gt, dims=[-2])

        # Random crop
        if self.random_crop:
            _, h, w = img.shape
            scale = random.uniform(*self.crop_scale)
            new_h, new_w = int(h * scale), int(w * scale)

            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)

            img = img[:, top:top + new_h, left:left + new_w]
            gt = gt[:, top:top + new_h, left:left + new_w]

        # Resize to target size
        img = F.interpolate(
            img.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
        ).squeeze(0)
        gt = F.interpolate(
            gt.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
        ).squeeze(0)

        return img, gt


class VideoTransform:
    """Transform for video data with temporal consistency."""

    def __init__(
        self,
        size: Tuple[int, int],
        num_frames: int,
        horizontal_flip: bool = True,
        random_crop: bool = True,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
    ) -> None:
        self.size = size
        self.num_frames = num_frames
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        self.crop_scale = crop_scale

    def __call__(
        self,
        video: torch.Tensor,
        gt_video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply video transforms.

        Args:
            video: Input video (C, T, H, W).
            gt_video: Groundtruth video (C, T, H, W).

        Returns:
            Transformed (video, gt_video) pair.
        """
        # Same flip for all frames
        if self.horizontal_flip and random.random() > 0.5:
            video = torch.flip(video, dims=[-1])
            gt_video = torch.flip(gt_video, dims=[-1])

        # Same crop for all frames
        if self.random_crop:
            _, _, h, w = video.shape
            scale = random.uniform(*self.crop_scale)
            new_h, new_w = int(h * scale), int(w * scale)

            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)

            video = video[:, :, top:top + new_h, left:left + new_w]
            gt_video = gt_video[:, :, top:top + new_h, left:left + new_w]

        # Resize
        c, t, h, w = video.shape
        video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
        video = F.interpolate(video, size=self.size, mode="bilinear", align_corners=False)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        gt_video = gt_video.permute(1, 0, 2, 3)
        gt_video = F.interpolate(gt_video, size=self.size, mode="bilinear", align_corners=False)
        gt_video = gt_video.permute(1, 0, 2, 3)

        return video, gt_video


# ============================================================================
# Dataset Implementations
# ============================================================================

class ImageDataset(Dataset):
    """Dataset for single folder of images with on-the-fly degradation.

    The groundtruth is the clean image, and the input is created by
    applying degradation augmentation.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DataConfig] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: Path to directory containing images.
            config: Data configuration.
        """
        if not HAS_PIL:
            raise ImportError("PIL is required for ImageDataset")

        self.data_dir = Path(data_dir)
        self.config = config or DataConfig(data_dir=data_dir)

        # Find all images
        self.images = self._find_images()
        if not self.images:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.images)} images in {data_dir}")

        # Setup transforms
        self.transform = PairedTransform(
            size=self.config.image_size,
            horizontal_flip=self.config.horizontal_flip and self.config.use_augmentation,
            vertical_flip=self.config.vertical_flip and self.config.use_augmentation,
            random_crop=self.config.random_crop and self.config.use_augmentation,
            crop_scale=self.config.crop_scale,
        )

        # Setup degradation
        if self.config.apply_degradation:
            self.degradation = DegradationPipeline(self.config.degradation_config)
        else:
            self.degradation = None

    def _find_images(self) -> List[Path]:
        """Find all image files in directory."""
        images = []
        for ext in self.config.image_extensions:
            images.extend(self.data_dir.glob(f"**/*{ext}"))
            images.extend(self.data_dir.glob(f"**/*{ext.upper()}"))
        return sorted(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.

        Returns:
            Dictionary with:
                - clean: Groundtruth clean image (C, H, W)
                - degraded: Degraded input image (C, H, W)
                - degradation_params: Degradation parameters (D,)
        """
        img_path = self.images[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)  # (C, H, W) in [0, 1]

        # Apply paired transform (same to both since GT = clean)
        clean, _ = self.transform(image, image.clone())

        # Apply degradation
        if self.degradation is not None:
            # Add batch dimension for degradation
            clean_batch = clean.unsqueeze(0)
            degraded, deg_params = self.degradation(clean_batch)
            degraded = degraded.squeeze(0)
            deg_params = deg_params.squeeze(0)
        else:
            degraded = clean.clone()
            deg_params = torch.zeros(32)

        return {
            "clean": clean,
            "degraded": degraded,
            "degradation_params": deg_params,
        }


class PairedImageDataset(Dataset):
    """Dataset for paired input/groundtruth images.

    Expects two directories:
    - input_dir: Contains degraded/input images
    - groundtruth_dir: Contains clean groundtruth images

    Files are matched by name.
    """

    def __init__(
        self,
        input_dir: str,
        groundtruth_dir: str,
        config: Optional[DataConfig] = None,
    ) -> None:
        """Initialize paired dataset.

        Args:
            input_dir: Path to input images.
            groundtruth_dir: Path to groundtruth images.
            config: Data configuration.
        """
        if not HAS_PIL:
            raise ImportError("PIL is required for PairedImageDataset")

        self.input_dir = Path(input_dir)
        self.gt_dir = Path(groundtruth_dir)
        self.config = config or DataConfig()

        # Find paired images
        self.pairs = self._find_pairs()
        if not self.pairs:
            raise ValueError(f"No image pairs found in {input_dir} and {groundtruth_dir}")

        print(f"Found {len(self.pairs)} image pairs")

        # Setup transforms
        self.transform = PairedTransform(
            size=self.config.image_size,
            horizontal_flip=self.config.horizontal_flip and self.config.use_augmentation,
            vertical_flip=self.config.vertical_flip and self.config.use_augmentation,
            random_crop=self.config.random_crop and self.config.use_augmentation,
            crop_scale=self.config.crop_scale,
        )

        # For paired data, degradation is already applied
        # But we still need to encode the degradation type
        self.degradation = DegradationPipeline(self.config.degradation_config)

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching input/groundtruth pairs."""
        pairs = []

        # Get all input images
        input_images = {}
        for ext in self.config.image_extensions:
            for img_path in self.input_dir.glob(f"**/*{ext}"):
                stem = img_path.stem
                input_images[stem] = img_path
            for img_path in self.input_dir.glob(f"**/*{ext.upper()}"):
                stem = img_path.stem
                input_images[stem] = img_path

        # Match with groundtruth
        for stem, input_path in input_images.items():
            # Try to find matching groundtruth
            for ext in self.config.image_extensions:
                gt_path = self.gt_dir / f"{stem}{ext}"
                if gt_path.exists():
                    pairs.append((input_path, gt_path))
                    break
                gt_path = self.gt_dir / f"{stem}{ext.upper()}"
                if gt_path.exists():
                    pairs.append((input_path, gt_path))
                    break

        return sorted(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        input_path, gt_path = self.pairs[idx]

        # Load images
        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        input_tensor = TF.to_tensor(input_img)
        gt_tensor = TF.to_tensor(gt_img)

        # Apply paired transforms
        degraded, clean = self.transform(input_tensor, gt_tensor)

        # Estimate degradation params (since it's pre-degraded)
        deg_params = self._estimate_degradation_params(degraded, clean)

        return {
            "clean": clean,
            "degraded": degraded,
            "degradation_params": deg_params,
        }

    def _estimate_degradation_params(
        self,
        degraded: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate degradation parameters from input/gt pair."""
        params = torch.zeros(32)

        # Estimate blur from frequency content difference
        degraded_freq = torch.fft.fft2(degraded.mean(0)).abs().mean()
        clean_freq = torch.fft.fft2(clean.mean(0)).abs().mean()
        blur_level = 1.0 - (degraded_freq / clean_freq.clamp(min=1e-6)).clamp(0, 1)
        params[0] = 1.0 if blur_level > 0.1 else 0.0
        params[1] = blur_level.item()
        params[2] = blur_level.item()

        # Estimate noise from difference variance
        diff = (degraded - clean).abs()
        noise_level = diff.std().item()
        params[3] = 1.0 if noise_level > 0.01 else 0.0
        params[4] = min(noise_level * 10, 1.0)

        return params


class AnnotatedDataset(Dataset):
    """Dataset using CSV or JSON annotation file.

    Annotation format (CSV):
        input_path,groundtruth_path,degradation_type,param1,param2,...

    Annotation format (JSON):
        [
            {
                "input": "path/to/input.jpg",
                "groundtruth": "path/to/gt.jpg",
                "degradation": {"type": "blur", "strength": 0.5}
            },
            ...
        ]
    """

    def __init__(
        self,
        annotation_file: str,
        data_root: str = "",
        config: Optional[DataConfig] = None,
    ) -> None:
        """Initialize annotated dataset.

        Args:
            annotation_file: Path to CSV or JSON annotation file.
            data_root: Root directory for relative paths.
            config: Data configuration.
        """
        self.annotation_file = Path(annotation_file)
        self.data_root = Path(data_root) if data_root else self.annotation_file.parent
        self.config = config or DataConfig()

        # Load annotations
        self.annotations = self._load_annotations()
        print(f"Loaded {len(self.annotations)} annotations")

        # Setup transforms
        self.transform = PairedTransform(
            size=self.config.image_size,
            horizontal_flip=self.config.horizontal_flip and self.config.use_augmentation,
            vertical_flip=self.config.vertical_flip and self.config.use_augmentation,
            random_crop=self.config.random_crop and self.config.use_augmentation,
            crop_scale=self.config.crop_scale,
        )

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load annotations from file."""
        suffix = self.annotation_file.suffix.lower()

        if suffix == ".json":
            with open(self.annotation_file, "r") as f:
                return json.load(f)
        elif suffix == ".csv":
            import csv
            annotations = []
            with open(self.annotation_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    annotations.append(dict(row))
            return annotations
        else:
            raise ValueError(f"Unsupported annotation format: {suffix}")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        ann = self.annotations[idx]

        # Get paths
        input_path = self.data_root / ann.get("input", ann.get("input_path", ""))
        gt_path = self.data_root / ann.get("groundtruth", ann.get("groundtruth_path", ann.get("gt", "")))

        # Load images
        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        input_tensor = TF.to_tensor(input_img)
        gt_tensor = TF.to_tensor(gt_img)

        # Apply transforms
        degraded, clean = self.transform(input_tensor, gt_tensor)

        # Get degradation params from annotation
        deg_params = self._parse_degradation_params(ann)

        return {
            "clean": clean,
            "degraded": degraded,
            "degradation_params": deg_params,
        }

    def _parse_degradation_params(self, ann: Dict[str, Any]) -> torch.Tensor:
        """Parse degradation parameters from annotation."""
        params = torch.zeros(32)

        deg_info = ann.get("degradation", {})
        if isinstance(deg_info, str):
            deg_type = deg_info
            strength = 0.5
        else:
            deg_type = deg_info.get("type", "unknown")
            strength = deg_info.get("strength", 0.5)

        if deg_type == "blur":
            params[0] = 1.0
            params[1] = strength
            params[2] = strength
        elif deg_type == "noise":
            params[3] = 1.0
            params[4] = strength
        elif deg_type == "jpeg":
            params[5] = 1.0
            params[6] = strength
        elif deg_type == "sr" or deg_type == "resize":
            params[7] = 1.0
            params[8] = strength

        return params


class VideoDataset(Dataset):
    """Dataset for video data.

    Loads video frames as sequences for temporal modeling.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[DataConfig] = None,
    ) -> None:
        """Initialize video dataset.

        Args:
            data_dir: Path to video directory (subfolders are video sequences).
            config: Data configuration.
        """
        self.data_dir = Path(data_dir)
        self.config = config or DataConfig(is_video=True)

        # Find video sequences (each subfolder is a sequence)
        self.sequences = self._find_sequences()
        print(f"Found {len(self.sequences)} video sequences")

        # Setup transforms
        self.transform = VideoTransform(
            size=self.config.image_size,
            num_frames=self.config.num_frames,
            horizontal_flip=self.config.horizontal_flip and self.config.use_augmentation,
            random_crop=self.config.random_crop and self.config.use_augmentation,
            crop_scale=self.config.crop_scale,
        )

        # Setup degradation
        if self.config.apply_degradation:
            self.degradation = DegradationPipeline(self.config.degradation_config)
        else:
            self.degradation = None

    def _find_sequences(self) -> List[List[Path]]:
        """Find video frame sequences."""
        sequences = []

        # Each subfolder is a video sequence
        for seq_dir in sorted(self.data_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            # Find frames in sequence
            frames = []
            for ext in self.config.image_extensions:
                frames.extend(seq_dir.glob(f"*{ext}"))
                frames.extend(seq_dir.glob(f"*{ext.upper()}"))

            frames = sorted(frames)

            if len(frames) >= self.config.num_frames:
                sequences.append(frames)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a video sample."""
        frames = self.sequences[idx]

        # Sample frames
        total_frames = len(frames)
        num_frames = self.config.num_frames
        stride = self.config.frame_stride

        # Random start position
        max_start = total_frames - num_frames * stride
        start = random.randint(0, max(0, max_start))

        # Load frames
        frame_tensors = []
        for i in range(num_frames):
            frame_idx = min(start + i * stride, total_frames - 1)
            frame = Image.open(frames[frame_idx]).convert("RGB")
            frame_tensors.append(TF.to_tensor(frame))

        # Stack to video tensor: (C, T, H, W)
        video = torch.stack(frame_tensors, dim=1)

        # Apply video transform
        clean, _ = self.transform(video, video.clone())

        # Apply degradation
        if self.degradation is not None:
            clean_batch = clean.unsqueeze(0)
            degraded, deg_params = self.degradation(clean_batch)
            degraded = degraded.squeeze(0)
            deg_params = deg_params.squeeze(0)
        else:
            degraded = clean.clone()
            deg_params = torch.zeros(32)

        return {
            "clean": clean,
            "degraded": degraded,
            "degradation_params": deg_params,
        }


# ============================================================================
# DataLoader Factory
# ============================================================================

def create_diffusion_dataloader(
    data_dir: str,
    groundtruth_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    config: Optional[DataConfig] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Create DataLoader for diffusion generator training.

    Args:
        data_dir: Path to data directory.
        groundtruth_dir: Path to groundtruth directory (for paired data).
        batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        config: Data configuration.
        shuffle: Whether to shuffle data.
        drop_last: Whether to drop last incomplete batch.
        pin_memory: Whether to pin memory for faster GPU transfer.
        distributed: Whether to use distributed sampling.
        world_size: Total number of processes (for distributed).
        rank: Current process rank (for distributed).

    Returns:
        DataLoader instance.

    Example:
        >>> # Single folder with degradation augmentation
        >>> loader = create_diffusion_dataloader(
        ...     data_dir="/path/to/images",
        ...     batch_size=8,
        ... )

        >>> # Paired input/groundtruth folders
        >>> loader = create_diffusion_dataloader(
        ...     data_dir="/path/to/degraded",
        ...     groundtruth_dir="/path/to/clean",
        ...     batch_size=8,
        ... )

        >>> # With custom config
        >>> config = DataConfig(
        ...     image_size=(512, 512),
        ...     apply_degradation=True,
        ...     use_augmentation=True,
        ... )
        >>> loader = create_diffusion_dataloader(
        ...     data_dir="/path/to/images",
        ...     batch_size=8,
        ...     config=config,
        ... )
    """
    config = config or DataConfig(data_dir=data_dir)

    # Select dataset type
    if config.is_video:
        dataset = VideoDataset(data_dir, config=config)
    elif groundtruth_dir is not None:
        dataset = PairedImageDataset(data_dir, groundtruth_dir, config=config)
    elif config.annotation_file is not None:
        dataset = AnnotatedDataset(config.annotation_file, data_dir, config=config)
    else:
        dataset = ImageDataset(data_dir, config=config)

    # Create sampler for distributed training
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_diffusion_batch,
    )

    return loader


def collate_diffusion_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function for diffusion training batches.

    Handles both image and video data.
    """
    clean = torch.stack([item["clean"] for item in batch])
    degraded = torch.stack([item["degraded"] for item in batch])
    deg_params = torch.stack([item["degradation_params"] for item in batch])

    return {
        "clean": clean,
        "degraded": degraded,
        "degradation_params": deg_params,
    }


# ============================================================================
# Utility Functions
# ============================================================================

def get_dataset_stats(dataset: Dataset, num_samples: int = 100) -> Dict[str, float]:
    """Compute dataset statistics for normalization.

    Args:
        dataset: Dataset to analyze.
        num_samples: Number of samples to use for estimation.

    Returns:
        Dictionary with mean and std for each channel.
    """
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    samples = []
    for idx in indices:
        sample = dataset[idx]
        samples.append(sample["clean"])

    stacked = torch.stack(samples)

    # Compute per-channel statistics
    if stacked.dim() == 4:  # (N, C, H, W)
        mean = stacked.mean(dim=(0, 2, 3))
        std = stacked.std(dim=(0, 2, 3))
    else:  # (N, C, T, H, W)
        mean = stacked.mean(dim=(0, 2, 3, 4))
        std = stacked.std(dim=(0, 2, 3, 4))

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


__all__ = [
    "DataConfig",
    "ImageDataset",
    "PairedImageDataset",
    "AnnotatedDataset",
    "VideoDataset",
    "create_diffusion_dataloader",
    "collate_diffusion_batch",
    "get_dataset_stats",
    "PairedTransform",
    "VideoTransform",
]
