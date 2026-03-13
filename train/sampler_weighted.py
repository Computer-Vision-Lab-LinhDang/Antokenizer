"""Modality-weighted sampler for progressive training stages.

Handles:
- Different dataset sizes (image 100M vs 3D 800K)
- Task-specific sampling ratios (recon vs understanding)
- Oversampling for small datasets
- Mixed-modality batches

Implements AToken's proven task ratios:
- Stage 1: Image 100%
- Stage 2: Image 22.2%, Video-understand 11.1%, Video-recon 66.7%
- Stage 3: Image 22.2%, Video-understand 11.1%, Video-recon 44.4%,
            3D-understand 11.1%, 3D-recon 11.1%
"""
from __future__ import annotations

import logging
import random
from typing import Dict, Optional

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ModalityWeightedSampler:
    """Sample raw data from per-modality datasets according to task weights.

    This sampler ensures that each batch contains samples in the correct
    proportions for the current training stage.
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        task_weights: Dict[str, float],
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        """Initialize sampler.

        Args:
            datasets: Dict mapping modality -> Dataset
                e.g., {"image": ImageDataset, "video": VideoDataset}
            task_weights: Dict mapping task -> weight (must sum to ~1.0)
                e.g., {"image_recon": 0.222, "video_recon": 0.667, ...}
            num_workers: Number of DataLoader workers per dataset
            prefetch_factor: Prefetch factor for DataLoaders
        """
        self.datasets = datasets
        self.task_weights = task_weights
        self.num_workers = num_workers

        # Validate weights
        total_weight = sum(task_weights.values())
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(
                f"Task weights sum to {total_weight:.3f}, not 1.0. "
                "Normalizing..."
            )
            # Normalize
            self.task_weights = {
                k: v / total_weight for k, v in task_weights.items()
            }

        # Build task -> modality mapping
        self.task_to_modality = {}
        for task in task_weights.keys():
            if "image" in task:
                self.task_to_modality[task] = "image"
            elif "video" in task:
                self.task_to_modality[task] = "video"
            elif "3d" in task:
                self.task_to_modality[task] = "3d"
            else:
                logger.warning(f"Unknown task: {task}")

        # Create iterators for each modality
        self.iterators = {}
        self.loaders = {}
        for modality, ds in datasets.items():
            loader = DataLoader(
                ds,
                batch_size=1,  # Sample one at a time
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
            self.loaders[modality] = loader
            self.iterators[modality] = iter(loader)

        # Precompute task list for faster sampling
        self.tasks = list(task_weights.keys())
        self.probs = [task_weights[t] for t in self.tasks]

        logger.info(
            f"ModalityWeightedSampler initialized with tasks: "
            f"{', '.join(f'{t}={w:.3f}' for t, w in task_weights.items())}"
        )

    def sample(self, n: int) -> list[dict]:
        """Sample n raw data items according to task weights.

        Args:
            n: Number of samples to draw

        Returns:
            List of sample dicts, each with:
                - data: raw image/video/triplane tensor
                - modality: str
                - task: str
                - caption: str
                - (other modality-specific keys)
        """
        samples = []

        # Draw tasks according to weights
        tasks = random.choices(self.tasks, weights=self.probs, k=n)

        for task in tasks:
            modality = self.task_to_modality[task]

            # Get next sample from this modality's iterator
            try:
                raw = next(self.iterators[modality])
            except StopIteration:
                # Restart iterator when exhausted
                self.iterators[modality] = iter(self.loaders[modality])
                raw = next(self.iterators[modality])

            # Extract sample from batch (batch_size=1)
            sample = {}
            for k, v in raw.items():
                if isinstance(v, (list, tuple)):
                    sample[k] = v[0]
                else:
                    # Remove batch dimension if tensor
                    sample[k] = v[0] if hasattr(v, 'shape') and len(v.shape) > 0 else v

            # Add task info
            sample["task"] = task
            if "modality" not in sample:
                sample["modality"] = modality

            samples.append(sample)

        return samples

    def get_batch(self, batch_size: int) -> list[dict]:
        """Convenience method to get a batch.

        Args:
            batch_size: Number of samples

        Returns:
            List of samples
        """
        return self.sample(batch_size)


class StageDatasetFactory:
    """Factory for creating datasets for each training stage.

    Handles:
    - Loading appropriate datasets for each stage
    - Setting up sampling weights
    - Managing resolution ranges
    """

    @staticmethod
    def create_stage_datasets(
        stage_config,
        data_roots: Dict[str, Optional[str]],
    ) -> Dict[str, Dataset]:
        """Create datasets for a training stage.

        Args:
            stage_config: StageConfig object
            data_roots: Dict mapping modality -> root path
                e.g., {"image": "/data/images", "video": "/data/videos", "3d": "/data/3d"}

        Returns:
            Dict mapping modality -> Dataset
        """
        from .datasets_modality import ImageDataset, VideoDataset, Object3DDataset

        datasets = {}

        # Image dataset
        if "image" in stage_config.modalities:
            image_root = data_roots.get("image")
            if image_root:
                datasets["image"] = ImageDataset(
                    data_paths=[image_root],
                    resolution_range=(
                        min(stage_config.image_resolutions),
                        max(stage_config.image_resolutions)
                    ),
                    buckets=stage_config.image_resolutions,
                    augment=True,
                )
                logger.info(
                    f"Created ImageDataset: {len(datasets['image'])} images, "
                    f"res {stage_config.image_resolutions}"
                )

        # Video dataset
        if "video" in stage_config.modalities:
            video_root = data_roots.get("video")
            if video_root and stage_config.video_resolutions:
                datasets["video"] = VideoDataset(
                    data_paths=[video_root],
                    resolution_range=(
                        min(stage_config.video_resolutions),
                        max(stage_config.video_resolutions)
                    ),
                    frame_counts=[4, 8, 16, 32],
                    temporal_patch=2,
                    augment=True,
                )
                logger.info(
                    f"Created VideoDataset: {len(datasets['video'])} videos, "
                    f"res {stage_config.video_resolutions}"
                )

        # 3D dataset
        if "3d" in stage_config.modalities:
            obj3d_root = data_roots.get("3d")
            if obj3d_root and stage_config.triplane_sizes:
                datasets["3d"] = Object3DDataset(
                    render_dir=obj3d_root,
                    triplane_res=stage_config.triplane_sizes[0],  # Use first size
                    n_views=8,
                    augment=True,
                )
                logger.info(
                    f"Created Object3DDataset: {len(datasets['3d'])} objects, "
                    f"triplane_res {stage_config.triplane_sizes}"
                )

        if not datasets:
            raise ValueError(
                f"No datasets created for stage {stage_config.name}. "
                f"Check data_roots: {data_roots}"
            )

        return datasets

    @staticmethod
    def create_sampler(
        stage_config,
        datasets: Dict[str, Dataset],
        num_workers: int = 4,
    ) -> ModalityWeightedSampler:
        """Create weighted sampler for a stage.

        Args:
            stage_config: StageConfig object
            datasets: Dict mapping modality -> Dataset
            num_workers: Number of workers per dataset

        Returns:
            ModalityWeightedSampler
        """
        return ModalityWeightedSampler(
            datasets=datasets,
            task_weights=stage_config.task_ratios,
            num_workers=num_workers,
        )


__all__ = [
    "ModalityWeightedSampler",
    "StageDatasetFactory",
]
