"""PyTorch Lightning DataModule for MAVT multi-modal training."""

from __future__ import annotations
import os
import random
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
import lightning as L

from mavt.data.datasets import (
    SyntheticMultiModalDataset,
    ImageFolderDataset,
    VideoDataset,
    ThreeDDataset,
    UniversalImageDataset,
    UniversalVideoDataset,
    UniversalThreeDDataset,
)


def _collate(batch):
    """Collate a batch — all items must share the same modality."""
    modality = batch[0]['modality']
    data = torch.stack([b['data'] for b in batch])
    result = {'data': data, 'modality': modality}
    if 'caption' in batch[0]:
        result['caption'] = [b['caption'] for b in batch]
    if 'id' in batch[0]:
        result['id'] = [b['id'] for b in batch]
    return result


class ModalityGroupedBatchSampler(Sampler):
    """Batch sampler for ConcatDataset that keeps each batch single-modality.

    Indices are grouped by sub-dataset (= modality), shuffled within each
    group, then all batches are interleaved and globally shuffled so that
    modalities alternate throughout the epoch.

    DDP: set rank / world_size before the DataLoader is iterated so that
    each process receives a disjoint subset of batches.
    Use trainer: use_distributed_sampler: false to stop Lightning from
    replacing this sampler with a DistributedSampler.
    """

    def __init__(self, concat_dataset: ConcatDataset, batch_size: int,
                 drop_last: bool = True, shuffle: bool = True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Build per-modality index lists from ConcatDataset boundaries
        self.groups: List[List[int]] = []
        offset = 0
        for ds in concat_dataset.datasets:
            n = len(ds)
            self.groups.append(list(range(offset, offset + n)))
            offset += n

    @staticmethod
    def _dist_info() -> tuple:
        """Return (rank, world_size) from the live process group if available."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _make_batches(self) -> List[List[int]]:
        all_batches: List[List[int]] = []
        for indices in self.groups:
            idx = indices[:]
            if self.shuffle:
                random.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                batch = idx[start: start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        rank, world_size = self._dist_info()
        all_batches = self._make_batches()
        # Truncate to a multiple of world_size so every rank gets the same count
        n = (len(all_batches) // world_size) * world_size
        all_batches = all_batches[:n]
        for i, batch in enumerate(all_batches):
            if i % world_size == rank:
                yield batch

    def __len__(self) -> int:
        _, world_size = self._dist_info()
        # If DDP not yet initialised (e.g. called during DataLoader construction),
        # fall back to WORLD_SIZE env var so the length matches what __iter__ yields.
        if world_size == 1:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        total = sum(
            len(g) // self.batch_size if self.drop_last
            else (len(g) + self.batch_size - 1) // self.batch_size
            for g in self.groups
        )
        return total // world_size


class MAVTDataModule(L.LightningDataModule):
    """DataModule supporting 3-stage curriculum.

    Stage 1: image only
    Stage 2: image + video
    Stage 3: image + video + 3D

    Set active_modalities to control which modalities are included.
    For synthetic smoke-testing leave all *_root paths as None.
    """

    def __init__(
        self,
        # Stage control
        active_modalities: List[str] = ('image',),
        # Paths (None → use synthetic data)
        image_root: Optional[str] = None,
        video_root: Optional[str] = None,
        threed_root: Optional[str] = None,
        # Universal data root — if set, overrides the three roots above
        universal_data_root: Optional[str] = None,
        # Data params
        image_resolution: int = 256,
        video_frames: int = 16,
        video_resolution: int = 256,
        triplane_res: int = 256,
        # Synthetic (smoke test)
        synthetic_n: int = 64,
        # DataLoader params
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    # ------------------------------------------------------------------ #

    def _make_dataset(self, modality: str) -> Dataset:
        hp = self.hparams
        if hp.universal_data_root:
            if modality == 'image':
                return UniversalImageDataset(hp.universal_data_root, hp.image_resolution)
            elif modality == 'video':
                return UniversalVideoDataset(hp.universal_data_root, hp.video_frames, hp.video_resolution)
            elif modality == 'threed':
                return UniversalThreeDDataset(hp.universal_data_root, hp.triplane_res)
        if modality == 'image':
            root = hp.image_root
            return (ImageFolderDataset(root, hp.image_resolution)
                    if root else
                    SyntheticMultiModalDataset(hp.synthetic_n, 'image', hp.image_resolution))
        elif modality == 'video':
            root = hp.video_root
            return (VideoDataset(root, hp.video_frames, hp.video_resolution)
                    if root else
                    SyntheticMultiModalDataset(hp.synthetic_n, 'video',
                                               hp.video_resolution, hp.video_frames))
        elif modality == 'threed':
            root = hp.threed_root
            return (ThreeDDataset(root, hp.triplane_res)
                    if root else
                    SyntheticMultiModalDataset(hp.synthetic_n, 'threed',
                                               triplane_res=hp.triplane_res))
        raise ValueError(modality)

    # ------------------------------------------------------------------ #

    def setup(self, stage: Optional[str] = None) -> None:
        datasets = [self._make_dataset(m) for m in self.hparams.active_modalities]
        if len(datasets) == 1:
            self._train_ds: Dataset = datasets[0]
            self._batch_sampler = None
        else:
            self._train_ds = ConcatDataset(datasets)
            self._batch_sampler = ModalityGroupedBatchSampler(
                self._train_ds, self.hparams.batch_size,
                drop_last=True, shuffle=True,
            )

    def train_dataloader(self) -> DataLoader:
        hp = self.hparams
        if self._batch_sampler is not None:
            return DataLoader(
                self._train_ds,
                batch_sampler=self._batch_sampler,
                num_workers=hp.num_workers,
                pin_memory=hp.pin_memory,
                collate_fn=_collate,
            )
        return DataLoader(
            self._train_ds,
            batch_size=hp.batch_size,
            shuffle=True,
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            collate_fn=_collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_ds = SyntheticMultiModalDataset(
            16, self.hparams.active_modalities[0],
            self.hparams.image_resolution,
        )
        return DataLoader(
            val_ds, batch_size=4, shuffle=False,
            num_workers=2, collate_fn=_collate,
        )
