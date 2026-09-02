"""PyTorch Lightning DataModule for MAVT multi-modal training."""

from __future__ import annotations
import os
import random
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Sampler, random_split
import lightning as L

from pathlib import Path

from mavt.data.datasets import (
    SyntheticMultiModalDataset,
    UniversalImageDataset,
    UniversalVideoDataset,
    UniversalThreeDDataset,
    WDSImageDataset,
    ShardVideoDataset,
    ManifestImageDataset,
    ManifestVideoDataset,
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
                 drop_last: bool = True, shuffle: bool = True,
                 modalities: Optional[Sequence[str]] = None,
                 modality_batch_sizes: Optional[Dict[str, int]] = None,
                 modality_weights: Optional[Dict[str, float]] = None):
        """
        batch_size           : default batch size (per modality unless overridden)
        modalities           : name of each sub-dataset, in ConcatDataset order
        modality_batch_sizes : per-modality override, e.g. {"video": 8, "threed": 8}
        modality_weights     : share of batches per modality, e.g. {"image": .5, "video": .35, "threed": .15}.
                               The epoch length is set by the modality that would be under-sampled
                               most (max over n_batches/weight); smaller modalities are oversampled
                               (re-shuffled and re-chunked) to reach their share. None = proportional.
        """
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
        self.modalities = list(modalities) if modalities else [f"ds{i}" for i in range(len(self.groups))]
        if len(self.modalities) != len(self.groups):
            raise ValueError(f"{len(self.modalities)} modality names for {len(self.groups)} datasets")
        mbs = modality_batch_sizes or {}
        self.batch_sizes = [int(mbs.get(m, batch_size)) for m in self.modalities]
        self.weights: Optional[List[float]] = None
        if modality_weights:
            w = [float(modality_weights.get(m, 0.0)) for m in self.modalities]
            if min(w) <= 0:
                raise ValueError(f"modality_weights must be > 0 for every active modality: {modality_weights}")
            tot = sum(w); self.weights = [x / tot for x in w]

    def _natural_counts(self) -> List[int]:
        """Batches per modality when each dataset is seen exactly once."""
        out = []
        for indices, bs in zip(self.groups, self.batch_sizes):
            n = len(indices) // bs if self.drop_last else -(-len(indices) // bs)
            out.append(n)
        return out

    def _target_counts(self) -> List[int]:
        nat = self._natural_counts()
        if not self.weights:
            return nat
        total = max(n / w for n, w in zip(nat, self.weights))
        return [int(round(w * total)) for w in self.weights]

    def _chunk(self, indices: List[int], bs: int) -> List[List[int]]:
        idx = indices[:]
        if self.shuffle:
            random.shuffle(idx)
        out = []
        for start in range(0, len(idx), bs):
            batch = idx[start: start + bs]
            if self.drop_last and len(batch) < bs:
                continue
            out.append(batch)
        return out

    @staticmethod
    def _dist_info() -> tuple:
        """Return (rank, world_size) from the live process group if available."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _make_batches(self) -> List[List[int]]:
        all_batches: List[List[int]] = []
        for indices, bs, target in zip(self.groups, self.batch_sizes, self._target_counts()):
            if not indices or target <= 0:
                continue
            batches = self._chunk(indices, bs)
            while len(batches) < target:                     # oversample: fresh shuffle + re-chunk
                batches.extend(self._chunk(indices, bs))
            all_batches.extend(batches[:target])
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
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        total = sum(self._target_counts())
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
        # Data root (None → synthetic smoke-test)
        universal_data_root: Optional[str] = None,
        # Per-modality shard roots (override universal_data_root when set)
        image_shards_dir: Optional[str] = None,
        video_shards_dir: Optional[str] = None,
        # Manifest-backed datasets (.jsonl from mavt.data.manifest) — highest precedence
        image_manifest: Optional[str] = None,
        video_manifest: Optional[str] = None,
        video_max_shards: Optional[int] = None,
        # Data params
        image_resolution: int = 256,
        video_frames: int = 16,
        video_frame_stride: int = 2,
        modality_batch_sizes: Optional[Dict[str, int]] = None,   # e.g. {video: 8, threed: 8}
        loader_timeout: int = 0,          # seconds a rank waits for a worker batch before raising (0 = forever)
        modality_weights: Optional[Dict[str, float]] = None,     # e.g. {image: .5, video: .35, threed: .15}
        video_resolution: int = 256,
        triplane_res: int = 256,
        # Synthetic (smoke test)
        synthetic_n: int = 64,
        # Train/val/test split
        val_split: float = 0.05,
        test_split: float = 0.0,
        # DataLoader params
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    # ------------------------------------------------------------------ #

    def _make_dataset(self, modality: str) -> Dataset:
        hp = self.hparams
        if modality == 'image' and hp.image_manifest:
            return ManifestImageDataset(hp.image_manifest, hp.image_resolution)
        if modality == 'video' and hp.video_manifest:
            return ManifestVideoDataset(hp.video_manifest, hp.video_frames, hp.video_resolution,
                                        frame_stride=hp.video_frame_stride)
        # Per-modality shard roots take precedence over universal_data_root
        if modality == 'image' and hp.image_shards_dir:
            return WDSImageDataset(hp.image_shards_dir, hp.image_resolution)
        if modality == 'video' and hp.video_shards_dir:
            return ShardVideoDataset(
                hp.video_shards_dir, hp.video_frames, hp.video_resolution,
                max_shards=hp.video_max_shards,
            )
        if hp.universal_data_root:
            if modality == 'image':
                root = Path(hp.universal_data_root)
                # Auto-detect WebDataset .tar shard layout vs. raw images/ directory
                if any(root.glob("*.tar")):
                    return WDSImageDataset(str(root), hp.image_resolution)
                return UniversalImageDataset(str(root), hp.image_resolution)
            elif modality == 'video':
                return UniversalVideoDataset(hp.universal_data_root, hp.video_frames, hp.video_resolution)
            elif modality == 'threed':
                return UniversalThreeDDataset(hp.universal_data_root, hp.triplane_res)
            raise ValueError(modality)
        # Synthetic fallback for smoke-testing
        if modality == 'image':
            return SyntheticMultiModalDataset(hp.synthetic_n, 'image', hp.image_resolution)
        elif modality == 'video':
            return SyntheticMultiModalDataset(hp.synthetic_n, 'video',
                                              hp.video_resolution, hp.video_frames)
        elif modality == 'threed':
            return SyntheticMultiModalDataset(hp.synthetic_n, 'threed',
                                              triplane_res=hp.triplane_res)
        raise ValueError(modality)

    # ------------------------------------------------------------------ #

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        full_datasets = [self._make_dataset(m) for m in hp.active_modalities]

        train_splits, val_splits, test_splits = [], [], []
        for ds in full_datasets:
            n = len(ds)
            n_val = max(1, int(n * hp.val_split))
            n_test = max(1, int(n * hp.test_split)) if hp.test_split > 0 else 0
            n_train = n - n_val - n_test
            parts = random_split(ds, [n_train, n_val, n_test]) if n_test else random_split(ds, [n_train, n_val])
            train_splits.append(parts[0])
            val_splits.append(parts[1])
            if n_test:
                test_splits.append(parts[2])

        if len(train_splits) == 1:
            self._train_ds: Dataset = train_splits[0]
            self._batch_sampler = None
        else:
            self._train_ds = ConcatDataset(train_splits)
            self._batch_sampler = ModalityGroupedBatchSampler(
                self._train_ds, hp.batch_size,
                modalities=list(hp.active_modalities),
                modality_batch_sizes=hp.modality_batch_sizes, modality_weights=hp.modality_weights,
                drop_last=True, shuffle=True,
            )

        if len(val_splits) == 1:
            self._val_ds: Dataset = val_splits[0]
            self._val_batch_sampler: Optional[ModalityGroupedBatchSampler] = None
        else:
            self._val_ds = ConcatDataset(val_splits)
            self._val_batch_sampler = ModalityGroupedBatchSampler(
                self._val_ds, hp.batch_size,
                drop_last=False, shuffle=False,
                modalities=list(hp.active_modalities), modality_batch_sizes=hp.modality_batch_sizes,
            )

        if test_splits:
            if len(test_splits) == 1:
                self._test_ds: Optional[Dataset] = test_splits[0]
                self._test_batch_sampler: Optional[ModalityGroupedBatchSampler] = None
            else:
                self._test_ds = ConcatDataset(test_splits)
                self._test_batch_sampler = ModalityGroupedBatchSampler(
                    self._test_ds, hp.batch_size,
                    drop_last=False, shuffle=False,
                )
        else:
            self._test_ds = None
            self._test_batch_sampler = None

    def _loader_extras(self) -> Dict:
        hp = self.hparams
        kw = {}
        if hp.num_workers > 0:
            kw['persistent_workers'] = bool(hp.persistent_workers)
            if hp.prefetch_factor is not None:
                kw['prefetch_factor'] = int(hp.prefetch_factor)
            if hp.loader_timeout and hp.loader_timeout > 0:
                kw['timeout'] = int(hp.loader_timeout)   # a hung worker must fail loudly, not stall the DDP group
        return kw

    def train_dataloader(self) -> DataLoader:
        hp = self.hparams
        if self._batch_sampler is not None:
            return DataLoader(
                self._train_ds,
                batch_sampler=self._batch_sampler,
                num_workers=hp.num_workers,
                pin_memory=hp.pin_memory,
                collate_fn=_collate,
                **self._loader_extras(),
            )
        # Single modality: shard across DDP ranks ourselves (the YAML disables
        # Lightning's sampler replacement for the multi-modality batch sampler).
        sampler = None
        shuffle = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self._train_ds, shuffle=True, drop_last=True)
            shuffle = False
        return DataLoader(
            self._train_ds,
            batch_size=hp.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            collate_fn=_collate,
            drop_last=True,
            **self._loader_extras(),
        )

    def val_dataloader(self) -> DataLoader:
        hp = self.hparams
        if self._val_batch_sampler is not None:
            return DataLoader(
                self._val_ds,
                batch_sampler=self._val_batch_sampler,
                num_workers=hp.num_workers,
                pin_memory=hp.pin_memory,
                collate_fn=_collate,
                **self._loader_extras(),
            )
        return DataLoader(
            self._val_ds, batch_size=hp.batch_size, shuffle=False,
            num_workers=hp.num_workers, pin_memory=hp.pin_memory,
            collate_fn=_collate, drop_last=False,
            **self._loader_extras(),
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self._test_ds is None:
            return None
        hp = self.hparams
        if self._test_batch_sampler is not None:
            return DataLoader(
                self._test_ds,
                batch_sampler=self._test_batch_sampler,
                num_workers=hp.num_workers,
                pin_memory=hp.pin_memory,
                collate_fn=_collate,
                **self._loader_extras(),
            )
        return DataLoader(
            self._test_ds, batch_size=hp.batch_size, shuffle=False,
            num_workers=hp.num_workers, pin_memory=hp.pin_memory,
            collate_fn=_collate, drop_last=False,
        )