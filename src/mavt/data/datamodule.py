"""PyTorch Lightning DataModule for MAVT multi-modal training."""

from __future__ import annotations
import os
import random
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, random_split
import lightning as L

from pathlib import Path

from mavt.data.datasets import (
    HFParquetImageDataset,
    ShardVideoDataset,
    SyntheticMultiModalDataset,
    UniversalImageDataset,
    UniversalVideoDataset,
    UniversalThreeDDataset,
    WDSImageDataset,
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
    """Batch sampler for ConcatDataset that keeps each step single-modality on
    every DDP rank.

    Each global step is a *super-batch* of ``world_size`` per-rank batches that
    all share the same modality. This is required for DDP: a single forward /
    backward across ranks must touch the same compute graph (otherwise
    DDP gradient sync stalls or wastes work even with
    ``find_unused_parameters=True``).

    Algorithm
    ---------
    1.  Per modality, indices are shuffled and chunked into per-rank batches of
        ``batch_size``.
    2.  Optional ``modality_weights`` rescale the number of batches per modality
        (oversampling when target > natural, truncation when target < natural).
    3.  Each modality's batch list is truncated to a multiple of ``world_size``
        and grouped into super-batches.
    4.  All super-batches are concatenated and globally shuffled.
    5.  Rank ``r`` yields the ``r``-th batch of each super-batch.

    DDP usage
    ---------
    Set ``trainer.use_distributed_sampler: false`` so Lightning does not wrap
    this with ``DistributedSampler``. Call ``set_epoch(epoch)`` at the start of
    each epoch on every rank — required for shuffles to stay in lockstep.
    """

    def __init__(self, concat_dataset: ConcatDataset, batch_size: int,
                 drop_last: bool = True, shuffle: bool = True,
                 seed: int = 42,
                 modalities: Optional[Sequence[str]] = None,
                 modality_weights: Optional[Dict[str, float]] = None):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Build per-modality index lists from ConcatDataset boundaries
        self.groups: List[List[int]] = []
        offset = 0
        for ds in concat_dataset.datasets:
            n = len(ds)
            self.groups.append(list(range(offset, offset + n)))
            offset += n

        # Default modality names ('group0', 'group1', ...) if caller did not
        # supply explicit names. Names only matter when modality_weights is set.
        if modalities is None:
            self.modalities: List[str] = [f'group{i}' for i in range(len(self.groups))]
        else:
            if len(modalities) != len(self.groups):
                raise ValueError(
                    f"modalities length ({len(modalities)}) does not match "
                    f"number of sub-datasets ({len(self.groups)})"
                )
            self.modalities = list(modalities)

        if modality_weights is not None:
            missing = set(self.modalities) - set(modality_weights)
            if missing:
                raise ValueError(
                    f"modality_weights missing entries for: {sorted(missing)}"
                )
            if any(w <= 0 for w in modality_weights.values()):
                raise ValueError("modality_weights must be strictly positive")
        self.modality_weights = modality_weights

    def set_epoch(self, epoch: int) -> None:
        """Sync per-epoch shuffle seed across DDP ranks. Call once per epoch."""
        self.epoch = int(epoch)

    @staticmethod
    def _dist_info() -> tuple:
        """Return (rank, world_size) from the live process group if available.

        Falls back to the WORLD_SIZE env var so ``__len__`` is correct before
        Lightning has spun up the process group.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        return rank, world_size

    def _per_modality_batches(self, rng: random.Random) -> Dict[str, List[List[int]]]:
        """Chunk each modality's index list into per-rank batches of batch_size."""
        out: Dict[str, List[List[int]]] = {}
        for name, indices in zip(self.modalities, self.groups):
            idx = indices[:]
            if self.shuffle:
                rng.shuffle(idx)
            batches: List[List[int]] = []
            for start in range(0, len(idx), self.batch_size):
                b = idx[start: start + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                batches.append(b)
            out[name] = batches
        return out

    def _apply_weights(self, per_mod: Dict[str, List[List[int]]],
                       rng: random.Random) -> Dict[str, List[List[int]]]:
        """Resample per-modality batch lists to match modality_weights.

        Targets are computed so the sum of batches stays close to the natural
        total: target_m = round(total_natural * w_m / sum(w)).
        Modalities short of target are oversampled (re-shuffled repeats);
        modalities over target are truncated.
        """
        if not self.modality_weights:
            return per_mod
        natural = {m: len(b) for m, b in per_mod.items()}
        total_natural = sum(natural.values())
        if total_natural == 0:
            return per_mod
        sum_w = sum(self.modality_weights[m] for m in per_mod)
        out: Dict[str, List[List[int]]] = {}
        for m, batches in per_mod.items():
            if not batches:
                out[m] = batches
                continue
            target = max(1, int(round(total_natural * self.modality_weights[m] / sum_w)))
            if target == len(batches):
                out[m] = batches
            elif target < len(batches):
                out[m] = batches[:target]
            else:
                extended: List[List[int]] = []
                while len(extended) < target:
                    cp = batches[:]
                    if self.shuffle:
                        rng.shuffle(cp)
                    extended.extend(cp)
                out[m] = extended[:target]
        return out

    def _make_super_batches(self) -> List[List[List[int]]]:
        """Build the list of super-batches (each: world_size batches, same modality)."""
        rng = random.Random(self.seed + self.epoch)
        _, world_size = self._dist_info()
        per_mod = self._per_modality_batches(rng)
        per_mod = self._apply_weights(per_mod, rng)
        super_batches: List[List[List[int]]] = []
        for batches in per_mod.values():
            n = (len(batches) // world_size) * world_size
            for i in range(0, n, world_size):
                super_batches.append(batches[i: i + world_size])
        if self.shuffle:
            rng.shuffle(super_batches)
        return super_batches

    def __iter__(self) -> Iterator[List[int]]:
        rank, _ = self._dist_info()
        for sb in self._make_super_batches():
            yield sb[rank]

    def __len__(self) -> int:
        _, world_size = self._dist_info()
        # Pure length computation — must match __iter__ output count without
        # actually allocating index lists.
        natural: Dict[str, int] = {}
        for name, indices in zip(self.modalities, self.groups):
            n = len(indices)
            natural[name] = (
                n // self.batch_size
                if self.drop_last
                else (n + self.batch_size - 1) // self.batch_size
            )
        if self.modality_weights:
            total_natural = sum(natural.values())
            sum_w = sum(self.modality_weights[m] for m in natural)
            target: Dict[str, int] = {}
            for m, n in natural.items():
                if n == 0:
                    target[m] = 0
                else:
                    target[m] = max(1, int(round(
                        total_natural * self.modality_weights[m] / sum_w
                    )))
            natural = target
        return sum(n // world_size for n in natural.values())


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
        # Per-modality overrides (override universal_data_root when set)
        image_shards_dir: Optional[str] = None,
        image_max_shards: Optional[int] = None,
        video_shards_dir: Optional[str] = None,
        video_max_shards: Optional[int] = None,
        triplane_dir: Optional[str] = None,
        # Data params
        image_resolution: int = 256,
        video_frames: int = 16,
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
        # Optional balance-modality weights, e.g. {'image': 1.0, 'video': 1.0,
        # 'threed': 1.0} for an equal mix. None → use natural batch counts.
        modality_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    # ------------------------------------------------------------------ #

    def _make_dataset(self, modality: str) -> Dataset:
        hp = self.hparams
        # Per-modality overrides take precedence over universal_data_root
        if modality == 'image' and hp.image_shards_dir:
            shards_path = Path(hp.image_shards_dir)
            if any(shards_path.glob('train-*.parquet')):
                return HFParquetImageDataset(hp.image_shards_dir, hp.image_resolution, split='train')
            return WDSImageDataset(
                hp.image_shards_dir, hp.image_resolution,
                max_shards=hp.image_max_shards,
            )
        if modality == 'video' and hp.video_shards_dir:
            return ShardVideoDataset(
                hp.video_shards_dir,
                n_frames=hp.video_frames,
                resolution=hp.video_resolution,
                max_shards=hp.video_max_shards,
            )
        if modality == 'threed' and hp.triplane_dir:
            return UniversalThreeDDataset(
                root='', resolution=hp.triplane_res, renders_dir=hp.triplane_dir,
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

        modalities = list(hp.active_modalities)

        if len(train_splits) == 1:
            self._train_ds: Dataset = train_splits[0]
            self._batch_sampler = None
        else:
            self._train_ds = ConcatDataset(train_splits)
            self._batch_sampler = ModalityGroupedBatchSampler(
                self._train_ds, hp.batch_size,
                drop_last=True, shuffle=True,
                modalities=modalities,
                modality_weights=hp.modality_weights,
            )

        if len(val_splits) == 1:
            self._val_ds: Dataset = val_splits[0]
            self._val_batch_sampler = None
        else:
            self._val_ds = ConcatDataset(val_splits)
            # Validation: keep natural ratios so val/loss reflects the actual
            # data distribution, regardless of the train-time balance setting.
            self._val_batch_sampler = ModalityGroupedBatchSampler(
                self._val_ds, hp.batch_size,
                drop_last=False, shuffle=False,
                modalities=modalities,
                modality_weights=None,
            )

        if not test_splits:
            self._test_ds: Optional[Dataset] = None
            self._test_batch_sampler = None
        elif len(test_splits) == 1:
            self._test_ds = test_splits[0]
            self._test_batch_sampler = None
        else:
            self._test_ds = ConcatDataset(test_splits)
            self._test_batch_sampler = ModalityGroupedBatchSampler(
                self._test_ds, hp.batch_size,
                drop_last=False, shuffle=False,
                modalities=modalities,
                modality_weights=None,
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
        hp = self.hparams
        if self._val_batch_sampler is not None:
            return DataLoader(
                self._val_ds,
                batch_sampler=self._val_batch_sampler,
                num_workers=hp.num_workers,
                pin_memory=hp.pin_memory,
                collate_fn=_collate,
            )
        return DataLoader(
            self._val_ds, batch_size=hp.batch_size, shuffle=False,
            num_workers=hp.num_workers, pin_memory=hp.pin_memory,
            collate_fn=_collate, drop_last=False,
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
            )
        return DataLoader(
            self._test_ds, batch_size=hp.batch_size, shuffle=False,
            num_workers=hp.num_workers, pin_memory=hp.pin_memory,
            collate_fn=_collate, drop_last=False,
        )