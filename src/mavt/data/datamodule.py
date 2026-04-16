from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from mavt.compat import LightningDataModule
from mavt.data.image_dataset import ImageDataset
from mavt.data.samplers import WeightedModalLoader
from mavt.data.video_dataset import VideoDataset


class UnifiedDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str = "./data",
        *,
        stage: int = 1,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: int = 256,
        video_size: int = 256,
        num_video_frames: int = 16,
        steps_per_epoch: int = 1000,
        modality_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.stage = stage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.video_size = video_size
        self.num_video_frames = num_video_frames
        self.steps_per_epoch = steps_per_epoch
        self.modality_weights = modality_weights or {"image": 1.0, "video": 1.0}
        self.train_datasets = {}
        self.val_datasets = {}

    def setup(self, stage: str | None = None) -> None:
        self.train_datasets["image"] = ImageDataset(
            self.data_root,
            split="train",
            image_size=self.image_size,
        )
        self.val_datasets["image"] = ImageDataset(
            self.data_root,
            split="val",
            image_size=self.image_size,
        )
        if self.stage >= 2:
            self.train_datasets["video"] = VideoDataset(
                self.data_root,
                split="train",
                image_size=self.video_size,
                num_frames=self.num_video_frames,
            )
            self.val_datasets["video"] = VideoDataset(
                self.data_root,
                split="val",
                image_size=self.video_size,
                num_frames=self.num_video_frames,
            )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        loaders = {name: self._loader(dataset, shuffle=True) for name, dataset in self.train_datasets.items()}
        if len(loaders) == 1:
            return next(iter(loaders.values()))
        return WeightedModalLoader(loaders, self.modality_weights, self.steps_per_epoch)

    def val_dataloader(self):
        return [self._loader(dataset, shuffle=False) for dataset in self.val_datasets.values()]

    def test_dataloader(self):
        return self.val_dataloader()
