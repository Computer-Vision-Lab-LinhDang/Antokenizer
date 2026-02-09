from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def _image_transforms(
    image_size: int,
    train: bool,
    mean: List[float],
    std: List[float],
) -> transforms.Compose:
    t_list = []
    if train:
        t_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
        t_list.append(transforms.RandomHorizontalFlip())
    else:
        t_list.append(transforms.Resize(image_size))
        t_list.append(transforms.CenterCrop(image_size))
    t_list.append(transforms.ToTensor())
    t_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t_list)


class ReconstructionWrapper(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, _ = self.base[index]
        return {"image": image}


class SemanticWrapper(Dataset):
    def __init__(self, base: Dataset, num_classes: int) -> None:
        self.base = base
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, target = self.base[index]
        if isinstance(target, list):
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            for idx in target:
                multi_hot[idx] = 1.0
        else:
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            multi_hot[target] = 1.0
        return {"image": image, "target": multi_hot}


class ClassificationWrapper(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, target = self.base[index]
        if isinstance(target, torch.Tensor):
            label = target.long()
        else:
            label = torch.tensor(target, dtype=torch.long)
        return {"image": image, "target": label}


class COCOMultiLabelDataset(datasets.CocoDetection):
    def __init__(
        self,
        root: str,
        ann_file: str,
        transform,
        categories: Optional[List[int]] = None,
    ) -> None:
        super().__init__(root=root, annFile=ann_file, transform=transform)
        if categories is None:
            categories = sorted(self.coco.cats.keys())
        self.categories = categories
        self.cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.categories)}

    def __getitem__(self, index: int):
        image, annotations = super().__getitem__(index)
        multi_hot = torch.zeros(len(self.categories), dtype=torch.float32)
        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id in self.cat_to_idx:
                multi_hot[self.cat_to_idx[cat_id]] = 1.0
        return image, multi_hot


@dataclass
class DataConfig:
    name: str
    root: str = "./data"
    ann_file: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    mean: List[float] = (0.5, 0.5, 0.5)
    std: List[float] = (0.5, 0.5, 0.5)
    pin_memory: bool = True
    persistent_workers: bool = False


def create_dataloader(
    cfg: Dict[str, Any] | DataConfig,
    *,
    task: str,
    train: bool = True,
) -> DataLoader:
    if isinstance(cfg, dict):
        cfg = DataConfig(**cfg)

    name = cfg.name.lower()
    transform = _image_transforms(
        image_size=cfg.image_size,
        train=train,
        mean=list(cfg.mean),
        std=list(cfg.std),
    )

    if name == "mnist":
        base_dataset = datasets.MNIST(
            root=cfg.root,
            train=train,
            transform=transforms.Compose(
                [
                    transforms.Resize(cfg.image_size),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
                ]
            ),
            download=True,
        )
        num_classes = 10
    elif name == "cifar10":
        base_dataset = datasets.CIFAR10(
            root=cfg.root,
            train=train,
            transform=transform,
            download=True,
        )
        num_classes = 10
    elif name == "coco":
        if cfg.ann_file is None:
            raise ValueError("COCO dataset requires annotation file path.")
        base_dataset = COCOMultiLabelDataset(
            root=cfg.root,
            ann_file=cfg.ann_file,
            transform=transform,
        )
        num_classes = len(base_dataset.categories)
    else:
        base_dataset = datasets.ImageFolder(
            root=cfg.root,
            transform=transform,
        )
        num_classes = len(base_dataset.classes)

    if task == "reconstruction":
        dataset = ReconstructionWrapper(base_dataset)
    elif task == "semantic":
        dataset = SemanticWrapper(base_dataset, num_classes=num_classes)
    elif task == "classification":
        dataset = ClassificationWrapper(base_dataset)
    else:
        raise ValueError(f"Unsupported task: {task}")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
    )
    return dataloader
