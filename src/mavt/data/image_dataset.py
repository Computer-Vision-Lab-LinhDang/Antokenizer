from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mavt.data.transforms import transform_image


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        image_size: int = 384,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.image_dir = self.root / "images"
        self.caption_path = self.root / "captions" / "images.json"
        self.captions = json.loads(self.caption_path.read_text()) if self.caption_path.exists() else {}
        self.files = sorted(
            path for path in self.image_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        tensor = transform_image(tensor, self.image_size, train=self.split == "train")
        stem = path.stem
        return {
            "id": stem,
            "image": tensor,
            "caption": self.captions.get(stem, ""),
            "modality": "image",
        }
