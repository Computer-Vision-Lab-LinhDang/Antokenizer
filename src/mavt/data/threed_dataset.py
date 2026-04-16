from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class ThreeDDataset(Dataset):
    """
    Deferred 3D path.

    This dataset exposes the expected interface but is intentionally not wired
    into training until the reconstruction target is finalized.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.render_root = self.root / "3d_objects" / "renders"
        caption_path = self.root / "captions" / "3d.json"
        self.captions = json.loads(caption_path.read_text()) if caption_path.exists() else {}
        self.object_dirs = sorted(path for path in self.render_root.glob("*") if path.is_dir())

    def __len__(self) -> int:
        return len(self.object_dirs)

    def __getitem__(self, index: int) -> dict:
        obj_dir = self.object_dirs[index]
        cameras_path = obj_dir / "cameras.json"
        cameras = json.loads(cameras_path.read_text()) if cameras_path.exists() else {}
        views = []
        for view_path in sorted(obj_dir.glob("view_*.png")):
            image = Image.open(view_path).convert("RGB")
            views.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0)
        return {
            "id": obj_dir.name,
            "views": views,
            "cameras": cameras,
            "caption": self.captions.get(obj_dir.name, ""),
            "modality": "threed",
        }
