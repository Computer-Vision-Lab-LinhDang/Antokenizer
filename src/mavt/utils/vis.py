from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_reconstruction_grid(target: torch.Tensor, reconstruction: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if target.ndim == 5:
        target = target[:, :, 0]
        reconstruction = reconstruction[:, :, 0]
    images = torch.cat([target, reconstruction], dim=0).clamp(-1, 1)
    images = (images + 1.0) / 2.0
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    tiles = [(image * 255.0).astype(np.uint8) for image in images]
    height, width = tiles[0].shape[:2]
    canvas = np.zeros((height, width * len(tiles), 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        canvas[:, idx * width : (idx + 1) * width] = tile
    Image.fromarray(canvas).save(output_path)
