from __future__ import annotations

import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


def _normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def _resize_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    resized = F.interpolate(
        x.unsqueeze(0),
        size=(size, size),
        mode="bicubic",
        align_corners=False,
    )
    return resized.squeeze(0)


def _crop_tensor(x: torch.Tensor, size: int, train: bool) -> torch.Tensor:
    _, height, width = x.shape
    if height == size and width == size:
        return x
    max_top = max(height - size, 0)
    max_left = max(width - size, 0)
    if train:
        top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
        left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0
    else:
        top = max_top // 2
        left = max_left // 2
    return x[:, top : top + size, left : left + size]


def transform_image(x: torch.Tensor, size: int, train: bool) -> torch.Tensor:
    x = _resize_tensor(x, size)
    x = _crop_tensor(x, size, train=train)
    return _normalize_tensor(x)


def transform_video(x: torch.Tensor, size: int, train: bool) -> torch.Tensor:
    frames = []
    for frame in x.permute(1, 0, 2, 3):
        frame = _resize_tensor(frame, size)
        frame = _crop_tensor(frame, size, train=train)
        frames.append(_normalize_tensor(frame))
    return torch.stack(frames, dim=1)
