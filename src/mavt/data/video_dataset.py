from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mavt.data.transforms import transform_video


class VideoDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        image_size: int = 256,
        num_frames: int = 16,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames
        self.video_dir = self.root / "videos"
        self.caption_path = self.root / "captions" / "videos.json"
        self.captions = json.loads(self.caption_path.read_text()) if self.caption_path.exists() else {}
        self.files = sorted(path for path in self.video_dir.glob("*.mp4"))

    def __len__(self) -> int:
        return len(self.files)

    def _load_frames(self, path: Path) -> torch.Tensor:
        try:  # pragma: no cover - optional dependency
            from decord import VideoReader
        except Exception:
            VideoReader = None
        if VideoReader is not None:
            reader = VideoReader(str(path))
            indices = self._sample_indices(len(reader))
            frames = reader.get_batch(indices).asnumpy()
            tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
            return tensor
        try:  # pragma: no cover - optional dependency
            from torchvision.io import read_video
        except Exception:
            read_video = None
        if read_video is None:
            raise ImportError("Install decord or torchvision video support to read videos.")
        frames, _, _ = read_video(str(path), pts_unit="sec")
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        indices = self._sample_indices(frames.shape[0])
        return frames[indices].permute(1, 0, 2, 3)

    def _sample_indices(self, total_frames: int) -> list[int]:
        if total_frames <= self.num_frames:
            return [min(idx, total_frames - 1) for idx in range(self.num_frames)]
        if self.split == "train":
            start = torch.randint(0, total_frames - self.num_frames + 1, (1,)).item()
        else:
            start = max((total_frames - self.num_frames) // 2, 0)
        return list(range(start, start + self.num_frames))

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        tensor = self._load_frames(path)
        tensor = transform_video(tensor, self.image_size, train=self.split == "train")
        stem = path.stem
        return {
            "id": stem,
            "video": tensor,
            "caption": self.captions.get(stem, ""),
            "modality": "video",
        }
