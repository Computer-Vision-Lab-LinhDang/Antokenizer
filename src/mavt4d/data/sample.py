"""SceneSample: one training/eval example (cameras over time, optional LiDAR sweeps and boxes)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

Tier = Literal["A", "B", "C"]


@dataclass
class Boxes3D:
    center: Tensor  # (M, 3) float32, world frame, metres
    size: Tensor  # (M, 3) (w, l, h)
    yaw: Tensor  # (M,) rad about world z
    instance_id: Tensor  # (M,) int64, stable within a scene


@dataclass
class SceneSample:
    scene_id: str
    source: str  # "synthetic" | "nuscenes" | "openimages" | "openvid"
    images: Tensor  # (V, T, 3, H, W) float32 in [-1, 1]
    frame_times: Tensor  # (V, T) float64, seconds from t0
    intrinsics: Tensor | None = None  # (V, 3, 3), pixels at (H, W)
    cam_to_world: Tensor | None = None  # (V, T, 4, 4)
    lidar_points: list[Tensor] | None = None  # S x (N_s, 4): xyz in LiDAR frame + intensity
    lidar_to_world: Tensor | None = None  # (S, 4, 4)
    lidar_times: Tensor | None = None  # (S,) float64
    lidar_keys: list[str] | None = None  # S stable keys (nuScenes sample_data token)
    heldout: list[Tensor] | None = None  # S x (N_s,) bool, True = held out for evaluation
    boxes: list[Boxes3D] | None = None  # T entries
    patch_flow: Tensor | None = None  # (V, T-1, Hp, Wp, 3): (du, dv) in patch units, valid

    @property
    def tier(self) -> Tier:
        """A: LiDAR + intrinsics + cam_to_world; B: video (T > 1); otherwise C."""
        if self.lidar_points and self.intrinsics is not None and self.cam_to_world is not None:
            return "A"
        return "B" if self.images.shape[1] > 1 else "C"

    def input_points(self, s: int) -> Tensor:
        """Points of sweep ``s`` the model may see: all of them minus the held-out ones."""
        if self.lidar_points is None:
            raise ValueError("lidar_points: sample has no LiDAR")
        pts = self.lidar_points[s]
        return pts if self.heldout is None else pts[~self.heldout[s]]

    def validate(self) -> None:
        """Raise ValueError whose message starts with the name of the first inconsistent field."""
        im = self.images
        if im.ndim != 5 or im.shape[2] != 3 or not im.is_floating_point():
            raise ValueError(f"images: expected float (V, T, 3, H, W), got {tuple(im.shape)} {im.dtype}")
        V, T = im.shape[:2]
        _check("frame_times", self.frame_times, (V, T))
        if self.intrinsics is not None:
            _check("intrinsics", self.intrinsics, (V, 3, 3))
        if self.cam_to_world is not None:
            _check("cam_to_world", self.cam_to_world, (V, T, 4, 4))
        self._validate_lidar()
        if self.boxes is not None:
            if len(self.boxes) != T:
                raise ValueError(f"boxes: expected {T} entries (one per frame), got {len(self.boxes)}")
            for t, b in enumerate(self.boxes):
                m = b.center.shape[0]
                for name, shape in (("center", (m, 3)), ("size", (m, 3)), ("yaw", (m,)), ("instance_id", (m,))):
                    _check(f"boxes[{t}].{name}", getattr(b, name), shape)
                if b.instance_id.dtype != torch.int64:
                    raise ValueError(f"boxes[{t}].instance_id: expected int64, got {b.instance_id.dtype}")
        if self.patch_flow is not None:
            _check("patch_flow", self.patch_flow, (V, T - 1, None, None, 3))

    def _validate_lidar(self) -> None:
        companions = ("lidar_to_world", "lidar_times", "lidar_keys", "heldout")
        if self.lidar_points is None:
            for name in companions:
                if getattr(self, name) is not None:
                    raise ValueError(f"{name}: set while lidar_points is None")
            return
        S = len(self.lidar_points)
        for s, p in enumerate(self.lidar_points):
            _check(f"lidar_points[{s}]", p, (None, 4))
        for name in ("lidar_to_world", "lidar_times", "lidar_keys"):
            if getattr(self, name) is None:
                raise ValueError(f"{name}: required when lidar_points is set")
        _check("lidar_to_world", self.lidar_to_world, (S, 4, 4))
        _check("lidar_times", self.lidar_times, (S,))
        if len(self.lidar_keys) != S:
            raise ValueError(f"lidar_keys: expected {S} keys, got {len(self.lidar_keys)}")
        if self.heldout is not None:
            if len(self.heldout) != S:
                raise ValueError(f"heldout: expected {S} masks, got {len(self.heldout)}")
            for s, (h, p) in enumerate(zip(self.heldout, self.lidar_points)):
                _check(f"heldout[{s}]", h, (p.shape[0],))
                if h.dtype != torch.bool:
                    raise ValueError(f"heldout[{s}]: expected bool, got {h.dtype}")


def _check(name: str, x: Tensor, shape: tuple[int | None, ...]) -> None:
    """Raise ValueError naming ``name`` unless ``x`` has ``shape`` (None matches any size)."""
    ok = x.ndim == len(shape) and all(e is None or e == d for e, d in zip(shape, x.shape))
    if not ok:
        want = tuple("*" if e is None else e for e in shape)
        raise ValueError(f"{name}: expected shape {want}, got {tuple(x.shape)}")
