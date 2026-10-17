"""First-frame external benchmark interface; future cameras are readout controls.

This adapter never accepts future RGB, depth, object poses, phase boundaries or
event labels as model evidence. It does not implement or replace benchmark
metrics. The present model has no text/event conditioning and keeps C fixed
during rollout, which must be declared when scoring changing-state videos.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .decoder import make_ray_query
from .observation import ObservationStep


@dataclass(frozen=True)
class BenchmarkInput:
    first_frame: Tensor  # [3,H,W], RGB [0,1]; exactly one observed image
    intrinsics: Tensor  # [F,3,3], pixel coordinates at first_frame resolution
    camera_c2w: Tensor  # [F,4,4], OpenCV camera axes
    times: Tensor  # [F], physical seconds with declared source fps
    benchmark: str
    case_id: str
    camera_provenance: str
    calibrated: bool
    metric_geometry_valid: bool
    camera_units: str
    gauge: str = "source"

    def validate(self):
        image, k, poses, times = (
            self.first_frame,
            self.intrinsics,
            self.camera_c2w,
            self.times,
        )
        if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) < 1:
            raise ValueError("first_frame must contain exactly one [3,H,W] RGB image")
        if (
            not image.is_floating_point()
            or not torch.isfinite(image).all()
            or image.min() < 0
            or image.max() > 1
        ):
            raise ValueError("first_frame must be finite float RGB in [0,1]")
        if (
            times.ndim != 1
            or len(times) < 1
            or not torch.isfinite(times).all()
            or (times[1:] <= times[:-1]).any()
        ):
            raise ValueError("times must be a finite strictly increasing sequence")
        if k.shape != (len(times), 3, 3) or poses.shape != (len(times), 4, 4):
            raise ValueError("camera arrays must match the number of timestamps")
        if not torch.isfinite(k).all() or not torch.isfinite(poses).all():
            raise ValueError("camera geometry must be finite")
        if (
            (k[:, 0, 0] <= 0).any()
            or (k[:, 1, 1] <= 0).any()
            or (torch.linalg.det(k.float()).abs() < 1e-8).any()
        ):
            raise ValueError(
                "intrinsics must be invertible with positive focal lengths"
            )
        rotation = poses[:, :3, :3].float()
        identity = torch.eye(3, device=poses.device).expand_as(rotation)
        if not torch.allclose(
            rotation.transpose(-1, -2) @ rotation, identity, atol=2e-3, rtol=0
        ):
            raise ValueError("camera rotations must be orthonormal")
        if (torch.linalg.det(rotation) < 0.99).any() or not torch.allclose(
            poses[:, 3].float(),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=poses.device).expand(
                len(times), -1
            ),
            atol=1e-6,
        ):
            raise ValueError("camera_c2w must be proper homogeneous transforms")
        if self.metric_geometry_valid and (
            not self.calibrated or self.camera_units != "metres"
        ):
            raise ValueError(
                "metric geometry requires calibrated cameras and verified metre units"
            )
        if not self.camera_provenance or not self.camera_units:
            raise ValueError("camera provenance and units must be declared")


def prepare_input(
    source: BenchmarkInput, *, output_hw: tuple[int, int] | None = None
) -> BenchmarkInput:
    """Resize RGB/K and choose a gauge using only the first observed camera.

    No translation rescaling is inferred from later cameras or target images.
    Non-metric estimated/assumed camera units remain explicitly non-metric.
    """
    source.validate()
    image = source.first_frame.detach().float().cpu().clone()
    k, poses = (
        source.intrinsics.detach().float().cpu().clone(),
        source.camera_c2w.detach().float().cpu().clone(),
    )
    if output_hw is not None:
        height, width = output_hw
        if type(height) is not int or type(width) is not int or min(height, width) < 1:
            raise ValueError("output_hw must be positive integers")
        old_h, old_w = image.shape[-2:]
        image = F.interpolate(
            image[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].clamp(
            0, 1
        )  # FP32 antialias weights can overshoot by a few ulps.
        k[:, 0] *= width / old_w
        k[:, 1] *= height / old_h
    poses = torch.linalg.inv(poses[0]) @ poses
    result = replace(
        source,
        first_frame=image,
        intrinsics=k,
        camera_c2w=poses,
        times=source.times.detach().double().cpu().clone(),
        gauge="first_camera",
    )
    result.validate()
    return result


def read_memo_cameras(path: str | Path, *, real: bool) -> tuple[Tensor, Tensor]:
    """Read released camera files only; no RGB/phase/object annotation access.

    For synthetic, path contains intrinsics.npy and poses.npy. For real, path
    is the released MapAnything .npz. Official synthetic preprocessing uses
    c2w_UE @ flip_yz (OpenCV camera axes); real MapAnything output also uses
    OpenCV axes. No additional axis conversion is applied to these released
    arrays. Neither format alone establishes metre scale.
    """
    path = Path(path)
    if real:
        with np.load(path, allow_pickle=False) as data:
            k, poses = np.array(data["intrinsic"]), np.array(data["cam_c2w"])
    else:
        k, poses = np.load(path / "intrinsics.npy", allow_pickle=False), np.load(
            path / "poses.npy", allow_pickle=False
        )
    k, poses = torch.as_tensor(k, dtype=torch.float32), torch.as_tensor(
        poses, dtype=torch.float32
    )
    if k.shape == (4,) or (k.ndim == 2 and k.shape[-1] == 4):
        values = k.reshape(-1, 4)
        k = torch.eye(3).repeat(len(values), 1, 1)
        k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2] = values.unbind(-1)
    elif k.shape == (3, 3):
        k = k[None]
    if k.shape == (1, 3, 3):
        k = k.expand(len(poses), -1, -1).clone()
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4) or k.shape != (len(poses), 3, 3):
        raise ValueError("unsupported released camera array shapes")
    return k, poses


def camera_sequence(
    intrinsics: Tensor, camera_c2w: Tensor, *, output_frames: int, source_fps: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Uniform source-index sampling, preserving source duration and endpoints."""
    count = len(camera_c2w)
    if type(output_frames) is not int or not 2 <= output_frames <= count:
        raise ValueError("output_frames must be between 2 and source frame count")
    if (
        not math.isfinite(source_fps)
        or source_fps <= 0
        or intrinsics.shape != (count, 3, 3)
    ):
        raise ValueError("positive source_fps and per-frame intrinsics required")
    indices = (
        torch.linspace(0, count - 1, output_frames, dtype=torch.float64).round().long()
    )
    return (
        intrinsics[indices].clone(),
        camera_c2w[indices].clone(),
        indices.double() / source_fps,
        indices,
    )


def assumed_wrbench_input(
    first_frame: Tensor,
    camera_c2w: Tensor,
    times: Tensor,
    *,
    case_id: str,
    fov_degrees: float,
) -> BenchmarkInput:
    """Explicit uncalibrated camera prior for a no-event-conditioning baseline."""
    if not math.isfinite(fov_degrees) or not 0 < fov_degrees < 180:
        raise ValueError("fov_degrees must be finite and in (0,180)")
    height, width = first_frame.shape[-2:]
    focal = width / (2 * math.tan(math.radians(fov_degrees) / 2))
    k = torch.tensor(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
    )
    result = BenchmarkInput(
        first_frame,
        k.repeat(len(times), 1, 1),
        camera_c2w,
        times,
        "WRBench",
        case_id,
        f"assumed_pinhole_horizontal_fov_{fov_degrees:g}",
        False,
        False,
        "assumed_scene_units",
    )
    result.validate()
    return result


@torch.inference_mode()
def render_first_frame(
    model,
    prepared: BenchmarkInput,
    *,
    near: float,
    far: float,
    ray_chunk_size: int = 256,
    use_assumed_rays: bool = False,
) -> dict:
    """One Correct followed solely by Predict and camera-conditioned Read.

    Returns CPU RGB [F,3,H,W], without substituting the observed first image for
    the model's first render. Decoder chunking is additional to this outer cap.
    """
    prepared.validate()
    if prepared.gauge != "first_camera":
        raise ValueError("call prepare_input to establish first_camera gauge")
    if not prepared.calibrated and not use_assumed_rays:
        raise ValueError(
            "uncalibrated input requires explicit use_assumed_rays=True baseline opt-in"
        )
    if (
        type(ray_chunk_size) is not int
        or ray_chunk_size < 1
        or not 0 < near < far
        or not math.isfinite(far)
    ):
        raise ValueError("positive ray_chunk_size and finite 0 < near < far required")
    device = next(model.parameters()).device
    image = prepared.first_frame.to(device)
    k, poses, times = (
        prepared.intrinsics.to(device),
        prepared.camera_c2w.to(device),
        prepared.times.to(device),
    )
    height, width = image.shape[-2:]
    observation = ObservationStep(
        image[None, None, None],
        time=times[:1],
        intrinsics=k[:1, None],
        camera_c2w=poses[:1, None],
        # Core geometry_valid controls content writes. Opt-in ray support is an
        # approximation for this adapter, not a claim of measured calibration.
        geometry_valid=torch.tensor(
            [[prepared.calibrated or use_assumed_rays]], device=device
        ),
    )
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device).float() + 0.5,
        torch.arange(width, device=device).float() + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((xx, yy), -1).reshape(1, -1, 2)
    was_training = model.training
    model.eval()
    try:
        state = model.initial_state(1, device=device, time=times[:1])
        state = model(state, observation)
        frames = []
        for index in range(len(times)):
            if index:
                state = model.predict(state, times[index : index + 1] - state.t)
            chunks = []
            for start in range(0, pixels.shape[1], ray_chunk_size):
                pixel = pixels[:, start : start + ray_chunk_size]
                query = make_ray_query(
                    k[index : index + 1],
                    poses[index : index + 1],
                    pixel,
                    time=times[index : index + 1],
                    near=near,
                    far=far,
                    valid=torch.ones(pixel.shape[:2], dtype=torch.bool, device=device),
                )
                chunks.append(model.read(state, query)["rgb"].float().cpu())
            frame = torch.cat(chunks, 1)[0].reshape(height, width, 3).permute(2, 0, 1)
            if not torch.isfinite(frame).all():
                raise ValueError("model emitted nonfinite RGB; generation failed")
            frames.append(frame)
    finally:
        model.train(was_training)
    provenance = {
        name: getattr(prepared, name)
        for name in (
            "benchmark",
            "case_id",
            "camera_provenance",
            "calibrated",
            "metric_geometry_valid",
            "camera_units",
            "gauge",
        )
    }
    provenance.update(
        observed_frames=1,
        future_camera_conditioned_readout=True,
        future_visual_evidence=False,
        event_conditioning=False,
        content_constant_during_rollout=True,
        ray_geometry_assumed=not prepared.calibrated,
        assumed_ray_content_write=not prepared.calibrated and use_assumed_rays,
        height=height,
        width=width,
        frame_count=len(times),
        times_seconds=times.cpu().tolist(),
        near=near,
        far=far,
        first_frame_output="model_render",
    )
    return {"rgb": torch.stack(frames), "provenance": provenance}
