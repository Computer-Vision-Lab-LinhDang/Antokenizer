"""Controlled temporal events with identical terminal observations.

The four cores describe motion order, never colour or another appearance label.
Each source seed is crossed with every core and configuration. Reversing an
event reuses exactly the same poses in reverse order; all four cores end at the
same visible pose for two frames. Thus neither the final tubelet nor unordered
frame pooling can solve each reversed pair.

This is a small diagnostic for retaining temporal structure, not a claim that
these labels capture the essence of arbitrary natural events. A static camera
observes a textured 3D box and ground plane, using the existing ray caster.
There is no LiDAR, occluder, optical flow, or model-specific preprocessing.

Use ``observation_prefix`` or ``model_inputs`` for model inputs. Scene IDs,
source seeds, labels, trajectories, boxes, and pixel ground truth are audit or
evaluation information and are excluded from those adapters.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mavt.data.splits import assign
from mavt4d.contracts import ObservationStep
from mavt4d.data.sample import Boxes3D, SceneSample
from mavt4d.data.synthetic import (
    SKY,
    SyntheticGT,
    SyntheticSceneConfig,
    _camera_rotations,
    _raycast,
    texture,
)
from mavt4d.geometry.boxes import world_to_box


EVENT_CORES = ("left_then_right", "right_then_left", "clockwise", "counterclockwise")
EVENT_CONFIGURATIONS = ("compact", "wide")
EVENT_VERSION = "event-cd-v1"


@dataclass(frozen=True)
class EventSceneConfig:
    """Versioned single-camera diagnostic; physical steps may use two frames.

    ``compact`` places the terminal box at x=7.5 m and uses a 0.9 m path
    radius; ``wide`` uses x=10 m and radius 1.4 m. Both are crossed with every
    core. Colour, camera height, and camera pitch depend only on source seed.
    Validation sources use the repository's salted grouped hash, not an RNG.
    """

    version: str = EVENT_VERSION
    T: int = 12
    H: int = 64
    W: int = 96
    dt: float = 1 / 12
    val_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.version != EVENT_VERSION:
            raise ValueError(f"version: expected {EVENT_VERSION!r}")
        if type(self.T) is not int or self.T < 8 or self.T % 2:
            raise ValueError("T: expected an even integer >= 8")
        for name, minimum in (("H", 16), ("W", 24)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name}: expected an integer >= {minimum}")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt: expected finite positive seconds")
        if not math.isfinite(self.val_fraction) or not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction: expected a finite value in [0, 1)")


@dataclass(frozen=True)
class EventKey:
    """Audit-only index; every variant from one source has the same group."""

    source_seed: int
    core: str
    configuration: str
    group_key: str
    split: str


@dataclass(frozen=True)
class EventEvaluation:
    """Labels and geometry for evaluation; never passed to the model adapter."""

    version: str
    source_seed: int
    group_key: str
    split: str
    core: str
    configuration: str
    positions: Tensor  # [T, 3], world-space target centres, metres
    boxes: tuple[Boxes3D, ...]
    gt: SyntheticGT
    target_instance_id: int = 1


@dataclass(frozen=True)
class EventFixture:
    """Compatible camera-only SceneSample paired with separate evaluation GT."""

    scene: SceneSample
    evaluation: EventEvaluation

    def observation_prefix(
        self,
        stop_frame: int,
        availability: Tensor | None = None,
        frames_per_step: int = 2,
    ) -> tuple[ObservationStep, ...]:
        """Copy frames ``[0, stop_frame)`` into causal, observation-only steps.

        Availability is boolean [V,T,H,W] for the full scene or
        [V,stop_frame,H,W] for this prefix. Unavailable RGB is replaced by zero
        before returning; the availability mask is retained separately. Every
        tensor is a compact copy, so serialization cannot expose future frames
        through a tensor view's backing storage. The last step may be partial;
        it carries real frames only, without implicit padding.
        """
        views, total, _, height, width = self.scene.images.shape
        if type(stop_frame) is not int or not 1 <= stop_frame <= total:
            raise ValueError("stop_frame: expected an integer in [1, T]")
        if type(frames_per_step) is not int or frames_per_step <= 0:
            raise ValueError("frames_per_step: expected a positive integer")
        if availability is None:
            availability = torch.ones(views, stop_frame, height, width, dtype=torch.bool,
                                      device=self.scene.images.device)
        elif (
            not isinstance(availability, Tensor)
            or availability.dtype != torch.bool
            or tuple(availability.shape) not in {
                (views, total, height, width), (views, stop_frame, height, width),
            }
            or availability.device != self.scene.images.device
        ):
            raise ValueError("availability: expected boolean [V,T,H,W] or [V,stop_frame,H,W] on image device")

        steps = []
        for start in range(0, stop_frame, frames_per_step):
            stop = min(start + frames_per_step, stop_frame)
            observed = availability[:, start:stop].contiguous().clone()
            rgb = self.scene.images[:, start:stop]
            images = torch.where(observed[:, :, None], rgb, torch.zeros_like(rgb)).contiguous().clone()
            intrinsics = self.scene.intrinsics
            poses = self.scene.cam_to_world
            steps.append(ObservationStep(
                images=images,
                observed=observed,
                frame_times=self.scene.frame_times[:, start:stop].contiguous().clone(),
                intrinsics=None if intrinsics is None else intrinsics.contiguous().clone(),
                cam_to_world=None if poses is None else poses[:, start:stop].contiguous().clone(),
            ))
        return tuple(steps)


def _key(cfg: EventSceneConfig, source_seed: int, core: str, configuration: str) -> EventKey:
    if type(source_seed) is not int or not 0 <= source_seed < 2**63:
        raise ValueError("source_seed: expected an integer in [0, 2**63)")
    if core not in EVENT_CORES:
        raise ValueError(f"core: expected one of {EVENT_CORES}")
    if configuration not in EVENT_CONFIGURATIONS:
        raise ValueError(f"configuration: expected one of {EVENT_CONFIGURATIONS}")
    # No version/core/config suffix: adding variants never reassigns a source.
    group = f"synthetic-event:source-{source_seed}"
    return EventKey(source_seed, core, configuration, group, assign(group, cfg.val_fraction))


def _positions(cfg: EventSceneConfig, core: str, configuration: str) -> Tensor:
    """Exact reversed pose sequences plus a repeated, visible two-frame end."""
    anchor_x, radius = (7.5, 0.9) if configuration == "compact" else (10.0, 1.4)
    phase = torch.linspace(0, 2 * math.pi, cfg.T - 2, dtype=torch.float64)
    positions = torch.zeros(cfg.T - 2, 3, dtype=torch.float64)
    positions[:, 0] = anchor_x
    positions[:, 1] = radius * torch.sin(phase)
    positions[:, 2] = 0.7
    if core in ("clockwise", "counterclockwise"):
        # Viewed from +z, starting at the leftmost point and moving toward +y
        # traces a clockwise circle in the world x/y plane.
        positions[:, 0] += radius * (1 - torch.cos(phase))
    origin = torch.tensor([anchor_x, 0.0, 0.7], dtype=torch.float64)
    positions[0] = origin
    positions[-1] = origin  # do not let sin(2*pi) rounding encode direction
    if core in ("right_then_left", "counterclockwise"):
        positions = positions.flip(0)
    return torch.cat((positions, origin.repeat(2, 1)), dim=0)


def make_event_fixture(
    cfg: EventSceneConfig,
    source_seed: int,
    core: str,
    configuration: str,
) -> EventFixture:
    """Render one deterministic 3D temporal diagnostic without global RNG use."""
    key = _key(cfg, source_seed, core, configuration)
    # CPU torch RNG streams can collide for seeds separated by 2**32 while
    # their source groups land in different splits. Hash the full accepted
    # seed separately for each nuisance value instead of reseeding that RNG.
    nuisance_values = []
    for index in range(5):
        digest = hashlib.sha256(f"{cfg.version}\0{source_seed}\0nuisance\0{index}".encode()).digest()
        nuisance_values.append((int.from_bytes(digest[:8], "big") >> 11) / 2**53)
    nuisance = torch.tensor(nuisance_values, dtype=torch.float64)
    texture_shift = (nuisance[:3] * 4 - 2).float()
    camera_height = 1.6 + 0.2 * float(nuisance[3])
    camera_pitch = 7.0 + 2.0 * float(nuisance[4])

    positions64 = _positions(cfg, core, configuration)
    size = torch.tensor([[1.4, 1.4, 1.4]], dtype=torch.float64)
    yaw = torch.zeros(1, dtype=torch.float64)
    instance_id = torch.ones(1, dtype=torch.int64)
    boxes64 = tuple(Boxes3D(p[None], size, yaw, instance_id) for p in positions64)
    positions = positions64.float()
    boxes = tuple(Boxes3D(positions[t:t + 1].clone(), size.float(), yaw.float(), instance_id.clone())
                  for t in range(cfg.T))

    focal = (cfg.W / 2) / math.tan(math.radians(68.0) / 2)
    intrinsic = torch.tensor([[focal, 0, cfg.W / 2], [0, focal, cfg.H / 2], [0, 0, 1]], dtype=torch.float64)
    camera_cfg = SyntheticSceneConfig(V=1, cam_pitch_deg=camera_pitch)
    rotation = _camera_rotations(camera_cfg)[0]
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = torch.tensor([0, 0, camera_height], dtype=torch.float64)
    poses = pose[None, None].repeat(1, cfg.T, 1, 1)
    times = (torch.arange(cfg.T, dtype=torch.float64) * cfg.dt)[None]

    columns, rows = torch.meshgrid(torch.arange(cfg.W, dtype=torch.float64),
                                   torch.arange(cfg.H, dtype=torch.float64), indexing="xy")
    pixels = torch.stack((columns + 0.5, rows + 0.5, torch.ones_like(columns)), dim=-1).reshape(-1, 3)
    rays = (pixels @ torch.linalg.inv(intrinsic).T) @ rotation.T
    origins = pose[:3, 3].expand_as(rays)
    count = cfg.H * cfg.W
    images = torch.tensor(SKY).view(1, 1, 3).repeat(cfg.T, count, 1)
    world = torch.zeros(cfg.T, count, 3, dtype=torch.float32)
    depth = torch.zeros(cfg.T, count, dtype=torch.float32)
    instance = torch.zeros(cfg.T, count, dtype=torch.int64)
    valid = torch.zeros(cfg.T, count, dtype=torch.bool)
    for t, box in enumerate(boxes64):
        hit_depth, hit_instance = _raycast(origins, rays, box)
        seen = hit_instance >= 0
        points = origins[seen] + hit_depth[seen, None] * rays[seen]
        world[t, seen] = points.float()
        depth[t, seen] = hit_depth[seen].float()
        instance[t] = hit_instance.clamp(min=0)
        valid[t] = seen
        object_points = points.float()
        target = hit_instance[seen] == 1
        object_points[target] = world_to_box(object_points[target], boxes[t].center[0], boxes[t].yaw[0])
        images[t, seen] = texture(object_points + texture_shift, hit_instance[seen])

    scene = SceneSample(
        scene_id=f"{cfg.version}:source-{source_seed}:{core}:{configuration}",
        source="synthetic",
        images=images.reshape(1, cfg.T, cfg.H, cfg.W, 3).permute(0, 1, 4, 2, 3).contiguous(),
        frame_times=times,
        intrinsics=intrinsic.float()[None],
        cam_to_world=poses.float(),
    )
    gt = SyntheticGT(
        world_xyz=world.reshape(1, cfg.T, cfg.H, cfg.W, 3),
        depth=depth.reshape(1, cfg.T, cfg.H, cfg.W),
        instance=instance.reshape(1, cfg.T, cfg.H, cfg.W),
        valid=valid.reshape(1, cfg.T, cfg.H, cfg.W),
    )
    evaluation = EventEvaluation(cfg.version, source_seed, key.group_key, key.split, core,
                                 configuration, positions, boxes, gt)
    return EventFixture(scene, evaluation)


class EventSyntheticDataset(Dataset):
    """On-demand full crossing with an explicit grouped train/val selection.

    ``__getitem__`` returns the fixture including evaluation targets. Training
    code should call ``model_inputs``; it returns only the declared observation
    interface. ``keys`` is cheap audit metadata and performs no rendering.
    """

    def __init__(self, cfg: EventSceneConfig, source_seeds: Sequence[int], split: str) -> None:
        if split not in ("train", "val"):
            raise ValueError("split: expected 'train' or 'val'")
        seeds = tuple(source_seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("source_seeds: duplicate source seeds are not allowed")
        keys = tuple(_key(cfg, seed, core, configuration)
                     for seed, core, configuration in product(seeds, EVENT_CORES, EVENT_CONFIGURATIONS))
        self.cfg = cfg
        self.split = split
        self.keys = tuple(key for key in keys if key.split == split)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> EventFixture:
        key = self.keys[index]
        return make_event_fixture(self.cfg, key.source_seed, key.core, key.configuration)

    def model_inputs(
        self, index: int, stop_frame: int, availability: Tensor | None = None, frames_per_step: int = 2,
    ) -> tuple[ObservationStep, ...]:
        return self[index].observation_prefix(stop_frame, availability, frames_per_step)
