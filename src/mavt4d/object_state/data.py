"""Procedural rigid factorial scenes, with GT kept outside model observations.

Each source contains eight shape x appearance x motion interventions. All motion
pairs meet at the last observed frame, so the current image cannot identify
velocity. Cameras use OpenCV optical axes; depth is camera z, in metres. This
internal synthetic suite is not WRBench or MemoBench.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch
from torch import Tensor
from torch.utils.data import Dataset, default_collate

from mavt4d.data.sample import Boxes3D
from mavt4d.data.synthetic import _raycast, texture
from mavt4d.geometry.boxes import rot_z, world_to_box


@dataclass(frozen=True)
class FactorialSceneConfig:
    frames: int = 12
    prefix_frames: int = 6
    views: int = 2
    observed_views: int = 1
    height: int = 128
    width: int = 192
    max_objects: int = 3
    dt: float = 0.12
    fov_degrees: float = 65.0
    near: float = 0.1
    far: float = 30.0
    gauge: str = "first_camera"

    def __post_init__(self):
        for name in ("frames", "prefix_frames", "views", "observed_views", "height", "width", "max_objects"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.prefix_frames >= self.frames or self.observed_views > self.views:
            raise ValueError("prefix must leave forecast frames and observed views must exist")
        if not all(math.isfinite(x) for x in (self.dt, self.fov_degrees, self.near, self.far)):
            raise ValueError("camera and time parameters must be finite")
        if self.dt <= 0 or not 10 < self.fov_degrees < 150 or not 0 < self.near < self.far:
            raise ValueError("invalid camera or time parameters")
        if self.gauge not in ("first_camera", "world"):
            raise ValueError("gauge must be first_camera or world")


def _source_seed(split: str, base_seed: int, index: int) -> int:
    key = f"mavt-object-factorial-v1/{split}/{base_seed}/{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def _camera_sequence(cfg: FactorialSceneConfig, times: Tensor, phase: float):
    f = cfg.width / (2 * math.tan(math.radians(cfg.fov_degrees) / 2))
    k = torch.tensor([[f, 0, cfg.width / 2], [0, f, cfg.height / 2], [0, 0, 1.]], dtype=torch.float32)
    cameras = torch.eye(4, dtype=torch.float32).repeat(cfg.frames, cfg.views, 1, 1)
    up = torch.tensor([0., 0., 1.], dtype=torch.float32)
    look_at = torch.tensor([5.7, 0., .65], dtype=torch.float32)
    for t, time in enumerate(times):
        for v in range(cfg.views):
            origin = torch.tensor([.12 * float(time), (v - (cfg.views - 1) / 2) * 1.1
                                   + .15 * math.sin(float(time) + phase), 1.7], dtype=torch.float32)
            forward = torch.nn.functional.normalize(look_at - origin, dim=0)
            right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=0)
            cameras[t, v, :3, :3] = torch.stack((right, torch.linalg.cross(forward, right), forward), -1)
            cameras[t, v, :3, 3] = origin
    return k.expand(cfg.frames, cfg.views, 3, 3).clone(), cameras


def _scene_parameters(cfg: FactorialSceneConfig, seed: int):
    g = torch.Generator().manual_seed(seed)
    def uniform(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(shape, generator=g, dtype=torch.float32)
    n = int(torch.randint(1, cfg.max_objects + 1, (), generator=g))
    centres = torch.zeros(n, 3, dtype=torch.float32)
    centres[:, 0] = uniform(4.8, 6.8, n)
    centres[:, 1] = torch.linspace(-1.1, 1.1, n) if n > 1 else uniform(-.4, .4, n)
    sizes = uniform(.8, 1.3, n, 3)  # w,l,h; height unchanged by shape interventions
    centres[:, 2] = sizes[:, 2] / 2
    yaw = uniform(-.6, .6, n)
    velocity = torch.stack((uniform(-.25, .25, n), uniform(.65, 1.2, n), torch.zeros(n)), -1)
    yaw_rate = uniform(-.3, .3, n)
    if n > 1:
        velocity[1] = 0
        yaw_rate[1] = 0
    if n > 2:  # foreground occluder; re-entry depends on the moving object's path
        centres[-1, :2] = torch.tensor([4., .15])
        velocity[-1] = 0
        yaw_rate[-1] = 0
    return centres, sizes, yaw, velocity, yaw_rate, float(uniform(0, 2 * math.pi))


def _render_geometry(cfg, centres, sizes, yaw, velocity, yaw_rate, times, intrinsics, cameras):
    t_count, v_count, h, w = cfg.frames, cfg.views, cfg.height, cfg.width
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32) + .5,
                            torch.arange(w, dtype=torch.float32) + .5, indexing="ij")
    pixel = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
    camera_rays = pixel @ torch.linalg.inv(intrinsics[0, 0]).T
    world = torch.zeros(t_count, v_count, h * w, 3, dtype=torch.float32)
    local = torch.zeros_like(world)
    depth = torch.zeros(t_count, v_count, h * w, dtype=torch.float32)
    instance = torch.full((t_count, v_count, h * w), -1, dtype=torch.int64)
    poses = torch.eye(4, dtype=torch.float32).repeat(t_count, cfg.max_objects, 1, 1)
    n = len(centres)
    relative_times = times - times[cfg.prefix_frames - 1]
    for t, tau in enumerate(relative_times):
        centre_t, yaw_t = centres + velocity * tau, yaw + yaw_rate * tau
        poses[t, :n, :3, :3] = rot_z(yaw_t)
        poses[t, :n, :3, 3] = centre_t
        boxes = Boxes3D(centre_t, sizes, yaw_t, torch.arange(1, n + 1))
        for v in range(v_count):
            origin = cameras[t, v, :3, 3].expand(h * w, 3)
            direction = camera_rays @ cameras[t, v, :3, :3].T
            hit_z, hit_instance = _raycast(origin, direction, boxes)
            ray_range = hit_z * direction.norm(dim=-1)
            valid = (hit_instance >= 0) & (ray_range >= cfg.near) & (ray_range <= cfg.far)
            instance[t, v] = torch.where(valid, hit_instance, -1)
            depth[t, v, valid] = hit_z[valid]
            world[t, v, valid] = origin[valid] + hit_z[valid, None] * direction[valid]
            local[t, v] = world[t, v]
            for i in range(n):
                select = instance[t, v] == i + 1
                local[t, v, select] = world_to_box(world[t, v, select], centre_t[i], yaw_t[i])
    return local.float(), depth.float(), instance, poses.float()


def _colour(local, instance, appearance: int):
    flat_instance, points = instance.flatten(), local.reshape(-1, 3)
    # Ground stays the same; only object appearance changes under this intervention.
    texture_ids = flat_instance.clamp_min(0) + (flat_instance > 0).long() * appearance * 7
    colour = (texture(points, texture_ids) + 1) / 2
    sky = torch.tensor([.42, .58, .78], dtype=colour.dtype)
    colour = torch.where((flat_instance >= 0)[:, None], colour, sky)
    return colour.reshape(*instance.shape, 3)


def _render_source(cfg: FactorialSceneConfig, seed: int, split_code: int, source_index: int):
    centres, sizes, yaw, base_velocity, base_yaw_rate, phase = _scene_parameters(cfg, seed)
    times = torch.arange(cfg.frames, dtype=torch.float64) * cfg.dt
    intrinsics, cameras = _camera_sequence(cfg, times, phase)
    # Gauge is fixed by the first observed camera, never by a future frame or GT object.
    world_to_gauge = (torch.linalg.inv(cameras[0, 0]) if cfg.gauge == "first_camera"
                      else torch.eye(4, dtype=cameras.dtype))
    output_cameras = world_to_gauge @ cameras
    variants = {}
    n = len(centres)
    for shape in range(2):
        shape_sizes = sizes.clone()
        if shape:
            shape_sizes[:, 0] *= .7
            shape_sizes[:, 1] *= 1.35
        padded_sizes = torch.zeros(cfg.max_objects, 3)
        padded_sizes[:n] = shape_sizes.float()
        for motion in range(2):
            velocity, yaw_rate = base_velocity * (1 - 2 * motion), base_yaw_rate * (1 - 2 * motion)
            local, depth, instance, poses = _render_geometry(
                cfg, centres, shape_sizes, yaw, velocity, yaw_rate, times, intrinsics, cameras)
            padded_velocity = torch.zeros(cfg.frames, cfg.max_objects, 3)
            padded_velocity[:, :n] = velocity.float()
            poses = world_to_gauge @ poses
            padded_velocity = padded_velocity @ world_to_gauge[:3, :3].T
            # Existence, not visibility: an occluded object remains a valid GT object.
            object_valid = torch.arange(cfg.max_objects)[None].expand(cfg.frames, -1) < n
            for appearance in range(2):
                variant = 4 * shape + 2 * appearance + motion
                rgb = _colour(local, instance, appearance).reshape(
                    cfg.frames, cfg.views, cfg.height, cfg.width, 3).permute(0, 1, 4, 2, 3).contiguous()
                depth_image = depth.reshape(cfg.frames, cfg.views, 1, cfg.height, cfg.width)
                inst_image = instance.reshape(cfg.frames, cfg.views, cfg.height, cfg.width)
                observations = {
                    "rgb": rgb[:cfg.prefix_frames, :cfg.observed_views].clone(),
                    "valid": torch.ones(cfg.prefix_frames, cfg.observed_views, 1, cfg.height, cfg.width, dtype=torch.bool),
                    "intrinsics": intrinsics[:cfg.prefix_frames, :cfg.observed_views].float().clone(),
                    "camera_c2w": output_cameras[:cfg.prefix_frames, :cfg.observed_views].float().clone(),
                    "times": times[:cfg.prefix_frames].clone(),
                    "geometry_valid": torch.ones(cfg.prefix_frames, cfg.observed_views, dtype=torch.bool),
                }
                targets = {
                    "rgb": rgb, "depth": depth_image, "depth_valid": (inst_image >= 0).unsqueeze(2),
                    "instance": inst_image, "object_poses": poses.clone(),
                    "object_sizes": padded_sizes.clone(), "object_velocity": padded_velocity.clone(),
                    "object_valid": object_valid.clone(), "intrinsics": intrinsics.float().clone(),
                    "camera_c2w": output_cameras.float().clone(), "times": times.clone(),
                    "near": torch.tensor(cfg.near), "far": torch.tensor(cfg.far),
                }
                metadata = {"source_uid": seed, "source_index": source_index, "variant_index": variant,
                            "shape_factor": shape, "appearance_factor": appearance, "motion_factor": motion,
                            "split_code": split_code, "cutoff_index": cfg.prefix_frames - 1,
                            "gauge_code": int(cfg.gauge == "first_camera")}
                variants[variant] = {"observations": observations, "targets": targets,
                                     "metadata": {k: torch.tensor(v, dtype=torch.int64) for k, v in metadata.items()}}
    return default_collate([variants[i] for i in range(8)])


class FactorialSceneDataset(Dataset):
    """One item is an entire eight-variant source group, rendered on CPU.

    Train/dev/test source seeds are disjoint by construction. This is a source
    split, not a claim that every shape or motion primitive is held out.
    """
    def __init__(self, n_sources: int, config: FactorialSceneConfig | None = None,
                 *, split: str = "train", base_seed: int = 17):
        if type(n_sources) is not int or n_sources < 1 or split not in ("train", "dev", "test"):
            raise ValueError("positive source count and train/dev/test split required")
        self.n_sources, self.config, self.split, self.base_seed = n_sources, config or FactorialSceneConfig(), split, base_seed

    def __len__(self):
        return self.n_sources

    def __getitem__(self, index):
        if not 0 <= index < self.n_sources:
            raise IndexError(index)
        return _render_source(self.config, _source_seed(self.split, self.base_seed, index),
                              ("train", "dev", "test").index(self.split), index)


def _tensor_tree_digest(sample) -> str:
    """Content hash independent of torch ZIP serialization and storage IDs."""
    digest = hashlib.sha256()
    def visit(value, path):
        if isinstance(value, dict):
            for name in sorted(value):
                visit(value[name], path + "/" + name)
        elif isinstance(value, Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(json.dumps([path, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.numpy().tobytes())
        else:
            raise ValueError("cache sample must contain tensors only")
    visit(sample, "")
    return digest.hexdigest()


class CachedFactorialDataset(FactorialSceneDataset):
    """Immutable per-source cache, safe for concurrent preparation and training.

    A missing source is rendered on demand. Publication uses an exclusive hard
    link from a unique, fsynced temporary file in the same directory; a racing
    worker cannot overwrite an existing source. Invalid existing files raise
    instead of being silently replaced. No global sample cache is retained.
    """
    CACHE_SCHEMA = "mavt-object-factorial-fp32-v2"

    def __init__(self, data_dir, n_sources: int, config: FactorialSceneConfig | None = None,
                 *, split: str = "train", base_seed: int = 17):
        super().__init__(n_sources, config, split=split, base_seed=base_seed)
        self.contract = {"schema": self.CACHE_SCHEMA, "config": asdict(self.config),
                         "split": self.split, "base_seed": self.base_seed}
        encoded = json.dumps(self.contract, sort_keys=True, separators=(",", ":")).encode()
        self.fingerprint = hashlib.sha256(encoded).hexdigest()
        self.cache_dir = Path(data_dir) / self.split / self.fingerprint

    def cache_path(self, index: int) -> Path:
        if not 0 <= index < self.n_sources:
            raise IndexError(index)
        return self.cache_dir / f"source-{index:08d}.pt"

    def _load(self, path: Path, index: int):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict) or payload.get("contract") != self.contract:
            raise ValueError(f"cache contract mismatch: {path.name}")
        if payload.get("fingerprint") != self.fingerprint or payload.get("source_index") != index:
            raise ValueError(f"cache source/fingerprint mismatch: {path.name}")
        if payload.get("source_seed") != _source_seed(self.split, self.base_seed, index):
            raise ValueError(f"cache source seed mismatch: {path.name}")
        sample = payload.get("sample")
        if not isinstance(sample, dict) or payload.get("sample_sha256") != _tensor_tree_digest(sample):
            raise ValueError(f"cache content digest mismatch: {path.name}")
        return sample

    def __getitem__(self, index):
        path = self.cache_path(index)
        if path.exists():
            return self._load(path, index)
        sample = super().__getitem__(index)
        payload = {"contract": self.contract, "fingerprint": self.fingerprint,
                   "source_index": index, "source_seed": _source_seed(self.split, self.base_seed, index),
                   "sample_sha256": _tensor_tree_digest(sample), "sample": sample}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        won_publication = False
        try:
            with temporary.open("xb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)  # atomic publish-if-absent; never replaces
                won_publication = True
            except FileExistsError:
                pass
        finally:
            # Only this call's exclusively created temporary name is removed.
            if temporary.exists():
                temporary.unlink()
        return sample if won_publication else self._load(path, index)


def flatten_source_batch(batch):
    """Merge DataLoader's [source,variant,...] into model batch [B,...]."""
    if isinstance(batch, dict):
        return {k: flatten_source_batch(v) for k, v in batch.items()}
    if not isinstance(batch, Tensor) or batch.ndim < 2 or batch.shape[1] != 8:
        raise ValueError("expected a collated source batch with eight variants")
    return batch.flatten(0, 1)


def observation_step(batch, end_frame: int, tubelet_size: int = 2):
    """Build encoder input from prefix RGB/cameras only; no target field is read."""
    from .observation import ObservationStep
    obs = batch["observations"]
    if not 0 <= end_frame < obs["rgb"].shape[1]:
        raise ValueError("observation endpoint must lie inside the observed prefix")
    if tubelet_size not in (1, 2):
        raise ValueError("tubelet_size must be one or two")
    start = max(0, end_frame + 1 - tubelet_size)
    return ObservationStep(
        images=obs["rgb"][:, start:end_frame + 1].transpose(1, 2).contiguous(),
        valid=obs["valid"][:, start:end_frame + 1].transpose(1, 2).contiguous(),
        time=obs["times"][:, end_frame], intrinsics=obs["intrinsics"][:, end_frame],
        camera_c2w=obs["camera_c2w"][:, end_frame], geometry_valid=obs["geometry_valid"][:, end_frame],
    )


def patch_instance_targets(batch, end_frame: int, patch_size: int = 16):
    """Loss-only instance labels in encoder endpoint patch order.

    Reads GT only after the model step. Future frames and held-out views are
    deliberately ineligible for observation association supervision.
    """
    obs = batch["observations"]
    if not 0 <= end_frame < obs["rgb"].shape[1]:
        raise ValueError("association targets must lie inside the observed prefix")
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError("patch_size must be positive")
    views, h, w = obs["rgb"].shape[2], obs["rgb"].shape[-2], obs["rgb"].shape[-1]
    device = obs["rgb"].device
    yy = (torch.arange(math.ceil(h / patch_size), device=device) * patch_size + patch_size // 2).clamp(max=h - 1)
    xx = (torch.arange(math.ceil(w / patch_size), device=device) * patch_size + patch_size // 2).clamp(max=w - 1)
    labels = batch["targets"]["instance"][:, end_frame, :views][:, :, yy][:, :, :, xx]
    valid = obs["valid"][:, end_frame, :, 0][:, :, yy][:, :, :, xx]
    return labels.flatten(1), valid.flatten(1)


def sample_ray_targets(batch, frame: int, n_rays: int | None, *, generator=None, view_indices=None):
    """Sample equal foreground/background ray counts when both are present.

    Samples are with replacement, so sparse foreground stays trainable. Set
    n_rays=None for all pixels; full-frame evaluation should use that option.
    Depth is converted to metric range to match unit-direction RayQuery.
    """
    from .decoder import RayQuery
    target = batch["targets"]
    if not 0 <= frame < target["rgb"].shape[1]:
        raise ValueError("query frame lies outside target sequence")
    rgb = target["rgb"][:, frame]
    b, v, _, h, w = rgb.shape
    if n_rays is not None and (type(n_rays) is not int or n_rays < 1):
        raise ValueError("n_rays must be positive or None")
    device = rgb.device
    views = torch.arange(v, device=device) if view_indices is None else torch.as_tensor(view_indices, device=device)
    if views.dtype not in (torch.int32, torch.int64) or views.ndim != 1 or not len(views) or ((views < 0) | (views >= v)).any():
        raise ValueError("invalid query views")
    pool = (views[:, None] * (h * w) + torch.arange(h * w, device=device)[None]).flatten().long()
    instance_all = target["instance"][:, frame].flatten(1)
    if n_rays is None:
        ids = pool.expand(b, -1)
    else:
        selected = []
        for i in range(b):
            foreground = pool[instance_all[i, pool] > 0]
            background = pool[instance_all[i, pool] <= 0]
            if len(foreground) and len(background):
                nf = n_rays // 2
                ids_i = torch.cat([foreground[torch.randint(len(foreground), (nf,), device=device, generator=generator)],
                                   background[torch.randint(len(background), (n_rays - nf,), device=device, generator=generator)]])
            else:
                ids_i = pool[torch.randint(len(pool), (n_rays,), device=device, generator=generator)]
            selected.append(ids_i)
        ids = torch.stack(selected)
    view_id, pixel_id = ids // (h * w), ids % (h * w)
    u, row = (pixel_id % w).float() + .5, (pixel_id // w).float() + .5
    bi = torch.arange(b, device=device)[:, None]
    # The training caller may be inside BF16 autocast. Calibration, range and
    # unit-ray normalization retain metric precision independently of AMP.
    with torch.autocast(device_type=device.type, enabled=False):
        k = target["intrinsics"][:, frame][bi, view_id].float()
        cameras = target["camera_c2w"][:, frame][bi, view_id].float()
        pixels = torch.stack([u, row, torch.ones_like(u)], -1).float()
        ray_camera = torch.linalg.solve(k, pixels[..., None]).squeeze(-1)
        direction = (cameras[..., :3, :3] @ ray_camera[..., None]).squeeze(-1)
        norm = direction.norm(dim=-1)
        direction = direction / norm[..., None]
        origins = cameras[..., :3, 3]
    flat_rgb = rgb.permute(0, 1, 3, 4, 2).reshape(b, v * h * w, 3)
    depth = target["depth"][:, frame].flatten(1)[bi, ids]
    valid_depth = target["depth_valid"][:, frame].flatten(1)[bi, ids]
    valid = torch.ones_like(valid_depth)
    return {"query": RayQuery(origins=origins, directions=direction,
                              near=target["near"][:, None].expand_as(depth), far=target["far"][:, None].expand_as(depth),
                              time=target["times"][:, frame], valid=valid),
            "rgb": flat_rgb[bi, ids], "depth": depth, "range": depth * norm,
            "depth_valid": valid_depth, "instance": instance_all[bi, ids], "valid": valid,
            "object_poses": target["object_poses"][:, frame], "object_valid": target["object_valid"][:, frame],
            "pixel_indices": ids, "view_indices": view_id}


def paired_variant_indices(batch):
    """Indices for C_a,D_b -> known target; slot alignment remains loss-side.

    Donor flips all three factors, while target keeps receiver shape/appearance
    and donor motion. Caller must match learned slots; ordering is not oracle
    association input to the model.
    """
    meta = batch["metadata"]
    lookup = {(int(s), int(v)): i for i, (s, v) in enumerate(zip(meta["source_uid"], meta["variant_index"]))}
    if len(lookup) != len(meta["source_uid"]):
        raise ValueError("duplicate source variants in paired batch")
    content, dynamic, targets = [], [], []
    for (source, variant), i in lookup.items():
        if (source, variant ^ 7) not in lookup or (source, variant ^ 1) not in lookup:
            raise ValueError("paired intervention requires a complete source group")
        content.append(i); dynamic.append(lookup[source, variant ^ 7]); targets.append(lookup[source, variant ^ 1])
    return {k: torch.tensor(v, device=meta["source_uid"].device, dtype=torch.int64)
            for k, v in (("content", content), ("dynamic", dynamic), ("target", targets))}
