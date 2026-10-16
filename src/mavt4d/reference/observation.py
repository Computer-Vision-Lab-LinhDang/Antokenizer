"""Stateless shared Patchify and current-step sparse observation attention.

Local attention gathers a fixed spatial stencil directly: O(N k) attention
scores, not an N-by-N graph followed by masking. The separate learned-query
pool costs O(K N), where K is the number of C/D evidence slots. Neither part
reads previous state, retains patch history, nor claims linear cost in K.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.contracts import CDStateSpec, ObservationStep


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name}: expected {'nonnegative' if allow_zero else 'positive'} integer")


def _availability_weights(scores: Tensor, coverage: Tensor) -> Tensor:
    """Normalized attention with exact zeros, including an all-missing row."""
    available = coverage > 0
    scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
    weights = scores.softmax(dim=-1) * coverage
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)


def _local_neighbor_attention(
    queries: Tensor, keys: Tensor, values: Tensor, coverage: Tensor, *, radius: int
) -> Tensor:
    """Attend within each view's square spatial stencil, never across views.

    Q/K/V are [V,Hpatch,Wpatch,E], coverage is [V,Hpatch,Wpatch]. Padding
    neighbors have zero availability. Stacked temporary arrays have a stencil
    axis of (2r+1)^2 rather than a second patch axis. This small reference
    implementation favors inspectability over a fused CUDA kernel.
    """
    _positive_int("radius", radius, allow_zero=True)
    if queries.ndim != 4 or keys.shape != queries.shape or values.shape != queries.shape:
        raise ValueError("local attention: Q/K/V must have equal [V,H,W,E] shapes")
    if coverage.shape != queries.shape[:-1]:
        raise ValueError("local attention: coverage must have [V,H,W] shape")
    _, height, width, channels = queries.shape
    padded_keys = F.pad(keys.permute(0, 3, 1, 2), (radius,) * 4)
    padded_values = F.pad(values.permute(0, 3, 1, 2), (radius,) * 4)
    padded_coverage = F.pad(coverage, (radius,) * 4)
    scores, neighbors, validity = [], [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y, x = radius + dy, radius + dx
            neighbor_key = padded_keys[:, :, y:y + height, x:x + width].permute(0, 2, 3, 1)
            neighbor_value = padded_values[:, :, y:y + height, x:x + width].permute(0, 2, 3, 1)
            scores.append((queries * neighbor_key).sum(dim=-1) / math.sqrt(channels))
            neighbors.append(neighbor_value)
            validity.append(padded_coverage[:, y:y + height, x:x + width])
    attention = _availability_weights(torch.stack(scores, dim=-1), torch.stack(validity, dim=-1))
    return (attention.unsqueeze(-1) * torch.stack(neighbors, dim=-2)).sum(dim=-2)


class ObservationEncoder(nn.Module):
    """Produce observation-only evidence for the fixed C/D write slots.

    A single Conv3d consumes camera images, video tubelets, or three triplanes.
    Short steps are left-padded internally with unavailable frames; the caller
    retains responsibility for supplying complete physical update steps. No
    unpaired frame or convolution cache is retained here. Camera view order is
    a set order; only explicit calibration gives a camera a geometric identity.
    Triplane order has the declared semantic identities XY, XZ, YZ.
    """

    def __init__(
        self, spec: CDStateSpec, embed_dim: int = 32, patch_size: int = 4,
        neighborhood_radius: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(spec, CDStateSpec):
            raise TypeError("spec: expected CDStateSpec")
        _positive_int("embed_dim", embed_dim)
        _positive_int("patch_size", patch_size)
        _positive_int("neighborhood_radius", neighborhood_radius, allow_zero=True)
        self.spec = spec
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.neighborhood_radius = neighborhood_radius
        self.patchify = nn.Conv3d(3, embed_dim,
                                 kernel_size=(spec.frames_per_step, patch_size, patch_size),
                                 stride=(spec.frames_per_step, patch_size, patch_size), bias=False)
        # xy, aggregate coverage, geometry-present, and per-frame time,
        # coverage, world ray (3), camera origin (3).
        self.metadata_projection = nn.Linear(4 + 8 * spec.frames_per_step, embed_dim)
        self.modality_embedding = nn.Embedding(2, embed_dim)
        self.plane_embedding = nn.Embedding(3, embed_dim)
        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.local_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.local_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.local_output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pool_keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pool_values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.evidence_queries = nn.Parameter(torch.empty(spec.c_tokens + spec.d_tokens, embed_dim))
        nn.init.normal_(self.evidence_queries, std=1 / math.sqrt(embed_dim))
        self.output_norm = nn.LayerNorm(embed_dim)

    def _metadata(self, observation: ObservationStep, frame_coverage: Tensor) -> Tensor:
        views, frames, _, height, width = observation.images.shape
        hp, wp = height // self.patch_size, width // self.patch_size
        target_frames = self.spec.frames_per_step
        pad_frames = target_frames - frames
        dtype, device = observation.images.dtype, observation.images.device
        yy, xx = torch.meshgrid(torch.arange(hp, dtype=dtype, device=device),
                                torch.arange(wp, dtype=dtype, device=device), indexing="ij")
        xy = torch.stack(((xx + .5) * 2 / wp - 1, (yy + .5) * 2 / hp - 1), dim=-1)
        xy = xy.unsqueeze(0).expand(views, -1, -1, -1)
        # Subtract in float64 before converting, preserving small physical
        # intervals even when absolute acquisition timestamps are large.
        relative_times = (observation.frame_times - observation.frame_times.max()).to(dtype)
        relative_times = F.pad(relative_times, (pad_frames, 0))
        times = torch.tanh(relative_times)[:, None, None, :].expand(-1, hp, wp, -1)
        per_frame_coverage = F.pad(frame_coverage, (0, 0, 0, 0, pad_frames, 0))
        per_frame_coverage = per_frame_coverage.permute(0, 2, 3, 1)
        overall_coverage = per_frame_coverage.mean(dim=-1, keepdim=True)
        geometry_flag = observation.images.new_full((views, hp, wp, 1),
                                                    float(observation.intrinsics is not None))
        rays = observation.images.new_zeros(views, target_frames, hp, wp, 3)
        origins = torch.zeros_like(rays)
        if observation.intrinsics is not None:
            # Geometry is computed in calibration precision, independently of
            # image dtype, and supplied only from current input metadata.
            pixels = torch.stack(((xx.double() + .5) * self.patch_size,
                                  (yy.double() + .5) * self.patch_size,
                                  torch.ones_like(xx, dtype=torch.float64)), dim=-1)
            camera_rays = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(observation.intrinsics.double()), pixels)
            world_rays = torch.einsum("vfij,vhwj->vfhwi", observation.cam_to_world[..., :3, :3].double(), camera_rays)
            world_rays = F.normalize(world_rays, dim=-1)
            rays[:, pad_frames:] = world_rays.to(dtype)
            origins[:, pad_frames:] = torch.tanh(observation.cam_to_world[..., :3, 3].double() / 10).to(dtype)[:, :, None, None, :]
        rays = rays.permute(0, 2, 3, 1, 4).reshape(views, hp, wp, target_frames * 3)
        origins = origins.permute(0, 2, 3, 1, 4).reshape(views, hp, wp, target_frames * 3)
        return torch.cat((xy, overall_coverage, geometry_flag, times, per_frame_coverage, rays, origins), dim=-1)

    def forward(self, observation: ObservationStep, *, modality: str = "camera") -> tuple[Tensor, Tensor]:
        if not isinstance(observation, ObservationStep):
            raise TypeError("observation: expected ObservationStep")
        observation.validate_against(self.spec)
        if modality not in ("camera", "triplane"):
            raise ValueError("modality: expected camera or triplane")
        views, frames, _, height, width = observation.images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        if observation.images.dtype not in (torch.float32, torch.float64):
            raise ValueError("reference observation dtype must be float32 or float64")
        if observation.images.dtype != self.patchify.weight.dtype:
            raise ValueError("observation dtype must match encoder weights")
        if observation.images.device != self.patchify.weight.device:
            raise ValueError("observation device must match encoder weights")
        if modality == "triplane" and (views != 3 or frames != 1 or observation.intrinsics is not None):
            raise ValueError("triplane requires three planes, one frame, and no camera calibration")

        masked = observation.masked_images().permute(0, 2, 1, 3, 4)
        masked = F.pad(masked, (0, 0, 0, 0, self.spec.frames_per_step - frames, 0))
        patches = self.patchify(masked).squeeze(2).permute(0, 2, 3, 1)
        frame_coverage = F.avg_pool2d(observation.observed.to(patches.dtype).reshape(views * frames, 1, height, width),
                                      self.patch_size, self.patch_size).reshape(views, frames, height // self.patch_size, width // self.patch_size)
        coverage = frame_coverage.sum(dim=1) / self.spec.frames_per_step
        patches = patches + self.metadata_projection(self._metadata(observation, frame_coverage))
        modality_index = torch.tensor(0 if modality == "camera" else 1, device=patches.device)
        patches = patches + self.modality_embedding(modality_index)
        if modality == "triplane":
            patches = patches + self.plane_embedding(torch.arange(3, device=patches.device))[:, None, None, :]
        normalized = self.local_norm(patches)
        local = _local_neighbor_attention(self.local_query(normalized), self.local_key(normalized),
                                          self.local_value(normalized), coverage, radius=self.neighborhood_radius)
        patches = patches + self.local_output(local)
        tokens = patches.reshape(-1, self.embed_dim)
        scores = self.evidence_queries @ self.pool_keys(tokens).transpose(0, 1) / math.sqrt(self.embed_dim)
        weights = _availability_weights(scores, coverage.reshape(1, -1))
        evidence = self.output_norm(weights @ self.pool_values(tokens))
        has_observation = observation.observed.any()
        return torch.where(has_observation, evidence, torch.zeros_like(evidence)), has_observation
