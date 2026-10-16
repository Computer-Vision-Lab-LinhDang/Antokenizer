"""Shared, stateless Patchify and spatial Transformer for hybrid v2.

Windows are physically partitioned, including shifted boundary windows. No
quadratic global mask is used to simulate window attention. Only every fourth
block is global across the current step's patches/views. Neither operation
retains a frame, convolution state, or attention cache between calls.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.contracts import ObservationStep

from .config import HybridConfig


def _attention(query: Tensor, key: Tensor, value: Tensor, heads: int,
               coverage: Tensor | None = None) -> Tensor:
    """Multihead attention on [B,N,E], with exact all-missing zero rows."""
    batch, count, width = query.shape
    head_dim = width // heads
    q = query.reshape(batch, count, heads, head_dim).transpose(1, 2)
    k = key.reshape(batch, key.shape[1], heads, head_dim).transpose(1, 2)
    v = value.reshape(batch, value.shape[1], heads, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    if coverage is None:
        weights = scores.softmax(dim=-1)
    else:
        available = coverage[:, None, None, :]
        scores = scores.masked_fill(available <= 0, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1) * available
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return (weights @ v).transpose(1, 2).reshape(batch, count, width)


class SpatialBlock(nn.Module):
    """Current-step pre-normalized attention and FFN, without dropout.

    Shifted windows use non-cyclic padding. Pixels at opposite image edges
    never become neighbors through a roll operation. Local blocks isolate
    views; global blocks can mix views, with geometry supplied by the caller.
    """

    def __init__(self, *, width: int, heads: int, window_size: int,
                 shift: int, global_attention: bool) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("spatial width must be positive and divisible by heads")
        if window_size <= 0 or not 0 <= shift < window_size:
            raise ValueError("window_size and shift must define a valid spatial window")
        self.width, self.heads = width, heads
        self.window_size, self.shift = window_size, shift
        self.global_attention = global_attention
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def _attend(self, tokens: Tensor, coverage: Tensor) -> Tensor:
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        return self.output(_attention(q, k, v, self.heads, coverage))

    def _windows(self, tokens: Tensor, coverage: Tensor) -> Tensor:
        views, height, width, channels = tokens.shape
        window, shift = self.window_size, self.shift
        padded_h = math.ceil((height + shift) / window) * window
        padded_w = math.ceil((width + shift) / window) * window
        pads = (shift, padded_w - width - shift, shift, padded_h - height - shift)
        padded = F.pad(tokens.permute(0, 3, 1, 2), pads).permute(0, 2, 3, 1)
        validity = F.pad(coverage, pads)
        rows, columns = padded_h // window, padded_w // window
        windows = padded.reshape(views, rows, window, columns, window, channels)
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)
        masks = validity.reshape(views, rows, window, columns, window)
        masks = masks.permute(0, 1, 3, 2, 4).reshape(-1, window * window)
        output = self._attend(windows, masks)
        output = output.reshape(views, rows, columns, window, window, channels)
        output = output.permute(0, 1, 3, 2, 4, 5).reshape(views, padded_h, padded_w, channels)
        return output[:, shift:shift + height, shift:shift + width]

    def forward(self, tokens: Tensor, coverage: Tensor) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != self.width or coverage.shape != tokens.shape[:-1]:
            raise ValueError("spatial input must be [V,H,W,E] and coverage [V,H,W]")
        valid = coverage.unsqueeze(-1) > 0
        tokens = torch.where(valid, tokens, torch.zeros_like(tokens))
        normalized = self.norm(tokens)
        if self.global_attention:
            update = self._attend(normalized.reshape(1, -1, self.width), coverage.reshape(1, -1))
            update = update.reshape_as(tokens)
        else:
            update = self._windows(normalized, coverage)
        output = tokens + update
        output = output + self.ffn(self.ffn_norm(output))
        return torch.where(valid, output, torch.zeros_like(output))


class HybridObservationEncoder(nn.Module):
    """Map one physical step to fixed summary tokens, without prior state.

    Images use one real frame and one unavailable left-padding frame. A
    triplane uses three explicitly identified planes, each with one real
    frame; plane order is never interpreted as temporal order. Camera input
    row i is the stable camera identity used by DecoderQuery.view_ids==i.
    Learned row identity is distinct from optional calibration and poses;
    it does not imply geometric information for uncalibrated cameras.
    """

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        if not isinstance(config, HybridConfig):
            raise TypeError("config must be HybridConfig")
        self.config = config
        width = config.spatial_width
        self.patchify = nn.Conv3d(3, width,
                                 kernel_size=(config.frames_per_step, config.patch_size, config.patch_size),
                                 stride=(config.frames_per_step, config.patch_size, config.patch_size), bias=False)
        self.metadata_projection = nn.Linear(4 + 8 * config.frames_per_step, width)
        self.modality_embedding = nn.Embedding(2, width)
        self.camera_view_embedding = nn.Embedding(config.reference_views, width)
        self.plane_embedding = nn.Embedding(3, width)
        local_count = 0
        blocks = []
        for index in range(config.spatial_blocks):
            global_attention = (index + 1) % 4 == 0
            shift = 0 if local_count % 2 == 0 else config.window_size // 2
            if not global_attention:
                local_count += 1
            blocks.append(SpatialBlock(width=width, heads=config.spatial_heads,
                                       window_size=config.window_size, shift=shift,
                                       global_attention=global_attention))
        self.blocks = nn.ModuleList(blocks)
        self.pool_norm = nn.LayerNorm(width)
        self.pool_keys = nn.Linear(width, config.width)
        self.pool_values = nn.Linear(width, config.width)
        self.pool_queries = nn.Parameter(torch.empty(config.summary_tokens, config.width))
        nn.init.normal_(self.pool_queries, std=1 / math.sqrt(config.width))
        self.pool_output = nn.Linear(config.width, config.width)
        self.pool_ffn_norm = nn.LayerNorm(config.width)
        self.pool_ffn = nn.Sequential(nn.Linear(config.width, config.width * 4), nn.GELU(),
                                      nn.Linear(config.width * 4, config.width))

    def _metadata(self, observation: ObservationStep, frame_coverage: Tensor) -> Tensor:
        views, frames, _, height, width = observation.images.shape
        hp, wp = frame_coverage.shape[-2:]
        target_frames = self.config.frames_per_step
        pad_frames = target_frames - frames
        dtype, device = observation.images.dtype, observation.images.device
        yy, xx = torch.meshgrid(torch.arange(hp, dtype=dtype, device=device),
                                torch.arange(wp, dtype=dtype, device=device), indexing="ij")
        pixel_x, pixel_y = (xx + .5) * self.config.patch_size, (yy + .5) * self.config.patch_size
        xy = torch.stack((pixel_x * 2 / width - 1, pixel_y * 2 / height - 1), dim=-1)
        xy = xy.unsqueeze(0).expand(views, -1, -1, -1)
        relative_times = (observation.frame_times - observation.frame_times.max()).to(dtype)
        relative_times = F.pad(relative_times, (pad_frames, 0))
        times = torch.tanh(relative_times)[:, None, None, :].expand(-1, hp, wp, -1)
        per_frame = F.pad(frame_coverage, (0, 0, 0, 0, pad_frames, 0)).permute(0, 2, 3, 1)
        coverage = per_frame.mean(dim=-1, keepdim=True)
        geometry = observation.images.new_full((views, hp, wp, 1), float(observation.intrinsics is not None))
        rays = observation.images.new_zeros(views, target_frames, hp, wp, 3)
        origins = torch.zeros_like(rays)
        if observation.intrinsics is not None:
            pixels = torch.stack((pixel_x.double(), pixel_y.double(),
                                  torch.ones_like(xx, dtype=torch.float64)), dim=-1)
            camera_rays = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(observation.intrinsics.double()), pixels)
            world_rays = torch.einsum("vfij,vhwj->vfhwi", observation.cam_to_world[..., :3, :3].double(), camera_rays)
            rays[:, pad_frames:] = F.normalize(world_rays, dim=-1).to(dtype)
            origins[:, pad_frames:] = torch.tanh(observation.cam_to_world[..., :3, 3].double() / 10).to(dtype)[:, :, None, None, :]
        rays = rays.permute(0, 2, 3, 1, 4).reshape(views, hp, wp, -1)
        origins = origins.permute(0, 2, 3, 1, 4).reshape(views, hp, wp, -1)
        return torch.cat((xy, coverage, geometry, times, per_frame, rays, origins), dim=-1)

    def forward(self, observation: ObservationStep, *, modality: str = "camera") -> tuple[Tensor, Tensor]:
        if not isinstance(observation, ObservationStep):
            raise TypeError("observation must be ObservationStep")
        observation.validate()
        if modality not in ("camera", "triplane"):
            raise ValueError("modality must be camera or triplane")
        views, frames, _, height, width = observation.images.shape
        if views != self.config.reference_views:
            raise ValueError("view count must match fixed reference_views")
        if frames > self.config.frames_per_step:
            raise ValueError("frame count exceeds frames_per_step")
        if observation.images.dtype != torch.float32 or self.patchify.weight.dtype != torch.float32:
            raise ValueError("hybrid observation and encoder weights must be FP32")
        if observation.images.device != self.patchify.weight.device:
            raise ValueError("observation and encoder weights must share a device")
        if modality == "triplane" and (views != 3 or frames != 1 or observation.intrinsics is not None):
            raise ValueError("triplane requires three planes, one real frame, and no camera calibration")
        observed = observation.observed.any()
        if not bool(observed):
            return observation.images.new_zeros(self.config.summary_tokens, self.config.width), observed
        patch = self.config.patch_size
        bottom, right = (-height) % patch, (-width) % patch
        masked = observation.masked_images().permute(0, 2, 1, 3, 4)
        masked = F.pad(masked, (0, right, 0, bottom, self.config.frames_per_step - frames, 0))
        patches = self.patchify(masked).squeeze(2).permute(0, 2, 3, 1)
        availability = F.pad(observation.observed.to(patches.dtype), (0, right, 0, bottom))
        hp, wp = patches.shape[1:3]
        frame_coverage = F.avg_pool2d(availability.reshape(views * frames, 1, height + bottom, width + right),
                                      patch, patch).reshape(views, frames, hp, wp)
        coverage = frame_coverage.sum(dim=1) / self.config.frames_per_step
        patches = patches + self.metadata_projection(self._metadata(observation, frame_coverage))
        modality_index = torch.tensor(0 if modality == "camera" else 1, device=patches.device)
        patches = patches + self.modality_embedding(modality_index)
        if modality == "camera":
            # Persist identity in evidence even when two cameras have absent
            # or identical calibration. Input rows must stay stable over time.
            view_ids = torch.arange(views, device=patches.device)
            patches = patches + self.camera_view_embedding(view_ids)[:, None, None, :]
        else:
            patches = patches + self.plane_embedding(torch.arange(3, device=patches.device))[:, None, None, :]
        for block in self.blocks:
            patches = block(patches, coverage)
        tokens = self.pool_norm(patches.reshape(1, -1, self.config.spatial_width))
        queries = self.pool_queries.unsqueeze(0)
        pooled = _attention(queries, self.pool_keys(tokens), self.pool_values(tokens),
                            self.config.q_heads, coverage.reshape(1, -1))
        pooled = queries + self.pool_output(pooled)
        pooled = pooled + self.pool_ffn(self.pool_ffn_norm(pooled))
        return pooled.squeeze(0), observed
