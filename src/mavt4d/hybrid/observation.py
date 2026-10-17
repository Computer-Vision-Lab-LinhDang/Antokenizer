"""Shared, stateless Patchify and spatial Transformer for hybrid v2.

Windows are physically partitioned, including shifted boundary windows. No
quadratic global mask is used to simulate window attention. Only every fourth
block is global across the current step's patches/views. Neither operation
retains a frame, convolution state, or attention cache between calls.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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
        if tokens.ndim not in (4, 5) or tokens.shape[-1] != self.width or coverage.shape != tokens.shape[:-1]:
            raise ValueError("spatial input must be [V,H,W,E] or [B,V,H,W,E], with matching coverage")
        batch = 1 if tokens.ndim == 4 else tokens.shape[0]
        valid = coverage.unsqueeze(-1) > 0
        tokens = torch.where(valid, tokens, torch.zeros_like(tokens))
        normalized = self.norm(tokens)
        if self.global_attention:
            update = self._attend(normalized.reshape(batch, -1, self.width), coverage.reshape(batch, -1))
            update = update.reshape_as(tokens)
        else:
            # Local windows may share one kernel across batch/view rows,
            # while global attention must retain the batch boundary.
            update = self._windows(normalized.reshape(-1, *tokens.shape[-3:]),
                                   coverage.reshape(-1, *coverage.shape[-2:])).reshape_as(tokens)
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
        # Runtime precision policy: parameters and returned evidence remain
        # FP32. This option only autocasts Patchify and spatial Transformer
        # projections; it never changes temporal C/D or serialized KV dtypes.
        self.spatial_autocast = False
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

    def _validate_observation(self, observation: ObservationStep, modality: str) -> Tensor:
        if not isinstance(observation, ObservationStep):
            raise TypeError("observation must be ObservationStep")
        observation.validate()
        if modality not in ("camera", "triplane"):
            raise ValueError("modality must be camera or triplane")
        views, frames = observation.images.shape[:2]
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
        return observation.observed.any()

    def _patch_tokens(self, masked: Tensor, metadata: Tensor, modality: str,
                      batch: int, views: int, hp: int, wp: int) -> Tensor:
        patches = self.patchify(masked).squeeze(2).permute(0, 2, 3, 1)
        patches = patches.reshape(batch, views, hp, wp, self.config.spatial_width)
        patches = patches + self.metadata_projection(metadata)
        modality_index = torch.tensor(0 if modality == "camera" else 1, device=patches.device)
        patches = patches + self.modality_embedding(modality_index)
        view_ids = torch.arange(views, device=patches.device)
        identity = self.camera_view_embedding(view_ids) if modality == "camera" else self.plane_embedding(view_ids)
        return patches + identity[None, :, None, None, :]

    def _encode_group(self, observations: Sequence[ObservationStep], *, modality: str,
                      single: bool = False, return_patches: bool = False):
        """Encode a homogeneous shape group without mixing examples."""
        observation = observations[0]
        batch = len(observations)
        views, frames, _, height, width = observation.images.shape
        patch = self.config.patch_size
        bottom, right = (-height) % patch, (-width) % patch
        images = torch.stack([item.images for item in observations])
        observed = torch.stack([item.observed for item in observations])
        masked = torch.where(observed.unsqueeze(3), images, torch.zeros_like(images))
        masked = masked.permute(0, 1, 3, 2, 4, 5).reshape(batch * views, 3, frames, height, width)
        masked = F.pad(masked, (0, right, 0, bottom, self.config.frames_per_step - frames, 0))
        hp, wp = (height + bottom) // patch, (width + right) // patch
        availability = F.pad(observed.to(torch.float32), (0, right, 0, bottom))
        frame_coverage = F.avg_pool2d(
            availability.reshape(batch * views * frames, 1, height + bottom, width + right),
            patch, patch).reshape(batch, views, frames, hp, wp)
        coverage = frame_coverage.sum(dim=2) / self.config.frames_per_step
        metadata = torch.stack([self._metadata(item, frame_coverage[index])
                                for index, item in enumerate(observations)])
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                            enabled=self.spatial_autocast):
            patches = self._patch_tokens(masked, metadata, modality, batch, views, hp, wp)
            if single:
                # Keep the original single-item module interface for callers
                # that inspect per-view spatial activations using hooks.
                patches, spatial_coverage = patches[0], coverage[0]
            else:
                spatial_coverage = coverage
            for block in self.blocks:
                patches = block(patches, spatial_coverage)
        with torch.autocast(device_type=images.device.type, enabled=False):
            tokens = self.pool_norm(patches.float().reshape(batch, -1, self.config.spatial_width))
            queries = self.pool_queries.unsqueeze(0).expand(batch, -1, -1)
            pooled = _attention(queries, self.pool_keys(tokens), self.pool_values(tokens),
                                self.config.q_heads, coverage.reshape(batch, -1))
            pooled = queries + self.pool_output(pooled)
            evidence = pooled + self.pool_ffn(self.pool_ffn_norm(pooled))
        if not return_patches:
            return evidence
        # Innovation tokens: pre-block (own pixels + metadata only), FP32, so a
        # mask elsewhere in a window never changes an observed patch's token.
        with torch.autocast(device_type=images.device.type, enabled=False):
            tokens = self._patch_tokens(masked, metadata, modality, batch, views, hp, wp)
        return evidence, tokens, coverage

    def forward(self, observation: ObservationStep, *, modality: str = "camera") -> tuple[Tensor, Tensor]:
        observed = self._validate_observation(observation, modality)
        if not bool(observed):
            return observation.images.new_zeros(self.config.summary_tokens, self.config.width), observed
        return self._encode_group((observation,), modality=modality, single=True)[0], observed

    def forward_batch(self, observations: Sequence[ObservationStep], *,
                      modality: str = "camera", return_patches: bool = False) -> tuple[Tensor, ...]:
        """Return independent FP32 evidence [B,P,E] and availability [B].

        Equal shapes share Patchify/Transformer kernels. Different spatial
        sizes or frame counts form separate groups rather than adding fake
        image pixels or changing each example's positional metadata. Missing
        examples skip spatial computation, as in the single-item interface.
        """
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
            raise TypeError("observations must be a sequence of ObservationStep")
        if not observations:
            raise ValueError("observations must be nonempty")
        available = torch.stack([self._validate_observation(item, modality) for item in observations])
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, present in enumerate(available.tolist()):
            if present:
                groups.setdefault(tuple(observations[index].images.shape), []).append(index)
        output = [observations[0].images.new_zeros(self.config.summary_tokens, self.config.width)
                  for _ in observations]
        if return_patches and len(groups) > 1:
            raise ValueError("return_patches requires all available observations to share one image shape")
        extra: list[Tensor | None] = [None] * len(observations)
        cov_rows: list[Tensor | None] = [None] * len(observations)
        for indices in groups.values():
            encoded = self._encode_group([observations[index] for index in indices], modality=modality,
                                         return_patches=return_patches)
            if return_patches:
                encoded, tokens, covs = encoded
            for row, index in enumerate(indices):
                output[index] = encoded[row]
                if return_patches:
                    extra[index], cov_rows[index] = tokens[row], covs[row]
        if not return_patches:
            return torch.stack(output), available
        views, _, _, height, width = observations[0].images.shape
        grid = (views, -(-height // self.config.patch_size), -(-width // self.config.patch_size))
        device = observations[0].images.device
        zero_p = torch.zeros(*grid, self.config.spatial_width, device=device)
        zero_c = torch.zeros(*grid, device=device)
        patches = torch.stack([zero_p if t is None else t for t in extra])
        coverage = torch.stack([zero_c if c is None else c for c in cov_rows])
        return torch.stack(output), available, patches, coverage
