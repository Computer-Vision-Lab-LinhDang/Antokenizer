"""Canonical content readout with explicit SE(3) controls and bounded ray chunks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ObjectStateConfig
from .state import ObjectState


@dataclass(frozen=True)
class RayQuery:
    origins: Tensor  # [B,R,3], reference coordinates
    directions: Tensor  # [B,R,3], unit world direction
    near: Tensor | float  # scalar, [B,1] or [B,R], metric range
    far: Tensor | float
    time: Tensor  # [B], must equal state time
    valid: Tensor | None = None  # boolean [B,R]

    def to(self, device) -> RayQuery:
        return RayQuery(
            **{
                name: value.to(device) if isinstance(value, Tensor) else value
                for name, value in vars(self).items()
            }
        )


def make_ray_query(
    intrinsics: Tensor,
    camera_c2w: Tensor,
    pixels: Tensor,
    *,
    time: Tensor,
    near: Tensor | float,
    far: Tensor | float,
    valid: Tensor | None = None,
) -> RayQuery:
    """Construct rays for pixel centers in the camera's native K convention.

    Inputs are [B,3,3], [B,4,4], and [B,R,2]. Multi-view callers flatten
    batch/view or concatenate the rays explicitly; camera-z is not ray range.
    """
    if pixels.ndim != 3 or pixels.shape[-1] != 2:
        raise ValueError("pixels must be [B,R,2]")
    b, count, _ = pixels.shape
    if intrinsics.shape != (b, 3, 3) or camera_c2w.shape != (b, 4, 4):
        raise ValueError("query cameras must be [B,3,3] and [B,4,4]")
    if not torch.isfinite(intrinsics).all() or not torch.isfinite(camera_c2w).all():
        raise ValueError("query camera geometry must be finite")
    with torch.autocast(device_type=pixels.device.type, enabled=False):
        intrinsics, camera_c2w, pixels = (
            intrinsics.float(),
            camera_c2w.float(),
            pixels.float(),
        )
        homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
        camera_rays = torch.einsum(
            "bij,bnj->bni", torch.linalg.inv(intrinsics), homogeneous
        )
        directions = F.normalize(
            torch.einsum("bij,bnj->bni", camera_c2w[:, :3, :3], camera_rays), dim=-1
        )
        origins = camera_c2w[:, None, :3, 3].expand(-1, count, -1).clone()
        return RayQuery(origins, directions, near, far, time, valid)


def volume_render(
    sigma: Tensor,
    color: Tensor,
    distances: Tensor,
    deltas: Tensor,
    background: Tensor,
    *,
    hit_threshold: float = 1e-4,
) -> dict[str, Tensor]:
    """Render nonnegative density; depth is conditional on a hit.

    Missing-hit ranges are zero placeholders, paired with an explicit mask.
    The caller's loss/evaluator must penalize missing hits on target-hit rays.
    """
    optical = sigma * deltas
    cumulative = optical.cumsum(-1)
    before = torch.cat(
        (torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1
    )
    transmittance = torch.exp(-before)
    alpha = -torch.expm1(-optical)
    weights = transmittance * alpha
    residual = torch.exp(-cumulative[..., -1])
    opacity = -torch.expm1(-cumulative[..., -1])
    rgb = (weights[..., None] * color).sum(-2) + residual[..., None] * background
    hit = opacity > hit_threshold
    ray_range = (weights * distances).sum(-1) / opacity.clamp_min(
        torch.finfo(opacity.dtype).eps
    )
    ray_range = torch.where(hit, ray_range, torch.zeros_like(ray_range))
    return {"rgb": rgb, "range": ray_range, "opacity": opacity, "hit": hit}


def _rotation_6d(values: Tensor) -> Tensor:
    first, second = values[..., :3], values[..., 3:]
    default_first = torch.zeros_like(first)
    default_first[..., 0] = 1
    first = torch.where((first.norm(dim=-1, keepdim=True) > 1e-7), first, default_first)
    b1 = F.normalize(first, dim=-1)
    remainder = second - (b1 * second).sum(-1, keepdim=True) * b1
    # Choose the least-aligned basis axis to obtain a nonsingular fallback.
    basis = F.one_hot(b1.abs().argmin(-1), 3).to(values.dtype)
    fallback = basis - (b1 * basis).sum(-1, keepdim=True) * b1
    remainder = torch.where(
        remainder.norm(dim=-1, keepdim=True) > 1e-7, remainder, fallback
    )
    b2 = F.normalize(remainder, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


class _CanonicalField(nn.Module):
    """Cross-attend coordinate queries to C; D is not an argument."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.register_buffer(
            "frequencies", torch.tensor([1.0, 2.0, 4.0, 8.0]), persistent=False
        )
        self.query = nn.Sequential(
            nn.Linear(27, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.keys = nn.Linear(width, width)
        self.values = nn.Linear(width, width)
        self.mix = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.density = nn.Linear(width, 1)
        self.color = nn.Sequential(
            nn.Linear(width + 3, width), nn.SiLU(), nn.Linear(width, 3)
        )
        nn.init.constant_(self.density.bias, -2.0)

    def forward(
        self, content: Tensor, points: Tensor, directions: Tensor
    ) -> tuple[Tensor, Tensor]:
        # [batch_or_batch_slots, tokens, Z], [batch_or_batch_slots, Q, 3]
        b, count, _ = points.shape
        angles = points[..., None] * self.frequencies
        embedding = torch.cat(
            (points, angles.sin().flatten(-2), angles.cos().flatten(-2)), -1
        )
        q = self.query(embedding)
        head_dim = q.shape[-1] // self.heads
        queries = q.reshape(b, count, self.heads, head_dim).transpose(1, 2)
        keys = self.keys(content).reshape(b, -1, self.heads, head_dim).transpose(1, 2)
        values = (
            self.values(content).reshape(b, -1, self.heads, head_dim).transpose(1, 2)
        )
        # Query count is bounded by ray_chunk_size * samples_per_ray.
        read = F.scaled_dot_product_attention(queries, keys, values)
        read = read.transpose(1, 2).reshape_as(q)
        features = self.mix(torch.cat((read, q), -1))
        sigma = F.softplus(self.density(features).squeeze(-1))
        rgb = torch.sigmoid(self.color(torch.cat((features, directions), -1)))
        return sigma, rgb


class CanonicalReadout(nn.Module):
    def __init__(self, config: ObjectStateConfig):
        super().__init__()
        self.config = config
        self.control_head = nn.Sequential(
            nn.Linear(config.dynamic_tokens * config.width, config.width),
            nn.SiLU(),
            nn.Linear(config.width, 9),
        )
        # Identity-safe at D == 0, but with a differentiable path from D.
        nn.init.zeros_(self.control_head[0].bias)
        nn.init.normal_(self.control_head[-1].weight, std=1e-3)
        with torch.no_grad():
            self.control_head[-1].bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            )
        self.object_field = _CanonicalField(config.width, config.resampler_heads)
        self.background_field = _CanonicalField(config.width, config.resampler_heads)
        self.environment = nn.Sequential(
            nn.Linear(config.width, config.width),
            nn.SiLU(),
            nn.Linear(config.width, 3),
            nn.Sigmoid(),
        )
        self.default_ray_chunk_size = config.query_chunk_size

    def controls(self, dynamic: Tensor) -> Tensor:
        if dynamic.ndim != 4 or dynamic.shape[-2:] != (
            self.config.dynamic_tokens,
            self.config.width,
        ):
            raise ValueError("dynamic controls require [B,K,ND,Z]")
        with torch.autocast(device_type=dynamic.device.type, enabled=False):
            values = self.control_head(dynamic.flatten(-2))
            rotation = _rotation_6d(values[..., :6])
            top = torch.cat((rotation, values[..., 6:, None]), -1)
            bottom = torch.zeros_like(top[..., :1, :])
            bottom[..., 0, 3] = 1
            return torch.cat((top, bottom), -2)

    def _field(
        self, field: nn.Module, content: Tensor, points: Tensor, directions: Tensor
    ):
        if self.training and self.config.decoder_checkpoint and torch.is_grad_enabled():
            return checkpoint(field, content, points, directions, use_reentrant=False)
        return field(content, points, directions)

    def _validate_query(self, state: ObjectState, query: RayQuery):
        if not isinstance(query, RayQuery):
            raise TypeError("query must be RayQuery")
        b = state.c.shape[0]
        if (
            query.origins.ndim != 3
            or query.origins.shape[0] != b
            or query.origins.shape[-1] != 3
        ):
            raise ValueError("ray origins must be [B,R,3]")
        count = query.origins.shape[1]
        if count < 1 or query.directions.shape != query.origins.shape:
            raise ValueError("ray directions must match nonempty origins")
        if query.time.shape != (b,) or not torch.isfinite(query.time).all():
            raise ValueError("query time must be finite [B]")
        if not torch.allclose(query.time.double(), state.t, atol=1e-9, rtol=0):
            raise ValueError(
                "query timestamp must equal state time; rollout future state first"
            )
        valid = query.valid
        if valid is None:
            valid = torch.ones(b, count, device=state.c.device, dtype=torch.bool)
        if valid.shape != (b, count) or valid.dtype != torch.bool:
            raise ValueError("query validity must be boolean [B,R]")
        for value in (query.origins, query.directions, query.time, valid):
            if value.device != state.c.device:
                raise ValueError("query and state must share a device")
        if (
            not torch.isfinite(query.origins[valid]).all()
            or not torch.isfinite(query.directions[valid]).all()
        ):
            raise ValueError("valid rays must be finite")
        lengths = query.directions[valid].norm(dim=-1)
        if not torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4, rtol=1e-4):
            raise ValueError("valid ray directions must be unit length")
        near = torch.as_tensor(query.near, device=state.c.device, dtype=state.c.dtype)
        far = torch.as_tensor(query.far, device=state.c.device, dtype=state.c.dtype)
        try:
            near, far = torch.broadcast_to(near, (b, count)), torch.broadcast_to(
                far, (b, count)
            )
        except RuntimeError as error:
            raise ValueError("near/far must broadcast to [B,R]") from error
        if (
            not torch.isfinite(near[valid]).all()
            or not torch.isfinite(far[valid]).all()
        ):
            raise ValueError("valid near/far must be finite")
        if ((near[valid] < 0) | (far[valid] <= near[valid])).any():
            raise ValueError("ray bounds must satisfy 0 <= near < far")
        return valid, near, far

    def forward(
        self, state: ObjectState, query: RayQuery, *, ray_chunk_size: int | None = None
    ) -> dict[str, Tensor]:
        # Metric queries, pose transforms and integration stay FP32 under AMP.
        with torch.autocast(device_type=state.c.device.type, enabled=False):
            return self._read_fp32(state, query, ray_chunk_size=ray_chunk_size)

    def _read_fp32(
        self, state: ObjectState, query: RayQuery, *, ray_chunk_size: int | None
    ):
        state.validate(self.config)
        valid, near, far = self._validate_query(state, query)
        chunk_size = (
            self.default_ray_chunk_size if ray_chunk_size is None else ray_chunk_size
        )
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("ray_chunk_size must be a positive integer")
        controls = self.controls(state.d)
        b, k = state.c.shape[:2]
        count = query.origins.shape[1]
        samples = self.config.samples_per_ray
        fractions = (
            torch.arange(samples, device=state.c.device, dtype=state.c.dtype) + 0.5
        ) / samples
        existence = state.p_exist * state.allocated
        environment = self.environment(state.c_bg.mean(-2))
        rotation, translation = controls[..., :3, :3], controls[..., :3, 3]
        outputs = []
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            mask = valid[:, start:end]
            origin = torch.where(
                mask[..., None],
                query.origins[:, start:end].float(),
                torch.zeros_like(query.origins[:, start:end], dtype=torch.float32),
            )
            default_direction = torch.zeros_like(origin)
            default_direction[..., 2] = 1
            direction = torch.where(
                mask[..., None],
                query.directions[:, start:end].float(),
                default_direction,
            )
            lo = torch.where(
                mask, near[:, start:end], torch.zeros_like(near[:, start:end])
            )
            hi = torch.where(
                mask, far[:, start:end], torch.ones_like(far[:, start:end])
            )
            distances = lo[..., None] + (hi - lo)[..., None] * fractions
            deltas = ((hi - lo) / samples)[..., None].expand_as(distances)
            points = origin[:, :, None] + distances[..., None] * direction[:, :, None]
            flat_points = points.flatten(1, 2)
            flat_directions = direction[:, :, None].expand_as(points).flatten(1, 2)
            local_points = torch.einsum(
                "bkji,bkqj->bkqi",
                rotation,
                flat_points[:, None] - translation[:, :, None],
            )
            local_directions = torch.einsum("bkji,bqj->bkqi", rotation, flat_directions)
            sigma_obj, color_obj = self._field(
                self.object_field,
                state.c.flatten(0, 1),
                local_points.flatten(0, 1),
                local_directions.flatten(0, 1),
            )
            sigma_obj = sigma_obj.reshape(b, k, -1) * existence[..., None]
            color_obj = color_obj.reshape(b, k, -1, 3)
            sigma_bg, color_bg = self._field(
                self.background_field, state.c_bg, flat_points, flat_directions
            )
            sigma = sigma_bg + sigma_obj.sum(1)
            weighted_color = sigma_bg[..., None] * color_bg + (
                sigma_obj[..., None] * color_obj
            ).sum(1)
            color = weighted_color / sigma[..., None].clamp_min(
                torch.finfo(sigma.dtype).eps
            )
            rendered = volume_render(
                sigma.reshape_as(distances),
                color.reshape(*distances.shape, 3),
                distances,
                deltas,
                environment[:, None].expand(-1, end - start, -1),
            )
            rendered["points"] = origin + rendered["range"][..., None] * direction
            for name, value in rendered.items():
                ray_mask = mask[..., None] if value.ndim == 3 else mask
                rendered[name] = torch.where(ray_mask, value, torch.zeros_like(value))
            outputs.append(rendered)
        result = {
            name: torch.cat([entry[name] for entry in outputs], dim=1)
            for name in outputs[0]
        }
        result["controls"] = controls
        result["valid"] = valid
        return result
