"""Stateless sparse encoding of endpoint-calibrated RGB observations.

Earlier tubelet frames supply context only. Rays and optional measured points
are anchored to the endpoint; no temporal feature/cache survives a call.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.hybrid.observation import SpatialBlock

from .config import ObjectStateConfig
from .state import Evidence


@dataclass(frozen=True)
class ObservationStep:
    images: Tensor  # [B,V,T,3,H,W], T <= tubelet_size
    time: Tensor  # [B], endpoint timestamp
    valid: Tensor | None = None  # boolean [B,V,T,1,H,W]
    intrinsics: Tensor | None = None  # endpoint [B,V,3,3]
    camera_c2w: Tensor | None = None  # endpoint [B,V,4,4]
    depth: Tensor | None = None  # measured camera-z [B,V,1,H,W]
    depth_valid: Tensor | None = None
    geometry_valid: Tensor | None = None  # calibrated/synchronized views [B,V]

    def to(self, device) -> ObservationStep:
        return ObservationStep(
            **{
                name: value.to(device) if isinstance(value, Tensor) else value
                for name, value in vars(self).items()
            }
        )


def _attention(
    query: Tensor, key: Tensor, value: Tensor, valid: Tensor, heads: int
) -> Tensor:
    """Linear-size cross attention; an all-missing row returns exactly zero."""
    b, qn, width = query.shape
    hn = width // heads
    q = query.reshape(b, qn, heads, hn).transpose(1, 2)
    k = key.reshape(b, -1, heads, hn).transpose(1, 2)
    v = value.reshape(b, -1, heads, hn).transpose(1, 2)
    mask = valid[:, None, None, :]
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(hn)
    weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1)
    weights = weights * mask
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return (weights @ v).transpose(1, 2).reshape(b, qn, width)


class _RelayFusion(nn.Module):
    def __init__(self, width: int, heads: int, count: int):
        super().__init__()
        self.heads = heads
        self.seeds = nn.Parameter(torch.randn(count, width) / math.sqrt(width))
        self.norm = nn.LayerNorm(width)
        self.kv = nn.Linear(width, 2 * width)
        self.read = nn.Linear(width, width)
        self.write_q = nn.Linear(width, width)
        self.write_kv = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        b, n, width = x.shape
        k, v = self.kv(self.norm(x)).chunk(2, -1)
        seeds = self.seeds[None].expand(b, -1, -1)
        relay = self.read(_attention(seeds, k, v, valid, self.heads))
        k, v = self.write_kv(relay).chunk(2, -1)
        relay_valid = valid.any(-1, keepdim=True).expand(-1, relay.shape[1])
        update = self.output(
            _attention(self.write_q(self.norm(x)), k, v, relay_valid, self.heads)
        )
        return torch.where(valid[..., None], x + update, torch.zeros_like(x))


class _GeometricFusion(nn.Module):
    """Gather at most M projected cross-view patch neighbors per token.

    Candidate construction costs O(N * views * 9), not an N-by-N mask.
    Measured endpoint depth is required on both ends of a correspondence.
    """

    def __init__(self, width: int, heads: int, neighbors: int):
        super().__init__()
        self.heads, self.neighbors = heads, neighbors
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.distance_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: Tensor,
        valid: Tensor,
        points: Tensor,
        point_valid: Tensor,
        intrinsics: Tensor,
        c2w: Tensor,
        geometry_valid: Tensor,
        grid_h: int,
        grid_w: int,
        patch_size: int,
    ) -> Tensor:
        b, n, width = x.shape
        views = c2w.shape[1]
        if views < 2:
            return x
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Each query point is projected into each target camera.
            rotation, translation = c2w[..., :3, :3], c2w[..., :3, 3]
            cam = torch.einsum(
                "bvij,bnvj->bnvi",
                rotation.transpose(-1, -2),
                points[:, :, None] - translation[:, None],
            )
            projected = torch.einsum("bvij,bnvj->bnvi", intrinsics, cam)
            uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
            base = torch.floor(uv / patch_size).long()
            offsets = torch.tensor(
                [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], device=x.device
            )
            xy = base[..., None, :] + offsets
            gx, gy = xy[..., 0], xy[..., 1]
            view_ids = torch.arange(views, device=x.device)[None, None, :, None]
            source = torch.arange(n, device=x.device) // (grid_h * grid_w)
            available = (
                (gx >= 0)
                & (gx < grid_w)
                & (gy >= 0)
                & (gy < grid_h)
                & (cam[..., 2, None] > 1e-6)
                & geometry_valid[:, None, :, None]
                & point_valid[:, :, None, None]
                & (view_ids != source[None, :, None, None])
            )
            indices = (
                view_ids * (grid_h * grid_w)
                + gy.clamp(0, grid_h - 1) * grid_w
                + gx.clamp(0, grid_w - 1)
            ).reshape(b, n, -1)
            available = available.reshape(b, n, -1)
            batch_index = torch.arange(b, device=x.device)[:, None, None]
            available = (
                available
                & point_valid[batch_index, indices]
                & valid[batch_index, indices]
            )
            distance = (
                (points[batch_index, indices] - points[:, :, None]).square().sum(-1)
            )
            count = min(self.neighbors, indices.shape[-1])
            ordering = (
                distance.masked_fill(~available, float("inf"))
                .topk(count, largest=False)
                .indices
            )
            indices = indices.gather(-1, ordering)
            available = available.gather(-1, ordering)
            distance = distance.gather(-1, ordering)
        q, k, v = self.qkv(self.norm(x)).chunk(3, -1)
        hd = width // self.heads
        q = q.reshape(b, n, self.heads, hd)
        k = k[batch_index, indices].reshape(b, n, count, self.heads, hd)
        v = v[batch_index, indices].reshape(b, n, count, self.heads, hd)
        logits = (q[:, :, None] * k).sum(-1) / math.sqrt(hd)
        logits = logits - F.softplus(self.distance_scale) * distance[..., None]
        weights = logits.masked_fill(
            ~available[..., None], torch.finfo(logits.dtype).min
        ).softmax(2)
        weights = weights * available[..., None]
        weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-12)
        update = self.output((weights[..., None] * v).sum(2).flatten(-2))
        update = torch.where(
            available.any(-1)[..., None], update, torch.zeros_like(update)
        )
        return torch.where(valid[..., None], x + update, torch.zeros_like(x))


class SparseObservationEncoder(nn.Module):
    def __init__(self, config: ObjectStateConfig):
        super().__init__()
        self.config = config
        p, t, width = config.patch_size, config.tubelet_size, config.evidence_width
        self.patchify = nn.Conv3d(
            config.image_channels, width, (t, p, p), (t, p, p), bias=False
        )
        self.position = nn.Linear(8, width)
        self.geometry = nn.Linear(9, width, bias=False)
        self.blocks = nn.ModuleList(
            [
                SpatialBlock(
                    width=width,
                    heads=config.spatial_heads,
                    window_size=config.window_size,
                    shift=0 if index % 2 == 0 else config.window_size // 2,
                    global_attention=False,
                )
                for index in range(config.spatial_blocks)
            ]
        )
        if config.mlp_ratio != 4:
            for block in self.blocks:
                hidden = width * config.mlp_ratio
                block.ffn = nn.Sequential(
                    nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)
                )
        indices = [
            index
            for index in range(config.spatial_blocks)
            if (index + 1) % 4 == 0 or index == config.spatial_blocks - 1
        ]
        self.relays = nn.ModuleDict(
            {
                str(index): _RelayFusion(
                    width, config.spatial_heads, config.relay_tokens
                )
                for index in indices
            }
        )
        self.geometric = nn.ModuleDict(
            {
                str(index): _GeometricFusion(
                    width, config.spatial_heads, config.geometric_neighbors
                )
                for index in indices
            }
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, observation: ObservationStep) -> Evidence:
        if not isinstance(observation, ObservationStep):
            raise TypeError("observation must be ObservationStep")
        cfg, obs = self.config, observation
        images = obs.images
        if images.ndim != 6 or not images.is_floating_point():
            raise ValueError("images must be floating [B,V,T,C,H,W]")
        b, views, frames, channels, height, width = images.shape
        if min(b, views, frames, height, width) < 1 or frames > cfg.tubelet_size:
            raise ValueError("invalid image/tubelet frames")
        if channels != cfg.image_channels:
            raise ValueError("image channels do not match config")
        device = images.device
        if (
            obs.time.shape != (b,)
            or not obs.time.is_floating_point()
            or not torch.isfinite(obs.time).all()
        ):
            raise ValueError("time must be finite floating [B]")
        for value in vars(obs).values():
            if isinstance(value, Tensor) and value.device != device:
                raise ValueError("observation tensors must share a device")
        valid = obs.valid
        if valid is None:
            valid = torch.ones(
                b, views, frames, 1, height, width, dtype=torch.bool, device=device
            )
        if (
            valid.shape != (b, views, frames, 1, height, width)
            or valid.dtype != torch.bool
        ):
            raise ValueError("valid must be boolean [B,V,T,1,H,W]")
        clean = torch.where(valid, images, torch.zeros_like(images))
        if not torch.isfinite(clean).all():
            raise ValueError("available image pixels must be finite")
        calibrated = obs.intrinsics is not None and obs.camera_c2w is not None
        if (obs.intrinsics is None) != (obs.camera_c2w is None):
            raise ValueError("camera intrinsics and camera_c2w are required together")
        geo_valid = obs.geometry_valid
        if geo_valid is None:
            geo_valid = torch.full(
                (b, views), calibrated, dtype=torch.bool, device=device
            )
        if geo_valid.shape != (b, views) or geo_valid.dtype != torch.bool:
            raise ValueError("geometry_valid must be boolean [B,V]")
        if not calibrated and geo_valid.any():
            raise ValueError("missing camera cannot have valid geometry")
        with torch.autocast(device_type=device.type, enabled=False):
            identity3 = torch.eye(3, device=device, dtype=torch.float32)
            identity4 = torch.eye(4, device=device, dtype=torch.float32)
            if calibrated:
                if obs.intrinsics.shape != (b, views, 3, 3) or obs.camera_c2w.shape != (
                    b,
                    views,
                    4,
                    4,
                ):
                    raise ValueError("invalid camera shape")
                intrinsics = torch.where(
                    geo_valid[..., None, None], obs.intrinsics, identity3
                ).float()
                poses = torch.where(
                    geo_valid[..., None, None], obs.camera_c2w, identity4
                ).float()
                if (
                    not torch.isfinite(intrinsics).all()
                    or not torch.isfinite(poses).all()
                ):
                    raise ValueError("valid camera must be finite")
                rotation = poses[..., :3, :3]
                if not torch.allclose(
                    rotation.transpose(-1, -2) @ rotation,
                    identity3.expand_as(rotation),
                    atol=1e-4,
                    rtol=1e-4,
                ):
                    raise ValueError("camera rotation must be orthonormal")
                if not torch.allclose(
                    torch.linalg.det(rotation),
                    torch.ones_like(geo_valid, dtype=torch.float32),
                    atol=1e-4,
                    rtol=1e-4,
                ):
                    raise ValueError("camera rotation determinant must be +1")
                if not torch.allclose(
                    poses[..., 3, :], identity4[3].expand(b, views, -1), atol=1e-5
                ):
                    raise ValueError("camera must have homogeneous last row")
                if (intrinsics[..., 0, 0] <= 0).any() or (
                    intrinsics[..., 1, 1] <= 0
                ).any():
                    raise ValueError("camera focal lengths must be positive")
                if not torch.allclose(
                    intrinsics[..., 2, :], identity3[2].expand(b, views, -1), atol=1e-5
                ):
                    raise ValueError("camera intrinsics must use z-depth convention")
            else:
                intrinsics = identity3.expand(b, views, 3, 3)
                poses = identity4.expand(b, views, 4, 4)
        p = cfg.patch_size
        gh, gw = math.ceil(height / p), math.ceil(width / p)
        pad = (0, gw * p - width, 0, gh * p - height)
        frame_coverage = F.avg_pool2d(
            F.pad(valid.float().reshape(b * views * frames, 1, height, width), pad),
            p,
            p,
        )
        coverage = frame_coverage.reshape(b, views, frames, gh, gw)[:, :, -1]
        token_valid = coverage.flatten(1) > 0
        x = clean.reshape(b * views, frames, channels, height, width).transpose(1, 2)
        x = F.pad(x, (*pad, cfg.tubelet_size - frames, 0))
        tokens = (
            self.patchify(x)
            .squeeze(2)
            .permute(0, 2, 3, 1)
            .reshape(b, views, gh, gw, cfg.evidence_width)
        )
        with torch.autocast(device_type=device.type, enabled=False):
            # Geometry samples the endpoint center pixel of each patch, without
            # averaging depths across object boundaries.
            yy = (torch.arange(gh, device=device) * p + p // 2).clamp(max=height - 1)
            xx = (torch.arange(gw, device=device) * p + p // 2).clamp(max=width - 1)
            y, xpixel = torch.meshgrid(yy, xx, indexing="ij")
            pixels = torch.stack(
                (
                    xpixel.float() + 0.5,
                    y.float() + 0.5,
                    torch.ones_like(xpixel).float(),
                ),
                -1,
            ).reshape(-1, 3)
            camera_rays = torch.einsum(
                "bvij,nj->bvni", torch.linalg.inv(intrinsics), pixels
            )
            world_rays = torch.einsum(
                "bvij,bvnj->bvni", poses[..., :3, :3], camera_rays
            )
            directions = F.normalize(world_rays, dim=-1).reshape(b, -1, 3)
            origins = (
                poses[..., :3, 3][:, :, None]
                .expand(-1, -1, gh * gw, -1)
                .reshape(b, -1, 3)
            )
            geometry_valid = (
                geo_valid[:, :, None].expand(-1, -1, gh * gw).reshape(b, -1)
                & token_valid
            )
            origins = torch.where(
                geometry_valid[..., None], origins, torch.zeros_like(origins)
            )
            directions = torch.where(
                geometry_valid[..., None], directions, torch.zeros_like(directions)
            )
            points, point_valid = None, None
            if obs.depth is not None:
                if (
                    obs.depth.shape != (b, views, 1, height, width)
                    or not obs.depth.is_floating_point()
                ):
                    raise ValueError("depth must be camera-z [B,V,1,H,W]")
                depth_valid = obs.depth_valid
                if depth_valid is None:
                    depth_valid = torch.isfinite(obs.depth) & (obs.depth > 0)
                if (
                    depth_valid.shape != obs.depth.shape
                    or depth_valid.dtype != torch.bool
                ):
                    raise ValueError("depth_valid must match depth")
                if (
                    not torch.isfinite(obs.depth[depth_valid]).all()
                    or (obs.depth[depth_valid] <= 0).any()
                ):
                    raise ValueError("valid camera-z depth must be finite and positive")
                z = torch.where(
                    depth_valid,
                    obs.depth.float(),
                    torch.zeros_like(obs.depth, dtype=torch.float32),
                )[:, :, 0, y, xpixel].reshape(b, views, -1)
                pv = depth_valid[:, :, 0, y, xpixel].reshape(b, -1)
                center_valid = valid[:, :, -1, 0, y, xpixel].reshape(b, -1)
                point_valid = pv & center_valid & geometry_valid
                points = origins + (world_rays * z[..., None]).reshape(b, -1, 3)
                points = torch.where(
                    point_valid[..., None], points, torch.zeros_like(points)
                )
            elif obs.depth_valid is not None:
                raise ValueError("depth_valid requires depth")
            coord = pixels[:, :2] / torch.tensor(
                [width, height], device=device, dtype=torch.float32
            )
            coord = coord[None, None].expand(b, views, -1, -1).reshape(b, -1, 2)
            phase = obs.time.float()[:, None].expand(b, views * gh * gw)
            view_index = (
                torch.arange(views, device=device, dtype=torch.float32)[None, :, None]
                .expand(b, -1, gh * gw)
                .reshape(b, -1)
            )
            features = torch.cat(
                (
                    coord,
                    phase.sin()[..., None],
                    phase.cos()[..., None],
                    geometry_valid[..., None].float(),
                    coverage.flatten(1)[..., None],
                    (view_index / max(views - 1, 1))[..., None],
                    torch.full_like(phase[..., None], frames / cfg.tubelet_size),
                ),
                -1,
            )
            geo_points = torch.zeros_like(origins) if points is None else points
            geom = torch.cat((origins, directions, geo_points), -1)
        tokens = (
            tokens.flatten(1, 3)
            + self.position(features)
            + self.geometry(torch.atan(geom))
        )
        tokens = torch.where(token_valid[..., None], tokens, torch.zeros_like(tokens))
        for index, block in enumerate(self.blocks):
            tokens = block(tokens.reshape(b, views, gh, gw, -1), coverage).reshape(
                b, -1, cfg.evidence_width
            )
            if str(index) in self.relays:
                tokens = self.relays[str(index)](tokens, token_valid)
                if points is not None:
                    tokens = self.geometric[str(index)](
                        tokens,
                        token_valid,
                        points,
                        point_valid,
                        intrinsics,
                        poses,
                        geo_valid,
                        gh,
                        gw,
                        p,
                    )
        tokens = torch.where(
            token_valid[..., None], self.norm(tokens), torch.zeros_like(tokens)
        ).float()
        result = Evidence(
            tokens,
            token_valid,
            origins.float(),
            directions.float(),
            obs.time.double().clone(),
            None if points is None else points.float(),
            point_valid,
            geometry_valid,
        )
        result.validate(cfg)
        return result
