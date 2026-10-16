"""C/D-only image-tile decoder; KV and observations are not accepted inputs.

Query FFNs and D modulation control routing, never supply RGB values. Every
content projection, including attention output and tile head, has no bias.
Consequently zero C produces exactly zero RGB, even with arbitrary D/query
metadata and learned biases elsewhere in the model.
"""

from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.contracts import DecoderQuery

from .config import HybridConfig
from .observation import _attention
from .state import CDCode


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def tile_grid_query(height: int, width: int, *, tile_size: int, time: float,
                    view_id: int = 0, device: torch.device | str | None = None,
                    intrinsics: Tensor | None = None, cam_to_world: Tensor | None = None) -> DecoderQuery:
    """Construct aligned tile centers in row-major order, including edge tiles."""
    for name, value in (("height", height), ("width", width), ("tile_size", tile_size)):
        _positive_int(name, value)
    if type(view_id) is not int or view_id < 0:
        raise ValueError("view_id must be a nonnegative integer")
    if isinstance(time, bool) or not isinstance(time, (float, int)) or not math.isfinite(time):
        raise ValueError("time must be finite")
    rows, columns = math.ceil(height / tile_size), math.ceil(width / tile_size)
    yy, xx = torch.meshgrid(torch.arange(rows, dtype=torch.float32, device=device),
                            torch.arange(columns, dtype=torch.float32, device=device), indexing="ij")
    centers = torch.stack(((xx + .5) * tile_size, (yy + .5) * tile_size, torch.zeros_like(xx)), dim=-1)
    count = rows * columns
    return DecoderQuery(centers.reshape(count, 3),
                        torch.full((count,), view_id, dtype=torch.int64, device=device),
                        torch.full((count,), float(time), dtype=torch.float64, device=device),
                        intrinsics=intrinsics, cam_to_world=cam_to_world)


def tiles_to_image(tiles: Tensor, *, height: int, width: int) -> Tensor:
    """Map row-major [B,Q,3,T,T] tiles to [B,3,H,W], cropping edge padding.

    Flattened tile channel index is c*T*T + dy*T + dx. No interpolation,
    blending, or query-order history is involved in reconstruction.
    """
    _positive_int("height", height)
    _positive_int("width", width)
    if not isinstance(tiles, Tensor) or tiles.ndim != 5 or tiles.shape[0] <= 0:
        raise ValueError("tiles must have nonempty [B,Q,3,T,T] shape")
    batch, count, channels, tile, tile_width = tiles.shape
    if channels != 3 or tile <= 0 or tile != tile_width:
        raise ValueError("tiles must have three channels and square nonempty tiles")
    rows, columns = math.ceil(height / tile), math.ceil(width / tile)
    if count != rows * columns:
        raise ValueError("tile count does not match image grid")
    image = tiles.reshape(batch, rows, columns, 3, tile, tile)
    image = image.permute(0, 3, 1, 4, 2, 5).reshape(batch, 3, rows * tile, columns * tile)
    return image[..., :height, :width].contiguous()


class _QueryBlock(nn.Module):
    """Pointwise query refinement followed by cross-attention to raw C values."""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.heads = config.q_heads
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(nn.Linear(config.width, config.width * 4), nn.GELU(),
                                 nn.Linear(config.width * 4, config.width))
        self.query_norm = nn.LayerNorm(config.width)
        self.query_projection = nn.Linear(config.width, config.width)
        self.content_keys = nn.Linear(config.state_dim, config.width, bias=False)
        self.content_values = nn.Linear(config.state_dim, config.width, bias=False)
        self.output_projection = nn.Linear(config.width, config.width, bias=False)
        self.d_routing = nn.Linear(config.modulation_dim, config.width, bias=False)
        nn.init.normal_(self.d_routing.weight, std=.02)

    def forward(self, queries: Tensor, c: Tensor, key_identity: Tensor,
                modulation: Tensor) -> tuple[Tensor, Tensor]:
        # The FFN precedes this block's attention, so even the final block's
        # FFN influences routing; it never becomes a direct content residual.
        queries = queries + self.ffn(self.ffn_norm(queries))
        routing = .1 * torch.tanh(self.d_routing(modulation))
        projected = self.query_projection(self.query_norm(queries + routing.unsqueeze(1)))
        keys = self.content_keys(c) + key_identity.unsqueeze(0)
        values = self.content_values(c)
        content = self.output_projection(_attention(projected, keys, values, self.heads))
        return queries + content, content


class HybridQueryDecoder(nn.Module):
    """Decode independent aligned tiles, or pixels selected from those tiles.

    The image_pixel coordinate API samples the containing raster pixel:
    floor(x), floor(y), including subpixel input. It does not claim continuous
    interpolation. Future queries require explicit state rollout first. World
    points and triplane UV grids need a separate specified output adapter.
    """

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        if not isinstance(config, HybridConfig):
            raise TypeError("config must be HybridConfig")
        self.config = config
        # xyz + sinusoidal xyz + geometry flag/ray/origin + tile size/view ID.
        self.coordinate_projection = nn.Linear(18, config.width)
        self.time_projection = nn.Linear(3, config.width, bias=False)
        self.stage_keys = nn.Parameter(torch.empty(config.stage_count, config.width))
        self.slot_keys = nn.Parameter(torch.empty(config.state_tokens, config.width))
        nn.init.normal_(self.stage_keys, std=1 / math.sqrt(config.width))
        nn.init.normal_(self.slot_keys, std=1 / math.sqrt(config.width))
        self.d_bottleneck = nn.Sequential(
            nn.Linear(config.stage_count * config.state_tokens * config.state_dim, 128),
            nn.GELU(), nn.Linear(128, config.modulation_dim), nn.Tanh(),
        )
        self.blocks = nn.ModuleList(_QueryBlock(config) for _ in range(config.decoder_blocks))
        self.d_gain = nn.Linear(config.modulation_dim, config.width, bias=False)
        nn.init.normal_(self.d_gain.weight, std=.02)
        self.tile_head = nn.Linear(config.width, 3 * config.decoder_tile**2, bias=False)

    def _validate(self, code: CDCode, query: DecoderQuery) -> None:
        if not isinstance(code, CDCode) or not isinstance(query, DecoderQuery):
            raise TypeError("decoder requires CDCode and DecoderQuery; KV state is not a decoder input")
        code.validate()
        query.validate()
        if code.config != self.config:
            raise ValueError("code config must match decoder config")
        if code.c.dtype != torch.float32 or self.tile_head.weight.dtype != torch.float32:
            raise ValueError("hybrid decoder requires FP32 C/D and weights")
        if code.c.device != self.tile_head.weight.device or code.c.device != query.coordinates.device:
            raise ValueError("code, query, and decoder weights must share a device")
        if query.coordinate_space != "image_pixel":
            raise ValueError("tile decoder supports only image_pixel queries")
        if not bool((query.coordinates[:, :2] >= 0).all()):
            raise ValueError("image_pixel coordinates must be nonnegative")
        if not bool((query.view_ids < self.config.reference_views).all()):
            raise ValueError("query view_ids exceed the configured view table")
        if bool((query.times.unsqueeze(0) > code.last_time.unsqueeze(1)).any()):
            raise ValueError("future query requires rollout before decoding")

    def _query_features(self, code: CDCode, query: DecoderQuery) -> Tensor:
        coordinates = query.coordinates.to(code.c.dtype)
        bounded = torch.tanh(coordinates / 64)
        geometry = coordinates.new_zeros(coordinates.shape[0], 7)
        if query.intrinsics is not None:
            intrinsics = query.intrinsics[query.view_ids].double()
            poses = query.cam_to_world[query.view_ids].double()
            pixels = torch.cat((query.coordinates[:, :2].double(),
                                torch.ones_like(query.coordinates[:, :1], dtype=torch.float64)), dim=-1)
            rays = torch.einsum("qij,qj->qi", torch.linalg.inv(intrinsics), pixels)
            world_rays = torch.einsum("qij,qj->qi", poses[:, :3, :3], rays)
            geometry[:, 0] = 1
            geometry[:, 1:4] = F.normalize(world_rays, dim=-1).to(code.c.dtype)
            geometry[:, 4:] = torch.tanh(poses[:, :3, 3] / 10).to(code.c.dtype)
        extent_view = torch.stack((coordinates.new_full((coordinates.shape[0],), self.config.decoder_tile / 64),
                                   torch.tanh(query.view_ids.to(code.c.dtype) / 16)), dim=-1)
        features = torch.cat((bounded, torch.sin(math.pi * bounded), torch.cos(math.pi * bounded),
                              geometry, extent_view), dim=-1)
        base = self.coordinate_projection(features)
        # Subtract before casting to preserve small intervals at large times.
        relative_time = (query.times.unsqueeze(0) - code.last_time.unsqueeze(1)).to(code.c.dtype)
        temporal = torch.stack((torch.tanh(relative_time), torch.sin(relative_time),
                                torch.cos(relative_time)), dim=-1)
        return base.unsqueeze(0) + self.time_projection(temporal)

    def decode_tiles(self, code: CDCode, query: DecoderQuery) -> Tensor:
        """Read [B,Q,3,T,T] tiles at aligned pixel-boundary tile centers."""
        self._validate(code, query)
        tile = self.config.decoder_tile
        grid_index = (query.coordinates[:, :2] - tile / 2) / tile
        if not torch.allclose(grid_index, grid_index.round(), rtol=0, atol=1e-6) or bool((grid_index < 0).any()):
            raise ValueError("tile query must use nonnegative aligned tile centers")
        queries = self._query_features(code, query)
        modulation = self.d_bottleneck(code.d.flatten(start_dim=1))
        identity = (self.stage_keys[:, None] + self.slot_keys[None]).reshape(-1, self.config.width)
        c = code.c.flatten(1, 2)
        content = None
        for block in self.blocks:
            queries, content = block(queries, c, identity, modulation)
        gain = 1 + .1 * torch.tanh(self.d_gain(modulation))
        rgb = self.tile_head(content * gain.unsqueeze(1))
        return rgb.reshape(code.c.shape[0], query.coordinates.shape[0], 3, tile, tile)

    def forward(self, code: CDCode, query: DecoderQuery) -> Tensor:
        """Return [B,Q,3] RGB by aligned-tile evaluation and pixel selection."""
        self._validate(code, query)
        tile = self.config.decoder_tile
        pixels = query.coordinates[:, :2].floor().long()
        tile_indices = torch.div(pixels, tile, rounding_mode="floor")
        centers = (tile_indices.to(query.coordinates.dtype) + .5) * tile
        tile_coordinates = torch.cat((centers, torch.zeros_like(centers[:, :1])), dim=-1)
        tiles = self.decode_tiles(code, replace(query, coordinates=tile_coordinates))
        offsets = pixels.remainder(tile)
        pixel_index = offsets[:, 1] * tile + offsets[:, 0]
        gather_index = pixel_index[None, :, None, None].expand(code.c.shape[0], -1, 3, 1)
        return tiles.flatten(-2).gather(-1, gather_index).squeeze(-1)

    def render(self, code: CDCode, *, height: int, width: int, view_id: int = 0,
               time: float | None = None, intrinsics: Tensor | None = None,
               cam_to_world: Tensor | None = None, chunk_size: int | None = None) -> Tensor:
        """Render a row-major tile grid and crop to an arbitrary image size."""
        if not isinstance(code, CDCode):
            raise TypeError("render requires CDCode")
        code.validate()
        if time is None:
            if not bool((code.last_time == code.last_time[0]).all()):
                raise ValueError("batched streams at different times require an explicit query time")
            time = float(code.last_time[0])
        query = tile_grid_query(height, width, tile_size=self.config.decoder_tile,
                                time=time, view_id=view_id, device=code.c.device,
                                intrinsics=intrinsics, cam_to_world=cam_to_world)
        if chunk_size is None:
            tiles = self.decode_tiles(code, query)
        else:
            _positive_int("chunk_size", chunk_size)
            chunks = []
            for start in range(0, query.coordinates.shape[0], chunk_size):
                indices = slice(start, start + chunk_size)
                part = replace(query, coordinates=query.coordinates[indices],
                               view_ids=query.view_ids[indices], times=query.times[indices])
                chunks.append(self.decode_tiles(code, part))
            tiles = torch.cat(chunks, dim=1)
        return tiles_to_image(tiles, height=height, width=width)
