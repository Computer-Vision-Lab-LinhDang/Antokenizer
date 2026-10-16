"""Read C-valued content using bounded D-dependent query routing/modulation.

These are readout heads, not state transitions. No observed pixels, prior
queries, or per-stream caches enter either head. Restricting D's path is an
inductive bias; it does not establish semantic disentanglement by itself.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.contracts import CDState, CDStateSpec, DecoderQuery


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected positive integer")


class StructuredQueryDecoder(nn.Module):
    """Query-independent RGB readout: C supplies values; D routes and scales.

    D is compressed into a small tanh bottleneck. It supplies bounded additive
    query routing and multiplicative content gain near one. Queries have no
    self-attention. Crucially neither queries nor D have a residual route to
    RGB: the RGB head sees only the weighted sum of bias-free C values, then
    multiplicative gain. Consequently C=0 makes RGB independent of D and query
    metadata, even after training. Nonzero C still permits D to encode through
    routing/modulation, so this constraint alone is not a semantic guarantee.
    """

    def __init__(self, spec: CDStateSpec, hidden_dim: int = 32, modulation_dim: int = 8) -> None:
        super().__init__()
        if not isinstance(spec, CDStateSpec):
            raise TypeError("spec: expected CDStateSpec")
        _positive_int("hidden_dim", hidden_dim)
        _positive_int("modulation_dim", modulation_dim)
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.modulation_dim = modulation_dim
        self.coordinate_projection = nn.Linear(16, hidden_dim)
        self.time_projection = nn.Linear(3, hidden_dim, bias=False)
        self.space_embedding = nn.Embedding(3, hidden_dim)
        self.plane_embedding = nn.Embedding(3, hidden_dim)
        self.query_projection = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.content_keys = nn.Linear(spec.token_dim, hidden_dim, bias=False)
        self.content_values = nn.Linear(spec.token_dim, hidden_dim, bias=False)
        self.slot_keys = nn.Parameter(torch.empty(spec.c_tokens, hidden_dim))
        nn.init.normal_(self.slot_keys, std=1 / math.sqrt(hidden_dim))
        self.d_bottleneck = nn.Linear(spec.d_tokens * spec.token_dim, modulation_dim, bias=False)
        self.d_routing = nn.Linear(modulation_dim, hidden_dim, bias=False)
        self.d_gain = nn.Linear(modulation_dim, hidden_dim, bias=False)
        nn.init.normal_(self.d_routing.weight, std=.02)
        nn.init.normal_(self.d_gain.weight, std=.02)
        self.rgb = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def _query_features(self, query: DecoderQuery, state: CDState) -> Tensor:
        dtype = state.c.dtype
        coordinates = query.coordinates.to(dtype)
        space = {"image_pixel": 0, "world_xyz": 1, "triplane_uv": 2}[query.coordinate_space]
        scale = (64., 10., 1.)[space]
        bounded_coordinates = torch.tanh(coordinates / scale) if space != 2 else coordinates
        geometric = coordinates.new_zeros(coordinates.shape[0], 7)
        if query.intrinsics is not None:
            intrinsics = query.intrinsics[query.view_ids].double()
            poses = query.cam_to_world[query.view_ids].double()
            if space == 0:
                pixels = torch.cat((query.coordinates[:, :2].double(),
                                    torch.ones_like(query.coordinates[:, :1], dtype=torch.float64)), dim=-1)
                camera_rays = torch.einsum("qij,qj->qi", torch.linalg.inv(intrinsics), pixels)
                directions = torch.einsum("qij,qj->qi", poses[:, :3, :3], camera_rays)
            else:
                # World coordinates already specify a point; the explicit
                # camera table supplies the camera-to-point direction/origin.
                directions = query.coordinates.double() - poses[:, :3, 3]
            geometric[:, 0] = 1
            geometric[:, 1:4] = F.normalize(directions, dim=-1).to(dtype)
            geometric[:, 4:] = torch.tanh(poses[:, :3, 3] / 10).to(dtype)
        features = torch.cat((bounded_coordinates, torch.sin(math.pi * bounded_coordinates),
                              torch.cos(math.pi * bounded_coordinates), geometric), dim=-1)
        base = self.coordinate_projection(features)
        space_index = torch.tensor(space, device=state.c.device)
        base = base + self.space_embedding(space_index)
        if space == 2:
            base = base + self.plane_embedding(query.view_ids)
        # Subtract while timestamps are still float64, before model casting.
        relative_time = (query.times.unsqueeze(0) - state.last_time.unsqueeze(1)).to(dtype)
        temporal = torch.stack((torch.tanh(relative_time), torch.sin(relative_time),
                                torch.cos(relative_time)), dim=-1)
        return self.query_projection(base.unsqueeze(0) + self.time_projection(temporal))

    def forward(self, state: CDState, query: DecoderQuery) -> Tensor:
        if not isinstance(state, CDState) or not isinstance(query, DecoderQuery):
            raise TypeError("decoder expects CDState and DecoderQuery")
        state.validate()
        query.validate()
        if state.spec != self.spec:
            raise ValueError("state spec must match decoder spec")
        if state.c.dtype not in (torch.float32, torch.float64) or state.c.dtype != self.content_values.weight.dtype:
            raise ValueError("state dtype must be float32 or float64 and match decoder weights")
        if state.c.device != self.content_values.weight.device or query.coordinates.device != state.c.device:
            raise ValueError("state, query, and decoder weights must share a device")
        if query.coordinates.dtype not in (torch.float32, torch.float64):
            raise ValueError("reference query coordinate dtype must be float32 or float64")
        if bool((query.times.unsqueeze(0) > state.last_time.unsqueeze(1)).any()):
            raise ValueError("future query requires rollout of the state before decoding")
        queries = self._query_features(query, state)
        modulation = torch.tanh(self.d_bottleneck(state.d.flatten(start_dim=1)))
        routing = .1 * torch.tanh(self.d_routing(modulation))
        gain = 1 + .1 * torch.tanh(self.d_gain(modulation))
        queries = queries + routing.unsqueeze(1)
        keys = self.content_keys(state.c) + self.slot_keys.unsqueeze(0)
        values = self.content_values(state.c)
        weights = (queries @ keys.transpose(-1, -2) / math.sqrt(self.hidden_dim)).softmax(dim=-1)
        content = (weights @ values) * gain.unsqueeze(1)
        return self.rgb(content)


class CTemporalHead(nn.Module):
    """Temporal training/probe head whose only input is the C tensor."""

    def __init__(self, spec: CDStateSpec, output_dim: int = 8) -> None:
        super().__init__()
        if not isinstance(spec, CDStateSpec):
            raise TypeError("spec: expected CDStateSpec")
        _positive_int("output_dim", output_dim)
        self.spec = spec
        self.norm = nn.LayerNorm(spec.token_dim)
        self.projection = nn.Linear(spec.c_tokens * spec.token_dim, output_dim)

    def forward(self, c: Tensor) -> Tensor:
        if not isinstance(c, Tensor) or c.ndim != 3 or c.shape[0] <= 0 or tuple(c.shape[1:]) != (self.spec.c_tokens, self.spec.token_dim):
            raise ValueError("c: expected nonempty [B,C_tokens,token_dim]")
        if c.dtype not in (torch.float32, torch.float64) or c.dtype != self.projection.weight.dtype:
            raise ValueError("c dtype must be float32 or float64 and match head weights")
        if c.device != self.projection.weight.device:
            raise ValueError("c device must match head weights")
        if not bool(torch.isfinite(c).all()):
            raise ValueError("c must contain finite values")
        return self.projection(self.norm(c).flatten(start_dim=1))
