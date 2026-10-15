"""Grid edges: spatial neighbours within a view-frame and the same-pixel temporal baseline."""
from __future__ import annotations

import torch
from torch import Tensor

from mavt4d.graph.types import EDGE_ATTR_DIM, EdgeSet, Provenance, TokenLayout


def _grid_edges(src: Tensor, dst: Tensor) -> EdgeSet:
    n = src.shape[0]
    return EdgeSet(src=src, dst=dst, weight=torch.ones(n), attr=torch.zeros(n, EDGE_ATTR_DIM),
                   provenance=torch.full((n,), int(Provenance.GRID), dtype=torch.int64))


def build_spatial_edges(layout: TokenLayout, radius: int = 2, cap: int = 24) -> EdgeSet:
    """Each patch receives from patches within Chebyshev ``radius`` in its own view-frame, no self-loop;
    at most ``cap`` neighbours, nearest (Euclidean in the grid) first."""
    Hp, Wp = layout.Hp, layout.Wp
    offsets = sorted(((dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
                      if (dr, dc) != (0, 0)), key=lambda o: (o[0] ** 2 + o[1] ** 2, o))[:cap]
    r, c = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
    r, c = r.flatten(), c.flatten()
    src, dst = [], []
    for dr, dc in offsets:
        ok = (r + dr >= 0) & (r + dr < Hp) & (c + dc >= 0) & (c + dc < Wp)
        src.append(((r + dr) * Wp + c + dc)[ok])
        dst.append((r * Wp + c)[ok])
    frames = torch.arange(layout.V * layout.T)[:, None] * (Hp * Wp)
    return _grid_edges((frames + torch.cat(src)).flatten(), (frames + torch.cat(dst)).flatten())


def build_same_pixel_track_edges(layout: TokenLayout, cap: int = 2) -> EdgeSet:
    """Baseline (iv): each patch receives from the same (r, c) of its camera at t-1 and t+1 (in that order,
    truncated to ``cap``)."""
    idx = torch.arange(layout.n_cam).reshape(layout.V, layout.T, layout.Hp, layout.Wp)
    pairs = [(idx[:, :-1], idx[:, 1:]), (idx[:, 1:], idx[:, :-1])][:cap]  # (src, dst): from t-1, from t+1
    return _grid_edges(torch.cat([s.flatten() for s, _ in pairs]), torch.cat([d.flatten() for _, d in pairs]))
