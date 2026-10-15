"""Correspondence-graph types. Messages flow src -> dst; in-degree counts dst."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch import Tensor

from mavt4d.geometry.camera import invert_se3, transform_points


class EdgeType(IntEnum):
    SPATIAL = 0  # patch -> patch, same view-frame (scatter pass)
    TRACK = 1  # patch -> patch, same camera, |dt| = 1 (scatter pass)
    PROJ_P2H = 2  # patch -> hub (gather pass)
    PROJ_H2P = 3  # hub -> patch (scatter pass)
    NB3D = 4  # hub -> hub (gather pass)


class Provenance(IntEnum):
    GRID = 0
    CALIB_LIDAR = 1
    BOX = 2
    FLOW = 3


EDGE_ATTR_DIM = 10
# [log_depth, footprint=fx/depth, cos_incidence, x/z, y/z, log_z, dir_x, dir_y, dir_z, attr_valid]
# x/z, y/z, log_z: target point in the receiving camera's frame (PROJ: hub centre; TRACK: 3D point moved to
# the other frame). dir: unit direction camera centre -> point, world frame. SPATIAL, FLOW track, NB3D: all
# zero with attr_valid = 0.


@dataclass(frozen=True)
class TokenLayout:
    V: int
    T: int
    Hp: int
    Wp: int
    S: int = 0  # sweeps with hubs (0 without geometry)
    K: int = 0

    @property
    def n_cam(self) -> int:
        return self.V * self.T * self.Hp * self.Wp

    @property
    def n_hub(self) -> int:
        return self.S * self.K

    @property
    def num_nodes(self) -> int:
        return self.n_cam + self.n_hub

    def cam_index(self, v, t, r, c):
        return ((v * self.T + t) * self.Hp + r) * self.Wp + c

    def hub_index(self, s, k):
        return self.n_cam + s * self.K + k


@dataclass
class EdgeSet:
    src: Tensor  # (E,) int64
    dst: Tensor  # (E,) int64
    weight: Tensor  # (E,) float32 in (0, 1]
    attr: Tensor  # (E, EDGE_ATTR_DIM) float32
    provenance: Tensor  # (E,) int64

    @staticmethod
    def empty() -> "EdgeSet":
        i = torch.zeros(0, dtype=torch.int64)
        return EdgeSet(src=i, dst=i.clone(), weight=torch.zeros(0), attr=torch.zeros(0, EDGE_ATTR_DIM),
                       provenance=i.clone())


def cat_edges(sets: list[EdgeSet]) -> EdgeSet:
    """Concatenate edge sets (empty list -> empty set)."""
    if not sets:
        return EdgeSet.empty()
    return EdgeSet(**{f.name: torch.cat([getattr(e, f.name) for e in sets]) for f in fields(EdgeSet)})


@dataclass
class CorrespondenceGraph:
    layout: TokenLayout
    edges: dict[EdgeType, EdgeSet]  # always all 5 keys; absent types are EdgeSet.empty()
    num_graphs: int = 1  # > 1 after batch_graphs: node indices span num_graphs * layout.num_nodes

    def __post_init__(self) -> None:
        self.edges = {et: self.edges.get(et, EdgeSet.empty()) for et in EdgeType}

    def in_degree(self, etype: EdgeType) -> Tensor:
        """(num_graphs * num_nodes,) number of ``etype`` edges arriving at each node."""
        return torch.bincount(self.edges[etype].dst, minlength=self.num_graphs * self.layout.num_nodes)

    def to(self, device) -> "CorrespondenceGraph":
        moved = {et: EdgeSet(**{f.name: getattr(e, f.name).to(device) for f in fields(EdgeSet)})
                 for et, e in self.edges.items()}
        return CorrespondenceGraph(self.layout, moved, self.num_graphs)


@dataclass
class GraphStats:
    num_edges: dict[str, int]
    mean_in_degree: dict[str, float]  # over receiving nodes that can carry that edge type
    provenance: dict[str, dict[str, int]] = field(default_factory=dict)


def batch_graphs(graphs: list[CorrespondenceGraph]) -> CorrespondenceGraph:
    """Join graphs sharing one layout; graph i's node indices are shifted by i * num_nodes."""
    if not graphs:
        raise ValueError("graphs: need at least one graph")
    layout = graphs[0].layout
    if any(g.layout != layout for g in graphs):
        raise ValueError(f"layout: all graphs must share {layout}")
    edges: dict[EdgeType, list[EdgeSet]] = {et: [] for et in EdgeType}
    offset = 0
    for g in graphs:
        for et, e in g.edges.items():
            edges[et].append(EdgeSet(src=e.src + offset, dst=e.dst + offset, weight=e.weight, attr=e.attr,
                                     provenance=e.provenance))
        offset += g.num_graphs * layout.num_nodes
    return CorrespondenceGraph(layout, {et: cat_edges(v) for et, v in edges.items()},
                               sum(g.num_graphs for g in graphs))


def edge_attr(p_world: Tensor, normal: Tensor, K: Tensor, cam_to_world: Tensor) -> Tensor:
    """(E, EDGE_ATTR_DIM) attributes of points (E, 3) in per-edge cameras K (E, 3, 3), cam_to_world (E, 4, 4).

    ``normal`` (E, 3) is a world-frame surface normal (sign irrelevant). Depth is camera-frame z, so
    log_depth and log_z coincide. Requires z > 0.
    """
    p_cam = transform_points(invert_se3(cam_to_world), p_world[:, None])[:, 0]
    z = p_cam[:, 2]
    d = F.normalize(p_world - cam_to_world[:, :3, 3], dim=-1)
    cos = (F.normalize(normal, dim=-1) * d).sum(-1).abs()
    return torch.stack([z.log(), K[:, 0, 0] / z, cos, p_cam[:, 0] / z, p_cam[:, 1] / z, z.log(),
                        d[:, 0], d[:, 1], d[:, 2], torch.ones_like(z)], -1).float()


def rank_in_group(group: Tensor, key: Tensor) -> Tensor:
    """(E,) rank of each element among elements of the same group by ascending ``key`` (ties by position)."""
    if group.numel() == 0:
        return group.clone()
    order = torch.argsort(key, stable=True)
    order = order[torch.argsort(group[order], stable=True)]
    g = group[order]
    pos = torch.arange(len(g), device=g.device)
    start = torch.ones_like(g, dtype=torch.bool)
    start[1:] = g[1:] != g[:-1]
    first = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
    rank = torch.empty_like(pos)
    rank[order] = pos - first
    return rank
