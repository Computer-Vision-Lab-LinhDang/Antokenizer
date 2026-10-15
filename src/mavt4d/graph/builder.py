"""Graph assembly per tier, graph statistics and training correspondences.

Tier C: spatial. Tier B: spatial + track (flow). Tier A with hubs: all five types. Tier A without hubs
(camera-only variant): spatial + track. ``track_mode="same_pixel"`` swaps track edges for the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from mavt4d.data.sample import SceneSample
from mavt4d.geometry.camera import pixel_to_patch, project_visible, transform_points
from mavt4d.geometry.hubs import HubGroups
from mavt4d.graph.projection import build_nb3d_edges, build_projection_edges
from mavt4d.graph.spatial import build_same_pixel_track_edges, build_spatial_edges
from mavt4d.graph.track import build_track_edges, sweep_for_frame
from mavt4d.graph.types import CorrespondenceGraph, EdgeType, GraphStats, Provenance, TokenLayout


@dataclass
class GraphConfig:
    patch: int = 16
    spatial_radius: int = 2
    spatial_cap: int = 24
    track_cap: int = 2
    track_mode: Literal["correspondence", "same_pixel"] = "correspondence"
    proj_hub_cap: int = 32
    proj_patch_cap: int = 8
    zbuffer_tol_m: float = 0.5
    zbuffer_cell_px: int = 4
    nb3d_k: int = 16
    hubs_K: int = 256
    hubs_M: int = 32


def build_graph(sample: SceneSample, hubs: HubGroups | None, cfg: GraphConfig) -> tuple[CorrespondenceGraph,
                                                                                       GraphStats]:
    V, T, _, H, W = sample.images.shape
    if H % cfg.patch or W % cfg.patch:
        raise ValueError(f"images: {H}x{W} is not a multiple of patch {cfg.patch}")
    if hubs is not None and sample.tier != "A":
        raise ValueError("hubs: projection edges need a tier-A sample (LiDAR + intrinsics + cam_to_world)")
    S, K = hubs.centers_world.shape[:2] if hubs is not None else (0, 0)
    layout = TokenLayout(V=V, T=T, Hp=H // cfg.patch, Wp=W // cfg.patch, S=S, K=K)
    edges = {EdgeType.SPATIAL: build_spatial_edges(layout, cfg.spatial_radius, cfg.spatial_cap)}
    if T > 1:
        edges[EdgeType.TRACK] = (build_same_pixel_track_edges(layout, cfg.track_cap) if cfg.track_mode == "same_pixel"
                                 else build_track_edges(sample, layout, cfg))
    if hubs is not None:
        edges[EdgeType.PROJ_P2H], edges[EdgeType.PROJ_H2P] = build_projection_edges(sample, hubs, layout, cfg)
        edges[EdgeType.NB3D] = build_nb3d_edges(hubs, layout, cfg.nb3d_k)
    graph = CorrespondenceGraph(layout, edges)
    return graph, graph_stats(graph)


def graph_stats(graph: CorrespondenceGraph) -> GraphStats:
    """Edge counts, mean in-degree over nodes that can receive the type, and provenance counts.

    Eligible receivers: SPATIAL, PROJ_H2P (only if there are hubs) and TRACK (only if T > 1) -> camera
    tokens; PROJ_P2H and NB3D -> hubs.
    """
    L, n = graph.layout, graph.num_graphs
    eligible = {EdgeType.SPATIAL: L.n_cam, EdgeType.TRACK: L.n_cam if L.T > 1 else 0,
                EdgeType.PROJ_P2H: L.n_hub, EdgeType.PROJ_H2P: L.n_cam if L.n_hub else 0, EdgeType.NB3D: L.n_hub}
    num, mean, prov = {}, {}, {}
    for et, e in graph.edges.items():
        num[et.name] = len(e.src)
        mean[et.name] = len(e.src) / (n * eligible[et]) if eligible[et] else 0.0
        vals, counts = torch.unique(e.provenance, return_counts=True)
        prov[et.name] = {Provenance(int(p)).name: int(c) for p, c in zip(vals, counts)}
    return GraphStats(num_edges=num, mean_in_degree=mean, provenance=prov)


def train_correspondences(sample: SceneSample, layout: TokenLayout, cfg: GraphConfig, max_pairs: int = 4096,
                          seed: int = 0) -> Tensor:
    """(P, 2) camera-token pairs in two different view-frames that both see one LiDAR train point (z-buffered).

    Only view-frames paired with the same sweep are compared (same capture time), so a single-camera clip
    yields none. Pairs are unique, sorted, and subsampled to ``max_pairs`` with a seeded generator.
    """
    empty = torch.zeros(0, 2, dtype=torch.int64)
    if not sample.lidar_points or sample.intrinsics is None or sample.cam_to_world is None:
        return empty
    H, W = sample.images.shape[-2:]
    pairs = []
    for s in range(len(sample.lidar_points)):
        frames = [(v, t) for v in range(layout.V) for t in range(layout.T) if sweep_for_frame(sample, v, t) == s]
        pts = transform_points(sample.lidar_to_world[s], sample.input_points(s)[:, :3])
        if len(frames) < 2 or len(pts) == 0:
            continue
        tok = []
        for v, t in frames:
            uv, _, vis = project_visible(pts, pts, sample.intrinsics[v], sample.cam_to_world[v, t], H, W,
                                         cfg.zbuffer_cell_px, cfg.zbuffer_tol_m)
            rc, inside = pixel_to_patch(uv, cfg.patch, layout.Hp, layout.Wp)
            tok.append(torch.where(vis & inside, layout.cam_index(v, t, rc[:, 0], rc[:, 1]), -1))
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                both = (tok[i] >= 0) & (tok[j] >= 0)
                pairs.append(torch.stack([tok[i][both], tok[j][both]], 1))
    if not pairs:
        return empty
    pairs = torch.unique(torch.cat(pairs), dim=0)
    if len(pairs) > max_pairs:
        pick = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed))[:max_pairs]
        pairs = pairs[pick.sort().values]
    return pairs
