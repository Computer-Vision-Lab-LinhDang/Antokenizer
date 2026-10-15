"""Projection edges (patch <-> hub) and 3D-neighbour edges (hub -> hub).

A hub's members are projected into every view-frame paired with the hub's sweep (nearest in time) and
z-buffered against all train points of that sweep. Each (hub, patch) pair with visible members becomes one
PROJ_P2H and one PROJ_H2P edge, weight = visible members in the patch / M, attributes from the hub centre.
Pairs whose hub surface is seen at grazing incidence (|PCA normal . view ray| < ``min_cos_incidence``) are
dropped before the degree caps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mavt4d.data.sample import SceneSample
from mavt4d.geometry.camera import pixel_to_patch, project_visible, transform_points
from mavt4d.geometry.hubs import HubGroups, pca_normals
from mavt4d.graph.track import sweep_for_frame
from mavt4d.graph.types import EDGE_ATTR_DIM, EdgeSet, Provenance, TokenLayout, edge_attr, rank_in_group

if TYPE_CHECKING:
    from mavt4d.graph.builder import GraphConfig


def visible_members(sample: SceneSample, hubs: HubGroups, s: int, v: int, t: int,
                    cfg: GraphConfig) -> tuple[Tensor, Tensor, Tensor]:
    """Members of sweep ``s``'s hubs, flattened (K*M,), projected into view-frame (v, t): uv, depth, visible."""
    pts = transform_points(sample.lidar_to_world[s], sample.input_points(s)[:, :3])
    H, W = sample.images.shape[-2:]
    return project_visible(pts[hubs.member_idx[s].flatten()], pts, sample.intrinsics[v], sample.cam_to_world[v, t],
                           H, W, cfg.zbuffer_cell_px, cfg.zbuffer_tol_m, cfg.zbuffer_tol_rel)


def build_projection_edges(sample: SceneSample, hubs: HubGroups, layout: TokenLayout,
                           cfg: GraphConfig) -> tuple[EdgeSet, EdgeSet]:
    """(PROJ_P2H, PROJ_H2P) over one shared pair set, degree-capped by weight: hub side ``proj_hub_cap``,
    patch side ``proj_patch_cap``; a pair is kept only if it passes both."""
    S, K, M = hubs.member_idx.shape
    n_patch = layout.Hp * layout.Wp
    R = sample.lidar_to_world[:, :3, :3]
    normals = pca_normals(hubs.rel_xyz.reshape(S * K, M, 3)).reshape(S, K, 3) @ R.transpose(1, 2)  # world
    cam, hub, weight, attr = [], [], [], []
    for v in range(layout.V):
        for t in range(layout.T):
            s = sweep_for_frame(sample, v, t)
            uv, _, vis = visible_members(sample, hubs, s, v, t, cfg)
            rc, inside = pixel_to_patch(uv, cfg.patch, layout.Hp, layout.Wp)
            ok = vis & inside
            k = torch.arange(K).repeat_interleave(M)[ok]
            pair, count = torch.unique(k * n_patch + rc[ok, 0] * layout.Wp + rc[ok, 1], return_counts=True)
            k, patch = pair // n_patch, pair % n_patch
            n = len(pair)
            a = edge_attr(hubs.centers_world[s, k], normals[s, k], sample.intrinsics[v].expand(n, 3, 3),
                          sample.cam_to_world[v, t].expand(n, 4, 4))
            front = a[:, 5].isfinite() & (a[:, 2] >= cfg.min_cos_incidence)  # in front, not grazing
            cam.append(layout.cam_index(v, t, patch // layout.Wp, patch % layout.Wp)[front])
            hub.append(layout.hub_index(s, k)[front])
            weight.append(count[front].float() / M)
            attr.append(a[front])
    if not cam:
        return EdgeSet.empty(), EdgeSet.empty()
    cam, hub, weight, attr = torch.cat(cam), torch.cat(hub), torch.cat(weight), torch.cat(attr)
    keep = (rank_in_group(hub, -weight) < cfg.proj_hub_cap) & (rank_in_group(cam, -weight) < cfg.proj_patch_cap)
    cam, hub, weight, attr = cam[keep], hub[keep], weight[keep], attr[keep]
    prov = torch.full((len(cam),), int(Provenance.CALIB_LIDAR), dtype=torch.int64)
    return (EdgeSet(src=cam, dst=hub, weight=weight, attr=attr, provenance=prov),
            EdgeSet(src=hub.clone(), dst=cam.clone(), weight=weight.clone(), attr=attr.clone(), provenance=prov.clone()))


def build_nb3d_edges(hubs: HubGroups, layout: TokenLayout, k: int = 16) -> EdgeSet:
    """Each hub receives from its ``k`` nearest hubs of the same sweep (no self-loop); attributes are zero."""
    S, K = hubs.centers_world.shape[:2]
    k = min(k, K - 1)
    src, dst = [], []
    for s in range(S):
        c = hubs.centers_world[s]
        d = torch.cdist(c, c, compute_mode="donot_use_mm_for_euclid_dist").fill_diagonal_(float("inf"))
        nb = d.topk(k, dim=1, largest=False).indices  # (K, k)
        src.append(layout.hub_index(s, nb.flatten()))
        dst.append(layout.hub_index(s, torch.arange(K).repeat_interleave(k)))
    src, dst = torch.cat(src), torch.cat(dst)
    n = len(src)
    return EdgeSet(src=src, dst=dst, weight=torch.ones(n), attr=torch.zeros(n, EDGE_ATTR_DIM),
                   provenance=torch.full((n,), int(Provenance.CALIB_LIDAR), dtype=torch.int64))
