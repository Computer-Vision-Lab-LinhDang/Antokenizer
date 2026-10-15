"""Track edges: patch -> same camera's patch at t +- 1, built from LiDAR train points or flow.

For each source patch and target frame, the first available source wins: BOX (the patch's median-depth
LiDAR point lies in a box: carry the point with the box's motion), CALIB_LIDAR (treat the point as static
in the world), FLOW (``sample.patch_flow``). A patch receives at most ``track_cap // 2`` edges from each
temporal direction, keeping the sources that land nearest its centre.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mavt4d.data.sample import SceneSample
from mavt4d.geometry.boxes import interpolate_boxes, move_with_boxes, points_in_boxes
from mavt4d.geometry.camera import pixel_to_patch, project, project_visible, transform_points, unproject
from mavt4d.geometry.hubs import knn, pca_normals
from mavt4d.graph.types import EDGE_ATTR_DIM, EdgeSet, Provenance, TokenLayout, edge_attr, rank_in_group

if TYPE_CHECKING:
    from mavt4d.graph.builder import GraphConfig

NORMAL_K = 8  # LiDAR neighbours for the PCA normal behind cos_incidence


def sweep_for_frame(sample: SceneSample, v: int, t: int) -> int:
    """Index of the LiDAR sweep nearest in time to view-frame (v, t)."""
    return int(torch.argmin((sample.lidar_times - sample.frame_times[v, t]).abs()))


def move_points(sample: SceneSample, p: Tensor, inst: Tensor, tau0: float, tau1: float) -> Tensor:
    """Carry world points of box ``inst`` (-1 = static) from time tau0 to tau1. A frame's box time is
    ``frame_times[0, t]``."""
    if not sample.boxes:
        return p
    times = sample.frame_times[0]
    return move_with_boxes(p, inst, interpolate_boxes(sample.boxes, times, tau0),
                           interpolate_boxes(sample.boxes, times, tau1))


def _centres(rc: Tensor, patch: int) -> Tensor:
    return torch.stack([(rc[:, 1] + 0.5) * patch, (rc[:, 0] + 0.5) * patch], -1).float()


def _lidar_reps(sample: SceneSample, layout: TokenLayout, cfg: GraphConfig, v: int, t: int):
    """Per patch of view-frame (v, t) with visible LiDAR train points: patch ids, representative world point
    (patch centre at the median depth), its box instance (-1 = none) and a world normal. None if no patch."""
    s = sweep_for_frame(sample, v, t)
    pts = sample.input_points(s)
    if len(pts) == 0:
        return None
    tau_s, tau = float(sample.lidar_times[s]), float(sample.frame_times[v, t])
    p = transform_points(sample.lidar_to_world[s], pts[:, :3])
    inst = (points_in_boxes(p, interpolate_boxes(sample.boxes, sample.frame_times[0], tau_s)) if sample.boxes
            else torch.full((len(p),), -1, dtype=torch.int64))
    p = move_points(sample, p, inst, tau_s, tau)
    K, c2w = sample.intrinsics[v], sample.cam_to_world[v, t]
    H, W = sample.images.shape[-2:]
    uv, depth, vis = project_visible(p, p, K, c2w, H, W, cfg.zbuffer_cell_px, cfg.zbuffer_tol_m)
    rc, inside = pixel_to_patch(uv, cfg.patch, layout.Hp, layout.Wp)
    idx = (vis & inside).nonzero().flatten()
    if len(idx) == 0:
        return None
    patch = rc[idx, 0] * layout.Wp + rc[idx, 1]
    order = torch.argsort(depth[idx], stable=True)
    order = order[torch.argsort(patch[order], stable=True)]  # grouped by patch, depth ascending
    counts = torch.bincount(patch, minlength=layout.Hp * layout.Wp)
    ids = counts.nonzero().flatten()
    starts = torch.cumsum(counts, 0) - counts
    med = idx[order[starts[ids] + (counts[ids] - 1) // 2]]  # median-depth point of each patch
    rc_ids = torch.stack([ids // layout.Wp, ids % layout.Wp], -1)
    rep = unproject(_centres(rc_ids, cfg.patch), depth[med], K, c2w)
    normal = pca_normals(p[knn(p[med], p, min(NORMAL_K, len(p)))])
    return ids, rep, inst[med], normal


def build_track_edges(sample: SceneSample, layout: TokenLayout, cfg: GraphConfig) -> EdgeSet:
    V, T, Hp, Wp, P = layout.V, layout.T, layout.Hp, layout.Wp, cfg.patch
    cand: dict[str, list[Tensor]] = {k: [] for k in ("src", "dst", "dist", "prov", "attr")}

    def add(src, dst, dist, prov, attr=None):
        cand["src"].append(src)
        cand["dst"].append(dst)
        cand["dist"].append(dist)
        cand["prov"].append(prov)
        cand["attr"].append(torch.zeros(len(src), EDGE_ATTR_DIM) if attr is None else attr)

    has_lidar = torch.zeros(V, T, Hp * Wp, dtype=torch.bool)
    geometric = sample.lidar_points and sample.intrinsics is not None and sample.cam_to_world is not None
    for v in range(V if geometric and T > 1 else 0):
        K = sample.intrinsics[v]
        for t in range(T):
            reps = _lidar_reps(sample, layout, cfg, v, t)
            if reps is None:
                continue
            ids, rep, inst, normal = reps
            has_lidar[v, t, ids] = True
            for t2 in (t - 1, t + 1):
                if not 0 <= t2 < T:
                    continue
                tau, tau2 = float(sample.frame_times[v, t]), float(sample.frame_times[v, t2])
                p2 = move_points(sample, rep, inst, tau, tau2)
                n2 = move_points(sample, rep + normal, inst, tau, tau2) - p2
                uv, z = project(p2, K, sample.cam_to_world[v, t2])
                rc, inside = pixel_to_patch(uv, P, Hp, Wp)
                ok = inside & (z > 0)
                n = int(ok.sum())
                prov = torch.where(inst[ok] >= 0, int(Provenance.BOX), int(Provenance.CALIB_LIDAR))
                add(layout.cam_index(v, t, ids[ok] // Wp, ids[ok] % Wp),
                    layout.cam_index(v, t2, rc[ok, 0], rc[ok, 1]),
                    (uv[ok] - _centres(rc[ok], P)).norm(dim=-1), prov,
                    edge_attr(p2[ok], n2[ok], K.expand(n, 3, 3), sample.cam_to_world[v, t2].expand(n, 4, 4)))

    if sample.patch_flow is not None:
        rc0 = torch.stack(torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij"), -1).reshape(-1, 2)
        for v in range(V):
            for t in range(T - 1):
                f = sample.patch_flow[v, t].reshape(-1, 3)
                uv = _centres(rc0, P) + f[:, :2] * P
                rc, inside = pixel_to_patch(uv, P, Hp, Wp)
                ok = inside & (f[:, 2] > 0.5)
                land = torch.where(ok, rc[:, 0] * Wp + rc[:, 1], 0)
                dist = (uv - _centres(rc, P)).norm(dim=-1)
                here = layout.cam_index(v, t, rc0[:, 0], rc0[:, 1])
                there = layout.cam_index(v, t + 1, rc[:, 0], rc[:, 1])
                fwd = ok & ~has_lidar[v, t]  # forward edge: source is the flow's own patch
                rev = ok & ~has_lidar[v, t + 1, land]  # backward edge: source is where the flow lands
                flow = int(Provenance.FLOW)
                add(here[fwd], there[fwd], dist[fwd], torch.full((int(fwd.sum()),), flow))
                add(there[rev], here[rev], dist[rev], torch.full((int(rev.sum()),), flow))

    if not cand["src"]:
        return EdgeSet.empty()
    src, dst, dist, prov, attr = (torch.cat(cand[k]) for k in ("src", "dst", "dist", "prov", "attr"))
    from_future = (src // (Hp * Wp)) % T > (dst // (Hp * Wp)) % T
    keep = rank_in_group(dst * 2 + from_future.long(), dist) < cfg.track_cap // 2
    return EdgeSet(src=src[keep], dst=dst[keep], weight=torch.ones(int(keep.sum())), attr=attr[keep],
                   provenance=prov[keep].long())
