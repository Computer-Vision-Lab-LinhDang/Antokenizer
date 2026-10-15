"""Checks of the projection edges on real data (E1.1 / plan T27).

``recheck_projection_edges``: every (hub, patch) pair of PROJ_P2H must have at least one hub member whose
float64 re-projection, computed apart from the builder's float32 path, lands in that patch in front of the
camera (spec T11, test 1); PROJ_H2P must be the exact reverse. Where the hub centre lands is reported for
information only: edges come from members, so a hub spanning several patches links to all of them.

``false_visible_counts``: members the graph uses (visible to the builder's z-buffer and inside the patch of
a kept edge) that the same z-buffer rule (cell, tol) rejects against a denser reference cloud.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mavt4d.data.sample import SceneSample
from mavt4d.geometry.camera import pixel_to_patch, project, transform_points, zbuffer_visibility
from mavt4d.geometry.hubs import HubGroups
from mavt4d.graph.projection import visible_members
from mavt4d.graph.track import sweep_for_frame
from mavt4d.graph.types import CorrespondenceGraph, EdgeType

if TYPE_CHECKING:
    from mavt4d.graph.builder import GraphConfig

DEPTH_BINS_M = (10.0, 20.0, 40.0)  # member depth bins [0, 10), [10, 20), [20, 40), [40, inf)
CHEB_MAX = 64  # last bin of the centre-distance histogram collects everything at or beyond it


def _decode(graph: CorrespondenceGraph, cam: Tensor, hub: Tensor) -> tuple[Tensor, ...]:
    """(v, t, row, col, sweep, hub-in-sweep) of camera-token and hub node indices."""
    L = graph.layout
    frame, rc = cam // (L.Hp * L.Wp), cam % (L.Hp * L.Wp)
    return frame // L.T, frame % L.T, rc // L.Wp, rc % L.Wp, (hub - L.n_cam) // L.K, (hub - L.n_cam) % L.K


def recheck_projection_edges(sample: SceneSample, hubs: HubGroups, graph: CorrespondenceGraph,
                             cfg: GraphConfig) -> dict:
    """Counts over PROJ_P2H edges: ``passed`` (some member re-projects into the edge's patch), the hub centre's
    Chebyshev patch distance to the edge's patch as a histogram, and for failed edges the distance in px from
    their nearest in-front member to the patch rectangle (to tell a float border flip from a real miss)."""
    p2h, h2p = graph.edges[EdgeType.PROJ_P2H], graph.edges[EdgeType.PROJ_H2P]
    v, t, r, c, s, k = _decode(graph, p2h.src, p2h.dst)
    E = len(v)
    passed = torch.zeros(E, dtype=torch.bool)
    gap = torch.zeros(E, dtype=torch.float64)
    cheb = torch.zeros(E, dtype=torch.int64)
    l2w = sample.lidar_to_world.double()
    for vv, tt, ss in sorted({(int(a), int(b), int(x)) for a, b, x in zip(v, t, s)}):
        e = ((v == vv) & (t == tt) & (s == ss)).nonzero().flatten()
        K, c2w = sample.intrinsics[vv].double(), sample.cam_to_world[vv, tt].double()
        members = transform_points(l2w[ss], sample.input_points(ss)[:, :3].double()[hubs.member_idx[ss, k[e]]])
        uv, d = project(members.reshape(-1, 3), K, c2w)
        uv, d = uv.reshape(len(e), -1, 2), d.reshape(len(e), -1)
        col, row = c[e, None], r[e, None]
        cr = torch.floor(uv / cfg.patch)
        passed[e] = ((d > 0) & (cr[..., 0] == col) & (cr[..., 1] == row)).any(1)
        lo = torch.stack([col, row], -1).double() * cfg.patch
        out = torch.maximum(lo - uv, uv - (lo + cfg.patch)).clamp(min=0).norm(dim=-1)
        gap[e] = torch.where(d > 0, out, torch.full_like(out, float("inf"))).min(1).values
        centre, _ = project(transform_points(l2w[ss], hubs.centers_sensor[ss, k[e]].double()), K, c2w)
        cc = torch.floor(centre / cfg.patch).long()
        cheb[e] = torch.maximum((cc[:, 0] - c[e]).abs(), (cc[:, 1] - r[e]).abs())
    return {"edges": E, "passed": int(passed.sum()),
            "h2p_is_reverse": torch.equal(p2h.src, h2p.dst) and torch.equal(p2h.dst, h2p.src),
            "centre_in_patch": int((cheb == 0).sum()),
            "centre_cheb_hist": torch.bincount(cheb.clamp(0, CHEB_MAX), minlength=CHEB_MAX + 1).tolist(),
            "fail_min_dist_px": gap[~passed].tolist()}


def false_visible_counts(sample: SceneSample, hubs: HubGroups, graph: CorrespondenceGraph,
                         ref_world: list[Tensor], cfg: GraphConfig) -> dict[tuple[int, int], dict[str, list[int]]]:
    """Per view-frame (v, t): ``used`` members (one per hub member slot) behind kept PROJ_P2H edges, and
    ``false`` = those that ``zbuffer_visibility`` with the graph's cell and tol rejects against
    ``ref_world[s]``, the world-frame reference cloud of the sweep s paired with (v, t). Both are counted
    per member-depth bin of DEPTH_BINS_M."""
    p2h, L = graph.edges[EdgeType.PROJ_P2H], graph.layout
    v, t, r, c, s, k = _decode(graph, p2h.src, p2h.dst)
    H, W = sample.images.shape[-2:]
    M, n_patch = hubs.member_idx.shape[2], L.Hp * L.Wp
    bins = torch.tensor(DEPTH_BINS_M)
    out = {}
    for vv in range(L.V):
        for tt in range(L.T):
            ss = sweep_for_frame(sample, vv, tt)
            sel = (v == vv) & (t == tt) & (s == ss)
            uv, depth, vis = visible_members(sample, hubs, ss, vv, tt, cfg)
            rc, inside = pixel_to_patch(uv, cfg.patch, L.Hp, L.Wp)
            key = torch.arange(L.K).repeat_interleave(M) * n_patch + rc[:, 0] * L.Wp + rc[:, 1]
            used = vis & inside & torch.isin(key, k[sel] * n_patch + r[sel] * L.Wp + c[sel])
            ref_uv, ref_d = project(ref_world[ss], sample.intrinsics[vv], sample.cam_to_world[vv, tt])
            ok = zbuffer_visibility(uv, depth, ref_uv, ref_d, H, W, cfg.zbuffer_cell_px, cfg.zbuffer_tol_m)
            b = torch.bucketize(depth.contiguous(), bins.to(depth.dtype), right=True)
            out[(vv, tt)] = {"used": torch.bincount(b[used], minlength=len(bins) + 1).tolist(),
                             "false": torch.bincount(b[used & ~ok], minlength=len(bins) + 1).tolist()}
    return out
