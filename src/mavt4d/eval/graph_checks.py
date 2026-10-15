"""Checks of the projection edges on real data (E1.1 / plan T27).

``recheck_projection_edges``: every (hub, patch) pair of PROJ_P2H must have at least one hub member whose
float64 re-projection, computed apart from the builder's float32 path, lands in that patch in front of the
camera (spec T11, test 1); PROJ_H2P must be the exact reverse. Where the hub centre lands is reported for
information only: edges come from members, so a hub spanning several patches links to all of them.

``false_visible_counts``: members the graph uses (visible to the builder's z-buffer and inside the patch of
a kept edge) that the same z-buffer rule (cell, tol) rejects against a denser reference cloud.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mavt4d.data.sample import SceneSample
from mavt4d.geometry.camera import cell_index, pixel_to_patch, project, transform_points, zbuffer_visibility
from mavt4d.geometry.hubs import HubGroups, pca_normals
from mavt4d.graph.projection import visible_members
from mavt4d.graph.track import sweep_for_frame
from mavt4d.graph.types import CorrespondenceGraph, EdgeType

if TYPE_CHECKING:
    from mavt4d.graph.builder import GraphConfig

DEPTH_BINS_M = (10.0, 20.0, 40.0)  # member depth bins [0, 10), [10, 20), [20, 40), [40, inf)
CHEB_MAX = 64  # last bin of the centre-distance histogram collects everything at or beyond it


@dataclass(frozen=True)
class FVRule:
    """Occlusion rule of the false-visible check: a point is hidden when a reference point in its ``cell_px``
    cell is nearer than depth - (tol_m + tol_rel * depth). With ``plane_tau`` set, reference points within
    plane_tau metres of the point's surface plane (its hub's PCA plane) are the same surface and do not count."""
    cell_px: int = 4
    tol_m: float = 0.5
    tol_rel: float = 0.0
    plane_tau: float | None = None


def rule_visible(uv: Tensor, z: Tensor, ref_uv: Tensor, ref_z: Tensor, H: int, W: int, rule: FVRule,
                 p: Tensor | None = None, n: Tensor | None = None, ref_p: Tensor | None = None) -> Tensor:
    """(N,) bool visible under ``rule``. ``p`` (N, 3), ``n`` (N, 3) unit normals and ``ref_p`` (R, 3) are world
    points, needed only when ``rule.plane_tau`` is set (every (point, reference) pair sharing a cell is tested)."""
    if rule.plane_tau is None:
        return zbuffer_visibility(uv, z, ref_uv, ref_z, H, W, rule.cell_px, rule.tol_m, rule.tol_rel)
    ref_ok, ref_idx, n_cells = cell_index(ref_uv, ref_z, H, W, rule.cell_px)
    ok, idx, _ = cell_index(uv, z, H, W, rule.cell_px)
    refs = ref_ok.nonzero().flatten()
    refs = refs[torch.argsort(ref_idx[refs], stable=True)]
    count = torch.bincount(ref_idx[refs], minlength=n_cells)
    start = torch.cumsum(count, 0) - count
    pts = ok.nonzero().flatten()
    k = count[idx[pts]]
    pair_pt = pts.repeat_interleave(k)
    offset = torch.arange(len(pair_pt), device=z.device) - (torch.cumsum(k, 0) - k).repeat_interleave(k)
    pair_ref = refs[start[idx[pair_pt]] + offset]
    nearer = ref_z[pair_ref] < z[pair_pt] - (rule.tol_m + rule.tol_rel * z[pair_pt])
    off_plane = ((ref_p[pair_ref] - p[pair_pt]) * n[pair_pt]).sum(-1).abs() > rule.plane_tau
    hidden = torch.zeros_like(ok)
    hidden[pair_pt[nearer & off_plane]] = True
    return ok & ~hidden


def hub_normals_world(sample: SceneSample, hubs: HubGroups) -> Tensor:
    """(S, K, 3) world-frame unit PCA normals of the hubs, as the projection-edge builder computes them."""
    S, K, M = hubs.member_idx.shape
    return pca_normals(hubs.rel_xyz.reshape(S * K, M, 3)).reshape(S, K, 3) @ sample.lidar_to_world[:, :3, :3].transpose(1, 2)


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


def used_members(sample: SceneSample, hubs: HubGroups, graph: CorrespondenceGraph, cfg: GraphConfig, v: int,
                 t: int) -> tuple[int, Tensor, Tensor, Tensor]:
    """Sweep s paired with view-frame (v, t), and for its hub members flattened (K*M,): uv, depth and ``used`` =
    visible to the builder's z-buffer and inside the patch of a kept PROJ_P2H edge in (v, t), i.e. exactly the
    members behind the edge weights."""
    p2h, L = graph.edges[EdgeType.PROJ_P2H], graph.layout
    ev, et, er, ec, es, ek = _decode(graph, p2h.src, p2h.dst)
    s = sweep_for_frame(sample, v, t)
    sel = (ev == v) & (et == t) & (es == s)
    M, n_patch = hubs.member_idx.shape[2], L.Hp * L.Wp
    uv, depth, vis = visible_members(sample, hubs, s, v, t, cfg)
    rc, inside = pixel_to_patch(uv, cfg.patch, L.Hp, L.Wp)
    key = torch.arange(L.K).repeat_interleave(M) * n_patch + rc[:, 0] * L.Wp + rc[:, 1]
    return s, uv, depth, vis & inside & torch.isin(key, ek[sel] * n_patch + er[sel] * L.Wp + ec[sel])


def false_visible_counts(sample: SceneSample, hubs: HubGroups, graph: CorrespondenceGraph, ref_world: list[Tensor],
                         cfg: GraphConfig, rule: FVRule | None = None) -> dict[tuple[int, int], dict[str, list[int]]]:
    """Per view-frame (v, t): ``used`` members (``used_members``) and ``false`` = those that ``rule`` rejects
    against ``ref_world[s]``, the world-frame reference cloud of the sweep s paired with (v, t). ``rule``
    defaults to the builder's own z-buffer; pass it explicitly to hold the yardstick fixed across builder
    variants. Counts are per member-depth bin of DEPTH_BINS_M."""
    rule = rule or FVRule(cfg.zbuffer_cell_px, cfg.zbuffer_tol_m, cfg.zbuffer_tol_rel)
    normals = hub_normals_world(sample, hubs) if rule.plane_tau is not None else None
    L, (H, W) = graph.layout, sample.images.shape[-2:]
    M = hubs.member_idx.shape[2]
    bins = torch.tensor(DEPTH_BINS_M)
    out = {}
    for vv in range(L.V):
        for tt in range(L.T):
            ss, uv, depth, used = used_members(sample, hubs, graph, cfg, vv, tt)
            ref_uv, ref_d = project(ref_world[ss], sample.intrinsics[vv], sample.cam_to_world[vv, tt])
            plane = ()
            if normals is not None:
                pts = transform_points(sample.lidar_to_world[ss], sample.input_points(ss)[:, :3])
                plane = (pts[hubs.member_idx[ss].flatten()], normals[ss].repeat_interleave(M, 0), ref_world[ss])
            ok = rule_visible(uv, depth, ref_uv, ref_d, H, W, rule, *plane)
            b = torch.bucketize(depth.contiguous(), bins.to(depth.dtype), right=True)
            out[(vv, tt)] = {"used": torch.bincount(b[used], minlength=len(bins) + 1).tolist(),
                             "false": torch.bincount(b[used & ~ok], minlength=len(bins) + 1).tolist()}
    return out
