"""Track edges on synthetic scenes, checked against the renderer's exact correspondences."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mavt4d.data.holdout import apply_holdout
from mavt4d.data.synthetic import SyntheticSceneConfig, gt_patch_flow, make_synthetic_scene
from mavt4d.geometry.boxes import move_with_boxes
from mavt4d.geometry.camera import pixel_to_patch, project, unproject
from mavt4d.graph.builder import GraphConfig
from mavt4d.graph.track import build_track_edges
from mavt4d.graph.types import EDGE_ATTR_DIM, Provenance, TokenLayout

GCFG = GraphConfig()
SEEDS = range(4)


def _layout(cfg: SyntheticSceneConfig) -> TokenLayout:
    return TokenLayout(V=cfg.V, T=cfg.T, Hp=cfg.H // cfg.patch, Wp=cfg.W // cfg.patch)


def _decode(L: TokenLayout, idx: torch.Tensor) -> tuple[torch.Tensor, ...]:
    frame, rc = idx // (L.Hp * L.Wp), idx % (L.Hp * L.Wp)
    return frame // L.T, frame % L.T, rc // L.Wp, rc % L.Wp


def _gt_dst(sample, gt, L: TokenLayout, p: int, src: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """GT target token of each source patch centre in frame t2 (-1 if it leaves the grid)."""
    out = torch.full_like(src, -1)
    v, t, r, c = _decode(L, src)
    for vv, tt, tt2 in {(int(a), int(b), int(d)) for a, b, d in zip(v, t, t2)}:
        sel = ((v == vv) & (t == tt) & (t2 == tt2)).nonzero().flatten()
        rr, cc = r[sel], c[sel]
        rows, cols = rr * p + p // 2, cc * p + p // 2
        d = torch.stack([gt.depth[vv, tt][rows - a, cols - b] for a in (0, 1) for b in (0, 1)]).mean(0)
        inst = gt.instance[vv, tt][rows, cols]
        centre = torch.stack([(cc + 0.5) * p, (rr + 0.5) * p], -1).float()
        X = unproject(centre, d, sample.intrinsics[vv], sample.cam_to_world[vv, tt])
        X2 = move_with_boxes(X, torch.where(inst > 0, inst, -1), sample.boxes[tt], sample.boxes[tt2])
        uv, z = project(X2, sample.intrinsics[vv], sample.cam_to_world[vv, tt2])
        rc, inside = pixel_to_patch(uv, p, L.Hp, L.Wp)
        out[sel] = torch.where(inside & (z > 0), L.cam_index(vv, tt2, rc[:, 0], rc[:, 1]), -1)
    return out


def _interior(gt, L: TokenLayout, p: int, idx: torch.Tensor) -> torch.Tensor:
    """Patch is one surface with GT depth range < 5%."""
    v, t, r, c = _decode(L, idx)
    d = gt.depth.reshape(*gt.depth.shape[:2], L.Hp, p, L.Wp, p)[v, t, r, :, c]
    inst = gt.instance.reshape(*gt.depth.shape[:2], L.Hp, p, L.Wp, p)[v, t, r, :, c]
    ok = gt.valid.reshape(*gt.depth.shape[:2], L.Hp, p, L.Wp, p)[v, t, r, :, c]
    lo, hi = d.flatten(1).min(1).values, d.flatten(1).max(1).values
    same = (inst.flatten(1) == inst.flatten(1)[:, :1]).all(1)
    return ok.flatten(1).all(1) & same & (hi - lo < 0.05 * lo)


def _hits(cfg: SyntheticSceneConfig, select) -> tuple[int, int]:
    """(hits, total) over edges kept by ``select(sample, gt, edges, L)`` across SEEDS."""
    hits = total = 0
    L = _layout(cfg)
    for seed in SEEDS:
        sample, gt = make_synthetic_scene(cfg, seed)
        sample = apply_holdout(sample)
        e = build_track_edges(sample, L, GCFG)
        keep = select(sample, gt, e, L)
        _, t2, _, _ = _decode(L, e.dst[keep])
        gt_dst = _gt_dst(sample, gt, L, cfg.patch, e.src[keep], t2)
        hits += int((gt_dst == e.dst[keep]).sum())
        total += int(keep.sum())
    return hits, total


def test_static_scene_edges_hit_gt_patch():
    # 128x224 (half nuScenes): at 64x112 no 16 px patch has a GT depth range under 5%
    cfg = SyntheticSceneConfig(n_moving=0, H=128, W=224)
    hits, total = _hits(cfg, lambda s, gt, e, L: _interior(gt, L, cfg.patch, e.src))
    assert total >= 20, total
    assert hits / total >= 0.9, (hits, total)


def test_moving_box_edges_box_provenance_hit_gt():
    cfg = SyntheticSceneConfig()

    def moving_box(sample, gt, e, L):
        v, t, r, c = _decode(L, e.src)
        rows, cols = r * cfg.patch + cfg.patch // 2, c * cfg.patch + cfg.patch // 2
        on_box = torch.stack([gt.instance[v, t, rows - a, cols - b] == 1 for a in (0, 1) for b in (0, 1)]).all(0)
        return on_box & (e.provenance == Provenance.BOX)

    hits, total = _hits(cfg, moving_box)
    assert total >= 10, total
    assert hits / total >= 0.9, (hits, total)


def test_heldout_points_not_used():
    sample, _ = make_synthetic_scene(SyntheticSceneConfig(), 0)
    sample = replace(sample, heldout=[torch.ones(len(p), dtype=torch.bool) for p in sample.lidar_points])
    e = build_track_edges(sample, _layout(SyntheticSceneConfig()), GCFG)
    assert not ((e.provenance == Provenance.BOX) | (e.provenance == Provenance.CALIB_LIDAR)).any()
    assert len(e.src) == 0  # no flow either


def test_flow_fallback_when_no_lidar():
    cfg = SyntheticSceneConfig()
    L = _layout(cfg)
    sample, gt = make_synthetic_scene(cfg, 0)
    tier_b = replace(sample, intrinsics=None, cam_to_world=None, lidar_points=None, lidar_to_world=None,
                     lidar_times=None, lidar_keys=None, patch_flow=gt_patch_flow(sample, gt, cfg))
    tier_b.validate()
    assert tier_b.tier == "B"
    e = build_track_edges(tier_b, L, GCFG)
    assert len(e.src) >= 50 and (e.provenance == Provenance.FLOW).all() and (e.attr == 0).all()
    assert e.attr.shape[1] == EDGE_ATTR_DIM
    _, ts, _, _ = _decode(L, e.src)
    _, td, _, _ = _decode(L, e.dst)
    fwd = td == ts + 1  # forward edges follow the flow; backward edges are its reverse
    ok_fwd = _gt_dst(sample, gt, L, cfg.patch, e.src[fwd], td[fwd]) == e.dst[fwd]
    ok_bwd = _gt_dst(sample, gt, L, cfg.patch, e.dst[~fwd], ts[~fwd]) == e.src[~fwd]
    assert fwd.any() and (~fwd).any()
    assert (ok_fwd.sum() + ok_bwd.sum()) / len(e.src) >= 0.9


def test_in_degree_at_most_two():
    cfg = SyntheticSceneConfig()
    L = _layout(cfg)
    sample, gt = make_synthetic_scene(cfg, 1)
    sample = replace(apply_holdout(sample), patch_flow=gt_patch_flow(sample, gt, cfg))  # LiDAR and flow together
    e = build_track_edges(sample, L, GCFG)
    assert len(e.src) > 0 and torch.bincount(e.dst, minlength=L.n_cam).max() <= 2
    _, ts, _, _ = _decode(L, e.src)
    _, td, _, _ = _decode(L, e.dst)
    for past in (True, False):
        sel = (ts < td) == past
        assert torch.bincount(e.dst[sel], minlength=L.n_cam).max() <= 1
    assert ((ts - td).abs() == 1).all()


def test_no_edges_when_T_is_1():
    cfg = SyntheticSceneConfig(T=1)
    sample, _ = make_synthetic_scene(cfg, 0)
    assert len(build_track_edges(apply_holdout(sample), _layout(cfg), GCFG).src) == 0
