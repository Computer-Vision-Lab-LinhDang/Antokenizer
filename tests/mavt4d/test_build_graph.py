"""Graph assembly per tier, stats, token geometry and training correspondences."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mavt4d.data.holdout import apply_holdout
from mavt4d.data.synthetic import SyntheticSceneConfig, gt_patch_flow, make_synthetic_scene
from mavt4d.geometry.hubs import build_hubs
from mavt4d.graph.builder import GraphConfig, build_graph, train_correspondences
from mavt4d.graph.geometry_tokens import token_geometry
from mavt4d.graph.spatial import build_same_pixel_track_edges
from mavt4d.graph.types import EdgeType, Provenance

CFG = SyntheticSceneConfig()
GCFG = GraphConfig()
NO_GEOMETRY = dict(intrinsics=None, cam_to_world=None, lidar_points=None, lidar_to_world=None, lidar_times=None,
                   lidar_keys=None, heldout=None, boxes=None)


@pytest.fixture(scope="module")
def scene():
    sample, gt = make_synthetic_scene(CFG, 0)
    sample = apply_holdout(sample)
    return sample, gt, build_hubs(sample, K=64, M=16)


def _counts(graph) -> dict[EdgeType, int]:
    return {et: len(graph.edges[et].src) for et in EdgeType}


def test_tier_C_only_spatial():
    sample, _ = make_synthetic_scene(SyntheticSceneConfig(V=1, T=1), 0)
    sample = replace(sample, **NO_GEOMETRY)
    assert sample.tier == "C"
    graph, _ = build_graph(sample, None, GCFG)
    n = _counts(graph)
    assert n[EdgeType.SPATIAL] > 0 and sum(n.values()) == n[EdgeType.SPATIAL]


def test_tier_B_spatial_and_flow_track(scene):
    sample, gt, _ = scene
    tier_b = replace(sample, **NO_GEOMETRY, patch_flow=gt_patch_flow(sample, gt, CFG))
    assert tier_b.tier == "B"
    graph, _ = build_graph(tier_b, None, GCFG)
    n = _counts(graph)
    assert n[EdgeType.SPATIAL] > 0 and n[EdgeType.TRACK] > 0
    assert (graph.edges[EdgeType.TRACK].provenance == Provenance.FLOW).all()
    assert n[EdgeType.PROJ_P2H] == n[EdgeType.PROJ_H2P] == n[EdgeType.NB3D] == 0


def test_tier_A_all_types_present(scene):
    sample, _, hubs = scene
    graph, _ = build_graph(sample, hubs, GCFG)
    assert all(v > 0 for v in _counts(graph).values())
    L = graph.layout
    assert (L.V, L.T, L.Hp, L.Wp, L.S, L.K) == (CFG.V, CFG.T, 4, 7, CFG.T, 64)
    track_prov = set(graph.edges[EdgeType.TRACK].provenance.tolist())
    assert {int(Provenance.BOX), int(Provenance.CALIB_LIDAR)} <= track_prov
    for e in graph.edges.values():
        assert e.src.max() < L.num_nodes and e.dst.max() < L.num_nodes


def test_no_hubs_means_no_proj_or_nb3d(scene):
    sample, _, _ = scene
    graph, _ = build_graph(sample, None, GCFG)
    n = _counts(graph)
    assert n[EdgeType.SPATIAL] > 0 and n[EdgeType.TRACK] > 0
    assert n[EdgeType.PROJ_P2H] == n[EdgeType.PROJ_H2P] == n[EdgeType.NB3D] == 0
    assert graph.layout.S == graph.layout.K == 0


def test_same_pixel_mode_replaces_track(scene):
    sample, _, hubs = scene
    graph, _ = build_graph(sample, hubs, replace(GCFG, track_mode="same_pixel"))
    track, ref = graph.edges[EdgeType.TRACK], build_same_pixel_track_edges(graph.layout, GCFG.track_cap)
    assert torch.equal(track.src, ref.src) and torch.equal(track.dst, ref.dst)
    assert (track.provenance == Provenance.GRID).all()


def test_stats_match_edges(scene):
    sample, _, hubs = scene
    graph, stats = build_graph(sample, hubs, GCFG)
    L = graph.layout
    eligible = {EdgeType.SPATIAL: L.n_cam, EdgeType.TRACK: L.n_cam, EdgeType.PROJ_P2H: L.n_hub,
                EdgeType.PROJ_H2P: L.n_cam, EdgeType.NB3D: L.n_hub}
    for et, e in graph.edges.items():
        assert stats.num_edges[et.name] == len(e.src)
        assert stats.mean_in_degree[et.name] == pytest.approx(len(e.src) / eligible[et])
        assert sum(stats.provenance[et.name].values()) == len(e.src)
        for name, count in stats.provenance[et.name].items():
            assert count == int((e.provenance == Provenance[name]).sum())


def test_token_geometry_none_fields_when_uncalibrated(scene):
    sample, gt, hubs = scene
    graph, _ = build_graph(sample, hubs, GCFG)
    L = graph.layout
    geom = token_geometry(sample, L, hubs)
    i = L.cam_index(2, 3, 1, 5)
    assert geom.patch_rc[i].tolist() == [1, 5] and geom.cam_id[i] == 2
    assert geom.patch_center_px[i].tolist() == [5.5 * 16, 1.5 * 16]
    assert geom.time[i] == pytest.approx(3 * CFG.dt) and geom.time.dtype == torch.float32
    assert torch.equal(geom.cam_to_world[i], sample.cam_to_world[2, 3]) and torch.equal(geom.intrinsics[i], sample.intrinsics[2])
    assert geom.hub_world.shape == (L.n_hub, 3) and torch.equal(geom.hub_world[L.K + 3], hubs.centers_world[1, 3])
    assert torch.equal(geom.lidar_to_world[L.K + 3], sample.lidar_to_world[1]) and geom.hub_time.shape == (L.n_hub,)

    tier_b = replace(sample, **NO_GEOMETRY, patch_flow=gt_patch_flow(sample, gt, CFG))
    graph_b, _ = build_graph(tier_b, None, GCFG)
    geom_b = token_geometry(tier_b, graph_b.layout, None)
    assert geom_b.patch_rc.shape == (graph_b.layout.n_cam, 2) and geom_b.cam_id.shape == (graph_b.layout.n_cam,)
    for f in ("cam_to_world", "intrinsics", "hub_world", "hub_sensor", "hub_time", "lidar_to_world"):
        assert getattr(geom_b, f) is None, f


def test_train_correspondences_training_points_only_cross_view(scene):
    sample, gt, hubs = scene
    graph, _ = build_graph(sample, hubs, GCFG)
    L, p = graph.layout, CFG.patch
    pairs = train_correspondences(sample, L, GCFG)
    assert pairs.dtype == torch.int64 and pairs.shape[1] == 2 and len(pairs) >= 20
    assert (pairs < L.n_cam).all()
    frame = pairs // (L.Hp * L.Wp)
    assert (frame[:, 0] != frame[:, 1]).all()  # two different view-frames
    # the two patches see a common surface point (renderer GT, independent of the LiDAR)
    hw = L.Hp * L.Wp

    def patch_points(tok):
        f, rc = tok // hw, tok % hw
        v, t, r, c = f // L.T, f % L.T, rc // L.Wp, rc % L.Wp
        xyz = gt.world_xyz[v, t, r * p:(r + 1) * p, c * p:(c + 1) * p].reshape(-1, 3)
        return xyz[gt.valid[v, t, r * p:(r + 1) * p, c * p:(c + 1) * p].flatten()]

    close = [torch.cdist(patch_points(a), patch_points(b)).min() < 0.5 for a, b in pairs.tolist()]
    assert sum(close) / len(close) >= 0.99
    assert torch.equal(train_correspondences(sample, L, GCFG, max_pairs=5, seed=1),
                       train_correspondences(sample, L, GCFG, max_pairs=5, seed=1))
    assert len(train_correspondences(sample, L, GCFG, max_pairs=5)) == 5
    all_held = replace(sample, heldout=[torch.ones(len(q), dtype=torch.bool) for q in sample.lidar_points])
    assert train_correspondences(all_held, L, GCFG).shape == (0, 2)
