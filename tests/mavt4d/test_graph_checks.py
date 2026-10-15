"""E1.1 graph-edge checks: projection re-check and false-visible counts against a reference cloud."""
from __future__ import annotations

import pytest
import torch

from mavt4d.data.holdout import apply_holdout
from mavt4d.data.sample import SceneSample
from mavt4d.data.synthetic import SyntheticSceneConfig, make_synthetic_scene
from mavt4d.eval.graph_checks import false_visible_counts, recheck_projection_edges
from mavt4d.geometry.camera import transform_points
from mavt4d.geometry.hubs import build_hubs
from mavt4d.graph.builder import GraphConfig, build_graph
from mavt4d.graph.types import CorrespondenceGraph, EdgeSet, EdgeType

GCFG = GraphConfig()


@pytest.fixture(scope="module")
def scene():
    sample, _ = make_synthetic_scene(SyntheticSceneConfig(), 0)
    sample = apply_holdout(sample)
    hubs = build_hubs(sample, K=64, M=16)
    graph, _ = build_graph(sample, hubs, GCFG)
    return sample, hubs, graph


def _train_refs(sample: SceneSample) -> list[torch.Tensor]:
    return [transform_points(sample.lidar_to_world[s], sample.input_points(s)[:, :3])
            for s in range(len(sample.lidar_points))]


def test_recheck_passes_on_built_graph(scene):
    r = recheck_projection_edges(*scene, GCFG)
    assert r["edges"] >= 100 and r["passed"] == r["edges"] and r["h2p_is_reverse"]
    assert 0 < r["centre_in_patch"] <= r["edges"]
    assert sum(r["centre_cheb_hist"]) == r["edges"]


def test_recheck_catches_shifted_patch(scene):
    sample, hubs, graph = scene
    L, e = graph.layout, graph.edges[EdgeType.PROJ_P2H]
    src = e.src.clone()
    c = src[:10] % L.Wp
    src[:10] += (c + L.Wp // 2) % L.Wp - c  # same row, half the image away
    p2h = EdgeSet(src=src, dst=e.dst, weight=e.weight, attr=e.attr, provenance=e.provenance)
    h2p = EdgeSet(src=e.dst, dst=src, weight=e.weight, attr=e.attr, provenance=e.provenance)
    bad = CorrespondenceGraph(L, {**graph.edges, EdgeType.PROJ_P2H: p2h, EdgeType.PROJ_H2P: h2p})
    r = recheck_projection_edges(sample, hubs, bad, GCFG)
    assert r["passed"] == r["edges"] - 10 and len(r["fail_min_dist_px"]) == 10
    assert min(r["fail_min_dist_px"]) > 0


def test_recheck_flags_h2p_not_reverse(scene):
    sample, hubs, graph = scene
    h2p = graph.edges[EdgeType.PROJ_H2P]
    swapped = EdgeSet(src=h2p.src.flip(0), dst=h2p.dst, weight=h2p.weight, attr=h2p.attr,
                      provenance=h2p.provenance)
    bad = CorrespondenceGraph(graph.layout, {**graph.edges, EdgeType.PROJ_H2P: swapped})
    assert not recheck_projection_edges(sample, hubs, bad, GCFG)["h2p_is_reverse"]


def test_false_visible_zero_against_graphs_own_reference(scene):
    sample, hubs, graph = scene
    fv = false_visible_counts(sample, hubs, graph, _train_refs(sample), GCFG)
    used = sum(sum(c["used"]) for c in fv.values())
    assert used >= 100 and all(sum(c["false"]) == 0 for c in fv.values())


def test_used_members_match_edge_weights(scene):
    """Members counted as used are exactly the visible members behind the kept edges' weights."""
    sample, hubs, graph = scene
    fv = false_visible_counts(sample, hubs, graph, _train_refs(sample), GCFG)
    M = hubs.member_idx.shape[2]
    want = round(float(graph.edges[EdgeType.PROJ_P2H].weight.sum()) * M)
    assert sum(sum(c["used"]) for c in fv.values()) == want


def test_wall_missing_from_sweep_counts_as_false_visible():
    """The sweep only has the back patch at x = 10 m, so the graph links it; the reference also has the
    wall at x = 5 m in front of it, so every used member is false-visible."""
    gy, gz = torch.meshgrid(torch.arange(-4, 4.01, 0.1), torch.arange(-2.5, 2.51, 0.1), indexing="ij")
    wall = torch.stack([torch.full_like(gy, 5.0), gy, gz], -1).reshape(-1, 3)
    by, bz = torch.meshgrid(torch.arange(-0.5, 0.51, 0.05), torch.arange(-0.5, 0.51, 0.05), indexing="ij")
    back = torch.stack([torch.full_like(by, 10.0), by, bz], -1).reshape(-1, 3)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # looks along +x
    sample = SceneSample(scene_id="occl", source="synthetic", images=torch.zeros(1, 1, 3, 64, 112),
                         frame_times=torch.zeros(1, 1, dtype=torch.float64),
                         intrinsics=torch.tensor([[[80.0, 0, 56], [0, 80.0, 32], [0, 0, 1]]]),
                         cam_to_world=c2w[None, None], lidar_points=[torch.cat([back, torch.ones(len(back), 1)], 1)],
                         lidar_to_world=torch.eye(4)[None], lidar_times=torch.zeros(1, dtype=torch.float64),
                         lidar_keys=["occl"])
    hubs = build_hubs(sample, K=16, M=8)
    graph, _ = build_graph(sample, hubs, GCFG)
    fv = false_visible_counts(sample, hubs, graph, [torch.cat([wall, back])], GCFG)[(0, 0)]
    assert sum(fv["used"]) > 0 and fv["false"] == fv["used"]
    assert fv["used"][1] == sum(fv["used"])  # every member sits at depth 10 m: bin [10, 20)


def test_depth_bins_cover_all_members(scene):
    sample, hubs, graph = scene
    for c in false_visible_counts(sample, hubs, graph, _train_refs(sample), GCFG).values():
        assert len(c["used"]) == len(c["false"]) == 4


def test_check_script_graph_pass_on_fake_nuscenes(tmp_path, monkeypatch):
    """Smoke of check_nuscenes.py's graph pass, timing pass and report section on the fake dataset."""
    import importlib.util
    from pathlib import Path

    from fixtures import fake_nuscenes as fk
    from mavt4d.data import nuscenes as nu

    fk.write_fake_nuscenes(tmp_path)
    monkeypatch.setattr(nu, "load_official_split", lambda version, split: list(fk.SCENE_NAMES))
    path = Path(__file__).resolve().parents[2] / "scripts" / "mavt4d" / "check_nuscenes.py"
    spec = importlib.util.spec_from_file_location("_check_nuscenes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ds = nu.NuScenesSceneDataset(tmp_path, version=fk.VERSION, split="mini_train", mode="m3", cameras=fk.CAMERAS)
    g = mod.check_graph(ds, 0.3, 5, workers=0)
    assert g["samples"] == len(ds) == 4 and g["neighbour_sweeps"]["min"] == 5
    assert g["recheck"]["edges"] > 0 and g["recheck"]["passed"] == g["recheck"]["edges"]
    assert sum(g["false_visible"]["train"]["used"]) > 0 and sum(g["false_visible"]["train"]["false"]) == 0
    assert g["degree"]["NB3D"]["max_in_degree"] == mod.GCFG.nb3d_k
    t = mod.graph_timing(ds, 0.3, 2, workers=0)
    assert t["n"] == 2 and t["s_per_sample"] > 0
    rep = {"graph": {sp: {**g, "wall_s": 1.0} for sp in mod.SPLITS}, "graph_config": mod.asdict(mod.GCFG),
           "heldout_frac": 0.3, "fv_sweeps": 5, "cameras": list(fk.CAMERAS), "timing": {"graph_m3": t},
           "loadavg_graph_timing": (1.0, 1.0, 1.0)}
    text = "\n".join(mod.render_graph(rep))
    assert "| projection edges re-project" in text and "**PASS**" in text and "pending" in text
