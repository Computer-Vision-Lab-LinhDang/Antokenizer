"""Calibration of the false-visible rule on synthetic scenes with exact visibility."""
from __future__ import annotations

import math

import pytest
import torch

from mavt4d.data.holdout import apply_holdout
from mavt4d.data.sample import Boxes3D
from mavt4d.data.synthetic import SyntheticSceneConfig, make_synthetic_scene
from mavt4d.eval.fv_calibration import boundary_cells, calibrate_scene, gt_normals, reference_clouds
from mavt4d.eval.graph_checks import FVRule
from mavt4d.geometry.boxes import points_in_boxes
from mavt4d.geometry.hubs import build_hubs
from mavt4d.graph.builder import GraphConfig, build_graph

CFG = SyntheticSceneConfig(T=1, n_boxes=6, n_moving=1)
SEED = 1  # 12 GT-occluded hub members in its 3 views


@pytest.fixture(scope="module")
def scene():
    sample, gt = make_synthetic_scene(CFG, SEED)
    sample = apply_holdout(sample)
    hubs = build_hubs(sample, K=64, M=16)
    graph, _ = build_graph(sample, hubs, GraphConfig())
    return sample, gt, hubs, graph


def test_gt_normals_ground_and_box_faces():
    yaw = math.pi / 6
    boxes = Boxes3D(center=torch.tensor([[10.0, 0.0, 1.0]]), size=torch.tensor([[2.0, 4.0, 2.0]]),
                    yaw=torch.tensor([yaw]), instance_id=torch.tensor([1]))
    heading = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0])
    side = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0])
    p = torch.stack([torch.tensor([3.0, 4.0, 0.0]),  # ground
                     boxes.center[0] + 2.0 * heading,  # front face (half length 2)
                     boxes.center[0] + 1.0 * side + 0.3 * heading,  # side face (half width 1)
                     boxes.center[0] + torch.tensor([0.2, 0.1, 1.0])])  # roof
    n = gt_normals(p, boxes)
    for got, want in zip(n, (torch.tensor([0.0, 0, 1]), heading, side, torch.tensor([0.0, 0, 1]))):
        assert torch.allclose(got.abs(), want.abs(), atol=1e-5)


def test_boundary_cells_mark_cells_with_two_labels_or_sky():
    inst = torch.zeros(8, 8, dtype=torch.int64)
    valid = torch.ones(8, 8, dtype=torch.bool)
    inst[0:4, 2] = 3  # a box column inside the top-left cell
    valid[6, 7] = False  # sky in the bottom-right cell
    assert boundary_cells(inst, valid, 4).tolist() == [[True, False], [False, True]]


def test_reference_clouds_add_neighbour_sweeps_without_box_points(scene):
    sample, _, _, _ = scene
    refs, info = reference_clouds(CFG, SEED, sample, sides=(0, 1, 5), drop="all")
    n_key = sample.lidar_points[0].shape[0]  # held-out points included
    assert refs[0].shape == (n_key, 3) and len(refs[1]) > n_key and len(refs[5]) > len(refs[1])
    assert info[5]["sweeps"] == 10 and info[1]["sweeps"] == 2 and info[5]["dropped_box"] > 0
    extra = refs[5][n_key:]
    assert (points_in_boxes(extra, sample.boxes[0], margin=0.0) < 0).float().mean() > 0.95  # moving boxes moved
    moving, info_m = reference_clouds(CFG, SEED, sample, sides=(5,), drop="moving")
    assert 0 < info_m[5]["dropped_box"] < info[5]["dropped_box"]  # static boxes stay as structure
    assert len(moving[5]) == len(refs[5]) + info[5]["dropped_box"] - info_m[5]["dropped_box"]


def test_calibrate_scene_counts_are_consistent(scene):
    sample, gt, hubs, graph = scene
    refs, _ = reference_clouds(CFG, SEED, sample, sides=(0,))
    rules = {"never": FVRule(tol_m=1e6), "always": FVRule(tol_m=-1e6), "m1": FVRule()}
    out = calibrate_scene(sample, gt, hubs, graph, GraphConfig(), refs, rules)
    n, f = out["n"], out["flag"]
    assert sum(n["correct"]["depth"]) > 100 and sum(n["wrong"]["depth"]) > 0
    assert sum(n["used"]["depth"]) <= sum(n["correct"]["depth"]) + sum(n["wrong"]["depth"])
    assert sum(n["used_true_fv"]["depth"]) <= sum(n["used"]["depth"])
    for split in ("correct", "wrong", "used"):
        assert sum(f["never"][0][split]["depth"]) == 0
        assert f["always"][0][split]["depth"] == n[split]["depth"]
    assert sum(n["correct"]["inc"]) == sum(n["correct"]["boundary"]) == sum(n["correct"]["depth"])
    assert sum(n["wrong"]["gap"]) == sum(n["wrong"]["depth"])


def test_choose_rule_youden_with_ties_to_the_simpler_rule():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "mavt4d" / "calibrate_false_visible.py"
    spec = importlib.util.spec_from_file_location("_calibrate_fv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def agg(err: dict, rec: dict) -> dict:
        n = {"correct": {"depth": [1000]}, "wrong": {"depth": [100]}}
        flag = {k: {"pm5": {"correct": {"depth": [round(1000 * err.get(k, 0.0))]},
                            "wrong": {"depth": [round(100 * rec.get(k, 0.0))]}}} for k in mod.RULES}
        return {"n": n, "flag": flag}

    assert mod.choose_rule(agg({"c4": 0.04}, {"c4": 0.5}))[0] is None  # below 5%: not miscalibrated
    # R0 over-flags (recall 0.9, error 0.6): J 0.3; c4+plane J 0.5; c2+rel+plane J 0.502 ties -> fewer changes
    err = {k: 0.6 for k in mod.RULES} | {"c4": 0.6, "c4+plane": 0.01, "c2+rel+plane": 0.008}
    rec = {k: 0.0 for k in mod.RULES} | {"c4": 0.9, "c4+plane": 0.51, "c2+rel+plane": 0.51}
    chosen, why = mod.choose_rule(agg(err, rec))
    assert chosen == "c4+plane" and why["youden_pm5"]["c4"] == pytest.approx(0.3)
