"""Bounded, real end-to-end wiring check; its scores are not pilot evidence.

Six single-step optimizations exercise all cells, checkpoint I/O, genuine feature
extraction, frozen ridge probes, reconstruction masks and final reporting. The
production configuration and loss are never modified by this test.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path

import torch

from mavt4d.pilot.config import PilotConfig
from mavt4d.pilot.runner import run_suite


ROOT = Path(__file__).resolve().parents[2]


def test_real_tiny_suite_runs_all_cells_through_frozen_evaluation(tmp_path):
    cfg = replace(
        PilotConfig(), source_count=5, seed_start=0, dev_fraction=0.,
        T=12, H=16, W=24, c_tokens=2, d_tokens=2, token_dim=4,
        embed_dim=8, patch_size=4, modulation_dim=2, semantic_dim=4,
        temporal_pool=(2, 3), query_count=4, overfit_steps=1, train_steps=1,
        seeds=(17,), overfit_seed=17, log_every=1, checkpoint_every=1,
        bootstrap=8, eval_reconstruction_groups=1,
    )
    assert cfg.config_sha256 != PilotConfig().config_sha256
    spec = importlib.util.spec_from_file_location(
        "_pilot_smoke_export", ROOT / "scripts/mavt4d/build_event_suite.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    dataset_root = tmp_path / "tiny-fixture"
    fixture = exporter.build_event_suite(dataset_root, cfg.event_config, range(5))
    assert fixture["counts"]["sources"] == {"train": 4, "val": 1}
    assert fixture["counts"]["samples"] == {"train": 32, "val": 8}

    output = tmp_path / "wiring-smoke"
    result = run_suite(cfg, dataset_root, output, device="cpu",
                       git_revision="1" * 40, allow_cpu=True)
    assert result["status"] == "complete", result
    assert result["evaluation_failures"] == []
    assert len(result["training"]) == 6
    assert {(run["phase"], run["cell"], run["seed"]) for run in result["training"]} == {
        (phase, cell, 17) for phase in ("overfit", "main") for cell in cfg.cells}
    for run in result["training"]:
        assert run["status"] == "complete" and run["step"] == 1
        assert run["budget"]["optimizer_steps"] == 1
        assert run["budget"]["total_state_bytes"] == 72
        assert all(math.isfinite(value) for value in run["last_training_metrics"].values())
        initial = torch.load(run["initial_checkpoint"], map_location="cpu", weights_only=True)
        final = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
        assert initial["metadata"]["step"] == 0
        assert final["metadata"]["step"] == 1
        assert final["metadata"]["config_sha256"] == cfg.config_sha256
        assert final["optimizer"]["state"]
        assert all(torch.isfinite(value).all() for value in final["model"].values())
        assert all(torch.isfinite(value).all() for value in final["heads"].values())
        assert any(not torch.equal(value, final["model"][name])
                   for name, value in initial["model"].items())

    evaluations = sorted((output / "evaluation").glob("*.json"))
    assert len(evaluations) == 6
    assert {path.name for path in evaluations} == {
        f"{cell}-seed17-{kind}.json" for cell in cfg.cells for kind in ("trained", "untrained")}
    for path in evaluations:
        evaluated = json.loads(path.read_text())["result"]
        assert evaluated["train_sources"] == 4 and evaluated["val_sources"] == 1
        # FeatureTable validates every actual extracted vector before the ridge
        # fit; finite reported scores additionally guard the full reporting path.
        assert all(dimension > 0 for dimension in evaluated["feature_dimensions"].values())
        assert all(math.isfinite(value) and 0. <= value <= 1.
                   for value in evaluated["metrics"].values())
        for control in ("last_c", "last_d", "orderless"):
            for suffix in ("direction", "direction_lateral", "direction_loop"):
                assert evaluated["metrics"][f"{control}_{suffix}"] == .5
        reconstruction = evaluated["reconstruction"]
        assert reconstruction["data_range"] == 2 and reconstruction["clipped"] is False
        assert reconstruction["source_groups"] == ["synthetic-event:source-4"]
        assert set(reconstruction["families"]) == set(cfg.mask_kinds)
        for family, measured in reconstruction["families"].items():
            assert measured["endpoint_frame"] == (7 if family == "snapshot" else 11)
            assert measured["all"]["rgb_count"] == 8 * cfg.H * cfg.W * 3
            assert math.isfinite(measured["all"]["l1"])
            assert math.isfinite(measured["all"]["mse"])
            for region in ("observed", "missing", "all"):
                for metric in ("l1", "mse", "psnr"):
                    value = measured[region][metric]
                    assert value is None or math.isfinite(value)

    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    barrier = next(i for i, event in enumerate(events) if event["action"] == "all_training_finished")
    assert events[barrier]["attempted"] == events[barrier]["expected"] == 6
    assert sum(event["action"] == "training_finished" for event in events[:barrier]) == 6
    assert all(i > barrier for i, event in enumerate(events) if event["action"] == "evaluation_started")
    assert sum(event["action"] == "evaluation_started" for event in events) == 6
    lock = json.loads((output / "run_lock.json").read_text())
    assert lock["environment"]["test_only_cpu"] is True
    assert lock["groups"]["val"] == ["synthetic-event:source-4"]
    assert lock["groups"]["dev"] == []
    assert json.loads((output / "config.json").read_text()) == cfg.to_dict()
    assert json.loads((output / "summary.json").read_text())["status"] == "complete"
    assert (output / "report.md").is_file()
    json.dumps(result, allow_nan=False)
