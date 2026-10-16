"""Pilot orchestration must preserve initialization, schedules and the val barrier."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch

from mavt4d.pilot.config import PilotConfig
from mavt4d.pilot import runner


def tiny_config(**changes):
    defaults = dict(H=16, W=24, c_tokens=2, d_tokens=2, token_dim=4,
                    embed_dim=8, modulation_dim=2, temporal_pool=(2, 3),
                    semantic_dim=4, query_count=12, overfit_steps=2, train_steps=2,
                    log_every=1, checkpoint_every=1)
    defaults.update(changes)
    return dataclasses.replace(PilotConfig(), **defaults)


def test_shared_initialization_is_identical_across_cells_and_reproducible():
    cfg = tiny_config()
    runs = [runner.initialize_run(cfg, cell, 17, "cpu") for cell in cfg.cells]
    assert runs[0][2] == runs[1][2] == runs[2][2]
    assert set(runs[0][2]) == {"observer", "decoder", "heads"}
    for index in (1, 2):
        for module in ("observer", "decoder"):
            assert all(torch.equal(a, b) for a, b in zip(
                getattr(runs[0][0], module).state_dict().values(),
                getattr(runs[index][0], module).state_dict().values()))
    again = runner.initialize_run(cfg, "coupled", 17, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(
        runs[0][0].state_dict().values(), again[0].state_dict().values()))
    assert runner.initialize_run(cfg, "gru", 29, "cpu")[2] != runs[0][2]
    counts = [sum(p.numel() for p in model.parameters()) for model, _, _ in runs]
    assert counts[2] > counts[0] > counts[1]


def test_group_schedule_and_objective_rng_are_independent_of_model_rng():
    groups = ("source-a", "source-b", "source-c")
    expected = runner.group_schedule(groups, seed=17, steps=30)
    torch.manual_seed(900)
    torch.rand(123)
    assert runner.group_schedule(groups, seed=17, steps=30) == expected
    assert runner.group_schedule(groups, seed=29, steps=30) != expected
    assert runner.group_schedule(groups, seed=17, steps=4, overfit_group="source-b") == ("source-b",) * 4
    first = runner.objective_generator(17)
    torch.rand(31)
    second = runner.objective_generator(17)
    assert torch.equal(torch.rand(64, generator=first), torch.rand(64, generator=second))


def test_checkpoint_restores_model_optimizer_and_explicit_random_stream(tmp_path):
    cfg = tiny_config()
    model, heads, hashes = runner.initialize_run(cfg, "coupled", 17, "cpu")
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=.001)
    generator = runner.objective_generator(17)
    loss = sum(p.square().mean() for p in model.parameters()) + sum(p.square().mean() for p in heads.parameters())
    loss.backward()
    optimizer.step()
    torch.rand(5, generator=generator)
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = {"config_sha256": cfg.config_sha256, "dataset_sha256": "a" * 64,
                "git_revision": "1" * 40, "phase": "main", "cell": "coupled",
                "seed": 17, "step": 1, "schedule_sha256": "b" * 64,
                "shared_init_hashes": hashes}
    runner.save_checkpoint(checkpoint, model, heads, optimizer, generator, metadata)
    expected_rng = torch.rand(16, generator=generator)
    clone, clone_heads, _ = runner.initialize_run(cfg, "coupled", 99, "cpu")
    clone_opt = torch.optim.AdamW(list(clone.parameters()) + list(clone_heads.parameters()), lr=.001)
    clone_rng = runner.objective_generator(99)
    loaded = runner.load_checkpoint(checkpoint, clone, clone_heads, clone_opt, clone_rng,
                                    config_sha256=cfg.config_sha256, dataset_sha256="a" * 64)
    assert loaded == metadata
    assert torch.equal(torch.rand(16, generator=clone_rng), expected_rng)
    for left, right in zip(model.state_dict().values(), clone.state_dict().values()):
        assert torch.equal(left, right)
    for left, right in zip(heads.state_dict().values(), clone_heads.state_dict().values()):
        assert torch.equal(left, right)
    assert clone_opt.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    assert len(clone_opt.state) == len(optimizer.state)
    with pytest.raises(ValueError, match="config"):
        runner.load_checkpoint(checkpoint, clone, clone_heads, clone_opt, clone_rng,
                               config_sha256="c" * 64, dataset_sha256="a" * 64)
    with pytest.raises(ValueError, match="dataset"):
        runner.load_checkpoint(checkpoint, clone, clone_heads, clone_opt, clone_rng,
                               config_sha256=cfg.config_sha256, dataset_sha256="c" * 64)


def test_overfit_gate_is_fixed_and_does_not_change_main_budget():
    cfg = tiny_config()
    good = runner.overfit_gate({"reconstruction_l1": 1.0},
        {"reconstruction_l1": .70, "temporal_mse": .7, "temporal_zero_mse": 1.0}, cfg)
    bad = runner.overfit_gate({"reconstruction_l1": 1.0},
        {"reconstruction_l1": .9, "temporal_mse": .9, "temporal_zero_mse": 1.0}, cfg)
    assert good["passed"] and not bad["passed"]
    assert good["reconstruction_ratio"] == .70
    assert bad["temporal_ratio"] == .9
    assert cfg.train_steps == 2


def test_tiny_training_loop_updates_parameters_and_saves_final_optimizer(tmp_path, monkeypatch):
    cfg = tiny_config()
    class FakeDataset:
        train_groups = ("train-a", "train-b")
        overfit_group = "train-a"
        def load_group(self, group, device):
            assert group in self.train_groups
            return object()
    def objective(model, heads, batch, config, generator, phase):
        # Exercise optimizer, clipping and checkpoint orchestration independently
        # of the genuine objective, which has its own all-cell gradient tests.
        loss = sum(p.square().mean() for p in model.parameters())
        loss = loss + sum(p.square().mean() for p in heads.parameters())
        loss = loss + torch.rand((), generator=generator) * .001
        return loss, {"reconstruction_l1": loss, "temporal_mse": loss,
                      "temporal_zero_mse": loss.new_tensor(1.)}
    monkeypatch.setattr(runner, "training_loss", objective)
    result = runner._train_run(cfg, FakeDataset(), "overfit", "coupled", 17,
                               tmp_path / "tiny-run", "cpu",
                               {"dataset_sha256": "a" * 64, "git_revision": "1" * 40})
    assert result["status"] == "complete" and result["step"] == 2
    assert result["budget"]["total_state_bytes"] == 72
    initial = torch.load(result["initial_checkpoint"], weights_only=True)
    final = torch.load(result["checkpoint"], weights_only=True)
    assert final["metadata"]["step"] == 2
    assert len(final["optimizer"]["state"]) > 0
    assert any(not torch.equal(value, final["model"][name]) for name, value in initial["model"].items())
    assert not torch.equal(initial["objective_rng"], final["objective_rng"])
    assert (tmp_path / "tiny-run" / "training.jsonl").read_text().count("\n") == 2


def test_suite_trains_all_cells_and_seeds_before_any_evaluation(tmp_path, monkeypatch):
    cfg = tiny_config()
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "summary.json").write_text('{"complete": true}')
    events = []
    class FakeDataset:
        def __init__(self, path, config):
            self.root = Path(path)
            self.train_groups = ("train",)
            self.dev_groups = ("dev",)
            self.val_groups = ("val",)
    monkeypatch.setattr(runner, "PilotDataset", FakeDataset)
    def train(config, dataset, phase, cell, seed, run_dir, device, lock):
        events.append(("train", phase, cell, seed))
        return {"phase": phase, "cell": cell, "seed": seed, "status": "complete",
                "step": config.overfit_steps if phase == "overfit" else config.train_steps,
                "checkpoint": "fake.pt", "initial_checkpoint": "fake-initial.pt",
                "overfit_gate": {"passed": False} if phase == "overfit" else None}
    def evaluate(config, dataset, run, device, *, untrained=False):
        assert len([e for e in events if e[:2] == ("train", "main")]) == len(cfg.cells) * len(cfg.seeds)
        events.append(("eval", run["cell"], run["seed"], untrained))
        return {"metrics": {"c_direction": .5}, "group_scores": {"c_direction": {"val": .5}}, "predictions": {}}
    monkeypatch.setattr(runner, "_train_run", train)
    monkeypatch.setattr(runner, "_evaluate_run", evaluate)
    monkeypatch.setattr(runner, "summarize_runs", lambda runs, cfg: {"number_of_runs": len(runs)})
    output = tmp_path / "pilot"
    result = runner.run_suite(cfg, root, output, device="cpu", git_revision="1" * 40, allow_cpu=True)
    training = [e for e in events if e[0] == "train"]
    assert len(training) == 3 + 9
    assert [e[1] for e in training] == ["overfit"] * 3 + ["main"] * 9
    assert len([e for e in events if e[0] == "eval"]) == 18
    assert result["status"] == "complete"
    frozen = json.loads((output / "run_lock.json").read_text())
    assert frozen["config_sha256"] == cfg.config_sha256
    assert frozen["dataset_sha256"] == hashlib.sha256((root / "summary.json").read_bytes()).hexdigest()
    assert (output / "report.md").exists()
    with pytest.raises(FileExistsError):
        runner.run_suite(cfg, root, output, device="cpu", git_revision="1" * 40, allow_cpu=True)


def test_production_cpu_is_rejected_before_creating_output(tmp_path):
    output = tmp_path / "pilot"
    with pytest.raises(ValueError, match="CUDA"):
        runner.run_suite(PilotConfig(), tmp_path / "dataset", output,
                         device="cpu", git_revision="1" * 40)
    assert not output.exists()


def test_partial_results_keep_unpaired_runs_without_fabricating_pairs(monkeypatch):
    cfg = tiny_config()
    entries = [{"cell": "coupled", "seed": 17, "result": {}},
               {"cell": "coupled", "seed": 29, "result": {}},
               {"cell": "two_scan", "seed": 17, "result": {}}]
    def summarize(runs, config):
        seeds_by_cell = {}
        for run in runs:
            seeds_by_cell.setdefault(run["cell"], set()).add(run["seed"])
        assert len({tuple(sorted(seeds)) for seeds in seeds_by_cell.values()}) == 1
        return {"pairs": [(run["cell"], run["seed"]) for run in runs]}
    monkeypatch.setattr(runner, "summarize_runs", summarize)
    result = runner._summarize_available(entries, cfg)
    assert result["status"] == "incomplete_comparison"
    assert len(result["per_cell"]["coupled"]["pairs"]) == 2
    assert result["paired_complete_seeds"] == []  # GRU has no usable result.
    assert result["paired"] is None
