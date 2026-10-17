"""Labelled upper bound U: a training-only linear core head on final C."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid_loss_v3.config import LossV3TrialConfig
from mavt4d.hybrid_loss_v3.evaluation import probe_metrics
from mavt4d.hybrid_loss_v3.objectives import (
    LossV3Module,
    core_head_cross_entropy,
    split_core_accuracy,
    variant_contrast,
)
from mavt4d.hybrid_pilot.config import HybridPilotConfig
from mavt4d.hybrid_pilot.objectives import full_observation_states
from mavt4d.pilot.objectives import make_training_mask

from test_hybrid_loss_v3_objectives import LABELS, batch, config

ROOT = Path(__file__).resolve().parents[2]


def test_existing_trial_configs_keep_hashes_and_arm_b():
    expected = {
        "hybrid_loss_v3_trial": "3de93fc72fc6adec2f48e08bf697a88bef72416923398adb43ca2b048c586339",
        "hybrid_loss_v3_vram_trial": "b40a4a9e4333232df64127c4eeda349da5784fc22865ea3d373d5f937ebf4eb0",
        "hybrid_loss_v3_vram_trial_v2": "2b955fb68df3a71f495a4b90b7a3c822d9d2fd122181e5eb7fb7fe1f9e1a1ce4",
    }
    for name, digest in expected.items():
        cfg = LossV3TrialConfig.load(ROOT / f"configs/mavt4d/{name}.json")
        assert cfg.sha256 == digest
        assert (cfg.arm, cfg.core_head_weight, cfg.overfit) == ("B", 0.0, False)


def test_upper_bound_profiles_are_versioned_and_production_valid():
    tiny = LossV3TrialConfig.load(ROOT / "configs/mavt4d/hybrid_loss_v3_upper_bound_tiny.json")
    main = LossV3TrialConfig.load(ROOT / "configs/mavt4d/hybrid_loss_v3_upper_bound.json")
    b0 = LossV3TrialConfig.load(ROOT / "configs/mavt4d/hybrid_loss_v3_trial.json")
    for cfg in (tiny, main):
        assert (cfg.arm, cfg.core_head_weight) == ("U", 1.0)
        assert cfg.capacity_policy == "fastest_safe"
        cfg.validate_production()
    assert tiny.overfit and (tiny.steps, tiny.effective_source_groups) == (200, 1)
    assert not main.overfit
    # U-main differs from the matched B0 only by version and logging cadence.
    differs = ("version", "log_every")
    assert {k: v for k, v in main.to_dict().items() if k not in differs} == {
        k: v for k, v in b0.to_dict().items() if k not in differs
    }
    assert main.log_every == 1
    with pytest.raises(ValueError):
        replace(tiny, effective_source_groups=8)


def test_default_module_has_no_head_and_head_does_not_change_model_init():
    torch.manual_seed(17)
    plain = LossV3Module(config())
    torch.manual_seed(17)
    upper = LossV3Module(config(), core_head_weight=1.0)
    assert not hasattr(plain, "core_head")
    assert not any("core_head" in key for key in plain.state_dict())
    arch = config().architecture
    assert isinstance(upper.core_head, nn.Linear)
    assert upper.core_head.weight.shape == (
        4,
        arch.stage_count * arch.state_tokens * arch.state_dim,
    )
    plain_model, upper_model = plain.model.state_dict(), upper.model.state_dict()
    assert plain_model.keys() == upper_model.keys()
    assert all(torch.equal(plain_model[k], upper_model[k]) for k in plain_model)
    with pytest.raises(ValueError):
        LossV3Module(config(), core_head_weight=-1.0)


def test_head_gradient_is_nonzero_and_distinct_when_c_is_collapsed():
    torch.manual_seed(3)
    head = nn.Linear(6, 4)
    c = torch.ones(8, 6, requires_grad=True)  # all variants identical
    cores = torch.tensor([core for core, _ in LABELS])
    core_head_cross_entropy(head, c, cores).mean().backward()
    assert c.grad.abs().sum() > 0
    assert not torch.allclose(c.grad[0], c.grad[2])  # different cores pull apart


def test_variant_contrast_is_zero_for_identical_variants():
    same = torch.ones(8, 3, 2)
    assert float(variant_contrast(same)) == 0
    split = torch.cat((torch.ones(4, 2), -torch.ones(4, 2)))
    assert float(variant_contrast(split)) == pytest.approx(1.0)


def test_upper_bound_forward_adds_weighted_ce_read_from_final_c():
    torch.manual_seed(5)
    module = LossV3Module(config(), core_head_weight=1.0)
    group = batch()
    loss, metrics = module([group], [12], [LABELS], phase="tiny")
    weighted = sum(v for k, v in metrics.items() if k.startswith("weighted/"))
    assert torch.allclose(loss.detach(), weighted)
    assert torch.allclose(metrics["weighted/core_ce"], metrics["core_ce"])
    with torch.no_grad():
        states = full_observation_states(module.model, [group], [group.observed])
        cores = torch.tensor([core for core, _ in LABELS])
        expected = core_head_cross_entropy(
            module.core_head, states[-1].c.flatten(1), cores
        ).mean()
    assert torch.allclose(metrics["core_ce"], expected, rtol=1e-5, atol=1e-6)
    assert 0 <= float(metrics["core_accuracy"]) <= 1
    assert float(metrics["active_source_fraction/core_clean"]) == 1
    assert torch.equal(metrics["core_accuracy_clean_sum"], metrics["core_accuracy"])
    assert float(metrics["active_source_fraction/core_masked"]) == 0
    assert float(metrics["core_accuracy_masked_sum"]) == 0
    assert "c_variant_contrast" in metrics
    loss.backward()
    assert module.core_head.weight.grad.abs().sum() > 0


def test_plain_v3_forward_reports_no_core_terms():
    torch.manual_seed(5)
    _, metrics = LossV3Module(config())([batch()], [12], [LABELS], phase="tiny")
    assert not any(key.startswith(("core_", "weighted/core")) for key in metrics)
    assert "c_variant_contrast" in metrics


def test_masked_sources_split_accuracy_metrics():
    cfg = config()
    group = batch()
    seeds = {
        bool((make_training_mask(group, cfg, torch.Generator().manual_seed(s), "main")
              & group.observed).all()): s
        for s in range(64)
    }
    assert set(seeds) == {True, False}
    torch.manual_seed(5)
    module = LossV3Module(cfg, core_head_weight=1.0)
    _, metrics = module(
        [group, group], [seeds[True], seeds[False]], [LABELS, LABELS], phase="main"
    )
    assert float(metrics["active_source_fraction/core_clean"]) == 0.5
    assert float(metrics["active_source_fraction/core_masked"]) == 0.5


def test_split_accuracy_is_identical_for_one_or_two_microbatches():
    cfg = config()
    first, second = batch(), batch(torch.rand(8, 1, 12, 3, 16, 24) * 2 - 1)
    flags = {
        bool((make_training_mask(first, cfg, torch.Generator().manual_seed(s), "main")
              & first.observed).all()): s
        for s in range(64)
    }
    seeds = [flags[True], flags[False]]
    torch.manual_seed(5)
    module = LossV3Module(cfg, core_head_weight=1.0)
    keys = ("core_accuracy_clean_sum", "core_accuracy_masked_sum",
            "active_source_fraction/core_clean", "active_source_fraction/core_masked")
    _, joint = module([first, second], seeds, [LABELS, LABELS], phase="main")
    together = split_core_accuracy({k: float(joint[k]) for k in keys})
    parts = [module([g], [s], [LABELS], phase="main")[1] for g, s in ((first, seeds[0]), (second, seeds[1]))]
    # optimization_step weights each microbatch by its share of source groups.
    apart = split_core_accuracy({k: sum(float(p[k]) / 2 for p in parts) for k in keys})
    for name in ("clean", "masked"):
        assert apart[f"core_accuracy_{name}"] == pytest.approx(together[f"core_accuracy_{name}"])
    assert apart["core_accuracy_clean"] == pytest.approx(float(parts[0]["core_accuracy"]))
    assert split_core_accuracy({"core_accuracy_clean_sum": 0.0,
                                "active_source_fraction/core_clean": 0.0})["core_accuracy_clean"] == 0.0


def test_labels_never_change_model_states():
    torch.manual_seed(5)
    module = LossV3Module(config(), core_head_weight=1.0)
    captured = []
    original = module.model.forward_sequence

    def record(*args, **kwargs):
        states = original(*args, **kwargs)
        captured.append(states[-1].c.detach().clone())
        return states

    module.model.forward_sequence = record
    relabelled = tuple((3 - core, config) for core, config in LABELS)
    first, _ = module([batch()], [12], [LABELS], phase="tiny")
    second, _ = module([batch()], [12], [relabelled], phase="tiny")
    assert torch.equal(captured[0], captured[1])
    assert not torch.equal(first.detach(), second.detach())


def test_upper_bound_reduction_is_invariant_to_microbatch_grouping():
    torch.manual_seed(5)
    module = LossV3Module(config(), core_head_weight=1.0)
    first, second = batch(), batch(torch.rand(8, 1, 12, 3, 16, 24) * 2 - 1)
    combined, _ = module([first, second], [11, 29], [LABELS, LABELS], phase="tiny")
    separate = sum(
        module([group], [seed], [LABELS], phase="tiny")[0] / 2
        for group, seed in ((first, 11), (second, 29))
    )
    assert torch.allclose(combined, separate, rtol=1e-5, atol=1e-6)


def test_probe_metrics_report_source_bootstrap_intervals():
    labels = [(c, d) for c in range(4) for d in range(2)]
    features = torch.eye(4).repeat_interleave(2, dim=0)
    entry = lambda name, x: {"source": name, "c": x, "d": x[:, :2], "labels": labels}
    dev = [entry(f"dev{i}", features + 0.01 * i) for i in range(5)]
    first = probe_metrics([entry("fit", features)], dev)
    second = probe_metrics([entry("fit", features)], dev)
    assert first == second  # fixed bootstrap seed
    for name in ("c_core4", "c_direction_family0", "d_configuration"):
        assert first[f"{name}_ci_low"] <= first[name] <= first[f"{name}_ci_high"]
        assert 0 <= first[f"{name}_train"] <= 1
    assert first["c_core4_train"] == 1  # separable training features fit exactly


def test_warm_start_from_b_checkpoint_is_rejected_with_arm_mismatch(tmp_path):
    from mavt4d.hybrid_loss_v3 import runner
    from mavt4d.hybrid_pilot.runner import DistributedContext

    module = LossV3Module(config(), core_head_weight=1.0)
    optimizer = runner._optimizer(module, config())
    lock_u = {key: f"value-{key}" for key in (
        "schema", "arm", "base_config", "base_config_sha256", "objective",
        "dataset_sha256", "manifest_sha256", "partition_sha256", "world_size",
        "spatial_bf16", "recurrent_decoder_dtype", "held_out_evaluated")}
    lock_u["arm"] = "U"
    metadata = {**lock_u, "arm": "B", "phase": "loss-v3-B", "step": 0, "next_step": 1,
                "selection_metric": "loss-v3-common-dev-v1"}
    path = tmp_path / "b.pt"
    torch.save({"schema_version": 1, "metadata": metadata, "module": {},
                "optimizer": {}, "rank_rng": [{}]}, path)
    with pytest.raises(ValueError, match="checkpoint arm mismatch"):
        runner.initialize_from_checkpoint(
            path, module, optimizer, DistributedContext(0, 1, 0, torch.device("cpu")), lock_u
        )


def test_tiny_upper_bound_trial_overfits_one_source_and_logs_u_evidence(
    tmp_path, monkeypatch
):
    from test_hybrid_pilot_objectives import tiny_config
    from mavt4d.hybrid_loss_v3 import runner

    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    base = replace(
        tiny_config(),
        source_count=12,
        dev_fraction=0.5,
        world_size=1,
        global_source_groups=1,
        microbatch_candidates=(1,),
    )
    script = ROOT / "scripts/mavt4d/build_event_suite.py"
    spec = importlib.util.spec_from_file_location("_ub_export", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    data = tmp_path / "data"
    exporter.build_event_suite(data, base.event_config, range(12))
    calls = []
    original_schedule, original_step = runner.scheduled_step, runner.optimization_step

    def schedule(*args, **kwargs):
        calls.append(("schedule", kwargs.get("overfit_group")))
        return original_schedule(*args, **kwargs)

    def step(*args, **kwargs):
        calls.append(("phase", kwargs.get("phase", "main")))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(runner, "scheduled_step", schedule)
    monkeypatch.setattr(runner, "optimization_step", step)
    cfg = replace(
        LossV3TrialConfig.load(ROOT / "configs/mavt4d/hybrid_loss_v3_upper_bound_tiny.json"),
        steps=2,
    )
    now = datetime.now(timezone.utc)
    output = tmp_path / "run"
    result = runner.run_trial(
        base, cfg, data, output,
        optimization_at=now + timedelta(hours=1),
        work_at=now + timedelta(hours=2),
        git_revision="b" * 40,
        device="cpu",
        allow_cpu=True,
        spatial_bf16=False,
    )
    assert result["step"] == 2
    lock = json.loads((output / "run_lock.json").read_text())
    assert lock["arm"] == "U"
    assert lock["objective"]["core_head"] == {
        "weight": 1.0, "classes": 4, "features": "final_c_flat", "overfit": True}
    overfit = [group for kind, group in calls if kind == "schedule"]
    assert overfit and all(group == lock["train_source_groups"][0] for group in overfit)
    assert {value for kind, value in calls if kind == "phase"} == {"tiny"}
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert "core_head.weight" in checkpoint["module"]
    assert checkpoint["metadata"]["phase"] == "loss-v3-U"
    journal = [json.loads(row) for row in (output / "progress.jsonl").read_text().splitlines()]
    assert any("train/core_accuracy" in row for row in journal)
    assert any("dev/c_variant_contrast" in row for row in journal)
    assert any("dev/c_core4_ci_low" in row for row in journal)


def test_b_trial_lock_has_no_core_head_and_keeps_b_phase(tmp_path, monkeypatch):
    from test_hybrid_pilot_objectives import tiny_config
    from mavt4d.hybrid_loss_v3 import runner

    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    base = replace(tiny_config(), source_count=12, dev_fraction=0.5, world_size=1,
                   global_source_groups=1, microbatch_candidates=(1,))
    spec = importlib.util.spec_from_file_location("_b_export", ROOT / "scripts/mavt4d/build_event_suite.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    data = tmp_path / "data"
    exporter.build_event_suite(data, base.event_config, range(12))
    now = datetime.now(timezone.utc)
    output = tmp_path / "b"
    runner.run_trial(
        base, LossV3TrialConfig(steps=1, effective_source_groups=1, microbatch_candidates=(1,)),
        data, output, optimization_at=now + timedelta(hours=1), work_at=now + timedelta(hours=2),
        git_revision="c" * 40, device="cpu", allow_cpu=True, spatial_bf16=False,
    )
    lock = json.loads((output / "run_lock.json").read_text())
    assert lock["arm"] == "B" and "core_head" not in lock["objective"]
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert checkpoint["metadata"]["phase"] == "loss-v3-B"
    assert not any("core_head" in key for key in checkpoint["module"])
    journal = [json.loads(row) for row in (output / "progress.jsonl").read_text().splitlines()]
    assert not any("train/core_accuracy" in row for row in journal)
