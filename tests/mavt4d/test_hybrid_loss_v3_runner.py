"""One-device v3 trial, frozen selection, and best-only restart evidence."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mavt4d.hybrid_loss_v3.config import LossV3TrialConfig
from mavt4d.hybrid_loss_v3.evaluation import probe_metrics
from mavt4d.hybrid_loss_v3.runner import run_trial, save_selected, validate_device
from mavt4d.hybrid_pilot.runner import DistributedContext


def test_config_locks_single_gpu_and_accumulation():
    cfg = LossV3TrialConfig()
    assert cfg.dev_every == 200 and cfg.seed == 17
    assert cfg.accumulation(1) == cfg.effective_source_groups
    assert cfg.accumulation(cfg.effective_source_groups) == 1
    assert LossV3TrialConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        replace(cfg, dev_every=100)
    with pytest.raises(ValueError):
        cfg.accumulation(3)
    with pytest.raises(ValueError):
        replace(cfg, seed=29)


def test_production_rejects_extra_gpu_and_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one"):
        validate_device("cpu", allow_cpu=True)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="one"):
        validate_device("cuda:0", allow_cpu=False)


def test_probe_selection_uses_train_only_and_both_core_metrics():
    labels = [(c, d) for c in range(4) for d in range(2)]
    features = torch.eye(4).repeat_interleave(2, dim=0)
    entries = lambda name: {
        "source": name,
        "c": features,
        "d": features[:, :2],
        "labels": labels,
    }
    scores = probe_metrics([entries("fit")], [entries("dev")])
    assert scores["c_direction"] == 1 and scores["c_core4"] == 1
    assert scores["E_C"] == 0
    with pytest.raises(ValueError, match="disjoint"):
        probe_metrics([entries("same")], [entries("same")])


def test_incomplete_dev_never_replaces_best_and_ties_keep_earlier(tmp_path):
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(module.parameters())
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    metadata = {"step": 0, "next_step": 1}
    best = save_selected(
        tmp_path,
        module,
        optimizer,
        {"complete": False},
        float("inf"),
        metadata,
        context,
    )
    assert best == float("inf") and not list(tmp_path.glob("*.pt"))
    best = save_selected(
        tmp_path,
        module,
        optimizer,
        {"complete": True, "S_dev": 0.2},
        best,
        metadata,
        context,
    )
    original = (tmp_path / "best.pt").read_bytes()
    save_selected(
        tmp_path,
        module,
        optimizer,
        {"complete": True, "S_dev": 0.2},
        best,
        {"step": 1, "next_step": 2},
        context,
    )
    assert (tmp_path / "best.pt").read_bytes() == original
    assert (
        torch.load(tmp_path / "best.pt", weights_only=True)["metadata"][
            "selection_metric"
        ]
        == "loss-v3-common-dev-v1"
    )


def test_one_update_real_tiny_trial_complete_dev_and_images(tmp_path, monkeypatch):
    from test_hybrid_pilot_objectives import tiny_config

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
    script = Path(__file__).resolve().parents[2] / "scripts/mavt4d/build_event_suite.py"
    spec = importlib.util.spec_from_file_location("_v3_export", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    data = tmp_path / "data"
    exporter.build_event_suite(data, base.event_config, range(12))
    now = datetime.now(timezone.utc)
    output = tmp_path / "run"
    result = run_trial(
        base,
        LossV3TrialConfig(
            steps=1, effective_source_groups=1, microbatch_candidates=(1,)
        ),
        data,
        output,
        optimization_at=now + timedelta(hours=1),
        work_at=now + timedelta(hours=2),
        git_revision="a" * 40,
        device="cpu",
        allow_cpu=True,
        spatial_bf16=False,
    )
    assert result["step"] == 1 and result["last_dev_step"] == 1
    assert result["held_out_evaluated"] is False
    assert {p.name for p in output.glob("*.pt")} == {"best.pt"}
    lock = json.loads((output / "run_lock.json").read_text())
    assert lock["world_size"] == 1 and len(lock["dev_source_groups"]) > 2
    assert set(lock["train_source_groups"]).isdisjoint(lock["dev_source_groups"])
    checkpoint = torch.load(output / "best.pt", weights_only=True)
    assert len(checkpoint["rank_rng"]) == 1
    assert checkpoint["metadata"]["next_step"] == result["best_step"] + 1
    assert not any(
        "head" in key for key in checkpoint["module"] if not key.startswith("model.")
    )
    assert len(list((output / "images").glob("*.jpg"))) == 2
    journal = [
        json.loads(row) for row in (output / "progress.jsonl").read_text().splitlines()
    ]
    assert any("dev/S_dev" in row for row in journal)
    assert any(
        "diagnostic/aligned_d/frame_003/modulation/configuration_rms" in row
        for row in journal
    )


def test_interrupted_accumulation_does_not_apply_partial_update():
    from mavt4d.hybrid_loss_v3.runner import optimization_step

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, batches, seeds, labels, phase):
            return self.weight.square(), {}

    class Data:
        def load_group(self, name, device):
            return None

    class Deadline:
        calls = 0

        def should_stop(self, context):
            self.calls += 1
            return self.calls > 1

    module = Module()
    optimizer = torch.optim.AdamW(module.parameters())
    result = optimization_step(
        module,
        optimizer,
        Data(),
        {"a": [], "b": []},
        [("a", 1), ("b", 2)],
        1,
        DistributedContext(0, 1, 0, torch.device("cpu")),
        Deadline(),
    )
    assert result is None and float(module.weight.detach()) == 2.0
    assert module.weight.grad is None and len(optimizer.state) == 0


def test_capacity_chooses_fastest_safe_complete_candidate():
    from mavt4d.hybrid_loss_v3.runner import choose_capacity

    rows = [
        dict(
            microbatch_groups=1,
            status="complete",
            headroom_ok=True,
            seconds_per_update=3.0,
        ),
        dict(
            microbatch_groups=2,
            status="complete",
            headroom_ok=True,
            seconds_per_update=2.0,
        ),
        dict(
            microbatch_groups=4,
            status="complete",
            headroom_ok=False,
            seconds_per_update=1.0,
        ),
        dict(microbatch_groups=8, status="incomplete_deadline"),
    ]
    assert choose_capacity(rows) == 2
    assert choose_capacity(rows[2:]) is None


def test_capacity_is_disposable_and_clean_peak_precedes_main_timing(
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
    script = Path(__file__).resolve().parents[2] / "scripts/mavt4d/build_event_suite.py"
    spec = importlib.util.spec_from_file_location("_v3_capacity_export", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    data = tmp_path / "data"
    exporter.build_event_suite(data, base.event_config, range(12))
    before = runner.initialize(
        base, torch.device("cpu"), spatial_bf16=False
    ).state_dict()
    phases = []
    original = runner.optimization_step

    def tracked(*args, **kwargs):
        phases.append(kwargs["phase"])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "optimization_step", tracked)
    now = datetime.now(timezone.utc)
    output = tmp_path / "capacity"
    result = runner.capacity_probe(
        base,
        LossV3TrialConfig(
            effective_source_groups=1, microbatch_candidates=(1,), capacity_repeats=1
        ),
        data,
        output,
        optimization_at=now + timedelta(hours=1),
        work_at=now + timedelta(hours=2),
        device="cpu",
        allow_cpu=True,
        spatial_bf16=False,
    )
    assert result["selected_microbatch"] == 1
    assert phases == ["tiny", "main"]
    assert {p.name for p in output.iterdir()} == {"capacity.json"}
    after = runner.initialize(
        base, torch.device("cpu"), spatial_bf16=False
    ).state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
