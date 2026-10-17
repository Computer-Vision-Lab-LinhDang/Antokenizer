"""Local optimizer-boundary recovery stays independent of selected W&B weights."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mavt4d.hybrid_loss_v3 import runner
from mavt4d.hybrid_loss_v3.config import LossV3TrialConfig


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from test_hybrid_pilot_objectives import tiny_config

    torch.set_num_threads(1)
    root = tmp_path / "data"
    root.mkdir()
    for filename in ("summary.json", "manifest_v1.jsonl"):
        (root / filename).write_text("{}\n")
    base = tiny_config()
    schedules, uploads = [], []

    class Dataset:
        train_groups = ("a", "b", "c", "d")
        dev_groups = ("e",)
        val_groups = ("f",)

        def __init__(self, *args):
            self.root = root

        def load_group(self, name, device):
            return float(ord(name) - ord("a") + 1)

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
            self.cfg = SimpleNamespace(grad_clip=1e6)

        def forward(self, batches, seeds, labels, phase):
            schedules.append(tuple(zip(batches, seeds)))
            # All three RNG streams must recover exactly after a restart.
            noise = (
                torch.rand((), dtype=torch.float64) + random.random() + np.random.rand()
            )
            loss = ((self.weight * torch.tensor(batches) - noise) ** 2).mean()
            return loss, {"objective": loss.detach()}

    class Logger:
        run = None
        run_id = "fresh-id"

        def __init__(self, output, **kwargs):
            self.output = output
            torch.rand(3)
            random.random()
            np.random.rand()

        def log(self, metrics, **kwargs):
            # SDK implementations are external and may consume any RNG stream.
            torch.rand(3)
            random.random()
            np.random.rand()

        def checkpoint(self, path, **kwargs):
            uploads.append(path.name)

        def finish(self, **kwargs):
            pass

    def dev(*args, **kwargs):
        torch.rand(4)
        random.random()
        np.random.rand()
        return {"complete": True, "S_dev": 0.2}

    monkeypatch.setattr(runner, "HybridPilotDataset", Dataset)
    monkeypatch.setattr(runner, "labels_for_dataset", lambda _: {})
    monkeypatch.setattr(runner, "initialize", lambda *a, **kw: Module())
    monkeypatch.setattr(runner, "RunLogger", Logger)
    monkeypatch.setattr(runner, "development", dev)
    monkeypatch.setattr(runner, "aligned_panel", lambda *a: ({}, torch.zeros(3, 2, 2)))
    # Labels exist even though the toy objective ignores them.
    monkeypatch.setattr(
        runner, "labels_for_dataset", lambda _: {g: [] for g in "abcdef"}
    )

    def run(name, steps, *, groups=2, microbatch=1, **kwargs):
        now = datetime.now(timezone.utc)
        return runner.run_trial(
            base,
            LossV3TrialConfig(
                steps=steps,
                effective_source_groups=groups,
                microbatch_candidates=(1, groups),
            ),
            root,
            tmp_path / name,
            optimization_at=now + timedelta(hours=1),
            work_at=now + timedelta(hours=2),
            git_revision="a" * 40,
            device="cpu",
            allow_cpu=True,
            spatial_bf16=False,
            microbatch_groups=microbatch,
            **kwargs,
        )

    return SimpleNamespace(run=run, root=tmp_path, schedules=schedules, uploads=uploads)


def read_last(harness, name):
    return torch.load(harness.root / name / "last.pt", weights_only=True)


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


def test_last_resume_matches_uninterrupted_optimizer_rng_and_schedule(harness):
    harness.run("full", 4)
    full_schedule = harness.schedules[:]
    harness.schedules.clear()
    first = harness.run("first", 2)
    resumed = harness.run("resumed", 4, resume=harness.root / "first")
    assert first["step"] == 2 and resumed["step"] == 4
    assert harness.schedules == full_schedule
    full, actual = read_last(harness, "full"), read_last(harness, "resumed")
    for key in ("module", "optimizer", "rank_rng"):
        assert_tree_equal(full[key], actual[key])
    assert actual["metadata"]["step"] == 4
    assert actual["metadata"]["next_step"] == 5
    assert actual["metadata"]["source_cursor"] == 8
    assert actual["metadata"]["clip_presentations"] == 64
    assert resumed["best_step"] == 0
    assert harness.uploads == ["best.pt"] * 3


def test_changed_effective_batch_retains_step_and_actual_exposure(harness):
    harness.run("first", 2)
    with pytest.raises(ValueError, match="allow.batch.change"):
        harness.run("reject", 3, groups=4, resume=harness.root / "first")
    resumed = harness.run(
        "resumed", 3, groups=4, resume=harness.root / "first", allow_batch_change=True
    )
    metadata = read_last(harness, "resumed")["metadata"]
    assert resumed["step"] == 3
    assert metadata["source_cursor"] == 8  # 2*2 + 1*4, never 3*4
    assert metadata["clip_presentations"] == 64
    assert metadata["initialization"]["exact_resume"] is False
    assert metadata["batch_transitions"][-1]["at_step"] == 2
    assert metadata["batch_transitions"][-1]["to_effective_source_groups"] == 4


def test_resume_requires_last_and_valid_historical_best(harness):
    harness.run("first", 1)
    with pytest.raises(ValueError, match="last"):
        harness.run("bad-best", 2, resume=harness.root / "first" / "best.pt")
    source = harness.root / "first"
    (source / "best.pt").write_bytes(b"corrupt selected model")
    with pytest.raises(ValueError, match="best.*(hash|companion)"):
        harness.run("bad-companion", 2, resume=source)
    with pytest.raises(FileNotFoundError, match="last"):
        harness.run("missing", 2, resume=harness.root / "missing-source")


def test_partial_accumulation_keeps_last_step_optimizer_and_cursor(
    harness, monkeypatch
):
    harness.run("clean", 1)
    clean = read_last(harness, "clean")
    original = runner.optimization_step
    count = 0

    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            # Use the real partial accumulation path, then return no commit.
            class Deadline:
                calls = 0

                def should_stop(self, context):
                    self.calls += 1
                    return self.calls > 1

            return original(*args[:7], Deadline(), **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "optimization_step", interrupt)
    result = harness.run("partial", 3)
    saved = read_last(harness, "partial")
    assert result["step"] == saved["metadata"]["step"] == 1
    assert saved["metadata"]["source_cursor"] == 2
    assert saved["optimizer"]["state"][0]["step"] == 1
    for key in ("module", "optimizer", "rank_rng"):
        assert_tree_equal(saved[key], clean[key])


def test_atomic_last_failure_preserves_previous_boundary_and_best(harness, monkeypatch):
    original = runner.save_checkpoint

    def fail_candidate(path, module, optimizer, metadata, context):
        if metadata.get("checkpoint_kind") == "last" and metadata["step"] == 2:
            raise OSError("simulated disk failure")
        return original(path, module, optimizer, metadata, context)

    monkeypatch.setattr(runner, "save_checkpoint", fail_candidate)
    with pytest.raises(OSError, match="disk failure"):
        harness.run("write-failed", 3)
    saved = read_last(harness, "write-failed")
    assert saved["metadata"]["step"] == 1
    assert saved["metadata"]["source_cursor"] == 2
    best = torch.load(harness.root / "write-failed" / "best.pt", weights_only=True)
    assert best["metadata"]["step"] == 0
    assert harness.uploads == []


def test_initial_snapshot_exists_before_initial_dev_failure(harness, monkeypatch):
    monkeypatch.setattr(
        runner,
        "development",
        lambda *a, **kw: {"complete": False, "reason": "deadline"},
    )
    with pytest.raises(RuntimeError, match="initial full dev"):
        harness.run("initial-failed", 1)
    saved = read_last(harness, "initial-failed")
    assert saved["metadata"]["step"] == 0
    assert saved["metadata"]["next_step"] == 1
    assert saved["optimizer"]["state"] == {}


def test_crash_after_best_replace_recovers_prior_selection_and_pending_eval(
    harness, monkeypatch
):
    calls = 0

    def improving_dev(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"complete": True, "S_dev": 0.2 if calls == 1 else 0.1}

    original = runner.save_last

    def crash_after_new_best(output, module, optimizer, metadata, context):
        best = metadata.get("best_selection")
        if metadata["step"] == 1 and best is not None and best["score"] == 0.1:
            raise OSError("crash after selected best replaced")
        return original(output, module, optimizer, metadata, context)

    monkeypatch.setattr(runner, "development", improving_dev)
    monkeypatch.setattr(runner, "save_last", crash_after_new_best)
    with pytest.raises(OSError, match="selected best"):
        harness.run("crash", 1)
    last = read_last(harness, "crash")["metadata"]
    assert last["step"] == 1 and last["best_selection"]["step"] == 0
    assert last["development_state"]["pending"] is True
    backup = harness.root / "crash" / ".resume-best.pt"
    assert backup.is_file()
    monkeypatch.setattr(runner, "save_last", original)
    monkeypatch.setattr(
        runner, "development", lambda *a, **kw: {"complete": True, "S_dev": 0.3}
    )
    result = harness.run("recovered", 1, resume=harness.root / "crash")
    assert result["step"] == 1 and result["best_step"] == 0
    assert (harness.root / "recovered" / "best.pt").read_bytes() == backup.read_bytes()
    assert (
        read_last(harness, "recovered")["metadata"]["development_state"]["pending"]
        is False
    )


def test_replace_failure_keeps_last_readable(harness, monkeypatch):
    original = runner._replace_durable
    replacements = 0

    def broken(candidate, destination):
        nonlocal replacements
        if destination.name == "last.pt":
            replacements += 1
            if replacements == 4:  # initial, initial-dev, step1, then step2
                raise OSError("replace unavailable")
        return original(candidate, destination)

    monkeypatch.setattr(runner, "_replace_durable", broken)
    with pytest.raises(OSError, match="replace unavailable"):
        harness.run("failed", 3)
    assert read_last(harness, "failed")["metadata"]["step"] == 1


@pytest.mark.parametrize(
    "key",
    [
        "base_config_sha256",
        "objective",
        "dataset_sha256",
        "manifest_sha256",
        "partition_sha256",
        "trial_seed",
        "spatial_bf16",
        "selection_metric",
    ],
)
def test_resume_rejects_immutable_contract_changes(harness, key):
    harness.run("first", 1)
    path = harness.root / "first" / "last.pt"
    saved = torch.load(path, weights_only=True)
    saved["metadata"][key] = "incompatible"
    torch.save(saved, path)
    with pytest.raises(ValueError, match=key):
        harness.run("rejected", 2, resume=path)


def test_microbatch_change_keeps_logical_schedule_without_explicit_batch_override(
    harness,
):
    harness.run("first", 1)
    result = harness.run("next", 2, microbatch=2, resume=harness.root / "first")
    metadata = read_last(harness, "next")["metadata"]
    expected = runner.scheduled_step(
        ("a", "b", "c", "d"), seed=17, step=2, global_groups=2
    )
    assert harness.schedules[-1] == tuple(
        (float(ord(g) - ord("a") + 1), seed) for g, seed in expected
    )
    assert result["step"] == 2 and metadata["source_cursor"] == 4
    assert metadata["initialization"]["same_effective_batch"] is True
    assert metadata["initialization"]["exact_resume"] is False


def test_resume_and_warmstart_are_mutually_exclusive(harness):
    with pytest.raises(ValueError, match="mutually exclusive"):
        harness.run("invalid", 1, resume="last.pt", initial_checkpoint="best.pt")
