"""Runtime invariants independent of the scientific objective implementation."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import random

import pytest
import torch

from mavt4d.hybrid_pilot import runner
from mavt4d.hybrid_pilot.logging import RunLogger


def test_schedule_is_global_deterministic_and_microbatch_independent():
    groups = ("source-a", "source-b", "source-c")
    whole = runner.scheduled_step(groups, seed=17, step=4, global_groups=32)
    parts = [runner.rank_schedule(whole, rank=r, world_size=8) for r in range(8)]
    assert sum((list(x) for x in parts), []) == list(whole)
    torch.rand(11)
    assert runner.scheduled_step(groups, seed=17, step=4, global_groups=32) == whole
    assert runner.scheduled_step(groups, seed=18, step=4, global_groups=32) != whole
    assert len({seed for _, seed in whole}) == 32
    assert {g for g, _ in runner.scheduled_step(groups, seed=17, step=4,
        global_groups=32, overfit_group="source-b")} == {"source-b"}


def test_checkpoint_restores_optimizer_rng_and_next_step(tmp_path):
    torch.manual_seed(4)
    random.seed(9)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    context = runner.DistributedContext(0, 1, 0, torch.device("cpu"))
    metadata = dict(config_sha256="a" * 64, dataset_sha256="b" * 64,
        git_revision="1" * 40, phase="main", step=7, next_step=8,
        source_cursor=224, world_size=1, wandb_run_id="run-id")
    path = tmp_path / "checkpoint.pt"
    runner.save_checkpoint(path, model, optimizer, metadata, context)
    expected_random = (torch.rand(5), random.random())
    clone = torch.nn.Linear(3, 1)
    clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=.9)
    loaded = runner.load_checkpoint(path, clone, clone_optimizer, context,
        expected={k: metadata[k] for k in ("config_sha256", "dataset_sha256", "phase", "world_size")})
    assert loaded["next_step"] == 8 and loaded["source_cursor"] == 224
    assert loaded["wandb_run_id"] == "run-id"
    assert torch.equal(torch.rand(5), expected_random[0])
    assert random.random() == expected_random[1]
    assert all(torch.equal(value, clone.state_dict()[name]) for name, value in expected.items())
    assert clone_optimizer.param_groups[0]["lr"] == .01
    with pytest.raises(ValueError, match="config_sha256"):
        runner.load_checkpoint(path, clone, clone_optimizer, context,
            expected={"config_sha256": "c" * 64})
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="immutable"):
        runner.save_checkpoint(path, clone, clone_optimizer, metadata, context)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_deadline_and_signal_only_request_boundary_stop():
    now = datetime.now(timezone.utc)
    context = runner.DistributedContext(0, 1, 0, torch.device("cpu"))
    deadline = runner.DeadlineController(now + timedelta(hours=1), now + timedelta(hours=2))
    assert not deadline.should_stop(context)
    deadline.handle_signal(15, None)
    assert deadline.requested and deadline.should_stop(context)
    expired = runner.DeadlineController(now - timedelta(seconds=1), now + timedelta(hours=1))
    assert expired.should_stop(context)
    assert not expired.should_stop(context, optimization=False)
    with pytest.raises(ValueError, match="before"):
        runner.DeadlineController(now + timedelta(hours=2), now + timedelta(hours=1))


def test_rank_zero_online_logging_only_and_checkpoint_artifact(tmp_path):
    calls = []
    class FakeRun:
        id = "run-id"
        url = "https://wandb.example/project/runs/run-id"
        def log(self, values, step=None): calls.append(("log", values, step))
        def define_metric(self, name, **kwargs): calls.append(("metric", name, kwargs))
        def log_artifact(self, artifact, aliases): calls.append(("artifact", artifact, aliases))
        def finish(self, exit_code): calls.append(("finish", exit_code))
    class Artifact:
        def __init__(self, name, type, metadata): self.name = name
        def add_file(self, filename, name): calls.append(("file", name))
    def initialize(**kwargs):
        calls.append(("init", kwargs))
        return FakeRun()
    FakeWandb = SimpleNamespace(Artifact=Artifact, init=initialize)
    # Supplying the fake SDK proves rank>0 never calls init or logs anything.
    other = RunLogger(tmp_path / "other", rank=1, config={"seed": 17}, sdk=FakeWandb)
    other.log({"train/loss": 1}, step=1)
    assert calls == []
    logger = RunLogger(tmp_path, rank=0, config={"seed": 17}, sdk=FakeWandb)
    logger.log({"train/loss": 1.25}, step=1)
    checkpoint = tmp_path / "step-1.pt"
    checkpoint.write_bytes(b"checkpoint")
    logger.checkpoint(checkpoint, metadata={"step": 1})
    logger.finish()
    assert calls[0][1]["mode"] == "online"
    assert calls[0][1]["name"] == "semantic router"
    assert (tmp_path / "wandb-run.json").exists()
    assert json.loads((tmp_path / "progress.jsonl").read_text())["train/loss"] == 1.25
    assert any(call[0] == "artifact" for call in calls)


def test_resumed_wandb_uses_backend_steps_when_sdk_step_is_stale(tmp_path):
    (tmp_path / "progress.jsonl").write_text(json.dumps({"step": 1403,
        "progress/optimizer_step": 3, "progress/phase": "main"}) + "\n")
    accepted, explicit_steps, definitions = [], [], []
    class StaleRun:
        id = "existing-run"
        url = "https://wandb.example/project/runs/existing-run"
        step = 0  # Observed W&B resume state despite the server's later history.
        server_next_step = 1404
        def log(self, values, **kwargs):
            explicit_steps.append(kwargs.get("step"))
            offered_step = kwargs.get("step", self.server_next_step)
            if offered_step < self.server_next_step:
                return  # W&B drops stale explicit history steps.
            accepted.append((offered_step, dict(values)))
            self.server_next_step = offered_step + 1
        def define_metric(self, name, **kwargs):
            definitions.append((name, kwargs))
        def finish(self, exit_code):
            pass
    run = StaleRun()
    sdk = SimpleNamespace(init=lambda **kwargs: run)
    logger = RunLogger(tmp_path, rank=0, config={"seed": 17}, run_id=run.id, sdk=sdk)
    logger.log({"main/loss": .8, "progress/optimizer_step": 4, "progress/phase": "main"}, step=204)
    logger.log({"progress/status": "stopped"}, step=1403)
    logger.finish()
    assert len(accepted) == 2 and explicit_steps == [None, None]
    assert accepted[0][1]["progress/optimizer_step"] == 4
    assert accepted[0][1]["progress/logical_step"] == 204
    assert [step for step, _ in accepted] == [1404, 1405]
    local = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [row["step"] for row in local] == [1403, 1404, 1405]
    assert ("*", {"step_metric": "progress/logical_step", "step_sync": True}) in definitions


def test_production_rejects_cpu_before_work(tmp_path):
    with pytest.raises(ValueError, match="eight GPU"):
        runner.validate_execution(runner.DistributedContext(0, 1, 0, torch.device("cpu")),
            allow_cpu=False)


def test_microbatch_accumulation_and_resume_match_uninterrupted_optimizer(tmp_path, monkeypatch):
    class Objective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.4, -.2]))
        def forward(self, groups, objective_seeds, phase):
            losses = []
            for group, seed in zip(groups, objective_seeds):
                noise = torch.rand((), generator=torch.Generator().manual_seed(seed))
                losses.append(((self.weight * group - noise) ** 2).mean())
            loss = torch.stack(losses).mean()
            return loss, {"loss": loss.detach()}
    class Dataset:
        train_groups = ("source-a", "source-b")
        overfit_group = "source-a"
        def load_group(self, group, device):
            return torch.tensor(1. if group == "source-a" else 2., device=device)
    cfg = SimpleNamespace(global_source_groups=4, microbatch_candidates=(1, 2, 4),
        overfit_steps=2, train_steps=2, seed=17, learning_rate=.01,
        betas=(.9, .95), weight_decay=.01, grad_clip=1., log_every=1,
        checkpoint_every=100, batch_size=8, T=12)
    cfg.to_dict = lambda: {"seed": cfg.seed}
    monkeypatch.setattr(runner, "initialize_module", lambda *args, **kwargs: Objective())
    context = runner.DistributedContext(0, 1, 0, torch.device("cpu"))
    now = datetime.now(timezone.utc)
    deadlines = runner.DeadlineController(now + timedelta(hours=1), now + timedelta(hours=2))
    lock = {"config_sha256": "a" * 64, "dataset_sha256": "b" * 64, "git_revision": "1" * 40}
    def train(folder, microbatch, resume=None, cap=None):
        output = tmp_path / folder
        logger = RunLogger(output, rank=0, config={}, enabled=False)
        return runner.train_phase(cfg, Dataset(), "main", output, context, deadlines,
            logger, lock, microbatch=microbatch, resume=resume, max_steps=cap)
    full, complete = train("full", 4)
    accumulated, _ = train("accumulated", 1)
    torch.testing.assert_close(full.weight, accumulated.weight, atol=1e-7, rtol=1e-7)
    _, partial = train("resume", 1, cap=1)
    assert partial["status"] == "stopped" and partial["step"] == 1
    checkpoint = tmp_path / "resume" / partial["checkpoint"]
    resumed, result = train("resume", 1, resume=checkpoint)
    assert result["status"] == complete["status"] == "complete"
    assert result["step"] == 2
    assert torch.equal(resumed.weight, accumulated.weight)
    payload = torch.load(tmp_path / "resume" / result["checkpoint"], weights_only=True)
    assert payload["metadata"]["config"] == cfg.to_dict()
    with pytest.raises(RuntimeError, match="rewind"):
        train("resume", 1, resume=checkpoint)


def _two_rank_worker(rank, rendezvous, output):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    class Objective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.4))
            self.unused = torch.nn.Parameter(torch.tensor(.9))
        def forward(self, groups, objective_seeds, phase):
            loss = torch.stack([(self.weight * group - torch.rand((),
                generator=torch.Generator().manual_seed(seed))) ** 2
                for group, seed in zip(groups, objective_seeds)]).mean()
            return loss, {"loss": loss.detach()}
    class Dataset:
        train_groups = ("source-a", "source-b")
        overfit_group = "source-a"
        def load_group(self, group, device):
            return torch.tensor(1. if group == "source-a" else 2., device=device)
    class StopOnSecondBoundary(runner.DeadlineController):
        calls = 0
        def should_stop(self, context, *, optimization=True):
            self.calls += 1
            if rank == 1 and self.calls == 2:
                self.requested = True
            return super().should_stop(context, optimization=optimization)
    cfg = SimpleNamespace(global_source_groups=4, microbatch_candidates=(1, 2),
        overfit_steps=3, train_steps=3, seed=17, learning_rate=.01,
        betas=(.9, .95), weight_decay=.01, grad_clip=1., log_every=1,
        checkpoint_every=100, batch_size=8, T=12)
    cfg.to_dict = lambda: {"seed": cfg.seed}
    runner.initialize_module = lambda *args, **kwargs: Objective()
    context = runner.DistributedContext(rank, 2, rank, torch.device("cpu"))
    now = datetime.now(timezone.utc)
    deadlines = StopOnSecondBoundary(now + timedelta(hours=1), now + timedelta(hours=2))
    logger = RunLogger(output, rank=rank, config={}, enabled=False)
    module, result = runner.train_phase(cfg, Dataset(), "main", Path(output), context, deadlines,
        logger, {"config_sha256": "a" * 64, "dataset_sha256": "b" * 64, "git_revision": "1" * 40},
        microbatch=1)
    weights = [torch.zeros_like(module.weight) for _ in range(2)]
    dist.all_gather(weights, module.weight.detach())
    assert torch.equal(weights[0], weights[1])
    assert result["status"] == "stopped" and result["step"] == 1
    assert module.unused.item() == pytest.approx(.9)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        runner.require_finite_loss(torch.tensor(float("nan") if rank == 1 else 1.), context)
    def fail_logging():
        raise OSError("simulated logging failure")
    with pytest.raises(RuntimeError, match="simulated logging failure"):
        runner.rank_zero_call(context, fail_logging)
    dist.destroy_process_group()


def test_two_rank_ddp_accumulation_unused_parameter_and_coherent_stop(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_two_rank_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path / "run")),
             nprocs=2, join=True)
    checkpoint = torch.load(tmp_path / "run" / "main" / "step-000001-stopped.pt", weights_only=True)
    assert len(checkpoint["rank_rng"]) == 2
    assert checkpoint["metadata"]["next_step"] == 2
    assert checkpoint["metadata"]["source_cursor"] == 4


def test_hybrid_feature_controls_reset_full_state_and_ignore_order(tmp_path):
    from mavt4d.hybrid.model import HybridCDModel
    from mavt4d.hybrid.config import HybridConfig
    from mavt4d.hybrid_pilot.evaluation import extract_features
    from mavt4d.hybrid_pilot.data import TrainingBatch
    from mavt4d.data.event_synthetic import EVENT_CORES, EVENT_CONFIGURATIONS
    architecture = HybridConfig.tiny()
    frames = torch.rand(12, 3, 4, 4, generator=torch.Generator().manual_seed(11)) * 2 - 1
    # The histories reverse their first ten frames but share the final two;
    # their complete frame multisets therefore remain exactly equal.
    reverse = torch.cat((frames[:10].flip(0), frames[10:]), dim=0)
    batch = TrainingBatch(
        images=torch.stack((frames, reverse))[:, None],
        observed=torch.ones(2, 1, 12, 4, 4, dtype=torch.bool),
        frame_times=torch.arange(12, dtype=torch.float64)[None, None].repeat(2, 1, 1) / 12,
        intrinsics=torch.eye(3).repeat(2, 1, 1, 1),
        cam_to_world=torch.eye(4).repeat(2, 1, 12, 1, 1))
    labels = [{"record_id": f"record-{i}", "group_key": "source", "split": "train",
               "core": EVENT_CORES[i], "configuration": EVENT_CONFIGURATIONS[0]} for i in range(2)]
    raw = "".join(json.dumps(row) + "\n" for row in labels).encode()
    (tmp_path / "eval_targets_v1.jsonl").write_bytes(raw)
    import hashlib
    (tmp_path / "summary.json").write_text(json.dumps({"sha256": {"evaluation": hashlib.sha256(raw).hexdigest()}}))
    dataset = SimpleNamespace(root=tmp_path, load_group=lambda *args, **kwargs: batch,
        records_for_group=lambda _: labels)
    torch.manual_seed(7)
    model = HybridCDModel(architecture)
    features = extract_features(model, dataset, ["source"], SimpleNamespace(T=12), "cpu")
    assert features.features["c"].shape == (2, 16)  # both stages
    for name in ("last_c", "last_d", "orderless"):
        assert torch.equal(features.features[name][0], features.features[name][1])
    assert not torch.equal(features.features["c"][0], features.features["c"][1])


@pytest.mark.parametrize("rows,columns", [(12, 40), (40, 12)])
def test_dual_ridge_matches_primal_parameters_and_predictions(rows, columns):
    from mavt4d.pilot.evaluation import fit_ridge_probe
    from mavt4d.hybrid_pilot.evaluation import fit_efficient_ridge_probe
    generator = torch.Generator().manual_seed(18)
    features = torch.randn(rows, columns, generator=generator)
    features[:, -1] = 4  # Keep the locked zero-variance scale floor covered.
    labels = torch.arange(rows) % 4
    first = fit_ridge_probe(features, labels, classes=4, ridge=.7)
    second = fit_efficient_ridge_probe(features, labels, classes=4, ridge=.7)
    for name in ("mean", "scale", "weights", "intercept"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), atol=1e-12, rtol=1e-12)
    heldout = torch.randn(8, columns, generator=generator)
    assert torch.equal(first.predict(heldout), second.predict(heldout))


def _probe_table(split, dimensions=40):
    from mavt4d.pilot.evaluation import FEATURE_NAMES, FeatureTable, FeatureRecord
    records = tuple(FeatureRecord(f"{split}-{c}-{d}", split, split, c, d)
                    for c in range(4) for d in range(2))
    semantic = torch.cat((torch.eye(4).repeat_interleave(2, dim=0), torch.eye(2).repeat(4, 1)), dim=1)
    projection = torch.randn(6, dimensions, generator=torch.Generator().manual_seed(91))
    return FeatureTable(records, {name: semantic @ projection for name in FEATURE_NAMES})


def test_probe_suite_default_unchanged_and_dual_preserves_all_scores():
    from mavt4d.pilot.evaluation import evaluate_probe_suite, fit_ridge_probe
    from mavt4d.hybrid_pilot.evaluation import fit_efficient_ridge_probe
    train, val, cfg = _probe_table("train"), _probe_table("val"), SimpleNamespace(probe_ridge=1.)
    original = evaluate_probe_suite(train, val, cfg)
    explicit = evaluate_probe_suite(train, val, cfg, fit_probe=fit_ridge_probe)
    dual = evaluate_probe_suite(train, val, cfg, fit_probe=fit_efficient_ridge_probe)
    assert original == explicit == dual


def test_evaluation_reports_source_intervals_and_wandb_progress(tmp_path, monkeypatch):
    from mavt4d.hybrid_pilot import evaluation
    monkeypatch.setattr(evaluation, "extract_features", lambda model, dataset, groups, cfg, device: _probe_table(groups[0]))
    monkeypatch.setattr(evaluation, "reconstruction_group", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "_reconstruction_summary", lambda groups, entries: {"tested": True})
    cfg = SimpleNamespace(probe_ridge=1., bootstrap=100, bootstrap_seed=31415, eval_reconstruction_groups=1)
    dataset = SimpleNamespace(train_groups=("train",), val_groups=("val",))
    context = runner.DistributedContext(0, 1, 0, torch.device("cpu"))
    now = datetime.now(timezone.utc)
    deadlines = runner.DeadlineController(now + timedelta(hours=1), now + timedelta(hours=2))
    logger = RunLogger(tmp_path, rank=0, config={}, enabled=False)
    result = evaluation.evaluate_distributed(torch.nn.Linear(1, 1), cfg, dataset, context,
        deadlines, logger, global_step=1)
    assert result["status"] == "complete"
    assert result["source_intervals"]["c_direction"] == {"mean": 1., "low": 1., "high": 1., "n_groups": 1}
    logged = [json.loads(row) for row in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert len(logged) == 3 and logged[-1]["eval/trained/c_direction"] == 1.
    assert [row["step"] for row in logged] == [1, 2, 3]


def test_d_swap_uses_opposite_configuration_same_core_with_c_fixed(tmp_path, monkeypatch):
    from dataclasses import dataclass
    import hashlib
    from mavt4d.hybrid_pilot import evaluation
    from mavt4d.data.event_synthetic import EVENT_CORES, EVENT_CONFIGURATIONS
    rows = [{"record_id": f"row-{core}-{configuration}", "group_key": "val", "split": "val",
             "core": EVENT_CORES[core], "configuration": EVENT_CONFIGURATIONS[configuration]}
            for core in range(4) for configuration in range(2)]
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (tmp_path / "eval_targets_v1.jsonl").write_bytes(raw)
    (tmp_path / "summary.json").write_text(json.dumps({"sha256": {"evaluation": hashlib.sha256(raw).hexdigest()}}))
    configuration = (torch.arange(8) % 2).float()
    batch = SimpleNamespace(images=configuration[:, None, None, None, None, None].expand(8, 1, 12, 3, 4, 4),
        observed=torch.ones(8, 1, 12, 4, 4, dtype=torch.bool))
    dataset = SimpleNamespace(root=tmp_path, load_group=lambda *args, **kwargs: batch,
                              records_for_group=lambda _: rows)
    @dataclass
    class State:
        c: torch.Tensor
        d: torch.Tensor
    state = State(torch.ones(8, 2, 1, 1), configuration[:, None, None, None].expand(8, 2, 1, 1).clone())
    monkeypatch.setattr(evaluation, "full_observation_states", lambda *args, **kwargs: (state,))
    def render(model, current, batch, frame):
        return (current.c[:, 0, 0, 0] * current.d[:, 0, 0, 0])[:, None, None, None].expand(8, 3, 4, 4)
    monkeypatch.setattr(evaluation, "render_group", render)
    entry = evaluation.reconstruction_group(None, dataset, "val", SimpleNamespace(eval_mask_seed=2718), "cpu", source_index=0)
    summary = evaluation._reconstruction_summary(["val"], [entry])
    assert summary["c_zero_exact_zero_verified"]
    for family in summary["families"].values():
        assert family["all"]["l1"] == 0.
        swap = family["interventions"]["d_swapped"]
        assert swap["all_l1_change"] == 1.
        assert swap["opposite_configuration_target"]["all"]["l1"] == 0.
        assert swap["opposite_configuration_l1_gain"] == 1.


@pytest.mark.parametrize("completed_training", [False, True])
def test_heldout_barrier_and_incomplete_matched_control_status(tmp_path, monkeypatch, completed_training):
    from mavt4d.hybrid_pilot import data, evaluation
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "summary.json").write_text("{}")
    cfg = SimpleNamespace(world_size=1, config_sha256="a" * 64, overfit_steps=1, train_steps=2,
                          to_dict=lambda: {"seed": 17})
    monkeypatch.setattr(data, "HybridPilotDataset", lambda *args: SimpleNamespace(root=dataset_root))
    monkeypatch.setattr(runner, "initialize_distributed", lambda **kwargs: runner.DistributedContext(0, 1, 0, torch.device("cpu")))
    module = SimpleNamespace(model=torch.nn.Linear(1, 1))
    monkeypatch.setattr(runner, "train_phase", lambda *args, **kwargs: (module,
        {"phase": "main", "status": "complete" if completed_training else "stopped", "step": 2 if completed_training else 1}))
    monkeypatch.setattr(runner, "initialize_module", lambda *args, **kwargs: module)
    calls = []
    def evaluate(*args, **kwargs):
        assert completed_training
        calls.append(kwargs.get("label", "trained"))
        return {"status": "complete" if len(calls) == 1 else "stopped"}
    monkeypatch.setattr(evaluation, "evaluate_distributed", evaluate)
    now = datetime.now(timezone.utc)
    result = runner.run_pilot(cfg, dataset_root, tmp_path / "run", phase="main", microbatch=1,
        optimization_at=now + timedelta(hours=1), work_at=now + timedelta(hours=2),
        git_revision="1" * 40, allow_cpu=True)
    assert result["status"] == ("training_complete_evaluation_incomplete" if completed_training else "stopped")
    assert calls == (["trained", "untrained"] if completed_training else [])
