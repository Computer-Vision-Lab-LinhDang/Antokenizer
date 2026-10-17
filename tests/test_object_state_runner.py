"""Runner correctness on CPU; capacity logic uses injected measurements only."""

from dataclasses import replace

import pytest
import torch

from mavt4d.object_state.runner import SourceBatchSampler, capacity_search, run_update
from mavt4d.object_state.evaluation import (
    MetricTotals,
    fixed_identity_pose_metrics,
    evaluate,
)


def test_source_cursor_resume_reproduces_indices_across_epoch_and_tail():
    whole = list(
        SourceBatchSampler(
            12, seed=9, cursor=0, updates=4, effective_groups=4, micro_groups=3
        )
    )
    resumed = list(
        SourceBatchSampler(
            12, seed=9, cursor=4, updates=3, effective_groups=4, micro_groups=3
        )
    )
    assert whole[2:] == resumed
    assert [len(x) for x in whole] == [3, 1] * 4
    assert set(sum(whole, [])[:12]) == set(range(12))
    with pytest.raises(ValueError, match="divide"):
        SourceBatchSampler(
            5, seed=9, cursor=0, updates=3, effective_groups=4, micro_groups=3
        )


def test_capacity_search_finds_largest_safe_group_count():
    calls = []

    def probe(groups):
        calls.append(groups)
        return {"safe": groups <= 7, "peak_reserved_bytes": groups * 100}

    result = capacity_search(probe, 16)
    assert result["micro_groups"] == 7
    assert len(calls) == len(set(calls))
    assert result["measurements"][-1]["groups"] in calls
    with pytest.raises(RuntimeError, match="one source"):
        capacity_search(lambda n: {"safe": False}, 4)


def test_capacity_fixture_restores_parameters_rng_optimizer_and_method_after_oom():
    from mavt4d.object_state.model import ObjectStateModel
    from mavt4d.object_state.runner import _capacity_fixture

    architecture, _, _ = _tiny()
    torch.manual_seed(12)
    model = ObjectStateModel(architecture)
    optimizer = torch.optim.AdamW(model.parameters())
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    original_method = model.initial_state.__func__
    rng = torch.get_rng_state()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with _capacity_fixture(model, optimizer, seed=17):
            active = model.initial_state(2)
            assert active.allocated.all() and (active.p_exist > 0).all()
            assert active.d.abs().sum() > 0 and active.c.abs().sum() > 0
            torch.rand(10)
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            raise torch.cuda.OutOfMemoryError("test fault")
    assert model.initial_state.__func__ is original_method
    assert not model.initial_state(2).allocated.any()
    assert not optimizer.state and torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_worker_budget_avoids_shared_memory_overcommit_for_large_batches():
    from mavt4d.object_state.runner import worker_budget

    assert (
        worker_budget(
            12, sample_bytes=133 * 1024**2, micro_groups=256, shm_available=32 * 1024**3
        )
        == 0
    )
    assert (
        worker_budget(
            12, sample_bytes=133 * 1024**2, micro_groups=1, shm_available=32 * 1024**3
        )
        == 12
    )
    assert (
        worker_budget(
            12, sample_bytes=133 * 1024**2, micro_groups=32, shm_available=32 * 1024**3
        )
        <= 2
    )


def test_metrics_count_missing_hits_and_foreground_separately():
    meter = MetricTotals()
    gt = {
        "rgb": torch.zeros(1, 2, 3),
        "range": torch.tensor([[2.0, 0.0]]),
        "depth_valid": torch.tensor([[True, False]]),
        "instance": torch.tensor([[1, -1]]),
        "valid": torch.ones(1, 2, dtype=torch.bool),
    }
    pred = {
        "rgb": torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]),
        "range": torch.zeros(1, 2),
        "hit": torch.zeros(1, 2, dtype=torch.bool),
    }
    meter.add(pred, gt)
    result = meter.report()
    assert result["rgb_mae"] == 0.5 and result["rgb_foreground_mae"] == 1.0
    assert result["range_mae"] == 2.0 and result["target_hit_coverage"] == 0.0


def test_fixed_identity_pose_metric_does_not_rematch_swapped_slots():
    truth = torch.eye(4).repeat(1, 2, 1, 1)
    truth[0, :, 0, 3] = torch.tensor([-1.0, 1.0])
    result = fixed_identity_pose_metrics(
        truth.flip(1), truth, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0, 1]])
    )
    assert result["translation_error_sum"] == 4.0
    assert result["matched_objects"] == result["target_objects"] == 2


def _tiny():
    from mavt4d.object_state.config import ObjectStateConfig
    from mavt4d.object_state.training import TrainingConfig
    from mavt4d.object_state.data import FactorialSceneConfig

    architecture = ObjectStateConfig(
        slots=3,
        content_tokens=2,
        dynamic_tokens=2,
        width=8,
        background_tokens=2,
        evidence_width=12,
        spatial_blocks=1,
        spatial_heads=3,
        window_size=2,
        relay_tokens=2,
        resampler_heads=2,
        samples_per_ray=4,
        query_chunk_size=16,
        birth_min_peak=0.0,
        birth_min_mass=0.01,
    )
    config = TrainingConfig(
        frames=6,
        prefix_frames=4,
        height=16,
        width=32,
        rays_per_view=4,
        effective_groups=1,
        workers=0,
        max_updates=1,
        wandb_online=False,
    )
    scene = FactorialSceneConfig(
        frames=6, prefix_frames=4, height=16, width=32, max_objects=2
    )
    return architecture, config, scene


def test_real_update_and_sampled_eval_keep_causal_inputs_and_restore_mode():
    from torch.utils.data import DataLoader
    from mavt4d.object_state.model import ObjectStateModel
    from mavt4d.object_state.data import FactorialSceneDataset

    architecture, config, scene = _tiny()
    torch.manual_seed(17)
    model = ObjectStateModel(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    dataset = FactorialSceneDataset(1, scene)
    loader = iter(DataLoader(dataset, batch_size=1))
    before = model.decoder.control_head[0].weight.detach().clone()
    metrics = run_update(
        model,
        optimizer,
        loader,
        config,
        micro_groups=1,
        step=0,
        device=torch.device("cpu"),
    )
    assert metrics["loss"] > 0 and metrics["source_groups"] == 1
    assert not torch.equal(before, model.decoder.control_head[0].weight)
    model.train()
    result = evaluate(
        model,
        dataset,
        config,
        source_limit=1,
        rays=8,
        seed=55,
        deadline=lambda: True,
        visuals=False,
    )
    assert model.training and result["complete"]
    assert result["variants_completed"] == 8
    assert result["sampling"] == "fixed_seed_balanced_rays"
    assert "future/copy_last_rgb_observed" in result["metrics"]
    assert result["metrics"]["future/model"]["rays"] == 64


def test_eval_deadline_reports_incomplete_coverage():
    from mavt4d.object_state.model import ObjectStateModel
    from mavt4d.object_state.data import FactorialSceneDataset

    architecture, config, scene = _tiny()
    result = evaluate(
        ObjectStateModel(architecture),
        FactorialSceneDataset(1, scene),
        config,
        source_limit=1,
        rays=None,
        deadline=lambda: False,
        visuals=False,
    )
    assert not result["complete"] and result["sources_completed"] == 0
    assert result["sources_requested"] == 1 and result["variants_expected"] == 8


def test_cpu_checkpoint_resume_replays_optimizer_and_sampled_rays_exactly(tmp_path):
    from mavt4d.object_state.model import ObjectStateModel
    from mavt4d.object_state.data import FactorialSceneDataset
    from mavt4d.object_state.runner import _loader
    from mavt4d.object_state.training import save_checkpoint, restore_checkpoint

    architecture, config, scene = _tiny()
    dataset = FactorialSceneDataset(3, scene)
    torch.manual_seed(91)
    original = ObjectStateModel(architecture)
    optimizer = torch.optim.AdamW(original.parameters(), lr=config.lr)
    batches = iter(_loader(dataset, config, cursor=0, updates=2, micro_groups=1))
    run_update(
        original,
        optimizer,
        batches,
        config,
        micro_groups=1,
        step=0,
        device=torch.device("cpu"),
    )
    checkpoint = tmp_path / "step1.pt"
    contract = {"architecture": architecture.to_dict(), "training": config.to_dict()}
    save_checkpoint(
        checkpoint,
        original,
        optimizer,
        step=1,
        config=contract,
        extra={"source_cursor": 1},
    )
    expected = run_update(
        original,
        optimizer,
        batches,
        config,
        micro_groups=1,
        step=1,
        device=torch.device("cpu"),
    )
    resumed = ObjectStateModel(architecture)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=config.lr)
    restored = restore_checkpoint(
        checkpoint, resumed, resumed_optimizer, expected_config=contract
    )
    resumed_batches = iter(
        _loader(
            dataset,
            config,
            cursor=restored["extra"]["source_cursor"],
            updates=1,
            micro_groups=1,
        )
    )
    actual = run_update(
        resumed,
        resumed_optimizer,
        resumed_batches,
        config,
        micro_groups=1,
        step=1,
        device=torch.device("cpu"),
    )
    assert actual["loss"] == expected["loss"]
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_full_cpu_runner_writes_checkpoints_images_and_full_frame_counts(tmp_path):
    import json
    from datetime import datetime, timedelta, timezone
    from mavt4d.object_state.model import ObjectStateModel
    from mavt4d.object_state.data import FactorialSceneDataset
    from mavt4d.object_state.runner import RunLogger, RuntimeConfig, run_training

    architecture, config, scene = _tiny()
    now = datetime.now(timezone.utc)
    config = replace(
        config,
        train_until=(now + timedelta(minutes=5)).isoformat(),
        eval_until=(now + timedelta(minutes=6)).isoformat(),
        hard_stop=(now + timedelta(minutes=7)).isoformat(),
    )
    runtime = RuntimeConfig(eval_sources=1, periodic_eval_sources=1, eval_rays=8)
    model = ObjectStateModel(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    datasets = {
        split: FactorialSceneDataset(1, scene, split=split)
        for split in ("train", "dev", "test")
    }
    contract = {"architecture": architecture.to_dict(), "training": config.to_dict()}
    logger = RunLogger(tmp_path, config=contract, runtime=runtime, enabled=False)
    summary = run_training(
        model,
        optimizer,
        datasets,
        config,
        runtime,
        output=tmp_path,
        logger=logger,
        device=torch.device("cpu"),
        micro_groups=1,
        config_payload=contract,
    )
    assert summary["updates"] == 1 and summary["source_cursor"] == 1
    assert summary["final_evaluation_complete"]
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert (tmp_path / latest["checkpoint"]).exists()
    result = json.loads(
        (tmp_path / "final_test_full_frame-step-00000001.json").read_text()
    )
    assert result["frames_completed"] == result["frames_expected"] == 16
    assert result["metrics"]["future/model"]["rays"] == 8 * 2 * 16 * 32
    assert list((tmp_path / "visuals").glob("*.png"))
