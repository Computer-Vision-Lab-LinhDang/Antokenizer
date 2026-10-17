import json
from datetime import datetime, timezone

import pytest
import torch

from mavt4d.object_state.training import (
    TrainingConfig,
    Deadline,
    microbatch_slices,
    save_checkpoint,
    restore_checkpoint,
)


def test_microbatch_weighting_uses_actual_tail():
    spans = list(microbatch_slices(64, 63))
    assert spans == [(0, 63, 63 / 64), (63, 64, 1 / 64)]
    assert sum(w for _, _, w in spans) == 1
    with pytest.raises(ValueError):
        list(microbatch_slices(0, 2))


def test_deadlines_absolute_and_leave_eval_buffer():
    cfg = TrainingConfig(
        train_until="2026-10-18T00:00:00Z",
        eval_until="2026-10-18T00:45:00Z",
        hard_stop="2026-10-18T00:55:00Z",
    )
    now = [datetime(2026, 10, 17, 23, 59, tzinfo=timezone.utc).timestamp()]
    d = Deadline(cfg, clock=lambda: now[0])
    assert d.can_train(estimated_seconds=30)
    assert not d.can_train(estimated_seconds=60)
    now[0] += 60
    assert not d.can_train()
    assert d.can_evaluate()
    now[0] += 45 * 60
    assert not d.can_evaluate()
    with pytest.raises(ValueError):
        TrainingConfig(train_until=cfg.hard_stop, eval_until=cfg.eval_until)


def test_checkpoint_restarts_optimizer_and_rng(tmp_path):
    torch.manual_seed(19)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(3, 2)).sum().backward()
    opt.step()
    opt.zero_grad()
    path = tmp_path / "checkpoint-00001.pt"
    save_checkpoint(path, model, opt, step=1, config={"seed": 19}, extra={"best": 2.0})
    expected_rng = torch.rand(4)
    expected = model(torch.ones(3, 2)).detach().clone()
    with torch.no_grad():
        model.weight.add_(10)
    result = restore_checkpoint(path, model, opt, expected_config={"seed": 19})
    assert result["step"] == 1
    assert torch.equal(torch.rand(4), expected_rng)
    assert torch.equal(model(torch.ones(3, 2)), expected)
    with pytest.raises(ValueError, match="config"):
        restore_checkpoint(path, model, opt, expected_config={"seed": 20})
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, opt, step=2, config={}, extra={})


def test_cpu_checkpoint_never_probes_cuda_availability(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    def forbidden():
        raise AssertionError("CPU checkpoint must not enumerate CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    path = tmp_path / "cpu.pt"
    save_checkpoint(path, model, optimizer, step=0, config={}, extra={})
    result = restore_checkpoint(path, model, optimizer, expected_config={})
    assert "cuda" not in result["rng"]
    result["rng"]["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
    gpu_path = tmp_path / "cuda-state.pt"
    torch.save(result, gpu_path)
    with pytest.raises(ValueError, match="initialized CUDA"):
        restore_checkpoint(gpu_path, model, optimizer, expected_config={})


def test_production_defaults_are_full_architecture_and_online():
    cfg = TrainingConfig()
    assert cfg.wandb_online
    assert cfg.effective_groups >= 16
    assert cfg.frames == 12 and cfg.prefix_frames == 6
    assert cfg.train_sources > cfg.dev_sources
    assert cfg.rays_per_view > 0 and cfg.width == 192
    json.dumps(cfg.to_dict())


def test_full_objective_filters_prefix_and_rolls_out_without_future_encoder():
    from torch.utils.data import default_collate
    from mavt4d.object_state.config import ObjectStateConfig
    from mavt4d.object_state.model import ObjectStateModel
    from mavt4d.object_state.data import (
        FactorialSceneConfig,
        FactorialSceneDataset,
        flatten_source_batch,
    )
    from mavt4d.object_state.training import compute_training_loss

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
        query_chunk_size=8,
        birth_min_peak=0.0,
        # Explicit cold-start fixture: no birth is admitted, regardless of the
        # random background/object ordering. Proposals must still learn.
        birth_min_mass=1e6,
    )
    scene = FactorialSceneConfig(
        frames=6, prefix_frames=4, height=16, width=32, max_objects=2
    )
    data = FactorialSceneDataset(1, scene)[0]
    batch = flatten_source_batch(default_collate([data]))
    config = TrainingConfig(
        frames=6,
        prefix_frames=4,
        height=16,
        width=32,
        rays_per_view=8,
        effective_groups=1,
        workers=0,
    )
    torch.manual_seed(17)
    model = ObjectStateModel(architecture)
    calls = []
    handle = model.observer.register_forward_hook(
        lambda m, args, out: calls.append(out.time.clone())
    )
    loss, metrics, state = compute_training_loss(model, batch, config, step=0)
    loss.backward()
    handle.remove()
    assert len(calls) == 2
    assert torch.isfinite(loss)
    assert float(state.t[0]) == pytest.approx(0.36)
    assert {
        "rgb",
        "depth",
        "opacity",
        "pose",
        "association",
        "existence",
        "swap",
    } <= set(metrics)
    assert not state.allocated.any()
    for parameter in (
        model.resampler.birth_descriptor,
        model.resampler.birth_evidence.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(
        p.grad is None or (p.grad == 0).all() for p in model.dynamics.parameters()
    )


def test_active_object_forecast_reaches_rates_and_forcing_without_encoder():
    from dataclasses import replace
    from mavt4d.object_state.config import ObjectStateConfig
    from mavt4d.object_state.model import ObjectStateModel
    from mavt4d.object_state.decoder import RayQuery
    from mavt4d.object_state.training import fixed_pose_loss

    torch.manual_seed(73)
    architecture = ObjectStateConfig(
        slots=2,
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
        query_chunk_size=8,
    )
    model = ObjectStateModel(architecture)
    initial = model.initial_state(1)
    initial = replace(
        initial,
        c=torch.randn_like(initial.c),
        d=torch.randn_like(initial.d),
        slot_status=torch.ones_like(initial.slot_status),
        slot_uid=torch.ones_like(initial.slot_uid),
        p_exist=torch.ones_like(initial.p_exist),
    )
    calls = []
    handle = model.observer.register_forward_hook(lambda *args: calls.append(True))
    futures = model.rollout(
        initial, torch.tensor([[0.12], [0.24]], dtype=torch.float64)
    )
    controls = model.decoder.controls(futures[-1].d)
    target = torch.eye(4).repeat(1, 2, 1, 1)
    target[0, :, 0, 3] = torch.tensor([-1.0, 1.0])
    pose = fixed_pose_loss(
        controls,
        target,
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0, 1]]),
        futures[-1].allocated,
    )
    query = RayQuery(
        torch.zeros(1, 8, 3),
        torch.tensor([0.0, 0.0, 1.0]).expand(1, 8, 3),
        0.1,
        3.0,
        futures[-1].t,
    )
    rendered = model.read(futures[-1], query)
    (
        pose + (rendered["rgb"] - 0.3).square().mean() + rendered["range"].mean()
    ).backward()
    handle.remove()
    assert not calls
    for state in futures:
        torch.testing.assert_close(state.c, initial.c, rtol=0, atol=0)
    for name in (
        "rates.weight",
        "forcing_u.weight",
        "forcing_b.weight",
        "transition.0.weight",
    ):
        gradient = dict(model.dynamics.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0, name


def test_future_pose_keeps_prefix_identity_assignment():
    from mavt4d.object_state.training import fixed_pose_loss

    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[0, :, 0, 3] = torch.tensor([-1.0, 1.0])
    reversed_slots = poses.flip(1).requires_grad_()
    loss = fixed_pose_loss(
        reversed_slots,
        poses,
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0, 1]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert float(loss.detach()) == pytest.approx(2 / 3)
    loss.backward()
    assert reversed_slots.grad.abs().sum() > 0
