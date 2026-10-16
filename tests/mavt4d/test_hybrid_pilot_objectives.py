"""Actual hybrid optimization, causal masking, and source-group normalization."""
from dataclasses import replace

import pytest
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid_pilot.config import HybridPilotConfig
from mavt4d.hybrid_pilot.data import TrainingBatch
from mavt4d.hybrid_pilot.objectives import (
    HybridPilotModule, full_observation_states, ordered_history_target,
)


def tiny_config():
    return replace(HybridPilotConfig(), architecture=HybridConfig.tiny(),
                   H=16, W=24, temporal_pool=(2, 3), semantic_dim=4)


def tiny_batch(seed=88, camera_offset=0.):
    return TrainingBatch(
        images=torch.rand(8, 1, 12, 3, 16, 24, generator=torch.Generator().manual_seed(seed)) * 2 - 1,
        observed=torch.ones(8, 1, 12, 16, 24, dtype=torch.bool),
        frame_times=torch.arange(12, dtype=torch.float64)[None, None].repeat(8, 1, 1) / 12,
        intrinsics=torch.tensor([[[12., 0., 12.], [0., 12., 8.], [0., 0., 1.]]]).repeat(8, 1, 1, 1),
        cam_to_world=(torch.eye(4) + torch.tensor([[0., 0., 0., camera_offset], [0., 0., 0., 0.],
                                                 [0., 0., 0., 0.], [0., 0., 0., 0.]])).repeat(8, 1, 12, 1, 1),
    )


def test_real_loss_updates_all_modules_and_returns_detached_metrics():
    torch.manual_seed(10)
    module = HybridPilotModule(tiny_config())
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    optimizer = torch.optim.AdamW(module.parameters(), lr=3e-4)
    loss, metrics = module([tiny_batch()], [16], "main")
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"rec", "missing", "future", "temporal", "invariance", "noncollapse"} <= metrics.keys()
    assert all(value.ndim == 0 and torch.isfinite(value) and not value.requires_grad for value in metrics.values())
    loss.backward()
    for component in (module.model.observer, module.model.cells, module.model.attentions,
                      module.model.decoder, module.history_head, module.semantic_head):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in component.parameters())
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in module.named_parameters())


def test_microbatch_grouping_preserves_loss_and_gradient_with_different_cameras():
    torch.manual_seed(13)
    module = HybridPilotModule(tiny_config())
    first, second = tiny_batch(), tiny_batch(99, .2)
    together, _ = module([first, second], [1, 16])
    together.backward()
    together_grad = module.history_head.weight.grad.detach().clone()
    module.zero_grad(set_to_none=True)
    separate = sum(module([group], [seed])[0] / 2 for group, seed in zip((first, second), (1, 16)))
    separate.backward()
    assert torch.allclose(together, separate, rtol=2e-5, atol=2e-6)
    assert torch.allclose(together_grad, module.history_head.weight.grad, rtol=3e-4, atol=2e-6)


def test_masked_pixels_and_later_frames_never_enter_prefix_state():
    torch.manual_seed(4)
    module, batch = HybridPilotModule(tiny_config()), tiny_batch()
    available = batch.observed.clone()
    available[:, :, :6, :5, :5] = False
    images = batch.images.clone()
    images[:, :, 6:] = 700
    images[:, :, :6, :, :5, :5] = 900
    changed = replace(batch, images=images)
    first = full_observation_states(module.model, [batch], [available], stop_frame=6)
    second = full_observation_states(module.model, [changed], [available], stop_frame=6)
    assert len(first) == 3
    for a, b in zip(first, second):
        assert torch.equal(a.c, b.c) and torch.equal(a.d, b.d)
        assert torch.equal(a.k, b.k) and torch.equal(a.v, b.v)


def test_ordered_target_reverses_sign_and_never_uses_labels():
    cfg, batch = tiny_config(), tiny_batch()
    reverse = replace(cfg, temporal_pairs=tuple((b, a) for a, b in cfg.temporal_pairs))
    assert torch.equal(ordered_history_target(batch.images, reverse), -ordered_history_target(batch.images, cfg))


def test_groups_require_eight_variants_and_matching_seed_count():
    module = HybridPilotModule(tiny_config())
    with pytest.raises(ValueError, match="seed"):
        module([tiny_batch()], [])
