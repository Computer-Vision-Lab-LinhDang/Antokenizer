"""Batched spatial encoding preserves per-stream evidence and FP32 state."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from mavt4d.contracts import ObservationStep
from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.observation import HybridObservationEncoder


def sample(seed, *, views=2, frames=2, height=5, width=7, calibrated=False, missing=False):
    rng = torch.Generator().manual_seed(seed)
    images = torch.rand(views, frames, 3, height, width, generator=rng)
    observed = torch.rand(views, frames, height, width, generator=rng) > .25
    if missing:
        observed.zero_()
    images = images.masked_fill(~observed.unsqueeze(2), float("nan")).requires_grad_()
    times = torch.arange(1, frames + 1, dtype=torch.float64)[None].expand(views, -1).clone()
    intrinsics = torch.eye(3).repeat(views, 1, 1) if calibrated else None
    poses = torch.eye(4).repeat(views, frames, 1, 1) if calibrated else None
    if poses is not None:
        poses[:, :, :3, 3] = torch.rand(views, frames, 3, generator=rng)
    return ObservationStep(images, observed, times, intrinsics, poses)


def copied_observations(observations):
    return [replace(item, images=item.images.detach().clone().requires_grad_()) for item in observations]


def test_batched_encoder_matches_single_outputs_and_gradients_with_mixed_geometry():
    torch.manual_seed(905)
    single = HybridObservationEncoder(HybridConfig.tiny(reference_views=2, spatial_blocks=4))
    batched = deepcopy(single)
    observations = [sample(1, calibrated=True), sample(2), sample(3, missing=True)]
    batch_observations = copied_observations(observations)
    expected = torch.stack([single(item)[0] for item in observations])
    actual, available = batched.forward_batch(batch_observations)
    assert available.tolist() == [True, True, False]
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[2]) == 0
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, first), (_, second) in zip(single.named_parameters(), batched.named_parameters()):
        if first.grad is None:
            assert second.grad is None, name
        else:
            torch.testing.assert_close(second.grad, first.grad, rtol=2e-4, atol=2e-5, msg=name)
    for before, after in zip(observations[:2], batch_observations[:2]):
        torch.testing.assert_close(after.images.grad, before.images.grad, rtol=2e-4, atol=2e-5)
        assert torch.isfinite(after.images.grad).all()
        assert torch.count_nonzero(after.images.grad[~after.observed.unsqueeze(2).expand_as(after.images)]) == 0


def test_batched_encoder_groups_ragged_shapes_and_preserves_input_order():
    encoder = HybridObservationEncoder(HybridConfig.tiny(reference_views=2, spatial_blocks=4))
    observations = [sample(10), sample(11, frames=1, height=3, width=4), sample(12),
                    sample(13, height=2, width=1, missing=True)]
    expected = torch.stack([encoder(item)[0] for item in observations])
    calls = []
    hook = encoder.patchify.register_forward_pre_hook(lambda _module, args: calls.append(args[0].shape[0]))
    actual, available = encoder.forward_batch(observations)
    hook.remove()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert available.tolist() == [True, True, True, False]
    assert sorted(calls) == [2, 4]  # Two shape groups, with V=2 per observation.


def test_batched_all_missing_skips_spatial_compute_and_invalid_input_is_rejected():
    encoder = HybridObservationEncoder(HybridConfig.tiny(reference_views=2))
    calls = []
    hook = encoder.patchify.register_forward_hook(lambda *_args: calls.append(True))
    values, available = encoder.forward_batch([sample(21, missing=True), sample(22, missing=True)])
    hook.remove()
    assert not calls and not available.any() and torch.count_nonzero(values) == 0
    with pytest.raises(ValueError, match="nonempty"):
        encoder.forward_batch([])
    with pytest.raises(TypeError, match="ObservationStep"):
        encoder.forward_batch([object()])


def test_batched_triplane_keeps_plane_identity():
    encoder = HybridObservationEncoder(HybridConfig.tiny(reference_views=3, spatial_blocks=4))
    observations = [sample(30, views=3, frames=1), sample(31, views=3, frames=1)]
    expected = torch.stack([encoder(item, modality="triplane")[0] for item in observations])
    actual, _ = encoder.forward_batch(observations, modality="triplane")
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_spatial_bfloat16_is_explicit_and_returns_fp32_evidence_and_gradients():
    encoder = HybridObservationEncoder(HybridConfig.tiny(reference_views=2, spatial_blocks=4))
    assert encoder.spatial_autocast is False
    encoder.spatial_autocast = True
    dtypes = {}
    handles = [encoder.patchify.register_forward_hook(lambda _m, _i, out: dtypes.update(patchify=out.dtype)),
               encoder.blocks[0].qkv.register_forward_hook(lambda _m, _i, out: dtypes.update(qkv=out.dtype)),
               encoder.pool_keys.register_forward_hook(lambda _m, _i, out: dtypes.update(pool=out.dtype))]
    values, _ = encoder.forward_batch([sample(40, calibrated=True), sample(41)])
    for handle in handles:
        handle.remove()
    assert dtypes == {"patchify": torch.bfloat16, "qkv": torch.bfloat16, "pool": torch.float32}
    assert values.dtype == torch.float32 and torch.isfinite(values).all()
    values.square().sum().backward()
    for parameter in encoder.parameters():
        assert parameter.dtype == torch.float32
        if parameter.grad is not None:
            assert parameter.grad.dtype == torch.float32 and torch.isfinite(parameter.grad).all()


def test_model_uses_one_batched_observer_call_per_physical_step(monkeypatch):
    model = HybridCDModel(HybridConfig.tiny(reference_views=2))
    original = model.observer.forward_batch
    calls = []

    def observe(items, *, modality="camera"):
        calls.append(len(items))
        return original(items, modality=modality)

    monkeypatch.setattr(model.observer, "forward_batch", observe)
    model.observer.spatial_autocast = True
    state = model.step(model.initial_state(batch_size=2), [sample(50), sample(51)])
    assert calls == [2]
    assert state.c.dtype == state.d.dtype == torch.float32
    assert state.k.dtype == state.v.dtype == torch.bfloat16
    assert torch.isfinite(state.c).all() and torch.isfinite(state.d).all()
