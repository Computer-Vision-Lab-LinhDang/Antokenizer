"""A single stateless spatial batch must preserve physical-step causality."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from mavt4d.contracts import ObservationStep
from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.model import HybridCDModel


def observation(seed, time, *, frames=2, height=5, width=7,
                calibrated=False, missing=False):
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(2, frames, 3, height, width, generator=generator)
    observed = torch.rand(2, frames, height, width, generator=generator) > .25
    if missing:
        observed.zero_()
    images = images.masked_fill(~observed.unsqueeze(2), float("nan")).requires_grad_()
    times = torch.linspace(time + .1, time + .4, frames, dtype=torch.float64)
    times = times[None].expand(2, -1).clone()
    intrinsics = torch.eye(3).repeat(2, 1, 1) if calibrated else None
    poses = torch.eye(4).repeat(2, frames, 1, 1) if calibrated else None
    if poses is not None:
        poses[:, :, :3, 3] = torch.rand(2, frames, 3, generator=generator)
    return ObservationStep(images, observed, times, intrinsics, poses)


def schedule():
    return (
        [observation(1, 0, calibrated=True), observation(2, 0)],
        [observation(3, 1, frames=1, height=3), observation(4, 1, missing=True)],
        [observation(5, 2), observation(6, 2, calibrated=True)],
    )


def assert_state_close(actual, expected):
    for name in ("c", "d", "k", "v", "last_time", "step_count", "write_index",
                 "steps_filled", "slot_times", "slot_step_ids", "token_valid"):
        left, right = getattr(actual, name), getattr(expected, name)
        if name in ("c", "d"):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6, msg=name)
        elif name in ("k", "v"):
            torch.testing.assert_close(left.float(), right.float(), rtol=.008, atol=1e-4, msg=name)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)


def per_step_encoding(model, initial, steps):
    evidence, available, times = [], [], []
    previous_times = initial.last_time
    for item in steps:
        encoded, present, current = model._encode_step(previous_times, item, "camera")
        evidence.append(encoded)
        available.append(present)
        times.append(current)
        previous_times = current
    return model.forward_encoded(initial, torch.stack(evidence), torch.stack(available),
                                 torch.stack(times), method="associative")


def test_sequence_calls_stateless_observer_once_for_all_physical_steps(monkeypatch):
    model = HybridCDModel(HybridConfig.tiny(reference_views=2, spatial_blocks=4))
    original = model.observer.forward_batch
    calls = []

    def observe(items, *, modality="camera"):
        calls.append(len(items))
        return original(items, modality=modality)

    monkeypatch.setattr(model.observer, "forward_batch", observe)
    states = model.forward_sequence(model.initial_state(batch_size=2), schedule(),
                                    method="associative")
    assert len(states) == 3
    assert calls == [6]


@pytest.mark.parametrize("spatial_bf16", [False, True])
def test_flattened_sequence_matches_per_step_encoding_values_and_gradients(spatial_bf16):
    torch.manual_seed(431)
    reference = HybridCDModel(HybridConfig.tiny(reference_views=2, spatial_blocks=4))
    reference.observer.spatial_autocast = spatial_bf16
    flattened = deepcopy(reference)
    original_steps = schedule()
    copied_steps = tuple([
        replace(item, images=item.images.detach().clone().requires_grad_()) for item in step
    ] for step in original_steps)
    expected = per_step_encoding(reference, reference.initial_state(batch_size=2), original_steps)
    actual = flattened.forward_sequence(flattened.initial_state(batch_size=2), copied_steps,
                                        method="associative")
    for got, wanted in zip(actual, expected):
        assert_state_close(got, wanted)
    sum(item.c.square().sum() + item.d.square().sum() for item in expected).backward()
    sum(item.c.square().sum() + item.d.square().sum() for item in actual).backward()
    for (name, first), (_, second) in zip(reference.named_parameters(), flattened.named_parameters()):
        if first.grad is None:
            assert second.grad is None, name
        else:
            torch.testing.assert_close(second.grad, first.grad, rtol=.03, atol=2e-5, msg=name)
    for before_step, after_step in zip(original_steps, copied_steps):
        for before, after in zip(before_step, after_step):
            if before.images.grad is None:
                assert after.images.grad is None
            else:
                torch.testing.assert_close(after.images.grad, before.images.grad,
                                           rtol=.03, atol=2e-5)
                hidden = ~after.observed.unsqueeze(2).expand_as(after.images)
                assert torch.isfinite(after.images.grad).all()
                assert torch.count_nonzero(after.images.grad[hidden]) == 0


def test_flattened_spatial_batch_keeps_future_and_other_streams_isolated():
    torch.manual_seed(876)
    model = HybridCDModel(HybridConfig.tiny(reference_views=2, spatial_blocks=4))
    steps = schedule()
    changed = tuple([
        item if index < 2 and row == 0 else observation(90 + index * 2 + row, index,
                                                      calibrated=True)
        for row, item in enumerate(step)
    ] for index, step in enumerate(steps))
    with torch.no_grad():
        initial = model.initial_state(batch_size=2)
        original = model.forward_sequence(initial, steps, method="associative")
        altered = model.forward_sequence(initial, changed, method="associative")
        first = model.forward_sequence(initial, steps[:1], method="associative")
        chunked = first + model.forward_sequence(first[-1], steps[1:], method="associative")
        stepped, current = [], initial
        for item in steps:
            current = model.step(current, item)
            stepped.append(current)
    for expected, chunk, step in zip(original, chunked, stepped):
        assert_state_close(chunk, expected)
        assert_state_close(step, expected)
    for before, after in zip(original[:2], altered[:2]):
        for name in ("c", "d", "k", "v"):
            torch.testing.assert_close(getattr(after, name)[0], getattr(before, name)[0],
                                       rtol=3e-5, atol=3e-6, msg=name)


def test_empty_and_all_missing_sequences_preserve_noop_and_skip_spatial_compute():
    model = HybridCDModel(HybridConfig.tiny(reference_views=2, spatial_blocks=4))
    initial = model.initial_state()
    assert model.forward_sequence(initial, []) == ()
    calls = []
    handle = model.observer.patchify.register_forward_hook(lambda *_args: calls.append(True))
    try:
        first = observation(21, 0, missing=True)
        noop = replace(first, images=first.images[:, -1:], observed=first.observed[:, -1:],
                       frame_times=first.frame_times[:, -1:])
        states = model.forward_sequence(initial, [first, [noop], observation(22, 1, missing=True)])
    finally:
        handle.remove()
    assert not calls
    assert_state_close(states[1], states[0])
    assert states[2].step_count.tolist() == [2]
    assert torch.isfinite(states[-1].c).all() and torch.isfinite(states[-1].d).all()


@pytest.mark.parametrize("invalid,exception,match", [
    ("stale", ValueError, "stale"),
    ("batch", ValueError, "batch must match"),
    ("entry", TypeError, "entries must be ObservationStep"),
    ("physical", TypeError, "ObservationStep or a batch sequence"),
])
def test_sequence_keeps_physical_step_validation(invalid, exception, match):
    model = HybridCDModel(HybridConfig.tiny(reference_views=2))
    steps = list(schedule())
    if invalid == "stale":
        steps[1][0] = replace(steps[1][0], frame_times=steps[1][0].frame_times - 1)
    elif invalid == "batch":
        steps[1] = steps[1][:1]
    elif invalid == "entry":
        steps[1][0] = object()
    else:
        steps[1] = "invalid physical step"
    with pytest.raises(exception, match=match):
        model.forward_sequence(model.initial_state(batch_size=2), steps)
