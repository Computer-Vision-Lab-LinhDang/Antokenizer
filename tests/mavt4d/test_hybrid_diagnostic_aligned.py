"""The D readout chain is measured at one fixed, causal endpoint."""

from dataclasses import fields, replace
import math

import pytest
import torch

from mavt4d.hybrid_diagnostic import experiment
from mavt4d.hybrid_pilot.objectives import HybridPilotModule
from test_hybrid_pilot_objectives import tiny_batch, tiny_config


PREFIX = "aligned_d/frame_003"


def _rms(value):
    return float(value.detach().square().mean().sqrt())


@pytest.mark.parametrize("arm,strength", [("baseline", 0.1), ("d_strength", 1.0)])
def test_aligned_chain_uses_early_state_and_correct_configuration_donor(
    monkeypatch, arm, strength
):
    torch.manual_seed(19)
    module = HybridPilotModule(tiny_config())
    experiment.apply_intervention(module, experiment.DiagnosticConfig(arm=arm))
    # Saturated controls make omission of tanh/strength easy to detect.
    with torch.no_grad():
        for block in module.model.decoder.blocks:
            block.d_routing.weight.fill_(2.0)
        module.model.decoder.d_gain.weight.fill_(3.0)
    # Deliberately shuffled labels: an adjacent-index swap is wrong.
    labels = [(3, 1), (0, 0), (2, 1), (1, 0), (3, 0), (0, 1), (2, 0), (1, 1)]
    swap, _ = experiment.permutations(labels, "cpu")
    initial = module.model.initial_state(8)
    core = torch.tensor([c + 1.0 for c, _ in labels]).reshape(8, 1, 1, 1)
    configuration = torch.tensor([float(d) for _, d in labels]).reshape(8, 1, 1, 1)
    states = [
        replace(
            initial,
            c=torch.ones_like(initial.c) * core,
            d=torch.ones_like(initial.d) * (configuration + index),
        )
        for index in range(6)
    ]
    states[1] = replace(states[1], d=torch.ones_like(initial.d) * configuration)
    # The old unqualified diagnostics must keep reading the final state.
    states[-1] = replace(
        states[-1], d=torch.ones_like(initial.d) * (10 + 100 * configuration)
    )
    saved_states = [
        {
            field.name: getattr(state, field.name).clone()
            for field in fields(state)
            if isinstance(getattr(state, field.name), torch.Tensor)
        }
        for state in states
    ]
    original_weights = {
        key: value.clone() for key, value in module.state_dict().items()
    }
    batch = tiny_batch()
    pattern = torch.zeros(1, 3, batch.images.shape[-2], batch.images.shape[-1])
    pattern[..., :8, :] = 1.0
    calls = []

    def render(model, state, supplied_batch, frame):
        assert model is module.model and supplied_batch is batch
        calls.append((state, frame))
        c = state.c.mean((1, 2, 3)).reshape(-1, 1, 1, 1)
        d = state.d.mean((1, 2, 3)).reshape(-1, 1, 1, 1)
        return c * (0.1 + 0.1 * d) * pattern

    images = torch.zeros_like(batch.images)
    images[:, 0, 3] = core * (0.1 + 0.1 * configuration) * pattern
    batch = replace(batch, images=images)
    monkeypatch.setattr(
        experiment, "full_observation_states", lambda *args, **kwargs: states
    )
    monkeypatch.setattr(experiment, "render_group", render)
    monkeypatch.setattr(
        module.model,
        "rollout",
        lambda *args, **kwargs: pytest.fail("D swap must not roll out"),
    )
    result, panel = experiment.measure(module, batch, labels, images=False)

    assert result["aligned_d/protocol_version"] == 2.0
    assert result[f"{PREFIX}/state/d_configuration_rms"] == pytest.approx(1.0)
    assert result["state/d_configuration_distance"] == pytest.approx(100.0)
    early = states[1]
    modulation = module.model.decoder.d_bottleneck(early.d.flatten(1))
    assert result[f"{PREFIX}/modulation/configuration_rms"] == pytest.approx(
        _rms(modulation - modulation[swap])
    )
    final_modulation = module.model.decoder.d_bottleneck(states[-1].d.flatten(1))
    assert result["decoder/modulation_configuration_distance"] == pytest.approx(
        _rms(final_modulation - final_modulation[swap])
    )
    for index, block in enumerate(module.model.decoder.blocks):
        routing = strength * block.d_routing(modulation).tanh()
        assert result[
            f"{PREFIX}/routing/block_{index}/configuration_rms"
        ] == pytest.approx(_rms(routing - routing[swap]))
    gain_deviation = strength * module.model.decoder.d_gain(modulation).tanh()
    assert result[f"{PREFIX}/gain/configuration_rms"] == pytest.approx(
        _rms(gain_deviation - gain_deviation[swap])
    )
    assert result[f"{PREFIX}/gain/deviation_rms"] == pytest.approx(_rms(gain_deviation))

    # Half the image moves, making all-pixel and region metrics distinct.
    assert result[f"{PREFIX}/moving_pixel_fraction"] == pytest.approx(0.5)
    assert result[f"{PREFIX}/d_swap/output_l1"] == pytest.approx(0.125)
    assert result[f"{PREFIX}/d_swap/moving_output_l1"] == pytest.approx(0.25)
    assert result[f"{PREFIX}/base/original_target_l1"] == pytest.approx(0.0)
    assert result[f"{PREFIX}/base/paired_target_l1"] == pytest.approx(0.125)
    assert result[f"{PREFIX}/d_swap/original_target_l1"] == pytest.approx(0.125)
    assert result[f"{PREFIX}/d_swap/paired_target_l1"] == pytest.approx(0.0)
    assert result[f"{PREFIX}/d_swap/paired_target_gain"] == pytest.approx(0.125)
    assert result[f"{PREFIX}/d_swap/moving_paired_target_gain"] == pytest.approx(0.25)
    assert result[f"{PREFIX}/d_zero/paired_target_gain"] == pytest.approx(0.0625)
    assert result["intervention/d_swap/pixel_l1"] == pytest.approx(0.125)
    assert result["intervention/d_swap/moving_pixel_l1"] == pytest.approx(0.25)
    assert result["intervention/d_swap/paired_target_gain"] == pytest.approx(0.25)

    # Rendering keeps the receiver's C, cache, and clocks, never the donor's.
    interventions = calls[6:]
    assert len(interventions) == 3 and all(frame == 3 for _, frame in interventions)
    d_swap_state = interventions[1][0]
    assert torch.equal(d_swap_state.d, early.d[swap])
    for name in ("c", "k", "v", "last_time", "step_count", "slot_times"):
        assert getattr(d_swap_state, name) is getattr(early, name)
    assert panel is None and all(
        type(value) is float and math.isfinite(value) for value in result.values()
    )
    assert all(
        torch.equal(saved[name], getattr(state, name))
        for saved, state in zip(saved_states, states)
        for name in saved
    )
    assert all(
        torch.equal(original_weights[key], value)
        for key, value in module.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in module.parameters())
    assert module.training


def test_aligned_outputs_keep_the_legacy_target_only_motion_support():
    """Frame alignment does not silently redefine the reported pixel region."""
    torch.manual_seed(20)
    module = HybridPilotModule(tiny_config())
    batch = tiny_batch()
    labels = [(c, d) for c in range(4) for d in range(2)]
    images = torch.zeros_like(batch.images)
    images[:, 0, 3, :, :4] = 0.2
    images[:, 0, 8, :, 8:] = 0.8
    batch = replace(batch, images=images)
    result, panel = experiment.measure(module, batch, labels, images=False)
    assert result[f"{PREFIX}/moving_pixel_fraction"] == pytest.approx(0.75)
    assert result["image/moving_pixel_fraction"] == pytest.approx(0.75)
    assert result[f"{PREFIX}/moving_pixel_count"] == 8.0 * 12.0 * 24.0
    assert result[f"{PREFIX}/moving_valid"] == 1.0
    assert (
        result[f"{PREFIX}/d_swap/moving_output_l1"]
        == result["intervention/d_swap/moving_pixel_l1"]
    )
    assert (
        result[f"{PREFIX}/d_swap/moving_paired_target_gain"]
        == result["intervention/d_swap/paired_target_gain"]
    )
    assert panel is None


def test_empty_same_frame_motion_support_is_explicit_and_finite():
    torch.manual_seed(21)
    module = HybridPilotModule(tiny_config())
    batch = tiny_batch()
    images = batch.images[:, :, :1].expand_as(batch.images).clone()
    result, _ = experiment.measure(
        module,
        replace(batch, images=images),
        [(c, d) for c in range(4) for d in range(2)],
        images=False,
    )
    assert result[f"{PREFIX}/moving_valid"] == 0.0
    assert result[f"{PREFIX}/moving_pixel_fraction"] == 0.0
    assert result[f"{PREFIX}/d_swap/moving_output_l1"] == 0.0
    assert all(
        type(value) is float and math.isfinite(value) for value in result.values()
    )


def test_paired_target_gain_remains_signed_when_intervention_hurts():
    module = HybridPilotModule(tiny_config())
    early = module.model.initial_state(8)
    swap, _ = experiment.permutations(
        [(c, d) for c in range(4) for d in range(2)], "cpu"
    )
    target = torch.arange(8.0).reshape(8, 1, 1, 1).expand(8, 3, 2, 2)
    moving = torch.ones(8, 2, 2, dtype=torch.bool)
    result = experiment._aligned_d_readout_metrics(
        module.model.decoder,
        early,
        swap,
        target,
        {"d_swap": target + 10.0},
        target,
        moving,
        frame=3,
    )
    assert result[f"{PREFIX}/d_swap/paired_target_gain"] == pytest.approx(-9.0)
    assert result[f"{PREFIX}/d_swap/moving_paired_target_gain"] == pytest.approx(-9.0)
