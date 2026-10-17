"""Loss-v3 gates: pair evidence, reductions and gradients, without optimizer steps."""

from dataclasses import replace

import pytest
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid_pilot.config import HybridPilotConfig
from mavt4d.hybrid_pilot.data import TrainingBatch
from mavt4d.hybrid_loss_v3.objectives import (
    LossV3Module,
    c_objectives,
    pair_eligibility,
    region_balanced_l1,
    region_support,
    validate_labels,
)


LABELS = tuple((core, config) for core in range(4) for config in range(2))


def config():
    return replace(
        HybridPilotConfig(),
        architecture=HybridConfig.tiny(),
        H=16,
        W=24,
        temporal_pool=(2, 3),
        semantic_dim=4,
    )


def batch(images=None):
    if images is None:
        generator = torch.Generator().manual_seed(81)
        images = torch.rand(8, 1, 12, 3, 16, 24, generator=generator) * 2 - 1
    return TrainingBatch(
        images=images,
        observed=torch.ones(8, 1, 12, 16, 24, dtype=torch.bool),
        frame_times=torch.arange(12, dtype=torch.float64)[None, None].repeat(8, 1, 1)
        / 12,
        intrinsics=torch.tensor(
            [[[12.0, 0.0, 12.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]]]
        ).repeat(8, 1, 1, 1),
        cam_to_world=torch.eye(4).repeat(8, 1, 12, 1, 1),
    )


def permute(group, order):
    return TrainingBatch(
        **{
            name: getattr(group, name)[order]
            for name in (
                "images",
                "observed",
                "frame_times",
                "intrinsics",
                "cam_to_world",
            )
        }
    )


def test_labels_are_bijective_and_pairs_follow_labels_not_stream_position():
    order = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3])
    labels = tuple(LABELS[i] for i in order)
    group = permute(batch(), order)
    lookup = validate_labels(labels)
    eligibility = pair_eligibility(group, group.observed, labels, 3)
    for index, (core, configuration) in enumerate(labels):
        assert int(lookup[core, configuration]) == index
        assert labels[int(eligibility.config_partner[index])] == (
            core,
            1 - configuration,
        )
    with pytest.raises(ValueError, match="four cores"):
        validate_labels(LABELS[:-1] + (LABELS[0],))


def test_witness_requires_two_prefix_frames_both_configs_and_full_availability():
    images = torch.zeros(8, 1, 12, 3, 16, 24)
    # Reversal differs at frame 2 then 3, in BOTH configurations, four pixels.
    images[2:4, 0, 2:4, :, :2, :2] = 1
    group = batch(images)
    assert not pair_eligibility(group, group.observed, LABELS, 1).negative_witness[0, 1]
    assert not pair_eligibility(group, group.observed, LABELS, 2).negative_witness[0, 1]
    eligible = pair_eligibility(group, group.observed, LABELS, 3)
    assert eligible.negative_witness[0, 1]
    modified = images.clone()
    modified[:, :, 4:] = torch.rand_like(modified[:, :, 4:])
    assert torch.equal(
        eligible.negative_witness,
        pair_eligibility(batch(modified), group.observed, LABELS, 3).negative_witness,
    )
    # Equal rectangular masks do not establish equal evidence. Any missing pixel
    # in an involved prefix disqualifies its structured objectives.
    availability = group.observed.clone()
    availability[:, :, 0, 15, 23] = False
    hidden = pair_eligibility(group, availability, LABELS, 3)
    assert not hidden.negative_witness.any()
    assert not hidden.pair_valid.any() and not hidden.swap_valid.any()
    one_config_only = images.clone()
    one_config_only[3] = 0
    assert not pair_eligibility(
        batch(one_config_only), group.observed, LABELS, 3
    ).negative_witness[0, 1]


def test_region_union_includes_static_configuration_and_balances_nonempty_regions():
    images = torch.zeros(8, 1, 12, 3, 16, 24)
    images[1::2, :, :, :, 0, 0] = 0.5  # Static configuration difference.
    images[:, :, 2:, :, 1, 1] += 0.2  # Temporal difference.
    group = batch(images)
    support = region_support(group, LABELS)
    assert support.sum() == 2 and support[0, 0] and support[1, 1]
    prediction = torch.ones(2, 3, 16, 24, requires_grad=True)
    with torch.no_grad():
        prediction[:, :, support] = 3
    valid = torch.ones(2, 16, 24, dtype=torch.bool)
    valid[1] = False
    error, active, parts = region_balanced_l1(
        prediction, torch.zeros_like(prediction), valid, support
    )
    assert torch.equal(active, torch.tensor([True, False]))
    assert torch.allclose(error, torch.tensor([2.0, 0.0]))
    assert parts["support_l1"][0] == 3 and parts["background_l1"][0] == 1
    assert float(parts["whole_l1"][0].detach()) == pytest.approx((3 * 2 + 382) / 384)
    error.sum().backward()
    assert prediction.grad[0].abs().sum() > 0 and prediction.grad[1].abs().sum() == 0


def test_c_losses_use_raw_banks_and_balance_negative_categories():
    group = batch()
    eligibility = pair_eligibility(group, group.observed, LABELS, 3)
    # C0/C1 reversal separated; C2 duplicates C0 and C3 duplicates C1. The other
    # family still supplies violating negatives despite perfect reversal loss.
    c = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    c = c.repeat_interleave(2, 0).requires_grad_()
    terms = c_objectives(c, LABELS, eligibility)
    assert float(terms["order"][0].detach()) == pytest.approx(0.05)
    assert terms["c_pair"][0] == 0
    assert terms["c_var"][1]
    changed = c.detach().clone()
    changed[1, 1] += 0.2
    changed.requires_grad_()
    paired = c_objectives(changed, LABELS, eligibility)["c_pair"][0]
    # Mean over all coordinates per pair, then over four pairs: .04 / 2 / 4 / .01.
    assert float(paired.detach()) == pytest.approx(0.5)
    paired.backward()
    assert changed.grad[0, 1] != 0 and changed.grad[1, 1] != 0


def test_skipped_terms_are_zero_and_counted_without_nan():
    group = batch(torch.zeros(8, 1, 12, 3, 16, 24))
    eligibility = pair_eligibility(group, group.observed, LABELS, 3)
    assert eligibility.pair_valid.all()
    assert not eligibility.swap_valid.any() and not eligibility.negative_witness.any()
    c = torch.ones(8, 16, requires_grad=True)
    terms = c_objectives(c, LABELS, eligibility)
    assert terms["order"][0] == 0 and not terms["order"][1]
    assert terms["order"][2] == 0
    assert terms["c_var"][0] == 0 and not terms["c_var"][1]
    sum(value[0] for value in terms.values()).backward()
    assert torch.isfinite(c.grad).all()


def test_reciprocal_swap_uses_donor_target_and_never_resumes_swapped_code(monkeypatch):
    import mavt4d.hybrid_loss_v3.objectives as objectives

    module = LossV3Module(config())
    order = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3])
    labels = tuple(LABELS[index] for index in order)
    target_values = torch.tensor(
        [core * 0.2 + configuration * 0.1 for core, configuration in labels]
    )
    images = (
        target_values[:, None, None, None, None, None]
        .expand(8, 1, 12, 3, 16, 24)
        .clone()
    )
    group = batch(images)
    initial = module.model.initial_state(8, last_time=0.0)
    c = torch.ones_like(initial.c, requires_grad=True)
    d = target_values[:, None, None, None].expand_as(initial.d).clone().requires_grad_()
    source_state = replace(initial, c=c, d=d)
    monkeypatch.setattr(
        objectives,
        "full_observation_states",
        lambda *args, **kwargs: (source_state,) * 6,
    )
    rollout_prefixes = []

    def rollout(state, *args, **kwargs):
        rollout_prefixes.append(state)
        return (state, state)

    monkeypatch.setattr(module.model, "rollout", rollout)
    codes = []

    def render(code, **kwargs):
        codes.append(code)
        # The additive 0.03 deliberately gives a nonzero signed gradient to D.
        return (code.d[:, 0, 0, 0] + 0.03)[:, None, None, None].expand(8, 3, 16, 24)

    monkeypatch.setattr(module.model.decoder, "render", render)
    loss, metrics = module([group], [21], [labels], phase="tiny")
    assert metrics["swap"] == pytest.approx(0.03, abs=1e-6)
    assert metrics["valid_count/swap"] == 48
    assert len(rollout_prefixes) == 1 and rollout_prefixes[0] is source_state
    partners = pair_eligibility(group, group.observed, labels, 3).config_partner
    for index in range(6):
        self_code, swapped_code = codes[index * 2 : index * 2 + 2]
        assert torch.equal(self_code.c, swapped_code.c)
        assert torch.equal(swapped_code.d, self_code.d[partners])
        assert torch.equal(self_code.last_time, swapped_code.last_time)
    loss.backward()
    assert d.grad[:, 0, 0, 0].min() > 0


def test_missing_negative_category_renormalizes_and_partial_target_is_skipped():
    group = batch()
    eligibility = pair_eligibility(group, group.observed, LABELS, 3)
    # Only reversal is eligible, so it gets full weight rather than one half.
    witness = torch.eye(4, dtype=torch.bool)[:, torch.tensor([1, 0, 3, 2])]
    eligibility = replace(eligibility, negative_witness=witness)
    c = torch.ones(8, 8, requires_grad=True)
    terms = c_objectives(c, LABELS, eligibility)
    assert float(terms["order"][0].detach()) == pytest.approx(0.2)
    assert not terms["c_var"][1]
    images = torch.zeros_like(group.images)
    images[1::2, :, :, :, 0, :3] = 1  # Three pixels cannot activate the swap loss.
    too_small = batch(images)
    eligibility = pair_eligibility(too_small, too_small.observed, LABELS, 3)
    assert not eligibility.swap_valid.any()


def test_forward_preserves_model_contract_and_has_actual_c_d_gradients(monkeypatch):
    import mavt4d.hybrid_loss_v3.objectives as objectives

    torch.manual_seed(5)
    module = LossV3Module(config())
    assert set(module._modules) == {"model"}
    original = objectives.full_observation_states
    retained = []

    def record(*args, **kwargs):
        states = original(*args, **kwargs)
        for state in states:
            state.c.retain_grad()
            state.d.retain_grad()
        retained.extend(states)
        return states

    monkeypatch.setattr(objectives, "full_observation_states", record)
    loss, metrics = module([batch()], [12], [LABELS], phase="tiny")
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {
        "rec",
        "missing",
        "future",
        "order",
        "c_pair",
        "swap",
        "c_var",
        "loss",
    } <= metrics.keys()
    assert all(
        value.ndim == 0 and torch.isfinite(value) and not value.requires_grad
        for value in metrics.values()
    )
    assert len(retained) == 6  # No second augmentation assimilation.
    loss.backward()
    assert any(
        state.c.grad is not None and state.c.grad.abs().sum() > 0 for state in retained
    )
    assert any(
        state.d.grad is not None and state.d.grad.abs().sum() > 0 for state in retained
    )
    for component in (
        module.model.observer,
        module.model.cells,
        module.model.attentions,
        module.model.decoder,
    ):
        assert any(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
            for p in component.parameters()
        )


def test_equal_source_reduction_is_invariant_to_microbatch_grouping():
    torch.manual_seed(24)
    module = LossV3Module(config())
    first = batch()
    second = batch(torch.zeros_like(first.images))
    combined, _ = module([first, second], [11, 29], [LABELS, LABELS], phase="tiny")
    separate = sum(
        module([group], [seed], [LABELS], phase="tiny")[0] / 2
        for group, seed in ((first, 11), (second, 29))
    )
    assert torch.allclose(combined, separate, rtol=2e-5, atol=2e-6)
    parameter = module.model.decoder.tile_head.weight
    together_gradient = torch.autograd.grad(combined, parameter)[0]
    separate_gradient = torch.autograd.grad(separate, parameter)[0]
    assert torch.allclose(together_gradient, separate_gradient, rtol=3e-4, atol=2e-6)
