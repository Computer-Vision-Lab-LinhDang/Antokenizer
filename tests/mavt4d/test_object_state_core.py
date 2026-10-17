"""State sufficiency, causal transitions and typed object-memory contracts."""

from dataclasses import replace
import io

import pytest
import torch
from torch import nn

from mavt4d.object_state.config import ObjectStateConfig
from mavt4d.object_state.state import ObjectState, Evidence
from mavt4d.object_state.dynamics import SelectiveDynamics
from mavt4d.object_state.resampler import ObjectResampler
from mavt4d.object_state.model import ObjectStateModel


def config():
    return ObjectStateConfig(
        slots=3,
        content_tokens=2,
        dynamic_tokens=2,
        width=12,
        background_tokens=3,
        evidence_width=12,
        spatial_heads=3,
        resampler_heads=3,
        candidates=5,
        spatial_blocks=2,
        relay_tokens=2,
    )


class Controls(nn.Module):
    def controls(self, d):
        result = (
            torch.eye(4, device=d.device, dtype=d.dtype)
            .expand(*d.shape[:2], 4, 4)
            .clone()
        )
        result[..., :3, 3] = d.mean(-2)[..., :3]
        return result

    def forward(self, state, query):
        return {"value": state.c.sum((-2, -1)) + state.d.sum((-2, -1))}


def network():
    torch.manual_seed(140)
    return ObjectStateModel(config(), observer=nn.Identity(), decoder=Controls())


def evidence(batch=2, count=8, time=1.0):
    torch.manual_seed(51)
    origins = torch.zeros(batch, count, 3)
    directions = torch.randn(batch, count, 3)
    directions = torch.nn.functional.normalize(directions, dim=-1)
    return Evidence(
        torch.randn(batch, count, 12),
        torch.ones(batch, count, dtype=torch.bool),
        origins,
        directions,
        torch.full((batch,), time, dtype=torch.float64),
    )


def allocated(model):
    state = model.initial_state(2)
    return replace(
        state,
        c=torch.randn_like(state.c),
        d=torch.randn_like(state.d),
        slot_status=torch.ones_like(state.slot_status),
        slot_uid=torch.ones_like(state.slot_uid),
        p_exist=torch.ones_like(state.p_exist),
    )


def assert_equal(a, b):
    for name, value in a.tensor_dict().items():
        assert torch.equal(value, b.tensor_dict()[name]), name


def test_default_full_tensor_contract_and_native_dynamic_memory():
    cfg = ObjectStateConfig()
    state = ObjectState.initial(cfg, 2, device="cpu")
    assert state.c.shape == (2, 16, 8, 128)
    assert state.d.shape == (2, 16, 8, 128)
    assert state.c_bg.shape == (2, 16, 128)
    assert state.flat_tokens().shape == (2, 272, 128)
    assert state.native_d.shape == (2, 16, 128, 8)
    assert (
        state.native_d.untyped_storage().data_ptr()
        == state.d.untyped_storage().data_ptr()
    )
    assert state.learned_nbytes == 2 * 139264
    state.validate(cfg)


def test_zero_dt_identity_and_missing_observation_exact_bypass():
    model = network()
    state = allocated(model)
    same = model.predict(state, torch.zeros(2))
    assert_equal(same, state)
    future = model.predict(state, torch.tensor([0.25, 0.5]))
    assert torch.equal(future.c, state.c)
    assert torch.equal(future.c_bg, state.c_bg)
    assert torch.equal(future.slot_uid, state.slot_uid)
    assert torch.equal(future.last_seen, state.last_seen)
    assert torch.equal(future.p_exist, state.p_exist)
    assert (future.slot_status == 2).all()
    missing = replace(
        evidence(),
        valid=torch.zeros(2, 8, dtype=torch.bool),
        time=future.t,
        tokens=torch.full((2, 8, 12), float("nan")),
    )
    assert_equal(model.correct(future, missing), future)


def test_mixed_batch_missing_row_and_empty_slots_do_not_evolve():
    model = network()
    state = model.initial_state(2)
    future = model.predict(state, torch.ones(2))
    assert torch.equal(future.d, state.d)
    item = evidence()
    item = replace(
        item,
        valid=torch.stack(
            [torch.ones(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool)]
        ),
        tokens=item.tokens.masked_fill(
            torch.arange(2)[:, None, None] == 1, float("nan")
        ),
    )
    result = model.correct(future, item)
    assert torch.equal(result.c[1], future.c[1])
    assert torch.equal(result.d[1], future.d[1])
    assert torch.equal(result.c_bg[1], future.c_bg[1])
    assert torch.isfinite(result.flat_tokens()).all()
    assert (result.slot_status[0] > 0).any()
    assert not result.slot_status[1].any()


def test_ownership_competes_across_objects_and_background_and_rebinds_birth():
    model = network()
    state = model.predict(model.initial_state(2), torch.ones(2))
    result, details = model.correct(state, evidence(), return_diagnostics=True)
    weights = details["ownership"]
    assert weights.shape == (2, 8, 4)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 8))
    assert torch.equal(details["candidate_uid"], result.slot_uid)
    assert (result.slot_uid[result.slot_status > 0] == 1).all()
    again = model.correct(result, evidence())
    assert torch.equal(
        again.slot_uid[result.slot_status > 0], result.slot_uid[result.slot_status > 0]
    )


def test_restart_restores_all_stream_state_without_module_mutation():
    model = network().eval()
    initial = model.predict(model.initial_state(2), torch.ones(2))
    state = model.correct(initial, evidence())
    saved = io.BytesIO()
    torch.save(state.to_payload(), saved)
    saved.seek(0)
    restored = ObjectState.from_payload(
        torch.load(saved, weights_only=True), model.config
    )
    clone = network().eval()
    clone.load_state_dict(model.state_dict())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    expected = model.correct(
        model.predict(state, torch.full((2,), 0.5)),
        replace(evidence(), time=state.t + 0.5),
    )
    actual = clone.correct(
        clone.predict(restored, torch.full((2,), 0.5)),
        replace(evidence(), time=state.t + 0.5),
    )
    assert_equal(actual, expected)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert not list(model.named_buffers())


def test_rollout_only_predicts_and_keeps_content_fixed():
    model = network()
    state = allocated(model)
    steps = model.rollout(state, torch.full((3, 2), 0.1))
    assert len(steps) == 3
    for result in steps:
        assert torch.equal(result.c, state.c)
        assert torch.equal(result.c_bg, state.c_bg)
    torch.testing.assert_close(steps[-1].t, state.t + 0.3)
    with pytest.raises(ValueError, match="nonnegative"):
        model.predict(state, torch.tensor([-0.1, 0.1]))


def test_gradients_flow_through_correction_canonical_write_and_predict():
    model = network()
    prior = model.predict(allocated(model), torch.full((2,), 0.1))
    result = model.correct(prior, replace(evidence(), time=prior.t))
    predicted = model.predict(result, torch.full((2,), 0.2))
    loss = (
        predicted.d.square().mean()
        + predicted.c.square().mean()
        + predicted.c_bg.square().mean()
    )
    loss.backward()
    for prefix in ("dynamics.", "resampler."):
        gradients = [
            p.grad for name, p in model.named_parameters() if name.startswith(prefix)
        ]
        assert any(
            g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
            for g in gradients
        )


def test_serialization_rejects_hidden_unlisted_state_and_wrong_shapes():
    cfg = config()
    state = ObjectState.initial(cfg, 1, device="cpu")
    payload = state.to_payload()
    payload["secret_kv"] = torch.ones(3)
    with pytest.raises(ValueError, match="fields"):
        ObjectState.from_payload(payload, cfg)
    with pytest.raises(ValueError):
        replace(state, d=state.d[..., :1]).validate(cfg)


def test_per_stream_zero_dt_preserves_every_field_and_positive_dt_has_gradient():
    model = network()
    state = allocated(model)
    result = model.predict(state, torch.tensor([0.0, 0.1]))
    for name, value in state.tensor_dict().items():
        assert torch.equal(result.tensor_dict()[name][0], value[0]), name
    result.d[1].sum().backward()
    assert model.dynamics.forcing_u.weight.grad.abs().sum() > 0


def test_transition_is_slot_permutation_equivariant_for_allocated_objects():
    model = network()
    state = allocated(model)
    permutation = torch.tensor([2, 0, 1])
    names = {"c", "d", "slot_status", "slot_uid", "p_exist", "last_seen"}
    permuted = replace(
        state,
        **{
            name: value[:, permutation]
            for name, value in state.tensor_dict().items()
            if name in names
        },
    )
    expected = model.predict(state, 0.3)
    actual = model.predict(permuted, 0.3)
    torch.testing.assert_close(
        actual.d, expected.d[:, permutation], rtol=1e-5, atol=1e-6
    )


def test_missing_geometry_updates_dynamic_state_but_not_canonical_content():
    model = network()
    state = allocated(model)
    item = replace(
        evidence(time=0.0),
        geometry_valid=torch.zeros(2, 8, dtype=torch.bool),
        ray_origins=torch.full((2, 8, 3), float("nan")),
        ray_directions=torch.full((2, 8, 3), float("nan")),
    )
    result = model.correct(state, item)
    assert torch.equal(result.c, state.c)
    assert torch.equal(result.c_bg, state.c_bg)
    assert not torch.equal(result.d, state.d)
    assert torch.isfinite(result.d).all()


def test_action_conditioning_and_action_shape_are_explicit():
    torch.manual_seed(42)
    cfg = replace(config(), action_dim=2)
    dynamic = SelectiveDynamics(cfg)
    state = replace(
        ObjectState.initial(cfg, 2), slot_status=torch.ones(2, 3, dtype=torch.int64)
    )
    a = dynamic(state, 0.2, torch.zeros(2, 2))
    b = dynamic(state, 0.2, torch.ones(2, 2))
    assert not torch.equal(a.d, b.d)
    with pytest.raises(ValueError, match="action"):
        dynamic(state, 0.1)
    with pytest.raises(ValueError, match="action"):
        dynamic(state, 0.1, torch.zeros(2, 1))


def test_snapshot_storage_is_compact_and_correction_checks_physical_timestamp():
    model = network()
    state = allocated(model)
    assert state.retained_storage_nbytes == state.total_nbytes
    loaded = ObjectState.from_payload(state.to_payload(), model.config)
    assert loaded.retained_storage_nbytes == loaded.total_nbytes
    with pytest.raises(ValueError, match="timestamp"):
        model.correct(state, evidence(time=1.0))


def test_invalid_available_geometry_and_fractional_configuration_are_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        replace(config(), candidates=1.5)
    item = replace(evidence(), ray_directions=torch.zeros(2, 8, 3))
    with pytest.raises(ValueError, match="nonzero"):
        item.validate(config())


def test_birth_proposals_receive_gradient_through_initialized_dynamic_state():
    model = network()
    state = model.predict(model.initial_state(2), torch.ones(2))
    result, details = model.correct(state, evidence(), return_diagnostics=True)
    assert details["births"].any()
    model.predict(result, 0.1).d.square().sum().backward()
    gradient = model.resampler.birth_descriptor.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_background_dominant_tails_do_not_correct_occluded_allocated_objects(
    monkeypatch,
):
    model = network()
    state = model.predict(allocated(model), torch.ones(2))
    item = evidence(count=128)

    def background_ownership(state, evidence, controls, *, proposals=False):
        probabilities = evidence.tokens.new_full((2, 128, 4), 0.001)
        probabilities[..., -1] = 0.997
        return probabilities

    monkeypatch.setattr(model.resampler, "ownership", background_ownership)
    result, details = model.correct(state, item, return_diagnostics=True)
    assert not details["observed"].any()
    for name in ("c", "d", "slot_status", "slot_uid", "last_seen", "p_exist"):
        assert torch.equal(getattr(result, name), getattr(state, name)), name
    assert details["support"].count_nonzero() == 0


def test_candidate_budget_explicitly_enables_soft_all_evidence_warmup():
    model = network()
    state = model.predict(model.initial_state(2), torch.ones(2))
    _, warm = model.correct(
        state, evidence(), candidate_budget=0, return_diagnostics=True
    )
    _, sparse = model.correct(
        state, evidence(), candidate_budget=3, return_diagnostics=True
    )
    assert warm["candidate_indices"].shape == (2, 3, 8)
    assert sparse["candidate_indices"].shape == (2, 3, 3)
    assert torch.equal(warm["candidate_indices"][0, 0], torch.arange(8))
    with pytest.raises(ValueError, match="candidate_budget"):
        model.correct(state, evidence(), candidate_budget=-1)
