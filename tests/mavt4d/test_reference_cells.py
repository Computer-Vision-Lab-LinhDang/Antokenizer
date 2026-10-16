"""Executable recurrence contract, including chronological affine scan."""

import copy
import io

import pytest
import torch

from mavt4d.contracts import CDState, CDStateSpec
from mavt4d.reference.cells import (
    CoupledAffineCell,
    GRUCDCell,
    TwoScanCell,
    affine_prefix_scan,
)


CELL_TYPES = (CoupledAffineCell, TwoScanCell, GRUCDCell)


def _case(cell_type=CoupledAffineCell, dtype=torch.float64, requires_grad=False):
    torch.manual_seed(127)
    spec = CDStateSpec(c_tokens=2, d_tokens=2, token_dim=3)
    cell = cell_type(spec, evidence_dim=4).to(dtype=dtype)
    state = CDState(
        torch.randn(2, 2, 3, dtype=dtype).requires_grad_(requires_grad),
        torch.randn(2, 2, 3, dtype=dtype).requires_grad_(requires_grad),
        torch.tensor([0.0, 0.25], dtype=torch.float64), spec=spec,
    )
    evidence = torch.randn(7, 2, 4, 4, dtype=dtype)
    observed = torch.tensor([
        [True, True], [False, True], [True, False], [False, False],
        [True, True], [False, True], [True, False],
    ])
    times = state.last_time + torch.arange(1, 8, dtype=torch.float64)[:, None] * 0.3
    return cell, state, evidence, observed, times


def _assert_state_equal(actual, expected, *, atol=1e-10, rtol=1e-9):
    torch.testing.assert_close(actual.c, expected.c, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.d, expected.d, atol=atol, rtol=rtol)
    assert torch.equal(actual.last_time, expected.last_time)
    assert actual.spec == expected.spec


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_step_sequential_and_chunked_continuation_share_the_same_schedule(cell_type, dtype):
    cell, initial, evidence, observed, times = _case(cell_type, dtype)
    sequence = cell.scan(initial, evidence, observed, times)
    assert isinstance(sequence, tuple) and len(sequence) == 7
    state = initial
    for index, expected in enumerate(sequence):
        state = cell.step(state, evidence[index], observed[index], times[index])
        _assert_state_equal(state, expected)
        assert state.retained_storage_nbytes == state.total_nbytes
    prefix = cell.scan(initial, evidence[:3], observed[:3], times[:3])
    suffix = cell.scan(prefix[-1].detach(), evidence[3:], observed[3:], times[3:])
    for actual, expected in zip(suffix, sequence[3:], strict=True):
        _assert_state_equal(actual, expected)


@pytest.mark.parametrize("cell_type", [CoupledAffineCell, TwoScanCell])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_associative_scan_matches_online_and_chunked_scan(cell_type, dtype):
    cell, initial, evidence, observed, times = _case(cell_type, dtype)
    reference = cell.scan(initial, evidence, observed, times, method="sequential")
    scanned = cell.scan(initial, evidence, observed, times, method="associative")
    atol, rtol = (2e-6, 2e-5) if dtype == torch.float32 else (1e-10, 1e-9)
    for actual, expected in zip(scanned, reference, strict=True):
        _assert_state_equal(actual, expected, atol=atol, rtol=rtol)
        assert actual.retained_storage_nbytes == actual.total_nbytes
    split = cell.scan(initial, evidence[:3], observed[:3], times[:3], method="associative")
    rest = cell.scan(split[-1], evidence[3:], observed[3:], times[3:], method="associative")
    _assert_state_equal(rest[-1], reference[-1], atol=atol, rtol=rtol)


def test_affine_prefix_scan_composes_noncommuting_blocks_in_chronological_order():
    a = torch.tensor([
        [[1.0, 2.0], [0.0, 1.0]],
        [[2.0, 0.0], [3.0, 1.0]],
        [[0.5, 1.0], [1.0, 0.0]],
    ], dtype=torch.float64)
    b = torch.tensor([[1.0, -1.0], [2.0, 3.0], [4.0, -2.0]], dtype=torch.float64)
    prefix_a, prefix_b = affine_prefix_scan(a, b)
    expected_a = torch.eye(2, dtype=torch.float64)
    expected_b = torch.zeros(2, dtype=torch.float64)
    for index in range(3):
        expected_b = a[index] @ expected_b + b[index]
        expected_a = a[index] @ expected_a
        torch.testing.assert_close(prefix_a[index], expected_a)
        torch.testing.assert_close(prefix_b[index], expected_b)
    assert not torch.allclose(a[0] @ a[1], prefix_a[1])


@pytest.mark.parametrize("cell_type", [CoupledAffineCell, TwoScanCell])
def test_scan_gradients_match_step_for_every_parameter_and_initial_state(cell_type):
    cell, initial, evidence, observed, times = _case(cell_type, requires_grad=True)
    other = copy.deepcopy(cell)
    second_initial = CDState(
        initial.c.detach().clone().requires_grad_(),
        initial.d.detach().clone().requires_grad_(), initial.last_time.clone(), spec=initial.spec,
    )
    first_evidence = evidence.clone().requires_grad_()
    second_evidence = evidence.clone().requires_grad_()
    reference = cell.scan(initial, first_evidence, observed, times)
    scanned = other.scan(second_initial, second_evidence, observed, times, method="associative")
    sum((index + 1) * (s.c.square().sum() + 0.7 * s.d.square().sum())
        for index, s in enumerate(reference)).backward()
    sum((index + 1) * (s.c.square().sum() + 0.7 * s.d.square().sum())
        for index, s in enumerate(scanned)).backward()
    pairs = [(initial.c, second_initial.c), (initial.d, second_initial.d),
             (first_evidence, second_evidence)]
    pairs.extend(zip(cell.parameters(), other.parameters(), strict=True))
    for first, second in pairs:
        assert first.grad is not None and second.grad is not None
        assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
        torch.testing.assert_close(first.grad, second.grad, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_missing_zero_time_is_exact_identity_and_ignores_even_nan_evidence(cell_type):
    cell, state, evidence, _, _ = _case(cell_type)
    hidden = torch.full_like(evidence[0], float("nan"))
    missing = torch.zeros(2, dtype=torch.bool)
    unchanged = cell.step(state, hidden, missing, state.last_time)
    assert torch.equal(unchanged.c, state.c)
    assert torch.equal(unchanged.d, state.d)
    advanced = cell.step(state, hidden, missing, state.last_time + 0.5)
    finite_input = cell.step(state, torch.zeros_like(hidden), missing, state.last_time + 0.5)
    _assert_state_equal(advanced, finite_input)
    assert not (torch.equal(advanced.c, state.c) and torch.equal(advanced.d, state.d))
    if cell_type is TwoScanCell:
        assert torch.equal(advanced.c, state.c)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_batch_missing_masks_do_not_suppress_other_streams(cell_type):
    cell, state, evidence, observed, times = _case(cell_type)
    observed = torch.tensor([False, True])
    altered = evidence[0].clone()
    altered[0] = float("nan")
    first = cell.step(state, evidence[0], observed, times[0])
    second = cell.step(state, altered, observed, times[0])
    _assert_state_equal(first, second)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_restore_only_cd_payload_into_fresh_cell_reproduces_continuation(cell_type):
    cell, initial, evidence, observed, times = _case(cell_type)
    reference = cell.scan(initial, evidence, observed, times)
    stream = io.BytesIO()
    torch.save({"weights": cell.state_dict(), "state": reference[2].to_payload()}, stream)
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    fresh = cell_type(initial.spec, evidence_dim=4).double()
    fresh.load_state_dict(payload["weights"])
    resumed = fresh.scan(CDState.from_payload(payload["state"]),
                         evidence[3:], observed[3:], times[3:])
    for actual, expected in zip(resumed, reference[3:], strict=True):
        _assert_state_equal(actual, expected)
    # Parameters and declared buffers cannot change with the video.
    for name, value in cell.state_dict().items():
        assert torch.equal(value, payload["weights"][name])


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_changing_future_evidence_never_changes_prefix_states(cell_type):
    cell, state, evidence, observed, times = _case(cell_type)
    changed = evidence.clone()
    changed[3:] = changed[3:] * 19 - 4
    method = "sequential" if cell_type is GRUCDCell else "associative"
    first = cell.scan(state, evidence, observed, times, method=method)
    second = cell.scan(state, changed, observed, times, method=method)
    for actual, expected in zip(first[:3], second[:3], strict=True):
        _assert_state_equal(actual, expected)


@pytest.mark.parametrize("is_observed", [False, True])
def test_feedback_ablation_changes_only_cd_entry_with_common_normalizer(is_observed):
    full, state, evidence, _, _ = _case()
    ablated = CoupledAffineCell(state.spec, evidence_dim=4, feedback=False).double()
    ablated.load_state_dict(full.state_dict())
    mask = torch.full((2,), is_observed, dtype=torch.bool)
    dt = torch.tensor([0.2, 2.0], dtype=torch.float64)
    full_a, full_b = full.coefficients(evidence[0], mask, dt)
    cut_a, cut_b = ablated.coefficients(evidence[0], mask, dt)
    assert torch.equal(full_b, cut_b)
    for row, col in [(0, 0), (1, 0), (1, 1)]:
        assert torch.equal(full_a[..., row, col], cut_a[..., row, col])
    assert torch.count_nonzero(cut_a[..., 0, 1]) == 0
    assert torch.count_nonzero(full_a[..., 0, 1]) > 0
    for operator in (full_a, cut_a):
        assert (torch.linalg.matrix_norm(operator, ord=2) <= 1 + 1e-10).all()


def test_feedback_changes_c_when_only_previous_d_changes():
    cell, initial, evidence, observed, times = _case()
    changed = CDState(initial.c, initial.d + 2, initial.last_time, spec=initial.spec)
    full_a = cell.step(initial, evidence[0], observed[0], times[0])
    full_b = cell.step(changed, evidence[0], observed[0], times[0])
    assert not torch.allclose(full_a.c, full_b.c)
    no_feedback = CoupledAffineCell(initial.spec, 4, feedback=False).double()
    no_feedback.load_state_dict(cell.state_dict())
    cut_a = no_feedback.step(initial, evidence[0], observed[0], times[0])
    cut_b = no_feedback.step(changed, evidence[0], observed[0], times[0])
    assert torch.equal(cut_a.c, cut_b.c)


def test_coupled_step_is_exactly_the_declared_block_affine_map():
    cell, state, evidence, observed, times = _case()
    a, b = cell.coefficients(evidence[0], observed[0], times[0] - state.last_time)
    assert a.shape == (2, 2, 3, 2, 2) and b.shape == (2, 2, 3, 2)
    expected = (a @ torch.stack((state.c, state.d), dim=-1).unsqueeze(-1)).squeeze(-1) + b
    result = cell.step(state, evidence[0], observed[0], times[0])
    torch.testing.assert_close(result.c, expected[..., 0], atol=0, rtol=0)
    torch.testing.assert_close(result.d, expected[..., 1], atol=0, rtol=0)
    identity = torch.eye(2, dtype=a.dtype)
    assert torch.linalg.matrix_norm(a - identity, ord=2).max() < 0.05


def test_two_scan_c_has_no_dependency_on_previous_d():
    cell, state, evidence, observed, times = _case(TwoScanCell)
    altered = CDState(state.c, state.d * 10 - 3, state.last_time, spec=state.spec)
    first = cell.step(state, evidence[0], observed[0], times[0])
    second = cell.step(altered, evidence[0], observed[0], times[0])
    assert torch.equal(first.c, second.c)
    assert not torch.equal(first.d, second.d)


@pytest.mark.parametrize("feedback", [True, False])
def test_full_block_stability_and_long_missing_rollout(feedback):
    cell, state, evidence, _, _ = _case()
    cell.feedback = feedback
    missing = torch.zeros(2, dtype=torch.bool)
    norm = torch.stack((state.c, state.d), dim=-1).norm(dim=-1)
    for step in range(1, 65):
        updated = cell.step(state, evidence[0], missing, state.last_time + 1.5)
        new_norm = torch.stack((updated.c, updated.d), dim=-1).norm(dim=-1)
        assert (new_norm <= norm + 1e-10).all()
        state, norm = updated, new_norm
    operators, bias = cell.coefficients(evidence[0] * 1e6, ~missing,
                                       torch.tensor([0.0, 1e4], dtype=torch.float64))
    assert (torch.linalg.matrix_norm(operators, ord=2) <= 1 + 1e-10).all()
    assert torch.isfinite(bias).all() and bias.abs().max() <= 0.1


def test_gru_gradients_reach_initial_c_d_and_every_parameter():
    cell, initial, evidence, observed, times = _case(GRUCDCell, requires_grad=True)
    last = cell.scan(initial, evidence, observed, times)[-1]
    (last.c.square().sum() + last.d.square().sum()).backward()
    for value in [initial.c, initial.d, *cell.parameters()]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize("cell_type", CELL_TYPES)
@pytest.mark.parametrize("invalid", ["evidence_shape", "evidence_dtype", "mask_dtype",
                                     "time_dtype", "past", "observed_nan", "spec"])
def test_step_rejects_invalid_input(cell_type, invalid):
    cell, state, evidence, observed, times = _case(cell_type)
    e, m, t = evidence[0], observed[0], times[0]
    if invalid == "evidence_shape":
        e = e[:, :-1]
    elif invalid == "evidence_dtype":
        e = e.float()
    elif invalid == "mask_dtype":
        m = m.float()
    elif invalid == "time_dtype":
        t = t.float()
    elif invalid == "past":
        t = state.last_time - 0.1
    elif invalid == "observed_nan":
        e = torch.full_like(e, float("nan"))
    elif invalid == "spec":
        state = CDState(state.c, state.d, state.last_time,
                        spec=CDStateSpec(c_tokens=2, d_tokens=2, token_dim=3, reference_views=2))
    with pytest.raises((ValueError, TypeError)):
        cell.step(state, e, m, t)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_scan_rejects_reversed_time_and_unknown_method(cell_type):
    cell, state, evidence, observed, times = _case(cell_type)
    times[4] = times[2]
    with pytest.raises(ValueError):
        cell.scan(state, evidence, observed, times)
    with pytest.raises(ValueError):
        cell.scan(state, evidence, observed, times, method="fast")


def test_gru_does_not_claim_associative_scan():
    cell, state, evidence, observed, times = _case(GRUCDCell)
    with pytest.raises(ValueError, match="associative"):
        cell.scan(state, evidence, observed, times, method="associative")


def test_coupled_requires_equal_c_and_d_token_counts():
    with pytest.raises(ValueError, match="equal"):
        CoupledAffineCell(CDStateSpec(c_tokens=2, d_tokens=3, token_dim=4), 8)


@pytest.mark.parametrize("cell_type", CELL_TYPES)
def test_empty_scan_returns_no_states_and_half_precision_is_explicitly_unsupported(cell_type):
    cell, state, evidence, observed, times = _case(cell_type)
    assert cell.scan(state, evidence[:0], observed[:0], times[:0]) == ()
    cell = cell.half()
    half_state = CDState(state.c.half(), state.d.half(), state.last_time, spec=state.spec)
    with pytest.raises(ValueError, match="float32.*float64"):
        cell.step(half_state, evidence[0].half(), observed[0], times[0])


@pytest.mark.parametrize("cell_type", [TwoScanCell, GRUCDCell])
def test_baselines_accept_unequal_slot_counts_without_expanding_persistent_memory(cell_type):
    spec = CDStateSpec(c_tokens=2, d_tokens=3, token_dim=2)
    cell = cell_type(spec, 4)
    state = CDState(torch.zeros(1, 2, 2), torch.zeros(1, 3, 2),
                    torch.zeros(1, dtype=torch.float64), spec=spec)
    updated = cell.step(state, torch.randn(1, 5, 4), torch.ones(1, dtype=torch.bool),
                        torch.ones(1, dtype=torch.float64))
    assert updated.retained_storage_nbytes == state.total_nbytes
    assert updated.c.shape == state.c.shape and updated.d.shape == state.d.shape
