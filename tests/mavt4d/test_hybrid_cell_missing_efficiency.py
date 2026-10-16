"""Missing dynamics can share equal time gaps without changing gradients."""
from copy import deepcopy

import pytest
import torch

from mavt4d.hybrid.cells import HybridCoupledCell
from mavt4d.hybrid.config import HybridConfig


def dense_coefficients(cell, evidence, observed, delta_time):
    """Pre-optimization reference: form every missing transition then overwrite."""
    cfg, batch = cell.config, len(evidence)
    clean = torch.where(observed[:, None, None], evidence, torch.zeros_like(evidence))
    identity = torch.eye(2, device=evidence.device, dtype=torch.float32)
    omega = cell.missing_rotation
    zero = torch.zeros_like(omega)
    skew = torch.stack((zero, -omega, omega, zero), dim=-1).reshape(cfg.state_tokens, cfg.state_dim, 2, 2)
    generator = skew - cell.missing_damping @ cell.missing_damping.transpose(-1, -2)
    missing_a = cell._normalize(torch.matrix_exp(delta_time.float()[:, None, None, None, None] * generator))
    missing_b = evidence.new_zeros(batch, cfg.state_tokens, cfg.state_dim, 2)
    indices = observed.nonzero(as_tuple=False).flatten()
    if len(indices) == 0:
        return missing_a, missing_b
    active = cell.evidence_norm(clean.index_select(0, indices))
    queries = cell.write_queries[None].expand(len(indices), -1, -1)
    context, _ = cell.write_attention(queries, active, active, need_weights=False)
    time_context = cell.time_projection(torch.log1p(delta_time[indices]).float()[:, None])
    raw = cell.write_head(context + time_context[:, None]).reshape(len(indices), cfg.state_tokens, cfg.state_dim, 7)
    matrix = identity + .1 * torch.tanh(raw[..., :4].reshape(*raw.shape[:-1], 2, 2))
    alpha = torch.sigmoid(raw[..., 6:7])
    observed_a = (1 - alpha[..., None]) * identity + alpha[..., None] * cell._normalize(matrix)
    observed_b = alpha * torch.tanh(raw[..., 4:6])
    return missing_a.index_copy(0, indices, observed_a), missing_b.index_copy(0, indices, observed_b)


def make_case(observed, deltas):
    torch.manual_seed(211)
    cfg = HybridConfig.tiny()
    cell = HybridCoupledCell(cfg)
    # Break repeated singular values for meaningful parameter-gradient checks.
    with torch.no_grad():
        cell.missing_damping.add_(torch.randn_like(cell.missing_damping) * .01)
    evidence = torch.randn(len(observed), cfg.summary_tokens, cfg.width)
    return cell, evidence, torch.tensor(observed), torch.tensor(deltas, dtype=torch.float64)


def test_observed_only_skips_matrix_exp_but_preserves_zero_grad_and_adamw(monkeypatch):
    cell, evidence, observed, deltas = make_case([True] * 4, [.1, .2, .2, .3])
    dense = deepcopy(cell)
    expected = dense_coefficients(dense, evidence, observed, deltas)
    calls = []
    original = torch.matrix_exp
    def counted(value):
        calls.append(value.shape)
        return original(value)
    monkeypatch.setattr(torch, "matrix_exp", counted)
    actual = cell.coefficients(evidence, observed, deltas)
    assert calls == []
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    sum(x.square().mean() for x in actual).backward()
    sum(x.square().mean() for x in expected).backward()
    for name in ("missing_rotation", "missing_damping"):
        parameter = getattr(cell, name)
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        assert torch.equal(parameter.grad, getattr(dense, name).grad)
    first_optimizer = torch.optim.AdamW(cell.parameters(), lr=.001, weight_decay=.1)
    second_optimizer = torch.optim.AdamW(dense.parameters(), lr=.001, weight_decay=.1)
    first_optimizer.step()
    second_optimizer.step()
    for first, second in zip(cell.parameters(), dense.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_missing_exp_uses_only_unique_missing_gaps(monkeypatch):
    cell, evidence, observed, deltas = make_case([True, False, False, True, False], [.1, .2, .2, .3, .2])
    calls = []
    original = torch.matrix_exp
    def counted(value):
        calls.append(value.shape[0])
        return original(value)
    monkeypatch.setattr(torch, "matrix_exp", counted)
    cell.coefficients(evidence, observed, deltas)
    assert calls == [1]


@pytest.mark.parametrize("observed,deltas", [
    ([True, False, False, True, False], [.1, .2, .2, .3, .2]),
    ([False, False, False, False], [.0, .125, .125, .25]),
])
def test_unique_missing_coefficients_and_parameter_gradients_match_dense(observed, deltas):
    cell, evidence, available, gaps = make_case(observed, deltas)
    dense = deepcopy(cell)
    actual = cell.coefficients(evidence, available, gaps)
    expected = dense_coefficients(dense, evidence, available, gaps)
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second, rtol=2e-6, atol=2e-7)
    weights = [torch.randn_like(x) for x in actual]
    sum((value * weight).sum() for value, weight in zip(actual, weights)).backward()
    sum((value * weight).sum() for value, weight in zip(expected, weights)).backward()
    for (name, first), (_, second) in zip(cell.named_parameters(), dense.named_parameters()):
        assert (first.grad is None) == (second.grad is None), name
        if first.grad is not None:
            assert torch.isfinite(first.grad).all(), name
            torch.testing.assert_close(first.grad, second.grad, rtol=3e-4, atol=2e-6)


def test_explicitly_differentiable_timestamps_retain_dense_gap_gradients():
    cell, evidence, observed, deltas = make_case([True, False, False], [.1, .2, .2])
    dense = deepcopy(cell)
    actual_gaps = deltas.clone().requires_grad_()
    expected_gaps = deltas.clone().requires_grad_()
    actual = cell.coefficients(evidence, observed, actual_gaps)
    expected = dense_coefficients(dense, evidence, observed, expected_gaps)
    weights = [torch.randn_like(value) for value in actual]
    sum((value * weight).sum() for value, weight in zip(actual, weights)).backward()
    sum((value * weight).sum() for value, weight in zip(expected, weights)).backward()
    torch.testing.assert_close(actual_gaps.grad, expected_gaps.grad, rtol=3e-4, atol=2e-6)
