"""Temporal GQA windows, step causality and common BF16 value path."""
from dataclasses import replace

import pytest
import torch

from mavt4d.hybrid.attention import TemporalGQA
from mavt4d.hybrid.cells import HybridCoupledCell
from mavt4d.hybrid.config import HybridConfig


def inputs(length=7):
    torch.manual_seed(2701)
    cfg = replace(HybridConfig.tiny(), cache_steps=3)
    x = torch.randn(length, 1, cfg.summary_tokens, cfg.width)
    observed = torch.ones(length, 1, dtype=torch.bool)
    times = torch.arange(1, length + 1, dtype=torch.float64)[:, None]
    ids = torch.arange(length)[:, None]
    shape = (1, cfg.cache_steps, cfg.summary_tokens, cfg.kv_heads, cfg.head_dim)
    k = torch.zeros(shape, dtype=torch.bfloat16)
    valid = torch.zeros(shape[:3], dtype=torch.bool)
    return cfg, x, k, valid, observed, times, ids


def run(layer, data, x=None, observed=None):
    cfg, values, k, valid, obs, times, ids = data
    return layer.forward_sequence(values if x is None else x, k, k.clone(), valid,
                                  torch.zeros(1, cfg.cache_steps, dtype=torch.float64),
                                  torch.full((1, cfg.cache_steps), -1, dtype=torch.int64),
                                  obs if observed is None else observed, times, ids)


def test_attention_quantizes_current_kv_before_reading_and_never_reads_future():
    data = inputs()
    layer = TemporalGQA(data[0]).eval()
    output, k, v = run(layer, data)
    assert k.dtype == v.dtype == torch.bfloat16
    perturbed = data[1].clone()
    perturbed[3:] += 10
    other, _, _ = run(layer, data, x=perturbed)
    torch.testing.assert_close(output[:3], other[:3], rtol=0, atol=0)
    # Current and stored paths share an explicit BF16 roundtrip.
    for t in range(len(output)):
        norm = layer.input_norm(data[1][t])
        raw = layer.k_projection(norm).reshape_as(k[t])
        torch.testing.assert_close(k[t], raw.to(torch.bfloat16), rtol=0, atol=0)


def test_current_attention_result_uses_rounded_values_not_only_rounded_storage():
    data = inputs(length=1)
    cfg, x, *_ = data
    layer = TemporalGQA(cfg).eval()
    output, k, v = run(layer, data)
    normalized = layer.input_norm(x[0, 0])
    q = layer.q_projection(normalized).reshape(cfg.summary_tokens, cfg.q_heads, cfg.head_dim).permute(1, 0, 2)

    def manual(keys, values):
        keys = keys.repeat_interleave(cfg.q_heads // cfg.kv_heads, dim=1).permute(1, 0, 2)
        values = values.repeat_interleave(cfg.q_heads // cfg.kv_heads, dim=1).permute(1, 0, 2)
        weights = torch.softmax(q @ keys.transpose(-1, -2) / cfg.head_dim**0.5, dim=-1)
        aggregate = (weights @ values).permute(1, 0, 2).reshape(cfg.summary_tokens, cfg.width)
        residual = x[0, 0] + layer.out_projection(aggregate)
        return residual + layer.ffn(layer.ffn_norm(residual))

    rounded = manual(k[0, 0].float(), v[0, 0].float())
    raw_k = layer.k_projection(normalized).reshape(cfg.summary_tokens, cfg.kv_heads, cfg.head_dim)
    raw_v = layer.v_projection(normalized).reshape_as(raw_k)
    unrounded = manual(raw_k, raw_v)
    torch.testing.assert_close(output[0, 0], rounded, rtol=2e-6, atol=2e-7)
    assert (rounded - unrounded).abs().max() > 1e-5


def test_window_includes_current_and_missing_steps_do_not_pull_older_keys():
    data = inputs()
    layer = TemporalGQA(data[0]).eval()
    observed = data[4].clone()
    observed[1:3] = False
    expected, k, v = run(layer, data, observed=observed)
    changed = data[1].clone()
    changed[0] += 1000
    actual, _, _ = run(layer, data, x=changed, observed=observed)
    torch.testing.assert_close(expected[3:], actual[3:], rtol=0, atol=0)
    assert torch.count_nonzero(k[1:3]) == torch.count_nonzero(v[1:3]) == 0
    assert torch.count_nonzero(expected[1:3]) == 0


def test_summary_slots_within_current_step_are_not_autoregressive():
    data = inputs(length=1)
    layer = TemporalGQA(data[0]).eval()
    original, _, _ = run(layer, data)
    changed = data[1].clone()
    changed[0, 0, -1] += torch.arange(data[0].width)
    actual, _, _ = run(layer, data, x=changed)
    assert not torch.equal(original[0, 0, 0], actual[0, 0, 0])


def test_coupled_operators_norm_feedback_and_scan_gradient_parity():
    cfg, x, _, _, observed, times, _ = inputs(length=5)
    full = HybridCoupledCell(cfg)
    cut = HybridCoupledCell(cfg, feedback=False)
    cut.load_state_dict(full.state_dict())
    dt = torch.ones(5, dtype=torch.float64)
    obs = observed.flatten()
    evidence = x[:, 0]
    a, b = full.coefficients(evidence, obs, dt)
    off, off_b = cut.coefficients(evidence, obs, dt)
    assert torch.linalg.matrix_norm(a, ord=2).max() <= 1 + 2e-6
    assert torch.linalg.matrix_norm(off, ord=2).max() <= 1 + 2e-6
    mask = a.new_tensor([[1, 0], [1, 1]])
    torch.testing.assert_close(off, a * mask, rtol=0, atol=0)
    torch.testing.assert_close(b, off_b, rtol=0, atol=0)
    c0 = torch.randn(1, cfg.state_tokens, cfg.state_dim, requires_grad=True)
    d0 = torch.randn_like(c0, requires_grad=True)
    outputs = []
    gradients = []
    for method in ('sequential', 'associative'):
        c, d = full.scan(c0, d0, x, observed, torch.ones_like(times), method=method)
        outputs.append((c, d))
        gradients.append(torch.autograd.grad(c.square().sum() + d.square().sum(),
                                             (c0, d0), retain_graph=True))
    for actual, expected in zip(outputs[1] + gradients[1], outputs[0] + gradients[0]):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_missing_zero_time_operator_is_identity_and_hidden_evidence_is_ignored():
    cfg, x, *_ = inputs(length=2)
    cell = HybridCoupledCell(cfg)
    a, b = cell.coefficients(torch.full_like(x[:, 0], float('nan')),
                             torch.zeros(2, dtype=torch.bool),
                             torch.tensor([0., 1.], dtype=torch.float64))
    torch.testing.assert_close(a[0], torch.eye(2).expand_as(a[0]), rtol=0, atol=0)
    assert torch.count_nonzero(b) == 0
    assert torch.isfinite(a).all()
    assert torch.linalg.matrix_norm(a, ord=2).max() <= 1 + 2e-6
