"""Batching the row-local GQA tail preserves the chronological reference."""

from copy import deepcopy
import math

import pytest
import torch

from mavt4d.hybrid.attention import TemporalGQA
from mavt4d.hybrid.config import HybridConfig


def attention_case(*, length=4, all_missing=False):
    cfg = HybridConfig.tiny()
    generator = torch.Generator().manual_seed(813)
    observed = torch.tensor([[True, False, True], [True, True, False],
                             [False, True, True], [True, False, True]])[:length]
    if all_missing:
        observed.zero_()
    values = torch.randn(length, 3, cfg.summary_tokens, cfg.width, generator=generator)
    values = values.masked_fill(~observed[..., None, None], float("nan")).requires_grad_()
    shape = (3, cfg.cache_steps, cfg.summary_tokens, cfg.kv_heads, cfg.head_dim)
    key = torch.randn(shape, generator=generator).to(torch.bfloat16).requires_grad_()
    value = torch.randn(shape, generator=generator).to(torch.bfloat16).requires_grad_()
    valid = torch.ones(shape[:3], dtype=torch.bool)
    valid[1, 0, 0] = False
    # A wrapped cache must be gathered in logical rather than physical order.
    initial_ids = torch.tensor([[2, 0, 1]]).expand(3, -1).clone()
    initial_times = initial_ids.double() / 10
    ids = torch.arange(3, 3 + length)[:, None].expand(-1, 3).clone()
    times = ids.double() / 10
    return cfg, (values, key, value, valid, initial_times, initial_ids,
                 observed, times, ids)


def cloned_case(args):
    return tuple(item.detach().clone().requires_grad_(item.requires_grad) for item in args)


def chronological_oracle(layer, inputs, initial_k, initial_v, initial_valid,
                         initial_times, initial_ids, observed, times, step_ids):
    """Frozen per-row output/FFN path from the original reference implementation."""
    cfg = layer.config
    length, batch = inputs.shape[:2]
    clean = torch.where(observed[..., None, None], inputs, torch.zeros_like(inputs))
    indices = observed.flatten().nonzero(as_tuple=False).flatten()
    active = clean.flatten(0, 1).index_select(0, indices)
    normalized = layer.input_norm(active)
    q_active = layer.q_projection(normalized).reshape(-1, cfg.summary_tokens, cfg.q_heads, cfg.head_dim)
    k_active = layer.k_projection(normalized).reshape(-1, cfg.summary_tokens, cfg.kv_heads, cfg.head_dim).to(torch.bfloat16)
    v_active = layer.v_projection(normalized).reshape_as(k_active).to(torch.bfloat16)
    q = inputs.new_zeros(length * batch, cfg.summary_tokens, cfg.q_heads, cfg.head_dim).index_copy(0, indices, q_active)
    shape = (length * batch, cfg.summary_tokens, cfg.kv_heads, cfg.head_dim)
    k = torch.zeros(shape, dtype=torch.bfloat16).index_copy(0, indices, k_active)
    v = torch.zeros_like(k).index_copy(0, indices, v_active)
    q = q.reshape(length, batch, *q.shape[1:])
    k = k.reshape(length, batch, *k.shape[1:])
    v = v.reshape(length, batch, *v.shape[1:])
    if length == 0:
        return inputs.clone(), k, v
    all_k = torch.cat((initial_k, k.transpose(0, 1)), dim=1)
    all_v = torch.cat((initial_v, v.transpose(0, 1)), dim=1)
    all_times = torch.cat((initial_times, times.transpose(0, 1)), dim=1)
    all_ids = torch.cat((initial_ids, step_ids.transpose(0, 1)), dim=1)
    all_valid = torch.cat((initial_valid, observed.T[..., None].expand(-1, -1, cfg.summary_tokens)), dim=1)
    result = []
    repetitions = cfg.q_heads // cfg.kv_heads
    for t in range(length):
        rows = []
        for stream in range(batch):
            if not bool(observed[t, stream]):
                rows.append(inputs.new_zeros(cfg.summary_tokens, cfg.width))
                continue
            current_id = step_ids[t, stream]
            eligible = ((all_ids[stream] >= current_id - cfg.cache_steps + 1)
                        & (all_ids[stream] <= current_id) & (all_ids[stream] >= 0))
            selection = eligible.nonzero(as_tuple=False).flatten()
            selection = selection[torch.argsort(all_ids[stream, selection])]
            valid = all_valid[stream].index_select(0, selection).flatten()
            keys = all_k[stream].index_select(0, selection).flatten(0, 1)[valid].float()
            values = all_v[stream].index_select(0, selection).flatten(0, 1)[valid].float()
            key_times = all_times[stream].index_select(0, selection)[:, None].expand(-1, cfg.summary_tokens).flatten()[valid]
            keys = keys.repeat_interleave(repetitions, dim=1).permute(1, 0, 2)
            values = values.repeat_interleave(repetitions, dim=1).permute(1, 0, 2)
            queries = q[t, stream].permute(1, 0, 2)
            logits = queries @ keys.transpose(-1, -2) / math.sqrt(cfg.head_dim)
            elapsed = (times[t, stream] - key_times).clamp_min(0)
            bias = layer.time_bias(torch.log1p(elapsed).float()[:, None]).T
            aggregate = (torch.softmax(logits + bias[:, None, :], dim=-1) @ values)
            aggregate = aggregate.permute(1, 0, 2).reshape(cfg.summary_tokens, cfg.width)
            output = inputs[t, stream] + layer.out_projection(aggregate)
            rows.append(output + layer.ffn(layer.ffn_norm(output)))
        result.append(torch.stack(rows))
    return torch.stack(result), k, v


def assert_matching_gradients(first, second):
    assert (first is None) == (second is None)
    if first is not None:
        assert torch.isfinite(second).all()
        # Same tolerances as the existing GPU value/gradient correctness gate.
        torch.testing.assert_close(second, first, rtol=3e-2, atol=2e-5)


def test_attention_tail_uses_one_batch_of_observed_rows():
    cfg, args = attention_case()
    layer = TemporalGQA(cfg)
    calls = {name: [] for name in ("out_projection", "ffn_norm", "ffn.0", "ffn.2")}
    handles = []
    for name in calls:
        def record(_module, inputs, *, name=name):
            calls[name].append(tuple(inputs[0].shape))
        handles.append(layer.get_submodule(name).register_forward_pre_hook(record))
    try:
        layer.forward_sequence(*args)
    finally:
        for handle in handles:
            handle.remove()
    count = int(args[6].sum())
    assert calls == {
        "out_projection": [(count, cfg.summary_tokens, cfg.width)],
        "ffn_norm": [(count, cfg.summary_tokens, cfg.width)],
        "ffn.0": [(count, cfg.summary_tokens, cfg.width)],
        "ffn.2": [(count, cfg.summary_tokens, 4 * cfg.width)],
    }


def test_attention_batch_matches_chronological_values_and_gradients():
    cfg, expected_args = attention_case()
    actual_args = cloned_case(expected_args)
    torch.manual_seed(814)
    reference = TemporalGQA(cfg)
    batched = deepcopy(reference)
    expected = chronological_oracle(reference, *expected_args)
    actual = batched.forward_sequence(*actual_args)
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-5, atol=3e-6)
    for first, second in zip(expected[1:], actual[1:]):
        torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert torch.count_nonzero(actual[0][~actual_args[6]]) == 0
    for result in (expected, actual):
        sum(value.float().square().mean() for value in result).backward()
    for (name, first), (_, second) in zip(reference.named_parameters(), batched.named_parameters()):
        assert_matching_gradients(first.grad, second.grad)
    for first, second in zip(expected_args[:3], actual_args[:3]):
        assert_matching_gradients(first.grad, second.grad)
    hidden = ~actual_args[6][..., None, None].expand_as(actual_args[0])
    assert torch.count_nonzero(actual_args[0].grad[hidden]) == 0


@pytest.mark.parametrize("length", [0, 4])
def test_all_missing_and_empty_attention_preserve_bypass_and_unused_gradients(length):
    cfg, expected_args = attention_case(length=length, all_missing=True)
    actual_args = cloned_case(expected_args)
    reference = TemporalGQA(cfg)
    batched = deepcopy(reference)
    calls = []
    handle = batched.out_projection.register_forward_pre_hook(lambda *_args: calls.append(True))
    try:
        actual = batched.forward_sequence(*actual_args)
    finally:
        handle.remove()
    expected = chronological_oracle(reference, *expected_args)
    assert not calls
    for first, second in zip(expected, actual):
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        assert first.requires_grad == second.requires_grad
    for result in (expected, actual):
        (result[1].float().sum() + result[2].float().sum()).backward()
    for (_, first), (_, second) in zip(reference.named_parameters(), batched.named_parameters()):
        assert_matching_gradients(first.grad, second.grad)
    for first, second in zip(expected_args[:3], actual_args[:3]):
        assert_matching_gradients(first.grad, second.grad)
