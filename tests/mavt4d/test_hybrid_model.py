"""Layerwise hybrid scans preserve causal and complete streaming state."""
from dataclasses import fields, replace

import pytest
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.state import HybridCDKVState, weights_fingerprint


def setup(length=8, batch=2):
    torch.manual_seed(2801)
    cfg = replace(HybridConfig.tiny(), cache_steps=3)
    model = HybridCDModel(cfg).eval()
    state = model.initial_state(batch_size=batch)
    state = replace(state, c=torch.randn_like(state.c), d=torch.randn_like(state.d))
    evidence = torch.randn(length, batch, cfg.summary_tokens, cfg.width)
    observed = torch.ones(length, batch, dtype=torch.bool)
    times = torch.arange(1, length + 1, dtype=torch.float64)[:, None].expand(-1, batch).clone()
    if length > 3:
        observed[2, 0] = False
        times[2, 0] = times[1, 0]  # per-stream no-op
        observed[3, -1] = False  # accepted missing step
    return model, state, evidence, observed, times


def same_state(actual, expected):
    for name in ('c', 'd', 'k', 'v', 'last_time', 'step_count', 'write_index',
                 'steps_filled', 'slot_times', 'slot_step_ids', 'token_valid'):
        a, b = getattr(actual, name), getattr(expected, name)
        if name in ('c', 'd'):
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        elif name in ('k', 'v'):
            # Storage is exactly BF16 in both paths. Roundoff in lower-layer
            # scan may only affect values at a quantization boundary.
            torch.testing.assert_close(a.float(), b.float(), rtol=0.008, atol=1e-4)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('method', ['sequential', 'associative'])
def test_layerwise_sequence_chunk_and_online_parity_with_wrap_and_ragged_noops(method):
    network, initial, e, observed, times = setup()
    expected = network.forward_encoded(initial, e, observed, times, method='sequential')
    actual = network.forward_encoded(initial, e, observed, times, method=method)
    first = network.forward_encoded(initial, e[:4], observed[:4], times[:4], method=method)
    chunks = first + network.forward_encoded(first[-1], e[4:], observed[4:], times[4:], method=method)
    current = initial
    online = []
    for index in range(len(e)):
        current = network.forward_encoded(current, e[index:index+1], observed[index:index+1],
                                          times[index:index+1], method=method)[0]
        online.append(current)
    for reference, scanned, chunked, stepped in zip(expected, actual, chunks, online):
        for candidate in (scanned, chunked, stepped):
            same_state(candidate, reference)
            candidate.validate()
            assert candidate.retained_storage_nbytes == candidate.total_nbytes


def test_future_evidence_masks_and_timestamps_cannot_change_prefix():
    network, state, e, observed, times = setup()
    expected = network.forward_encoded(state, e, observed, times, method='associative')
    e[4:] *= 1000
    observed[4:] = False
    times[4:] += 20
    actual = network.forward_encoded(state, e, observed, times, method='associative')
    for a, b in zip(actual[:4], expected[:4]):
        same_state(a, b)


def test_missing_noop_is_exact_and_cache_expires_by_accepted_steps():
    network, state, e, observed, times = setup(length=1, batch=1)
    state = network.forward_encoded(state, e, observed, times)[-1]
    initial = state.clone()
    noop = network.rollout(state, state.last_time[None], method='associative')[0]
    for field in fields(initial):
        if isinstance(getattr(initial, field.name), torch.Tensor):
            assert torch.equal(getattr(initial, field.name), getattr(noop, field.name)), field.name
    missing_times = state.last_time[None] + torch.arange(1, 4, dtype=torch.float64)[:, None]
    missing = network.rollout(state, missing_times)[-1]
    assert not missing.token_valid.any()
    assert torch.count_nonzero(missing.k) == torch.count_nonzero(missing.v) == 0
    assert missing.step_count.item() == state.step_count.item() + 3


def test_module_has_no_mutating_stream_buffers_and_stage1_is_not_changed_by_mixer():
    network, state, e, observed, times = setup(length=3, batch=1)
    before = {k: v.clone() for k, v in network.state_dict().items()}
    network.forward_encoded(state, e, observed, times, method='associative')
    assert network.state_dict().keys() == before.keys()
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert list(network.named_buffers()) == []


def test_gradient_parity_through_layerwise_scan_mixer_and_initial_cd():
    result = []
    for mode in ('sequential', 'associative', 'chunk', 'step'):
        network, initial, e, observed, times = setup(length=4, batch=1)
        initial = replace(initial, c=initial.c.requires_grad_(), d=initial.d.requires_grad_())
        e.requires_grad_()
        if mode == 'chunk':
            first = network.forward_encoded(initial, e[:2], observed[:2], times[:2])
            states = first + network.forward_encoded(first[-1], e[2:], observed[2:], times[2:])
        elif mode == 'step':
            current, states = initial, []
            for t in range(len(e)):
                current = network.forward_encoded(current, e[t:t+1], observed[t:t+1], times[t:t+1])[0]
                states.append(current)
        else:
            states = network.forward_encoded(initial, e, observed, times, method=mode)
        loss = sum((t+1) * (s.c.square().mean() + s.d.square().mean()) for t, s in enumerate(states))
        loss.backward()
        gradients = {'c0': initial.c.grad, 'd0': initial.d.grad, 'e': e.grad}
        gradients.update({name: p.grad for name, p in network.named_parameters()
                          if not name.startswith(('observer.', 'decoder.'))})
        for name, gradient in gradients.items():
            assert gradient is not None and torch.isfinite(gradient).all(), name
        assert any(gradients[name].abs().sum() > 0 for name in gradients if name.startswith('mixer.'))
        result.append(gradients)
    for actual in result[1:]:
        for name in result[0]:
            torch.testing.assert_close(actual[name], result[0][name], rtol=0.03, atol=2e-5, msg=name)


def test_encoded_input_validation_and_empty_sequence():
    network, state, e, observed, times = setup(length=1, batch=1)
    assert network.forward_encoded(state, e[:0], observed[:0], times[:0]) == ()
    with pytest.raises(ValueError, match='method'):
        network.forward_encoded(state, e, observed, times, method='bogus')
    with pytest.raises(ValueError, match='time|nonnegative|monotonic'):
        network.forward_encoded(state, e, observed, -times)
    with pytest.raises(ValueError, match='finite'):
        network.forward_encoded(state, e * float('nan'), observed, times)
    hidden = network.forward_encoded(state, e * float('nan'), ~observed, times)
    assert torch.isfinite(hidden[-1].c).all()


def test_model_feedback_mode_is_part_of_effective_restart_config():
    cfg = HybridConfig.tiny()
    full = HybridCDModel(cfg, feedback=True)
    cut = HybridCDModel(cfg, feedback=False)
    cut.load_state_dict(full.state_dict())
    assert full.config.feedback is True
    assert cut.config.feedback is False
    assert full.config.fingerprint() != cut.config.fingerprint()
    assert weights_fingerprint(full.state_dict()) == weights_fingerprint(cut.state_dict())
    assert cut.initial_state().config.feedback is False
    configured = HybridCDModel(replace(cfg, feedback=False))
    assert configured.cells[0].feedback is False
    assert configured.cells[1].feedback is False
    payload = full.initial_state().to_payload(weights_hash=weights_fingerprint(full.state_dict()))
    with pytest.raises(ValueError, match='config|hash'):
        HybridCDKVState.from_payload(payload, config=cut.config,
                                    weights_hash=weights_fingerprint(cut.state_dict()))
