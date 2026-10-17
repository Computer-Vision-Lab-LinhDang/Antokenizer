"""v3 hybrid model: innovation write and appearance bank wired into every entry point."""

from contextlib import contextmanager
from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mavt4d.contracts import DecoderQuery, ObservationStep
from mavt4d.hybrid import model as model_module
from mavt4d.hybrid.config import HybridConfig, HybridV3Config
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.state import HybridCDAKVState, HybridCDKVState, weights_fingerprint
from mavt4d.hybrid_pilot.objectives import slice_state

KINDS = {
    "both": dict(innovation=True, appearance_tokens=2),
    "innovation": dict(innovation=True),
    "appearance": dict(appearance_tokens=2),
}
V3_FIELDS = ("appearance", "appearance_valid", "prev_tokens", "prev_seen")
EXACT_FLOATS = ("last_time", "slot_times")


def v3(kind="both"):
    return HybridV3Config.tiny(cache_steps=3, **KINDS[kind])


def make_model(kind="both", seed=2801):
    torch.manual_seed(seed)
    return HybridCDModel(v3(kind)).eval()


@contextmanager
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def capture_innovation(model):
    """Record δ̂ [N,P,W] on every pool call."""
    captured = []
    model.innovation_pool.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    return captured


def observations(count=10, seed=1729):
    """8x12 two-frame steps on the (4,6) grid: missing steps 0,3,6,7; a corner hidden on steps 1-4."""
    generator = torch.Generator().manual_seed(seed)
    steps = []
    for index in range(count):
        observed = torch.full((1, 2, 8, 12), index not in (0, 3, 6, 7), dtype=torch.bool)
        if 1 <= index <= 4:
            observed[..., :4, :4] = False
        steps.append(ObservationStep(
            images=torch.randn(1, 2, 3, 8, 12, generator=generator), observed=observed,
            frame_times=torch.tensor([[index + .25, index + .5]], dtype=torch.float64)))
    return tuple(steps)


def missing_step(time, frames=1):
    times = torch.tensor([[time - .1 * (frames - 1 - f) for f in range(frames)]], dtype=torch.float64)
    return ObservationStep(images=torch.zeros(1, frames, 3, 8, 12),
                           observed=torch.zeros(1, frames, 8, 12, dtype=torch.bool), frame_times=times)


def encoded(config, length=6, batch=2, seed=7):
    """Synthetic post-encoder inputs; patches/coverage only when innovation is on."""
    generator = torch.Generator().manual_seed(seed)
    hp, wp = config.innovation_grid
    evidence = torch.randn(length, batch, config.summary_tokens, config.width, generator=generator)
    observed = torch.ones(length, batch, dtype=torch.bool)
    times = torch.arange(1, length + 1, dtype=torch.float64)[:, None].expand(-1, batch).clone()
    patches = torch.randn(length, batch, config.reference_views, hp, wp, config.spatial_width,
                          generator=generator)
    coverage = (torch.rand(length, batch, config.reference_views, hp, wp, generator=generator) > .3).float()
    return evidence, observed, times, patches, coverage


def patch_kwargs(config, patches, coverage):
    return dict(patches=patches, coverage=coverage) if config.innovation else {}


def assert_same(actual, expected, *, rtol=3e-5, atol=3e-6):
    assert type(actual) is HybridCDAKVState
    for name in HybridCDAKVState.STATE_FIELDS:
        left, right = getattr(actual, name), getattr(expected, name)
        if name in ("k", "v"):
            torch.testing.assert_close(left.float(), right.float(), rtol=.008, atol=1e-4, msg=name)
        elif left.is_floating_point() and name not in EXACT_FLOATS:
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=name)
        else:
            assert torch.equal(left, right), name


def assert_bit_identical(actual, expected, names=HybridCDAKVState.STATE_FIELDS):
    for name in names:
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def assert_exercised(states, config):
    """Guard against vacuous passes: the v3 machinery actually fired."""
    last = states[-1]
    if config.appearance_tokens:
        assert last.appearance_valid.all() and last.appearance.abs().sum() > 0
    if config.innovation:
        assert last.prev_seen.any()


RESTART_SCRIPT = r"""
import sys
import torch
from mavt4d.contracts import DecoderQuery, ObservationStep
from mavt4d.hybrid.config import HybridV3Config
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.state import HybridCDAKVState, weights_fingerprint
torch.set_num_threads(1)
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
values = dict(payload['config'], innovation_grid=tuple(payload['config']['innovation_grid']))
config = HybridV3Config(**values)
model = HybridCDModel(config).eval()
model.load_state_dict(payload['weights'])
weight_hash = weights_fingerprint(model.state_dict())
state = HybridCDAKVState.from_payload(payload['state'], config=config, weights_hash=weight_hash)
steps = tuple(ObservationStep(**item) for item in payload['observations'])
with torch.no_grad():
    states = model.forward_sequence(state, steps, method=payload['method'])
    rgb = model.decode(states[-1], DecoderQuery(**payload['query']))
torch.save({'states': [item.to_payload(weights_hash=weight_hash) for item in states],
            'rgb': rgb}, sys.argv[2])
"""


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("cut,method", [(2, "sequential"), (8, "associative")])
def test_v3_restart_in_fresh_process(kind, cut, method, tmp_path):
    model = make_model(kind)
    config, schedule = model.config, observations()
    query = DecoderQuery(torch.tensor([[0.5, 0.5, 0.0], [11.5, 7.5, 0.0]]),
                         torch.zeros(2, dtype=torch.int64), torch.full((2,), 9.5, dtype=torch.float64))
    weight_hash = weights_fingerprint(model.state_dict())
    with torch.no_grad(), single_thread():
        checkpoint = model.forward_sequence(model.initial_state(), schedule[:cut])[-1]
        expected = model.forward_sequence(checkpoint, schedule[cut:], method=method)
        expected_rgb = model.decode(expected[-1], query)
        restored = HybridCDAKVState.from_payload(checkpoint.to_payload(weights_hash=weight_hash),
                                                 config=config, weights_hash=weight_hash)
        for got, wanted in zip(model.forward_sequence(restored, schedule[cut:], method=method), expected):
            assert_bit_identical(got, wanted)
    assert_exercised(expected, config)
    if cut == 2 and config.innovation:  # the corner is still unseen at the cut and revealed after it
        assert not checkpoint.prev_seen.all()
    incoming, outgoing = tmp_path / "resume.pt", tmp_path / "continued.pt"
    torch.save({
        "config": asdict(config), "weights": model.state_dict(),
        "state": checkpoint.to_payload(weights_hash=weight_hash),
        "observations": [asdict(item) for item in schedule[cut:]],
        "query": asdict(query), "method": method,
    }, incoming)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run([sys.executable, "-c", RESTART_SCRIPT, str(incoming), str(outgoing)],
                            cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    actual = torch.load(outgoing, map_location="cpu", weights_only=True)
    assert len(actual["states"]) == len(expected)
    for got, wanted in zip(actual["states"], expected):
        reference = wanted.to_payload(weights_hash=weight_hash)
        assert got.keys() == reference.keys()
        for name, value in got.items():
            assert torch.equal(value, reference[name]), f"fresh-process payload mismatch: {name}"
        HybridCDAKVState.from_payload(got, config=config, weights_hash=weight_hash)
    assert torch.equal(actual["rgb"], expected_rgb)


@pytest.mark.parametrize("kind", list(KINDS))
def test_v3_step_chunk_scan_parity(kind):
    model = make_model(kind)
    config, schedule = model.config, observations()
    captured = capture_innovation(model) if config.innovation else [torch.ones(1)]
    initial = model.initial_state()
    initial = replace(initial, c=torch.randn_like(initial.c), d=torch.randn_like(initial.d))
    with torch.no_grad():
        full = model.forward_sequence(initial, schedule)
        scanned = model.forward_sequence(initial, schedule, method="associative")
        head = model.forward_sequence(initial, schedule[:4], method="associative")
        chunks = head + model.forward_sequence(head[-1], schedule[4:], method="associative")
        stepped, current = [], initial
        for item in schedule:
            current = model.step(current, item)
            stepped.append(current)
    assert_exercised(full, config)
    assert any(torch.count_nonzero(delta_hat) for delta_hat in captured)
    for reference, *candidates in zip(full, scanned, chunks, stepped):
        for candidate in candidates:
            assert_same(candidate, reference)
            assert candidate.retained_storage_nbytes == candidate.total_nbytes

    gradients = []
    for mode in ("sequential", "associative", "chunk", "step"):
        model = make_model(kind)
        evidence, observed, times, patches, coverage = encoded(config, length=5, batch=2)
        observed[1, 0] = False
        times[1, 0] = times[0, 0]  # per-stream no-op
        observed[2, 1] = False  # accepted missing step
        evidence.requires_grad_()
        patches.requires_grad_()
        state = model.initial_state(batch_size=2)
        state = replace(state, c=torch.randn_like(state.c).requires_grad_(),
                        d=torch.randn_like(state.d).requires_grad_())
        extra = lambda a, b: patch_kwargs(config, patches[a:b], coverage[a:b])
        if mode == "chunk":
            first = model.forward_encoded(state, evidence[:2], observed[:2], times[:2], **extra(0, 2))
            states = first + model.forward_encoded(first[-1], evidence[2:], observed[2:], times[2:], **extra(2, 5))
        elif mode == "step":
            current, states = state, []
            for t in range(5):
                current = model.forward_encoded(current, evidence[t:t + 1], observed[t:t + 1],
                                                times[t:t + 1], **extra(t, t + 1))[0]
                states.append(current)
        else:
            states = model.forward_encoded(state, evidence, observed, times, method=mode, **extra(0, 5))
        loss = sum((t + 1) * (s.c.square().mean() + s.d.square().mean() + s.appearance.square().sum()
                              + 1e-3 * s.prev_tokens.square().sum()) for t, s in enumerate(states))
        loss.backward()
        result = {"c0": state.c.grad, "d0": state.d.grad, "evidence": evidence.grad}
        if config.innovation:
            result["patches"] = patches.grad
        result.update({name: p.grad for name, p in model.named_parameters()
                       if name.startswith(("innovation_pool.", "appearance_writer.", "cells.", "mixer."))})
        for name, gradient in result.items():
            assert gradient is not None and torch.isfinite(gradient).all(), (mode, name)
        for prefix in ("innovation_pool.", "appearance_writer."):
            names = [n for n in result if n.startswith(prefix)]
            assert not names or any(result[n].abs().sum() > 0 for n in names), (mode, prefix)
        gradients.append(result)
    for actual in gradients[1:]:
        for name in gradients[0]:
            torch.testing.assert_close(actual[name], gradients[0][name], rtol=3e-5, atol=3e-6, msg=name)


@pytest.mark.parametrize("kind", list(KINDS))
def test_v3_causality(kind):
    model = make_model(kind)
    config, schedule = model.config, observations(count=7)
    changed = schedule[:3] + tuple(replace(item, images=item.images * 100 + 3) for item in schedule[3:])
    with torch.no_grad():
        initial = model.initial_state()
        original = model.forward_sequence(initial, schedule, method="associative")
        altered = model.forward_sequence(initial, changed, method="associative")
    assert_exercised(original[:3], config)
    for actual, expected in zip(altered[:3], original[:3]):
        assert_same(actual, expected)
        assert_bit_identical(actual, expected, V3_FIELDS)
    if config.innovation:
        assert not torch.equal(altered[-1].prev_tokens, original[-1].prev_tokens)

    evidence, observed, times, patches, coverage = encoded(config)
    with torch.no_grad():
        state = model.initial_state(batch_size=2)
        before = model.forward_encoded(state, evidence, observed, times, **patch_kwargs(config, patches, coverage))
        evidence[3:] *= 1000
        patches[3:] = -patches[3:] * 50
        coverage[3:] = 1 - coverage[3:]
        observed[3:, 0] = False
        times[3:] += 20
        after = model.forward_encoded(state, evidence, observed, times, **patch_kwargs(config, patches, coverage))
    for actual, expected in zip(after[:3], before[:3]):
        assert_same(actual, expected)
        assert_bit_identical(actual, expected, V3_FIELDS)


@pytest.mark.parametrize("kind", list(KINDS))
def test_first_observation_after_missing_writes_A_and_prev(kind):
    model = make_model(kind)
    config = model.config
    captured = capture_innovation(model) if config.innovation else []
    evidence, observed, times, patches, coverage = encoded(config, length=4, batch=1)
    observed[:2] = False  # accepted missing steps (dt > 0) before the first observation
    with torch.no_grad():
        initial = model.initial_state()
        states = model.forward_encoded(initial, evidence, observed, times, **patch_kwargs(config, patches, coverage))
    for state in states[:2]:
        assert_bit_identical(state, initial, V3_FIELDS)
        assert state.step_count.item() > initial.step_count.item()
    first, after = states[2], states[3]
    if config.appearance_tokens:
        assert first.appearance_valid.all()
        torch.testing.assert_close(first.appearance, model.appearance_writer(evidence[2]), rtol=1e-5, atol=1e-6)
        assert torch.equal(after.appearance, first.appearance)
    else:
        assert not first.appearance_valid.any() and not after.appearance_valid.any()
    if config.innovation:
        seen = coverage[2] >= 1 - 1e-6
        assert seen.any() and not seen.all()
        assert torch.equal(first.prev_seen, seen)
        assert torch.equal(first.prev_tokens, torch.where(seen[..., None], patches[2], torch.zeros_like(patches[2])))
        delta_hat = captured[-1].reshape(4, 1, config.summary_tokens, config.width)
        assert torch.count_nonzero(delta_hat[:2]) == 0
        # With A on, A holds the first observed step; without it, that step is new evidence for C.
        assert (torch.count_nonzero(delta_hat[2]) == 0) == bool(config.appearance_tokens)


@pytest.mark.parametrize("kind", list(KINDS))
def test_missing_dt_zero_is_exact_noop_and_dt_positive_keeps_A_prev(kind):
    model = make_model(kind)
    config = model.config
    with torch.no_grad():
        state = model.forward_sequence(model.initial_state(), observations(count=6))[-1]
        assert_exercised((state,), config)
        snapshot = state.clone()
        noops = (model.rollout(state, state.last_time[None], method="associative")[0],
                 model.step(state, missing_step(float(state.last_time))))
        later = (model.rollout(state, state.last_time[None] + 1)[0],
                 model.step(state, missing_step(float(state.last_time) + 1, frames=2)))
    for noop in noops:
        assert_bit_identical(noop, snapshot)
    for moved in later:
        assert_bit_identical(moved, snapshot, V3_FIELDS)
        assert moved.step_count.item() == snapshot.step_count.item() + 1
        assert not torch.equal(moved.c, snapshot.c)


def test_streams_at_different_progress():
    model = make_model("both")
    config = model.config
    captured = capture_innovation(model)
    evidence, observed, times, patches, coverage = encoded(config, length=5, batch=2)
    observed[:2, 1] = False  # stream 1 first observed at step 2; stream 0 from step 0
    kwargs = lambda a, b, rows=slice(None): dict(patches=patches[a:b, rows], coverage=coverage[a:b, rows])
    with torch.no_grad():
        initial = model.initial_state(batch_size=2)
        whole = model.forward_encoded(initial, evidence, observed, times, **kwargs(0, 5))
        whole_delta = captured[-1].reshape(5, 2, config.summary_tokens, config.width)
        head = model.forward_encoded(initial, evidence[:2], observed[:2], times[:2], **kwargs(0, 2))
        assert head[-1].appearance_valid.tolist() == [True, False]
        tail = model.forward_encoded(head[-1], evidence[2:], observed[2:], times[2:], **kwargs(2, 5))
        batched = head + tail
        tail_delta = captured[-1].reshape(3, 2, config.summary_tokens, config.width)
        alone = []
        for b in range(2):
            rows = slice(b, b + 1)
            single = slice_state(initial, b, b + 1)
            first = model.forward_encoded(single, evidence[:2, rows], observed[:2, rows], times[:2, rows],
                                          **kwargs(0, 2, rows))
            alone.append(first + model.forward_encoded(first[-1], evidence[2:, rows], observed[2:, rows],
                                                       times[2:, rows], **kwargs(2, 5, rows)))
    for delta_hat in (whole_delta[2], tail_delta[0]):  # within one call and across a chunk boundary
        assert torch.count_nonzero(delta_hat[1]) == 0  # stream 1's first observation goes to A
        assert torch.count_nonzero(delta_hat[0]) > 0  # stream 0 keeps writing change into C
    assert torch.equal(batched[-1].appearance[0], batched[0].appearance[0])  # stream 0 never rewritten
    assert not batched[1].appearance[1].any() and batched[2].appearance_valid[1]
    assert torch.equal(batched[-1].appearance[1], batched[2].appearance[1])
    for t, state in enumerate(batched):
        assert_same(whole[t], state)
        for b in range(2):
            assert_same(slice_state(state, b, b + 1), alone[b][t])


def test_revealed_patch_writes_C_not_A():
    model = make_model("both")
    config = model.config
    captured = capture_innovation(model)
    evidence, observed, times, patches, _ = encoded(config, length=3, batch=1)
    patches = patches[:1].expand(3, -1, -1, -1, -1, -1).clone()  # static scene
    visible = torch.ones(3, 1, config.reference_views, *config.innovation_grid)
    hidden = visible.clone()
    hidden[0, :, :, 0, 0] = 0  # hidden at the first observed step, revealed at step 1
    with torch.no_grad():
        initial = model.initial_state()
        runs = {}
        for name, coverage in (("visible", visible), ("hidden", hidden)):
            states = model.forward_encoded(initial, evidence, observed, times, patches=patches, coverage=coverage)
            runs[name] = states, captured[-1].reshape(3, config.summary_tokens, config.width)
    (seen_all, delta_all), (revealed, delta_rev) = runs["visible"], runs["hidden"]
    assert torch.count_nonzero(delta_all) == 0
    assert torch.count_nonzero(delta_rev[0]) == 0 and torch.count_nonzero(delta_rev[2]) == 0
    assert torch.count_nonzero(delta_rev[1]) > 0
    assert not revealed[0].prev_seen[..., 0, 0].any() and revealed[1].prev_seen.all()
    assert_bit_identical(revealed[0], seen_all[0], ("c", "d", "appearance", "appearance_valid"))
    assert torch.equal(revealed[1].appearance, revealed[0].appearance)
    assert torch.equal(revealed[1].appearance, seen_all[1].appearance)
    assert not torch.equal(revealed[1].c[:, 0], seen_all[1].c[:, 0])  # the reveal writes C
    assert torch.equal(revealed[1].d[:, 0], seen_all[1].d[:, 0])  # and only C


def test_dropped_step_with_real_patches_is_inert():
    model = make_model("both")
    config = model.config
    captured = capture_innovation(model)
    evidence, observed, times, patches, coverage = encoded(config, length=3, batch=1)
    observed[1] = False
    coverage[1] = 1
    blank_patches, blank_coverage = patches.clone(), coverage.clone()
    blank_patches[1], blank_coverage[1] = 0, 0
    with torch.no_grad():
        initial = model.initial_state()
        dropped = model.forward_encoded(initial, evidence, observed, times, patches=patches, coverage=coverage)
        delta_hat = captured[-1].reshape(3, config.summary_tokens, config.width)
        blank = model.forward_encoded(initial, evidence, observed, times,
                                      patches=blank_patches, coverage=blank_coverage)
    assert torch.count_nonzero(delta_hat[1]) == 0
    assert_bit_identical(dropped[1], dropped[0], V3_FIELDS)
    for got, wanted in zip(dropped, blank):
        assert_bit_identical(got, wanted)


@pytest.mark.parametrize("kind", ["both", "appearance"])
def test_v3_render_reads_appearance(kind):
    model = make_model(kind)
    config = model.config
    evidence, observed, times, patches, coverage = encoded(config, length=3, batch=1)
    with torch.no_grad():
        state = model.forward_encoded(model.initial_state(), evidence, observed, times,
                                      **patch_kwargs(config, patches, coverage))[-1]
        code = state.code()
        whole = model.decoder.render(code, height=8, width=12)
        chunked = model.decoder.render(code, height=8, width=12, chunk_size=5)
        moved = model.decoder.render(replace(code, appearance=code.appearance + 1), height=8, width=12)
    assert whole.shape == (1, 3, 8, 12)
    torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(moved, whole)


def test_forward_encoded_validates_patches():
    for kind in ("both", "innovation"):
        model = make_model(kind)
        config = model.config
        evidence, observed, times, patches, coverage = encoded(config, length=2, batch=1)
        state = model.initial_state()
        with pytest.raises(ValueError, match="patches"):
            model.forward_encoded(state, evidence, observed, times)
        with pytest.raises(ValueError, match="patches"):
            model.forward_encoded(state, evidence, observed, times, patches=patches)
        wide = torch.zeros(*patches.shape[:4], patches.shape[4] + 1, patches.shape[5])
        with pytest.raises(ValueError, match="innovation_grid"):
            model.forward_encoded(state, evidence, observed, times, patches=wide, coverage=wide[..., 0])
        with pytest.raises(ValueError, match="FP32"):
            model.forward_encoded(state, evidence, observed, times, patches=patches.double(), coverage=coverage)
    torch.manual_seed(0)
    for other in (HybridCDModel(v3("appearance")), HybridCDModel(HybridConfig.tiny())):
        state = other.initial_state()
        e, o, t, p, c = encoded(v3("both"), length=2, batch=1)
        with pytest.raises(ValueError, match="innovation is off"):
            other.forward_encoded(state, e, o, t, patches=p, coverage=c)


def test_forward_sequence_diagnostics():
    model = make_model("both")
    diagnostics = {}
    with torch.no_grad():
        model.forward_sequence(model.initial_state(), observations(count=5), diagnostics=diagnostics)
    for stage in ("stage1", "stage2"):
        ratio = diagnostics[stage]["dc_feedback_ratio"]
        assert ratio.ndim == 0 and torch.isfinite(ratio)


def test_v3_model_states_and_v2_modules():
    for kind, config in KINDS.items():
        model = make_model(kind)
        assert type(model.initial_state()) is HybridCDAKVState
        assert hasattr(model, "innovation_pool") == bool(config.get("innovation"))
        assert hasattr(model, "appearance_writer") == bool(config.get("appearance_tokens"))
        with pytest.raises(TypeError, match="HybridCDAKVState"):
            model.step(HybridCDKVState.initial(HybridConfig.tiny(), batch_size=1, device="cpu", last_time=0.0),
                       missing_step(1.0))


def test_v2_path_never_calls_v3_components(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("v2 path called a v3 component")

    for name in ("patch_innovation", "InnovationPool", "AppearanceWriter", "write_once", "observed_patches"):
        monkeypatch.setattr(model_module, name, forbidden)
    torch.manual_seed(5)
    model = HybridCDModel(replace(HybridConfig.tiny(), cache_steps=3)).eval()
    assert not hasattr(model, "innovation_pool") and not hasattr(model, "appearance_writer")
    schedule = observations(count=4)
    with torch.no_grad():
        states = model.forward_sequence(model.initial_state(), schedule, method="associative")
        stepped = model.step(states[-1], missing_step(float(states[-1].last_time) + 1, frames=2))
        rolled = model.rollout(stepped, stepped.last_time[None] + 1)[-1]
        query = DecoderQuery(torch.tensor([[0.5, 0.5, 0.0]]), torch.zeros(1, dtype=torch.int64),
                             rolled.last_time.clone())
        rgb = model.decode(rolled, query)
        image = model.decoder.render(rolled.code(), height=8, width=12)
    assert type(rolled) is HybridCDKVState
    assert torch.isfinite(rgb).all() and torch.isfinite(image).all()
