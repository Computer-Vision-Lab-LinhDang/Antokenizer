"""Integration gates for the observation -> explicit C/D -> query path."""

from __future__ import annotations

from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mavt4d.contracts import CDState, CDStateSpec, DecoderQuery, ObservationStep
from mavt4d.data.event_synthetic import EventSceneConfig, make_event_fixture
from mavt4d.reference.model import ReferenceCDModel


KINDS = ("coupled", "two_scan", "gru")


def spec() -> CDStateSpec:
    return CDStateSpec(c_tokens=2, d_tokens=2, token_dim=4)


def model(kind: str = "coupled") -> ReferenceCDModel:
    torch.manual_seed(1701)
    return ReferenceCDModel(spec(), cell_kind=kind, embed_dim=8, patch_size=2,
                            modulation_dim=2).eval()


def observation(start: float, seed: int = 1, *, missing: bool = False) -> ObservationStep:
    generator = torch.Generator().manual_seed(seed)
    return ObservationStep(
        images=torch.rand(1, 2, 3, 4, 6, generator=generator),
        observed=torch.full((1, 2, 4, 6), not missing, dtype=torch.bool),
        frame_times=torch.tensor([[start + 0.05, start + 0.10]], dtype=torch.float64),
    )


def steps() -> tuple[ObservationStep, ...]:
    return tuple(observation(index / 4, index + 2) for index in range(4))


def query(time: float) -> DecoderQuery:
    return DecoderQuery(
        coordinates=torch.tensor([[0.5, 0.5, 0.0], [3.5, 2.5, 0.0]]),
        view_ids=torch.zeros(2, dtype=torch.int64),
        times=torch.full((2,), time, dtype=torch.float64),
    )


def assert_state_equal(actual: CDState, expected: CDState) -> None:
    assert actual.spec == expected.spec
    torch.testing.assert_close(actual.c, expected.c, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual.d, expected.d, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual.last_time, expected.last_time, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_initial_state_is_only_declared_zero_memory(kind):
    network = model(kind).double()
    state = network.initial_state(batch_size=2, last_time=1.234567890123)
    assert state.spec == spec()
    assert state.c.shape == (2, 2, 4)
    assert state.d.shape == (2, 2, 4)
    assert state.c.dtype == state.d.dtype == torch.float64
    assert state.last_time.dtype == torch.float64
    assert torch.count_nonzero(state.c) == torch.count_nonzero(state.d) == 0
    torch.testing.assert_close(state.last_time, torch.full((2,), 1.234567890123,
                                                        dtype=torch.float64), rtol=0, atol=0)
    assert state.retained_storage_nbytes == state.total_nbytes


@pytest.mark.parametrize("kind", KINDS)
def test_future_observations_cannot_change_prefix_states_or_decode(kind):
    network = model(kind)
    schedule = steps()
    changed = schedule[:2] + tuple(replace(item, images=1 - item.images)
                                  for item in schedule[2:])
    with torch.no_grad():
        left = network.forward_sequence(network.initial_state(), schedule)
        right = network.forward_sequence(network.initial_state(), changed)
        for original, perturbed in zip(left[:2], right[:2]):
            assert_state_equal(original, perturbed)
            torch.testing.assert_close(network.decode(original, query(float(original.last_time[0]))),
                                       network.decode(perturbed, query(float(perturbed.last_time[0]))),
                                       rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_full_chunk_and_step_parity_with_compact_returned_states(kind):
    network = model(kind)
    schedule = steps()
    with torch.no_grad():
        initial = network.initial_state()
        full = network.forward_sequence(initial, schedule)
        first = network.forward_sequence(initial, schedule[:2])
        second = network.forward_sequence(first[-1], schedule[2:])
        online = []
        current = initial
        for item in schedule:
            current = network.step(current, item)
            online.append(current)
        for expected, chunked, stepped in zip(full, first + second, online):
            assert_state_equal(expected, chunked)
            assert_state_equal(expected, stepped)
            assert expected.retained_storage_nbytes == expected.total_nbytes
            assert stepped.retained_storage_nbytes == stepped.total_nbytes
        assert network.forward_sequence(initial, ()) == ()
        if kind != "gru":
            scanned = network.forward_sequence(initial, schedule, method="associative")
            for expected, actual in zip(full, scanned):
                assert_state_equal(actual, expected)
                assert actual.retained_storage_nbytes == actual.total_nbytes
        else:
            with pytest.raises(ValueError, match="associative|method|scan"):
                network.forward_sequence(initial, schedule, method="associative")


def gradients(network, schedule, execution):
    initial = network.initial_state()
    initial = CDState(initial.c.clone().requires_grad_(), initial.d.clone().requires_grad_(),
                      initial.last_time, spec=initial.spec)
    if execution == "step":
        states = []
        current = initial
        for item in schedule:
            current = network.step(current, item)
            states.append(current)
    elif execution == "chunk":
        first = network.forward_sequence(initial, schedule[:2])
        states = first + network.forward_sequence(first[-1], schedule[2:])
    else:
        states = network.forward_sequence(initial, schedule, method=execution)
    loss = sum((index + 1) * (state.c.square().mean() + state.d.square().mean())
               for index, state in enumerate(states))
    loss = loss + network.decode(states[-1], query(float(states[-1].last_time[0]))).square().mean()
    if getattr(network, "temporal_head", None) is not None:
        loss = loss + network.temporal_head(states[-1].c).square().mean()
    loss.backward()
    result = {"initial.c": initial.c.grad, "initial.d": initial.d.grad}
    result.update({name: value.grad for name, value in network.named_parameters()
                   if value.requires_grad})
    # This schedule is camera-only. The explicit triplane-only lookup tables
    # are inactive here; every other parameter must participate in the graph.
    for name in ("observer.plane_embedding.weight", "decoder.plane_embedding.weight"):
        assert name in result
        assert result.pop(name) is None, f"triplane table unexpectedly active: {name}"
    for name, value in result.items():
        assert value is not None, f"active parameter/state has no gradient: {name}"
        assert torch.isfinite(value).all(), name
    assert result["initial.c"].abs().sum() > 0
    assert result["initial.d"].abs().sum() > 0
    return {name: value.detach().clone() for name, value in result.items()}


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_match_for_full_chunk_step_and_associative_execution(kind):
    schedule = steps()
    schedule = schedule[:2] + (observation(0.5, missing=True),) + schedule[3:]
    expected = gradients(model(kind), schedule, "sequential")
    methods = ("step", "chunk") if kind == "gru" else ("step", "chunk", "associative")
    for method in methods:
        actual = gradients(model(kind), schedule, method)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=2e-4, atol=2e-6,
                                       msg=f"gradient mismatch {kind}/{method}/{name}")


@pytest.mark.parametrize("kind", KINDS)
def test_hidden_rgb_including_nan_is_isolated_and_has_zero_gradient(kind):
    network = model(kind)
    item = observation(0)
    availability = item.observed.clone()
    availability[..., :2, :3] = False
    clean = replace(item, observed=availability)
    hidden = ~availability.unsqueeze(2).expand_as(item.images)
    corrupt_rgb = item.images.clone()
    corrupt_rgb[hidden] = float("nan")
    corrupt_rgb.requires_grad_()
    corrupt = replace(item, images=corrupt_rgb, observed=availability)
    expected = network.step(network.initial_state(), clean)
    actual = network.step(network.initial_state(), corrupt)
    assert_state_equal(actual, expected)
    readout = network.decode(actual, query(float(actual.last_time[0])))
    (readout.square().sum() + actual.c.square().sum() + actual.d.square().sum()).backward()
    assert torch.isfinite(corrupt_rgb.grad).all()
    assert torch.count_nonzero(corrupt_rgb.grad[hidden]) == 0
    assert corrupt_rgb.grad[~hidden].abs().sum() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_missing_zero_time_is_identity_and_rollout_advances_physical_time(kind):
    network = model(kind)
    state = network.step(network.initial_state(), observation(0))
    instant = ObservationStep(torch.full((1, 1, 3, 4, 6), float("nan")),
                              torch.zeros((1, 1, 4, 6), dtype=torch.bool),
                              state.last_time.reshape(1, 1).clone())
    with torch.no_grad():
        same = network.step(state, instant)
        assert_state_equal(same, state)
        times = torch.tensor([[0.2], [0.4], [0.8]], dtype=torch.float64)
        future = network.rollout(state, times)
        assert len(future) == 3
        for index, value in enumerate(future):
            assert torch.isfinite(value.c).all() and torch.isfinite(value.d).all()
            torch.testing.assert_close(value.last_time, times[index], rtol=0, atol=0)
            assert value.retained_storage_nbytes == value.total_nbytes
        assert not (torch.equal(future[-1].c, state.c) and torch.equal(future[-1].d, state.d))
        if kind != "gru":
            scanned = network.rollout(state, times, method="associative")
            for expected, actual in zip(future, scanned):
                assert_state_equal(actual, expected)


@pytest.mark.parametrize("kind", KINDS)
def test_interleaved_stream_has_no_module_memory_or_cross_stream_influence(kind):
    network = model(kind)
    before = {name: value.detach().clone() for name, value in network.state_dict().items()}
    with torch.no_grad():
        expected = network.forward_sequence(network.initial_state(), steps())
        current = network.initial_state()
        unrelated = network.initial_state(last_time=10)
        for index, item in enumerate(steps()):
            unrelated = network.step(unrelated, observation(10 + index, 50 + index))
            current = network.step(current, item)
            assert_state_equal(current, expected[index])
        after = network.state_dict()
        assert before.keys() == after.keys()
        for name in before:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_batch_streams_equal_independent_runs(kind):
    network = model(kind)
    left = steps()
    right = tuple(observation(index / 4, index + 20) for index in range(4))
    with torch.no_grad():
        batched = network.forward_sequence(network.initial_state(2), tuple(zip(left, right)))
        singles = [network.forward_sequence(network.initial_state(), schedule)
                   for schedule in (left, right)]
        for index, state in enumerate(batched):
            for batch in range(2):
                part = CDState(state.c[batch:batch + 1], state.d[batch:batch + 1],
                               state.last_time[batch:batch + 1], spec=state.spec)
                assert_state_equal(part, singles[batch][index])


@pytest.mark.parametrize("kind", KINDS)
def test_stale_early_frame_is_rejected_even_when_max_timestamp_is_new(kind):
    network = model(kind)
    state = network.initial_state(last_time=1)
    stale = replace(observation(0), frame_times=torch.tensor([[0.9, 1.1]], dtype=torch.float64))
    with pytest.raises(ValueError, match="time|chronolog|stale"):
        network.step(state, stale)
    with pytest.raises(ValueError, match="time|chronolog|stale"):
        network.forward_sequence(network.initial_state(), (observation(0.9), stale))
    same_time = ObservationStep(torch.ones(1, 1, 3, 4, 6),
                                torch.ones(1, 1, 4, 6, dtype=torch.bool),
                                torch.ones(1, 1, dtype=torch.float64))
    assert network.step(state, same_time).last_time.item() == 1


@pytest.mark.parametrize("kind", KINDS)
def test_future_decode_requires_explicit_rollout_and_does_not_mutate_state(kind):
    network = model(kind)
    state = network.step(network.initial_state(), observation(0))
    before = state.snapshot()
    with pytest.raises(ValueError, match="future|time|rollout"):
        network.decode(state, query(0.3))
    assert_state_equal(state, before)
    predicted = network.rollout(state, torch.tensor([[0.3]], dtype=torch.float64))[-1]
    assert network.decode(predicted, query(0.3)).shape == (1, 2, 3)


RESTART_SCRIPT = r"""
import sys
import torch
from mavt4d.contracts import CDState, CDStateSpec, DecoderQuery, ObservationStep
from mavt4d.reference.model import ReferenceCDModel
torch.set_num_threads(1)
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
network = ReferenceCDModel(CDStateSpec(**payload['spec']), **payload['kwargs']).eval()
network.load_state_dict(payload['weights'])
state = CDState.from_payload(payload['state'])
schedule = tuple(ObservationStep(**value) for value in payload['observations'])
with torch.no_grad():
    states = network.forward_sequence(state, schedule)
    rgb = network.decode(states[-1], DecoderQuery(**payload['query']))
torch.save({'states': [state.to_payload() for state in states], 'rgb': rgb}, sys.argv[2])
"""


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_process_restart_needs_only_weights_cd_payload_and_new_inputs(kind, tmp_path):
    network = model(kind)
    schedule = steps()
    with torch.no_grad():
        checkpoint = network.forward_sequence(network.initial_state(), schedule[:2])[-1]
        expected = network.forward_sequence(checkpoint, schedule[2:])
        read_query = query(float(expected[-1].last_time[0]))
        expected_rgb = network.decode(expected[-1], read_query)
    payload = {
        "spec": asdict(spec()),
        "kwargs": {"cell_kind": kind, "embed_dim": 8, "patch_size": 2, "modulation_dim": 2},
        "weights": network.state_dict(),
        "state": checkpoint.to_payload(),
        "observations": [dict(images=item.images, observed=item.observed,
                              frame_times=item.frame_times, intrinsics=item.intrinsics,
                              cam_to_world=item.cam_to_world) for item in schedule[2:]],
        "query": dict(coordinates=read_query.coordinates, view_ids=read_query.view_ids,
                      times=read_query.times, coordinate_space=read_query.coordinate_space),
    }
    incoming, outgoing = tmp_path / "resume.pt", tmp_path / "result.pt"
    torch.save(payload, incoming)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run([sys.executable, "-c", RESTART_SCRIPT, str(incoming), str(outgoing)],
                            env=env, cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    restored = torch.load(outgoing, map_location="cpu", weights_only=True)
    for actual, wanted in zip(restored["states"], expected):
        assert_state_equal(CDState.from_payload(actual), wanted)
    torch.testing.assert_close(restored["rgb"], expected_rgb, rtol=3e-5, atol=3e-6)


def test_actual_event_fixture_runs_through_declared_inputs_and_readout_without_labels():
    fixture = make_event_fixture(EventSceneConfig(T=8, H=16, W=24), source_seed=0,
                                 core="left_then_right", configuration="compact")
    network = model()
    schedule = fixture.observation_prefix(8)
    with torch.no_grad():
        states = network.forward_sequence(network.initial_state(), schedule)
        rgb = network.decode(states[-1], query(float(states[-1].last_time[0])))
    assert len(states) == 4
    assert rgb.shape == (1, 2, 3)
    assert torch.isfinite(rgb).all()
    assert states[-1].retained_storage_nbytes == states[-1].total_nbytes


def test_invalid_model_execution_and_rollout_inputs_are_rejected():
    with pytest.raises(ValueError, match="cell_kind|cell|kind"):
        ReferenceCDModel(spec(), cell_kind="unknown")
    network = model()
    state = network.initial_state()
    with pytest.raises(ValueError, match="method|scan"):
        network.forward_sequence(state, steps(), method="unknown")
    with pytest.raises(ValueError, match="batch|observation"):
        network.step(network.initial_state(2), observation(0))
    with pytest.raises(ValueError, match="time|shape|float64"):
        network.rollout(state, torch.tensor([[0.1]], dtype=torch.float32))
    with pytest.raises(ValueError, match="time|chronolog"):
        network.rollout(state, torch.tensor([[0.2], [0.1]], dtype=torch.float64))
    assert network.rollout(state, torch.empty(0, 1, dtype=torch.float64)) == ()
