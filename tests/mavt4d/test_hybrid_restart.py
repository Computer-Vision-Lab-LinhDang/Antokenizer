"""Fresh-process continuation with only declared hybrid state and model weights."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mavt4d.contracts import DecoderQuery, ObservationStep
from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.state import HybridCDKVState, weights_fingerprint


RESTART_SCRIPT = r"""
import sys
import torch
from mavt4d.contracts import DecoderQuery, ObservationStep
from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.state import HybridCDKVState, weights_fingerprint
torch.set_num_threads(1)
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
config = HybridConfig(**payload['config'])
model = HybridCDModel(config).eval()
model.load_state_dict(payload['weights'])
weight_hash = weights_fingerprint(model.state_dict())
state = HybridCDKVState.from_payload(payload['state'], config=config,
                                     weights_hash=weight_hash)
steps = tuple(ObservationStep(**item) for item in payload['observations'])
with torch.no_grad():
    states = model.forward_sequence(state, steps, method=payload['method'])
    rgb = model.decode(states[-1], DecoderQuery(**payload['query']))
torch.save({'states': [item.to_payload(weights_hash=weight_hash) for item in states],
            'rgb': rgb}, sys.argv[2])
"""


def _observations():
    generator = torch.Generator().manual_seed(1729)
    return tuple(
        ObservationStep(
            images=torch.randn(1, 2, 3, 5, 7, generator=generator),
            observed=torch.full((1, 2, 5, 7), index not in (3, 6, 7), dtype=torch.bool),
            frame_times=torch.tensor([[index + 0.25, index + 0.5]], dtype=torch.float64),
        )
        for index in range(10)
    )


def _assert_stream_equal(actual, expected):
    for name in ("c", "d", "k", "v", "last_time", "step_count", "write_index",
                 "steps_filled", "slot_times", "slot_step_ids", "token_valid"):
        left, right = getattr(actual, name), getattr(expected, name)
        if name in ("c", "d"):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6, msg=name)
        elif name in ("k", "v"):
            torch.testing.assert_close(left.float(), right.float(), rtol=0.008, atol=1e-4, msg=name)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)


def test_raw_observation_full_chunk_step_and_scan_parity():
    torch.manual_seed(1776)
    model = HybridCDModel(replace(HybridConfig.tiny(), cache_steps=3)).eval()
    schedule = _observations()
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
        for reference, a, b, c in zip(full, scanned, chunks, stepped):
            for actual in (a, b, c):
                _assert_stream_equal(actual, reference)
                query = DecoderQuery(torch.tensor([[0.5, 0.5, 0.0]]),
                                     torch.zeros(1, dtype=torch.int64),
                                     reference.last_time.clone())
                torch.testing.assert_close(model.decode(actual, query), model.decode(reference, query),
                                           rtol=3e-5, atol=3e-6)


def test_future_raw_images_masks_and_geometry_cannot_change_prefix():
    torch.manual_seed(1776)
    model = HybridCDModel(HybridConfig.tiny()).eval()
    schedule = _observations()[:5]
    changed = schedule[:2] + tuple(
        replace(item, images=item.images * 100, observed=~item.observed,
                frame_times=item.frame_times + 10,
                intrinsics=torch.eye(3)[None],
                cam_to_world=torch.eye(4)[None, None].expand(1, 2, 4, 4).clone())
        for item in schedule[2:]
    )
    with torch.no_grad():
        initial = model.initial_state()
        original = model.forward_sequence(initial, schedule, method="associative")
        altered = model.forward_sequence(initial, changed, method="associative")
    for actual, expected in zip(altered[:2], original[:2]):
        _assert_stream_equal(actual, expected)


@pytest.mark.parametrize("cut,method", [(1, "sequential"), (5, "associative"), (8, "sequential")])
def test_fresh_process_restart_before_and_after_cache_wrap(cut, method, tmp_path):
    torch.manual_seed(2718)
    config = replace(HybridConfig.tiny(), cache_steps=3)
    model = HybridCDModel(config).eval()
    schedule = _observations()
    with torch.no_grad():
        checkpoint = model.forward_sequence(model.initial_state(), schedule[:cut])[-1]
        expected = model.forward_sequence(checkpoint, schedule[cut:], method=method)
        query = DecoderQuery(
            coordinates=torch.tensor([[0.5, 0.5, 0.0], [6.5, 4.5, 0.0]]),
            view_ids=torch.zeros(2, dtype=torch.int64),
            times=torch.full((2,), 9.5, dtype=torch.float64),
        )
        expected_rgb = model.decode(expected[-1], query)
    weight_hash = weights_fingerprint(model.state_dict())
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
    result = subprocess.run(
        [sys.executable, "-c", RESTART_SCRIPT, str(incoming), str(outgoing)],
        cwd=root, env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    actual = torch.load(outgoing, map_location="cpu", weights_only=True)
    assert len(actual["states"]) == len(expected)
    for got, wanted in zip(actual["states"], expected):
        restored = HybridCDKVState.from_payload(got, config=config, weights_hash=weight_hash)
        reference = wanted.to_payload(weights_hash=weight_hash)
        assert got.keys() == reference.keys()
        for name, value in got.items():
            torch.testing.assert_close(value, reference[name], rtol=1e-5, atol=1e-6,
                                       msg=f"fresh-process payload mismatch: {name}")
        assert restored.retained_storage_nbytes == restored.total_nbytes
    torch.testing.assert_close(actual["rgb"], expected_rgb, rtol=1e-5, atol=1e-6)


def test_full_width_preset_has_declared_memory_and_runs_without_training():
    """Exercise default modules with a small spatial grid, not a training pilot."""
    torch.manual_seed(314)
    preset_path = Path(__file__).resolve().parents[2] / "configs/mavt4d/hybrid_cd2_kv_v2.json"
    config = HybridConfig(**json.loads(preset_path.read_text()))
    assert config == HybridConfig()
    model = HybridCDModel(config).eval()
    initial = model.initial_state()
    assert initial.total_nbytes == 148256
    item = ObservationStep(
        images=torch.zeros(1, 2, 3, 17, 33),
        observed=torch.ones(1, 2, 17, 33, dtype=torch.bool),
        frame_times=torch.tensor([[0.0, 0.1]], dtype=torch.float64),
    )
    query = DecoderQuery(
        coordinates=torch.tensor([[0.5, 0.5, 0.0], [32.5, 16.5, 0.0]]),
        view_ids=torch.zeros(2, dtype=torch.int64),
        times=torch.full((2,), 0.1, dtype=torch.float64),
    )
    with torch.no_grad():
        state = model.step(initial, item)
        rgb = model.decode(state, query)
    assert state.c.shape == state.d.shape == (1, 2, 32, 32)
    assert state.k.shape == state.v.shape == (1, 2, 16, 16, 2, 32)
    assert state.total_nbytes == state.retained_storage_nbytes == 148256
    assert rgb.shape == (1, 2, 3)
    assert torch.isfinite(rgb).all()
