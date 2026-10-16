#!/usr/bin/env python3
"""Bounded GPU correctness gate for the hybrid reference, without training.

Run inside an allocated GPUardian job. Both FP32 spatial compute and selective
BF16 spatial autocast are checked independently; a failed precision mode is
reported without relaxing the existing hybrid test tolerances.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mavt4d.contracts import DecoderQuery, ObservationStep
from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.state import HybridCDKVState, weights_fingerprint


STATE_FIELDS = ("c", "d", "k", "v", "last_time", "step_count", "write_index",
                "steps_filled", "slot_times", "slot_step_ids", "token_valid")
TOLERANCES = {
    "state": {"rtol": 3e-5, "atol": 3e-6},
    "kv": {"rtol": .008, "atol": 1e-4},
    "gradient": {"rtol": .03, "atol": 2e-5},
}


def compare_tensor(actual, expected, *, name, tolerance, errors):
    a, b = actual.detach().float(), expected.detach().float()
    maximum = float((a - b).abs().max().item()) if a.numel() else 0.0
    errors[name] = maximum if math.isfinite(maximum) else None
    torch.testing.assert_close(actual, expected, **tolerance, msg=name)


def compare_states(actual, expected, *, name, errors):
    assert len(actual) == len(expected), name
    for index, (left, right) in enumerate(zip(actual, expected)):
        left.validate()
        assert left.c.dtype == left.d.dtype == torch.float32
        assert left.k.dtype == left.v.dtype == torch.bfloat16
        for field in STATE_FIELDS:
            a, b = getattr(left, field), getattr(right, field)
            if field in ("c", "d"):
                tolerance = TOLERANCES["state"]
            elif field in ("k", "v"):
                a, b = a.float(), b.float()
                tolerance = TOLERANCES["kv"]
            else:
                tolerance = {"rtol": 0, "atol": 0}
            compare_tensor(a, b, name=f"{name}/{index}/{field}", tolerance=tolerance, errors=errors)


def observations(device):
    generator = torch.Generator().manual_seed(91017)
    schedule = []
    for index in range(4):
        images = torch.rand(1, 2, 3, 16, 24, generator=generator).to(device)
        mask = torch.ones(1, 2, 16, 24, dtype=torch.bool, device=device)
        mask[..., 3:10, 5:13] = False
        images = images.masked_fill(~mask.unsqueeze(2), float("nan"))
        times = torch.tensor([[index + .25, index + .5]], dtype=torch.float64, device=device)
        schedule.append(ObservationStep(images, mask, times))
    return tuple(schedule)


def query_at(time_value, device):
    return DecoderQuery(
        torch.tensor([[.5, .5, 0.], [7.5, 5.5, 0.], [23.5, 15.5, 0.]], device=device),
        torch.zeros(3, dtype=torch.int64, device=device),
        torch.full((3,), time_value, dtype=torch.float64, device=device),
    )


def run_path(model, schedule, mode, query):
    model.zero_grad(set_to_none=True)
    initial = model.initial_state()
    if mode == "chunk":
        head = model.forward_sequence(initial, schedule[:2], method="associative")
        states = head + model.forward_sequence(head[-1], schedule[2:], method="associative")
    elif mode == "step":
        current, collected = initial, []
        for item in schedule:
            current = model.step(current, item)
            collected.append(current)
        states = tuple(collected)
    else:
        states = model.forward_sequence(initial, schedule, method=mode)
    decoded = model.decode(states[-1], query)
    loss = sum((index + 1) * (state.c.square().mean() + state.d.square().mean())
               for index, state in enumerate(states)) + decoded.square().mean()
    assert torch.isfinite(loss), f"{mode}: nonfinite loss"
    loss.backward()
    gradients = {}
    gradient_norms = {"observer": 0.0, "cells": 0.0, "decoder": 0.0}
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32, name
        if parameter.grad is None:
            continue  # The unused modality embedding can legitimately be idle.
        assert parameter.grad.dtype == torch.float32 and torch.isfinite(parameter.grad).all(), name
        gradients[name] = parameter.grad.detach().cpu().clone()
        group = name.split(".", 1)[0]
        if group in gradient_norms:
            gradient_norms[group] += float(parameter.grad.detach().abs().sum().item())
    assert all(value > 0 for value in gradient_norms.values()), gradient_norms
    return tuple(state.snapshot() for state in states), decoded.detach().clone(), gradients, gradient_norms


def check_precision(device, spatial_bf16, report):
    torch.manual_seed(20261017)
    model = HybridCDModel(HybridConfig.tiny(spatial_blocks=4)).to(device).eval()
    model.observer.spatial_autocast = spatial_bf16
    schedule = observations(device)
    query = query_at(3.5, device)
    errors = report["maximum_absolute_errors"]
    evidence, _ = model.observer.forward_batch([schedule[0]])
    assert evidence.dtype == torch.float32 and torch.isfinite(evidence).all()
    report["dtypes"] = {"parameters": "float32", "evidence": str(evidence.dtype),
                        "c_d": "float32", "kv": "bfloat16"}

    reference, reference_rgb, reference_gradients, norms = run_path(model, schedule, "sequential", query)
    report["gradient_l1_by_module"] = {"sequential": norms}
    for mode in ("associative", "chunk", "step"):
        states, rgb, gradients, norms = run_path(model, schedule, mode, query)
        compare_states(states, reference, name=mode, errors=errors)
        compare_tensor(rgb, reference_rgb, name=f"{mode}/decoded", tolerance=TOLERANCES["state"], errors=errors)
        assert gradients.keys() == reference_gradients.keys(), f"{mode}: gradient participation differs"
        for name, gradient in gradients.items():
            compare_tensor(gradient, reference_gradients[name], name=f"{mode}/gradient/{name}",
                           tolerance=TOLERANCES["gradient"], errors=errors)
        report["gradient_l1_by_module"][mode] = norms

    with torch.no_grad():
        # Serialize the complete declared C/D + KV payload through weights-only
        # loading. A fresh model object proves no module-owned cache is needed.
        digest = weights_fingerprint(model.state_dict())
        buffer = io.BytesIO()
        torch.save(reference[1].to_payload(weights_hash=digest), buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=True, map_location=device)
        restored = HybridCDKVState.from_payload(payload, config=model.config, weights_hash=digest)
        fresh = HybridCDModel(model.config).to(device).eval()
        fresh.load_state_dict(model.state_dict())
        fresh.observer.spatial_autocast = spatial_bf16
        continued = fresh.forward_sequence(restored, schedule[2:], method="associative")
        compare_states(continued, reference[2:], name="restart", errors=errors)
        compare_tensor(fresh.decode(continued[-1], query), reference_rgb, name="restart/decoded",
                       tolerance=TOLERANCES["state"], errors=errors)

        changed = schedule[:2] + tuple(replace(
            item, images=torch.nan_to_num(item.images) * 100,
            observed=~item.observed, frame_times=item.frame_times + 20,
        ) for item in schedule[2:])
        altered = model.forward_sequence(model.initial_state(), changed, method="associative")
        compare_states(altered[:2], reference[:2], name="future_causality", errors=errors)

        filled = tuple(replace(item, images=torch.nan_to_num(item.images, nan=12345.)) for item in schedule)
        hidden = model.forward_sequence(model.initial_state(), filled, method="associative")
        compare_states(hidden, reference, name="hidden_pixels", errors=errors)

        future_times = torch.tensor([[4.5], [5.5]], dtype=torch.float64, device=device)
        rolled = model.rollout(reference[-1], future_times, method="associative")
        missing = tuple(ObservationStep(
            torch.full_like(schedule[0].images, float("nan")), torch.zeros_like(schedule[0].observed),
            torch.tensor([[index + .25, index + .5]], dtype=torch.float64, device=device),
        ) for index in (4, 5))
        explicit = model.forward_sequence(reference[-1], missing, method="associative")
        compare_states(rolled, explicit, name="missing_rollout", errors=errors)
        assert torch.isfinite(model.decode(rolled[-1], query_at(5.5, device))).all()
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This gate requires an allocated CUDA/ROCm GPU")
    if not 0 <= args.device_index < torch.cuda.device_count():
        parser.error("--device-index is outside the visible GPU range")
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device_index)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", args.device_index)
    started = time.monotonic()
    result = {"status": "passed", "device": torch.cuda.get_device_name(device),
              "visible_gpu_count": torch.cuda.device_count(), "torch": torch.__version__,
              "hip": torch.version.hip, "tolerances": TOLERANCES, "precision_modes": []}
    for spatial_bf16 in (False, True):
        report = {"spatial_bf16": spatial_bf16, "status": "running", "maximum_absolute_errors": {}}
        result["precision_modes"].append(report)
        try:
            check_precision(device, spatial_bf16, report)
            torch.cuda.synchronize(device)
        except (AssertionError, RuntimeError, ValueError) as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            result["status"] = "failed"
    result["elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
