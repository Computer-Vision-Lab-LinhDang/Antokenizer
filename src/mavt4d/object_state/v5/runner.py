"""Resumable full-budget natural training, with measured physical microbatches.

No scheduler submission or credential discovery occurs here. The explicit run
lock pins source weights and inventory; missing sources are never substituted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from ..runner import RunLogger, RuntimeConfig
from .natural import NaturalSources, SUPPORTED_SOURCES, mixture_counts
from .training import clip_signature, compute_loss, curriculum, learning_rate


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _rng():
    result = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_initialized():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _optimizer(model):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": .05},
                             {"params": no_decay, "weight_decay": 0.}], lr=3e-4, betas=(.9, .95))


def validate_run_config(config):
    if config.get("schema") != "natural-v5-run-v1":
        raise ValueError("explicit versioned run config required")
    training = config["training"]
    effective, updates = training.get("effective_batch", 256), training.get("updates", 20000)
    if effective < 256 or updates != 20000:
        raise ValueError("v5 full run requires effective batch >=256 and exactly20000 updates")
    mixture_counts(effective, config["mixture"])
    configured = {s for s in SUPPORTED_SOURCES if s + "_manifest" in config["data"] or
                  (s == "nuscenes" and "nuscenes_root" in config["data"])}
    if configured != set(config["mixture"]) or config["data"].get("mixture", config["mixture"]) != config["mixture"]:
        raise ValueError("run mixture must match data sources and sampler weights exactly")


def implementation_digest():
    """Pin executed Python sources, including dirty/uncommitted implementations."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parents[2].rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).resolve().parents[2])).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _save(path, model, optimizer, *, lock, step, run_id, capacities):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    payload = {"schema": "natural-v5-run-v1", "lock": lock, "step": step,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "rng": _rng(), "wandb_run_id": run_id, "capacities": capacities}
    with temp.open("wb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CalibrationStopped(RuntimeError):
    """Stop requested before every source had a measured safe capacity."""


def measure_capacity(model, optimizer, dataset, *, device, rays, effective, sources, logger,
                     calibration_seconds=900., deadline=None, should_stop=None):
    """Measure mature G/H6/bridge backwards + Adam, then restore training state.

    Probe every source at batch one before round-robin growth. Search is bounded
    by each source's logical draw count and a wall-time budget checked between
    probes; a single in-flight probe cannot be preempted safely. CPU is only a
    correctness path. No unmeasured capacity is substituted on timeout or OOM.
    """
    if not math.isfinite(calibration_seconds) or calibration_seconds <= 0:
        raise ValueError("calibration_seconds must be finite and positive")
    limits = mixture_counts(effective, sources)
    end = time.monotonic() + calibration_seconds
    if deadline is not None:
        end = min(end, deadline)
    should_stop = should_stop or (lambda: False)
    candidates = {}
    for source, limit in limits.items():
        powers = [2**i for i in range(limit.bit_length())]
        candidates[source] = sorted(set(powers + [limit])) if device.type == "cuda" else [1]
    saved_model, saved_optimizer, saved_rng = _cpu_state(model), copy.deepcopy(optimizer.state_dict()), _rng()
    capacities, records, best_rates, clips = {}, [], {}, {}
    blocked = set()

    def boundary_reached():
        stopped = should_stop()
        if not stopped and time.monotonic() < end:
            return False
        missing = [name for name in sources if name not in capacities]
        if missing:
            if stopped:
                raise CalibrationStopped("calibration stopped before measuring all sources")
            raise RuntimeError("calibration budget expired: no safe measured microbatch for " + ", ".join(missing))
        return True

    try:
        for index in range(max(map(len, candidates.values()))):
            for source in sources:
                if source in blocked or index >= len(candidates[source]):
                    continue
                if boundary_reached():
                    return capacities, records
                if source not in clips:
                    clips[source] = dataset.draw(source, 20, profile="G", horizon=6, plan_seed=17)
                clip, candidate = clips[source], candidates[source][index]
                # A slow decode must not bypass the stop/deadline check.
                if boundary_reached():
                    return capacities, records
                model.load_state_dict(saved_model); optimizer.load_state_dict(copy.deepcopy(saved_optimizer)); _restore_rng(saved_rng)
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
                started = time.monotonic()
                try:
                    loss, _ = compute_loss(model, [clip] * candidate, step=4000, rays=rays, seed=20, bridge=True)
                    if not math.isfinite(float(loss.detach())):
                        raise FloatingPointError("nonfinite loss in capacity probe")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not math.isfinite(float(norm)):
                        raise FloatingPointError("nonfinite gradient in capacity probe")
                    optimizer.step()
                    _sync(device)
                    elapsed = max(time.monotonic() - started, 1e-9)
                    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                    total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 1
                    eligible = reserved <= .9 * total
                    rate = candidate / elapsed
                    records.append({"source": source, "microbatch": candidate, "seconds": elapsed,
                                    "peak_reserved": reserved, "eligible": eligible, "draws_per_second": rate,
                                    "realized_profile": clip["plan"]["profile"], "future_targets": len(clip["plan"]["future"])})
                    logger.log({f"capacity/{source}/microbatch": candidate, f"capacity/{source}/draws_per_second": rate,
                                f"capacity/{source}/peak_reserved": reserved}, step=0)
                    if eligible and rate > best_rates.get(source, -1.):
                        capacities[source], best_rates[source] = candidate, rate
                    if not eligible:
                        blocked.add(source)
                except torch.OutOfMemoryError:
                    records.append({"source": source, "microbatch": candidate, "oom": True})
                    blocked.add(source)
                finally:
                    optimizer.zero_grad(set_to_none=True)
                if source not in capacities:
                    raise RuntimeError(f"no safe measured microbatch for {source}")
    finally:
        model.load_state_dict(saved_model); optimizer.load_state_dict(copy.deepcopy(saved_optimizer)); _restore_rng(saved_rng)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return capacities, records


def _draw_chunk(dataset, source, seeds, profile, horizon, plan_seed, pool):
    futures = [pool.submit(dataset.draw, source, seed, profile=profile, horizon=horizon, plan_seed=plan_seed) for seed in seeds]
    clips = [future.result() for future in futures]
    buckets = defaultdict(list)
    for clip in clips:
        buckets[clip_signature(clip)].append(clip)
    return list(buckets.values())


def train_update(model, optimizer, dataset, *, step, effective, capacities, weights, seed, rays, pool):
    counts = mixture_counts(effective, weights)
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step)
    metrics = defaultdict(float)
    actual_batches, total_loss, decode_seconds = 0, 0., 0.
    for source_index, (source, count) in enumerate(counts.items()):
        metrics[f"source/{source}/draws"] = count
        profile, horizon, _ = curriculum(step, seed + step * 103 + source_index)
        micro = min(capacities[source], count)
        for start in range(0, count, micro):
            # Counter-based seed is the sampler cursor; retry/resume repeats exactly.
            seeds = [seed + step * 1000003 + source_index * 10007 + j for j in range(start, min(start + micro, count))]
            decode_start = time.monotonic()
            buckets = _draw_chunk(dataset, source, seeds, profile, horizon, step, pool)
            decode_seconds += time.monotonic() - decode_start
            for clips in buckets:
                loss, current = compute_loss(model, clips, step=step, rays=rays, seed=seed + step, bridge=True)
                weight = weights[source] * len(clips) / count
                (loss * weight).backward()
                total_loss += float(loss.detach()) * weight
                for key, value in current.items():
                    metrics[f"source/{source}/{key}"] += float(value) * len(clips) / count
                actual_batches += 1
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not math.isfinite(float(norm)) or not math.isfinite(total_loss):
        raise FloatingPointError("nonfinite objective/gradient; optimizer update not committed")
    optimizer.step()
    return {**metrics, "loss": total_loss, "optimizer/grad_norm": float(norm), "optimizer/lr": optimizer.param_groups[0]["lr"],
            "batch/effective": effective, "batch/physical_mean": effective / actual_batches,
            "timing/decode_seconds": decode_seconds}


@torch.no_grad()
def evaluate(model, dataset, *, step, rays, seed, weights):
    was_training = model.training
    model.eval()
    values = {}
    try:
        for source in weights:
            for profile in ("S", "L", "G") if source != "openimages" else ("S",):
                clip = dataset.draw(source, seed, profile=profile, horizon=6, plan_seed=17)
                loss, metrics = compute_loss(model, [clip], step=4000, rays=rays, seed=seed, bridge=False)
                values[f"dev/{source}/{profile}/loss"] = float(loss)
                values.update({f"dev/{source}/{profile}/{k}": v for k, v in metrics.items()})
    finally:
        model.train(was_training)
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allocation-seconds", type=int, default=42600)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    from .config import V5Config
    from .model import V5Model
    config = json.loads(args.config.read_text())
    validate_run_config(config)
    # Full budget is never automatically shortened after a throughput measurement.
    training = config["training"]
    effective, updates = training.get("effective_batch", 256), training.get("updates", 20000)
    if effective < 256 or updates != 20000:
        raise ValueError("v5 full run requires effective batch >=256 and exactly20000 updates")
    weights = dict(config["mixture"])
    mixture_counts(effective, weights)
    device = torch.device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise ValueError("training requires allocated CUDA; CPU only for explicit correctness checks")
    if not os.environ.get("WANDB_API_KEY") or os.environ.get("MAVT_WANDB_CREDENTIAL_OWNER") != config["wandb"]["entity"]:
        raise RuntimeError("approved process W&B credential required")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError("online W&B required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and args.resume is None:
        raise FileExistsError("existing run requires explicit --resume")
    seed, rays = training.get("seed", 17), training.get("rays", 256)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.set_num_threads(training.get("cpu_threads", 4))
    data_config = {**config["data"], "mixture": weights}
    dataset = NaturalSources(data_config, split="train")
    # Separate dev group indexes. Metadata is loaded once per split; this is a
    # known host-memory cost to measure before deployment, not a GPU allocation.
    dev = NaturalSources(data_config, split="dev")
    lock = {"schema": config["schema"], "implementation_stage": "v5-implementation",
            "model": config.get("model", {}), "training": training,
            "mixture": weights, "inventory": dataset.inventory(), "dev_inventory": dev.inventory(),
            "implementation_digest": implementation_digest(), "revision": config.get("revision"),
            "claim": "configured-source-coverage-only; academic G3 requires frozen val and trained baselines"}
    lock["inventory"]["mixture"] = weights
    lock["dev_inventory"]["mixture"] = weights
    lock_path = output / "run-lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("run lock changed; refusing incompatible resume")
    else:
        _atomic_json(lock_path, lock)
    model = V5Model(V5Config(**config.get("model", {}))).to(device)
    optimizer = _optimizer(model)
    step, run_id, capacities = 0, None, None
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("schema") != config["schema"] or saved["lock"] != lock:
            raise ValueError("checkpoint schema/lock mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"])
        step, run_id, capacities = saved["step"], saved["wandb_run_id"], saved["capacities"]
    os.environ["WANDB_ENTITY"] = config["wandb"]["entity"]
    logger = RunLogger(output, config=lock, runtime=RuntimeConfig(project=config["wandb"]["project"],
                       run_name=config["wandb"].get("name", "MAVT-v5-full")), enabled=True, run_id=run_id)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    deadline = time.monotonic() + args.allocation_seconds
    exit_code, reason = 1, "error"
    try:
        if capacities is None:
            capacities, probes = measure_capacity(model, optimizer, dataset, device=device, rays=rays,
                                                    effective=effective, sources=weights, logger=logger,
                                                    calibration_seconds=training.get("calibration_seconds", 900.),
                                                    deadline=deadline - 300., should_stop=lambda: stop[0])
            _atomic_json(output / "capacity.json", {"microbatches": capacities, "probes": probes})
        _save(output / "last.pt", model, optimizer, lock=lock, step=step, run_id=logger.run_id, capacities=capacities)
        estimate = 60.
        with ThreadPoolExecutor(max_workers=training.get("decode_workers", 4)) as pool:
            while step < updates and not stop[0] and time.monotonic() + max(300., estimate * 1.5) < deadline:
                started = time.monotonic()
                rng_before = _rng()
                try:
                    metrics = train_update(model, optimizer, dataset, step=step, effective=effective,
                                           capacities=capacities, weights=weights, seed=seed, rays=rays, pool=pool)
                except torch.OutOfMemoryError:
                    # No Adam call occurred until all microbatches finished. Restore
                    # full boundary, halve physical batches, retain logical objective.
                    saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
                    model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
                    optimizer.zero_grad(set_to_none=True); _restore_rng(rng_before)
                    if all(v == 1 for v in capacities.values()):
                        raise
                    capacities = {key: max(1, value // 2) for key, value in capacities.items()}
                    torch.cuda.empty_cache()
                    logger.log({"runtime/oom_retry": 1}, step=step)
                    continue
                step += 1
                _sync(device)
                elapsed = time.monotonic() - started
                estimate = max(estimate * .9, elapsed)
                _save(output / "last.pt", model, optimizer, lock=lock, step=step, run_id=logger.run_id, capacities=capacities)
                if step % 500 == 0:
                    snapshot = output / f"checkpoint-{step:06d}.pt"
                    if snapshot.exists():
                        raise FileExistsError(snapshot)
                    _save(snapshot, model, optimizer, lock=lock, step=step, run_id=logger.run_id, capacities=capacities)
                logger.log({**metrics, "timing/update_seconds": elapsed, "progress/draws": step * effective,
                            "progress/remaining_updates": updates - step}, step=step)
                if step % training.get("eval_every", 500) == 0 and time.monotonic() + 600 < deadline:
                    logger.log(evaluate(model, dev, step=step, rays=rays, seed=seed + 500, weights=weights), step=step)
        reason = "complete" if step == updates else "resume_required"
        _atomic_json(output / "summary.json", {"status": reason, "updates": step, "target_updates": updates,
                                               "effective_batch": effective, "wandb_url": logger.url,
                                               "checkpoint": "last.pt", "mixture": weights})
        exit_code = 0
    except CalibrationStopped:
        reason, exit_code = "calibration_stopped", 0
        _atomic_json(output / "summary.json", {"status": reason, "updates": step,
                                               "target_updates": updates, "effective_batch": effective,
                                               "wandb_url": logger.url, "checkpoint": None,
                                               "mixture": weights})
    finally:
        logger.finish(exit_code=exit_code)
    print(json.dumps({"status": reason, "updates": step, "wandb_url": logger.url}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
