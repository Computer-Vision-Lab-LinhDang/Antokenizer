"""Resumable full-budget natural training, with measured physical microbatches.

No scheduler submission or credential discovery occurs here. The explicit run
lock pins source weights and inventory; missing sources are never substituted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict, dataclass, field
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


def logical_draw_plan(*, step, effective, weights, seed):
    """Counter-addressed choices, created before decoding or physical batching."""
    result = []
    for source, count in sorted(mixture_counts(effective, dict(sorted(weights.items()))).items()):
        for index in range(count):
            draw_id = f"v5-draw-v2:{seed}:{step}:{source}:{index}"
            draw_seed = int.from_bytes(hashlib.sha256(draw_id.encode()).digest()[:8], "big")
            if source == "openvid":
                # Even seeds select CelebV. Floor-half cap also holds for odd counts.
                draw_seed = (draw_seed & ~1) | ((index + 1) % 2)
            profile, horizon, _ = curriculum(step, draw_seed)
            result.append(dict(draw_id=draw_id, source=source, seed=draw_seed,
                               plan_seed=draw_seed, profile=profile, horizon=horizon))
    return result


def validate_capacity_clip(source, clip, *, profile="G", horizon=6):
    """A requested profile is not evidence that the adapter realized it."""
    plan = clip["plan"]
    future = len(plan.get("future", []))
    if source == "openimages":
        valid = plan["profile"] == "image"
    elif not clip.get("time_valid", True):
        valid = (plan["profile"] in (profile, profile + "-views") and sum(plan.get("observed", [])) == (3 if profile == "G" else 2)
                 and bool(plan.get("heldout")))
        if profile == "G":
            valid = valid and plan.get("observed", []).count(False) == 3
    else:
        valid = plan["profile"] == profile and future == horizon
        observed = plan.get("observed", [])
        valid = valid and sum(observed) == (6 if profile == "G" else 3)
        if profile == "G":
            valid = valid and observed.count(False) == 6 and observed[0] and observed[-1]
    if not valid:
        raise ValueError(f"capacity for {source} did not realize {profile}/H{horizon} (or declared static/image maximum): {plan}")
    return {"realized_profile": plan["profile"], "future_targets": future,
            "heldout_views": len(plan.get("heldout", [])),
            "observed_steps": sum(plan.get("observed", []))}


@dataclass
class OwnershipMonitor:
    """Persist evidence for the predeclared fallback; missing diagnostics never trigger it."""
    bad_updates: int = 0
    diagnostics: list = field(default_factory=list)
    softened_at: int | None = None

    def update(self, entropy, gradient):
        bad = math.isfinite(entropy) and math.isfinite(gradient) and entropy < .1 and gradient < 1e-6
        self.bad_updates = self.bad_updates + 1 if bad else 0

    def record_diagnostic(self, assignment_accuracy):
        if math.isfinite(assignment_accuracy) and 0 <= assignment_accuracy <= 1:
            self.diagnostics = (self.diagnostics + [float(assignment_accuracy)])[-2:]

    @property
    def should_soften(self):
        return (self.softened_at is None and self.bad_updates >= 500 and len(self.diagnostics) == 2
                and self.diagnostics[-1] <= self.diagnostics[-2])

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def record_diagnostic_report(monitor, report):
    score = report.get("assignment_accuracy")
    if report.get("complete") and score is not None:
        monitor.record_diagnostic(float(score))


def validate_readiness(receipt, model_config, inventory=None):
    """Fail closed on missing launch evidence; preserve unmeasured source scope."""
    if receipt.get("schema") != "v5-readiness-v1":
        raise ValueError("v5 readiness receipt required before full training")
    census = receipt.get("slot_census", {})
    n = census.get("total_steps", 0)
    over = census.get("steps_over_16", -1)
    if n <= 0 or not 0 <= over <= n or not census.get("input_views"):
        raise ValueError("measured annotation frustum slot census required")
    from mavt4d.data.nuscenes import CAM6
    if census.get("schema") != "v5-slot-census-v2" or census.get("input_distribution") != "uniform_single_camera" or set(census["input_views"]) != set(CAM6):
        raise ValueError("slot census must match the uniform single-input-camera distribution over all six cameras")
    expected = (32, 4, 4) if over / n > .2 else (16, 8, 8)
    actual = tuple(model_config.get(k, v) for k, v in zip(
        ("slots", "content_tokens", "dynamic_tokens"), (16, 8, 8)))
    if actual != expected:
        raise ValueError(f"slot tier must follow locked census: expected {expected}, got {actual}")
    if inventory is not None:
        manifests = inventory["manifest_sha256"]
        if census.get("metadata_sha256") != manifests.get("nuscenes"):
            raise ValueError("slot census metadata differs from frozen training inventory")
        measured_hashes = receipt.get("role_coverage", {}).get("manifest_sha256", {})
        for source in receipt.get("role_coverage", {}).get("sources", {}):
            if measured_hashes.get(source) != manifests.get(source) or source not in manifests:
                raise ValueError("role coverage inventory differs from frozen training source")
    sources = receipt.get("role_coverage", {}).get("sources", {})
    trusted = [r for r in sources.values() if r.get("trusted_draws", 0) > 0]
    if not trusted or not any(r.get("reuse_valid", 0) > 0 and r.get("canonical_valid", 0) > 0 for r in trusted):
        raise ValueError("measured real-data reuse/canonical supervision coverage required")


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


def _save(path, model, optimizer, *, lock, step, run_id, capacities, ownership=None):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    payload = {"schema": "natural-v5-run-v1", "lock": lock, "step": step,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "rng": _rng(), "wandb_run_id": run_id, "capacities": capacities,
               "ownership_monitor": ownership.to_dict() if ownership is not None else OwnershipMonitor().to_dict()}
    with temp.open("wb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CalibrationStopped(RuntimeError):
    """Stop requested before every source had a measured safe capacity."""


def capacity_probe_batch(clip, candidate, *, mask):
    """Exercise all possible hidden-render endpoints in a physical microbatch."""
    plan = clip["plan"]
    observed = [frame for frame, admitted in zip(plan.get("prefix", []), plan.get("observed", [])) if admitted]
    force_mask = mask and clip.get("time_valid", True) and len(observed) >= 2
    result = []
    for index in range(candidate):
        probe = {**clip, "draw_seed": (int(clip.get("draw_seed", 20)) + index) % (2**63),
                 "force_bridge": True, "force_tube_mask": force_mask}
        if force_mask:
            probe["force_tube_frame"] = observed[index % len(observed)]
        result.append(probe)
    return result


def capacity_for(capacities, source, *, step):
    measured = capacities[source]
    if isinstance(measured, int):
        return measured  # Legacy checkpoint/test compatibility, not new production receipt.
    return measured["S/H1/bridge" if step < 1000 else "G/H6/bridge"]


def halve_capacities(capacities):
    return {source: ({key: max(1, value // 2) for key, value in measured.items()} if isinstance(measured, dict)
                     else max(1, measured // 2)) for source, measured in capacities.items()}


def _capacity_values(capacities):
    return [value for measured in capacities.values() for value in
            (measured.values() if isinstance(measured, dict) else [measured])]


def measure_capacity(model, optimizer, dataset, *, device, rays, effective, sources, logger,
                     calibration_seconds=900., deadline=None, should_stop=None, bucketed=False, workload="G/H6/bridge"):
    if bucketed:
        # Two measured workloads. Intermediate schedules use the mature upper
        # bound rather than pretending an unmeasured bucket has been calibrated.
        capacities, records = {source: {} for source in sources}, []
        for bucket in ("G/H6/bridge", "S/H1/bridge"):
            measured, probes = measure_capacity(model, optimizer, dataset, device=device, rays=rays,
                effective=effective, sources=sources, logger=logger, calibration_seconds=calibration_seconds / 2,
                deadline=deadline, should_stop=should_stop, bucketed=False, workload=bucket)
            for source, value in measured.items():
                capacities[source][bucket] = value
            records.extend(probes)
        return capacities, records
    return _measure_capacity_workload(model, optimizer, dataset, device=device, rays=rays,
        effective=effective, sources=sources, logger=logger, calibration_seconds=calibration_seconds,
        deadline=deadline, should_stop=should_stop, workload=workload)


def _measure_capacity_workload(model, optimizer, dataset, *, device, rays, effective, sources, logger,
                     calibration_seconds=900., deadline=None, should_stop=None, workload="G/H6/bridge"):

    """Measure mature G/H6/bridge backwards + Adam, then restore training state.

    Probe every source at batch one before round-robin growth. Search is bounded
    by each source's logical draw count and a wall-time budget checked between
    probes; a single in-flight probe cannot be preempted safely. CPU is only a
    correctness path. No unmeasured capacity is substituted on timeout or OOM.
    """
    if not math.isfinite(calibration_seconds) or calibration_seconds <= 0:
        raise ValueError("calibration_seconds must be finite and positive")
    profile, horizon = ("S", 1) if workload == "S/H1/bridge" else ("G", 6)
    probe_step = 0 if profile == "S" else 4000
    limits = mixture_counts(effective, dict(sorted(sources.items())))
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
                    capacity_draw = getattr(type(dataset), "capacity_draw", None)
                    clips[source] = (dataset.capacity_draw(source, 20, horizon=horizon, profile=profile) if capacity_draw is not None
                                     else dataset.draw(source, 20, profile=profile, horizon=horizon, plan_seed=17))
                    validate_capacity_clip(source, clips[source], profile=profile, horizon=horizon)
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
                    probe_clips = capacity_probe_batch(clip, candidate, mask=profile == "G")
                    expected_mask_endpoints = len({p["force_tube_frame"] for p in probe_clips if "force_tube_frame" in p})
                    loss, probe_metrics = compute_loss(model, probe_clips, step=probe_step, rays=rays, seed=20, bridge=True)
                    if "rgb" in clip and probe_metrics.get("tube_mask_distinct_endpoints") != expected_mask_endpoints:
                        raise RuntimeError("capacity probe did not realize worst-case distinct hidden-render endpoints")
                    if "intrinsics" in clip and probe_metrics.get("bridge_active") != 1:
                        raise RuntimeError("capacity probe failed to exercise measured geometry bridge")
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
                                    **validate_capacity_clip(source, clip, profile=profile, horizon=horizon), "workload": workload, "bridge_requested": True,
                                    "bridge_realized": float(probe_metrics.get("bridge_active", 0.)),
                                    "distinct_mask_endpoints": expected_mask_endpoints})
                    logger.log({f"capacity/{source}/{workload}/microbatch": candidate, f"capacity/{source}/{workload}/draws_per_second": rate,
                                f"capacity/{source}/{workload}/peak_reserved": reserved}, step=0)
                    if eligible and rate > best_rates.get(source, -1.):
                        capacities[source], best_rates[source] = candidate, rate
                    if not eligible:
                        blocked.add(source)
                except torch.OutOfMemoryError:
                    records.append({"source": source, "microbatch": candidate, "workload": workload, "oom": True})
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


def _draw_planned(dataset, choice):
    clip = dataset.draw(choice["source"], choice["seed"], profile=choice["profile"],
                        horizon=choice["horizon"], plan_seed=choice["plan_seed"])
    return {**clip, "draw_seed": choice["seed"], "draw_id": choice["draw_id"]}


def train_update(model, optimizer, dataset, *, step, effective, capacities, weights, seed, rays, pool):
    counts = mixture_counts(effective, dict(sorted(weights.items())))
    draw_plan = logical_draw_plan(step=step, effective=effective, weights=weights, seed=seed)
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate(step)
    metrics = defaultdict(float)
    actual_batches, total_loss, decode_seconds = 0, 0., 0.
    ownership_samples = []
    for source, count in counts.items():
        metrics[f"source/{source}/draws"] = count
        choices = [p for p in draw_plan if p["source"] == source]
        started = time.monotonic()
        # Bucket the entire logical source partition before physical slicing.
        # This avoids changing random choices or wasting capacity on small mixed chunks.
        clips = list(pool.map(lambda choice: _draw_planned(dataset, choice), choices))
        unique_groups = len({clip.get("group", clip["draw_id"]) for clip in clips})
        metrics[f"source/{source}/unique_groups"] = unique_groups
        metrics[f"source/{source}/repeat_draw_fraction"] = 1 - unique_groups / count
        buckets = defaultdict(list)
        for clip in clips:
            buckets[clip_signature(clip)].append(clip)
            metrics[f"source/{source}/profile/{clip['plan']['profile']}"] += 1
        decode_seconds += time.monotonic() - started
        micro = min(capacity_for(capacities, source, step=step), count)
        for bucket in buckets.values():
            for start in range(0, len(bucket), micro):
                batch = bucket[start:start + micro]
                loss, current = compute_loss(model, batch, step=step, rays=rays, seed=seed + step, bridge=True)
                weight = weights[source] * len(batch) / count
                (loss * weight).backward()
                total_loss += float(loss.detach()) * weight
                ownership_samples.extend(current.get("_ownership_samples", []))
                for key, value in current.items():
                    if not key.startswith("_"):
                        metrics[f"source/{source}/{key}"] += float(value) * len(batch) / count
                actual_batches += 1
    for key, output_key in (("entropy_values", "ownership/entropy_median"),
                            ("grad_values", "ownership/gradient_median")):
        values = [s[key].reshape(-1) for s in ownership_samples if s.get(key) is not None and s[key].numel()]
        if values:
            metrics[output_key] = float(torch.cat(values).median())
    metrics["ownership/gradient_available"] = float("ownership/gradient_median" in metrics)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not math.isfinite(float(norm)) or not math.isfinite(total_loss):
        raise FloatingPointError("nonfinite objective/gradient; optimizer update not committed")
    optimizer.step()
    return {**metrics, "loss": total_loss, "optimizer/grad_norm": float(norm), "optimizer/lr": optimizer.param_groups[0]["lr"],
            "batch/effective": effective, "batch/physical_mean": effective / actual_batches,
            "sampler/plan_sha256": hashlib.sha256(json.dumps(draw_plan, sort_keys=True).encode()).hexdigest(),
            "timing/decode_seconds": decode_seconds}


@torch.no_grad()
def evaluate(model, dataset, *, step, rays, seed, weights, seconds=60.):
    was_training = model.training
    model.eval()
    values = {}
    started = time.monotonic()
    completed = 0
    try:
        for source in weights:
            for profile in ("S", "L", "G") if source != "openimages" else ("S",):
                if time.monotonic() - started >= seconds:
                    values["dev/evaluated_cases"] = completed
                    values["dev/complete"] = 0
                    return values
                clip = dataset.draw(source, seed, profile=profile, horizon=6, plan_seed=17)
                loss, metrics = compute_loss(model, [clip], step=4000, rays=rays, seed=seed, bridge=False)
                completed += 1
                values[f"dev/{source}/{profile}/loss"] = float(loss)
                values.update({f"dev/{source}/{profile}/{k}": v for k, v in metrics.items() if not k.startswith("_")})
    finally:
        model.train(was_training)
    values["dev/evaluated_cases"] = completed
    values["dev/complete"] = 1
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
    validate_readiness(config.get("readiness", {}), config.get("model", {}))
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
    validate_readiness(config["readiness"], config.get("model", {}), dataset.inventory())
    suite_path = Path(training.get("diagnostic_suite", "")) / "manifest.json"
    if not suite_path.is_file():
        raise ValueError("CPU-prepared fixed gap diagnostic_suite required before full run")
    diagnostic_sha256 = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    lock = {"schema": config["schema"], "implementation_stage": "v5-implementation",
            "model": config.get("model", {}), "training": training,
            "mixture": weights, "readiness": config["readiness"],
            "sampler": "v5-draw-v2", "diagnostic_suite_sha256": diagnostic_sha256,
            "logical_source_counts": mixture_counts(effective, dict(sorted(weights.items()))),
            "ownership_fallback": {"entropy": .1, "gradient": 1e-6, "updates": 500, "temperature": 2.},
            "inventory": dataset.inventory(), "dev_inventory": dev.inventory(),
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
    ownership = OwnershipMonitor()
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("schema") != config["schema"] or saved["lock"] != lock:
            raise ValueError("checkpoint schema/lock mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"])
        step, run_id, capacities = saved["step"], saved["wandb_run_id"], saved["capacities"]
        ownership = OwnershipMonitor.from_dict(saved.get("ownership_monitor", {}))
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
                                                    deadline=deadline - 300., should_stop=lambda: stop[0], bucketed=True)
            _atomic_json(output / "capacity.json", {"microbatches": capacities, "probes": probes})
        _save(output / "last.pt", model, optimizer, lock=lock, step=step, run_id=logger.run_id, capacities=capacities, ownership=ownership)
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
                    if all(v == 1 for v in _capacity_values(capacities)):
                        raise
                    capacities = halve_capacities(capacities)
                    torch.cuda.empty_cache()
                    logger.log({"runtime/oom_retry": 1}, step=step)
                    continue
                step += 1
                ownership.update(metrics.get("ownership/entropy_median", float("nan")),
                                 metrics.get("ownership/gradient_median", float("nan")))
                if ownership.should_soften:
                    model.resampler.temperature.fill_(2.)
                    ownership.softened_at = step
                    _atomic_json(output / f"ownership-fallback-{step:06d}.json", ownership.to_dict())
                metrics["ownership/consecutive_bad_updates"] = ownership.bad_updates
                metrics["ownership/temperature"] = float(model.resampler.temperature)
                metrics["ownership/diagnostic_passes"] = len(ownership.diagnostics)
                _sync(device)
                elapsed = time.monotonic() - started
                estimate = max(estimate * .9, elapsed)
                _save(output / "last.pt", model, optimizer, lock=lock, step=step, run_id=logger.run_id, capacities=capacities, ownership=ownership)
                logger.log({**metrics, "timing/update_seconds": elapsed, "progress/draws": step * effective,
                            "progress/remaining_updates": updates - step}, step=step)
                if step % training.get("eval_every", 500) == 0 and time.monotonic() + 600 < deadline:
                    from .diagnostic import evaluate_gap_diagnostic
                    dev_started, dev_rng = time.monotonic(), _rng()
                    natural_seconds = training.get("natural_eval_seconds", 60.)
                    diagnostic_seconds = training.get("diagnostic_seconds", 60.)
                    if not (0 < natural_seconds <= 60 and 0 < diagnostic_seconds <= 60):
                        raise ValueError("natural and diagnostic dev budgets must each be in (0, 60] seconds")
                    try:
                        if natural_seconds > 0:
                            logger.log(evaluate(model, dev, step=step, rays=rays, seed=seed + 500,
                                                weights=weights, seconds=natural_seconds), step=step)
                        if diagnostic_seconds > 0:
                            report = evaluate_gap_diagnostic(model, seed=training.get("diagnostic_seed", 17),
                                seconds=diagnostic_seconds, suite_dir=training["diagnostic_suite"],
                                batch_size=training.get("diagnostic_batch_size", 32))
                            if report.get("suite_sha256") != diagnostic_sha256:
                                raise ValueError("diagnostic suite changed after run lock")
                            record_diagnostic_report(ownership, report)
                            _atomic_json(output / f"diagnostic-{step:06d}.json", report)
                            logger.log({"diagnostic/" + key: value for key, value in report.items()
                                        if isinstance(value, (int, float, bool))}, step=step)
                    finally:
                        _restore_rng(dev_rng)
                    logger.log({"dev/total_seconds": time.monotonic() - dev_started}, step=step)
                    # Diagnostics are part of the resumable fallback decision.
                    _save(output / "last.pt", model, optimizer, lock=lock, step=step,
                          run_id=logger.run_id, capacities=capacities, ownership=ownership)
                if step % 500 == 0:
                    snapshot = output / f"checkpoint-{step:06d}.pt"
                    if snapshot.exists():
                        raise FileExistsError(snapshot)
                    _save(snapshot, model, optimizer, lock=lock, step=step, run_id=logger.run_id, capacities=capacities, ownership=ownership)
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
