"""Single-device capacity sizing, training, and internal evaluation entry point.

All paths are supplied by the caller. Production requires one visible CUDA
device and online W&B. Capacity trials are real optimizer updates, rolled back
before training. Checkpoints resume deterministic source order and RNG state;
cached sources have no stochastic worker augmentation.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import importlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, default_collate

from .config import ObjectStateConfig
from .data import CachedFactorialDataset, FactorialSceneConfig, flatten_source_batch
from .evaluation import evaluate
from .model import ObjectStateModel
from .training import (
    Deadline,
    TrainingConfig,
    compute_training_loss,
    microbatch_slices,
    restore_checkpoint,
    save_checkpoint,
    tree_to,
)


@dataclass(frozen=True)
class RuntimeConfig:
    eval_sources: int = 64
    periodic_eval_sources: int = 2
    eval_rays: int = 256
    periodic_eval_seconds: float = 180.0
    run_name: str = "MAVT-object-state-full"
    project: str = "semantic-router"
    capacity_fraction: float = 0.9

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.eval_sources, self.periodic_eval_sources, self.eval_rays)
        ):
            raise ValueError("evaluation counts must be positive integers")
        if not 0 < self.capacity_fraction <= 0.9 or self.periodic_eval_seconds <= 0:
            raise ValueError(
                "capacity must leave at least 10% VRAM and evaluation needs a deadline"
            )


class SourceBatchSampler(Sampler[list[int]]):
    """Deterministic source order; cursor counts consumed sources, not prefetches."""

    def __init__(
        self,
        n_sources: int,
        *,
        seed: int,
        cursor: int,
        updates: int,
        effective_groups: int,
        micro_groups: int,
    ):
        if (
            min(n_sources, effective_groups, micro_groups) < 1
            or min(cursor, updates) < 0
        ):
            raise ValueError("invalid source sampler dimensions")
        if n_sources % effective_groups or cursor % effective_groups:
            raise ValueError("effective_groups must divide n_sources and resume cursor")
        self.n_sources, self.seed, self.cursor, self.updates = (
            n_sources,
            seed,
            cursor,
            updates,
        )
        self.effective_groups, self.micro_groups = effective_groups, micro_groups

    def __len__(self):
        return self.updates * math.ceil(self.effective_groups / self.micro_groups)

    def __iter__(self) -> Iterator[list[int]]:
        cached_epoch, permutation = -1, None
        cursor = self.cursor
        for _ in range(self.updates):
            for start, end, _ in microbatch_slices(
                self.effective_groups, self.micro_groups
            ):
                selected = []
                for _ in range(end - start):
                    epoch, offset = divmod(cursor, self.n_sources)
                    if epoch != cached_epoch:
                        generator = torch.Generator().manual_seed(self.seed + epoch)
                        permutation = torch.randperm(
                            self.n_sources, generator=generator
                        ).tolist()
                        cached_epoch = epoch
                    selected.append(permutation[offset])
                    cursor += 1
                yield selected


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_metrics(device) -> dict:
    if device.type != "cuda":
        return {"device_is_cuda": 0}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device_is_cuda": 1,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def run_update(
    model,
    optimizer,
    batches: Iterator,
    config: TrainingConfig,
    *,
    micro_groups: int,
    step: int,
    device: torch.device,
    materialize_optimizer_state: bool = False,
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.monotonic()
    metrics: dict[str, float] = {}
    for start, end, weight in microbatch_slices(config.effective_groups, micro_groups):
        collated = next(batches)
        if len(collated["metadata"]["source_uid"]) != end - start:
            raise ValueError(
                "loader microbatch does not match source accumulation plan"
            )
        batch = tree_to(flatten_source_batch(collated), device)
        loss, terms, _ = compute_training_loss(model, batch, config, step=step)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss; update was not applied")
        (loss * weight).backward()
        for name, value in {"loss": loss.detach(), **terms}.items():
            metrics[name] = metrics.get(name, 0.0) + float(value) * weight
        del loss, terms, batch, collated
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.grad_clip, error_if_nonfinite=True
    )
    if materialize_optimizer_state:
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    elapsed = time.monotonic() - started
    metrics.update(
        {
            "gradient_norm": float(norm),
            "update_seconds": elapsed,
            "source_groups": config.effective_groups,
            "examples": config.effective_groups * 8,
            "micro_groups": micro_groups,
            "examples_per_second": config.effective_groups * 8 / max(elapsed, 1e-9),
            **memory_metrics(device),
        }
    )
    return metrics


def capacity_search(
    probe: Callable[[int], dict], maximum: int, *, capacity_fraction: float = 0.9
) -> dict:
    """One exponential/bracketed search; each group count is measured once."""
    if maximum < 1:
        raise ValueError("maximum capacity must be positive")
    measurements = []

    def test(groups):
        result = {"groups": groups, **probe(groups)}
        measurements.append(result)
        return result["safe"]

    if not test(1):
        raise RuntimeError("one source group cannot fit safely")
    low, high = 1, None
    candidate = min(2, maximum)
    while candidate > low:
        if not test(candidate):
            high = candidate
            break
        low = candidate
        if low == maximum:
            break
        candidate = min(maximum, low * 2)
        # A measured memory slope proposes a jump; acceptance still requires a
        # full gradient + optimizer update at the proposed group count.
        if len(measurements) == 2:
            first, second = measurements
            total = second.get("device_total_bytes")
            slope = second.get("peak_reserved_bytes", 0) - first.get(
                "peak_reserved_bytes", 0
            )
            if total and slope > 0:
                intercept = second["peak_reserved_bytes"] - 2 * slope
                estimate = math.floor((capacity_fraction * total - intercept) / slope)
                candidate = min(maximum, max(candidate, estimate))
    if high is not None:
        while high - low > 1:
            middle = (low + high) // 2
            if test(middle):
                low = middle
            else:
                high = middle
    return {
        "micro_groups": low,
        "measurements": measurements,
        "maximum_groups_considered": maximum,
    }


def _rng_state():
    result = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def _capacity_fixture(model, optimizer, *, seed: int):
    """All-slot mature-state stress fixture, rolled back even after an OOM."""
    initial_model = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    rng, training = _rng_state(), model.training
    original_initial = model.initial_state

    def mature_initial(batch_size, **kwargs):
        state = original_initial(batch_size, **kwargs)
        generator = torch.Generator(device=state.c.device).manual_seed(seed + 919)

        def nonzero(value):
            return (
                torch.randn(
                    value.shape,
                    device=value.device,
                    dtype=value.dtype,
                    generator=generator,
                )
                * 0.1
            )

        return replace(
            state,
            c=nonzero(state.c),
            d=nonzero(state.d),
            c_bg=nonzero(state.c_bg),
            slot_status=torch.ones_like(state.slot_status),
            slot_uid=torch.ones_like(state.slot_uid),
            p_exist=torch.full_like(state.p_exist, 0.95),
        )

    def reset():
        optimizer.zero_grad(set_to_none=True)
        model.load_state_dict(initial_model)
        optimizer.load_state_dict(initial_optimizer)
        _restore_rng(rng)
        if next(model.parameters()).is_cuda:
            torch.cuda.empty_cache()

    model.initial_state = mature_initial
    try:
        yield reset
    finally:
        model.initial_state = original_initial
        reset()
        model.train(training)


def measure_capacity(
    model, optimizer, dataset, config, runtime, *, device, logger, deadline
):
    if device.type != "cuda":
        raise ValueError("capacity mode requires actual CUDA measurements")
    if config.effective_groups > len(dataset):
        raise ValueError("capacity source groups must remain distinct")
    measured_times = []
    with _capacity_fixture(model, optimizer, seed=config.seed) as reset:

        def probe(groups):
            if not deadline.can_train():
                raise TimeoutError("capacity deadline reached before training cutoff")
            reset()
            collated = default_collate([dataset[i] for i in range(groups)])
            trial = replace(config, effective_groups=groups)
            try:
                metrics = run_update(
                    model,
                    optimizer,
                    iter([collated]),
                    trial,
                    micro_groups=groups,
                    step=0,
                    device=device,
                    materialize_optimizer_state=True,
                )
                safe = (
                    metrics["peak_reserved_bytes"]
                    <= runtime.capacity_fraction * metrics["device_total_bytes"]
                )
                result = {"safe": safe, **metrics}
                measured_times.append(metrics["update_seconds"] / groups)
            except torch.cuda.OutOfMemoryError:
                result = {"safe": False, "oom": True, **memory_metrics(device)}
            finally:
                del collated
                reset()
            result["fixture"] = "all_slots_allocated_nonzero_content_dynamic"
            result["optimizer_state_policy"] = (
                "real_AdamW_step_zero_grads_for_unused_parameters"
            )
            logger.log(
                {
                    "capacity/groups": groups,
                    **{"capacity/" + k: v for k, v in result.items()},
                },
                step=0,
            )
            if len(measured_times) == 2:
                estimate = max(measured_times) * config.effective_groups * 1.3
                logger.log(
                    {
                        "capacity/estimated_effective_update_seconds": estimate,
                        "capacity/train_seconds_remaining": max(
                            0.0, deadline.train - deadline.clock()
                        ),
                    },
                    step=0,
                )
                if not deadline.can_train(estimate):
                    raise TimeoutError(
                        "measured effective update cannot fit before training cutoff; no larger capacity trial started"
                    )
            return result

        result = capacity_search(
            probe, config.effective_groups, capacity_fraction=runtime.capacity_fraction
        )
    result["capacity_fraction"] = runtime.capacity_fraction
    result["model_optimizer_rng_and_initial_state_method_restored"] = True
    return result


class RunLogger:
    """Numeric local journal and online W&B; no environment/source/host upload."""

    def __init__(
        self,
        output: Path,
        *,
        config: dict,
        runtime: RuntimeConfig,
        enabled: bool,
        run_id: str | None = None,
    ):
        self.output, self.run, self.sdk = output, None, None
        self.run_id, self.url = run_id, None
        output.mkdir(parents=True, exist_ok=True)
        if enabled:
            if os.environ.get("WANDB_MODE", "online") != "online":
                raise ValueError("production logging requires W&B online mode")
            self.sdk = importlib.import_module("wandb")
            self.run = self.sdk.init(
                project=runtime.project,
                name=runtime.run_name,
                mode="online",
                id=run_id,
                resume="must" if run_id else None,
                dir=str(output),
                config=config,
                settings={
                    "disable_code": True,
                    "disable_git": True,
                    "x_disable_meta": True,
                    "x_disable_stats": True,
                    "x_disable_machine_info": True,
                    "console": "off",
                    "init_timeout": 60,
                },
            )
            if self.run is None:
                raise RuntimeError("W&B online initialization returned no run")
            self.run_id, self.url = self.run.id, self.run.url
            self.run.define_metric("progress/update")
            self.run.define_metric("*", step_metric="progress/update", step_sync=True)
            _json_write(output / "wandb-run.json", {"id": self.run_id, "url": self.url})

    def log(self, values: dict, *, step: int):
        record = {"progress/update": step, "utc_unix": time.time(), **values}
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
        if self.run is not None:
            self.run.log(record)

    def images(self, images: dict[str, torch.Tensor], *, step: int, phase: str):
        if not images:
            return
        from PIL import Image

        folder = self.output / "visuals"
        folder.mkdir(exist_ok=True)
        upload = {"progress/update": step}
        for name, image in images.items():
            array = (image.numpy() * 255).round().astype(np.uint8)
            filename = f"{phase}-{step:08d}-{name}.png"
            Image.fromarray(array).save(folder / filename)
            if self.run is not None:
                upload["images/" + phase + "/" + name] = self.sdk.Image(
                    array, caption=name
                )
        if self.run is not None:
            self.run.log(upload)

    def finish(self, *, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


def _json_write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _record_evaluation(result, *, output, logger, step, phase):
    images = result.pop("visuals")
    _json_write(output / f"{phase}-step-{step:08d}.json", result)
    flattened = {
        phase + "/" + group + "/" + key: value
        for group, metrics in result["metrics"].items()
        for key, value in metrics.items()
    }
    flattened.update(
        {
            phase + "/" + key: result[key]
            for key in (
                "complete",
                "sources_completed",
                "sources_requested",
                "variants_completed",
                "frames_completed",
                "frames_expected",
                "elapsed_seconds",
            )
        }
    )
    logger.log(flattened, step=step)
    logger.images(images, step=step, phase=phase)
    return result


def worker_budget(
    requested: int, *, sample_bytes: int, micro_groups: int, shm_available: int
) -> int:
    """Reserve half shared memory; allow one queued batch per worker plus consumer."""
    batch_bytes = sample_bytes * micro_groups
    if batch_bytes <= 0 or shm_available <= 0:
        return 0
    return min(requested, max(0, int(0.5 * shm_available) // batch_bytes - 1))


def _tensor_bytes(value):
    if isinstance(value, dict):
        return sum(_tensor_bytes(child) for child in value.values())
    return (
        value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0
    )


def _loader(dataset, config, *, cursor, updates, micro_groups):
    sampler = SourceBatchSampler(
        len(dataset),
        seed=config.seed,
        cursor=cursor,
        updates=updates,
        effective_groups=config.effective_groups,
        micro_groups=micro_groups,
    )
    workers, sample_bytes, shm_available = 0, 0, 0
    if config.workers:
        sample_bytes = _tensor_bytes(dataset[0])
        try:
            shm = os.statvfs("/dev/shm")
            shm_available = shm.f_bavail * shm.f_frsize
        except OSError:
            pass
        workers = worker_budget(
            config.workers,
            sample_bytes=sample_bytes,
            micro_groups=micro_groups,
            shm_available=shm_available,
        )
    options = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": torch.Generator().manual_seed(config.seed + 991),
    }
    if workers:
        options.update(prefetch_factor=1, persistent_workers=False)
    loader = DataLoader(dataset, batch_sampler=sampler, **options)
    loader.resource_budget = {
        "requested_workers": config.workers,
        "effective_workers": workers,
        "source_tensor_bytes": sample_bytes,
        "shm_available_bytes": shm_available,
        "prefetch_batches_per_worker": 1 if workers else 0,
    }
    return loader


def run_training(
    model,
    optimizer,
    datasets,
    config,
    runtime,
    *,
    output,
    logger,
    device,
    micro_groups,
    config_payload,
    checkpoint=None,
    initial_update_estimate=0.0,
):
    deadline = Deadline(config)
    step, cursor, best = 0, 0, float("inf")
    if checkpoint:
        restored = restore_checkpoint(
            checkpoint, model, optimizer, expected_config=config_payload
        )
        step, cursor = restored["step"], restored["extra"]["source_cursor"]
        best = restored["extra"].get("best_dev", float("inf"))
        if restored["extra"]["micro_groups"] != micro_groups:
            raise ValueError("exact resume requires unchanged micro_groups")
    consumed_cursor = cursor
    last_checkpoint_time = time.monotonic()
    last_eval = None

    def snapshot(tag, *, is_best=False):
        path = output / "checkpoints" / f"{tag}-{step:08d}-{time.time_ns()}.pt"
        extra = {
            "source_cursor": consumed_cursor,
            "micro_groups": micro_groups,
            "best_dev": best,
            "run_id": logger.run_id,
            "data_order": "source_cursor_seeded_epoch_permutation_v1",
            "resume_scope": "optimizer-boundary_same_config_and_cached_deterministic_sources",
            "bitwise_gpu_replay_guaranteed": False,
        }
        save_checkpoint(
            path, model, optimizer, step=step, config=config_payload, extra=extra
        )
        reference = {
            "checkpoint": str(path.relative_to(output)),
            "step": step,
            "source_cursor": consumed_cursor,
            "best_dev": None if math.isinf(best) else best,
        }
        _json_write(output / "latest.json", reference)
        if is_best:
            _json_write(output / "best.json", reference)
        return path

    def periodic(phase):
        nonlocal best, last_eval
        end = min(time.time() + runtime.periodic_eval_seconds, deadline.train)
        result = evaluate(
            model,
            datasets["dev"],
            config,
            source_limit=min(runtime.periodic_eval_sources, len(datasets["dev"])),
            rays=runtime.eval_rays,
            seed=config.seed + 7000,
            deadline=lambda: time.time() < end,
            visuals=True,
        )
        last_eval = _record_evaluation(
            result, output=output, logger=logger, step=step, phase=phase
        )
        score = result["metrics"].get("future/model", {}).get("rgb_region_balanced_mae")
        improved = result["complete"] and score is not None and score < best
        if improved:
            best = score
            snapshot("best", is_best=True)
        return improved

    if not checkpoint:
        periodic("initial_dev_sampled")
        snapshot("initial")
    loader = _loader(
        datasets["train"],
        config,
        cursor=consumed_cursor,
        updates=max(0, config.max_updates - step),
        micro_groups=micro_groups,
    )
    logger.log(
        {"data/" + key: value for key, value in loader.resource_budget.items()},
        step=step,
    )
    batches = iter(loader)
    estimated_update = max(0.0, initial_update_estimate)
    while step < config.max_updates and deadline.can_train(estimated_update):
        metrics = run_update(
            model,
            optimizer,
            batches,
            config,
            micro_groups=micro_groups,
            step=step,
            device=device,
        )
        step += 1
        consumed_cursor += config.effective_groups
        estimated_update = max(estimated_update * 0.8, metrics["update_seconds"] * 1.2)
        logger.log({"train/" + key: value for key, value in metrics.items()}, step=step)
        if step % config.eval_every == 0 and deadline.can_train(
            runtime.periodic_eval_seconds
        ):
            periodic("dev_sampled")
        if time.monotonic() - last_checkpoint_time >= config.save_every_seconds:
            snapshot("periodic")
            last_checkpoint_time = time.monotonic()
    final_checkpoint = snapshot("final")
    # Workers may have prefetched deterministic indices. Resume stores only the
    # consumed cursor; queued work does not enter a checkpoint or a next run.
    del batches, loader
    result = evaluate(
        model,
        datasets["test"],
        config,
        source_limit=runtime.eval_sources,
        rays=None,
        seed=config.seed + 9000,
        deadline=lambda: deadline.can_evaluate(),
        visuals=True,
    )
    final_eval = _record_evaluation(
        result, output=output, logger=logger, step=step, phase="final_test_full_frame"
    )
    summary = {
        "updates": step,
        "source_cursor": consumed_cursor,
        "micro_groups": micro_groups,
        "checkpoint": str(final_checkpoint.relative_to(output)),
        "final_evaluation_complete": final_eval["complete"],
        "final_sources_completed": final_eval["sources_completed"],
        "final_sources_requested": final_eval["sources_requested"],
        "run_url": logger.url,
    }
    _json_write(output / "summary.json", summary)
    return summary


def _make_datasets(data_dir, config, runtime):
    scene = FactorialSceneConfig(
        frames=config.frames,
        prefix_frames=config.prefix_frames,
        height=config.height,
        width=config.width,
    )
    counts = {
        "train": config.train_sources,
        "dev": config.dev_sources,
        "test": runtime.eval_sources,
    }
    return {
        split: CachedFactorialDataset(
            data_dir, count, scene, split=split, base_seed=config.seed
        )
        for split, count in counts.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capacity", "train", "eval"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-groups", type=int)
    parser.add_argument(
        "--eval-full-frame",
        action="store_true",
        help="eval defaults to full frame; retained for explicit invocations",
    )
    args = parser.parse_args(argv)
    payload = json.loads(args.config.read_text())
    unknown = set(payload) - {"architecture", "training", "runtime"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    architecture = ObjectStateConfig(**payload.get("architecture", {}))
    config = TrainingConfig(**payload.get("training", {}))
    runtime = RuntimeConfig(**payload.get("runtime", {}))
    payload = {
        "architecture": architecture.to_dict(),
        "training": config.to_dict(),
        "runtime": asdict(runtime),
    }
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index not in (None, 0) or torch.cuda.device_count() != 1:
            raise ValueError(
                "production requires exactly one isolated visible GPU, logical cuda:0"
            )
        if not config.wandb_online:
            raise ValueError("production W&B online logging is mandatory")
        torch.cuda.set_device(0)
    elif device.type != "cpu":
        raise ValueError(
            "only isolated CUDA production or CPU correctness is supported"
        )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = ObjectStateModel(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    datasets = _make_datasets(args.data_dir, config, runtime)
    args.output.mkdir(parents=True, exist_ok=True)
    _json_write(args.output / "resolved-config.json", payload)
    run_id = None
    if args.checkpoint:
        header = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        run_id = header.get("extra", {}).get("run_id") if args.mode == "train" else None
        del header
    rng = _rng_state()
    logger = RunLogger(
        args.output,
        config=payload,
        runtime=runtime,
        enabled=config.wandb_online,
        run_id=run_id,
    )
    _restore_rng(rng)
    exit_code = 1
    try:
        if args.mode == "capacity":
            result = measure_capacity(
                model,
                optimizer,
                datasets["train"],
                config,
                runtime,
                device=device,
                logger=logger,
                deadline=Deadline(config),
            )
            result["config"] = payload
            _json_write(args.output / "capacity.json", result)
        elif args.mode == "train":
            micro = args.micro_groups
            capacity = None
            capacity_path = args.output / "capacity.json"
            if capacity_path.exists():
                capacity = json.loads(capacity_path.read_text())
                if capacity["config"] != payload:
                    raise ValueError(
                        "capacity result must match the exact production config"
                    )
            if device.type == "cuda" and capacity is None:
                raise ValueError(
                    "production training requires the completed capacity result"
                )
            if micro is None and capacity is not None:
                micro = capacity["micro_groups"]
            if micro is None:
                raise ValueError("CPU correctness needs explicit --micro-groups")
            if not 1 <= micro <= config.effective_groups:
                raise ValueError(
                    "micro_groups must be within the effective source batch"
                )
            if capacity is not None and micro > capacity["micro_groups"]:
                raise ValueError("micro_groups exceeds measured safe capacity")
            measured = (
                []
                if capacity is None
                else [
                    trial["update_seconds"]
                    for trial in capacity["measurements"]
                    if trial["groups"] == micro
                    and trial["safe"]
                    and "update_seconds" in trial
                ]
            )
            estimate = (
                max(measured, default=0.0)
                * math.ceil(config.effective_groups / micro)
                * 1.3
            )
            result = run_training(
                model,
                optimizer,
                datasets,
                config,
                runtime,
                output=args.output,
                logger=logger,
                device=device,
                micro_groups=micro,
                config_payload=payload,
                checkpoint=args.checkpoint,
                initial_update_estimate=estimate,
            )
        else:
            if args.checkpoint is None:
                raise ValueError("eval requires a trained checkpoint")
            restored = restore_checkpoint(
                args.checkpoint, model, optimizer, expected_config=payload
            )
            result = evaluate(
                model,
                datasets["test"],
                config,
                source_limit=runtime.eval_sources,
                rays=None,
                seed=config.seed + 9000,
                deadline=Deadline(config).can_evaluate,
                visuals=True,
            )
            result = _record_evaluation(
                result,
                output=args.output,
                logger=logger,
                step=restored["step"],
                phase="test_full_frame",
            )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("config", "measurements", "metrics")
                },
                allow_nan=False,
            )
        )
        exit_code = 0
    finally:
        logger.finish(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
