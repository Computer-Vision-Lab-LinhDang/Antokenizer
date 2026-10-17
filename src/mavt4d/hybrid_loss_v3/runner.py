"""Single-GPU B trial with local boundary recovery and best-only W&B artifacts."""

from contextlib import contextmanager
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any
import uuid

import numpy as np
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid_pilot.data import HybridPilotDataset, file_sha256
from mavt4d.hybrid_pilot.logging import RunLogger
from mavt4d.hybrid_pilot.runner import (
    DeadlineController,
    DistributedContext,
    _write_json,
    require_finite_loss,
    load_checkpoint,
    save_checkpoint,
    scheduled_step,
    _rng_state,
    _restore_rng,
)

from mavt4d.hybrid_diagnostic.telemetry import selected_gpu_metrics
from .config import LossV3TrialConfig
from .evaluation import aligned_panel, development, labels_for_dataset
from .objectives import (
    LossV3Module,
    LOSS_WEIGHTS,
    ENDPOINTS,
    PIXEL_THRESHOLD,
    MIN_WITNESS_PIXELS,
    MIN_WITNESS_FRAMES,
    C_PAIR_SCALE,
    C_STD_FLOOR,
    ORDER_MARGIN,
)


RESUME_VERSION = "hybrid-loss-v3-last-v1"
RESUME_KEYS = (
    "schema",
    "arm",
    "base_config",
    "base_config_sha256",
    "objective",
    "dataset_sha256",
    "manifest_sha256",
    "partition_sha256",
    "world_size",
    "spatial_bf16",
    "recurrent_decoder_dtype",
    "held_out_evaluated",
    "trial_seed",
    "selection_metric",
    "schedule_schema",
)


@contextmanager
def preserve_rng(context: DistributedContext) -> Iterator[None]:
    """Evaluation, panels and SDK activity must not advance training RNG."""
    state = _rng_state(context)
    try:
        yield
    finally:
        _restore_rng(state, context)


def _replace_durable(candidate, destination):
    os.replace(candidate, destination)
    descriptor = os.open(Path(destination).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_durable(source, destination):
    candidate = Path(destination).with_name(f".copy-{uuid.uuid4().hex}.pt")
    with Path(source).open("rb") as src, candidate.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    _replace_durable(candidate, destination)


def save_last(
    output: str | Path,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
    context: DistributedContext,
) -> None:
    """Replace local recovery state only after the full snapshot is durable."""
    candidate = Path(output) / f".last-{uuid.uuid4().hex}.pt"
    save_checkpoint(
        candidate,
        module,
        optimizer,
        {**metadata, "checkpoint_kind": "last", "resume_version": RESUME_VERSION},
        context,
    )
    _replace_durable(candidate, Path(output) / "last.pt")


def prepare_resume(
    path: str | Path,
    lock: Mapping[str, Any],
    *,
    allow_batch_change: bool,
) -> tuple[Path, str, dict[str, Any], Path | None, bool]:
    """Validate last and its selected companion before mutating model or output."""
    path = Path(path)
    if path.is_dir() or not path.exists():
        path = path / "last.pt" if path.suffix != ".pt" else path
    if not path.is_file():
        raise FileNotFoundError(
            f"required local last checkpoint does not exist: {path}"
        )
    digest = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if (
        metadata.get("checkpoint_kind") != "last"
        or metadata.get("resume_version") != RESUME_VERSION
    ):
        raise ValueError(
            "resume requires a local last checkpoint; best is not a fallback"
        )
    for key in RESUME_KEYS:
        if metadata.get(key) != lock[key]:
            raise ValueError(f"checkpoint {key} mismatch")
    step, cursor = metadata.get("step"), metadata.get("source_cursor")
    if (
        type(step) is not int
        or step < 0
        or metadata.get("next_step") != step + 1
        or type(cursor) is not int
        or cursor < 0
        or metadata.get("source_presentations") != cursor
        or metadata.get("clip_presentations") != cursor * 8
    ):
        raise ValueError("invalid last optimizer boundary or cumulative exposure")
    changed = metadata["effective_source_groups"] != lock["effective_source_groups"]
    if changed and not allow_batch_change:
        raise ValueError("effective batch changed; require --allow-batch-change")
    best = metadata.get("best_selection")
    companion = None
    if best is not None:
        if (
            type(best.get("step")) is not int
            or not 0 <= best["step"] <= step
            or not math.isfinite(best.get("score", float("nan")))
        ):
            raise ValueError("invalid historical best selection")
        for name in ("best.pt", ".resume-best.pt"):
            candidate = path.parent / name
            if candidate.is_file() and file_sha256(candidate) == best.get("sha256"):
                companion = candidate
                break
        if companion is None:
            raise ValueError("historical best companion missing or hash mismatch")
        best_payload = torch.load(companion, map_location="cpu", weights_only=True)
        selected = best_payload["metadata"]
        for key in RESUME_KEYS:
            if selected.get(key) != lock[key]:
                raise ValueError(f"best checkpoint {key} mismatch")
        if (
            selected.get("step") != best["step"]
            or selected.get("selection_score") != best["score"]
        ):
            raise ValueError("historical best companion metadata mismatch")
    if file_sha256(path) != digest:
        raise ValueError("last checkpoint changed during resume validation")
    return path, digest, metadata, companion, changed


def validate_device(device, *, allow_cpu=False):
    device = torch.device(device)
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) != 0
    ):
        raise ValueError("v3 trial requires exactly one process/rank")
    if allow_cpu:
        if device.type != "cpu":
            raise ValueError("CPU correctness mode must use CPU")
    elif (
        device.type != "cuda"
        or device.index not in (None, 0)
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError(
            "production requires exactly one visible allocated GPU as cuda:0"
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return DistributedContext(0, 1, 0, device)


def initialize(base, device, *, spatial_bf16):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(base.seed)
        module = LossV3Module(base)
    module.to(device=device, dtype=torch.float32)
    if spatial_bf16:
        if device.type != "cuda":
            raise ValueError("spatial BF16 requires GPU")
        module.model.observer.spatial_autocast = True
    return module


def _optimizer(module, base):
    return torch.optim.AdamW(
        module.parameters(),
        lr=base.learning_rate,
        betas=base.betas,
        weight_decay=base.weight_decay,
    )


def initialize_from_checkpoint(path, module, optimizer, context, lock):
    """Start an explicit new batch segment, retaining model, Adam and rank RNG."""
    expected = {
        key: lock[key]
        for key in (
            "schema",
            "arm",
            "base_config",
            "base_config_sha256",
            "objective",
            "dataset_sha256",
            "manifest_sha256",
            "partition_sha256",
            "world_size",
            "spatial_bf16",
            "recurrent_decoder_dtype",
            "held_out_evaluated",
        )
    }
    expected.update(phase="loss-v3-B", selection_metric="loss-v3-common-dev-v1")
    digest = file_sha256(path)
    metadata = load_checkpoint(path, module, optimizer, context, expected=expected)
    if type(metadata.get("step")) is not int or metadata["step"] < 0:
        raise ValueError("checkpoint step must be a nonnegative optimizer boundary")
    if file_sha256(path) != digest:
        raise ValueError("checkpoint changed during initialization")
    return {
        "kind": "new_batch_segment",
        "parent_checkpoint_sha256": digest,
        "parent_step": metadata["step"],
        "segment_start_step": 0,
        "exact_resume": False,
    }


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def optimization_step(
    module,
    optimizer,
    dataset,
    labels,
    schedule,
    microbatch,
    context,
    deadlines=None,
    *,
    phase="main",
):
    """Equal-source accumulation; interrupted accumulation never steps optimizer."""
    optimizer.zero_grad(set_to_none=True)
    count = len(schedule)
    totals = {}
    for start in range(0, count, microbatch):
        if deadlines is not None and deadlines.should_stop(context):
            optimizer.zero_grad(set_to_none=True)
            return None
        part = schedule[start : start + microbatch]
        batches = [dataset.load_group(group, context.device) for group, _ in part]
        loss, metrics = module(
            batches,
            [seed for _, seed in part],
            [labels[group] for group, _ in part],
            phase=phase,
        )
        require_finite_loss(loss, context)
        fraction = len(part) / count
        (loss * fraction).backward()
        for key, value in {**metrics, "loss": loss.detach()}.items():
            totals[key] = totals.get(key, 0.0) + float(value) * fraction
    if deadlines is not None and deadlines.should_stop(context):
        optimizer.zero_grad(set_to_none=True)
        return None
    norm = torch.nn.utils.clip_grad_norm_(
        module.parameters(), module.cfg.grad_clip, error_if_nonfinite=True
    )
    optimizer.step()
    totals["gradient_norm"] = float(norm)
    if not all(math.isfinite(value) for value in totals.values()):
        raise FloatingPointError("nonfinite optimizer metric")
    return totals


def save_selected(output, module, optimizer, scores, best_score, metadata, context):
    """Only complete S_dev, with strict improvements and atomic replacement."""
    if not scores.get("complete", False):
        return best_score
    score = scores["S_dev"]
    if not math.isfinite(score):
        raise FloatingPointError("nonfinite checkpoint selection score")
    if score >= best_score:
        return best_score
    candidate = Path(output) / f".best-{uuid.uuid4().hex}.pt"
    save_checkpoint(
        candidate,
        module,
        optimizer,
        {
            **metadata,
            "selection_metric": "loss-v3-common-dev-v1",
            "selection_score": score,
        },
        context,
    )
    best_path = Path(output) / "best.pt"
    if best_path.exists():
        # If a crash occurs before last refresh, its prior best remains recoverable.
        _copy_durable(best_path, Path(output) / ".resume-best.pt")
    _replace_durable(candidate, best_path)
    return score


def _write_panel(logger, panel, step):
    from PIL import Image

    folder = logger.output / "images"
    folder.mkdir(exist_ok=True)
    values = (panel.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
    image = Image.fromarray(values)
    image.save(folder / f"current-{step:06d}.jpg", quality=85)
    if logger.run is not None:
        logger.run.log(
            {
                "diagnostic/images_current": logger.sdk.Image(
                    image,
                    caption=f"Current weights step {step}; clean frame3. Rows: configurations of core0. "
                    "Columns: own target; own reconstruction; D-swap; donor target; D-zero. Display clipped only.",
                ),
                "progress/logical_step": step,
                "progress/optimizer_step": step,
            }
        )


def capacity_probe(
    base,
    cfg,
    dataset_root,
    output,
    *,
    optimization_at,
    work_at,
    device="cuda:0",
    spatial_bf16=True,
    allow_cpu=False,
):
    """Disposable models; each candidate measures complete effective updates."""
    context = validate_device(device, allow_cpu=allow_cpu)
    if not allow_cpu:
        cfg.validate_production()
    deadlines = DeadlineController(optimization_at, work_at)
    if datetime.now(timezone.utc) >= optimization_at:
        raise ValueError("capacity optimization deadline has already passed")
    if not allow_cpu and (base.architecture != HybridConfig() or base.seed != cfg.seed):
        raise ValueError("capacity uses the frozen production model and seed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = HybridPilotDataset(dataset_root, base)
    labels = labels_for_dataset(dataset)
    schedule = scheduled_step(
        dataset.train_groups,
        seed=cfg.seed,
        step=1,
        global_groups=cfg.effective_source_groups,
    )
    records = []
    deadlines.install()
    try:
        for microbatch in cfg.capacity_order:
            if deadlines.should_stop(context):
                break
            module = optimizer = None
            record = {
                "microbatch_groups": microbatch,
                "accumulation": cfg.accumulation(microbatch),
                "mask_policy": "clean_memory_admission_then_main_timing",
            }
            try:
                module = initialize(base, context.device, spatial_bf16=spatial_bf16)
                optimizer = _optimizer(module, base)
                if context.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(context.device)
                measured = []
                for repeat in range(cfg.capacity_warmup + cfg.capacity_repeats):
                    _synchronize(context.device)
                    started = time.monotonic()
                    metrics = optimization_step(
                        module,
                        optimizer,
                        dataset,
                        labels,
                        schedule,
                        microbatch,
                        context,
                        deadlines,
                        phase="tiny" if repeat < cfg.capacity_warmup else "main",
                    )
                    _synchronize(context.device)
                    if metrics is None:
                        break
                    if repeat >= cfg.capacity_warmup:
                        measured.append(time.monotonic() - started)
                if len(measured) != cfg.capacity_repeats:
                    record["status"] = "incomplete_deadline"
                else:
                    elapsed = sum(measured) / len(measured)
                    record.update(
                        status="complete",
                        seconds_per_update=elapsed,
                        clips_per_second=cfg.effective_source_groups * 8 / elapsed,
                    )
                    if context.device.type == "cuda":
                        allocated = torch.cuda.max_memory_allocated(context.device)
                        reserved = torch.cuda.max_memory_reserved(context.device)
                        total = torch.cuda.get_device_properties(
                            context.device
                        ).total_memory
                        free, _ = torch.cuda.mem_get_info(context.device)
                        record.update(
                            peak_allocated_bytes=allocated,
                            peak_reserved_bytes=reserved,
                            total_bytes=total,
                            free_bytes=free,
                            headroom_ok=reserved <= total * (1 - cfg.memory_headroom)
                            and free >= total * cfg.memory_headroom,
                        )
                    else:
                        record["headroom_ok"] = True
            except torch.cuda.OutOfMemoryError:
                record["status"] = "out_of_memory"
            finally:
                del module, optimizer
                gc.collect()
                if context.device.type == "cuda":
                    torch.cuda.empty_cache()
            records.append(record)
            _write_json(
                output / "capacity.json",
                {
                    "version": cfg.version,
                    "config_sha256": cfg.sha256,
                    "effective_source_groups": cfg.effective_source_groups,
                    "capacity_policy": cfg.capacity_policy,
                    "candidate_order": cfg.capacity_order,
                    "records": records,
                    "selected_microbatch": choose_capacity(
                        records, policy=cfg.capacity_policy
                    ),
                    "held_out_evaluated": False,
                },
            )
            print(json.dumps(record), flush=True)
            if (
                cfg.capacity_policy == "largest_safe"
                and choose_capacity(records, policy=cfg.capacity_policy) is not None
            ):
                break
        return {
            "records": records,
            "selected_microbatch": choose_capacity(records, policy=cfg.capacity_policy),
            "capacity_policy": cfg.capacity_policy,
            "candidate_order": cfg.capacity_order,
            "config_sha256": cfg.sha256,
        }
    finally:
        deadlines.restore()


def choose_capacity(records, *, policy="fastest_safe"):
    if policy not in ("fastest_safe", "largest_safe"):
        raise ValueError("unknown capacity selection policy")
    eligible = [
        row
        for row in records
        if row.get("status") == "complete" and row.get("headroom_ok")
    ]
    if policy == "largest_safe":
        return max((row["microbatch_groups"] for row in eligible), default=None)
    return (
        min(eligible, key=lambda row: row["seconds_per_update"])["microbatch_groups"]
        if eligible
        else None
    )


def run_trial(
    base_config,
    cfg: LossV3TrialConfig,
    dataset_root,
    output,
    *,
    optimization_at,
    work_at,
    git_revision,
    device="cuda:0",
    microbatch_groups=1,
    spatial_bf16=True,
    allow_cpu=False,
    initial_checkpoint=None,
    resume=None,
    allow_batch_change=False,
):
    if resume is not None and initial_checkpoint is not None:
        raise ValueError("resume and initial_checkpoint are mutually exclusive")
    if allow_batch_change and resume is None:
        raise ValueError("allow_batch_change requires resume")
    context = validate_device(device, allow_cpu=allow_cpu)
    if not allow_cpu:
        cfg.validate_production()
    if base_config.seed != cfg.seed:
        raise ValueError("base initialization seed differs from trial seed")
    if not allow_cpu and base_config.architecture != HybridConfig():
        raise ValueError("production B keeps the frozen full backbone/decoder")
    cfg.accumulation(microbatch_groups)
    deadlines = DeadlineController(optimization_at, work_at)
    if datetime.now(timezone.utc) >= optimization_at:
        raise ValueError("optimization deadline has already passed")
    if len(git_revision) != 40 or any(
        c not in "0123456789abcdef" for c in git_revision
    ):
        raise ValueError("git_revision must be a full lowercase SHA")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = HybridPilotDataset(dataset_root, base_config)
    if not dataset.dev_groups:
        raise ValueError("complete fixed dev requires a nonempty source partition")
    labels = labels_for_dataset(dataset)
    partition = {
        "train": list(dataset.train_groups),
        "dev": list(dataset.dev_groups),
        "held_out": list(dataset.val_groups),
    }
    lock = {
        "schema": "hybrid-cd-loss-v3-run-v1",
        "arm": "B",
        "trial_seed": cfg.seed,
        "selection_metric": cfg.selection,
        "schedule_schema": "hybrid-pilot-step-source-mask-v1",
        "trial_config": cfg.to_dict(),
        "trial_config_sha256": cfg.sha256,
        "base_config": base_config.to_dict(),
        "base_config_sha256": base_config.config_sha256,
        "git_revision": git_revision,
        "objective": {
            "weights": LOSS_WEIGHTS,
            "endpoints": ENDPOINTS,
            "pixel_threshold": PIXEL_THRESHOLD,
            "witness_pixels": MIN_WITNESS_PIXELS,
            "witness_frames": MIN_WITNESS_FRAMES,
            "c_pair_scale": C_PAIR_SCALE,
            "c_std_floor": C_STD_FLOOR,
            "order_margin": ORDER_MARGIN,
        },
        "dataset_sha256": file_sha256(dataset.root / "summary.json"),
        "manifest_sha256": file_sha256(dataset.root / "manifest_v1.jsonl"),
        "partition_sha256": hashlib.sha256(
            json.dumps(partition, sort_keys=True).encode()
        ).hexdigest(),
        "train_source_groups": list(dataset.train_groups),
        "dev_source_groups": list(dataset.dev_groups),
        "world_size": 1,
        "microbatch_groups": microbatch_groups,
        "effective_source_groups": cfg.effective_source_groups,
        "accumulation": cfg.accumulation(microbatch_groups),
        "spatial_bf16": spatial_bf16,
        "recurrent_decoder_dtype": "float32",
        "held_out_evaluated": False,
        "optimization_at": optimization_at.isoformat(),
        "work_at": work_at.isoformat(),
    }
    recovery = None
    resume_path = digest = None
    step, source_cursor, best_step, last_dev_step = 0, 0, None, None
    best_score, last_dev, best_metadata, best_selection = float("inf"), None, None, None
    pending_dev = True
    latest_eval_seconds = latest_step_seconds = 0.0
    batch_transitions = []
    if resume is not None:
        recovery = prepare_resume(resume, lock, allow_batch_change=allow_batch_change)
        resume_path, digest, prior, companion, changed = recovery
        step, source_cursor = prior["step"], prior["source_cursor"]
        if cfg.steps < step:
            raise ValueError(
                "target steps is an absolute global target below resumed step"
            )
        batch_transitions = list(prior.get("batch_transitions", []))
        if changed or prior["microbatch_groups"] != microbatch_groups:
            batch_transitions.append(
                {
                    "at_step": step,
                    "source_cursor": source_cursor,
                    "from_effective_source_groups": prior["effective_source_groups"],
                    "to_effective_source_groups": cfg.effective_source_groups,
                    "from_microbatch_groups": prior["microbatch_groups"],
                    "to_microbatch_groups": microbatch_groups,
                    "same_effective_batch": not changed,
                }
            )
        lock["initialization"] = {
            "kind": "resume_last",
            "parent_checkpoint_sha256": digest,
            "parent_step": step,
            "segment_start_step": step,
            "parent_wandb_run_id": prior.get("wandb_run_id"),
            "exact_resume": not changed
            and prior["microbatch_groups"] == microbatch_groups,
            "same_effective_batch": not changed,
            "trajectory": (
                "compatible_continuation" if changed else "same_effective_batch"
            ),
        }
        best_selection = prior["best_selection"]
        state = prior["development_state"]
        last_dev, last_dev_step = state["last_dev"], state["last_dev_step"]
        pending_dev = state["pending"]
        latest_eval_seconds = state["latest_eval_seconds"]
        if best_selection is not None:
            best_score, best_step = best_selection["score"], best_selection["step"]
            _copy_durable(companion, output / "best.pt")
            if file_sha256(output / "best.pt") != best_selection["sha256"]:
                raise ValueError("best companion changed during resume copy")
            best_metadata = torch.load(
                output / "best.pt", map_location="cpu", weights_only=True
            )["metadata"]
    module = initialize(base_config, context.device, spatial_bf16=spatial_bf16)
    optimizer = _optimizer(module, base_config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if initial_checkpoint is not None:
        lock["initialization"] = initialize_from_checkpoint(
            initial_checkpoint, module, optimizer, context, lock
        )
    elif resume is None:
        lock["initialization"] = {"kind": "fresh_seed", "segment_start_step": 0}
    lock["batch_transitions"] = batch_transitions
    _write_json(output / "run_lock.json", lock)
    # A fresh W&B run preserves the parent history; global/logical step continues.
    with preserve_rng(context):
        logger = RunLogger(output, rank=0, config=lock, enabled=not allow_cpu)
        if logger.run is not None:
            logger.run.tags = ("loss-v3", "B", "single-gpu", "synthetic")
    deadlines.install()
    error = None
    try:
        if recovery is not None:
            load_checkpoint(
                resume_path,
                module,
                optimizer,
                context,
                expected={key: lock[key] for key in RESUME_KEYS},
            )
            if file_sha256(resume_path) != digest:
                raise ValueError("last checkpoint changed during restore")

        def boundary_metadata():
            return {
                **lock,
                "phase": "loss-v3-B",
                "step": step,
                "next_step": step + 1,
                "source_cursor": source_cursor,
                "source_presentations": source_cursor,
                "clip_presentations": source_cursor * 8,
                "wandb_run_id": logger.run_id,
                "best_selection": best_selection,
                "development_state": {
                    "last_dev_step": last_dev_step,
                    "last_dev": last_dev,
                    "pending": pending_dev,
                    "latest_eval_seconds": latest_eval_seconds,
                },
            }

        def persist_last():
            save_last(output, module, optimizer, boundary_metadata(), context)

        # Initialization or validated restore is recoverable before any evaluation.
        persist_last()

        def evaluate(current_step, *, force_image=False):
            nonlocal best_score, best_step, last_dev_step, last_dev, best_metadata, latest_eval_seconds, best_selection, pending_dev
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            scores = development(
                module,
                dataset,
                labels,
                deadlines,
                context.device,
                progress=lambda split, done, total: logger.log(
                    {
                        f"dev/progress_{split}": done / total,
                        "progress/optimizer_step": current_step,
                    },
                    step=current_step,
                ),
            )
            latest_eval_seconds = time.monotonic() - started
            if not scores["complete"]:
                logger.log(
                    {"dev/complete": False, "dev/reason": scores["reason"]},
                    step=current_step,
                )
                return False
            last_dev, last_dev_step = scores, current_step
            logger.log(
                {
                    **{f"dev/{key}": value for key, value in scores.items()},
                    "progress/optimizer_step": current_step,
                    "performance/dev_seconds": latest_eval_seconds,
                },
                step=current_step,
            )
            metadata = boundary_metadata()
            previous = best_score
            best_score = save_selected(
                output, module, optimizer, scores, best_score, metadata, context
            )
            if best_score < previous:
                best_step, best_metadata = current_step, {
                    **metadata,
                    "selection_score": best_score,
                }
            if best_score < previous:
                best_selection = {
                    "step": best_step,
                    "score": best_score,
                    "sha256": file_sha256(output / "best.pt"),
                }
            pending_dev = False
            _write_json(
                output / "dev-latest.json",
                {
                    "step": current_step,
                    "metrics": scores,
                    "best_step": best_step,
                    "best_score": best_score,
                },
            )
            if not deadlines.should_stop(context, optimization=False):
                group = dataset.dev_groups[0]
                metrics, panel = aligned_panel(
                    module, dataset.load_group(group, context.device), labels[group]
                )
                logger.log(
                    {f"diagnostic/{key}": value for key, value in metrics.items()},
                    step=current_step,
                )
                if (
                    force_image
                    or current_step == 0
                    or current_step % cfg.image_every == 0
                    or current_step == cfg.steps
                    or deadlines.should_stop(context)
                ):
                    _write_panel(logger, panel, current_step)
            return True

        def evaluate_preserved(current_step, *, force_image=False):
            nonlocal pending_dev
            if not pending_dev:
                pending_dev = True
                persist_last()
            with preserve_rng(context):
                completed = evaluate(current_step, force_image=force_image)
            if completed:
                persist_last()
            return completed

        if pending_dev and not evaluate_preserved(step):
            raise RuntimeError("initial full dev incomplete; training was not started")
        module.train()
        last_metrics = {}
        while step < cfg.steps and not deadlines.should_stop(context):
            # Reserve a measured final full-dev pass, checkpoint and logger flush.
            remaining = (work_at - datetime.now(timezone.utc)).total_seconds()
            if remaining < 1.2 * latest_eval_seconds + 1.2 * latest_step_seconds + 30:
                break
            schedule = scheduled_step(
                dataset.train_groups,
                seed=cfg.seed,
                step=step + 1,
                global_groups=cfg.effective_source_groups,
            )
            _synchronize(context.device)
            started = time.monotonic()
            metrics = optimization_step(
                module,
                optimizer,
                dataset,
                labels,
                schedule,
                microbatch_groups,
                context,
                deadlines,
            )
            _synchronize(context.device)
            if metrics is None:
                # Partial accumulation may consume RNG/buffers but is not a boundary.
                load_checkpoint(
                    output / "last.pt",
                    module,
                    optimizer,
                    context,
                    expected={key: lock[key] for key in RESUME_KEYS},
                )
                break
            latest_step_seconds = time.monotonic() - started
            step += 1
            source_cursor += len(schedule)
            pending_dev = step % cfg.dev_every == 0 or step == cfg.steps
            # Do this before telemetry, evaluation or any network side effect.
            persist_last()
            last_metrics = metrics
            if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
                values = {f"train/{key}": value for key, value in metrics.items()}
                values.update(
                    {
                        "progress/optimizer_step": step,
                        "progress/target_steps": cfg.steps,
                        "performance/step_seconds": latest_step_seconds,
                        "performance/clips_per_second": cfg.effective_source_groups
                        * 8
                        / latest_step_seconds,
                        "progress/source_presentations": source_cursor,
                        "progress/clip_presentations": source_cursor * 8,
                    }
                )
                if context.device.type == "cuda":
                    values.update(selected_gpu_metrics())
                    values.update(
                        {
                            "system/peak_allocated_gib": torch.cuda.max_memory_allocated(
                                context.device
                            )
                            / 2**30,
                            "system/peak_reserved_gib": torch.cuda.max_memory_reserved(
                                context.device
                            )
                            / 2**30,
                        }
                    )
                with preserve_rng(context):
                    logger.log(values, step=step)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": metrics["loss"],
                            "seconds": latest_step_seconds,
                        }
                    ),
                    flush=True,
                )
            if step % cfg.dev_every == 0 and not evaluate_preserved(step):
                break
        if last_dev_step != step and not deadlines.should_stop(
            context, optimization=False
        ):
            evaluate_preserved(step, force_image=True)
        result = {
            "status": "complete" if step == cfg.steps else "stopped",
            "step": step,
            "target_steps": cfg.steps,
            "last_dev_step": last_dev_step,
            "last_dev": last_dev,
            "best_step": best_step,
            "best_score": best_score if math.isfinite(best_score) else None,
            "last_training_metrics": last_metrics,
            "checkpoint": "best.pt" if best_metadata else None,
            "last_checkpoint": "last.pt",
            "source_presentations": source_cursor,
            "clip_presentations": source_cursor * 8,
            "held_out_evaluated": False,
            "wandb_run_id": logger.run_id,
        }
        _write_json(output / "result.json", result)
        logger.log(
            {
                "progress/status": result["status"],
                "progress/optimizer_step": step,
                "result/best_step": best_step,
                "result/best_score": result["best_score"],
            },
            step=step,
        )
        if best_metadata is not None:
            logger.checkpoint(output / "best.pt", metadata=best_metadata)
        return result
    except Exception as exc:
        error = type(exc).__name__
        _write_json(
            output / "failure.json", {"error_type": error, "last_optimizer_step": step}
        )
        raise
    finally:
        deadlines.restore()
        logger.finish(exit_code=1 if error else 0)
