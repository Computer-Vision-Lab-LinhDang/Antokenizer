"""Fresh single-GPU B trial with full fixed-dev selection and best-only saving."""

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
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
    save_checkpoint,
    scheduled_step,
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
    os.replace(candidate, Path(output) / "best.pt")
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
    if not allow_cpu and cfg.effective_source_groups != 8:
        raise ValueError("production trial freezes eight effective source groups")
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
        for microbatch in cfg.microbatch_candidates:
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
                    "records": records,
                    "selected_microbatch": choose_capacity(records),
                    "held_out_evaluated": False,
                },
            )
            print(json.dumps(record), flush=True)
        return {
            "records": records,
            "selected_microbatch": choose_capacity(records),
            "config_sha256": cfg.sha256,
        }
    finally:
        deadlines.restore()


def choose_capacity(records):
    eligible = [
        row
        for row in records
        if row.get("status") == "complete" and row.get("headroom_ok")
    ]
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
):
    context = validate_device(device, allow_cpu=allow_cpu)
    if not allow_cpu and cfg.effective_source_groups != 8:
        raise ValueError("production trial freezes eight effective source groups")
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
    _write_json(output / "run_lock.json", lock)
    logger = RunLogger(output, rank=0, config=lock, enabled=not allow_cpu)
    if logger.run is not None:
        logger.run.tags = ("loss-v3", "B", "single-gpu", "synthetic")
    deadlines.install()
    step, best_step, last_dev_step = 0, None, None
    best_score, last_dev, best_metadata = float("inf"), None, None
    error = None
    try:
        module = initialize(base_config, context.device, spatial_bf16=spatial_bf16)
        optimizer = _optimizer(module, base_config)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        latest_eval_seconds = latest_step_seconds = 0.0

        def evaluate(current_step, *, force_image=False):
            nonlocal best_score, best_step, last_dev_step, last_dev, best_metadata, latest_eval_seconds
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
            metadata = {
                **lock,
                "phase": "loss-v3-B",
                "step": current_step,
                "next_step": current_step + 1,
                "source_cursor": current_step * cfg.effective_source_groups,
                "wandb_run_id": logger.run_id,
            }
            previous = best_score
            best_score = save_selected(
                output, module, optimizer, scores, best_score, metadata, context
            )
            if best_score < previous:
                best_step, best_metadata = current_step, {
                    **metadata,
                    "selection_score": best_score,
                }
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

        if not evaluate(0):
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
                break
            latest_step_seconds = time.monotonic() - started
            step += 1
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
                        "progress/source_presentations": step
                        * cfg.effective_source_groups,
                        "progress/clip_presentations": step
                        * cfg.effective_source_groups
                        * 8,
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
            if step % cfg.dev_every == 0 and not evaluate(step):
                break
        if last_dev_step != step and not deadlines.should_stop(
            context, optimization=False
        ):
            evaluate(step, force_image=True)
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
