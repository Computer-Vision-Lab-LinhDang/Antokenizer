"""Fresh two-GPU C/D training after a separately reviewed diagnostic arm."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mavt4d.data.event_synthetic import EVENT_CORES, EVENT_CONFIGURATIONS
from mavt4d.hybrid_pilot.data import HybridPilotDataset, file_sha256
from mavt4d.hybrid_pilot.logging import RunLogger
from mavt4d.hybrid_pilot.runner import (
    DeadlineController, DistributedContext, _broadcast, _elapsed_max, _memory_metrics, _reduce_metrics,
    _seed, _write_json, initialize_module, rank_zero_call, require_finite_loss,
    save_checkpoint, scheduled_step,
)
from mavt4d.pilot.evaluation import _evaluation_records
from .experiment import DiagnosticConfig, apply_intervention, measure
from .runner import write_panel
from .telemetry import selected_gpu_metrics


@dataclass(frozen=True)
class LongRunConfig:
    arm: str
    version: str = "hybrid-cd-longrun-v1"
    target_steps: int = 10000
    max_steps: int = 20000
    extension_steps: int = 5000
    extension_safety: float = 1.15
    dev_every: int = 200
    image_every: int = 1000
    log_every: int = 10
    selection: str = "fixed_dev_weighted_loss"

    def __post_init__(self):
        DiagnosticConfig(arm=self.arm)
        if self.version != "hybrid-cd-longrun-v1" or self.selection != "fixed_dev_weighted_loss":
            raise ValueError("unsupported long-run protocol")
        for name in ("target_steps", "max_steps", "extension_steps", "dev_every", "image_every", "log_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_steps > self.max_steps:
            raise ValueError("target_steps exceeds max_steps")
        if not math.isfinite(self.extension_safety) or self.extension_safety < 1:
            raise ValueError("extension_safety must be finite and >=1")

    def to_dict(self):
        return {**asdict(self), "world_size": 2, "global_source_groups": 2,
                "clips_per_update": 16, "microbatch_groups_per_rank": 1,
                "fresh_initialization": True, "held_out_evaluation": False}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def extend_budget(cfg, current_target, *, remaining_seconds, seconds_per_update):
    """Only measured wall time enters extension decisions, never model quality."""
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0:
        return current_target
    candidate = min(current_target + cfg.extension_steps, cfg.max_steps)
    required = (candidate-current_target) * seconds_per_update * cfg.extension_safety
    return candidate if remaining_seconds >= required else current_target


def save_distributed_best(output, module, optimizer, *, score, best_score, metadata, context):
    if not math.isfinite(score):
        raise FloatingPointError("nonfinite development score")
    if score >= best_score:
        return best_score
    filename = _broadcast(f".best-{uuid.uuid4().hex}.pt" if context.rank == 0 else None, context)
    candidate = Path(output) / filename
    save_checkpoint(candidate, module, optimizer,
        {**metadata, "selection_metric": "fixed_dev_weighted_loss", "selection_score": score}, context)
    rank_zero_call(context, lambda: os.replace(candidate, Path(output) / "best.pt"))
    return score


def _context(allow_cpu):
    rank, local, world = (int(os.environ.get(name, default))
                          for name, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    if world != 2 or not 0 <= rank < 2 or not 0 <= local < 2:
        raise ValueError("long-run production and correctness paths both require exactly two ranks")
    device = torch.device("cpu" if allow_cpu else f"cuda:{local}")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise ValueError("long-run training requires exactly two visible allocated GPUs")
        torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("gloo" if allow_cpu else "nccl", timeout=timedelta(seconds=180))
    return DistributedContext(rank, world, local, device)


@torch.no_grad()
def development(module, dataset, cfg, context, deadlines):
    """Equal-source aggregation across the complete fixed dev partition."""
    groups = dataset.dev_groups[context.rank::context.world_size]
    names = ("loss", "rec", "missing", "future", "temporal", "invariance", "noncollapse",
             "reconstruction_l1", "temporal_mse", "temporal_zero_mse", "temporal_cosine")
    sums = torch.zeros(len(names)+1, device=context.device, dtype=torch.float64)
    completed = True
    was_training = module.training
    module.eval()
    try:
        for group in groups:
            if deadlines.requested or datetime.now(timezone.utc) >= deadlines.work_at:
                completed = False
                break
            batch = dataset.load_group(group, context.device)
            loss, metrics = module([batch], [_seed(cfg.seed, "longrun-fixed-dev", group)], phase="main")
            values = {**metrics, "loss": loss}
            sums[:-1] += torch.stack([values[name].detach().double() for name in names])
            sums[-1] += 1
    finally:
        module.train(was_training)
    dist.all_reduce(sums)
    complete = torch.tensor(int(completed), device=context.device)
    dist.all_reduce(complete, op=dist.ReduceOp.MIN)
    if not bool(complete) or int(sums[-1]) != len(dataset.dev_groups):
        return None
    values = sums[:-1]/sums[-1]
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("nonfinite fixed development objective")
    return dict(zip(names, values.cpu().tolist()))


def run_long(base_config, cfg: LongRunConfig, dataset_root, output, *, optimization_at,
             work_at, git_revision, spatial_bf16=True, allow_cpu=False):
    """No automatic arm choice and no pilot/diagnostic-weight continuation."""
    deadlines = DeadlineController(optimization_at, work_at)
    if datetime.now(timezone.utc) >= optimization_at:
        raise ValueError("optimization deadline has already passed")
    if len(git_revision) != 40 or any(c not in "0123456789abcdef" for c in git_revision):
        raise ValueError("git_revision must be a complete lowercase commit SHA")
    context = _context(allow_cpu)
    output = Path(output)
    logger = None
    success = False
    try:
        dataset = HybridPilotDataset(dataset_root, base_config)
        if not dataset.dev_groups:
            raise ValueError("long-run selection requires a nonempty fixed dev partition")
        lock = {"schema": cfg.version, "base_config": base_config.to_dict(),
                "base_config_sha256": base_config.config_sha256, "longrun": cfg.to_dict(),
                "longrun_sha256": cfg.sha256, "dataset_sha256": file_sha256(dataset.root / "summary.json"),
                "git_revision": git_revision, "spatial_bf16": spatial_bf16,
                "dev_source_groups": list(dataset.dev_groups), "world_size": 2,
                "optimization_at": optimization_at.isoformat(), "work_at": work_at.isoformat()}
        def prepare():
            output.mkdir(parents=True, exist_ok=False)
            _write_json(output / "run_lock.json", lock)
        rank_zero_call(context, prepare)
        startup = None
        try:
            logger = RunLogger(output, rank=context.rank, config=lock, enabled=not allow_cpu)
            if logger.run is not None:
                logger.run.tags = ("synthetic-longrun", cfg.arm, cfg.version)
        except Exception as exc:
            startup = f"W&B startup failed: {type(exc).__name__}"
        startup = _broadcast(startup, context)
        if startup:
            raise RuntimeError(startup)
        logger.run_id = _broadcast(logger.run_id, context)
        deadlines.install()
        module = initialize_module(base_config, context.device, spatial_bf16=spatial_bf16)
        apply_intervention(module, DiagnosticConfig(arm=cfg.arm))
        random.seed(base_config.seed+context.rank)
        np.random.seed((base_config.seed+context.rank) % 2**32)
        torch.manual_seed(base_config.seed+context.rank)
        optimizer = torch.optim.AdamW(module.parameters(), lr=base_config.learning_rate,
                                      betas=base_config.betas, weight_decay=base_config.weight_decay)
        wrapped = DistributedDataParallel(module,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            find_unused_parameters=True, broadcast_buffers=False)
        best_score, best_step = float("inf"), None
        best_metadata, last_dev, last_dev_step = None, None, None
        step, target = 0, cfg.target_steps
        update_times = deque(maxlen=100)
        latest_dev_seconds = 0.

        def report_images(current_step):
            def one_rank():
                group = dataset.dev_groups[0]
                batch = dataset.load_group(group, context.device)
                labels = _evaluation_records(dataset)
                pairs = [(EVENT_CORES.index(labels[r["record_id"]]["core"]),
                          EVENT_CONFIGURATIONS.index(labels[r["record_id"]]["configuration"]))
                         for r in dataset.records_for_group(group)]
                metrics, panel = measure(module, batch, pairs, gradients=False, images=True)
                logger.log({f"dev_image_source/{k}": v for k, v in metrics.items()}, step=current_step)
                write_panel(logger, panel, step=current_step, prefix="development")
            rank_zero_call(context, one_rank)

        def evaluate_and_save(current_step):
            nonlocal best_score, best_step, best_metadata, last_dev, last_dev_step, latest_dev_seconds
            started = time.monotonic()
            scores = development(module, dataset, base_config, context, deadlines)
            if scores is None:
                return False
            last_dev, last_dev_step = scores, current_step
            metadata = {**lock, "phase": "synthetic_longrun", "step": current_step,
                        "next_step": current_step+1, "source_cursor": current_step*2,
                        "wandb_run_id": logger.run_id, "target_steps": target}
            if scores["loss"] < best_score:
                best_score = save_distributed_best(output, module, optimizer, score=scores["loss"],
                    best_score=best_score, metadata=metadata, context=context)
                best_step = current_step
                best_metadata = {**metadata, "selection_score": best_score,
                                 "selection_metric": "fixed_dev_weighted_loss"}
            def report():
                logger.log({**{f"dev/{key}": value for key, value in scores.items()},
                            "progress/optimizer_step": current_step,
                            "selection/best_step": best_step, "selection/best_dev_loss": best_score}, step=current_step)
                _write_json(output / "development-latest.json", {"step": current_step, "metrics": scores,
                            "best_step": best_step, "best_score": best_score})
            rank_zero_call(context, report)
            latest_dev_seconds = _elapsed_max(time.monotonic()-started, context)
            return True

        if not deadlines.should_stop(context, optimization=False):
            evaluate_and_save(0)
        if not deadlines.should_stop(context, optimization=False):
            report_images(0)
        last_metrics = {}
        module.train()
        while step < target and not deadlines.should_stop(context):
            next_step = step+1
            schedule = scheduled_step(dataset.train_groups, seed=base_config.seed, step=next_step,
                                      global_groups=2)
            group, objective_seed = schedule[context.rank]
            batch = dataset.load_group(group, context.device)
            optimizer.zero_grad(set_to_none=True)
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            started = time.monotonic()
            loss, components = wrapped([batch], [objective_seed], phase="main")
            require_finite_loss(loss, context)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), base_config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            step = next_step
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            elapsed = _elapsed_max(time.monotonic()-started, context)
            update_times.append(elapsed)
            if step == 1 or step % cfg.log_every == 0 or step == target:
                last_metrics = _reduce_metrics({**components, "gradient_norm": norm}, context)
                memory = _memory_metrics(context)
                def report_training():
                    telemetry = selected_gpu_metrics() if context.device.type == "cuda" and step % 100 == 0 else {}
                    logger.log({**{f"train/{key}": value for key, value in last_metrics.items()},
                                "progress/optimizer_step": step, "progress/target_steps": target,
                                "performance/step_seconds": elapsed,
                                "performance/clips_per_second": 16/elapsed, **memory, **telemetry}, step=step)
                    print(json.dumps({"phase": "synthetic_longrun", "step": step,
                                      "target_steps": target, "loss": last_metrics["loss"],
                                      "step_seconds": elapsed}), flush=True)
                rank_zero_call(context, report_training)
            if (step % cfg.dev_every == 0 or step == target) and not deadlines.should_stop(context, optimization=False):
                evaluate_and_save(step)
            if step % cfg.image_every == 0 and not deadlines.should_stop(context, optimization=False):
                report_images(step)
            if step == target and target < cfg.max_steps:
                new_target = None
                if context.rank == 0:
                    # Include the last measured evaluation/checkpoint overhead
                    # amortized across updates before applying the safety factor.
                    seconds_per_update = sum(update_times)/len(update_times) + latest_dev_seconds/cfg.dev_every
                    new_target = extend_budget(cfg, target,
                        remaining_seconds=(optimization_at-datetime.now(timezone.utc)).total_seconds(),
                        seconds_per_update=seconds_per_update)
                new_target = _broadcast(new_target, context)
                if new_target != target:
                    target = new_target
                    rank_zero_call(context, lambda: logger.log({"progress/target_steps": target,
                        "budget/extension_reason": "measured_time_only", "progress/optimizer_step": step}, step=step))
        if step != last_dev_step and not deadlines.should_stop(context, optimization=False):
            evaluate_and_save(step)
        if not deadlines.should_stop(context, optimization=False):
            report_images(step)
        result = {"status": "complete" if step == target else "stopped", "step": step,
                  "target_steps": target, "maximum_steps": cfg.max_steps, "arm": cfg.arm,
                  "longrun_sha256": cfg.sha256, "best_step": best_step,
                  "best_dev_loss": best_score if math.isfinite(best_score) else None,
                  "last_dev_step": last_dev_step, "last_dev": last_dev,
                  "last_training_metrics": last_metrics, "checkpoint": "best.pt" if best_metadata else None,
                  "held_out_evaluated": False, "wandb_run_id": logger.run_id}
        rank_zero_call(context, lambda: _write_json(output / "result.json", result))
        rank_zero_call(context, lambda: logger.log({"progress/status": result["status"],
                        "progress/optimizer_step": step}, step=step))
        if best_metadata is not None:
            rank_zero_call(context, lambda: logger.checkpoint(output / "best.pt", metadata=best_metadata))
        success = True
        return result
    finally:
        deadlines.restore()
        if dist.is_initialized():
            dist.destroy_process_group()
        if logger is not None:
            logger.finish(exit_code=0 if success else 1)
