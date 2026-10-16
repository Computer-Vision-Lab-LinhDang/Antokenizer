"""Six-GPU natural RGB trial, frozen dev selection and one best checkpoint."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import re
import time
import uuid

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mavt4d.hybrid_pilot.logging import RunLogger
from mavt4d.hybrid_pilot.runner import (
    DistributedContext, DeadlineController, _broadcast, _write_json, _reduce_metrics,
    _memory_metrics, _elapsed_max, require_finite_loss, rank_zero_call,
    save_checkpoint, load_checkpoint,
)
from .data import NaturalDataset
from .model import NaturalTrialModule, TrialConfig


def save_best_checkpoint(path, module, optimizer, metadata, context):
    """Gather all RNGs, fsync a candidate, then atomically replace only best.pt."""
    path = Path(path)
    candidate = path.with_name(f".best-candidate-{uuid.uuid4().hex}.pt") if context.rank == 0 else None
    candidate = Path(_broadcast(str(candidate) if candidate else None, context))
    save_checkpoint(candidate, module, optimizer, metadata, context)
    rank_zero_call(context, lambda: os.replace(candidate, path))


def _context(allow_cpu):
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not allow_cpu and world != 6:
        raise ValueError("natural trial requires exactly six allocated GPU ranks")
    if not 0 <= rank < world or local < 0:
        raise ValueError("invalid distributed rank")
    device = torch.device("cpu" if allow_cpu else f"cuda:{local}")
    if not allow_cpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 6:
            raise RuntimeError("natural trial requires exactly six visible allocated GPUs")
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo" if allow_cpu else "nccl", timeout=timedelta(seconds=180))
    return DistributedContext(rank, world, local, device)


def _schedule(dataset, cfg, step):
    generator = random.Random(f"natural-trial-v1:{cfg.seed}:{step}")
    return [(generator.choice(dataset.train), generator.randrange(2**63)) for _ in range(cfg.global_clips)]


def _allocated_gpu_metrics():
    from mavt4d.hybrid_pilot.telemetry import collect_gpu_metrics
    visible = os.environ.get("ROCR_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES", ""))
    physical = {int(item) for item in visible.split(",") if item.isdigit()}
    if len(physical) != 6:
        return {"system/telemetry_available": 0.}
    values = collect_gpu_metrics()
    return {key: value for key, value in values.items()
            if key == "system/telemetry_available" or
            ((match := re.match(r"system/gpu_(\d+)/", key)) and int(match[1]) in physical)}


@torch.no_grad()
def _dev(module, dataset, context, deadlines):
    was_training = module.training
    module.eval()
    totals = torch.zeros(7, dtype=torch.float64, device=context.device)
    names = ("loss", "observed_l1", "missing_l1", "future_l1", "history_mse", "history_zero_mse", "clips")
    try:
        rounds = (len(dataset.dev) + context.world_size - 1) // context.world_size
        for round_index in range(rounds):
            # All ranks execute the same rounds, including those with no clip.
            # An interrupted dev pass never selects a partial-set checkpoint.
            if deadlines.should_stop(context, optimization=False):
                return None
            i = round_index * context.world_size + context.rank
            if i >= len(dataset.dev):
                continue
            rgb, times = dataset.batch([dataset.dev[i]], context.device)
            _, metrics = module(rgb, times, [10000 + i])
            values = [float(metrics[n]) for n in names[:-1]] + [1.]
            totals[:7] += torch.tensor(values, dtype=torch.float64, device=context.device)
        if context.distributed:
            dist.all_reduce(totals)
        if totals[6] <= 0:
            raise ValueError("empty development partition")
        return {name: float(totals[i] / totals[6]) for i, name in enumerate(names[:-1])} | {"clips": int(totals[6])}
    finally:
        module.train(was_training)


@torch.no_grad()
def _image_panel(module, dataset, logger, context, step, output):
    if context.rank != 0:
        return
    from PIL import Image
    module.eval()
    rgb, times = dataset.batch([dataset.dev[0]], context.device)
    _, _, panel = module(rgb, times, [10000], return_panel=True)
    array = panel[0].clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    path = output / "images" / f"dev-step-{step:06d}.png"
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(array).save(path)
    if logger.run is not None:
        logger.run.log({"progress/logical_step": step, "samples/dev": logger.sdk.Image(
            array, caption="Fixed dev clip: masked last input | reconstruction | last GT | prefix-only future prediction | future GT")})
    module.train()


def run_trial(dataset_root, output, *, optimization_at, work_at, git_revision,
              microbatch=2, resume=None, allow_cpu=False, cfg=None, max_steps=None):
    cfg = cfg or TrialConfig()
    context = _context(allow_cpu)
    deadlines = DeadlineController(optimization_at, work_at)
    deadlines.install()
    logger = None
    exit_code = 1
    output = Path(output)
    try:
        dataset = NaturalDataset(dataset_root, cfg)
        if microbatch < 1:
            raise ValueError("microbatch must be positive")
        if cfg.global_clips % context.world_size or (cfg.global_clips // context.world_size) % microbatch:
            raise ValueError("microbatch must divide the fixed local clip budget")
        accumulation = cfg.global_clips // context.world_size // microbatch
        lock = {"config_sha256": cfg.fingerprint(), "dataset_sha256": dataset.fingerprint,
                "git_revision": git_revision, "world_size": context.world_size,
                "config": cfg.to_dict(), "phase": "natural_trial"}
        def prepare_output():
            if output.exists() and resume is None:
                raise FileExistsError("fresh natural runs require a new output directory")
            output.mkdir(parents=True, exist_ok=resume is not None)
            if resume is not None and json.loads((output / "run_lock.json").read_text()) != lock:
                raise ValueError("resume run lock differs")
            if resume is None:
                _write_json(output / "run_lock.json", lock)
        rank_zero_call(context, prepare_output)
        torch.manual_seed(cfg.seed)
        module = NaturalTrialModule(cfg).to(context.device)
        if not allow_cpu:
            module.model.observer.spatial_autocast = True
        optimizer = torch.optim.AdamW(module.parameters(), lr=cfg.learning_rate,
                                     betas=(.9, .95), weight_decay=cfg.weight_decay)
        step, best_loss, best_step, run_id = 0, float("inf"), None, None
        if resume is not None:
            metadata = load_checkpoint(resume, module, optimizer, context, expected=lock)
            step, best_loss = metadata["step"], metadata["dev_loss"]
            if metadata["source_cursor"] != step * cfg.global_clips:
                raise ValueError("resume source cursor differs from optimizer boundary")
            best_step, run_id = step, metadata["wandb_run_id"]
        def start_logger():
            nonlocal logger
            logger = RunLogger(output, rank=0, config={**cfg.to_dict(),
                "config_sha256": cfg.fingerprint(), "dataset_sha256": dataset.fingerprint,
                "git_revision": git_revision, "dataset_counts": dataset.summary["counts"],
                "trial_kind": "natural_video_feasibility", "checkpoint_policy": "best_dev_only"},
                run_id=run_id, enabled=not allow_cpu)
            return logger.run_id
        shared_run_id = rank_zero_call(context, start_logger)
        if context.rank != 0:
            logger = RunLogger(output, rank=context.rank, config={}, run_id=shared_run_id, enabled=False)
        if resume is not None:
            def record_resume():
                prior = max((json.loads(line).get("progress/optimizer_step", 0)
                             for line in (output / "progress.jsonl").read_text().splitlines()), default=0)
                logger.log({"resume/from_optimizer_step": step, "resume/prior_observed_optimizer_step": prior,
                            "resume/rewinds_to_best": step < prior}, step=step)
            rank_zero_call(context, record_resume)
        wrapped = DistributedDataParallel(module, device_ids=[context.local_rank] if not allow_cpu else None,
                    find_unused_parameters=True, broadcast_buffers=False) if context.distributed else module
        limit = min(cfg.steps, max_steps) if max_steps is not None else cfg.steps
        if limit < 1:
            raise ValueError("step budget must be positive")
        last_dev, last_eval_step = {}, None

        def evaluate_and_select():
            nonlocal best_loss, best_step, last_dev, last_eval_step
            measured = _dev(module, dataset, context, deadlines)
            if measured is None:
                rank_zero_call(context, lambda: logger.log({"dev/incomplete": True,
                    "progress/optimizer_step": step}, step=step))
                return
            last_dev = measured
            last_eval_step = step
            if not all(torch.isfinite(torch.tensor(value)) for value in last_dev.values()):
                raise FloatingPointError("nonfinite natural development metrics")
            improved = last_dev["loss"] < best_loss
            if improved:
                best_loss, best_step = last_dev["loss"], step
                metadata = {**lock, "step": step, "next_step": step + 1,
                            "source_cursor": step * cfg.global_clips, "dev_loss": best_loss,
                            "wandb_run_id": shared_run_id}
                save_best_checkpoint(output / "best.pt", module, optimizer, metadata, context)
                rank_zero_call(context, lambda: _write_json(output / "best.json", metadata))
            rank_zero_call(context, lambda: logger.log({**{f"dev/{k}": v for k, v in last_dev.items()},
                "checkpoint/best_step": best_step, "checkpoint/best_dev_loss": best_loss,
                "progress/optimizer_step": step, "progress/phase": "natural_trial"}, step=step))
            if step % cfg.image_every == 0:
                rank_zero_call(context, lambda: _image_panel(module, dataset, logger, context, step, output))

        if not deadlines.should_stop(context, optimization=False):
            evaluate_and_select()
        module.train()
        while step < limit and not deadlines.should_stop(context):
            if not allow_cpu:
                torch.cuda.synchronize(context.device)
            started = time.monotonic()
            schedule = _schedule(dataset, cfg, step + 1)
            local_count = cfg.global_clips // context.world_size
            local = schedule[context.rank * local_count:(context.rank + 1) * local_count]
            optimizer.zero_grad(set_to_none=True)
            means = {}
            for index in range(accumulation):
                portion = local[index * microbatch:(index + 1) * microbatch]
                rgb, times = dataset.batch([r[0] for r in portion], context.device)
                syncing = index + 1 == accumulation
                manager = nullcontext() if syncing or not context.distributed else wrapped.no_sync()
                with manager:
                    loss, metrics = wrapped(rgb, times, [r[1] for r in portion])
                    require_finite_loss(loss, context)
                    (loss / accumulation).backward()
                for name, value in metrics.items():
                    means[name] = means.get(name, 0.) + value / accumulation
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if not allow_cpu:
                torch.cuda.synchronize(context.device)
            elapsed = _elapsed_max(time.monotonic() - started, context)
            if step % cfg.log_every == 0 or step == 1:
                metrics = _reduce_metrics(means | {"gradient_norm": norm.detach()}, context)
                memory = _memory_metrics(context)
                rank_zero_call(context, lambda: logger.log({**{f"train/{k}": v for k, v in metrics.items()},
                    **memory, **({} if allow_cpu else _allocated_gpu_metrics()), "system/seconds_per_update": elapsed,
                    "system/clips_per_second": cfg.global_clips / elapsed,
                    "progress/optimizer_step": step, "progress/phase": "natural_trial"}, step=step))
            if step % cfg.dev_every == 0 and not deadlines.should_stop(context, optimization=False):
                evaluate_and_select()
        if step != last_eval_step and not deadlines.should_stop(context, optimization=False):
            evaluate_and_select()
        result = {"status": "complete" if step == cfg.steps else "stopped", "step": step,
                  "planned_steps": cfg.steps, "best_step": best_step, "best_dev_loss": best_loss if best_step is not None else None,
                  "last_dev": last_dev, "config_sha256": cfg.fingerprint(),
                  "dataset_sha256": dataset.fingerprint, "wandb_run_id": shared_run_id,
                  "held_out": "not_run; development-only feasibility trial"}
        rank_zero_call(context, lambda: _write_json(output / "result.json", result))
        if best_step is not None and not deadlines.should_stop(context, optimization=False):
            metadata = {"phase": "natural_trial", "step": best_step,
                        "dev_loss": best_loss, "config_sha256": cfg.fingerprint(), "dataset_sha256": dataset.fingerprint}
            rank_zero_call(context, lambda: logger.checkpoint(output / "best.pt", metadata=metadata))
        exit_code = 0
        return result
    finally:
        deadlines.restore()
        if dist.is_initialized():
            dist.destroy_process_group()
        if logger is not None:
            logger.finish(exit_code=exit_code)
