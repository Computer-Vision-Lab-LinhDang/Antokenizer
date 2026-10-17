"""Bounded six-rank batch capacity probe; never produces training checkpoints."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector

from mavt4d.hybrid_pilot.runner import _write_json, rank_zero_call
from .data import NaturalDataset
from .model import NaturalTrialModule, TrialConfig
from .runner import _context


def choose_candidate(rows, *, throughput_fraction=.9):
    good = [r for r in rows if r.get("status") == "stable" and r["peak_fraction"] <= .85]
    if not good:
        return None
    cutoff = max(r["clips_per_second"] for r in good)*throughput_fraction
    return max(r["batch_per_gpu"] for r in good if r["clips_per_second"] >= cutoff)


def _local_phase(operation, cleanup, flag):
    """Coordinate every allocating local phase before the next collective."""
    failed = False
    try:
        operation()
    except (torch.OutOfMemoryError, FloatingPointError, MemoryError):
        failed = True
        cleanup()  # Release local partial graphs/buffers before communication.
    flag.fill_(int(failed))
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if bool(flag):
        cleanup()  # Successful peers also discard this failed candidate.
        return False
    return True


def probe(dataset_root, output, *, deadline, git_revision, candidates=(8, 32, 128, 256, 512)):
    context = _context(False)
    output = Path(output)
    rows = []
    try:
        rank_zero_call(context, lambda: output.mkdir(parents=True, exist_ok=False))
        cfg = TrialConfig()
        dataset = NaturalDataset(dataset_root, cfg)
        torch.manual_seed(cfg.seed)
        module = NaturalTrialModule(cfg).to(context.device)
        module.model.observer.spatial_autocast = True
        optimizer = torch.optim.AdamW(module.parameters(), lr=cfg.learning_rate, betas=(.9,.95), weight_decay=cfg.weight_decay)
        # Cycling audited real frames is calibration only, never a training
        # manifest or an enlarged scientific dataset.
        rgb0, times0 = dataset.batch([dataset.train[context.rank % len(dataset.train)]], context.device)
        flag = torch.zeros((), device=context.device, dtype=torch.int32)
        elapsed = torch.zeros((), device=context.device, dtype=torch.float64)
        stats = torch.zeros(5, device=context.device, dtype=torch.float64)
        gathered = [torch.zeros_like(stats) for _ in range(context.world_size)]
        dist.all_reduce(flag)  # Initialize status communication before any probe.
        params = list(module.parameters())
        previous_peak, previous_batch, previous_seconds = 0., 1, 0.
        for batch in candidates:
            flag.fill_(int(datetime.now(timezone.utc) >= deadline))
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if bool(flag):
                break
            remaining = (deadline-datetime.now(timezone.utc)).total_seconds()
            flag.fill_(int(previous_seconds == 0 or
                previous_seconds*batch/previous_batch*1.3 + 15 < remaining))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not bool(flag):
                rows.append({"batch_per_gpu": batch, "status": "skipped_time_estimate"})
                break
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(context.device)
            base = torch.cuda.memory_allocated(context.device)
            predicted = base + max(0., previous_peak-base)*batch/previous_batch if previous_peak else base
            flag.fill_(int(predicted < .85*(free+base)))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not bool(flag):
                rows.append({"batch_per_gpu": batch, "status": "skipped_memory_estimate"})
                break
            seeds = [100000+context.rank*10000+i for i in range(batch)]
            torch.cuda.reset_peak_memory_stats(context.device)
            torch.cuda.synchronize(context.device)
            started = time.monotonic()
            storage = {}
            def cleanup():
                storage.clear()
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            def allocate():
                storage["rgb"] = rgb0.repeat(batch,1,1,1,1)
                storage["times"] = times0.repeat(batch,1)
            valid = _local_phase(allocate, cleanup, flag)
            interrupted = False
            for _ in range(2):
                if not valid:
                    break
                flag.fill_(int(datetime.now(timezone.utc) >= deadline))
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if bool(flag):
                    valid, interrupted = False, True
                    cleanup()
                    break
                optimizer.zero_grad(set_to_none=True)
                def backward():
                    loss, _ = module(storage["rgb"], storage["times"], seeds)
                    storage["loss"] = loss
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("nonfinite batch calibration loss")
                    loss.backward()
                    del storage["loss"]
                if not _local_phase(backward, cleanup, flag):
                    valid = False
                    break
                def flatten():
                    storage["flat"] = parameters_to_vector([
                        p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
                if not _local_phase(flatten, cleanup, flag):
                    valid = False
                    break
                dist.all_reduce(storage["flat"])
                def update():
                    flat = storage["flat"]
                    flat /= context.world_size
                    offset = 0
                    for parameter in params:
                        size = parameter.numel()
                        parameter.grad = flat[offset:offset+size].view_as(parameter)
                        offset += size
                    norm = torch.nn.utils.clip_grad_norm_(params,1.)
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("nonfinite batch calibration gradient")
                    optimizer.step()
                    del storage["flat"]
                if not _local_phase(update, cleanup, flag):
                    valid = False
                    break
            torch.cuda.synchronize(context.device)
            elapsed.fill_(time.monotonic()-started)
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            peak = torch.cuda.max_memory_allocated(context.device)
            reserved = torch.cuda.max_memory_reserved(context.device)
            stats.copy_(torch.tensor([peak, reserved, free, total, peak/total],dtype=torch.float64))
            dist.all_gather(gathered, stats)
            row = {"batch_per_gpu": batch, "global_clips": batch*6,
                   "status": "stable" if valid else ("interrupted_deadline" if interrupted else "failed_oom_or_nonfinite"),
                   "seconds_per_update": float(elapsed)/2 if valid else None,
                   "clips_per_second": batch*6*2/float(elapsed) if valid else 0.,
                   "peak_fraction": max(float(item[4]) for item in gathered),
                   "ranks": [{"rank":i,"peak_allocated":float(item[0]),"peak_reserved":float(item[1]),
                              "free_before":float(item[2]),"total":float(item[3])} for i,item in enumerate(gathered)]}
            rows.append(row)
            report = {"schema":"natural-batch-calibration-v1","git_revision":git_revision,
                      "requested_batches":list(candidates),"deadline":deadline.isoformat(),
                      "dataset_sha256":dataset.fingerprint,"candidates":rows,"chosen_batch_per_gpu":choose_candidate(rows),
                      "learning_rate":cfg.learning_rate,"calibration_only":True,
                      "distributed_probe":"six-rank explicit averaged gradients; not DDP overlap timing"}
            rank_zero_call(context,lambda:_write_json(output/"result.json",report))
            if context.rank == 0:
                print(json.dumps(row),flush=True)
            cleanup()
            if not valid:
                break
            previous_peak, previous_batch, previous_seconds = peak, batch, float(elapsed)
        report = {"schema":"natural-batch-calibration-v1","git_revision":git_revision,
                  "requested_batches":list(candidates),"deadline":deadline.isoformat(),
                  "dataset_sha256":dataset.fingerprint,"candidates":rows,"chosen_batch_per_gpu":choose_candidate(rows),
                  "learning_rate":cfg.learning_rate,"calibration_only":True,
                  "distributed_probe":"six-rank explicit averaged gradients; not DDP overlap timing"}
        rank_zero_call(context,lambda:_write_json(output/"result.json",report))
        return report
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
