"""One GPU per mechanism arm, with immutable run locks and best-only retention."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from mavt4d.data.event_synthetic import EVENT_CORES, EVENT_CONFIGURATIONS
from mavt4d.hybrid_pilot.data import HybridPilotDataset, file_sha256
from mavt4d.hybrid_pilot.logging import RunLogger
from mavt4d.hybrid_pilot.runner import (
    DeadlineController, DistributedContext, _write_json, initialize_module,
    require_finite_loss, scheduled_step,
)
from mavt4d.pilot.evaluation import _evaluation_records
from .experiment import DiagnosticConfig, apply_intervention, measure, save_best
from .telemetry import selected_gpu_metrics


def write_panel(logger, panel, *, step, prefix="diagnostic"):
    """Keep small visual evidence locally and online; never upload source RGB files."""
    if panel is None:
        return
    from PIL import Image
    values = (panel.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
    destination = logger.output / "images"
    destination.mkdir(exist_ok=True)
    image = Image.fromarray(values)
    image.save(destination / f"{prefix}-{step:06d}.jpg", quality=85)
    if logger.run is not None:
        caption = ("Rows: same source/core, opposite configurations. Columns: masked input t=3; "
                   "masked reconstruction t=3; target t=3; future prediction t=7 from prefix through5; "
                   "target t=7; D-swap reconstruction t=3. Values display-clipped to [-1,1].")
        logger.run.log({f"{prefix}/images": logger.sdk.Image(image, caption=caption),
                        "progress/logical_step": step, "progress/optimizer_step": step})


def run_diagnostic(base_config, diagnostic: DiagnosticConfig, dataset_root, output, *,
                   optimization_at, work_at, git_revision, device="cuda:0", spatial_bf16=True,
                   audit_checkpoints=(), allow_cpu=False):
    """Fresh standalone arm; intentionally no resume or held-out tuning interface."""
    device = torch.device(device)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("diagnostic arms each require a standalone single GPU process")
    if not allow_cpu and (device.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("production diagnostic requires an allocated GPU")
    if allow_cpu and device.type != "cpu":
        raise ValueError("CPU test mode must use the CPU")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if len(git_revision) != 40 or any(c not in "0123456789abcdef" for c in git_revision):
        raise ValueError("git_revision must be a full lowercase commit SHA")
    deadlines = DeadlineController(optimization_at, work_at)
    if datetime.now(timezone.utc) >= optimization_at:
        raise ValueError("optimization deadline has already passed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = HybridPilotDataset(dataset_root, base_config)
    group = dataset.overfit_group
    batch = dataset.load_group(group, device=device)
    records = _evaluation_records(dataset)
    labels = [(EVENT_CORES.index(records[r["record_id"]]["core"]),
               EVENT_CONFIGURATIONS.index(records[r["record_id"]]["configuration"]))
              for r in dataset.records_for_group(group)]
    lock = {"schema": "hybrid-cd-diagnostic-v1", "base_config": base_config.to_dict(),
            "base_config_sha256": base_config.config_sha256,
            "diagnostic": diagnostic.to_dict(), "diagnostic_sha256": diagnostic.sha256,
            "dataset_sha256": file_sha256(dataset.root / "summary.json"),
            "git_revision": git_revision, "world_size": 1, "training_source": group,
            "spatial_bf16": spatial_bf16, "selection_scope": "same eight tiny training variants",
            "optimization_at": optimization_at.isoformat(), "work_at": work_at.isoformat()}
    _write_json(output / "run_lock.json", lock)
    logger = RunLogger(output, rank=0, config=lock, enabled=not allow_cpu)
    if logger.run is not None:
        logger.run.tags = ("diagnostic", diagnostic.arm, diagnostic.version)
    context = DistributedContext(0, 1, device.index or 0, device)
    deadlines.install()
    step, best_step, best_score = 0, 0, float("inf")
    best_metadata = None
    error = None
    try:
        # Only inspect old snapshots. Their files and optimizer state are never
        # modified, and their trained weights never initialize a new arm.
        for index, checkpoint in enumerate(audit_checkpoints):
            if deadlines.should_stop(context):
                break
            audit_module = initialize_module(base_config, device, spatial_bf16=spatial_bf16)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            audit_module.load_state_dict(payload["module"], strict=True)
            metrics, panel = measure(audit_module, batch, labels, gradients=True, images=True)
            evidence = {"checkpoint_sha256": file_sha256(Path(checkpoint)),
                        "source_phase": payload["metadata"].get("phase"),
                        "source_step": payload["metadata"].get("step"), "metrics": metrics}
            _write_json(output / f"audit-{index}.json", evidence)
            logger.log({f"audit_{index}/{k}": v for k, v in metrics.items()}, step=0)
            write_panel(logger, panel, step=0, prefix=f"audit-{index}")
            del payload, audit_module
        module = initialize_module(base_config, device, spatial_bf16=spatial_bf16)
        apply_intervention(module, diagnostic)
        random.seed(base_config.seed)
        np.random.seed(base_config.seed % 2**32)
        torch.manual_seed(base_config.seed)
        optimizer = torch.optim.AdamW(module.parameters(), lr=base_config.learning_rate,
                                      betas=base_config.betas, weight_decay=base_config.weight_decay)

        def record_probe(current_step):
            nonlocal best_score, best_step, best_metadata
            metrics, panel = measure(module, batch, labels, gradients=True,
                images=current_step == 0 or current_step % diagnostic.image_every == 0 or current_step == diagnostic.steps)
            logger.log({**{f"diagnostic/{k}": v for k, v in metrics.items()},
                        "progress/optimizer_step": current_step, "progress/arm": diagnostic.arm,
                        **(selected_gpu_metrics() if device.type == "cuda" else {})}, step=current_step)
            write_panel(logger, panel, step=current_step)
            metadata = {**lock, "phase": diagnostic.arm, "step": current_step,
                        "next_step": current_step+1, "source_cursor": current_step,
                        "wandb_run_id": logger.run_id}
            score = metrics["history/mse"]
            if score < best_score:
                best_score = save_best(output, module, optimizer, score=score,
                    best_score=best_score, metadata=metadata)
                best_step, best_metadata = current_step, {**metadata, "selection_score": score}
            _write_json(output / "probe-latest.json", {"step": current_step, "metrics": metrics,
                        "best_step": best_step, "best_score": best_score})
            return metrics

        initial = record_probe(0)
        last_probe, last_probe_step = initial, 0
        last_metrics = {}
        module.train()
        while step < diagnostic.steps and not deadlines.should_stop(context):
            next_step = step + 1
            objective_seed = scheduled_step(dataset.train_groups, seed=base_config.seed, step=next_step,
                global_groups=1, overfit_group=group)[0][1]
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.monotonic()
            loss, components = module([batch], [objective_seed], phase="tiny")
            require_finite_loss(loss, context)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), base_config.grad_clip,
                                                 error_if_nonfinite=True)
            optimizer.step()
            step = next_step
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic()-started
            last_metrics = {k: float(v) for k, v in components.items()}
            if step == 1 or step % 5 == 0 or step == diagnostic.steps:
                values = {f"train/{k}": v for k, v in last_metrics.items()}
                values.update({"train/gradient_norm": float(norm), "performance/step_seconds": elapsed,
                               "performance/clips_per_second": 8/elapsed,
                               "progress/optimizer_step": step, "progress/target_steps": diagnostic.steps,
                               "progress/arm": diagnostic.arm})
                if device.type == "cuda":
                    values["system/allocated_gib"] = torch.cuda.memory_allocated(device)/2**30
                    values["system/reserved_gib"] = torch.cuda.memory_reserved(device)/2**30
                logger.log(values, step=step)
                print(json.dumps({"arm": diagnostic.arm, "step": step, "loss": float(loss.detach()),
                                  "step_seconds": elapsed}), flush=True)
            if (step % diagnostic.probe_every == 0 or step == diagnostic.steps) and not deadlines.should_stop(context, optimization=False):
                optimizer.zero_grad(set_to_none=True)
                last_probe = record_probe(step)
                last_probe_step = step
        if step != last_probe_step and not deadlines.should_stop(context, optimization=False):
            optimizer.zero_grad(set_to_none=True)
            last_probe = record_probe(step)
            last_probe_step = step
        result = {"status": "complete" if step == diagnostic.steps else "stopped",
                  "step": step, "target_steps": diagnostic.steps, "arm": diagnostic.arm,
                  "diagnostic_sha256": diagnostic.sha256, "initial": initial,
                  "last_probe_step": last_probe_step, "last_probe": last_probe,
                  "last_training_metrics": last_metrics, "best_step": best_step,
                  "best_history_mse": best_score, "checkpoint": "best.pt",
                  "held_out_evaluated": False, "wandb_run_id": logger.run_id}
        _write_json(output / "result.json", result)
        logger.log({"progress/status": result["status"], "progress/optimizer_step": step,
                    "result/best_step": best_step, "result/best_history_mse": best_score}, step=step)
        # Local snapshots update throughout; W&B gets exactly the selected final
        # artifact rather than accumulating versions of intermediate snapshots.
        if best_metadata is not None:
            logger.checkpoint(output / "best.pt", metadata=best_metadata)
        return result
    except Exception as exc:
        error = type(exc).__name__
        _write_json(output / "failure.json", {"error_type": error, "last_optimizer_step": step})
        raise
    finally:
        deadlines.restore()
        logger.finish(exit_code=1 if error else 0)
