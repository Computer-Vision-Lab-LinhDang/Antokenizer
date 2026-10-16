"""Deadline-bounded distributed optimization for the locked hybrid pilot.

Every DDP call encloses the complete objective. Source and augmentation streams
are indexed by global optimizer step, independent of microbatch packing. A
checkpoint is written only at a common optimizer boundary and includes every
rank's RNG state. There is no automatic CPU fallback in production.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import time
from types import FrameType
from typing import Any, Mapping, Sequence, TYPE_CHECKING
import uuid

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .logging import RunLogger

if TYPE_CHECKING:
    from .config import HybridPilotConfig
    from .data import HybridPilotDataset
    from .objectives import HybridPilotModule


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()


def initialize_distributed(*, allow_cpu: bool = False) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu" if allow_cpu else f"cuda:{local}")
    context = DistributedContext(rank, world, local, device)
    validate_execution(context, allow_cpu=allow_cpu)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("gloo" if allow_cpu else "nccl", timeout=timedelta(seconds=180))
    return context


def validate_execution(context: DistributedContext, *, allow_cpu: bool) -> None:
    if not allow_cpu and (context.world_size != 8 or context.device.type != "cuda"):
        raise ValueError("production pilot requires exactly eight GPU ranks")
    if context.device.type == "cuda" and (
            not torch.cuda.is_available() or torch.cuda.device_count() != 8):
        raise RuntimeError("production pilot requires eight visible GPUs on every rank")
    if not 0 <= context.rank < context.world_size or context.world_size < 1:
        raise ValueError("invalid distributed rank/world size")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


class DeadlineController:
    """Signals set a flag; only common boundaries perform collective decisions."""
    def __init__(self, optimization_at: datetime, work_at: datetime) -> None:
        if optimization_at.tzinfo is None or work_at.tzinfo is None:
            raise ValueError("deadlines require explicit timezones")
        if optimization_at >= work_at:
            raise ValueError("optimization deadline must be before work deadline")
        self.optimization_at = optimization_at
        self.work_at = work_at
        self.requested = False
        self._previous: dict[int, Any] = {}

    def handle_signal(self, signum: int, frame: FrameType | None) -> None:
        self.requested = True

    def install(self) -> None:
        for number in (signal.SIGTERM, signal.SIGINT):
            self._previous[number] = signal.signal(number, self.handle_signal)

    def restore(self) -> None:
        for number, handler in self._previous.items():
            signal.signal(number, handler)
        self._previous.clear()

    def should_stop(self, context: DistributedContext, *, optimization: bool = True) -> bool:
        cutoff = self.optimization_at if optimization else self.work_at
        stop = self.requested or datetime.now(timezone.utc) >= cutoff
        flag = torch.tensor(int(stop), device=context.device, dtype=torch.int32)
        if context.distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


def _seed(seed: int, *parts: Any) -> int:
    value = ":".join(str(part) for part in ("hybrid-pilot-v1", seed, *parts))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % 2**63


def scheduled_step(groups: Sequence[str], *, seed: int, step: int,
                   global_groups: int, overfit_group: str | None = None) -> tuple[tuple[str, int], ...]:
    if not groups or step < 1 or global_groups < 1:
        raise ValueError("schedule requires sources, a positive step and group count")
    if overfit_group is not None and overfit_group not in groups:
        raise ValueError("overfit source must belong to training sources")
    generator = torch.Generator().manual_seed(_seed(seed, "sources", step))
    indices = torch.randint(len(groups), (global_groups,), generator=generator).tolist()
    return tuple((overfit_group or groups[index], _seed(seed, "objective", step, position))
                 for position, index in enumerate(indices))


def rank_schedule(schedule: Sequence, *, rank: int, world_size: int) -> tuple:
    if len(schedule) % world_size or not 0 <= rank < world_size:
        raise ValueError("global source groups must divide evenly across ranks")
    size = len(schedule) // world_size
    return tuple(schedule[rank * size:(rank + 1) * size])


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def _rng_state(context: DistributedContext) -> dict:
    import numpy as np
    numpy = np.random.get_state()
    return {"torch": torch.random.get_rng_state(), "python": random.getstate(),
            "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
            "device": torch.cuda.get_rng_state(context.device).cpu()
                      if context.device.type == "cuda" else None}


def _restore_rng(state: Mapping, context: DistributedContext) -> None:
    import numpy as np
    torch.random.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype="uint32"), *numpy[2:]))
    if context.device.type == "cuda":
        if state["device"] is None:
            raise ValueError("checkpoint lacks GPU RNG state")
        torch.cuda.set_rng_state(state["device"].cpu(), context.device)


def _broadcast(value, context: DistributedContext):
    values = [value]
    if context.distributed:
        dist.broadcast_object_list(values, src=0)
    return values[0]


def rank_zero_call(context: DistributedContext, operation) -> Any:
    """Run a side effect once and broadcast failure before any rank continues."""
    response = {"error": None, "result": None}
    if context.rank == 0:
        try:
            response["result"] = operation()
        except Exception as exc:
            response["error"] = f"rank-zero operation failed: {type(exc).__name__}: {exc}"
    response = _broadcast(response, context)
    if response["error"]:
        raise RuntimeError(response["error"])
    return response["result"]


def _log(logger, values, step, context):
    rank_zero_call(context, lambda: logger.log(values, step=step))


def require_finite_loss(loss: torch.Tensor, context: DistributedContext) -> None:
    """Fail every rank before backward if any rank reports a nonfinite loss."""
    finite = torch.isfinite(loss.detach()).all().to(dtype=torch.int32, device=context.device)
    if context.distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError("nonfinite hybrid training loss on at least one rank")


def save_checkpoint(path: str | Path, module: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, metadata: Mapping[str, Any],
                    context: DistributedContext) -> None:
    """Collect rank RNGs, atomically replace one snapshot, and report errors to all."""
    rng = _rng_state(context)
    rngs = [None] * context.world_size if context.rank == 0 else None
    if context.distributed:
        dist.gather_object(rng, rngs, dst=0)
    else:
        rngs = [rng]
    error = None
    if context.rank == 0:
        path = Path(path)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError("checkpoint snapshots are immutable")
            payload = {"schema_version": 1, "metadata": dict(metadata),
                       "module": _cpu(module.state_dict()),
                       "optimizer": _cpu(optimizer.state_dict()), "rank_rng": rngs}
            with temporary.open("xb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception as exc:
            error = f"checkpoint write failed: {type(exc).__name__}: {exc}"
    error = _broadcast(error, context)
    if error:
        raise RuntimeError(error)


def load_checkpoint(path: str | Path, module: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, context: DistributedContext,
                    *, expected: Mapping[str, Any]) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if set(payload) != {"schema_version", "metadata", "module", "optimizer", "rank_rng"} or payload["schema_version"] != 1:
        raise ValueError("checkpoint schema mismatch")
    metadata = payload["metadata"]
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch")
    if metadata.get("next_step") != metadata.get("step", -1) + 1:
        raise ValueError("checkpoint next_step is not the next optimizer boundary")
    if len(payload["rank_rng"]) != context.world_size:
        raise ValueError("checkpoint rank RNG count differs from world size")
    module.load_state_dict(payload["module"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng(payload["rank_rng"][context.rank], context)
    return metadata


def _write_json(path: Path, values: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def initialize_module(cfg: HybridPilotConfig, device: torch.device, *, spatial_bf16: bool = False) -> HybridPilotModule:
    from .objectives import HybridPilotModule
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        module = HybridPilotModule(cfg)
    module.to(device=device, dtype=torch.float32)
    if spatial_bf16:
        if device.type != "cuda":
            raise ValueError("spatial BF16 requires a GPU")
        module.model.observer.spatial_autocast = True
    return module


def _reduce_metrics(values: Mapping[str, torch.Tensor], context: DistributedContext) -> dict[str, float]:
    names = sorted(values)
    tensors = torch.stack([values[name].detach().double().reshape(()) for name in names])
    if context.distributed:
        dist.all_reduce(tensors)
        tensors /= context.world_size
    if not bool(torch.isfinite(tensors).all()):
        raise FloatingPointError("nonfinite objective metric")
    return dict(zip(names, tensors.cpu().tolist()))


def _elapsed_max(seconds: float, context: DistributedContext) -> float:
    value = torch.tensor(seconds, device=context.device, dtype=torch.float64)
    if context.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def _memory_metrics(context: DistributedContext) -> dict[str, float]:
    if context.device.type != "cuda":
        return {}
    values = torch.tensor([torch.cuda.memory_allocated(context.device),
                           torch.cuda.memory_reserved(context.device),
                           torch.cuda.max_memory_allocated(context.device)],
                          device=context.device, dtype=torch.float64) / 2**30
    maximum = values.clone()
    if context.distributed:
        dist.all_reduce(values)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    names = ("allocated_gib", "reserved_gib", "peak_allocated_gib")
    return {**{f"system/{name}_sum": val for name, val in zip(names, values.tolist())},
            **{f"system/{name}_max_rank": val for name, val in zip(names, maximum.tolist())}}


def _metadata(lock, phase, step, cfg, context, logger):
    return {**lock, "phase": phase, "step": step, "next_step": step + 1,
            "config": cfg.to_dict(),
            "source_cursor": step * cfg.global_source_groups,
            "world_size": context.world_size, "wandb_run_id": logger.run_id}


@torch.no_grad()
def _diagnostics(module, cfg, dataset, phase, context):
    groups = (dataset.overfit_group,) if phase == "tiny" else tuple(getattr(dataset, "dev_groups", ()))[:context.world_size]
    if not groups:
        return {}
    # Every rank evaluates the same fixed tiny source. For development, choose
    # one distinct source per rank when enough are available; otherwise all
    # ranks use the same first source so averaging cannot overweight a source.
    selected = groups[context.rank] if len(groups) == context.world_size else groups[0]
    was_training = module.training
    module.eval()
    try:
        batch = dataset.load_group(selected, device=context.device)
        loss, metrics = module([batch], [_seed(cfg.seed, "diagnostic", selected)], phase=phase)
        return _reduce_metrics({"loss": loss.detach(), **metrics}, context)
    finally:
        module.train(was_training)


def train_phase(cfg: HybridPilotConfig, dataset: HybridPilotDataset, phase: str, output: Path, context: DistributedContext,
                deadlines: DeadlineController, logger: RunLogger, lock: Mapping, *,
                microbatch: int, spatial_bf16: bool = False, resume: str | Path | None = None,
                max_steps: int | None = None) -> tuple[HybridPilotModule, dict[str, Any]]:
    """Train one phase; a main phase always initializes from the canonical seed."""
    if phase not in ("tiny", "main"):
        raise ValueError("phase must be tiny or main")
    local_groups = cfg.global_source_groups // context.world_size
    if cfg.global_source_groups % context.world_size or microbatch not in cfg.microbatch_candidates or local_groups % microbatch:
        raise ValueError("microbatch must exactly divide the fixed local source-group budget")
    accumulation = local_groups // microbatch
    steps = cfg.overfit_steps if phase == "tiny" else cfg.train_steps
    step_limit = min(steps, max_steps) if max_steps is not None else steps
    phase_dir = output / phase
    def prepare_directory():
        phase_dir.mkdir(parents=True, exist_ok=resume is not None)
        if resume is not None:
            resume_step = torch.load(resume, map_location="cpu", weights_only=True)["metadata"]["step"]
            for existing in phase_dir.glob("step-*.pt"):
                match = re.match(r"step-(\d+)", existing.name)
                if match and int(match.group(1)) > resume_step:
                    raise ValueError("resume would rewind an output directory with later checkpoints")
    rank_zero_call(context, prepare_directory)
    module = initialize_module(cfg, context.device, spatial_bf16=spatial_bf16)
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg.learning_rate,
                                 betas=cfg.betas, weight_decay=cfg.weight_decay)
    random.seed(_seed(cfg.seed, "python", context.rank))
    import numpy as np
    np.random.seed(_seed(cfg.seed, "numpy", context.rank) % 2**32)
    torch.manual_seed(_seed(cfg.seed, "torch", context.rank))
    expected = {**lock, "phase": phase, "world_size": context.world_size, "config": cfg.to_dict()}
    step = 0
    if resume is not None:
        restored = load_checkpoint(resume, module, optimizer, context, expected=expected)
        step = restored["step"]
        if restored["source_cursor"] != step * cfg.global_source_groups or not 0 <= step <= steps:
            raise ValueError("checkpoint source cursor or optimizer step is invalid")
    wrapped = DistributedDataParallel(module, device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        find_unused_parameters=True, broadcast_buffers=False) if context.distributed else module
    module.train()
    if resume is None:
        save_checkpoint(phase_dir / "initial.pt", module, optimizer,
                        _metadata(lock, phase, 0, cfg, context, logger), context)
    initial_diagnostics = {}
    if phase == "tiny" and resume is None and not deadlines.should_stop(context):
        initial_diagnostics = _diagnostics(module, cfg, dataset, phase, context)
        rank_zero_call(context, lambda: _write_json(phase_dir / "diagnostics-initial.json", initial_diagnostics))
        _log(logger, {f"tiny_initial/{key}": value for key, value in initial_diagnostics.items()}, 0, context)
    elif phase == "tiny" and (phase_dir / "diagnostics-initial.json").exists():
        initial_diagnostics = json.loads((phase_dir / "diagnostics-initial.json").read_text())
    last_checkpoint = Path(resume) if resume else phase_dir / "initial.pt"
    checkpoint_time = time.monotonic()
    last_metrics = {}
    phase_offset = 0 if phase == "tiny" else cfg.overfit_steps
    stopped = False
    while step < step_limit:
        if deadlines.should_stop(context):
            stopped = True
            break
        next_step = step + 1
        schedule = scheduled_step(dataset.train_groups, seed=cfg.seed, step=next_step,
            global_groups=cfg.global_source_groups,
            overfit_group=dataset.overfit_group if phase == "tiny" else None)
        local = rank_schedule(schedule, rank=context.rank, world_size=context.world_size)
        optimizer.zero_grad(set_to_none=True)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
            torch.cuda.synchronize(context.device)
        started = time.monotonic()
        accumulated = {}
        for index in range(accumulation):
            selected = local[index * microbatch:(index + 1) * microbatch]
            batches = [dataset.load_group(group, device=context.device) for group, _ in selected]
            sync = wrapped.no_sync() if context.distributed and index + 1 < accumulation else nullcontext()
            with sync:
                loss, components = wrapped(batches, [seed for _, seed in selected], phase=phase)
                require_finite_loss(loss, context)
                (loss / accumulation).backward()
            for name, value in {"loss": loss.detach(), **components}.items():
                accumulated[name] = accumulated.get(name, torch.zeros_like(value)) + value.detach() / accumulation
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        step = next_step
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        elapsed = _elapsed_max(time.monotonic() - started, context)
        if step == 1 or step % cfg.log_every == 0 or step == step_limit:
            last_metrics = _reduce_metrics({**accumulated, "gradient_norm": norm}, context)
            metrics = {f"{phase}/{name}": value for name, value in last_metrics.items()}
            metrics.update({"progress/phase": phase, "progress/optimizer_step": step,
                "progress/phase_fraction": step / steps, "performance/step_seconds": elapsed,
                "performance/clips_per_second": cfg.global_source_groups * cfg.batch_size / elapsed,
                "performance/frames_per_second": cfg.global_source_groups * cfg.batch_size * cfg.T / elapsed,
                "performance/microbatch_groups_per_rank": microbatch,
                "performance/gradient_accumulation": accumulation, **_memory_metrics(context)})
            def log_training():
                from .telemetry import collect_gpu_metrics
                metrics.update(collect_gpu_metrics() if context.device.type == "cuda" else {})
                logger.log(metrics, step=phase_offset + step)
                print(json.dumps({"phase": phase, "step": step, "loss": last_metrics["loss"],
                                  "step_seconds": elapsed}, sort_keys=True), flush=True)
            rank_zero_call(context, log_training)
        # Rank zero decides elapsed-time saves so divergent host scheduling can
        # never make only some ranks enter RNG-gather/checkpoint collectives.
        timed_save = context.rank == 0 and time.monotonic() - checkpoint_time >= 600
        timed_save = _broadcast(timed_save, context)
        if step % cfg.checkpoint_every == 0 or step == steps or timed_save:
            last_checkpoint = phase_dir / f"step-{step:06d}.pt"
            metadata = _metadata(lock, phase, step, cfg, context, logger)
            save_checkpoint(last_checkpoint, module, optimizer, metadata, context)
            rank_zero_call(context, lambda: logger.checkpoint(last_checkpoint, metadata=metadata))
            checkpoint_time = time.monotonic()
            if phase == "main" and not deadlines.should_stop(context, optimization=False):
                development = _diagnostics(module, cfg, dataset, phase, context)
                _log(logger, {f"dev/{key}": value for key, value in development.items()}, phase_offset + step, context)
    stopped = stopped or step < steps
    if stopped:
        last_checkpoint = phase_dir / f"step-{step:06d}-stopped.pt"
        collision = _broadcast(last_checkpoint.exists() if context.rank == 0 else None, context)
        if collision:
            suffix = _broadcast(uuid.uuid4().hex[:8] if context.rank == 0 else None, context)
            last_checkpoint = phase_dir / f"step-{step:06d}-stopped-{suffix}.pt"
        metadata = _metadata(lock, phase, step, cfg, context, logger)
        save_checkpoint(last_checkpoint, module, optimizer, metadata, context)
        rank_zero_call(context, lambda: logger.checkpoint(last_checkpoint, metadata=metadata))
    result = {"phase": phase, "status": "stopped" if stopped else "complete", "step": step,
              "target_steps": steps, "checkpoint": str(last_checkpoint.relative_to(output))
                  if last_checkpoint.is_relative_to(output) else last_checkpoint.name,
              "last_metrics": last_metrics, "tiny_source_repeated": phase == "tiny",
              "verification_step_cap": max_steps,
              "main_starts_from_fresh_canonical_initialization": phase == "main" and resume is None}
    if phase == "tiny" and not stopped and not deadlines.should_stop(context, optimization=False):
        final_diagnostics = _diagnostics(module, cfg, dataset, phase, context)
        result.update(initial_diagnostics=initial_diagnostics, final_diagnostics=final_diagnostics)
        if "reconstruction_l1" in initial_diagnostics and "temporal_mse" in final_diagnostics:
            from mavt4d.pilot.runner import overfit_gate
            result["overfit_gate"] = overfit_gate(initial_diagnostics, final_diagnostics, cfg)
        _log(logger, {f"tiny_final/{key}": value for key, value in final_diagnostics.items()}, step, context)
    def save_result():
        _write_json(phase_dir / f"result-{uuid.uuid4().hex[:8]}.json", result)
        _write_json(phase_dir / "result.json", result)
    rank_zero_call(context, save_result)
    del wrapped, optimizer
    return module, result


def run_pilot(cfg: HybridPilotConfig, dataset_root: str | Path, output: str | Path, *, phase: str,
              microbatch: int, optimization_at: datetime, work_at: datetime,
              git_revision: str, resume: str | Path | None = None,
              spatial_bf16: bool = False, allow_cpu: bool = False, wandb_sdk=None,
              max_steps: int | None = None) -> dict:
    from .data import HybridPilotDataset
    if phase not in ("tiny", "main", "all"):
        raise ValueError("phase must be tiny, main, or all")
    if not re.fullmatch(r"[0-9a-f]{40}", git_revision):
        raise ValueError("git_revision must be a full commit SHA")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive explicit verification cap")
    deadlines = DeadlineController(optimization_at, work_at)
    if datetime.now(timezone.utc) >= optimization_at:
        raise ValueError("optimization deadline has already passed; refusing to launch")
    context = initialize_distributed(allow_cpu=allow_cpu)
    if context.world_size != cfg.world_size and not allow_cpu:
        raise ValueError("execution world size differs from locked configuration")
    output = Path(output).resolve()
    dataset = HybridPilotDataset(dataset_root, cfg)
    dataset_hash = hashlib.sha256((dataset.root / "summary.json").read_bytes()).hexdigest()
    lock = {"config_sha256": cfg.config_sha256, "dataset_sha256": dataset_hash,
            "git_revision": git_revision, "spatial_bf16": spatial_bf16, "microbatch": microbatch}
    resume_metadata = None
    if resume is not None:
        resume_metadata = torch.load(resume, map_location="cpu", weights_only=True)["metadata"]
        for name, value in lock.items():
            if resume_metadata.get(name) != value:
                raise ValueError(f"checkpoint {name} mismatch")
        if phase != "all" and phase != resume_metadata["phase"]:
            raise ValueError("requested phase differs from resume checkpoint")
    def prepare_output():
        output.mkdir(parents=True, exist_ok=resume is not None)
        run_lock = {**lock, "config": cfg.to_dict(),
            "world_size": context.world_size, "microbatch": microbatch,
            "verification_step_cap": max_steps,
            "stop_optimization_at": optimization_at.isoformat(), "stop_work_at": work_at.isoformat()}
        if resume is None:
            _write_json(output / "run_lock.json", run_lock)
        _write_json(output / f"attempt-{uuid.uuid4().hex[:8]}.json", run_lock)
    rank_zero_call(context, prepare_output)
    logger = None
    completed = False
    try:
        startup = None
        try:
            logger = RunLogger(output, rank=context.rank,
                config={**lock, **cfg.to_dict()}, sdk=wandb_sdk,
                run_id=resume_metadata.get("wandb_run_id") if resume_metadata else None,
                enabled=not allow_cpu or wandb_sdk is not None)
        except Exception as exc:
            startup = f"W&B startup failed: {type(exc).__name__}: {exc}"
        startup = _broadcast(startup, context)
        if startup:
            raise RuntimeError(startup)
        logger.run_id = _broadcast(logger.run_id, context)
        deadlines.install()
        selected = ["tiny", "main"] if phase == "all" else [phase]
        if resume_metadata and resume_metadata["phase"] == "main":
            selected = ["main"]
        results = []
        module = None
        for selected_phase in selected:
            if module is not None:
                del module
                if context.device.type == "cuda":
                    torch.cuda.empty_cache()
            module, result = train_phase(cfg, dataset, selected_phase, output, context,
                deadlines, logger, lock, microbatch=microbatch, spatial_bf16=spatial_bf16,
                resume=resume if resume_metadata and selected_phase == resume_metadata["phase"] else None,
                max_steps=max_steps)
            results.append(result)
            if result["status"] != "complete":
                break
        evaluation = {"status": "not_run", "reason": "full main budget is required"}
        if results[-1]["phase"] == "main" and results[-1]["status"] == "complete":
            from .evaluation import evaluate_distributed
            evaluation = evaluate_distributed(module.model, cfg, dataset, context, deadlines, logger,
                                               global_step=cfg.overfit_steps + cfg.train_steps + 1)
            if evaluation["status"] == "complete":
                untrained = initialize_module(cfg, context.device, spatial_bf16=spatial_bf16)
                baseline = evaluate_distributed(untrained.model, cfg, dataset, context, deadlines, logger,
                    global_step=cfg.overfit_steps + cfg.train_steps + 2, label="untrained")
                evaluation["matched_untrained"] = baseline
                del untrained
        summary = {"status": "complete" if all(item["status"] == "complete" for item in results) else "stopped",
                   "training": results, "evaluation": evaluation, **lock}
        if results[-1]["phase"] == "main" and results[-1]["status"] == "complete" and (
                evaluation["status"] != "complete" or evaluation.get("matched_untrained", {}).get("status") != "complete"):
            summary["status"] = "training_complete_evaluation_incomplete"
        def save_summary():
            _write_json(output / f"result-{uuid.uuid4().hex[:8]}.json", summary)
            _write_json(output / "result.json", summary)
        rank_zero_call(context, save_summary)
        _log(logger, {"progress/status": summary["status"], "progress/evaluation_status": evaluation["status"]},
             cfg.overfit_steps + cfg.train_steps + 3, context)
        completed = True
        return summary
    finally:
        deadlines.restore()
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        if dist.is_initialized():
            dist.destroy_process_group()
        if logger is not None:
            logger.finish(exit_code=0 if completed else 1)
