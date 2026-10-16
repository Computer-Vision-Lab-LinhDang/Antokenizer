"""Fixed-budget synthetic pilot orchestration, with an explicit validation barrier.

The production entry point requires CUDA. CPU execution is an explicit test hook,
not an alternative training backend. This module never submits jobs or downloads
models, and it never changes the protocol after observing an outcome.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Mapping, Sequence
import uuid

import torch
from torch import nn

from mavt4d.reference import ReferenceCDModel

from .config import PilotConfig
from .data import PilotDataset
from .evaluation import (
    evaluate_probe_suite, evaluate_reconstruction, extract_features, summarize_runs,
)
from .objectives import PilotHeads, training_loss


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _event(path: Path, **values: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical({"time": _utc_now(), **values}) + "\n")
        stream.flush()


def _module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(_canonical([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def initialize_run(
    cfg: PilotConfig, cell: str, seed: int, device: str | torch.device,
) -> tuple[ReferenceCDModel, PilotHeads, dict[str, str]]:
    """Copy canonical shared modules; model construction order cannot skew them.

    Construction happens on CPU with a temporary CPU RNG state. Reinitializing a
    cell cannot perturb the independent source/mask/query random streams.
    """
    def model(kind: str) -> ReferenceCDModel:
        return ReferenceCDModel(cfg.state_spec, kind, embed_dim=cfg.embed_dim,
                                patch_size=cfg.patch_size, modulation_dim=cfg.modulation_dim)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        canonical = model("coupled")
        canonical_heads = PilotHeads(cfg.state_spec, cfg)
        torch.random.default_generator.manual_seed(seed)
        current = model(cell)
        heads = PilotHeads(cfg.state_spec, cfg)
        current.observer.load_state_dict(canonical.observer.state_dict())
        current.decoder.load_state_dict(canonical.decoder.state_dict())
        heads.load_state_dict(canonical_heads.state_dict())
    hashes = {"observer": _module_hash(current.observer),
              "decoder": _module_hash(current.decoder), "heads": _module_hash(heads)}
    return current.to(device=device, dtype=torch.float32), heads.to(device=device, dtype=torch.float32), hashes


def _stream_seed(seed: int, stream: str) -> int:
    return int.from_bytes(hashlib.sha256(f"pilot-v1:{stream}:{seed}".encode()).digest()[:8], "big") % (2**63)


def objective_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(_stream_seed(seed, "objective"))


def group_schedule(
    groups: Sequence[str], seed: int, steps: int, overfit_group: str | None = None,
) -> tuple[str, ...]:
    if not groups or type(steps) is not int or steps <= 0:
        raise ValueError("groups must be nonempty and steps positive")
    if overfit_group is not None:
        if overfit_group not in groups:
            raise ValueError("overfit group must belong to training groups")
        return (overfit_group,) * steps
    generator = torch.Generator(device="cpu").manual_seed(_stream_seed(seed, "source"))
    indices = torch.randint(len(groups), (steps,), generator=generator).tolist()
    return tuple(groups[index] for index in indices)


def save_checkpoint(
    path: str | Path, model: ReferenceCDModel, heads: PilotHeads,
    optimizer: torch.optim.Optimizer, generator: torch.Generator,
    metadata: Mapping[str, Any],
) -> None:
    """Write one immutable checkpoint, including optimizer and random streams."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {"schema_version": 1, "metadata": dict(metadata),
               "model": model.state_dict(), "heads": heads.state_dict(),
               "optimizer": optimizer.state_dict(), "objective_rng": generator.get_state(),
               "torch_rng": torch.random.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path, model: ReferenceCDModel, heads: PilotHeads,
    optimizer: torch.optim.Optimizer, generator: torch.Generator, *,
    config_sha256: str, dataset_sha256: str,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore a run only after validating its frozen configuration and dataset.

    Callers continuing optimization must also pass phase/cell/seed/schedule in
    ``expected_metadata``. The CLI intentionally does not resume partial suites.
    """
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    expected_keys = {"schema_version", "metadata", "model", "heads", "optimizer",
                     "objective_rng", "torch_rng", "cuda_rng"}
    if set(payload) != expected_keys or payload["schema_version"] != 1:
        raise ValueError("checkpoint schema differs from the locked contract")
    metadata = payload["metadata"]
    for name, expected in {"config_sha256": config_sha256, "dataset_sha256": dataset_sha256,
                           **dict(expected_metadata or {})}.items():
        if metadata.get(name) != expected:
            raise ValueError(f"checkpoint {name} mismatch")
    model.load_state_dict(payload["model"], strict=True)
    heads.load_state_dict(payload["heads"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["objective_rng"].cpu())
    torch.random.set_rng_state(payload["torch_rng"].cpu())
    if payload["cuda_rng"] and torch.cuda.is_available():
        if len(payload["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA device count differs")
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    return metadata


def overfit_gate(initial: Mapping[str, float], final: Mapping[str, float], cfg: PilotConfig) -> dict[str, Any]:
    reconstruction_ratio = final["reconstruction_l1"] / max(initial["reconstruction_l1"], 1e-12)
    temporal_ratio = final["temporal_mse"] / max(final["temporal_zero_mse"], 1e-12)
    return {"reconstruction_ratio": reconstruction_ratio, "temporal_ratio": temporal_ratio,
            "reconstruction_limit": cfg.overfit_recon_ratio,
            "temporal_limit": cfg.overfit_temporal_ratio,
            "passed": reconstruction_ratio <= cfg.overfit_recon_ratio and temporal_ratio <= cfg.overfit_temporal_ratio,
            "policy": "finite threshold failure does not change the main training budget"}


def _metrics(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    result = {name: float(value.detach().cpu()) for name, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite objective component")
    return result


def _optimizer(model: ReferenceCDModel, heads: PilotHeads, cfg: PilotConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()),
                             lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay)


def _train_run(
    cfg: PilotConfig, dataset: PilotDataset, phase: str, cell: str, seed: int,
    run_dir: Path, device: str | torch.device, lock: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    model, heads, hashes = initialize_run(cfg, cell, seed, device)
    optimizer = _optimizer(model, heads, cfg)
    generator = objective_generator(seed)
    steps = cfg.overfit_steps if phase == "overfit" else cfg.train_steps
    schedule = group_schedule(dataset.train_groups, seed, steps,
                              dataset.overfit_group if phase == "overfit" else None)
    schedule_hash = hashlib.sha256(_canonical(schedule).encode()).hexdigest()
    _write_json(run_dir / "schedule.json", {"groups": schedule, "sha256": schedule_hash})
    state = model.initial_state(1)
    budget = {"state_values": state.num_state_values, "state_bytes": state.state_nbytes,
              "metadata_bytes": state.metadata_nbytes, "total_state_bytes": state.total_nbytes,
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "observer_parameters": sum(p.numel() for p in model.observer.parameters()),
              "cell_parameters": sum(p.numel() for p in model.cell.parameters()),
              "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
              "head_parameters": sum(p.numel() for p in heads.parameters()),
              "optimizer_steps": steps, "clips_per_step": cfg.batch_size,
              "frames_per_clip": cfg.T, "query_count": cfg.query_count,
              "execution": "float32 sequential reference; parameters and FLOPs are not matched"}
    metadata = {"config_sha256": cfg.config_sha256, "dataset_sha256": lock["dataset_sha256"],
                "git_revision": lock["git_revision"], "phase": phase, "cell": cell,
                "seed": seed, "schedule_sha256": schedule_hash, "shared_init_hashes": hashes}
    initial_checkpoint = run_dir / "initial.pt"
    save_checkpoint(initial_checkpoint, model, heads, optimizer, generator, {**metadata, "step": 0})
    _write_json(run_dir / "budget.json", budget)
    diagnostics_initial = None
    if phase == "overfit":
        model.eval()
        heads.eval()
        with torch.no_grad():
            _, diagnostic = training_loss(model, heads, dataset.load_group(dataset.overfit_group, device),
                                           cfg, objective_generator(seed), phase="overfit")
        diagnostics_initial = _metrics(diagnostic)
    model.train()
    heads.train()
    started = time.perf_counter()
    last_metrics: dict[str, float] = {}
    final_checkpoint = None
    for step, group in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss, components = training_loss(model, heads, dataset.load_group(group, device),
                                          cfg, generator, phase=phase)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss in {phase}/{cell}/seed{seed}/step{step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()),
                                             cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % cfg.log_every == 0 or step == steps:
            last_metrics = _metrics({"loss": loss, "gradient_norm": norm, **components})
            _event(run_dir / "training.jsonl", step=step, metrics=last_metrics)
            print(_canonical({"phase": phase, "cell": cell, "seed": seed,
                              "step": step, "loss": last_metrics["loss"]}), flush=True)
        if step % cfg.checkpoint_every == 0 or step == steps:
            final_checkpoint = run_dir / f"step-{step:06d}.pt"
            save_checkpoint(final_checkpoint, model, heads, optimizer, generator, {**metadata, "step": step})
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {"phase": phase, "cell": cell, "seed": seed, "status": "complete", "step": steps,
              "checkpoint": str(final_checkpoint), "initial_checkpoint": str(initial_checkpoint),
              "config_sha256": cfg.config_sha256, "dataset_sha256": lock["dataset_sha256"],
              "shared_init_hashes": hashes, "schedule_sha256": schedule_hash,
              "budget": budget, "seconds": elapsed, "last_training_metrics": last_metrics,
              "overfit_gate": None}
    if phase == "overfit":
        model.eval()
        heads.eval()
        with torch.no_grad():
            _, diagnostic = training_loss(model, heads, dataset.load_group(dataset.overfit_group, device),
                                           cfg, objective_generator(seed), phase="overfit")
        diagnostics_final = _metrics(diagnostic)
        result.update({"initial_diagnostics": diagnostics_initial, "final_diagnostics": diagnostics_final,
                       "overfit_gate": overfit_gate(diagnostics_initial, diagnostics_final, cfg)})
    _write_json(run_dir / "result.json", result)
    return result


def _evaluate_run(
    cfg: PilotConfig, dataset: PilotDataset, run: Mapping[str, Any], device: str | torch.device,
    *, untrained: bool = False,
) -> dict[str, Any]:
    model, heads, _ = initialize_run(cfg, run["cell"], run["seed"], device)
    optimizer = _optimizer(model, heads, cfg)
    checkpoint = run["initial_checkpoint"] if untrained else run["checkpoint"]
    load_checkpoint(checkpoint, model, heads, optimizer, objective_generator(run["seed"]),
                    config_sha256=cfg.config_sha256, dataset_sha256=run["dataset_sha256"],
                    expected_metadata={"phase": "main", "cell": run["cell"], "seed": run["seed"],
                                       "schedule_sha256": run["schedule_sha256"],
                                       "step": 0 if untrained else cfg.train_steps})
    model.eval()
    with torch.no_grad():
        train = extract_features(model, dataset, dataset.train_groups, cfg, device)
        val = extract_features(model, dataset, dataset.val_groups, cfg, device)
        result = evaluate_probe_suite(train, val, cfg)
        result["reconstruction"] = evaluate_reconstruction(model, dataset, dataset.val_groups, cfg, device)
    return result


def _report(result: Mapping[str, Any]) -> str:
    lines = ["# Synthetic C/D pilot", "", f"Status: **{result['status']}**.", "",
             f"Config SHA256: `{result['config_sha256']}`.",
             f"Dataset SHA256: `{result['dataset_sha256']}`.",
             f"Code revision: `{result['git_revision']}`.", "",
             "All main training runs finish or fail before held-out extraction begins. "
             "Only final checkpoints are evaluated. Finite overfit threshold failures do not extend training.",
             "", "## Overfit diagnostics", "", "| Cell | Status | Reconstruction ratio | Temporal / zero MSE | Pass |",
             "|---|---|---:|---:|---|"]
    for run in result["training"]:
        if run["phase"] != "overfit":
            continue
        gate = run.get("overfit_gate") or {}
        lines.append(f"| {run['cell']} | {run['status']} | {gate.get('reconstruction_ratio', 'unmeasured')} | "
                     f"{gate.get('temporal_ratio', 'unmeasured')} | {gate.get('passed', 'unmeasured')} |")
    lines.extend(["", "## Fixed-budget training", "", "| Cell | Seed | Status | Steps | Cell parameters | State bytes |",
                  "|---|---:|---|---:|---:|---:|"])
    for run in result["training"]:
        if run["phase"] == "main":
            budget = run.get("budget", {})
            lines.append(f"| {run['cell']} | {run['seed']} | {run['status']} | {run.get('step', 'unknown')} | "
                         f"{budget.get('cell_parameters', 'unknown')} | {budget.get('total_state_bytes', 'unknown')} |")
    for label in ("trained", "untrained"):
        lines.extend(["", f"## {label.capitalize()} probes", "", "```json",
                      json.dumps(result["summaries"][label], indent=2, sort_keys=True, allow_nan=False), "```"])
    lines.extend(["", "Source groups are held out; all core × configuration combinations remain represented. "
                  "Direction within reversal pairs is the primary C history metric; four-way core is secondary. "
                  "D configuration can be visible in the final image, so this measurement alone does not establish "
                  "phase, velocity or disentanglement. Compare trained states with untrained and order-blind controls.",
                  "", "State capacity, samples, update counts and shared initialization match across cells; "
                  "parameter counts and FLOPs do not. These unoptimized reference runs cannot establish a speed advantage.", ""])
    return "\n".join(lines)


def _summarize_available(entries: list[dict[str, Any]], cfg: PilotConfig) -> dict[str, Any]:
    """Retain partial evidence without inventing a paired three-cell comparison."""
    if not entries:
        return {"status": "unmeasured"}
    by_cell = {cell: [entry for entry in entries if entry["cell"] == cell] for cell in cfg.cells}
    seed_sets = {cell: {entry["seed"] for entry in runs} for cell, runs in by_cell.items()}
    expected = set(cfg.seeds)
    if all(seeds == expected for seeds in seed_sets.values()):
        return summarize_runs(entries, cfg)
    complete_seeds = set.intersection(*(seeds for seeds in seed_sets.values()))
    paired = [entry for entry in entries if entry["seed"] in complete_seeds]
    return {"status": "incomplete_comparison",
            "missing_cell_seed_pairs": [{"cell": cell, "seed": seed} for cell in cfg.cells
                                        for seed in cfg.seeds if seed not in seed_sets[cell]],
            "per_cell": {cell: summarize_runs(runs, cfg) for cell, runs in by_cell.items() if runs},
            "paired_complete_seeds": sorted(complete_seeds),
            "paired": summarize_runs(paired, cfg) if paired else None}


def run_suite(
    cfg: PilotConfig, dataset_root: str | Path, output: str | Path, *,
    device: str | torch.device = "cuda", git_revision: str, allow_cpu: bool = False,
) -> dict[str, Any]:
    """Run all overfit/main jobs sequentially, then cross the validation barrier."""
    device = torch.device(device)
    if device.type != "cuda" and not allow_cpu:
        raise ValueError("production pilot requires CUDA in the authorized compute job")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("only CUDA production or explicit CPU tests are supported")
    if not re.fullmatch(r"[0-9a-f]{40}", git_revision):
        raise ValueError("git_revision must be the full 40-character commit SHA")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no fallback training is permitted")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    dataset = PilotDataset(dataset_root, cfg)
    if not dataset.train_groups or not dataset.val_groups:
        raise ValueError("pilot needs nonempty train and held-out source groups")
    summary_bytes = (Path(dataset.root) / "summary.json").read_bytes()
    dataset_summary = json.loads(summary_bytes)
    lock = {"schema_version": 1, "config_sha256": cfg.config_sha256,
            "dataset_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "dataset_checksums": dataset_summary.get("sha256", {}),
            "render_config_sha256": dataset_summary.get("render_config_sha256"),
            "git_revision": git_revision, "created_at": _utc_now(),
            "groups": {"train": list(dataset.train_groups), "dev": list(dataset.dev_groups),
                       "val": list(dataset.val_groups)},
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "cuda": torch.version.cuda, "device": str(device),
                            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "test_only_cpu": device.type == "cpu"},
            "validation_policy": "all main training attempts precede any held-out evaluation"}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", cfg.to_dict())
    _write_json(output / "run_lock.json", lock)
    events = output / "events.jsonl"
    planned = [("overfit", cell, cfg.overfit_seed) for cell in cfg.cells]
    planned.extend(("main", cell, seed) for seed in cfg.seeds for cell in cfg.cells)
    training = []
    for phase, cell, seed in planned:
        run_dir = output / phase / f"{cell}-seed{seed}"
        _event(events, action="training_started", phase=phase, cell=cell, seed=seed)
        try:
            run = _train_run(cfg, dataset, phase, cell, seed, run_dir, device, lock)
        except Exception as exc:
            run = {"phase": phase, "cell": cell, "seed": seed, "status": "failed",
                   "error_type": type(exc).__name__, "error": str(exc)}
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(run_dir / "failure.json", run)
        training.append(run)
        _event(events, action="training_finished", phase=phase, cell=cell, seed=seed, status=run["status"])
        _write_json(output / "training_progress.json", training)
    _event(events, action="all_training_finished", attempted=len(training), expected=len(planned))
    evaluations: dict[str, list[dict[str, Any]]] = {"trained": [], "untrained": []}
    failures = []
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir()
    for run in training:
        if run["phase"] != "main" or run["status"] != "complete":
            continue
        for label in ("trained", "untrained"):
            _event(events, action="evaluation_started", cell=run["cell"], seed=run["seed"], kind=label)
            try:
                evaluated = _evaluate_run(cfg, dataset, run, device, untrained=label == "untrained")
                entry = {"cell": run["cell"], "seed": run["seed"], "result": evaluated}
                evaluations[label].append(entry)
                _write_json(evaluation_dir / f"{run['cell']}-seed{run['seed']}-{label}.json", entry)
            except Exception as exc:
                failure = {"cell": run["cell"], "seed": run["seed"], "kind": label,
                           "error_type": type(exc).__name__, "error": str(exc)}
                failures.append(failure)
                _write_json(evaluation_dir / f"{run['cell']}-seed{run['seed']}-{label}-failure.json", failure)
    result = {"schema_version": 1, "status": "complete" if not failures and all(r["status"] == "complete" for r in training) else "incomplete",
              "config_sha256": cfg.config_sha256, "dataset_sha256": lock["dataset_sha256"],
              "git_revision": git_revision, "finished_at": _utc_now(), "training": training,
              "evaluation_failures": failures,
              "summaries": {label: _summarize_available(entries, cfg)
                            for label, entries in evaluations.items()}}
    _write_json(output / "summary.json", result)
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    _event(events, action="suite_finished", status=result["status"])
    return result
