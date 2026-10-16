"""Frozen hybrid C/D probes with source-disjoint fitting and explicit controls.

The final suffix control resets the *complete* C/D+KV state. The orderless
control encodes individual frames and sorts features before reduction so frame
reversal cannot leak through a floating-point summation order.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Sequence, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mavt4d.hybrid.model import HybridCDModel

from mavt4d.data.event_synthetic import EVENT_CONFIGURATIONS, EVENT_CORES
from mavt4d.pilot.evaluation import (
    FEATURE_NAMES, MASK_FAMILIES, FeatureRecord, FeatureTable, _evaluation_records,
    _observations, _error_sums, _finalize_errors, evaluate_probe_suite,
    reconstruction_mask, source_bootstrap, fit_ridge_probe, RidgeProbe,
)
from .objectives import full_observation_states, render_group

if TYPE_CHECKING:
    from .config import HybridPilotConfig
    from .data import HybridPilotDataset
    from .runner import DistributedContext, DeadlineController
    from .logging import RunLogger


def fit_efficient_ridge_probe(features: torch.Tensor, labels: torch.Tensor, *,
                              classes: int, ridge: float = 1.) -> RidgeProbe:
    """Same standardized mean-Gram ridge, solving the smaller dual when N<D.

    The regularizer remains lambda*I after division by N, and the intercept is
    unpenalized. Only the linear-system shape changes; training statistics,
    feature dimensions, targets and predictions retain the shared protocol.
    """
    if features.ndim == 2 and features.shape[0] >= features.shape[1]:
        return fit_ridge_probe(features, labels, classes=classes, ridge=ridge)
    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if type(classes) is not int or classes < 2:
        raise ValueError("classes must be an integer >= 2")
    x = features.detach().to(device="cpu", dtype=torch.float64)
    y = labels.detach().to(device="cpu")
    if x.ndim != 2 or min(x.shape) == 0 or not torch.isfinite(x).all():
        raise ValueError("features must be a finite nonempty matrix")
    if y.dtype != torch.int64 or y.shape != (len(x),) or (y < 0).any() or (y >= classes).any():
        raise ValueError("labels must be int64 [N] within the declared classes")
    if len(torch.unique(y)) != classes:
        raise ValueError("training labels must cover all declared classes")
    mean = x.mean(0)
    scale = x.std(0, correction=0).clamp_min(1e-4)
    z = (x - mean) / scale
    target = F.one_hot(y, classes).double()
    intercept = target.mean(0)
    gram = z @ z.T / len(x)
    gram.diagonal().add_(ridge)
    weights = z.T @ torch.linalg.solve(gram, (target - intercept) / len(x))
    return RidgeProbe(mean, scale, weights, intercept)


@torch.inference_mode()
def extract_features(model: HybridCDModel, dataset: HybridPilotDataset, groups: Sequence[str],
                     cfg: HybridPilotConfig, device: torch.device | str) -> FeatureTable:
    labels = _evaluation_records(dataset)
    records, values = [], {name: [] for name in FEATURE_NAMES}
    was_training = model.training
    model.eval()
    try:
        for group in groups:
            batch = dataset.load_group(group, device=device)
            final = full_observation_states(model, [batch])[-1]
            values["c"].append(final.c.flatten(1).cpu().clone())
            values["d"].append(final.d.flatten(1).cpu().clone())
            observations = _observations(batch, cfg.T - model.config.frames_per_step, cfg.T)
            initial = model.initial_state(len(observations),
                last_time=torch.stack([item.frame_times.min() for item in observations]))
            suffix = model.step(initial, observations)
            for name, attribute in (("last_c", "c"), ("last_d", "d")):
                values[name].append(getattr(suffix, attribute).flatten(1).cpu().clone())
            per_frame = []
            for frame in range(cfg.T):
                per_frame.append(model.observer.forward_batch(_observations(batch, frame, frame + 1))[0].flatten(1))
            ordered = torch.stack(per_frame).double().sort(dim=0).values
            values["orderless"].append(ordered.mean(0).float().cpu().clone())
            for row in dataset.records_for_group(group):
                label = labels[row["record_id"]]
                if label["group_key"] != group or label["split"] != row["split"]:
                    raise ValueError("evaluation metadata does not match source inputs")
                records.append(FeatureRecord(row["record_id"], group, row["split"],
                    EVENT_CORES.index(label["core"]), EVENT_CONFIGURATIONS.index(label["configuration"])))
    finally:
        model.train(was_training)
    return FeatureTable(tuple(records), {name: torch.cat(items) for name, items in values.items()})


def _merge(tables):
    return FeatureTable(tuple(record for table in tables for record in table.records),
        {name: torch.cat([table.features[name] for table in tables]) for name in FEATURE_NAMES})


def _gather(local, context):
    if not context.distributed:
        return [local]
    result = [None] * context.world_size if context.rank == 0 else None
    dist.gather_object(local, result, dst=0)
    return result


def _broadcast(value, context):
    result = [value]
    if context.distributed:
        dist.broadcast_object_list(result, src=0)
    return result[0]


@torch.inference_mode()
def reconstruction_group(model: HybridCDModel, dataset: HybridPilotDataset, group: str,
                         cfg: HybridPilotConfig, device: torch.device | str, *, source_index: int) -> dict[str, Any]:
    batch = dataset.load_group(group, device=device)
    if any(row["split"] != "val" for row in dataset.records_for_group(group)):
        raise ValueError("final reconstruction uses held-out sources only")
    labels = _evaluation_records(dataset)
    rows = [labels[row["record_id"]] for row in dataset.records_for_group(group)]
    by_variant = {(row["core"], row["configuration"]): index for index, row in enumerate(rows)}
    permutation = torch.tensor([by_variant[(row["core"], EVENT_CONFIGURATIONS[
        1 - EVENT_CONFIGURATIONS.index(row["configuration"])])] for row in rows], device=batch.images.device)
    families = {}
    for family_index, family in enumerate(MASK_FAMILIES):
        generator = torch.Generator().manual_seed(cfg.eval_mask_seed + 1009 * source_index + 101 * family_index)
        mask, endpoint = reconstruction_mask(batch, family, generator)
        state = full_observation_states(model, [batch], [mask], stop_frame=endpoint + 1)[-1]
        target = batch.images[:, 0, endpoint].permute(0, 2, 3, 1).flatten(1, 2)
        variants = {"baseline": state, "c_zero": replace(state, c=torch.zeros_like(state.c)),
                    "d_zero": replace(state, d=torch.zeros_like(state.d)),
                    "d_swapped": replace(state, d=state.d[permutation].contiguous())}
        entries = {}
        for name, variant in variants.items():
            prediction = render_group(model, variant, batch, endpoint).permute(0, 2, 3, 1).flatten(1, 2)
            entries[name] = _error_sums(prediction, target, mask[:, 0, endpoint].flatten(1))
            if name in ("baseline", "d_swapped"):
                entries[name + "_opposite_configuration_target"] = _error_sums(
                    prediction, target[permutation], mask[:, 0, endpoint].flatten(1))
            if name == "c_zero" and torch.count_nonzero(prediction):
                raise AssertionError("C-zero decoder contract violated")
        families[family] = entries
    return families


def _reconstruction_summary(groups, entries):
    result = {"source_groups": list(groups), "data_range": 2, "clipped": False, "families": {}}
    for family in MASK_FAMILIES:
        summaries = {}
        for variant in ("baseline", "c_zero", "d_zero", "d_swapped",
                        "baseline_opposite_configuration_target", "d_swapped_opposite_configuration_target"):
            sums = {region: {"rgb_count": 0, "absolute_sum": 0., "squared_sum": 0.}
                    for region in ("observed", "missing", "all")}
            for entry in entries:
                for region, metrics in entry[family][variant].items():
                    for key, value in metrics.items():
                        sums[region][key] += value
            summaries[variant] = _finalize_errors(sums)
        result["families"][family] = {**summaries["baseline"], "interventions": {
            name: {"errors": summaries[name],
                   "all_l1_change": summaries[name]["all"]["l1"] - summaries["baseline"]["all"]["l1"]}
            for name in ("c_zero", "d_zero", "d_swapped")}}
        result["families"][family]["interventions"]["d_swapped"].update(
            opposite_configuration_target=summaries["d_swapped_opposite_configuration_target"],
            opposite_configuration_l1_gain=(summaries["baseline_opposite_configuration_target"]["all"]["l1"]
                                            - summaries["d_swapped_opposite_configuration_target"]["all"]["l1"]))
    result["c_zero_exact_zero_verified"] = True
    result["intervention_interpretation"] = "D swap uses the same source/core at opposite configuration; C, query and time stay fixed"
    return result


@torch.inference_mode()
def evaluate_distributed(model: HybridCDModel, cfg: HybridPilotConfig, dataset: HybridPilotDataset,
                         context: DistributedContext, deadlines: DeadlineController, logger: RunLogger,
                         *, global_step: int, label: str = "trained") -> dict[str, Any]:
    """All ranks enter identical boundary collectives; only rank zero fits probes."""
    from .runner import rank_zero_call
    model.eval()
    tables = {}
    for split, groups in (("train", dataset.train_groups), ("val", dataset.val_groups)):
        local = []
        rounds = math.ceil(len(groups) / context.world_size)
        for round_index in range(rounds):
            if deadlines.should_stop(context, optimization=False):
                return {"status": "stopped", "reason": "work deadline or signal", "label": label}
            index = round_index * context.world_size + context.rank
            if index < len(groups):
                local.append(extract_features(model, dataset, [groups[index]], cfg, context.device))
            rank_zero_call(context, lambda: logger.log(
                {f"eval/{label}/feature_{split}_fraction": (round_index + 1) / rounds}, step=global_step))
        gathered = _gather(local, context)
        def merge_tables():
            tables[split] = _merge([table for partition in gathered for table in partition])
        rank_zero_call(context, merge_tables)
    if deadlines.should_stop(context, optimization=False):
        return {"status": "stopped", "reason": "work deadline before probes", "label": label}
    probe = error = None
    probe_stopped = False
    if context.rank == 0:
        try:
            # Standardization and ridge are fitted solely on train sources by
            # the shared, already audited probe protocol.
            def bounded_fitter(*args, **kwargs):
                if deadlines.requested or datetime.now(timezone.utc) >= deadlines.work_at:
                    raise TimeoutError("work deadline reached between probe fits")
                return fit_efficient_ridge_probe(*args, **kwargs)
            probe = evaluate_probe_suite(tables["train"], tables["val"], cfg, fit_probe=bounded_fitter)
            probe["source_intervals"] = {name: source_bootstrap(scores, samples=cfg.bootstrap,
                seed=cfg.bootstrap_seed) for name, scores in probe["group_scores"].items()}
        except TimeoutError:
            probe_stopped = True
        except Exception as exc:
            error = f"probe evaluation failed: {type(exc).__name__}: {exc}"
    if _broadcast(probe_stopped, context):
        return {"status": "stopped", "reason": "work deadline during probes", "label": label}
    error = _broadcast(error, context)
    if error:
        raise RuntimeError(error)
    if deadlines.should_stop(context, optimization=False):
        return {"status": "stopped", "reason": "work deadline after probes", "label": label}
    selected = sorted(dataset.val_groups)[:cfg.eval_reconstruction_groups]
    local_reconstruction = []
    for round_index in range(math.ceil(len(selected) / context.world_size)):
        if deadlines.should_stop(context, optimization=False):
            return {"status": "stopped", "reason": "work deadline during reconstruction", "label": label}
        index = round_index * context.world_size + context.rank
        if index < len(selected):
            local_reconstruction.append(reconstruction_group(model, dataset, selected[index], cfg,
                context.device, source_index=index))
    gathered = _gather(local_reconstruction, context)
    def finalize():
        result = {"status": "complete", "label": label, **probe,
            "reconstruction": _reconstruction_summary(selected, [x for items in gathered for x in items]),
            "control_contract": "last C/D uses fresh complete state; orderless uses individual frames; both C/D stages flattened"}
        logger.log({f"eval/{label}/{key}": value for key, value in result["metrics"].items()}, step=global_step)
        return result
    return rank_zero_call(context, finalize)
