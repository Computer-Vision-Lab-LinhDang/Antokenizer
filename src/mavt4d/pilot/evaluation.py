"""Frozen, train-only linear probes and bounded reconstruction diagnostics.

Labels enter this module only after representation training has finished.
Neither the observation adapter nor the model receives an evaluation record.
Bootstrap intervals condition on the trained seeds and resample source groups,
not the correlated eight core/configuration variants within a source.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from mavt4d.contracts import DecoderQuery, ObservationStep
from mavt4d.data.event_synthetic import EVENT_CONFIGURATIONS, EVENT_CORES


FEATURE_NAMES = ("c", "d", "last_c", "last_d", "orderless")
MASK_FAMILIES = ("clean", "wide", "tube", "snapshot", "never_seen")


@dataclass(frozen=True)
class FeatureRecord:
    """Evaluation metadata, kept separate from every model input."""

    record_id: str
    group_key: str
    split: str
    core: int
    configuration: int

    def __post_init__(self):
        if not self.record_id or not self.group_key:
            raise ValueError("record_id and group_key must be nonempty")
        if self.split not in ("train", "val"):
            raise ValueError("split must be train or val")
        if type(self.core) is not int or not 0 <= self.core < 4:
            raise ValueError("core must be an integer in [0, 4)")
        if type(self.configuration) is not int or not 0 <= self.configuration < 2:
            raise ValueError("configuration must be an integer in [0, 2)")


@dataclass(frozen=True)
class FeatureTable:
    records: tuple[FeatureRecord, ...]
    features: dict[str, Tensor]

    def __post_init__(self):
        if not self.records or len({r.record_id for r in self.records}) != len(self.records):
            raise ValueError("feature records must be nonempty, with no duplicate record IDs")
        if set(self.features) != set(FEATURE_NAMES):
            raise ValueError(f"features must have exactly {FEATURE_NAMES}")
        for name, values in self.features.items():
            if values.ndim != 2 or values.shape[0] != len(self.records) or values.shape[1] == 0:
                raise ValueError(f"{name}: expected nonempty [records, features]")
            if not torch.isfinite(values).all():
                raise ValueError(f"{name}: features must be finite")


@dataclass(frozen=True)
class RidgeProbe:
    mean: Tensor
    scale: Tensor
    weights: Tensor
    intercept: Tensor

    def predict(self, features: Tensor) -> Tensor:
        values = features.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.numel():
            raise ValueError("probe features have incorrect dimensionality")
        if not torch.isfinite(values).all():
            raise ValueError("probe features must be finite")
        return (((values - self.mean) / self.scale) @ self.weights + self.intercept).argmax(1)


def fit_ridge_probe(features: Tensor, labels: Tensor, *, classes: int, ridge: float = 1.0) -> RidgeProbe:
    """Fit standardized mean-Gram ridge; only the supplied training rows count."""
    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if type(classes) is not int or classes < 2:
        raise ValueError("classes must be an integer >= 2")
    x = features.detach().to(device="cpu", dtype=torch.float64)
    y = labels.detach().to(device="cpu")
    if x.ndim != 2 or min(x.shape) == 0 or not torch.isfinite(x).all():
        raise ValueError("features must be a finite nonempty matrix")
    if y.dtype != torch.int64 or y.shape != (x.shape[0],) or (y < 0).any() or (y >= classes).any():
        raise ValueError("labels must be int64 [N] within the declared classes")
    if len(torch.unique(y)) != classes:
        raise ValueError("training labels must cover all declared classes")
    mean = x.mean(0)
    scale = x.std(0, correction=0).clamp_min(1e-4)
    z = (x - mean) / scale
    target = F.one_hot(y, classes).double()
    intercept = target.mean(0)
    gram = z.T @ z / len(x)
    gram.diagonal().add_(ridge)
    weights = torch.linalg.solve(gram, z.T @ (target - intercept) / len(x))
    return RidgeProbe(mean, scale, weights, intercept)


def _crossing_groups(table: FeatureTable) -> set[str]:
    groups = defaultdict(list)
    for record in table.records:
        groups[record.group_key].append((record.core, record.configuration))
    expected = {(core, configuration) for core in range(4) for configuration in range(2)}
    if any(len(values) != 8 or set(values) != expected for values in groups.values()):
        raise ValueError("each source requires a complete eight-variant core/configuration crossing")
    return set(groups)


def evaluate_probe_suite(train: FeatureTable, val: FeatureTable, cfg, *, fit_probe=None) -> dict:
    """Fit on training sources only; emit per-source scores for paired inference."""
    fit_probe = fit_ridge_probe if fit_probe is None else fit_probe
    train_groups, val_groups = _crossing_groups(train), _crossing_groups(val)
    if train_groups & val_groups:
        raise ValueError("train and val source groups must be disjoint; overlap detected")
    if any(r.split != "train" for r in train.records) or any(r.split != "val" for r in val.records):
        raise ValueError("probe fit must use train records and evaluation must use val records")
    if any(train.features[name].shape[1] != val.features[name].shape[1] for name in FEATURE_NAMES):
        raise ValueError("train and val feature dimensions differ")
    result = {"metrics": {}, "group_scores": {}, "predictions": {},
              "feature_dimensions": {name: train.features[name].shape[1] for name in FEATURE_NAMES},
              "train_sources": len(train_groups), "val_sources": len(val_groups),
              "ridge": cfg.probe_ridge, "standardization": "train-only, population std floor 1e-4"}

    def score(name, feature, target, classes, selection=lambda r: True):
        train_ids = [i for i, row in enumerate(train.records) if selection(row)]
        val_ids = [i for i, row in enumerate(val.records) if selection(row)]
        labels = torch.tensor([target(train.records[i]) for i in train_ids], dtype=torch.int64)
        probe = fit_probe(train.features[feature][train_ids], labels,
                          classes=classes, ridge=cfg.probe_ridge)
        predictions = probe.predict(val.features[feature][val_ids]).tolist()
        rows, correct_by_group = [], defaultdict(list)
        for index, prediction in zip(val_ids, predictions):
            row = val.records[index]
            actual = target(row)
            correct_by_group[row.group_key].append(float(prediction == actual))
            rows.append({"record_id": row.record_id, "group_key": row.group_key,
                         "target": actual, "prediction": prediction})
        # Complete crossings make accuracy equal balanced accuracy, including
        # the two configuration classes within every core/source subset.
        group_scores = {group: sum(values) / len(values) for group, values in correct_by_group.items()}
        result["group_scores"][name] = group_scores
        result["metrics"][name] = sum(group_scores.values()) / len(group_scores)
        result["predictions"][name] = rows

    for feature in FEATURE_NAMES:
        score(f"{feature}_core", feature, lambda r: r.core, 4)
        score(f"{feature}_configuration", feature, lambda r: r.configuration, 2)
        family_names = []
        for family, lower in (("lateral", 0), ("loop", 2)):
            name = f"{feature}_direction_{family}"
            score(name, feature, lambda r: r.core % 2, 2,
                  lambda r, lower=lower: lower <= r.core < lower + 2)
            family_names.append(name)
        name = f"{feature}_direction"
        result["group_scores"][name] = {
            group: sum(result["group_scores"][family][group] for family in family_names) / 2
            for group in sorted(val_groups)
        }
        result["metrics"][name] = sum(result["group_scores"][name].values()) / len(val_groups)
        result["predictions"][name] = [row for family in family_names
                                         for row in result["predictions"][family]]
    for core in range(4):
        # Stratify the one global D/configuration probe. Refitting a different
        # probe per core would conceal incompatible configuration mappings.
        ids = {row.record_id for row in val.records if row.core == core}
        rows = [row for row in result["predictions"]["d_configuration"] if row["record_id"] in ids]
        by_group = defaultdict(list)
        for row in rows:
            by_group[row["group_key"]].append(float(row["prediction"] == row["target"]))
        name = f"d_configuration_core_{core}"
        group_scores = {group: sum(values) / len(values) for group, values in by_group.items()}
        result["predictions"][name] = rows
        result["group_scores"][name] = group_scores
        result["metrics"][name] = sum(group_scores.values()) / len(group_scores)
    return result


def source_bootstrap(group_scores: dict[str, float], *, samples: int = 1000, seed: int = 31415) -> dict:
    """Percentile interval from source groups, with every group weighted equally."""
    if type(samples) is not int or samples < 1 or not group_scores:
        raise ValueError("bootstrap needs positive samples and nonempty source scores")
    values = torch.tensor([group_scores[key] for key in sorted(group_scores)], dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("source scores must be finite")
    indices = torch.randint(len(values), (samples, len(values)), generator=torch.Generator().manual_seed(seed))
    distribution = values[indices].mean(1)
    return {"mean": float(values.mean()), "low": float(torch.quantile(distribution, .025)),
            "high": float(torch.quantile(distribution, .975)), "n_groups": len(values)}


def summarize_runs(run_results: list[dict], cfg) -> dict:
    """Summarize seed variability separately from paired source uncertainty."""
    if not run_results:
        raise ValueError("run_results must not be empty")
    by_cell = defaultdict(dict)
    expected_metrics = set(run_results[0]["result"]["metrics"])
    expected_groups = None
    for run in run_results:
        cell, seed, result = run["cell"], run["seed"], run["result"]
        if seed in by_cell[cell]:
            raise ValueError("duplicate cell/seed run")
        if set(result["metrics"]) != expected_metrics or set(result["group_scores"]) != expected_metrics:
            raise ValueError("all runs must share the same metrics")
        for scores in result["group_scores"].values():
            groups = set(scores)
            if expected_groups is None:
                expected_groups = groups
            if groups != expected_groups:
                raise ValueError("paired runs must use identical source groups")
        by_cell[cell][seed] = result
    seeds = set(next(iter(by_cell.values())))
    if any(set(runs) != seeds for runs in by_cell.values()):
        raise ValueError("paired cells must have identical training seed sets")
    result = {"cells": {}, "paired": {}, "seeds": sorted(seeds),
              "interval_interpretation": "Source bootstrap conditional on these fitted seeds; seed SD reported separately."}
    means = {}
    for cell, runs in sorted(by_cell.items()):
        result["cells"][cell] = {}
        means[cell] = {}
        for metric in sorted(expected_metrics):
            seed_values = torch.tensor([runs[seed]["metrics"][metric] for seed in sorted(seeds)], dtype=torch.float64)
            means[cell][metric] = {group: sum(runs[seed]["group_scores"][metric][group] for seed in seeds) / len(seeds)
                                   for group in sorted(expected_groups)}
            interval = source_bootstrap(means[cell][metric], samples=cfg.bootstrap, seed=cfg.bootstrap_seed)
            result["cells"][cell][metric] = {**interval, "mean": float(seed_values.mean()),
                "seed_sd": float(seed_values.std(correction=1)) if len(seeds) > 1 else 0.0,
                "n_seeds": len(seeds)}
    for first, second in combinations(sorted(by_cell), 2):
        result["paired"][f"{first}-minus-{second}"] = {
            metric: source_bootstrap({group: means[first][metric][group] - means[second][metric][group]
                                      for group in sorted(expected_groups)},
                                     samples=cfg.bootstrap, seed=cfg.bootstrap_seed)
            for metric in sorted(expected_metrics)
        }
    return result


def _observations(batch, start, stop, availability=None):
    availability = batch.observed if availability is None else availability
    observations = []
    for index in range(len(batch.images)):
        observed = availability[index, :, start:stop].clone().contiguous()
        pixels = batch.images[index, :, start:stop]
        observations.append(ObservationStep(
            images=torch.where(observed[:, :, None], pixels, torch.zeros_like(pixels)).clone().contiguous(),
            observed=observed,
            frame_times=batch.frame_times[index, :, start:stop].clone().contiguous(),
            intrinsics=batch.intrinsics[index].clone().contiguous(),
            cam_to_world=batch.cam_to_world[index, :, start:stop].clone().contiguous(),
        ))
    return observations


def _final_state(model, batch, *, start=0, stop=None, availability=None):
    stop = batch.images.shape[2] if stop is None else stop
    initial = model.initial_state(len(batch.images), last_time=float(batch.frame_times[:, :, start].min()))
    steps = [_observations(batch, offset, min(offset + model.spec.frames_per_step, stop), availability)
             for offset in range(start, stop, model.spec.frames_per_step)]
    return model.forward_sequence(initial, steps, method="sequential")[-1]


def _evaluation_records(dataset) -> dict[str, dict]:
    path = dataset.root / "eval_targets_v1.jsonl"
    raw = path.read_bytes()
    summary = json.loads((dataset.root / "summary.json").read_text())
    if hashlib.sha256(raw).hexdigest() != summary["sha256"]["evaluation"]:
        raise ValueError("evaluation manifest checksum/sha256 does not match the locked dataset")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    records = {row["record_id"]: row for row in rows}
    if len(records) != len(rows):
        raise ValueError("duplicate evaluation record IDs")
    return records


@torch.inference_mode()
def extract_features(model, dataset, groups, cfg, device) -> FeatureTable:
    """Extract recurrent, reset-last-tubelet and truly orderless features.

    Orderless pooling encodes *individual frames*. Sorting each feature's
    values before summation makes reversal invariance exact, including floating
    point summation order. The observer's single-frame relative time is zero.
    """
    labels = _evaluation_records(dataset)
    records, features = [], {name: [] for name in FEATURE_NAMES}
    was_training = model.training
    model.eval()
    try:
        for group in groups:
            batch = dataset.load_group(group, device=device)
            rows = dataset.records_for_group(group)
            if len(rows) != len(batch.images):
                raise ValueError("evaluation metadata ordering does not match the input batch")
            final = _final_state(model, batch)
            for name, tensor in (("c", final.c), ("d", final.d)):
                features[name].append(tensor.flatten(1).detach().cpu().clone())
            # Use the same single-stream execution shape for every suffix.
            # A batched linear kernel can round identical rows differently by
            # row position; fixed variant ordering must not become a cue.
            suffix_states = []
            for item in _observations(batch, batch.images.shape[2] - model.spec.frames_per_step,
                                      batch.images.shape[2]):
                reset = model.initial_state(1, last_time=float(item.frame_times.min()))
                suffix_states.append(model.step(reset, item))
            for name, attribute in (("last_c", "c"), ("last_d", "d")):
                values = torch.cat([getattr(state, attribute).flatten(1) for state in suffix_states])
                features[name].append(values.cpu().clone())
            frame_features = []
            for frame in range(batch.images.shape[2]):
                frame_features.append(torch.stack([model.observer(item)[0].flatten()
                                                    for item in _observations(batch, frame, frame + 1)]))
            ordered = torch.stack(frame_features).double().sort(dim=0).values
            features["orderless"].append(ordered.mean(0).float().cpu().clone())
            for row in rows:
                label = labels[row["record_id"]]
                if label["group_key"] != group or label["split"] != row["split"]:
                    raise ValueError("input/evaluation source metadata do not match")
                records.append(FeatureRecord(row["record_id"], group, row["split"],
                    EVENT_CORES.index(label["core"]), EVENT_CONFIGURATIONS.index(label["configuration"])))
    finally:
        model.train(was_training)
    if not records:
        raise ValueError("feature extraction requires at least one source")
    return FeatureTable(tuple(records), {name: torch.cat(values) for name, values in features.items()})


def reconstruction_mask(batch, family: str, generator: torch.Generator) -> tuple[Tensor, int]:
    """Fixed evaluation schedules guarantee that missing metrics use a gap endpoint."""
    if family not in MASK_FAMILIES:
        raise ValueError(f"family must be one of {MASK_FAMILIES}")
    _, views, frames, height, width = batch.observed.shape
    if frames < 8:
        raise ValueError("reconstruction evaluation requires at least eight frames")
    mask = torch.ones((1, views, frames, height, width), dtype=torch.bool)
    endpoint = 7 if family == "snapshot" else frames - 1
    if family == "snapshot":
        mask[:, :, 4:8] = False
    elif family != "clean":
        rh, rw = max(1, height // 2), max(1, width // (4 if family == "tube" else 2))
        y = int(torch.randint(height - rh + 1, (), generator=generator))
        x = int(torch.randint(width - rw + 1, (), generator=generator))
        start = {"wide": frames - 2, "tube": 4, "never_seen": 0}[family]
        mask[:, :, start:, y:y + rh, x:x + rw] = False
    return mask.expand_as(batch.observed).to(batch.observed.device) & batch.observed, endpoint


def _error_sums(prediction, target, observed):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("RGB prediction/target must have equal [B,Q,3] shapes")
    if observed.shape != prediction.shape[:2] or observed.dtype != torch.bool:
        raise ValueError("observed must be bool [B,Q]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("RGB metrics require finite predictions and targets")
    error = (prediction.double() - target.double())
    result = {}
    for region, mask in (("observed", observed), ("missing", ~observed), ("all", torch.ones_like(observed))):
        selected = error[mask]
        result[region] = {"rgb_count": selected.numel(), "absolute_sum": float(selected.abs().sum()),
                          "squared_sum": float(selected.square().sum())}
    return result


def _finalize_errors(sums):
    result = {}
    for region, values in sums.items():
        count = values["rgb_count"]
        mse = values["squared_sum"] / count if count else None
        result[region] = {"rgb_count": count, "l1": values["absolute_sum"] / count if count else None,
                          "mse": mse, "psnr": 10 * math.log10(4 / mse) if mse is not None and mse > 0 else None,
                          "perfect": mse == 0 if count else False}
    return result


def rgb_error_summary(prediction: Tensor, target: Tensor, observed: Tensor) -> dict:
    """No output clipping. Empty or perfect-region PSNR is null (see perfect flag)."""
    return _finalize_errors(_error_sums(prediction, target, observed))


@torch.inference_mode()
def evaluate_reconstruction(model, dataset, groups, cfg, device) -> dict:
    """Aggregate RGB errors before division/PSNR over the fixed first val sources."""
    selected_groups = sorted(groups)[:cfg.eval_reconstruction_groups]
    if not selected_groups:
        raise ValueError("reconstruction evaluation requires held-out sources")
    was_training = model.training
    model.eval()
    result = {"source_groups": selected_groups, "data_range": 2, "clipped": False, "families": {}}
    try:
        for family_index, family in enumerate(MASK_FAMILIES):
            totals = {region: {"rgb_count": 0, "absolute_sum": 0., "squared_sum": 0.}
                      for region in ("observed", "missing", "all")}
            endpoint = None
            for group_index, group in enumerate(selected_groups):
                batch = dataset.load_group(group, device=device)
                if any(row["split"] != "val" for row in dataset.records_for_group(group)):
                    raise ValueError("reconstruction evaluation accepts held-out val sources only")
                generator = torch.Generator().manual_seed(cfg.eval_mask_seed + 1009 * group_index + 101 * family_index)
                mask, endpoint = reconstruction_mask(batch, family, generator)
                state = _final_state(model, batch, stop=endpoint + 1, availability=mask)
                _, views, _, _, height, width = batch.images.shape
                if views != 1:
                    raise ValueError("pilot reconstruction expects one camera view")
                if not torch.equal(batch.intrinsics, batch.intrinsics[:1].expand_as(batch.intrinsics)) or not torch.equal(
                        batch.cam_to_world[:, :, endpoint], batch.cam_to_world[:1, :, endpoint].expand_as(batch.cam_to_world[:, :, endpoint])):
                    raise ValueError("shared decoder query requires identical source calibration")
                times = batch.frame_times[:, 0, endpoint]
                if not torch.equal(times, times[:1].expand_as(times)):
                    raise ValueError("shared decoder query requires identical source timestamps")
                y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
                coordinates = torch.stack((x + .5, y + .5, torch.zeros_like(x)), dim=-1).reshape(-1, 3).to(batch.images.dtype)
                query = DecoderQuery(coordinates=coordinates, view_ids=torch.zeros(height * width, dtype=torch.int64, device=device),
                    times=times[0].expand(height * width).clone(), intrinsics=batch.intrinsics[0],
                    cam_to_world=batch.cam_to_world[0, :, endpoint], coordinate_space="image_pixel")
                prediction = model.decode(state, query)
                target = batch.images[:, 0, endpoint].permute(0, 2, 3, 1).reshape(len(batch.images), -1, 3)
                sums = _error_sums(prediction, target, mask[:, 0, endpoint].reshape(len(batch.images), -1))
                for region in totals:
                    for key in totals[region]:
                        totals[region][key] += sums[region][key]
            result["families"][family] = {"endpoint_frame": endpoint, **_finalize_errors(totals)}
    finally:
        model.train(was_training)
    return result
