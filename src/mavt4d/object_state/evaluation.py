"""Internal causal evaluation; sampled and full-frame coverage stay explicit.

This evaluates the factorial fixture only. Copy-last RGB has access to observed
views only; copy-last state and the learned rollout use the same posterior and
query cameras. Pose identities are matched once at the prefix cutoff.
"""

from __future__ import annotations

from dataclasses import replace
import math
import time
from typing import Callable

import torch
from torch import Tensor

from .data import observation_step, sample_ray_targets
from .decoder import RayQuery
from .objectives import matched_pose_loss
from .training import TrainingConfig, select_batch, tree_to


class MetricTotals:
    """Accumulate sums/counts, never average incompatible batch averages."""

    def __init__(self):
        self.values: dict[str, float] = {}

    def _add(self, name: str, value: Tensor | float):
        self.values[name] = self.values.get(name, 0.0) + float(value)

    def add(self, prediction: dict, target: dict, mask: Tensor | None = None):
        valid = target["valid"] if mask is None else target["valid"] & mask
        error = (prediction["rgb"] - target["rgb"]).float()
        self._add("rays", valid.sum())
        self._add("rgb_abs", torch.where(valid[..., None], error.abs(), 0.0).sum())
        self._add(
            "rgb_square", torch.where(valid[..., None], error.square(), 0.0).sum()
        )
        for name, region in (
            ("foreground", target["instance"] > 0),
            ("background", target["instance"] <= 0),
        ):
            selected = valid & region
            self._add(name + "_rays", selected.sum())
            self._add(
                name + "_abs", torch.where(selected[..., None], error.abs(), 0.0).sum()
            )
        if "range" in prediction:
            hit = valid & target["depth_valid"]
            self._add("target_hits", hit.sum())
            self._add(
                "range_abs",
                torch.where(
                    hit, (prediction["range"] - target["range"]).abs(), 0.0
                ).sum(),
            )
            self._add("predicted_target_hits", (prediction["hit"] & hit).sum())

    def add_pose(self, values: dict):
        for name, value in values.items():
            self._add(name, value)

    def report(self) -> dict:
        v = self.values
        result = {"rays": int(v.get("rays", 0))}
        if result["rays"]:
            count = 3 * result["rays"]
            result["rgb_mae"] = v["rgb_abs"] / count
            result["psnr_db"] = -10 * math.log10(max(v["rgb_square"] / count, 1e-12))
            available = []
            for name in ("foreground", "background"):
                if v.get(name + "_rays", 0):
                    result["rgb_" + name + "_mae"] = v[name + "_abs"] / (
                        3 * v[name + "_rays"]
                    )
                    available.append(result["rgb_" + name + "_mae"])
            result["rgb_region_balanced_mae"] = sum(available) / len(available)
        if v.get("target_hits", 0):
            result["range_mae"] = v["range_abs"] / v["target_hits"]
            result["target_hit_coverage"] = (
                v["predicted_target_hits"] / v["target_hits"]
            )
            result["target_hit_rays"] = int(v["target_hits"])
        if v.get("target_objects", 0):
            result["matched_gt_coverage"] = (
                v.get("matched_objects", 0) / v["target_objects"]
            )
            result["target_object_instances"] = int(v["target_objects"])
        if v.get("matched_objects", 0):
            result["fixed_identity_translation_l2"] = (
                v["translation_error_sum"] / v["matched_objects"]
            )
            result["fixed_identity_rotation_chordal"] = (
                v["rotation_error_sum"] / v["matched_objects"]
            )
        return result


def fixed_identity_pose_metrics(
    predicted: Tensor, target: Tensor, valid: Tensor, matching: Tensor
) -> dict:
    good = valid & (matching >= 0)
    b, g = good.nonzero(as_tuple=True)
    selected = predicted[b, matching[b, g]]
    truth = target[b, g]
    return {
        "translation_error_sum": float(
            (selected[..., :3, 3] - truth[..., :3, 3]).norm(dim=-1).sum()
        ),
        "rotation_error_sum": float(
            (selected[..., :3, :3] - truth[..., :3, :3]).square().mean((-1, -2)).sum()
        ),
        "matched_objects": int(good.sum()),
        "target_objects": int(valid.sum()),
    }


def _query_slice(query: RayQuery, start: int, end: int) -> RayQuery:
    def bound(value):
        return (
            value[:, start:end]
            if isinstance(value, Tensor) and value.ndim == 2 and value.shape[1] > 1
            else value
        )

    return replace(
        query,
        origins=query.origins[:, start:end],
        directions=query.directions[:, start:end],
        near=bound(query.near),
        far=bound(query.far),
        valid=None if query.valid is None else query.valid[:, start:end],
    )


def _read_with_deadline(model, state, query: RayQuery, deadline: Callable[[], bool]):
    """Stop between renderer chunks; incomplete frames are counted, not scored."""
    chunks = []
    done = 0
    size = model.config.query_chunk_size
    for start in range(0, query.origins.shape[1], size):
        if not deadline():
            return None, done
        end = min(query.origins.shape[1], start + size)
        chunks.append(model.read(state, _query_slice(query, start, end)))
        done += end - start
    result = {
        name: torch.cat([part[name] for part in chunks], 1)
        for name in chunks[0]
        if name != "controls"
    }
    result["controls"] = chunks[0]["controls"]
    return result, done


def _copy_last_rgb(batch, rays):
    obs = batch["observations"]["rgb"][:, -1]
    b, views, _, h, w = obs.shape
    valid = rays["view_indices"] < views
    ids = rays["pixel_indices"].clamp(max=views * h * w - 1)
    image = obs.permute(0, 1, 3, 4, 2).reshape(b, views * h * w, 3)
    rgb = image[torch.arange(b, device=image.device)[:, None], ids]
    return {"rgb": rgb}, valid


def evaluate(
    model,
    dataset,
    config: TrainingConfig,
    *,
    source_limit: int,
    rays: int | None,
    seed: int = 1701,
    deadline: Callable[[], bool] = lambda: True,
    visuals: bool = True,
) -> dict:
    if not 0 < source_limit <= len(dataset):
        raise ValueError("source_limit must be within the dataset")
    if rays is not None and (type(rays) is not int or rays < 1):
        raise ValueError("rays must be positive or None for full frame")
    started = time.monotonic()
    device = next(model.parameters()).device
    was_training = model.training
    meters: dict[str, MetricTotals] = {}
    images: dict[str, Tensor] = {}
    frames = [
        config.prefix_frames - 1,
        *range(config.prefix_frames + 1, config.frames, 2),
    ]
    result = {
        "sampling": "full_frame" if rays is None else "fixed_seed_balanced_rays",
        "seed": seed,
        "rays_per_frame": rays,
        "sources_requested": source_limit,
        "sources_available": len(dataset),
        "sources_completed": 0,
        "variants_expected": source_limit * 8,
        "variants_completed": 0,
        "frames_expected": source_limit * 8 * len(frames),
        "frames_completed": 0,
        "incomplete_frame_rays_discarded": 0,
        "complete": False,
        "benchmark": "internal_factorial_fixture",
        "occlusion_return_metric": "not_implemented",
        "identity_protocol": "prefix_cutoff_assignment_fixed_for_future",
        "copy_last_rgb_protocol": "last_observed_image_same_pixel_observed_views_only",
    }
    model.eval()
    try:
        with torch.no_grad():
            for source in range(source_limit):
                if not deadline():
                    break
                group = dataset[source]
                source_complete = True
                for variant in range(8):
                    if not deadline():
                        source_complete = False
                        break
                    batch = tree_to(
                        select_batch(group, torch.tensor([variant])), device
                    )
                    state = model.initial_state(
                        1, device=device, time=batch["observations"]["times"][:, 0]
                    )
                    for frame in range(1, config.prefix_frames, 2):
                        if not deadline():
                            break
                        observation = observation_step(
                            batch, frame, tubelet_size=model.config.tubelet_size
                        )
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=device.type == "cuda",
                        ):
                            evidence = model.encode_observation(observation)
                        state = model.correct(
                            model.predict(state, evidence.time.double() - state.t),
                            evidence,
                        )
                    if not deadline():
                        source_complete = False
                        break
                    posterior = state
                    cutoff = config.prefix_frames - 1
                    _, matching = matched_pose_loss(
                        model.decoder.controls(posterior.d),
                        batch["targets"]["object_poses"][:, cutoff],
                        batch["targets"]["object_valid"][:, cutoff],
                        posterior.allocated,
                    )
                    variant_complete = True
                    for frame in frames:
                        if not deadline():
                            variant_complete = False
                            break
                        current = frame == cutoff
                        if not current:
                            state = model.predict(
                                state, batch["targets"]["times"][:, frame] - state.t
                            )
                        frozen = replace(posterior, t=state.t)
                        generator = torch.Generator(device=device).manual_seed(
                            seed + source * 1000 + variant * 31 + frame
                        )
                        targets = sample_ray_targets(
                            batch, frame, rays, generator=generator
                        )
                        prediction, partial = _read_with_deadline(
                            model, state, targets["query"], deadline
                        )
                        if prediction is None:
                            result["incomplete_frame_rays_discarded"] += partial
                            variant_complete = False
                            break
                        baseline = None
                        if not current:
                            baseline, partial = _read_with_deadline(
                                model, frozen, targets["query"], deadline
                            )
                            if baseline is None:
                                result["incomplete_frame_rays_discarded"] += (
                                    partial + targets["query"].origins.shape[1]
                                )
                                variant_complete = False
                                break
                        category = "current" if current else "future"
                        meter = meters.setdefault(category + "/model", MetricTotals())
                        meter.add(prediction, targets)
                        meter.add_pose(
                            fixed_identity_pose_metrics(
                                prediction["controls"],
                                targets["object_poses"],
                                targets["object_valid"],
                                matching,
                            )
                        )
                        if not current:
                            frozen_meter = meters.setdefault(
                                "future/copy_last_state", MetricTotals()
                            )
                            frozen_meter.add(baseline, targets)
                            frozen_meter.add_pose(
                                fixed_identity_pose_metrics(
                                    baseline["controls"],
                                    targets["object_poses"],
                                    targets["object_valid"],
                                    matching,
                                )
                            )
                            copy_rgb, observed = _copy_last_rgb(batch, targets)
                            meters.setdefault(
                                "future/copy_last_rgb_observed", MetricTotals()
                            ).add(copy_rgb, targets, observed)
                            meters.setdefault(
                                "future/model_observed", MetricTotals()
                            ).add(prediction, targets, observed)
                            meters.setdefault(
                                "future/copy_last_state_observed", MetricTotals()
                            ).add(baseline, targets, observed)
                        result["frames_completed"] += 1
                        # One actual rendered current and final-future example per evaluation.
                        if (
                            visuals
                            and source == 0
                            and variant == 0
                            and frame in (cutoff, frames[-1])
                        ):
                            full_targets = (
                                targets
                                if rays is None
                                else sample_ray_targets(batch, frame, None)
                            )
                            full_pred = prediction
                            if rays is not None:
                                full_pred, _ = _read_with_deadline(
                                    model, state, full_targets["query"], deadline
                                )
                            if full_pred is not None:
                                v, _, h, w = batch["targets"]["rgb"].shape[2:]
                                truth = full_targets["rgb"][0].reshape(v, h, w, 3)
                                pred = full_pred["rgb"][0].reshape(v, h, w, 3)
                                # Rows: GT, learned readout; columns: camera views.
                                images[category + "_GT_top_prediction_bottom"] = (
                                    torch.cat(
                                        [
                                            torch.cat(list(truth), 1),
                                            torch.cat(list(pred), 1),
                                        ],
                                        0,
                                    )
                                    .clamp(0, 1)
                                    .cpu()
                                )
                    if not variant_complete:
                        source_complete = False
                        break
                    result["variants_completed"] += 1
                if source_complete:
                    result["sources_completed"] += 1
                else:
                    break
    finally:
        model.train(was_training)
    result["complete"] = result["frames_completed"] == result["frames_expected"]
    result["metrics"] = {name: meter.report() for name, meter in meters.items()}
    result["visuals"] = images
    result["elapsed_seconds"] = time.monotonic() - started
    return result
