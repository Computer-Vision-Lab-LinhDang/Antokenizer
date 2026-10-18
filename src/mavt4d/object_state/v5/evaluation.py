"""Frozen group-held-out sampled-ray evaluation; no training-set fallback."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random

import torch


def build_suite(dataset, *, groups_per_source=64, seed=17):
    if dataset.split != "val" or groups_per_source < 1:
        raise ValueError("fixed evaluation requires val groups and a positive group count")
    groups = {}
    for source, keys in sorted(dataset.keys.items()):
        keys = sorted(keys)
        rng = random.Random(f"v5-eval:{seed}:{source}")
        rng.shuffle(keys)
        groups[source] = keys[:groups_per_source]
    return {"schema": "v5-fixed-evaluation-v1", "split": "val", "seed": seed,
            "groups": groups, "manifest_hashes": dict(dataset.manifest_hashes),
            "profiles": ["S", "L", "G"], "horizon": 6,
            "requested_groups_per_source": groups_per_source}


def validate_suite(suite, dataset):
    if suite.get("schema") != "v5-fixed-evaluation-v1" or suite.get("split") != "val" or dataset.split != "val":
        raise ValueError("evaluation schema/split mismatch")
    if suite["manifest_hashes"] != dataset.manifest_hashes or set(suite["groups"]) != set(dataset.keys):
        raise ValueError("evaluation inventory changed")
    for source, groups in suite["groups"].items():
        if not groups or len(set(groups)) != len(groups) or not set(groups) <= set(dataset.keys[source]):
            raise ValueError("invalid frozen evaluation groups")


class ReadoutMetrics:
    """Additive sufficient statistics; depth is reported only with valid labels."""
    def __init__(self):
        self.sums = defaultdict(float)
        self.confusion = torch.zeros(3, 3, dtype=torch.long)
        self.status_probability, self.status_truth = [], []

    def add(self, prediction, targets):
        rgb, gt = prediction["rgb"].detach(), targets["rgb"]
        status = targets.get("status", torch.full_like(targets["rgb_valid"], -1, dtype=torch.long))
        valid = targets["rgb_valid"] & (status != 2)
        if not torch.isfinite(rgb[valid]).all():
            raise FloatingPointError("nonfinite RGB prediction on evaluated rays")
        error = rgb - gt
        self.sums["rgb_abs"] += float(error.abs().mean(-1)[valid].sum())
        self.sums["rgb_sq"] += float(error.square().mean(-1)[valid].sum())
        self.sums["rgb_rays"] += int(valid.sum())
        self.sums["all_rays"] += status.numel()
        self.sums["labeled_rays"] += int((status >= 0).sum())
        self.sums["missing_status"] += int((status < 0).sum())
        if "status_logits" in prediction:
            known = status >= 0
            logits = prediction["status_logits"].detach()
            if not torch.isfinite(logits[known]).all():
                raise FloatingPointError("nonfinite status prediction on evaluated rays")
            probability = logits.float().softmax(-1)
            self.status_probability.append(probability[known].cpu())
            self.status_truth.append(status[known].cpu())
            pred = logits.argmax(-1)
            self.confusion += torch.bincount((status[known] * 3 + pred[known]).cpu(), minlength=9).reshape(3, 3)
        if "range" in prediction and "range" in targets:
            mask = targets["depth_valid"] & (targets["range"] > 0)
            p, y = prediction["range"].detach()[mask], targets["range"][mask]
            if not torch.isfinite(p).all():
                raise FloatingPointError("nonfinite depth prediction on evaluated rays")
            self.sums["depth_rays"] += len(p)
            self.sums["depth_absrel_sum"] += float(((p-y).abs()/y).sum())
            self.sums["depth_sq"] += float((p-y).square().sum())

    def result(self):
        s = self.sums
        result = {"rgb_rays": int(s["rgb_rays"]), "depth_rays": int(s["depth_rays"]),
                  "status_label_coverage": s["labeled_rays"] / max(1, s["all_rays"]),
                  "status_missing_count": int(s["missing_status"])}
        if s["rgb_rays"]:
            mse = s["rgb_sq"] / s["rgb_rays"]
            result.update(rgb_mae=s["rgb_abs"]/s["rgb_rays"], rgb_mse=mse,
                          rgb_psnr=-10*math.log10(max(mse, 1e-12)))
        if s["depth_rays"]:
            result.update(depth_absrel=s["depth_absrel_sum"]/s["depth_rays"],
                          depth_rmse=math.sqrt(s["depth_sq"]/s["depth_rays"]))
        for truth in range(3):
            n = int(self.confusion[truth].sum())
            result[f"status_{truth}_count"] = n
            if n:
                result[f"status_{truth}_recall"] = int(self.confusion[truth, truth])/n
        if self.status_truth and sum(x.numel() for x in self.status_truth):
            result.update(status_calibration(torch.cat(self.status_probability), torch.cat(self.status_truth)))
        return result


def _binary_auc(score, positive):
    """Mann–Whitney AUROC with exact tie credit; undefined classes stay absent."""
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return None
    values, order = score.sort()
    _, inverse, counts = torch.unique_consecutive(values, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).double()
    average_rank = ends - (counts.double() - 1) / 2
    ranks = average_rank[inverse]
    return float((ranks[positive[order]].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def status_calibration(probabilities, truth, *, bins=15):
    if bins < 1 or probabilities.shape != (len(truth), 3) or not len(truth):
        raise ValueError("nonempty three-class probabilities and positive bins required")
    confidence, prediction = probabilities.max(-1)
    correct = prediction == truth
    bucket = (confidence * bins).long().clamp_max(bins - 1)
    ece = 0.
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs())
    result = {"status_ece": ece, "status_ece_bins": bins,
              "status_risk_coverage_count": len(truth)}
    for label in range(3):
        class_confidence = probabilities[:, label]
        class_correct = truth == label
        class_bucket = (class_confidence * bins).long().clamp_max(bins - 1)
        class_ece = 0.
        for index in range(bins):
            mask = class_bucket == index
            if mask.any():
                class_ece += float(mask.float().mean() * (class_confidence[mask].mean() - class_correct[mask].float().mean()).abs())
        result[f"status_{label}_ece"] = class_ece
        auc = _binary_auc(probabilities[:, label], truth == label)
        if auc is not None:
            result[f"status_{label}_auroc"] = auc
    unknown = truth == 2
    result["status_unknown_count"] = int(unknown.sum())
    if unknown.any():
        result["status_unknown_hallucination_rate"] = float((probabilities[unknown, 2] < .3).float().mean())
    pair = (truth == 1) | unknown
    pair_auc = _binary_auc(probabilities[pair, 2] / probabilities[pair, 1:].sum(-1).clamp_min(1e-12), unknown[pair])
    if pair_auc is not None:
        result["status_unknown_vs_inferred_auroc"] = pair_auc
    # Equal-confidence predictions enter together, so curve does not depend on
    # arbitrary input ordering. AURC uses right-endpoint rectangular integration.
    confidence, order = confidence.sort(descending=True)
    errors = (~correct[order]).double().cumsum(0)
    _, counts = torch.unique_consecutive(confidence, return_counts=True)
    ends = counts.cumsum(0)
    risk = errors[ends - 1] / ends
    result["status_aurc"] = float((risk * counts / len(truth)).sum())
    positions = torch.searchsorted(ends, torch.linspace(1, len(truth), min(21, len(truth))).long()).unique()
    result["status_risk_coverage"] = [{"coverage": float(ends[i] / len(truth)), "risk": float(risk[i])} for i in positions]
    return result


def group_bootstrap(group_values, *, seed=17, samples=1000):
    """Resample independent source groups, never individual correlated rays."""
    values = [float(group_values[k]) for k in sorted(group_values)]
    if not values or samples < 1 or not all(math.isfinite(v) for v in values):
        raise ValueError("finite nonempty group metrics and positive bootstrap count required")
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return {"mean": sum(values) / len(values), "groups": len(values), "samples": samples,
            "ci95": [draws[int(.025 * (samples - 1))], draws[int(.975 * (samples - 1))]]}


def build_gap_protocol(source_ids, *, seed=17):
    """Freeze the required diagnostic roster; does not claim it has been rendered."""
    sources = sorted(set(source_ids))
    if len(sources) < 200:
        raise ValueError("gap diagnostic requires at least 200 independent source groups")
    random.Random(seed).shuffle(sources)
    return {"schema": "v5-gap-protocol-v1", "sources": sources, "seed": seed,
            "steps": 24, "cameras": 2, "heldout_camera": 1, "gaps": [2, 4, 8, 16],
            "crossing_sources_required": math.ceil(.25 * len(sources)),
            "dev_seconds": 60, "metrics": ["reappearance_rgb", "reappearance_pose", "identity", "status"],
            "status": "protocol_only_requires_rendered_fixture_and_evaluation"}


def baseline_protocol():
    """Lock comparison requirements without presenting untrained arms as evidence."""
    return {"schema": "v5-baseline-protocol-v1", "shared_input_masks": True,
            "shared_group_split": True, "fixed_updates": 20000, "effective_batch_min": 256,
            "state_byte_budget_equal": True, "copy_last": "implemented_admitted_history",
            "trained_arms": [{"name": name, "status": "requires_independent_training"} for name in
                             ("dense_feature", "recurrent_unsplit", "transition_v4", "no_phi", "one_pass")]}


def horizon_seconds(clip, frame):
    return float(clip['times'][frame] - clip['times'][clip['plan']['prefix'][-1]])


def copy_last(clip, targets, *, frame, history=None):
    """Per-pixel persistence using exactly the RGB admitted into model history.

    The optional legacy path applies only when the caller supplied no masking
    history. All evaluation entry points supply actual history explicitly.
    """
    view = int(targets.get("view", 0))
    if view != 0:
        return None, None
    if history is None:
        history = [{"frame": f, "input_mask": clip["rgb_valid"][f:f+1]} for f, observed in
                   zip(clip["plan"]["prefix"], clip["plan"]["observed"]) if observed]
    allowed = sorted((r for r in history if r["frame"] < frame), key=lambda r: r["frame"])
    if not allowed:
        return None, None
    pixels = targets["pixels"].long().cpu()
    x, y = pixels[..., 0], pixels[..., 1]
    rgb = torch.zeros_like(targets["rgb"])
    admitted = torch.zeros_like(targets["rgb_valid"])
    for record in allowed:
        previous = record["frame"]
        mask = record["input_mask"][0, view, 0].cpu()[y, x].to(admitted.device)
        # Index only admitted pixels; a hidden RGB value is never read/copied.
        if mask.any():
            selected = mask.cpu()
            color = clip["rgb"][previous, view].permute(1, 2, 0)[y[selected], x[selected]].to(rgb.device)
            rgb[mask] = color
            admitted |= mask
    return {"rgb": rgb}, {**targets, "rgb_valid": targets["rgb_valid"] & admitted}


@torch.no_grad()
def evaluate_suite(model, dataset, suite, *, rays=256):
    from .training import compute_loss
    validate_suite(suite, dataset)
    accumulators = defaultdict(ReadoutMetrics)
    per_group = defaultdict(ReadoutMetrics)
    coverage = defaultdict(int)
    was_training = model.training
    model.eval()
    try:
        for source, groups in suite["groups"].items():
            for group in groups:
                digest = hashlib.sha256(f'{suite["seed"]}:{source}:{group}'.encode()).digest()
                seed = int.from_bytes(digest[:4], "big")
                for profile in (["S"] if source == "openimages" else suite["profiles"]):
                    clip = dataset.draw(source, seed, group=group, profile=profile,
                                        horizon=suite["horizon"], plan_seed=seed)
                    def collect(*, kind, frame, prediction, targets, history, **unused):
                        horizon = horizon_seconds(clip, frame) if kind == 'future' else 0
                        key = f'{source}/{profile}/{kind}/seconds_{horizon:.3f}'
                        accumulators[key].add(prediction, targets)
                        per_group[(key, group)].add(prediction, targets)
                        coverage[key + "/readouts"] += 1
                        if kind == "future":
                            baseline, baseline_targets = copy_last(clip, targets, frame=frame, history=history)
                            if baseline is not None:
                                accumulators[key+"/copy_last"].add(baseline, baseline_targets)
                                per_group[(key+"/copy_last", group)].add(baseline, baseline_targets)
                                accumulators[key+"/model_paired_copy_last"].add(prediction, baseline_targets)
                                per_group[(key+"/model_paired_copy_last", group)].add(prediction, baseline_targets)
                    loss, _ = compute_loss(model, [clip], step=4000, rays=rays, seed=seed,
                                           bridge=False, readout_callback=collect, evaluation_view=0)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite fixed evaluation loss")
                    coverage[f'{source}/{profile}/realized_{clip["plan"]["profile"]}'] += 1
    finally:
        model.train(was_training)
    grouped = defaultdict(dict)
    for (key, group), meter in per_group.items():
        for metric, value in meter.result().items():
            if metric in ("rgb_mae", "depth_absrel", "status_ece", "status_unknown_hallucination_rate"):
                grouped[(key, metric)][group] = value
    intervals = {f"{key}/{metric}": group_bootstrap(values, seed=suite["seed"]) for
                 (key, metric), values in grouped.items() if len(values) >= 2}
    return {"schema": "v5-evaluation-result-v1", "suite": suite,
            "protocol": "fixed sampled rays; no post-hoc scale fitting; not full-image SSIM/LPIPS",
            "dense_feature_baseline": "not_evaluated_requires_independently_trained_checkpoint",
            "baselines": baseline_protocol(),
            "group_bootstrap": {"unit": "independent source group", "weighting": "equal group means", "intervals": intervals},
            "metrics": {key: value.result() for key, value in accumulators.items()},
            "coverage": dict(coverage)}
