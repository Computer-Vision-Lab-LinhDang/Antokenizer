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
        if "status_logits" in prediction:
            known = status >= 0
            pred = prediction["status_logits"].argmax(-1)
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
                  "status_label_coverage": s["labeled_rays"] / max(1, s["all_rays"])}
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
        return result


def horizon_seconds(clip, frame):
    return float(clip['times'][frame] - clip['times'][clip['plan']['prefix'][-1]])


def copy_last(clip, targets, *, frame):
    """Same-view image persistence, restricted to previously observed frames."""
    allowed = [f for f, observed in zip(clip["plan"]["prefix"], clip["plan"]["observed"]) if observed and f < frame]
    view = int(targets.get("view", 0))
    # Only view zero is supplied to the observer by the natural training path.
    if not allowed or view != 0:
        return None, None
    previous = allowed[-1]
    pixels = targets["pixels"].long().cpu()
    x, y = pixels[..., 0], pixels[..., 1]
    rgb = clip["rgb"][previous, view].permute(1, 2, 0)[y, x].to(targets["rgb"].device)
    mask = clip["rgb_valid"][previous, view, 0][y, x].to(targets["rgb_valid"].device)
    return {"rgb": rgb}, {**targets, "rgb_valid": targets["rgb_valid"] & mask}


@torch.no_grad()
def evaluate_suite(model, dataset, suite, *, rays=256):
    from .training import compute_loss
    validate_suite(suite, dataset)
    accumulators = defaultdict(ReadoutMetrics)
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
                    def collect(*, kind, frame, prediction, targets, **unused):
                        horizon = horizon_seconds(clip, frame) if kind == 'future' else 0
                        key = f'{source}/{profile}/{kind}/seconds_{horizon:.3f}'
                        accumulators[key].add(prediction, targets)
                        coverage[key + "/readouts"] += 1
                        if kind == "future":
                            baseline, baseline_targets = copy_last(clip, targets, frame=frame)
                            if baseline is not None:
                                accumulators[key+"/copy_last"].add(baseline, baseline_targets)
                                accumulators[key+"/model_paired_copy_last"].add(prediction, baseline_targets)
                    loss, _ = compute_loss(model, [clip], step=4000, rays=rays, seed=seed,
                                           bridge=False, readout_callback=collect, evaluation_view=0)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite fixed evaluation loss")
                    coverage[f'{source}/{profile}/realized_{clip["plan"]["profile"]}'] += 1
    finally:
        model.train(was_training)
    return {"schema": "v5-evaluation-result-v1", "suite": suite,
            "protocol": "fixed sampled rays; no post-hoc scale fitting; not full-image SSIM/LPIPS",
            "dense_feature_baseline": "not_evaluated_requires_independently_trained_checkpoint",
            "metrics": {key: value.result() for key, value in accumulators.items()},
            "coverage": dict(coverage)}
