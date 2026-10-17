"""Deadline-safe full-training runtime primitives, without host-specific policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import os
import random
import time
from typing import Callable

import numpy as np
import torch


def utc_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must include UTC offset")
    return parsed.timestamp()


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    train_sources: int = 4096
    dev_sources: int = 64
    height: int = 128
    width: int = 192
    frames: int = 12
    prefix_frames: int = 6
    effective_groups: int = 32
    rays_per_view: int = 128
    workers: int = 12
    lr: float = 0.0002
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    gamma: float = 0.9
    max_updates: int = 100000
    eval_every: int = 200
    save_every_seconds: float = 600
    wandb_online: bool = True
    candidate_warmup_updates: int = 100
    train_until: str = "2026-10-18T00:00:00Z"
    eval_until: str = "2026-10-18T00:45:00Z"
    hard_stop: str = "2026-10-18T00:55:00Z"

    def __post_init__(self):
        if (
            not utc_timestamp(self.train_until)
            < utc_timestamp(self.eval_until)
            < utc_timestamp(self.hard_stop)
        ):
            raise ValueError("deadlines must reserve evaluation and shutdown time")
        for name in (
            "train_sources",
            "dev_sources",
            "height",
            "width",
            "frames",
            "prefix_frames",
            "effective_groups",
            "rays_per_view",
            "max_updates",
            "eval_every",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.prefix_frames >= self.frames
            or self.prefix_frames % 2
            or self.frames % 2
        ):
            raise ValueError("prefix/full lengths must be even with held-out future")
        if self.height % 16 or self.width % 16 or self.workers < 0:
            raise ValueError("invalid image size or worker count")
        if not (0 < self.gamma <= 1 and self.lr > 0 and self.grad_clip > 0):
            raise ValueError("invalid optimization parameters")

    def to_dict(self):
        return asdict(self)


class Deadline:
    def __init__(
        self, config: TrainingConfig, *, clock: Callable[[], float] = time.time
    ):
        self.train = utc_timestamp(config.train_until)
        self.evaluate = utc_timestamp(config.eval_until)
        self.hard = utc_timestamp(config.hard_stop)
        self.clock = clock

    def can_train(self, estimated_seconds=0.0):
        return self.clock() + max(0.0, estimated_seconds) < self.train

    def can_evaluate(self, estimated_seconds=0.0):
        return self.clock() + max(0.0, estimated_seconds) < self.evaluate


def microbatch_slices(total_groups: int, micro_groups: int):
    if total_groups < 1 or micro_groups < 1:
        raise ValueError("group counts must be positive")
    for start in range(0, total_groups, micro_groups):
        end = min(start + micro_groups, total_groups)
        yield start, end, (end - start) / total_groups


def save_checkpoint(path, model, optimizer, *, step, config, extra):
    """Immutable snapshots include optimizer/RNG; no credentials or observations."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    rng = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    payload = {
        "schema": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": config,
        "extra": extra,
        "rng": rng,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def restore_checkpoint(path, model, optimizer, *, expected_config):
    # Only trusted checkpoints produced by this training job are accepted.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != 1 or payload.get("config") != expected_config:
        raise ValueError("checkpoint config/schema mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"]["torch"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    if "cuda" in payload["rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG cannot be restored on CPU")
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return payload


LOSS_WEIGHTS = {
    "rgb": 1.0,
    "depth": 0.1,
    "opacity": 0.1,
    "pose": 0.1,
    "association": 0.1,
    "existence": 0.05,
    "swap": 0.1,
    "proposal_pose": 0.1,
    "proposal_association": 0.1,
}


def tree_to(value, device):
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    return (
        value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
    )


def select_batch(value, indices):
    if isinstance(value, dict):
        return {k: select_batch(v, indices) for k, v in value.items()}
    return value.index_select(0, indices)


def _aligned_swap(posterior, matching, pairs):
    """Loss-only GT correspondence aligns learned slots; inference never uses it."""
    dynamic = torch.zeros_like(posterior.d)
    exists = torch.zeros_like(posterior.p_exist)
    status = torch.zeros_like(posterior.slot_status)
    for receiver, donor in enumerate(pairs["dynamic"].tolist()):
        good = (matching[receiver] >= 0) & (matching[donor] >= 0)
        target_slots, donor_slots = matching[receiver, good], matching[donor, good]
        dynamic[receiver, target_slots] = posterior.d[donor, donor_slots]
        exists[receiver, target_slots] = posterior.p_exist[donor, donor_slots]
        status[receiver, target_slots] = posterior.slot_status[donor, donor_slots]
    return replace(posterior, d=dynamic, p_exist=exists, slot_status=status)


def fixed_pose_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    matching: torch.Tensor,
    allocated: torch.Tensor,
) -> torch.Tensor:
    """Future loss keeps the last-prefix identity assignment, including swaps."""
    selected = matching.clamp_min(0)
    batch = torch.arange(len(predicted), device=predicted.device)[:, None]
    eligible = valid & (matching >= 0) & allocated[batch, selected]
    pose = predicted[batch, selected]
    distance = (pose[..., :3, 3] - target[..., :3, 3]).abs().mean(-1)
    rotation = (pose[..., :3, :3] - target[..., :3, :3]).square().mean((-1, -2))
    error = torch.where(eligible, distance + rotation, 0.0)
    count = eligible.sum(-1)
    per_sample = error.sum(-1) / count.clamp_min(1)
    return per_sample.sum() / (count > 0).sum().clamp_min(1)


def compute_training_loss(model, batch, config: TrainingConfig, *, step: int):
    """Full causal filter + open-loop rollout, with no future encoder call.

    The first two/three observations influence the state, all future frames are
    targets only. Query camera geometry stays FP32 outside spatial autocast.
    """
    from .data import (
        observation_step,
        sample_ray_targets,
        patch_instance_targets,
        paired_variant_indices,
    )
    from .objectives import (
        readout_loss,
        matched_pose_loss,
        association_loss,
        existence_loss,
        weighted_objective,
    )

    batch_size = len(batch["observations"]["rgb"])
    device = batch["observations"]["rgb"].device
    state = model.initial_state(
        batch_size, device=device, time=batch["observations"]["times"][:, 0]
    )
    terms = {k: state.c.sum() * 0 for k in LOSS_WEIGHTS}
    observations = list(range(1, config.prefix_frames, 2))
    futures = list(range(config.prefix_frames + 1, config.frames, 2))
    discounts = [config.gamma ** (i + 1) for i in range(len(futures))]
    current_weight = 0.5 / len(observations)
    future_weights = [0.5 * v / sum(discounts) for v in discounts]
    matches = []

    def score(current, frame, weight, fixed_matching=None):
        rays = sample_ray_targets(batch, frame, config.rays_per_view * 2)
        prediction = model.read(current, rays["query"])
        read_terms = readout_loss(prediction, rays)
        if fixed_matching is None:
            pose, assignment = matched_pose_loss(
                prediction["controls"],
                rays["object_poses"],
                rays["object_valid"],
                current.allocated,
            )
        else:
            assignment = fixed_matching
            pose = fixed_pose_loss(
                prediction["controls"],
                rays["object_poses"],
                rays["object_valid"],
                assignment,
                current.allocated,
            )
        for name, value in read_terms.items():
            terms[name] = terms[name] + weight * value
        terms["pose"] = terms["pose"] + weight * pose
        matches.append(
            ((assignment >= 0) & rays["object_valid"]).float().sum()
            / rays["object_valid"].sum().clamp_min(1)
        )
        return assignment

    for frame in observations:
        observation = observation_step(batch, frame)
        # Only feature extraction uses low precision; state/rays/SE(3) remain FP32.
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            evidence = model.encode_observation(observation)
        state = model.predict(state, evidence.time.double() - state.t)
        state, diagnostics = model.correct(
            state,
            evidence,
            return_diagnostics=True,
            candidate_budget=0 if step < config.candidate_warmup_updates else None,
        )
        matching = score(state, frame, current_weight)
        labels, label_valid = patch_instance_targets(
            batch, frame, patch_size=model.config.patch_size
        )
        terms["association"] = terms["association"] + association_loss(
            diagnostics["ownership"], labels, matching, valid=label_valid
        ) / len(observations)
        terms["existence"] = terms["existence"] + existence_loss(
            state.p_exist, matching
        ) / len(observations)
        proposal_pose, proposal_match = matched_pose_loss(
            diagnostics["proposal_controls"],
            batch["targets"]["object_poses"][:, frame],
            batch["targets"]["object_valid"][:, frame],
            diagnostics["proposal_valid"],
        )
        terms["proposal_pose"] = terms["proposal_pose"] + proposal_pose / len(
            observations
        )
        terms["proposal_association"] = terms[
            "proposal_association"
        ] + association_loss(
            diagnostics["birth_proposals"], labels, proposal_match, valid=label_valid
        ) / len(
            observations
        )
    posterior = state
    for frame, weight in zip(futures, future_weights):
        state = model.predict(state, batch["targets"]["times"][:, frame] - state.t)
        score(state, frame, weight, fixed_matching=matching)

    pairs = paired_variant_indices(batch)
    swapped = _aligned_swap(posterior, matching, pairs)
    target_batch = select_batch(batch, pairs["target"])
    for frame in futures:
        swapped = model.predict(
            swapped, target_batch["targets"]["times"][:, frame] - swapped.t
        )
    swap_targets = sample_ray_targets(
        target_batch, futures[-1], config.rays_per_view * 2
    )
    swap_prediction = model.read(swapped, swap_targets["query"])
    # RGB/geometry swapping supplies an explicit role intervention; no arbitrary
    # latent-vector constancy loss forces newly observed surfaces to be forgotten.
    swap_read = readout_loss(swap_prediction, swap_targets)
    terms["swap"] = (
        swap_read["rgb"] + 0.1 * swap_read["depth"] + 0.1 * swap_read["opacity"]
    )
    total, metrics = weighted_objective(terms, LOSS_WEIGHTS)
    metrics["matched_gt_coverage"] = torch.stack(matches).mean().detach()
    metrics["allocated_slots"] = posterior.allocated.float().sum(-1).mean().detach()
    metrics["existence_mean"] = posterior.p_exist.mean().detach()
    return total, metrics, posterior
