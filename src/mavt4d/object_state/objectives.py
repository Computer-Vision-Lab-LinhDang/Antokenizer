"""Observed and predicted readout losses; assignment is supervision only."""
from __future__ import annotations

import math
import torch
from torch import Tensor


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Average valid elements per sample, then samples with available labels.

    Invalid NaN padding is removed with where *before* reduction. An entirely
    missing term is differentiable zero, never 0/0.
    """
    mask = torch.broadcast_to(valid.bool(), values.shape)
    if values.ndim < 1:
        raise ValueError("masked_mean needs a batch dimension")
    clean = torch.where(mask, values, torch.zeros_like(values))
    count = mask.reshape(values.shape[0], -1).sum(-1)
    summed = clean.reshape(values.shape[0], -1).sum(-1)
    means = summed / count.clamp_min(1)
    return means.sum() / (count > 0).sum().clamp_min(1)


def region_balanced_rgb_loss(prediction: Tensor, target: Tensor, instance: Tensor,
                             valid: Tensor | None = None) -> Tensor:
    if prediction.shape != target.shape or prediction.shape[:-1] != instance.shape or prediction.shape[-1] != 3:
        raise ValueError("RGB must be [B,...,3] with matching instance labels")
    mask = torch.ones_like(instance, dtype=torch.bool) if valid is None else valid.bool()
    # Sanitizing before subtraction also keeps masked NaN gradients finite.
    p = torch.where(mask[..., None], prediction, 0.)
    t = torch.where(mask[..., None], target, 0.)
    error = (p - t).abs().mean(-1)
    per_region, available = [], []
    for region in (instance > 0, instance <= 0):
        region = region & mask
        count = region.flatten(1).sum(-1)
        value = torch.where(region, error, 0.).flatten(1).sum(-1) / count.clamp_min(1)
        per_region.append(value)
        available.append(count > 0)
    count = torch.stack(available).sum(0)
    per_sample = torch.stack(per_region).sum(0) / count.clamp_min(1)
    return per_sample.sum() / (count > 0).sum().clamp_min(1)


def readout_loss(prediction: dict, targets: dict) -> dict[str, Tensor]:
    """No coefficients hidden in helpers; depth is absolute metric range L1."""
    valid = targets.get("valid", torch.ones_like(targets["depth_valid"]))
    depth_valid = valid & targets["depth_valid"]
    pred_range = torch.where(depth_valid, prediction["range"], 0.)
    true_range = torch.where(depth_valid, targets["range"], 0.)
    opacity_target = targets["depth_valid"].to(prediction["opacity"].dtype)
    return {
        "rgb": region_balanced_rgb_loss(prediction["rgb"], targets["rgb"], targets["instance"], valid),
        "depth": masked_mean((pred_range - true_range).abs(), depth_valid),
        "opacity": masked_mean((prediction["opacity"] - opacity_target).square(), valid),
    }


def _linear_assignment(cost: Tensor):
    """Exact rectangular Hungarian matching on detached CPU costs (rows<=cols)."""
    rows, cols = cost.shape
    if rows > cols:
        raise ValueError("more labeled objects than eligible predicted slots")
    a = cost.detach().double().cpu().tolist()
    u, v, p, way = [0.] * (rows + 1), [0.] * (cols + 1), [0] * (cols + 1), [0] * (cols + 1)
    for i in range(1, rows + 1):
        p[0], j0 = i, 0
        minimum, used = [math.inf] * (cols + 1), [False] * (cols + 1)
        while True:
            used[j0], i0, delta, j1 = True, p[j0], math.inf, 0
            for j in range(1, cols + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
            if j0 == 0:
                break
    assignment = [-1] * rows
    for j in range(1, cols + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def matched_pose_loss(predicted: Tensor, target: Tensor, valid: Tensor,
                      pred_valid: Tensor | None = None, *, rotation_weight: float = 1.):
    """Loss-side set matching; returns GT-object -> predicted-slot indices.

    Insufficient predicted slots are matched to a subset and unmatched GT remain
    -1; report that coverage separately. Matching uses translation plus rotation
    matrix chordal error, avoiding acos singular gradients at exact rotations.
    This loss assumes the synthetic texture resolves box orientation symmetry.
    """
    if predicted.ndim != 4 or target.ndim != 4 or predicted.shape[-2:] != (4, 4) or target.shape[-2:] != (4, 4):
        raise ValueError("poses must be [B,K,4,4]")
    if predicted.shape[0] != target.shape[0] or valid.shape != target.shape[:2]:
        raise ValueError("pose batch/validity mismatch")
    if not math.isfinite(rotation_weight) or rotation_weight < 0:
        raise ValueError("rotation weight must be finite and nonnegative")
    if pred_valid is None:
        pred_valid = torch.ones(predicted.shape[:2], dtype=torch.bool, device=predicted.device)
    if pred_valid.shape != predicted.shape[:2]:
        raise ValueError("predicted pose validity mismatch")
    matching = torch.full(target.shape[:2], -1, dtype=torch.int64, device=predicted.device)
    losses = []
    for b in range(len(predicted)):
        gi, pi = valid[b].nonzero().flatten(), pred_valid[b].nonzero().flatten()
        if not len(gi) or not len(pi):
            continue
        truth, pred = target[b, gi], predicted[b, pi]
        distance = (truth[:, None, :3, 3] - pred[None, :, :3, 3]).abs().mean(-1)
        rotation = (truth[:, None, :3, :3] - pred[None, :, :3, :3]).square().mean((-1, -2))
        cost = distance + rotation_weight * rotation
        if not torch.isfinite(cost).all():
            raise ValueError("valid pose matching costs must be finite")
        if len(gi) <= len(pi):
            picked = torch.tensor(_linear_assignment(cost), device=predicted.device)
            selected_g = torch.arange(len(gi), device=predicted.device)
        else:
            selected_g = torch.tensor(_linear_assignment(cost.T), device=predicted.device)
            picked = torch.arange(len(pi), device=predicted.device)
        matching[b, gi[selected_g]] = pi[picked]
        losses.append(cost[selected_g, picked].mean())
    zero = torch.where(torch.isfinite(predicted), predicted, 0.).sum() * 0
    return (torch.stack(losses).mean() if losses else zero), matching


def weighted_objective(terms: dict[str, Tensor], weights: dict[str, float]):
    """Apply the caller's frozen weights; never silently omit an unknown term."""
    if set(terms) != set(weights) or not terms:
        raise ValueError("loss weights must exactly match nonempty term names")
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("loss weights must be finite and nonnegative")
    if any(v.ndim != 0 for v in terms.values()):
        raise ValueError("objective terms must be scalar")
    return sum(terms[k] * weights[k] for k in terms), {k: v.detach() for k, v in terms.items()}


def association_loss(ownership: Tensor, labels: Tensor, matching: Tensor,
                     valid: Tensor | None = None) -> Tensor:
    """Balanced probability CE with background last; unmatched GT is excluded.

    GT IDs are 1-based; zero ground and negative sky map to background. This
    function produces a loss only and never changes a state or observation.
    """
    if ownership.ndim != 3 or labels.shape != ownership.shape[:2] or matching.shape[0] != ownership.shape[0]:
        raise ValueError("ownership/instance/matching shape mismatch")
    slots = ownership.shape[-1] - 1
    if ((matching < -1) | (matching >= slots)).any():
        raise ValueError("matching references an invalid slot")
    mask = torch.ones_like(labels, dtype=torch.bool) if valid is None else valid.bool()
    target = torch.full_like(labels, slots)
    foreground = labels > 0
    in_bounds = foreground & (labels <= matching.shape[1])
    if matching.shape[1]:
        slot = matching.gather(1, (labels - 1).clamp(0, matching.shape[1] - 1))
        target = torch.where(in_bounds, slot, target)
    mask = mask & (~foreground | (in_bounds & (target >= 0)))
    probability = ownership.gather(-1, target.clamp_min(0)[..., None]).squeeze(-1)
    error = -torch.where(mask, probability, torch.ones_like(probability)).clamp_min(1e-8).log()
    # As in RGB, give each available foreground/background region equal weight
    # within a sample, so small objects are not overwhelmed by background.
    region_losses, region_present = [], []
    for region in (foreground, ~foreground):
        use = region & mask
        count = use.sum(-1)
        region_losses.append(torch.where(use, error, 0.).sum(-1) / count.clamp_min(1))
        region_present.append(count > 0)
    count = torch.stack(region_present).sum(0)
    per_sample = torch.stack(region_losses).sum(0) / count.clamp_min(1)
    return per_sample.sum() / (count > 0).sum().clamp_min(1)


def existence_loss(probability: Tensor, matching: Tensor, valid: Tensor | None = None) -> Tensor:
    """One existence target per matched slot; unmatched duplicate slots are zero."""
    if probability.ndim != 2 or matching.ndim != 2 or matching.shape[0] != probability.shape[0]:
        raise ValueError("existence/matching shape mismatch")
    if ((matching < -1) | (matching >= probability.shape[1])).any():
        raise ValueError("matching references an invalid slot")
    target = torch.zeros_like(probability)
    batch, obj = (matching >= 0).nonzero(as_tuple=True)
    target[batch, matching[batch, obj]] = 1
    mask = torch.ones_like(probability, dtype=torch.bool) if valid is None else valid.bool()
    p = torch.where(mask, probability, torch.full_like(probability, .5)).clamp(1e-6, 1 - 1e-6)
    return masked_mean(-(target * p.log() + (1 - target) * torch.log1p(-p)), mask)
