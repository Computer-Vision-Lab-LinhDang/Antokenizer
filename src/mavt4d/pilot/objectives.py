"""RGB-only pilot objectives; synthetic class/configuration labels never enter.

The C temporal target is signed pooled RGB differences, divided by per-clip
RMS (with the locked floor). Semantic invariance uses the same support for two
photometric views. Noncollapse is mean ReLU(target-std), plus mean squared
 off-diagonal covariance of the raw semantic projection; both views contribute.
These are inductive biases, not a guarantee of semantic disentanglement.
"""
from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.contracts import CDState, CDStateSpec, DecoderQuery, ObservationStep
from mavt4d.reference.decoder import CTemporalHead
from mavt4d.reference.model import ReferenceCDModel
from .config import PilotConfig
from .data import TrainingBatch


class PilotHeads(nn.Module):
    """Training-only C heads, whose weights are shared initially across cells."""
    def __init__(self, spec: CDStateSpec, cfg: PilotConfig) -> None:
        super().__init__()
        self.temporal = CTemporalHead(spec, len(cfg.temporal_pairs) * 3 * cfg.temporal_pool[0] * cfg.temporal_pool[1])
        self.semantic = nn.Sequential(nn.Flatten(start_dim=1),
                                      nn.Linear(spec.c_tokens * spec.token_dim, cfg.semantic_dim))


def _cpu_generator(generator: torch.Generator) -> None:
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("pilot random choices require an explicit CPU torch.Generator")


def _randint(high: int, generator: torch.Generator) -> int:
    return int(torch.randint(high, (), generator=generator))


def make_training_mask(batch: TrainingBatch, cfg: PilotConfig, generator: torch.Generator,
                       phase: str = "main") -> Tensor:
    """Draw one label/RGB-independent anisotropic mask for an entire source group.

    Wide: H/2 x W/2 rectangle for one or two tubelets. Tube: H/2 x W/4
    rectangle from frame 4 onwards. Snapshot: all pixels in frames [4,8).
    Never-seen: H/2 x W/2 rectangle across all frames. Missing ground truth is
    used only by losses, never by assimilation or query selection.
    """
    _cpu_generator(generator)
    if phase not in ("main", "overfit"):
        raise ValueError("phase: expected main or overfit")
    mask = batch.observed.clone()
    if phase == "overfit":
        return mask
    kind = cfg.mask_kinds[int(torch.multinomial(torch.tensor(cfg.mask_weights), 1, generator=generator))]
    if kind == "clean":
        return mask
    frames, height, width = batch.images.shape[2], batch.images.shape[-2], batch.images.shape[-1]
    if kind == "snapshot":
        mask[:, :, 4:min(8, frames)] = False
        return mask
    rect_h, rect_w = max(1, height // 2), max(1, width // (4 if kind == "tube" else 2))
    top, left = _randint(height - rect_h + 1, generator), _randint(width - rect_w + 1, generator)
    if kind == "wide":
        tubelets = frames // 2
        length = 1 + _randint(min(2, tubelets), generator)
        start = 2 * _randint(tubelets - length + 1, generator)
        end = start + 2 * length
    elif kind == "tube":
        start, end = 4, frames
    else:
        start, end = 0, frames
    mask[:, :, start:end, top:top + rect_h, left:left + rect_w] = False
    return mask


def ordered_history_target(images: Tensor, cfg: PilotConfig) -> Tensor:
    """Signed C target [B, pairs*3*pool_H*pool_W], computed only from clean RGB."""
    if images.ndim != 6 or images.shape[1] != 1 or images.shape[2] != cfg.T:
        raise ValueError("images: expected [B,1,T,3,H,W] matching config")
    differences = torch.stack([images[:, 0, a] - images[:, 0, b]
                               for a, b in cfg.temporal_pairs], dim=1)
    pooled = F.adaptive_avg_pool2d(differences.flatten(0, 1), cfg.temporal_pool)
    flattened = pooled.reshape(images.shape[0], -1)
    rms = flattened.square().mean(dim=1, keepdim=True).sqrt().clamp_min(cfg.temporal_rms_floor)
    return (flattened / rms).detach()


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average over all valid scalar elements, including channels; empty -> zero."""
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    selected = torch.where(expanded, values, torch.zeros_like(values))
    return selected.sum() / expanded.sum().clamp_min(1).to(values.dtype)


def full_observation_states(model: ReferenceCDModel, batch: TrainingBatch,
                            availability: Tensor | None = None,
                            stop_frame: int | None = None) -> tuple[CDState, ...]:
    """Assimilate compact masked tubelets; no state input contains later storage.

    This helper starts at zero memory and retains no cache in the model. Callers
    needing continuation use the reference model's explicit state APIs.
    """
    total = batch.images.shape[2]
    stop = total if stop_frame is None else stop_frame
    if type(stop) is not int or not 1 <= stop <= total:
        raise ValueError("stop_frame must be a positive prefix length")
    available = batch.observed if availability is None else availability
    if available.dtype != torch.bool or available.shape != batch.observed.shape or available.device != batch.images.device:
        raise ValueError("availability must match batch observed shape/device and be boolean")
    available = available & batch.observed
    initial = model.initial_state(batch.images.shape[0], last_time=float(batch.frame_times[:, :, 0].min()))
    steps = []
    for start in range(0, stop, model.spec.frames_per_step):
        end = min(start + model.spec.frames_per_step, stop)
        observations = []
        for index in range(batch.images.shape[0]):
            mask = available[index, :, start:end].clone()
            rgb = batch.images[index, :, start:end]
            observations.append(ObservationStep(
                images=torch.where(mask.unsqueeze(2), rgb, torch.zeros_like(rgb)).contiguous().clone(),
                observed=mask.contiguous().clone(),
                frame_times=batch.frame_times[index, :, start:end].contiguous().clone(),
                intrinsics=batch.intrinsics[index].contiguous().clone(),
                cam_to_world=batch.cam_to_world[index, :, start:end].contiguous().clone(),
            ))
        steps.append(tuple(observations))
    return model.forward_sequence(initial, tuple(steps), method="sequential")


def _query(batch: TrainingBatch, frame: int, indices: Tensor) -> DecoderQuery:
    """One explicit shared camera/time table; verified by TrainingBatch."""
    height, width = batch.images.shape[-2:]
    del height
    device, dtype = batch.images.device, batch.images.dtype
    indices = indices.to(device)
    coordinates = torch.stack(((indices % width).to(dtype) + .5,
                               torch.div(indices, width, rounding_mode="floor").to(dtype) + .5,
                               torch.zeros_like(indices, dtype=dtype)), dim=-1)
    return DecoderQuery(coordinates=coordinates,
                        view_ids=torch.zeros(indices.shape[0], dtype=torch.int64, device=device),
                        times=batch.frame_times[0, 0, frame].expand(indices.shape[0]).clone(),
                        intrinsics=batch.intrinsics[0].clone(),
                        cam_to_world=batch.cam_to_world[0, :, frame].clone(),
                        coordinate_space="image_pixel")


def _pixels(batch: TrainingBatch, frame: int, indices: Tensor) -> Tensor:
    return batch.images[:, 0, frame].flatten(2).transpose(1, 2)[:, indices.to(batch.images.device)]


def _noncollapse(projection: Tensor, variance_target: float) -> Tensor:
    centered = projection - projection.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    # The epsilon is a numerical floor, below the locked target 0.1.
    std_penalty = F.relu(variance_target - torch.sqrt(variance + 1e-4)).mean()
    dimensions = projection.shape[1]
    if dimensions <= 1:
        return std_penalty
    covariance = centered.T @ centered / max(1, projection.shape[0] - 1)
    off_diagonal = ~torch.eye(dimensions, dtype=torch.bool, device=projection.device)
    return std_penalty + covariance[off_diagonal].square().mean()


def training_loss(model: ReferenceCDModel, heads: PilotHeads, batch: TrainingBatch,
                  cfg: PilotConfig, generator: torch.Generator,
                  phase: str = "main") -> tuple[Tensor, dict[str, Tensor]]:
    """Unroll one full source group and return the fixed RGB-only scalar loss.

    Every random choice uses the caller's CPU generator, independent of model
    RNG, initialization, gradients and labels. All three cells therefore receive
    identical masks, query pixels and photometric perturbations at each update.
    Future supervision starts from state after frame 5 by default and rolls out
    without observations to frames 7 and 9. Targets are read only by the loss.
    """
    _cpu_generator(generator)
    if model.spec != cfg.state_spec:
        raise ValueError("model state spec differs from pilot config")
    if batch.images.shape[2:] != (cfg.T, 3, cfg.H, cfg.W):
        raise ValueError("training batch dimensions differ from pilot config")
    availability = make_training_mask(batch, cfg, generator, phase)
    states = full_observation_states(model, batch, availability)
    indices = torch.randperm(cfg.H * cfg.W, generator=generator)[:cfg.query_count]
    observed_sum, missing_sum = batch.images.new_zeros(()), batch.images.new_zeros(())
    observed_count, missing_count = batch.images.new_zeros(()), batch.images.new_zeros(())
    all_sum = batch.images.new_zeros(())
    all_count = 0
    for step, state in enumerate(states):
        frame = min((step + 1) * model.spec.frames_per_step, cfg.T) - 1
        prediction = model.decode(state, _query(batch, frame, indices))
        errors = (prediction - _pixels(batch, frame, indices)).abs()
        observed = availability[:, 0, frame].flatten(1)[:, indices.to(batch.images.device)]
        expanded = observed.unsqueeze(-1).expand_as(errors)
        observed_sum = observed_sum + torch.where(expanded, errors, torch.zeros_like(errors)).sum()
        missing_sum = missing_sum + torch.where(~expanded, errors, torch.zeros_like(errors)).sum()
        observed_count = observed_count + expanded.sum()
        missing_count = missing_count + (~expanded).sum()
        all_sum = all_sum + errors.sum()
        all_count += errors.numel()
    rec = observed_sum / observed_count.clamp_min(1)
    missing = missing_sum / missing_count.clamp_min(1)
    prefix_state = states[cfg.future_prefix_frames // model.spec.frames_per_step - 1]
    future_times = torch.stack([batch.frame_times[:, 0, t] for t in cfg.future_target_frames])
    predicted_states = model.rollout(prefix_state, future_times)
    future = torch.stack([(model.decode(state, _query(batch, frame, indices)) - _pixels(batch, frame, indices)).abs().mean()
                          for state, frame in zip(predicted_states, cfg.future_target_frames)]).mean()
    target = ordered_history_target(batch.images, cfg)
    temporal_prediction = heads.temporal(states[-1].c)
    temporal = F.mse_loss(temporal_prediction, target)
    temporal_cosine = F.cosine_similarity(temporal_prediction, target, dim=1).mean()
    zero_temporal = target.square().mean()
    gain = .9 + .2 * float(torch.rand((), generator=generator))
    bias = -.05 + .1 * float(torch.rand((), generator=generator))
    augmented = replace(batch, images=(batch.images * gain + bias).clamp(-1, 1))
    other_states = full_observation_states(model, augmented, availability)
    projection = heads.semantic(states[-1].c)
    other_projection = heads.semantic(other_states[-1].c)
    invariance = F.mse_loss(F.normalize(projection, dim=1), F.normalize(other_projection, dim=1))
    noncollapse = .5 * (_noncollapse(projection, cfg.variance_target) + _noncollapse(other_projection, cfg.variance_target))
    components = {"rec": rec, "missing": missing, "future": future,
                  "temporal": temporal, "invariance": invariance, "noncollapse": noncollapse,
                  "zero_temporal_mse": zero_temporal,
                  "reconstruction_l1": all_sum / all_count,
                  "temporal_mse": temporal, "temporal_zero_mse": zero_temporal,
                  "temporal_cosine": temporal_cosine}
    total = sum(getattr(cfg, name + "_weight") * components[name]
                for name in ("rec", "missing", "future", "temporal", "invariance", "noncollapse"))
    return total, components
