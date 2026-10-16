"""RGB-only hybrid objectives with equal source-group normalization.

C supervision concatenates both stages. Full images are decoded from aligned
tiles, rather than allocating one independent decoder query per RGB pixel.
Every source retains its own camera metadata. Synthetic core/configuration
labels are absent from this module and never enter its forward interface.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.contracts import ObservationStep
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.state import HybridCDKVState
from mavt4d.pilot.objectives import make_training_mask, ordered_history_target, _noncollapse
from .config import HybridPilotConfig
from .data import TrainingBatch


def slice_state(state: HybridCDKVState, start: int, stop: int) -> HybridCDKVState:
    """Select streams, including every KV/clock field, preserving gradients."""
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(state.c):
        raise ValueError("state slice must select a nonempty valid stream range")
    return replace(state, **{f.name: getattr(state, f.name)[start:stop].contiguous()
                             for f in fields(state) if f.name != "config"})


def full_observation_states(model: HybridCDModel, groups: Sequence[TrainingBatch],
                            availability: Sequence[Tensor] | None = None, *,
                            start_frame: int = 0, stop_frame: int | None = None,
                            method: str = "associative") -> tuple[HybridCDKVState, ...]:
    """Batch physical steps across groups with compact causal masked inputs.

The helper also supports smaller variant batches for frozen evaluation. The
training forward below enforces all eight variants for every source.
"""
    if not isinstance(groups, Sequence) or not groups or not all(isinstance(g, TrainingBatch) for g in groups):
        raise ValueError("groups must be a nonempty sequence of TrainingBatch")
    total = groups[0].images.shape[2]
    stop = total if stop_frame is None else stop_frame
    if type(start_frame) is not int or type(stop) is not int or not 0 <= start_frame < stop <= total:
        raise ValueError("start_frame/stop_frame must select a valid nonempty prefix interval")
    masks = [g.observed for g in groups] if availability is None else availability
    if len(masks) != len(groups):
        raise ValueError("availability must provide one mask per group")
    for group, mask in zip(groups, masks):
        if group.images.shape[2:] != groups[0].images.shape[2:] or group.images.device != groups[0].images.device:
            raise ValueError("all groups must share image shape/device")
        if mask.dtype != torch.bool or mask.shape != group.observed.shape or mask.device != group.images.device:
            raise ValueError("availability must match group observation shape/device")
    initial_times = torch.cat([g.frame_times[:, :, start_frame].min(dim=1).values for g in groups])
    initial = model.initial_state(sum(len(g.images) for g in groups), last_time=initial_times)
    steps = []
    for start in range(start_frame, stop, model.config.frames_per_step):
        end = min(start + model.config.frames_per_step, stop)
        observations = []
        for group, available in zip(groups, masks):
            for index in range(len(group.images)):
                mask = (available[index, :, start:end] & group.observed[index, :, start:end]).contiguous().clone()
                rgb = group.images[index, :, start:end]
                observations.append(ObservationStep(
                    images=torch.where(mask.unsqueeze(2), rgb, torch.zeros_like(rgb)).contiguous().clone(),
                    observed=mask,
                    frame_times=group.frame_times[index, :, start:end].contiguous().clone(),
                    intrinsics=group.intrinsics[index].contiguous().clone(),
                    cam_to_world=group.cam_to_world[index, :, start:end].contiguous().clone(),
                ))
        steps.append(tuple(observations))
    return model.forward_sequence(initial, tuple(steps), method=method)


def render_group(model: HybridCDModel, state: HybridCDKVState,
                 batch: TrainingBatch, frame: int) -> Tensor:
    """Read only C/D, with this group's explicit camera at the target frame."""
    return model.decoder.render(
        state.code(), height=batch.images.shape[-2], width=batch.images.shape[-1],
        time=float(batch.frame_times[0, 0, frame]), intrinsics=batch.intrinsics[0],
        cam_to_world=batch.cam_to_world[0, :, frame],
    )


class HybridPilotModule(nn.Module):
    """DDP-safe owner of the hybrid and both training-only C readout heads."""

    def __init__(self, cfg: HybridPilotConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = HybridCDModel(cfg.architecture)
        c_values = cfg.architecture.stage_count * cfg.architecture.state_tokens * cfg.architecture.state_dim
        target_dim = len(cfg.temporal_pairs) * 3 * cfg.temporal_pool[0] * cfg.temporal_pool[1]
        self.history_head = nn.Linear(c_values, target_dim)
        self.semantic_head = nn.Linear(c_values, cfg.semantic_dim)

    def forward(self, groups: Sequence[TrainingBatch], objective_seeds: Sequence[int],
                phase: str = "main") -> tuple[Tensor, dict[str, Tensor]]:
        # The CLI calls the clean small-set phase "tiny"; v1's mask helper
        # calls the same fixed clean policy "overfit".
        phase = "overfit" if phase == "tiny" else phase
        if not isinstance(groups, Sequence) or not groups:
            raise ValueError("groups must be a nonempty sequence")
        if len(groups) != len(objective_seeds) or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in objective_seeds):
            raise ValueError("one nonnegative objective seed is required per source group")
        cfg = self.cfg
        for group in groups:
            if not isinstance(group, TrainingBatch) or group.images.shape != (8, 1, cfg.T, 3, cfg.H, cfg.W):
                raise ValueError("each source group requires eight variants and configured RGB dimensions")
        generators = [torch.Generator().manual_seed(seed) for seed in objective_seeds]
        availability = [make_training_mask(group, cfg, generator, phase)
                        for group, generator in zip(groups, generators)]
        states = full_observation_states(self.model, groups, availability)
        augmented = []
        for group, generator in zip(groups, generators):
            gain = .9 + .2 * float(torch.rand((), generator=generator))
            bias = -.05 + .1 * float(torch.rand((), generator=generator))
            augmented.append(replace(group, images=(group.images * gain + bias).clamp(-1, 1)))
        other_states = full_observation_states(self.model, augmented, availability)
        prefix = states[cfg.future_prefix_frames // cfg.architecture.frames_per_step - 1]
        future_times = torch.stack([torch.cat([group.frame_times[:, 0, frame] for group in groups])
                                    for frame in cfg.future_target_frames])
        future_states = self.model.rollout(prefix, future_times, method="associative")
        all_components = []
        for group_index, (group, mask) in enumerate(zip(groups, availability)):
            start, stop = group_index * 8, (group_index + 1) * 8
            observed_sum = group.images.new_zeros(())
            missing_sum = group.images.new_zeros(())
            observed_count = group.images.new_zeros(())
            missing_count = group.images.new_zeros(())
            for index, state in enumerate(states):
                frame = (index + 1) * cfg.architecture.frames_per_step - 1
                prediction = render_group(self.model, slice_state(state, start, stop), group, frame)
                errors = (prediction.float() - group.images[:, 0, frame]).abs()
                observed = mask[:, 0, frame, None].expand_as(errors)
                observed_sum = observed_sum + torch.where(observed, errors, 0.).sum()
                missing_sum = missing_sum + torch.where(~observed, errors, 0.).sum()
                observed_count = observed_count + observed.sum()
                missing_count = missing_count + (~observed).sum()
            rec = observed_sum / observed_count.clamp_min(1)
            missing = missing_sum / missing_count.clamp_min(1)
            future = torch.stack([
                (render_group(self.model, slice_state(state, start, stop), group, frame).float()
                 - group.images[:, 0, frame]).abs().mean()
                for state, frame in zip(future_states, cfg.future_target_frames)
            ]).mean()
            c = states[-1].c[start:stop].flatten(1)
            other_c = other_states[-1].c[start:stop].flatten(1)
            target = ordered_history_target(group.images, cfg)
            temporal_prediction = self.history_head(c).float()
            temporal = F.mse_loss(temporal_prediction, target)
            projection = self.semantic_head(c).float()
            other_projection = self.semantic_head(other_c).float()
            invariance = F.mse_loss(F.normalize(projection, dim=1), F.normalize(other_projection, dim=1))
            noncollapse = .5 * (_noncollapse(projection, cfg.variance_target)
                                + _noncollapse(other_projection, cfg.variance_target))
            all_components.append({
                "rec": rec, "missing": missing, "future": future, "temporal": temporal,
                "invariance": invariance, "noncollapse": noncollapse,
                "reconstruction_l1": (observed_sum + missing_sum) / (observed_count + missing_count),
                "temporal_mse": temporal, "temporal_zero_mse": target.square().mean(),
                "zero_temporal_mse": target.square().mean(),
                "temporal_cosine": F.cosine_similarity(temporal_prediction, target, dim=1).mean(),
            })
        components = {name: torch.stack([entry[name] for entry in all_components]).mean()
                      for name in all_components[0]}
        total = sum(getattr(cfg, name + "_weight") * components[name]
                    for name in ("rec", "missing", "future", "temporal", "invariance", "noncollapse"))
        return total, {**{name: value.detach() for name, value in components.items()}, "loss": total.detach()}
