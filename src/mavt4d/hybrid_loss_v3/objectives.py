"""Structured synthetic loss v3; targets/labels never enter model inputs.

The model and its persistent state are exactly the existing hybrid. This module
adds no training-only heads. Pair interventions decode once and never resume a
swapped state. Every objective is reduced within a source before averaging
sources, including zero contributions from sources with no eligible pair.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid_pilot.config import HybridPilotConfig
from mavt4d.hybrid_pilot.data import TrainingBatch
from mavt4d.hybrid_pilot.objectives import (
    full_observation_states,
    render_group,
    slice_state,
)
from mavt4d.pilot.objectives import make_training_mask


LABEL_SET = frozenset((core, config) for core in range(4) for config in range(2))
LOSS_WEIGHTS = {
    "rec": 1.0,
    "missing": 1.0,
    "future": 0.5,
    "order": 1.0,
    "c_pair": 0.1,
    "swap": 1.0,
    "c_var": 0.05,
}
ENDPOINTS = (1, 3, 5, 7, 9, 11)
PIXEL_THRESHOLD = 1e-5
MIN_WITNESS_PIXELS = 4
MIN_WITNESS_FRAMES = 2
C_PAIR_SCALE = 0.1
C_STD_FLOOR = 0.1
ORDER_MARGIN = 0.2


def validate_labels(
    labels: Sequence[tuple[int, int]], *, device: torch.device | str | None = None
) -> Tensor:
    """Return core/config -> stream indices, rejecting duplicates or missing cells."""
    if not isinstance(labels, Sequence) or len(labels) != 8:
        raise ValueError(
            "labels must contain four cores crossed with two configurations"
        )
    normalized = []
    for label in labels:
        if (
            not isinstance(label, (tuple, list))
            or len(label) != 2
            or any(type(value) is not int for value in label)
        ):
            raise ValueError(
                "labels must contain four cores crossed with two configurations"
            )
        normalized.append(tuple(label))
    if set(normalized) != LABEL_SET:
        raise ValueError(
            "labels must contain four cores crossed with two configurations"
        )
    indices = {label: index for index, label in enumerate(normalized)}
    return torch.tensor(
        [
            [indices[(core, configuration)] for configuration in range(2)]
            for core in range(4)
        ],
        dtype=torch.long,
        device=device,
    )


@torch.no_grad()
def region_support(group: TrainingBatch, labels: Sequence[tuple[int, int]]) -> Tensor:
    """Detached source-wide [H,W] temporal/configuration union; loss weighting only."""
    lookup = validate_labels(labels, device=group.images.device)
    if len(group.images) != 8:
        raise ValueError("region support requires all eight source variants")
    rgb = group.images.detach()[:, 0]
    temporal = (rgb - rgb[:, :1]).abs().amax(dim=2) > PIXEL_THRESHOLD
    configuration = (rgb[lookup[:, 0]] - rgb[lookup[:, 1]]).abs().amax(
        dim=2
    ) > PIXEL_THRESHOLD
    return temporal.any(dim=(0, 1)) | configuration.any(dim=(0, 1))


def _mean_valid(values: Tensor, valid: Tensor) -> Tensor:
    """Masked scalar mean with a graph-preserving finite zero for the empty set."""
    return torch.where(valid, values, 0.0).sum() / valid.sum().clamp_min(1)


def region_balanced_l1(
    prediction: Tensor, target: Tensor, valid: Tensor, support: Tensor
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Per-variant balanced RGB error and validity, with unweighted region errors.

    RGB has shape [N,3,H,W], valid is boolean [N,H,W], and support is boolean
    [H,W] (or [N,H,W]). Empty regions produce zero and an explicit validity flag;
    they never dilute the mean of nonempty regions. Targets and masks detach.
    """
    if (
        prediction.ndim != 4
        or prediction.shape != target.shape
        or prediction.shape[1] != 3
    ):
        raise ValueError("prediction and target must have matching [N,3,H,W] shapes")
    if (
        valid.dtype != torch.bool
        or valid.shape != prediction.shape[:1] + prediction.shape[-2:]
    ):
        raise ValueError("valid must be boolean [N,H,W]")
    if support.dtype != torch.bool or support.shape not in (
        prediction.shape[-2:],
        valid.shape,
    ):
        raise ValueError("support must be boolean [H,W] or [N,H,W]")
    error = (prediction.float() - target.detach().float()).abs().mean(dim=1)
    valid = valid.detach()
    support = support.detach()
    parts = {}
    for name, region in (
        ("support", valid & support),
        ("background", valid & ~support),
        ("whole", valid),
    ):
        count = region.sum(dim=(-2, -1))
        parts[f"{name}_l1"] = torch.where(region, error, 0.0).sum(
            dim=(-2, -1)
        ) / count.clamp_min(1)
        parts[f"{name}_valid"] = count > 0
    number_regions = parts["support_valid"].int() + parts["background_valid"].int()
    balanced = (
        parts["support_l1"] + parts["background_l1"]
    ) / number_regions.clamp_min(1)
    return balanced, number_regions > 0, parts


@dataclass(frozen=True)
class PairEligibility:
    """All tensors are detached loss-side eligibility, never model observations."""

    config_partner: Tensor  # [8], stream permutation
    prefix_observed: Tensor  # [8], entire prefix observed
    pair_valid: Tensor  # [8], both configurations have complete observed prefix
    negative_witness: (
        Tensor  # [4,4], complete prefixes + two-frame witness in BOTH configs
    )
    swap_valid: Tensor  # [8], pair_valid + distinct endpoint targets


@torch.no_grad()
def pair_eligibility(
    group: TrainingBatch,
    availability: Tensor,
    labels: Sequence[tuple[int, int]],
    frame: int,
) -> PairEligibility:
    """Build same-source pairs from observed prefix pixels, with no future witness."""
    lookup = validate_labels(labels, device=group.images.device)
    if type(frame) is not int or not 0 <= frame < group.images.shape[2]:
        raise ValueError("frame must index the available clip")
    if (
        availability.dtype != torch.bool
        or availability.shape != group.observed.shape
        or availability.device != group.images.device
        or len(group.images) != 8
    ):
        raise ValueError("availability must match all eight source variants")
    core = torch.tensor([label[0] for label in labels], device=lookup.device)
    configuration = torch.tensor([label[1] for label in labels], device=lookup.device)
    partner = lookup[core, 1 - configuration]
    observed = availability[:, :, : frame + 1] & group.observed[:, :, : frame + 1]
    complete = observed.flatten(1).all(dim=1)
    pair_valid = complete & complete[partner]
    # Use each configuration independently: a visible difference in only one
    # configuration cannot qualify an invariance/order relation for both.
    rgb = group.images.detach()[:, 0, : frame + 1]
    witness = torch.zeros(4, 4, dtype=torch.bool, device=lookup.device)
    for first in range(4):
        for second in range(first + 1, 4):
            difference = (rgb[lookup[first]] - rgb[lookup[second]]).abs().amax(dim=2)
            frame_pixels = (difference > PIXEL_THRESHOLD).sum(dim=(-2, -1))
            enough_frames = (frame_pixels >= MIN_WITNESS_PIXELS).sum(
                dim=-1
            ) >= MIN_WITNESS_FRAMES
            qualified = (
                enough_frames.all()
                & complete[lookup[first]].all()
                & complete[lookup[second]].all()
            )
            witness[first, second] = qualified
            witness[second, first] = qualified
    endpoint_difference = (rgb[:, frame] - rgb[partner, frame]).abs().amax(dim=1)
    distinct_target = (endpoint_difference > PIXEL_THRESHOLD).sum(
        dim=(-2, -1)
    ) >= MIN_WITNESS_PIXELS
    return PairEligibility(
        partner, complete, pair_valid, witness, pair_valid & distinct_target
    )


def c_objectives(
    c: Tensor, labels: Sequence[tuple[int, int]], eligibility: PairEligibility
) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    """C-bank losses for one source/endpoint, as (mean, valid, valid-count)."""
    lookup = validate_labels(labels, device=c.device)
    if c.shape[0] != 8:
        raise ValueError("C objectives require eight variants")
    c = c.float().flatten(start_dim=1)
    pair_values = (c[lookup[:, 0]] - c[lookup[:, 1]]).square().mean(
        dim=1
    ) / C_PAIR_SCALE**2
    pair_valid = eligibility.pair_valid[lookup[:, 0]]
    paired = _mean_valid(pair_values, pair_valid)

    u = F.normalize(c, dim=1, eps=1e-8)
    distance = (1 - u @ u.T) / 2
    positive_distance = distance[
        torch.arange(8, device=c.device), eligibility.config_partner
    ]
    triplet = F.relu(ORDER_MARGIN + positive_distance[:, None] - distance)
    core = torch.tensor([label[0] for label in labels], device=c.device)
    configuration = torch.tensor([label[1] for label in labels], device=c.device)
    witnessed = eligibility.negative_witness[core[:, None], core[None]]
    same_config = configuration[:, None] == configuration[None]
    reversal = witnessed & same_config & (core[:, None].bitwise_xor(1) == core[None])
    other_family = witnessed & same_config & (core[:, None] // 2 != core[None] // 2)
    reversal_count = reversal.sum(dim=1)
    other_count = other_family.sum(dim=1)
    category_count = (reversal_count > 0).int() + (other_count > 0).int()
    weights = (
        reversal / reversal_count[:, None].clamp_min(1)
        + other_family / other_count[:, None].clamp_min(1)
    ) / category_count[:, None].clamp_min(1)
    anchor_valid = category_count > 0
    order = _mean_valid((triplet * weights).sum(dim=1), anchor_valid)

    core_indices = torch.arange(4, device=c.device)
    reversal_qualified = eligibility.negative_witness[
        core_indices, core_indices.bitwise_xor(1)
    ]
    other_qualified = (
        eligibility.negative_witness
        & (core_indices[:, None] // 2 != core_indices[None] // 2)
    ).any(dim=1)
    variance_valid = (reversal_qualified & other_qualified).all()
    representatives = c[lookup].mean(dim=1)
    standard_deviation = (representatives.var(dim=0, unbiased=False) + 1e-8).sqrt()
    variance_penalty = F.relu(C_STD_FLOOR - standard_deviation).mean() / C_STD_FLOOR
    variance = torch.where(variance_valid, variance_penalty, c.sum() * 0)
    return {
        "c_pair": (paired, pair_valid.any(), pair_valid.sum()),
        "order": (order, anchor_valid.any(), anchor_valid.sum()),
        "c_var": (variance, variance_valid, variance_valid.long()),
    }


def core_head_cross_entropy(head: nn.Linear, c: Tensor, cores: Tensor) -> Tensor:
    """Per-clip CE of the labelled upper-bound head on flattened final C.

    Unlike the relative C objectives, its gradient W^T(p - y) stays nonzero and
    differs between variants even when every C row is identical.
    """
    return F.cross_entropy(head(c.float().flatten(start_dim=1)), cores, reduction="none")


def split_core_accuracy(totals: dict) -> dict:
    """Add clean/masked core accuracy to update-level totals, 0 when absent."""
    ratios = {}
    for name in ("clean", "masked"):
        key = f"core_accuracy_{name}_sum"
        if key in totals:
            fraction = totals[f"active_source_fraction/core_{name}"]
            ratios[f"core_accuracy_{name}"] = totals[key] / fraction if fraction > 0 else 0.0
    return {**totals, **ratios}


def variant_contrast(x: Tensor) -> Tensor:
    """||X - mean over variants|| / ||X|| for one source's variants on dim 0."""
    x = x.detach().double().flatten(start_dim=1)
    return (x - x.mean(dim=0, keepdim=True)).norm() / x.norm().clamp_min(1e-30)


class LossV3Module(nn.Module):
    """Unchanged hybrid plus frozen direction-B loss-side operations only.

    ``core_head_weight > 0`` selects the labelled upper bound (arm U): one
    training-only linear head reads final C; labels enter only its loss.
    """

    def __init__(self, cfg: HybridPilotConfig, core_head_weight: float = 0.0) -> None:
        super().__init__()
        if not isinstance(cfg, HybridPilotConfig):
            raise TypeError("cfg must be HybridPilotConfig")
        if not core_head_weight >= 0:
            raise ValueError("core_head_weight must be nonnegative")
        if (
            cfg.T != 12
            or cfg.future_prefix_frames != 6
            or cfg.future_target_frames != (7, 9)
        ):
            raise ValueError(
                "loss v3 freezes 12 frames, prefix through 5 and rollout targets 7/9"
            )
        self.cfg = cfg
        self.model = HybridCDModel(cfg.architecture)
        self.core_head_weight = float(core_head_weight)
        if self.core_head_weight:
            # Created after the model so the model's initialization is unchanged.
            arch = cfg.architecture
            features = arch.stage_count * arch.state_tokens * arch.state_dim
            self.core_head = nn.Linear(features, 4)

    def forward(
        self,
        groups: Sequence[TrainingBatch],
        objective_seeds: Sequence[int],
        labels: Sequence[Sequence[tuple[int, int]]],
        phase: str = "main",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(groups, Sequence) or not groups:
            raise ValueError("groups must be a nonempty sequence")
        if len(groups) != len(objective_seeds) or any(
            type(seed) is not int or not 0 <= seed < 2**63 for seed in objective_seeds
        ):
            raise ValueError(
                "one nonnegative objective seed is required per source group"
            )
        if len(labels) != len(groups):
            raise ValueError("one label table is required per source group")
        for group, source_labels in zip(groups, labels):
            if not isinstance(group, TrainingBatch) or group.images.shape != (
                8,
                1,
                self.cfg.T,
                3,
                self.cfg.H,
                self.cfg.W,
            ):
                raise ValueError(
                    "each source requires eight variants and configured RGB dimensions"
                )
            validate_labels(source_labels)
        mask_phase = "overfit" if phase == "tiny" else phase
        availability = [
            make_training_mask(
                group, self.cfg, torch.Generator().manual_seed(seed), mask_phase
            )
            & group.observed
            for group, seed in zip(groups, objective_seeds)
        ]
        # Only one assimilation: no auxiliary photometric augmentation or heads.
        states = full_observation_states(self.model, groups, availability)
        prefix = states[
            self.cfg.future_prefix_frames // self.cfg.architecture.frames_per_step - 1
        ]
        future_times = torch.stack(
            [
                torch.cat([group.frame_times[:, 0, frame] for group in groups])
                for frame in self.cfg.future_target_frames
            ]
        )
        future_states = self.model.rollout(prefix, future_times, method="associative")
        sources, clean_sources = [], []
        for group_index, (group, mask, source_labels) in enumerate(
            zip(groups, availability, labels)
        ):
            start, stop = group_index * 8, (group_index + 1) * 8
            support = region_support(group, source_labels)
            entries: dict[str, list[tuple[Tensor, Tensor, Tensor]]] = {
                name: [] for name in LOSS_WEIGHTS
            }
            region_entries: dict[str, list[tuple[Tensor, Tensor]]] = {}
            reference = states[0].c[start:stop]

            def rgb_term(name, prediction, target, valid, eligible=None):
                values, active, parts = region_balanced_l1(
                    prediction, target, valid, support
                )
                if eligible is not None:
                    active = active & eligible
                entries[name].append(
                    (_mean_valid(values, active), active.any(), active.sum())
                )
                for region in ("whole", "support", "background"):
                    selected = parts[f"{region}_valid"]
                    if eligible is not None:
                        selected = selected & eligible
                    region_entries.setdefault(f"{name}/{region}_l1", []).append(
                        (_mean_valid(parts[f"{region}_l1"], selected), selected.any())
                    )

            for state, frame in zip(states, ENDPOINTS):
                source_state = slice_state(state, start, stop)
                target = group.images[:, 0, frame]
                prediction = render_group(self.model, source_state, group, frame)
                observed = mask[:, 0, frame]
                rgb_term("rec", prediction, target, observed)
                rgb_term("missing", prediction, target, ~observed)
                eligibility = pair_eligibility(group, mask, source_labels, frame)
                for name, entry in c_objectives(
                    source_state.c, source_labels, eligibility
                ).items():
                    entries[name].append(entry)
                if bool(eligibility.swap_valid.any()):
                    # Only D changes. Receiver C/clock/camera stay fixed; the code
                    # is consumed by the decoder and is never recurrently resumed.
                    swapped = replace(
                        source_state.code(),
                        d=source_state.d[eligibility.config_partner],
                    )
                    swap_prediction = self.model.decoder.render(
                        swapped,
                        height=self.cfg.H,
                        width=self.cfg.W,
                        time=float(group.frame_times[0, 0, frame]),
                        intrinsics=group.intrinsics[0],
                        cam_to_world=group.cam_to_world[0, :, frame],
                    )
                    rgb_term(
                        "swap",
                        swap_prediction,
                        target[eligibility.config_partner],
                        torch.ones_like(observed),
                        eligibility.swap_valid,
                    )
                else:
                    zero = reference.sum() * 0
                    entries["swap"].append((zero, zero.bool(), zero.long()))
            for state, frame in zip(future_states, self.cfg.future_target_frames):
                prediction = render_group(
                    self.model, slice_state(state, start, stop), group, frame
                )
                rgb_term(
                    "future",
                    prediction,
                    group.images[:, 0, frame],
                    torch.ones_like(mask[:, 0, frame]),
                )
            source = {}
            possible = {
                "rec": 48,
                "missing": 48,
                "future": 16,
                "order": 48,
                "c_pair": 24,
                "swap": 48,
                "c_var": 6,
            }
            for name, items in entries.items():
                values, active, counts = (torch.stack(values) for values in zip(*items))
                source[name] = _mean_valid(values, active)
                source[f"active_source_fraction/{name}"] = active.any().float()
                source[f"valid_count/{name}"] = counts.sum().float()
                source[f"skipped_count/{name}"] = possible[name] - counts.sum().float()
                source[f"valid_endpoints/{name}"] = active.sum().float()
            for name, region_items in region_entries.items():
                values, active = (torch.stack(values) for values in zip(*region_items))
                source[name] = _mean_valid(values, active)
            # Always expose identical metric keys even when all swap terms skip.
            for region in ("whole", "support", "background"):
                source.setdefault(f"swap/{region}_l1", reference.sum() * 0)
            source["support_fraction"] = support.float().mean()
            final_c = slice_state(states[-1], start, stop).c
            source["c_variant_contrast"] = variant_contrast(final_c).float()
            if self.core_head_weight:
                cores = torch.tensor(
                    [core for core, _ in source_labels], device=final_c.device
                )
                source["core_ce"] = core_head_cross_entropy(
                    self.core_head, final_c, cores
                ).mean()
                logits = self.core_head(final_c.detach().float().flatten(start_dim=1))
                source["core_accuracy"] = (logits.argmax(dim=1) == cores).float().mean()
                clean_sources.append(mask.all())
            sources.append(source)
        components = {
            name: torch.stack([source[name] for source in sources]).mean()
            for name in sources[0]
        }
        weighted = {
            f"weighted/{name}": weight * components[name]
            for name, weight in LOSS_WEIGHTS.items()
        }
        if self.core_head_weight:
            weighted["weighted/core_ce"] = self.core_head_weight * components["core_ce"]
            accuracy = torch.stack([source["core_accuracy"] for source in sources])
            clean = torch.stack(clean_sources)
            # Source-mean sums aggregate exactly under microbatch weighting; the
            # ratio is formed only after a whole update (split_core_accuracy).
            for name, selected in (("clean", clean), ("masked", ~clean)):
                components[f"core_accuracy_{name}_sum"] = (accuracy * selected).mean()
                components[f"active_source_fraction/core_{name}"] = selected.float().mean()
        total = sum(weighted.values())
        metrics = {**components, **weighted, "loss": total}
        return total, {name: value.detach() for name, value in metrics.items()}
