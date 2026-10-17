"""Complete fixed-dev selection and aligned readout; never canonical held-out."""

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import math

import torch

from mavt4d.data.event_synthetic import EVENT_CORES, EVENT_CONFIGURATIONS
from mavt4d.hybrid.config import is_v3
from mavt4d.hybrid_diagnostic.experiment import _aligned_d_readout_metrics
from mavt4d.hybrid_pilot.evaluation import fit_efficient_ridge_probe
from mavt4d.hybrid_pilot.objectives import full_observation_states, render_group
from mavt4d.pilot.evaluation import _evaluation_records
from .objectives import (
    pair_eligibility,
    region_balanced_l1,
    region_support,
    variant_contrast,
)

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 31415
# Memory controls (v3 arms): core-4 probes on these features when present.
CONTROL_PROBES = (("c_core4_lastwrite", "c_lastwrite"), ("a_core4", "a"))


def last_write_only_states(model, batch):
    """Final state when C/D are zeroed before the last step.

    KV, previous tokens and the appearance bank are kept, so a core probe on
    this C measures history re-derived at the last write rather than carried.
    """
    last = batch.images.shape[2] - model.config.frames_per_step
    prefix = full_observation_states(model, [batch], stop_frame=last)[-1]
    zeroed = replace(prefix, c=torch.zeros_like(prefix.c), d=torch.zeros_like(prefix.d))
    return full_observation_states(model, [batch], start_frame=last, initial=zeroed)[-1]


def labels_for_dataset(dataset):
    records = _evaluation_records(dataset)
    result = {}
    for group in (*dataset.train_groups, *dataset.dev_groups):
        result[group] = [
            (
                EVENT_CORES.index(records[row["record_id"]]["core"]),
                EVENT_CONFIGURATIONS.index(records[row["record_id"]]["configuration"]),
            )
            for row in dataset.records_for_group(group)
        ]
    return result


def probe_metrics(train, dev, *, should_stop=None):
    """Exact existing mean-Gram/ridge protocol, fitting only train features."""
    if (
        not train
        or not dev
        or {x["source"] for x in train} & {x["source"] for x in dev}
    ):
        raise ValueError("probe fit and dev sources must be nonempty and disjoint")
    result = {}

    def fit_score(name, feature, label_index, classes, family=None):
        if should_stop is not None and should_stop():
            raise TimeoutError("deadline between probe fits")
        xs, ys, vs, targets, groups = [], [], [], [], []
        for rows, fitting in ((train, True), (dev, False)):
            for entry in rows:
                indices = [
                    i
                    for i, (core, _) in enumerate(entry["labels"])
                    if family is None or core // 2 == family
                ]
                labels = [entry["labels"][i][label_index] for i in indices]
                if family is not None:
                    labels = [value % 2 for value in labels]
                if fitting:
                    xs.append(entry[feature][indices])
                    ys.extend(labels)
                else:
                    vs.append(entry[feature][indices])
                    targets.extend(labels)
                    groups.extend([entry["source"]] * len(indices))
        probe = fit_efficient_ridge_probe(
            torch.cat(xs),
            torch.tensor(ys, dtype=torch.int64),
            classes=classes,
            ridge=1.0,
        )
        # Closed-form training fit: a capacity control independent of SGD.
        result[f"{name}_train"] = float(
            (probe.predict(torch.cat(xs)) == torch.tensor(ys)).float().mean()
        )
        correct = (
            (probe.predict(torch.cat(vs)) == torch.tensor(targets)).float().tolist()
        )
        by_source = defaultdict(list)
        for group, value in zip(groups, correct):
            by_source[group].append(value)
        scores = torch.tensor(
            [sum(v) / len(v) for v in by_source.values()], dtype=torch.float64
        )
        result[name] = float(scores.mean())
        # 95% source-bootstrap interval, conditional on this one trained model.
        draws = torch.randint(
            len(scores),
            (BOOTSTRAP_SAMPLES, len(scores)),
            generator=torch.Generator().manual_seed(BOOTSTRAP_SEED),
        )
        means = scores[draws].mean(dim=1)
        result[f"{name}_ci_low"] = float(means.quantile(0.025))
        result[f"{name}_ci_high"] = float(means.quantile(0.975))

    for feature in ("c", "d"):
        fit_score(f"{feature}_core4", feature, 0, 4)
        fit_score(f"{feature}_configuration", feature, 1, 2)
        for family in (0, 1):
            fit_score(f"{feature}_direction_family{family}", feature, 0, 2, family)
        result[f"{feature}_direction"] = (
            result[f"{feature}_direction_family0"]
            + result[f"{feature}_direction_family1"]
        ) / 2
    for name, feature in CONTROL_PROBES:
        if feature in train[0]:
            fit_score(name, feature, 0, 4)
    result["E_C"] = ((1 - result["c_direction"]) + (1 - result["c_core4"])) / 2
    return result


def _stopped(deadlines):
    return deadlines.requested or datetime.now(timezone.utc) >= deadlines.work_at


def _expected_reconstruction(model, states, batch, support, *, zero_c=False):
    """Mean region-balanced endpoint error, or None if any region is invalid."""
    errors = []
    for index_state, state in enumerate(states):
        frame = (index_state + 1) * 2 - 1
        if zero_c:
            state = replace(state, c=torch.zeros_like(state.c))
        rgb = render_group(model, state, batch, frame)
        target = batch.images[:, 0, frame]
        values, valid, _ = region_balanced_l1(
            rgb,
            target,
            torch.ones_like(batch.observed[:, 0, frame]),
            support,
        )
        if not bool(valid.all()):
            return None
        errors.append(float(values.mean()))
    return sum(errors) / len(errors)


@torch.no_grad()
def development(module, dataset, labels, deadlines, device, progress=None):
    """No partial pass can return a selection score. Stats refit per checkpoint."""
    was_training = module.training
    module.eval()
    tables = {"train": [], "dev": []}
    reconstruction, swaps = [], []
    # Memory controls exist only for v3 arms; B and U outputs are unchanged.
    v3 = is_v3(module.model.config)
    appearance = v3 and module.model.config.appearance_tokens > 0
    appearance_only, feedback = [], []
    try:
        for split, groups in (
            ("train", dataset.train_groups),
            ("dev", dataset.dev_groups),
        ):
            for index, group in enumerate(groups):
                if _stopped(deadlines):
                    return {"complete": False, "reason": "deadline during development"}
                batch = dataset.load_group(group, device)
                diagnostics = {} if v3 and split == "dev" else None
                states = full_observation_states(
                    module.model, [batch], diagnostics=diagnostics
                )
                entry = {
                    "source": group,
                    "labels": labels[group],
                    "c": states[-1].c.flatten(1).cpu(),
                    "d": states[-1].d.flatten(1).cpu(),
                }
                if v3:
                    last_write = last_write_only_states(module.model, batch)
                    entry["c_lastwrite"] = last_write.c.flatten(1).cpu()
                if appearance:
                    entry["a"] = states[-1].appearance.flatten(1).cpu()
                tables[split].append(entry)
                if split == "dev":
                    if diagnostics is not None:
                        try:
                            feedback.append(
                                [
                                    float(diagnostics[stage]["dc_feedback_ratio"])
                                    for stage in ("stage1", "stage2")
                                ]
                            )
                        except KeyError:
                            return {
                                "complete": False,
                                "reason": "missing feedback diagnostics",
                            }
                    support = region_support(batch, labels[group])
                    error = _expected_reconstruction(
                        module.model, states, batch, support
                    )
                    if error is None:
                        return {
                            "complete": False,
                            "reason": "invalid expected reconstruction",
                        }
                    reconstruction.append(error)
                    if appearance:
                        # Same regions as E_R; validity depends only on masks.
                        appearance_only.append(
                            _expected_reconstruction(
                                module.model, states, batch, support, zero_c=True
                            )
                        )
                    pair = pair_eligibility(batch, batch.observed, labels[group], 3)
                    if not bool(pair.swap_valid.all()):
                        return {
                            "complete": False,
                            "reason": "invalid expected swap pair",
                        }
                    early = states[1]
                    rgb = render_group(
                        module.model,
                        replace(early, d=early.d[pair.config_partner].contiguous()),
                        batch,
                        3,
                    )
                    values, valid, _ = region_balanced_l1(
                        rgb,
                        batch.images[pair.config_partner, 0, 3],
                        torch.ones_like(batch.observed[:, 0, 3]),
                        support,
                    )
                    if not bool(valid.all()):
                        return {
                            "complete": False,
                            "reason": "invalid expected swap region",
                        }
                    swaps.append(float(values.mean()))
                if progress:
                    progress(split, index + 1, len(groups))
        if _stopped(deadlines):
            return {"complete": False, "reason": "deadline before probes"}
        try:
            metrics = probe_metrics(
                tables["train"], tables["dev"], should_stop=lambda: _stopped(deadlines)
            )
        except TimeoutError:
            return {"complete": False, "reason": "deadline between probe fits"}
        if _stopped(deadlines):
            return {"complete": False, "reason": "deadline during probes"}
        metrics.update(
            E_D=sum(swaps) / len(swaps), E_R=sum(reconstruction) / len(reconstruction)
        )
        if appearance:
            # A-only render error (C zeroed); reported, never part of S_dev.
            metrics["E_R_aonly"] = sum(appearance_only) / len(appearance_only)
        if v3:
            for stage, values in zip(("stage1", "stage2"), zip(*feedback)):
                metrics[f"dc_feedback_ratio_{stage}"] = sum(values) / len(values)
        for feature in ("c", "d"):
            metrics[f"{feature}_variant_contrast"] = sum(
                float(variant_contrast(entry[feature])) for entry in tables["dev"]
            ) / len(tables["dev"])
        metrics["S_dev"] = (
            metrics["E_C"] + metrics["E_D"] / 2 + metrics["E_R"] / 2
        ) / 3
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite complete development metric")
        return {
            "complete": True,
            "train_sources": len(tables["train"]),
            "dev_sources": len(tables["dev"]),
            **metrics,
        }
    finally:
        module.train(was_training)


@torch.no_grad()
def aligned_panel(module, batch, labels):
    """Only real C/D paths; v3 has no history/semantic auxiliary heads."""
    was_training = module.training
    module.eval()
    try:
        states = full_observation_states(module.model, [batch], stop_frame=4)
        early = states[-1]
        pair = pair_eligibility(batch, batch.observed, labels, 3)
        base = render_group(module.model, early, batch, 3)
        altered = {
            "d_swap": render_group(
                module.model,
                replace(early, d=early.d[pair.config_partner].contiguous()),
                batch,
                3,
            ),
            "d_zero": render_group(
                module.model, replace(early, d=torch.zeros_like(early.d)), batch, 3
            ),
        }
        target = batch.images[:, 0, 3]
        moving = (batch.images[:, 0] - batch.images[:, 0, :1]).abs().amax(
            dim=(1, 2)
        ) > 1e-5
        metrics = _aligned_d_readout_metrics(
            module.model.decoder,
            early,
            pair.config_partner,
            base,
            altered,
            target,
            moving,
            frame=3,
        )
        indices = [labels.index((0, config)) for config in (0, 1)]
        rows = [
            torch.cat(
                [
                    image[index]
                    for image in (
                        target,
                        base,
                        altered["d_swap"],
                        target[pair.config_partner],
                        altered["d_zero"],
                    )
                ],
                dim=-1,
            )
            for index in indices
        ]
        panel = (torch.cat(rows, dim=-2).clamp(-1, 1).cpu() + 1) / 2
        return metrics, panel
    finally:
        module.train(was_training)
