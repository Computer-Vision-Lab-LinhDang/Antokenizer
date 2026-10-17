"""Complete fixed-dev selection and aligned readout; never canonical held-out."""

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import math

import torch

from mavt4d.data.event_synthetic import EVENT_CORES, EVENT_CONFIGURATIONS
from mavt4d.hybrid_diagnostic.experiment import _aligned_d_readout_metrics
from mavt4d.hybrid_pilot.evaluation import fit_efficient_ridge_probe
from mavt4d.hybrid_pilot.objectives import full_observation_states, render_group
from mavt4d.pilot.evaluation import _evaluation_records
from .objectives import pair_eligibility, region_balanced_l1, region_support


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
        correct = (
            (probe.predict(torch.cat(vs)) == torch.tensor(targets)).float().tolist()
        )
        by_source = defaultdict(list)
        for group, value in zip(groups, correct):
            by_source[group].append(value)
        result[name] = sum(sum(v) / len(v) for v in by_source.values()) / len(by_source)

    for feature in ("c", "d"):
        fit_score(f"{feature}_core4", feature, 0, 4)
        fit_score(f"{feature}_configuration", feature, 1, 2)
        for family in (0, 1):
            fit_score(f"{feature}_direction_family{family}", feature, 0, 2, family)
        result[f"{feature}_direction"] = (
            result[f"{feature}_direction_family0"]
            + result[f"{feature}_direction_family1"]
        ) / 2
    result["E_C"] = ((1 - result["c_direction"]) + (1 - result["c_core4"])) / 2
    return result


def _stopped(deadlines):
    return deadlines.requested or datetime.now(timezone.utc) >= deadlines.work_at


@torch.no_grad()
def development(module, dataset, labels, deadlines, device, progress=None):
    """No partial pass can return a selection score. Stats refit per checkpoint."""
    was_training = module.training
    module.eval()
    tables = {"train": [], "dev": []}
    reconstruction, swaps = [], []
    try:
        for split, groups in (
            ("train", dataset.train_groups),
            ("dev", dataset.dev_groups),
        ):
            for index, group in enumerate(groups):
                if _stopped(deadlines):
                    return {"complete": False, "reason": "deadline during development"}
                batch = dataset.load_group(group, device)
                states = full_observation_states(module.model, [batch])
                tables[split].append(
                    {
                        "source": group,
                        "labels": labels[group],
                        "c": states[-1].c.flatten(1).cpu(),
                        "d": states[-1].d.flatten(1).cpu(),
                    }
                )
                if split == "dev":
                    support = region_support(batch, labels[group])
                    errors = []
                    for index_state, state in enumerate(states):
                        frame = (index_state + 1) * 2 - 1
                        rgb = render_group(module.model, state, batch, frame)
                        target = batch.images[:, 0, frame]
                        values, valid, _ = region_balanced_l1(
                            rgb,
                            target,
                            torch.ones_like(batch.observed[:, 0, frame]),
                            support,
                        )
                        if not bool(valid.all()):
                            return {
                                "complete": False,
                                "reason": "invalid expected reconstruction",
                            }
                        errors.append(float(values.mean()))
                    reconstruction.append(sum(errors) / len(errors))
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
