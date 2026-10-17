#!/usr/bin/env python3
"""Fresh-init within-source contrast for arms B/I/A/IA (CPU, no training).

Each arm is built exactly as the trainer builds it, so I/A/IA share every
arm-B weight and differ only by the innovation write and appearance bank.
Sources are rendered in memory with the event generator (eight core x
configuration variants each); held-out sources are skipped. Contrast is
||X - mean over variants|| / ||X|| within one source.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from mavt4d.contracts import ObservationStep  # noqa: E402
from mavt4d.data.event_synthetic import (  # noqa: E402
    EVENT_CONFIGURATIONS,
    EVENT_CORES,
    make_event_fixture,
)
from mavt4d.hybrid.config import HybridConfig  # noqa: E402
from mavt4d.hybrid_loss_v3.config import ARCH_VERSIONS  # noqa: E402
from mavt4d.hybrid_loss_v3.objectives import variant_contrast  # noqa: E402
from mavt4d.hybrid_loss_v3.runner import initialize  # noqa: E402
from mavt4d.hybrid_pilot.config import HybridPilotConfig  # noqa: E402
from mavt4d.hybrid_pilot.data import TrainingBatch  # noqa: E402
from mavt4d.hybrid_pilot.objectives import full_observation_states  # noqa: E402


def source_batch(cfg, seed):
    """All eight variants of one source, or None for a held-out source."""
    fixtures = [
        make_event_fixture(cfg.event_config, seed, core, configuration)
        for core in EVENT_CORES
        for configuration in EVENT_CONFIGURATIONS
    ]
    if fixtures[0].evaluation.split == "val":
        return None
    scenes = [fixture.scene for fixture in fixtures]
    images = torch.stack([scene.images for scene in scenes])
    return TrainingBatch(
        images=images,
        observed=torch.ones(images.shape[:3] + images.shape[-2:], dtype=torch.bool),
        frame_times=torch.stack([scene.frame_times for scene in scenes]),
        intrinsics=torch.stack([scene.intrinsics for scene in scenes]),
        cam_to_world=torch.stack([scene.cam_to_world for scene in scenes]),
    )


def final_evidence(model, batch):
    """Summary evidence [8,P,W] of the final step (the observer is stateless)."""
    start = batch.images.shape[2] - model.config.frames_per_step
    steps = [
        ObservationStep(
            images=batch.images[i, :, start:].contiguous(),
            observed=batch.observed[i, :, start:].contiguous(),
            frame_times=batch.frame_times[i, :, start:].contiguous(),
            intrinsics=batch.intrinsics[i].contiguous(),
            cam_to_world=batch.cam_to_world[i, :, start:].contiguous(),
        )
        for i in range(len(batch.images))
    ]
    return model.observer.forward_batch(steps)[0]


def measure(model, batch):
    diagnostics = {}
    states = full_observation_states(model, [batch], diagnostics=diagnostics)
    row = {
        "evidence_final": float(variant_contrast(final_evidence(model, batch))),
        "c1": [float(variant_contrast(state.c[:, 0])) for state in states],
        "c2": [float(variant_contrast(state.c[:, 1])) for state in states],
        "d_final": float(variant_contrast(states[-1].d)),
    }
    if model.appearance_tokens:
        row["a_final"] = float(variant_contrast(states[-1].appearance))
    if model.innovation:
        row["innovation_rms"] = diagnostics["innovation_rms"].mean(dim=1).tolist()
    return row


def mean_rows(rows):
    """Elementwise mean over sources of scalar and per-step entries."""
    result = {}
    for key, value in rows[0].items():
        values = [row[key] for row in rows]
        if isinstance(value, list):
            result[key] = [sum(step) / len(rows) for step in zip(*values)]
        else:
            result[key] = sum(values) / len(rows)
    return result


def probe(base, sources, seed):
    seeds, batches, candidate = [], [], base.seed_start
    while len(batches) < sources:
        batch = source_batch(base, candidate)
        if batch is not None:
            seeds.append(candidate)
            batches.append(batch)
        candidate += 1
    base = replace(base, seed=seed)
    arms = {"B": None, **{arm: variant for arm, variant in ARCH_VERSIONS.values()}}
    report = {}
    for arm, variant in arms.items():
        module = initialize(
            base, torch.device("cpu"), spatial_bf16=False, architecture_variant=variant
        ).eval()
        with torch.no_grad():
            rows = [measure(module.model, batch) for batch in batches]
        report[arm] = {**mean_rows(rows), "per_source": rows}
    return {
        "schema": "hybrid-variant-contrast-probe-v1",
        "seed": seed,
        "sources": seeds,
        "H": base.H,
        "W": base.W,
        "architecture": asdict(base.architecture),
        "arms": report,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="tiny architecture at 16x24 (CPU correctness checks only)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sources <= 0:
        parser.error("--sources must be positive")
    base = HybridPilotConfig.load(args.base_config)
    if args.tiny:
        base = replace(
            base,
            architecture=HybridConfig.tiny(),
            H=16,
            W=24,
            temporal_pool=(2, 3),
            semantic_dim=4,
        )
    result = probe(base, args.sources, args.seed)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
