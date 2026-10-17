#!/usr/bin/env python3
"""Fresh one-GPU direction-B trial or isolated capacity calibration."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-optimization-at", required=True)
    parser.add_argument("--stop-work-at", required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--microbatch-groups", type=int, default=1)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-spatial-bf16", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mavt4d.hybrid_pilot.config import HybridPilotConfig
    from mavt4d.hybrid_pilot.runner import parse_utc
    from mavt4d.hybrid_loss_v3.config import LossV3TrialConfig
    from mavt4d.hybrid_loss_v3.runner import capacity_probe, run_trial

    common = dict(
        optimization_at=parse_utc(args.stop_optimization_at),
        work_at=parse_utc(args.stop_work_at),
        device="cpu" if args.allow_cpu else "cuda:0",
        allow_cpu=args.allow_cpu,
        spatial_bf16=not (args.allow_cpu or args.no_spatial_bf16),
    )
    positional = (
        HybridPilotConfig.load(args.base_config),
        LossV3TrialConfig.load(args.config),
        args.dataset,
        args.output,
    )
    if args.capacity_only:
        result = capacity_probe(*positional, **common)
    else:
        result = run_trial(
            *positional,
            **common,
            git_revision=args.git_revision,
            microbatch_groups=args.microbatch_groups,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
