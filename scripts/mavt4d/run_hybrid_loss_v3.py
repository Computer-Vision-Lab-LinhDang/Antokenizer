#!/usr/bin/env python3
"""Fresh or resumed one-GPU B trial, or isolated capacity calibration."""
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
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Deliberate warm-start from trusted best.pt: resets step0; use --resume for continuation",
    )
    initialization.add_argument(
        "--resume",
        type=Path,
        help="Trusted local last.pt or prior run directory; retains global step and Adam",
    )
    parser.add_argument(
        "--allow-batch-change",
        action="store_true",
        help="Explicit compatible continuation with a new effective batch",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-spatial-bf16", action="store_true")
    args = parser.parse_args()
    if args.capacity_only and (
        args.initial_checkpoint is not None or args.resume is not None
    ):
        parser.error("capacity probes must use disposable fresh models")
    if args.allow_batch_change and args.resume is None:
        parser.error("--allow-batch-change requires --resume")
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
            initial_checkpoint=args.initial_checkpoint,
            resume=args.resume,
            allow_batch_change=args.allow_batch_change,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
