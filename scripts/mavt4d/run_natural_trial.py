#!/usr/bin/env python3
"""Run the frozen six-GPU natural RGB feasibility trial."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--stop-optimization-at", required=True)
    parser.add_argument("--stop-work-at", required=True)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--global-clips", type=int, default=48)
    parser.add_argument("--dev-every", type=int, default=25)
    parser.add_argument("--image-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mavt4d.hybrid_pilot.runner import parse_utc
    from mavt4d.natural_trial.runner import run_trial
    from mavt4d.natural_trial.model import TrialConfig
    cfg = TrialConfig(steps=args.steps, global_clips=args.global_clips,
                      dev_every=args.dev_every, image_every=args.image_every, log_every=args.log_every)
    result = run_trial(args.dataset, args.output, optimization_at=parse_utc(args.stop_optimization_at),
        work_at=parse_utc(args.stop_work_at), git_revision=args.git_revision,
        microbatch=args.microbatch, resume=args.resume, max_steps=args.max_steps, allow_cpu=args.allow_cpu, cfg=cfg)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
