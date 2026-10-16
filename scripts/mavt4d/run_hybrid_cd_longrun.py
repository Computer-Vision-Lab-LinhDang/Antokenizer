#!/usr/bin/env python3
"""Train a reviewed C/D mechanism on two allocated GPUs with a strict deadline."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True, choices=("baseline", "write_gate", "d_strength"))
    parser.add_argument("--target-steps", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--dev-every", type=int, default=200)
    parser.add_argument("--image-every", type=int, default=1000)
    parser.add_argument("--stop-optimization-at", required=True)
    parser.add_argument("--stop-work-at", required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-spatial-bf16", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mavt4d.hybrid_pilot.config import HybridPilotConfig
    from mavt4d.hybrid_pilot.runner import parse_utc
    from mavt4d.hybrid_diagnostic.longrun import LongRunConfig, run_long
    result = run_long(HybridPilotConfig.load(args.config),
        LongRunConfig(arm=args.arm, target_steps=args.target_steps, max_steps=args.max_steps,
                      dev_every=args.dev_every, image_every=args.image_every),
        args.dataset, args.output, optimization_at=parse_utc(args.stop_optimization_at),
        work_at=parse_utc(args.stop_work_at), git_revision=args.git_revision,
        spatial_bf16=not (args.allow_cpu or args.no_spatial_bf16), allow_cpu=args.allow_cpu)
    print(json.dumps({k: result[k] for k in ("status", "step", "best_step", "best_dev_loss")}))


if __name__ == "__main__":
    main()
