#!/usr/bin/env python3
"""Run the locked synthetic hybrid pilot in an allocated eight-GPU job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("tiny", "main", "all"), default="all")
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-optimization-at", required=True, help="ISO8601 timestamp with timezone")
    parser.add_argument("--stop-work-at", required=True, help="ISO8601 timestamp with timezone")
    parser.add_argument("--git-revision", help="Full source commit SHA")
    parser.add_argument("--spatial-bf16", action="store_true", help="Autocast only the stateless spatial encoder")
    parser.add_argument("--max-steps", type=int, help="Explicit partial verification cap; suppresses held-out evaluation")
    parser.add_argument("--allow-cpu", action="store_true", help="Explicit CPU correctness-test mode, without W&B")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    revision = args.git_revision
    if revision is None:
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.PIPE).strip()
        except subprocess.CalledProcessError:
            parser.error("--git-revision is required for an exported source tree")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mavt4d.hybrid_pilot.config import HybridPilotConfig
    from mavt4d.hybrid_pilot.runner import parse_utc, run_pilot
    result = run_pilot(HybridPilotConfig.load(args.config), args.dataset, args.output,
        phase=args.phase, microbatch=args.microbatch, optimization_at=parse_utc(args.stop_optimization_at),
        work_at=parse_utc(args.stop_work_at), git_revision=revision, resume=args.resume,
        spatial_bf16=args.spatial_bf16, allow_cpu=args.allow_cpu, max_steps=args.max_steps)
    print(json.dumps({"status": result["status"], "config_sha256": result["config_sha256"]}), flush=True)
    # A bounded partial run is an expected success, and remains visibly stopped
    # in both its result and checkpoint instead of masquerading as completion.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
