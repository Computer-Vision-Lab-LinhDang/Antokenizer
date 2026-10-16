#!/usr/bin/env python3
"""Run one fresh, bounded C/D mechanism diagnostic on one allocated GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "write_gate", "d_strength"), required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--probe-every", type=int, default=50)
    parser.add_argument("--image-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-optimization-at", required=True)
    parser.add_argument("--stop-work-at", required=True)
    parser.add_argument("--git-revision", required=True)
    parser.add_argument("--audit-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-spatial-bf16", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mavt4d.hybrid_pilot.config import HybridPilotConfig
    from mavt4d.hybrid_pilot.runner import parse_utc
    from mavt4d.hybrid_diagnostic.experiment import DiagnosticConfig
    from mavt4d.hybrid_diagnostic.runner import run_diagnostic
    result = run_diagnostic(HybridPilotConfig.load(args.config),
        DiagnosticConfig(arm=args.arm, steps=args.steps, probe_every=args.probe_every, image_every=args.image_every),
        args.dataset, args.output, optimization_at=parse_utc(args.stop_optimization_at),
        work_at=parse_utc(args.stop_work_at), git_revision=args.git_revision,
        device="cpu" if args.allow_cpu else args.device, spatial_bf16=not (args.no_spatial_bf16 or args.allow_cpu),
        audit_checkpoints=args.audit_checkpoint, allow_cpu=args.allow_cpu)
    print(json.dumps({k: result[k] for k in ("status", "arm", "step", "best_step", "best_history_mse")}))


if __name__ == "__main__":
    main()
