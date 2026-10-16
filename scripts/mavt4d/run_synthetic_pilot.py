#!/usr/bin/env python3
"""Run the frozen synthetic pilot inside an already allocated CUDA compute job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0")
    parser.add_argument("--git-revision", help="Full source commit SHA (required for a source export without .git)")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("production pilot requires a CUDA device")
    revision = args.git_revision
    if revision is None:
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.PIPE).strip()
        except subprocess.CalledProcessError:
            parser.error("--git-revision is required when the source export has no .git metadata")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mavt4d.pilot.config import PilotConfig
    from mavt4d.pilot.runner import run_suite
    result = run_suite(PilotConfig.load(args.config), args.dataset, args.output,
                       device=args.device, git_revision=revision)
    print(json.dumps({"status": result["status"], "config_sha256": result["config_sha256"],
                      "output": str(args.output)}, sort_keys=True), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
