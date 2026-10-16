#!/usr/bin/env python3
"""Export a bounded, immutable EPIC RGB feasibility dataset without Torch."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--clips-per-source", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mavt4d.natural_trial.prepare import prepare
    print(json.dumps(prepare(args.metadata, args.archives, args.output,
                             sources=args.sources, clips_per_source=args.clips_per_source)), flush=True)


if __name__ == "__main__":
    main()
