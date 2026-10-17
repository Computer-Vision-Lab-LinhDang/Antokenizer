#!/usr/bin/env python3
"""Export a bounded, immutable EPIC RGB feasibility dataset without Torch."""
import argparse
from datetime import datetime
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--frame-selection", choices=("available_grid", "metadata_grid"), default="available_grid")
    parser.add_argument("--stop-at", help="Explicit timezone ISO8601 export cutoff")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mavt4d.natural_trial.prepare import prepare
    cutoff = None
    if args.stop_at:
        parsed = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parser.error("--stop-at requires an explicit timezone")
        cutoff = parsed.timestamp()
    print(json.dumps(prepare(args.metadata, args.archives, args.output,
                             sources=args.sources, clips_per_source=args.clips_per_source,
                             workers=args.workers, max_archives=args.max_archives,
                             frame_selection=args.frame_selection, stop_at=cutoff)), flush=True)


if __name__ == "__main__":
    main()
