#!/usr/bin/env python3
"""Scan MP4 files for 'moov atom not found' / decode-init errors.

Usage:
    python scripts/scan_broken_videos.py [ROOT] [--workers N] [--out PATH]

Writes broken file paths (one per line) to --out (default: logs/broken_videos.txt).
Also writes a JSON summary (counts + sample errors) next to it.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av


def probe(path: str) -> tuple[str, str | None]:
    try:
        with av.open(path, metadata_errors="ignore") as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return path, "no_video_stream"
            # Force header parse
            _ = stream.codec_context
        return path, None
    except av.error.InvalidDataError as e:
        return path, f"invalid_data:{e}"
    except av.error.FFmpegError as e:
        return path, f"ffmpeg:{e}"
    except Exception as e:
        return path, f"{type(e).__name__}:{e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="dataset/dataset_10m")
    ap.add_argument("--workers", type=int, default=max(8, os.cpu_count() or 8))
    ap.add_argument("--out", default="logs/broken_videos.txt")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files (0=all)")
    ap.add_argument("--print-every", type=int, default=5000)
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")

    print(f"[scan] root={root} workers={args.workers} out={out_path}", flush=True)
    print("[scan] enumerating MP4 files...", flush=True)
    files = []
    for p in root.rglob("*.mp4"):
        files.append(str(p))
        if args.limit and len(files) >= args.limit:
            break
    total = len(files)
    print(f"[scan] total={total}", flush=True)

    broken: list[tuple[str, str]] = []
    done = 0
    with open(out_path, "w") as fout, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path, err in ex.map(probe, files, chunksize=64):
            done += 1
            if err is not None:
                fout.write(path + "\n")
                fout.flush()
                broken.append((path, err))
            if done % args.print_every == 0:
                print(f"[scan] {done}/{total} broken={len(broken)}", flush=True)

    print(f"[scan] DONE {done}/{total} broken={len(broken)}", flush=True)

    err_counts: dict[str, int] = {}
    for _, e in broken:
        key = e.split(":", 1)[0]
        err_counts[key] = err_counts.get(key, 0) + 1
    summary = {
        "root": str(root),
        "total": total,
        "broken": len(broken),
        "error_kinds": err_counts,
        "sample": broken[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[scan] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
