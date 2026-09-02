#!/usr/bin/env python3
"""Build the training manifests (v2) from the check_data.py scan output.

    openimages : scan meta  →  {path, width, height, mode, bytes}
    openvid    : scan meta + OpenVid-1M.csv (aesthetic / motion / temporal-consistency /
                 camera motion)  →  filtered by --min-frames, sorted for determinism

Records the scan flagged as broken are never emitted (they are in <scan>.bad.jsonl).

Usage:
    python scripts/data/enrich_manifests.py image data/manifests/scan/openimages.meta.jsonl \
        --out data/manifests/openimages_v2.jsonl
    python scripts/data/enrich_manifests.py video data/manifests/scan/openvid.meta.jsonl \
        --csv data/datasets/openvid/data/train/OpenVid-1M.csv --min-frames 16 \
        --out data/manifests/openvid_v2.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Optional

IMAGE_KEYS = ("path", "caption", "caption_path", "width", "height", "mode", "bytes")
VIDEO_KEYS = ("path", "caption", "width", "height", "fps", "frames", "duration", "codec", "bytes")
CSV_FIELDS = {"aesthetic score": ("aesthetic", float), "motion score": ("motion", float),
              "temporal consistency score": ("temporal_consistency", float),
              "camera motion": ("camera_motion", str)}


def _read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _pick(rec: Dict, keys: Iterable[str]) -> Dict:
    return {k: rec[k] for k in keys if k in rec}


def load_openvid_csv(csv_path: str, wanted: Optional[set] = None) -> Dict[str, Dict]:
    """basename → {aesthetic, motion, temporal_consistency, camera_motion}; NaN-safe."""
    out: Dict[str, Dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("video", "")
            if wanted is not None and name not in wanted:
                continue
            rec: Dict = {}
            for col, (key, typ) in CSV_FIELDS.items():
                v = row.get(col)
                if v is None or v == "":
                    continue
                try:
                    rec[key] = typ(v) if typ is not str else v
                except ValueError:
                    continue
            out[name] = rec
    return out


def enrich_images(meta: List[Dict]) -> List[Dict]:
    return sorted((_pick(r, IMAGE_KEYS) for r in meta), key=lambda r: r["path"])


def enrich_videos(meta: List[Dict], csv_meta: Dict[str, Dict], min_frames: int) -> List[Dict]:
    out: List[Dict] = []
    for r in meta:
        if int(r.get("frames", 0)) < min_frames:
            continue
        rec = _pick(r, VIDEO_KEYS)
        rec.update(csv_meta.get(os.path.basename(r["path"]), {}))
        out.append(rec)
    return sorted(out, key=lambda r: r["path"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["image", "video"])
    ap.add_argument("meta", help="<scan>.meta.jsonl from check_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", help="OpenVid-1M.csv (video only)")
    ap.add_argument("--min-frames", type=int, default=16)
    args = ap.parse_args()

    meta = _read_jsonl(args.meta)
    if args.kind == "image":
        recs = enrich_images(meta)
    else:
        csv_meta = load_openvid_csv(args.csv, {os.path.basename(r["path"]) for r in meta}) if args.csv else {}
        recs = enrich_videos(meta, csv_meta, args.min_frames)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    dropped = len(meta) - len(recs)
    print(f"[enrich] {args.kind}: {len(recs)} records written to {args.out} (dropped {dropped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
