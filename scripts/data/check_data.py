#!/usr/bin/env python3
"""Integrity + metadata scan for manifest datasets (images / videos).

For every record in a .jsonl manifest, open the file the same way training does
(PIL for images, PyAV for videos), record its metadata and any decode error, and
write two files next to the output prefix:

    <out>.meta.jsonl   one line per good record: manifest fields + w/h/bytes (+ fps/frames/duration/codec)
    <out>.bad.jsonl    one line per broken record: manifest fields + "error"

Usage:
    python scripts/data/check_data.py image data/manifests/openimages.jsonl --out data/manifests/openimages
    python scripts/data/check_data.py video data/manifests/openvid.jsonl     --out data/manifests/openvid --workers 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, Tuple


def _probe_image(rec: Dict) -> Tuple[Dict, str]:
    from PIL import Image
    path = rec["path"]
    try:
        size = os.path.getsize(path)
        with Image.open(path) as im:
            im.verify()                       # header/CRC check without full decode
        with Image.open(path) as im:
            im.convert("RGB").load()          # full decode exactly like the loader
            w, h, mode = im.width, im.height, im.mode
        return {**rec, "width": w, "height": h, "mode": mode, "bytes": size}, ""
    except Exception as exc:  # noqa: BLE001
        return rec, f"{type(exc).__name__}: {exc}"


def _probe_video(rec: Dict) -> Tuple[Dict, str]:
    import av
    path = rec["path"]
    try:
        size = os.path.getsize(path)
        with av.open(path) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate) if s.average_rate else 0.0
            dur = float(s.duration * s.time_base) if s.duration is not None else (
                float(c.duration / 1e6) if c.duration else 0.0)
            n = int(s.frames) if s.frames else 0
            # Decode first + last frames: catches truncated files the header does not reveal.
            first = None
            for fr in c.decode(s):
                first = fr
                break
            if first is None:
                raise ValueError("no decodable frame")
            w, h = first.width, first.height
            if dur > 0.2:
                c.seek(int(max(0.0, dur - 0.2) / s.time_base), stream=s, backward=True, any_frame=False)
                got = False
                for fr in c.decode(s):
                    got = True
                    break
                if not got:
                    raise ValueError("cannot decode tail")
            if n == 0 and fps > 0:
                n = int(round(dur * fps))
        return {**rec, "width": w, "height": h, "fps": round(fps, 3), "frames": n,
                "duration": round(dur, 3), "codec": s.codec_context.name, "bytes": size}, ""
    except Exception as exc:  # noqa: BLE001
        return rec, f"{type(exc).__name__}: {exc}"


_PROBES = {"image": _probe_image, "video": _probe_video}


def _worker(args: Tuple[str, Dict]) -> Tuple[Dict, str]:
    kind, rec = args
    return _PROBES[kind](rec)


def _summarize(kind: str, metas: list, bads: list) -> Dict:
    import statistics as st
    n = len(metas) + len(bads)
    out: Dict = {"kind": kind, "total": n, "good": len(metas), "bad": len(bads),
                 "bad_ratio": round(len(bads) / max(1, n), 5),
                 "bytes_gb": round(sum(m["bytes"] for m in metas) / 1e9, 2)}
    if not metas:
        return out
    ws = [m["width"] for m in metas]; hs = [m["height"] for m in metas]
    mins = [min(w, h) for w, h in zip(ws, hs)]
    out["min_side"] = {"p05": _pct(mins, 5), "p50": _pct(mins, 50), "p95": _pct(mins, 95),
                       "lt256": sum(1 for m in mins if m < 256), "lt384": sum(1 for m in mins if m < 384)}
    ars = sorted(round(w / h, 2) for w, h in zip(ws, hs))
    out["aspect"] = {"p05": _pct(ars, 5), "p50": _pct(ars, 50), "p95": _pct(ars, 95)}
    if kind == "image":
        modes: Dict[str, int] = {}
        for m in metas:
            modes[m["mode"]] = modes.get(m["mode"], 0) + 1
        out["modes"] = modes
    else:
        fr = [m["frames"] for m in metas]; du = [m["duration"] for m in metas]; fp = [m["fps"] for m in metas]
        codecs: Dict[str, int] = {}
        res: Dict[str, int] = {}
        for m in metas:
            codecs[m["codec"]] = codecs.get(m["codec"], 0) + 1
            k = f'{m["width"]}x{m["height"]}'
            res[k] = res.get(k, 0) + 1
        out["frames"] = {"p05": _pct(fr, 5), "p50": _pct(fr, 50), "p95": _pct(fr, 95),
                         "lt16": sum(1 for f in fr if f < 16), "lt32": sum(1 for f in fr if f < 32)}
        out["duration_s"] = {"p05": _pct(du, 5), "p50": _pct(du, 50), "p95": _pct(du, 95), "sum_h": round(sum(du) / 3600, 1)}
        out["fps"] = {"p05": _pct(fp, 5), "p50": _pct(fp, 50), "p95": _pct(fp, 95)}
        out["codecs"] = codecs
        out["top_resolutions"] = dict(sorted(res.items(), key=lambda kv: -kv[1])[:8])
    return out


def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(_PROBES))
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.meta.jsonl / .bad.jsonl / .summary.json)")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 4))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.manifest) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        recs = recs[: args.limit]
    print(f"[check_data] {args.kind}: {len(recs)} records, {args.workers} workers", flush=True)

    metas, bads = [], []
    t0 = time.time()
    with Pool(args.workers) as pool, \
            open(args.out + ".meta.jsonl", "w") as fm, open(args.out + ".bad.jsonl", "w") as fb:
        for i, (rec, err) in enumerate(pool.imap_unordered(_worker, ((args.kind, r) for r in recs), chunksize=16), 1):
            if err:
                bads.append(rec)
                fb.write(json.dumps({**rec, "error": err}) + "\n")
            else:
                metas.append(rec)
                fm.write(json.dumps(rec) + "\n")
            if i % 2000 == 0 or i == len(recs):
                el = time.time() - t0
                print(f"  {i}/{len(recs)}  bad={len(bads)}  {i/el:.0f}/s  eta={(len(recs)-i)/max(1e-6,i/el)/60:.1f} min", flush=True)

    summary = _summarize(args.kind, metas, bads)
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("CHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
