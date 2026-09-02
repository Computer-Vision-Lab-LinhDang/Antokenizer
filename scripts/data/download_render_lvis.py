#!/usr/bin/env python3
"""Objaverse-LVIS → triplane dataset with a small disk footprint.

Per batch of uids: download GLBs (objaverse) → render oxoy/oxoz/oyoz PNGs (render_triplanes.py,
CPU pool) → delete the GLBs. Only the PNGs stay (~0.3 MB / object). Resumable: objects whose
three PNGs exist are skipped. Captions from Cap3D (uid → caption) are written to
<out>/captions/3d.json so UniversalThreeDDataset can read <out> directly.

    python scripts/data/download_render_lvis.py --out data/datasets/objaverse_lvis \
        --captions data/datasets/cap3d/captions/Cap3D_automated_Objaverse_full.csv \
        --max-objects 46207 --batch 256 --workers 48
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_triplanes as rt  # noqa: E402

PLANES = rt.PLANES


def rendered(out_root: str, uid: str) -> bool:
    d = os.path.join(out_root, "3d_objects", "renders", uid)
    return all(os.path.exists(os.path.join(d, f"{p}.png")) for p in PLANES)


def load_captions(csv_path: str) -> dict:
    caps = {}
    if not csv_path or not os.path.exists(csv_path):
        return caps
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                caps[row[0]] = row[1]
    return caps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--captions", default="")
    ap.add_argument("--max-objects", type=int, default=0, help="0 = all LVIS objects")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--download-procs", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--points", type=int, default=300_000)
    ap.add_argument("--keep-glb", action="store_true")
    a = ap.parse_args()

    import objaverse
    glb_root = os.path.join(a.out, "_glb_tmp")
    objaverse.BASE_PATH = glb_root
    objaverse._VERSIONED_PATH = os.path.join(glb_root, "hf-objaverse-v1")
    os.makedirs(os.path.join(a.out, "3d_objects", "renders"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "captions"), exist_ok=True)

    lvis = objaverse.load_lvis_annotations()
    uid2cat = {u: c for c, us in lvis.items() for u in us}
    uids = sorted(uid2cat)
    if a.max_objects:
        uids = uids[: a.max_objects]
    todo = [u for u in uids if not rendered(a.out, u)]
    print(f"[lvis] {len(lvis)} categories, {len(uids)} objects targeted, {len(uids) - len(todo)} already rendered, {len(todo)} to do", flush=True)

    caps = load_captions(a.captions)
    cap_path = os.path.join(a.out, "captions", "3d.json")
    cap_out = json.load(open(cap_path)) if os.path.exists(cap_path) else {}
    fail_log = open(os.path.join(a.out, "render_failures.jsonl"), "a")

    t0 = time.time(); done = 0; failed = 0
    for i in range(0, len(todo), a.batch):
        batch = todo[i: i + a.batch]
        try:
            paths = objaverse.load_objects(uids=batch, download_processes=a.download_procs)
        except Exception as exc:  # noqa: BLE001
            print(f"[download] batch {i // a.batch} failed: {exc!r} — retrying once", flush=True)
            time.sleep(10)
            paths = objaverse.load_objects(uids=batch, download_processes=a.download_procs)
        items = [(p, os.path.join(a.out, "3d_objects", "renders", u)) for u, p in paths.items() if p and os.path.exists(p)]
        ok, errs = rt.render_many(items, a.res, a.points, a.workers)
        for u, e in errs.items():
            fail_log.write(json.dumps({"uid": u, "error": e}) + "\n")
            shutil.rmtree(os.path.join(a.out, "3d_objects", "renders", u), ignore_errors=True)
        fail_log.flush()
        for u in paths:
            if rendered(a.out, u):
                cap_out[u] = caps.get(u) or uid2cat[u].replace("_", " ")   # fall back to the LVIS category name
        json.dump(cap_out, open(cap_path, "w"))
        if not a.keep_glb:
            for p in paths.values():
                try:
                    os.remove(p)
                except OSError:
                    pass
        done += ok; failed += len(errs)
        el = time.time() - t0
        print(f"[{i + len(batch)}/{len(todo)}] ok={done} failed={failed} {done / max(el, 1e-6) * 60:.0f} obj/min "
              f"eta={(len(todo) - i - len(batch)) / max(1, done / max(el, 1e-6)) / 60:.0f} min", flush=True)
    if not a.keep_glb:
        shutil.rmtree(glb_root, ignore_errors=True)
    print(f"RENDER_DONE ok={done} failed={failed} captions={len(cap_out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
