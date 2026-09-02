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
import glob
import json
import multiprocessing as mp
import os
import shutil
import socket
import sys
import time

socket.setdefaulttimeout(120)          # objaverse downloads via urllib with no timeout → hung request = hung batch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_triplanes as rt  # noqa: E402

PLANES = rt.PLANES


def _download_child(uids, glb_root, procs):
    os.setsid()                       # own process group → the parent can kill the whole download tree on timeout
    import objaverse as ov
    ov.BASE_PATH = glb_root
    ov._VERSIONED_PATH = os.path.join(glb_root, "hf-objaverse-v1")
    try:
        ov.load_objects(uids=uids, download_processes=procs)
    except Exception as exc:  # noqa: BLE001
        print(f"[download] load_objects raised {exc!r} — using whatever was fetched", flush=True)


def glb_paths_for(uids, glb_root):
    """uid → path for GLBs already on disk (objaverse layout: <root>/hf-objaverse-v1/glbs/<shard>/<uid>.glb)."""
    found = {}
    want = set(uids)
    for p in glob.glob(os.path.join(glb_root, "hf-objaverse-v1", "glbs", "*", "*.glb")):
        u = os.path.splitext(os.path.basename(p))[0]
        if u in want and os.path.getsize(p) > 0:
            found[u] = p
    return found


def download_batch(uids, glb_root, procs, timeout_s):
    """Download a batch in a separate process with a hard timeout (2026-09-02: one stuck request left
    the objaverse pool waiting forever with 383/384 files done). Returns uid → path for what exists."""
    import signal
    ctx = mp.get_context("spawn")
    # NOT daemon: objaverse spawns its own multiprocessing.Pool inside (daemonic processes may not have children)
    proc = ctx.Process(target=_download_child, args=(uids, glb_root, procs), daemon=False)
    proc.start(); proc.join(timeout_s)
    if proc.is_alive():
        print(f"[download] batch timed out after {timeout_s}s — killing downloader, rendering the {len(glb_paths_for(uids, glb_root))} files present", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # child called setsid() → pid == pgid
        except ProcessLookupError:
            proc.kill()
        proc.join(10)
    found = glb_paths_for(uids, glb_root)
    if not found and proc.exitcode not in (0, None):
        print(f"[download] downloader exited with code {proc.exitcode} and fetched nothing — check the log above", flush=True)
    return found


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
    ap.add_argument("--points", type=int, default=200_000)
    ap.add_argument("--keep-glb", action="store_true")
    ap.add_argument("--download-timeout", type=int, default=900, help="hard timeout (s) per batch download")
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

    t0 = time.time(); done = 0; failed = 0; empty_batches = 0
    for i in range(0, len(todo), a.batch):
        batch = todo[i: i + a.batch]
        paths = download_batch(batch, glb_root, a.download_procs, a.download_timeout)
        empty_batches = empty_batches + 1 if not paths else 0
        if empty_batches >= 3:
            print("[download] 3 consecutive batches fetched nothing — aborting (network/objaverse problem, uids stay unrendered for the next run)", flush=True)
            break
        missing = [u for u in batch if u not in paths]
        if missing:
            for u in missing:
                fail_log.write(json.dumps({"uid": u, "error": "download missing/timeout"}) + "\n")
            print(f"[download] {len(missing)} of {len(batch)} objects not fetched this batch", flush=True)
        items = [(p, os.path.join(a.out, "3d_objects", "renders", u)) for u, p in paths.items() if p and os.path.exists(p)]
        ok, errs = rt.render_many(items, a.res, a.points, a.workers)
        for u, e in errs.items():
            fail_log.write(json.dumps({"uid": u, "error": e}) + "\n")
            shutil.rmtree(os.path.join(a.out, "3d_objects", "renders", u), ignore_errors=True)
        fail_log.flush()
        for u in paths:
            if rendered(a.out, u):
                cap_out[u] = caps.get(u) or uid2cat[u].replace("_", " ")   # fall back to the LVIS category name
        tmp = cap_path + ".tmp"                      # atomic: a training job may read 3d.json at any time
        with open(tmp, "w") as f:
            json.dump(cap_out, f)
        os.replace(tmp, cap_path)
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
