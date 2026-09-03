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
import os
import shutil
import socket
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_triplanes as rt  # noqa: E402

PLANES = rt.PLANES


HF_OBJAVERSE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"


MAX_GLB_BYTES = 64 * 1024 * 1024     # a 256 px triplane render never needs a 360 MB model (seen: 360 MB @ 100 KB/s = 1 h stall)
FILE_TIME_BUDGET_S = 240


def _fetch_one(uid, rel_path, glb_root, timeout_s, retries=2, max_bytes=MAX_GLB_BYTES, budget_s=FILE_TIME_BUDGET_S):
    """Download one GLB to <glb_root>/hf-objaverse-v1/<rel_path> (atomic rename). Returns (uid, path|None).
    Skips files larger than max_bytes (Content-Length) and aborts a transfer exceeding budget_s."""
    import urllib.request
    dst = os.path.join(glb_root, "hf-objaverse-v1", rel_path)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return uid, dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(HF_OBJAVERSE + rel_path, timeout=timeout_s) as r:
                clen = int(r.headers.get("Content-Length") or 0)
                if clen > max_bytes:
                    return uid, None                                    # too large: skip for good
                t0 = time.time(); got = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk); got += len(chunk)
                        if got > max_bytes or time.time() - t0 > budget_s:
                            raise TimeoutError(f"{uid[:8]}: {got/1e6:.0f} MB in {time.time()-t0:.0f}s — over budget")
            if os.path.getsize(tmp) > 0:
                os.replace(tmp, dst)
                return uid, dst
        except TimeoutError:
            break                                                        # over budget: do not retry
        except Exception:  # noqa: BLE001  (404, socket timeout, reset — retry then give up)
            time.sleep(1 + attempt)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return uid, None


def _load_objects(uids, glb_root, procs, timeout_s=120):
    """Own threaded downloader. objaverse.load_objects() hangs forever when any file in the batch
    fails (its Pool never delivers the missing results — seen twice on 2026-09-02 at 383/384 and
    380/384). We only use objaverse for the uid → path mapping."""
    from concurrent.futures import ThreadPoolExecutor
    import objaverse as ov
    ov.BASE_PATH = glb_root
    ov._VERSIONED_PATH = os.path.join(glb_root, "hf-objaverse-v1")
    rel = ov._load_object_paths()
    jobs = [(u, rel[u]) for u in uids if u in rel]
    out = {}
    with ThreadPoolExecutor(max_workers=procs) as ex:
        for uid, path in ex.map(lambda j: _fetch_one(j[0], j[1], glb_root, timeout_s), jobs):
            if path:
                out[uid] = path
    return out


def glb_paths_for(uids, glb_root):
    """uid → path for GLBs already on disk (objaverse layout: <root>/hf-objaverse-v1/glbs/<shard>/<uid>.glb)."""
    found = {}
    want = set(uids)
    for p in glob.glob(os.path.join(glb_root, "hf-objaverse-v1", "glbs", "*", "*.glb")):
        u = os.path.splitext(os.path.basename(p))[0]
        if u in want and os.path.getsize(p) > 0:
            found[u] = p
    return found


def download_batch(uids, glb_root, procs, timeout_s=120):
    """Download a batch; on any error (timeouts raise thanks to the socket default timeout) fall back
    to whatever GLBs are on disk for these uids so the batch still renders."""
    try:
        paths = _load_objects(uids, glb_root, procs, timeout_s or 120)
        found = {u: p for u, p in paths.items() if p and os.path.exists(p) and os.path.getsize(p) > 0}
    except Exception as exc:  # noqa: BLE001
        print(f"[download] load_objects raised {exc!r} — rendering the files present", flush=True)
        found = {}
    if len(found) < len(uids):
        found.update(glb_paths_for([u for u in uids if u not in found], glb_root))
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
    ap.add_argument("--download-timeout", type=int, default=120, help="socket timeout (s) per request; a stuck request raises instead of hanging")
    a = ap.parse_args()
    socket.setdefaulttimeout(max(10, a.download_timeout))   # objaverse uses urllib with no timeout → a hung request would hang the batch

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
