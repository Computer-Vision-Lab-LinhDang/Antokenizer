#!/usr/bin/env python3
"""Objaverse-LVIS GLBs with a hard size cap: LVIS_CAP_GB=60 python download_lvis.py"""
import json, os
CAP_GB = float(os.environ.get("LVIS_CAP_GB", "60"))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "../../data/datasets/objaverse_lvis"))
OUT = os.path.abspath(OUT); os.makedirs(OUT, exist_ok=True)
import objaverse
objaverse.BASE_PATH = OUT; objaverse._VERSIONED_PATH = os.path.join(OUT, "hf-objaverse-v1")
lvis = objaverse.load_lvis_annotations()
uids = sorted({u for us in lvis.values() for u in us})
json.dump(lvis, open(os.path.join(OUT, "lvis_annotations.json"), "w"))
print(f"LVIS: {len(lvis)} categories, {len(uids)} uids | cap={CAP_GB} GB", flush=True)
def dir_gb(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 1e9
for i in range(0, len(uids), 500):
    objaverse.load_objects(uids=uids[i:i+500], download_processes=8)
    used = dir_gb(OUT); print(f"[{i+500}/{len(uids)}] size={used:.1f} GB", flush=True)
    if used >= CAP_GB: print(f"CAP {CAP_GB} GB reached — stop.", flush=True); break
print("LVIS_DONE", flush=True)
