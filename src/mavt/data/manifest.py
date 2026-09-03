"""Manifest builders: scan datasets once, write .jsonl, load fast at train time.

  python -m mavt.data.manifest --openimages-root data/datasets/open_images_v7/images/train \
      --webvid-root data/datasets/webvid_10m/videos --out-dir data/manifests [--limit N]
"""
from __future__ import annotations
import argparse
import json
import os
import time

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_openimages(root: str, out_path: str, limit: int = 0) -> int:
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    f.write(json.dumps({"path": os.path.join(dirpath, fn)}) + "\n")
                    n += 1
                    if limit and n >= limit:
                        return n
            if n and n % 500_000 == 0:
                print(f"  openimages: {n} files ({time.time()-t0:.0f}s)", flush=True)
    return n


def build_webvid(root: str, out_path: str, limit: int = 0, require_complete: bool = True) -> int:
    """video2dataset files layout: videos/<partition>/**/<key>.mp4 + <key>.txt"""
    n, t0 = 0, time.time()
    with open(out_path, "w") as f:
        for part in sorted(os.listdir(root)):
            pdir = os.path.join(root, part)
            if not os.path.isdir(pdir):
                continue
            if require_complete and not os.path.exists(os.path.join(pdir, ".download_complete")):
                continue
            for dirpath, _, filenames in os.walk(pdir):
                for fn in filenames:
                    if not fn.endswith(".mp4"):
                        continue
                    mp4 = os.path.join(dirpath, fn)
                    rec = {"path": mp4}
                    txt = mp4[:-4] + ".txt"
                    if os.path.exists(txt):
                        rec["caption_path"] = txt
                    f.write(json.dumps(rec) + "\n")
                    n += 1
                    if limit and n >= limit:
                        return n
            if n and n % 200_000 == 0:
                print(f"  webvid: {n} clips ({time.time()-t0:.0f}s)", flush=True)
    return n


def build_openvid(videos_root: str, caption_csv: str, out_path: str, limit: int = 0) -> int:
    """OpenVid-1M layout: videos/**/<name>.mp4 + OpenVid-1M.csv (columns: video, caption, ...)."""
    import csv
    captions = {}
    with open(caption_csv, newline="") as f:
        for row in csv.DictReader(f):
            captions[row["video"]] = row.get("caption", "")
    n = 0
    with open(out_path, "w") as out:
        for dirpath, _, filenames in os.walk(videos_root):
            for fn in filenames:
                if not fn.endswith(".mp4"):
                    continue
                rec = {"path": os.path.join(dirpath, fn)}
                if fn in captions:
                    rec["caption"] = captions[fn]
                out.write(json.dumps(rec) + "\n")
                n += 1
                if limit and n >= limit:
                    return n
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openimages-root")
    ap.add_argument("--webvid-root")
    ap.add_argument("--openvid-root", help="OpenVid videos/ dir")
    ap.add_argument("--openvid-csv", help="OpenVid-1M.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.openimages_root:
        out = os.path.join(a.out_dir, "openimages.jsonl")
        print(f"openimages: {build_openimages(a.openimages_root, out, a.limit)} -> {out}")
    if a.webvid_root:
        out = os.path.join(a.out_dir, "webvid.jsonl")
        print(f"webvid: {build_webvid(a.webvid_root, out, a.limit, not a.allow_incomplete)} -> {out}")
    if a.openvid_root:
        out = os.path.join(a.out_dir, "openvid.jsonl")
        print(f"openvid: {build_openvid(a.openvid_root, a.openvid_csv, out, a.limit)} -> {out}")


if __name__ == "__main__":
    main()
