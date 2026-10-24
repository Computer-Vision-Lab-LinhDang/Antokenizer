#!/usr/bin/env python3
"""Pack OmniWorld-Game for MAVT on DINOv3: rgb + camera + depth + dynamic-object mask + flow (format: omniworld.py).

    PYTHONPATH=src python scripts/pack_omniworld_dino.py --data-root ~/data/omniworld \
        --out ~/data/omniworld-game-dino-v1 --workers 32 [--scenes uid,uid | --limit N]

Each scene's tars are streamed once. A segment is a run of consecutive raw frames of one OmniWorld split with a
valid camera; it keeps every --base-stride-th frame. The raw forward flows between two kept frames are chained at
the aux-grid points (bilinear, source resolution), so a segment's flow.npy moves each point to the next kept frame.
Scenes without flow tars (59 of the 169 on alpha) are packed with NaN flow. Scenes go to train / val by uid hash
(the v6 rule, --val-frac). A scene is written to <uid>.tmp and renamed when complete: a rerun skips it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mavt.dino_v0.omniworld import (AUX, adjust_intrinsics, aux_centres, bilinear, cover_crop,  # noqa: E402
                                    decode_depth, decode_flow, depth_absrel, rigid_flow_epe, split_bucket,
                                    split_cameras)

SUB = "OmniWorld-Game"
_IMG = re.compile(r"(?:^|/)(color|depth)/(\d+)\.png$")
_FLOW = re.compile(r"(?:^|/)flow/(\d+)/flow_([uv])_16\.png$")


def segments(info: dict, cams: dict, base: int) -> list:
    """[(name, split index, kept raw frames)]: runs of consecutive camera-valid raw frames of each split."""
    out = []
    for i, cam in sorted(cams.items()):
        frames = list(info["split"][i])
        bad = {frames[p] for p in np.asarray(cam.get("invalid_frame", []), np.int64).ravel() if 0 <= p < len(frames)}
        runs, cur = [], []
        for f in frames:
            if f in bad or (cur and f != cur[-1] + 1):
                if cur:
                    runs.append(cur)
                cur = [] if f in bad else [f]
            else:
                cur.append(f)
        if cur:
            runs.append(cur)
        for r, run in enumerate(runs):
            kept = run[::base]
            if len(kept) >= 2:
                out.append((f"split{i}_{r}", i, kept, run[-1]))
    return out


def _stream(paths, rx):
    for path in paths:
        with tarfile.open(path, "r|gz") as tf:
            for m in tf:
                hit = rx.search(m.name) if m.isfile() else None
                if hit:
                    yield hit, tf.extractfile(m).read()


def pack_scene(src: Path, uid: str, fps: float, scale: float, dst: Path, args) -> int:
    from PIL import Image
    final, tmp = dst / uid, dst / f"{uid}.tmp"
    if final.exists():
        return 0
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "rgb").mkdir(parents=True)
    H, W, base = args.image_height, args.image_width, args.base_stride
    h, w = H // AUX, W // AUX
    ann = src / "annotations" / SUB / uid
    with tarfile.open(ann / f"{uid}_others.tar.gz") as tf:
        names = {re.sub(r"^\./", "", n): n for n in tf.getnames()}
        info = json.load(tf.extractfile(names["split_info.json"]))
        cams = {i: json.load(tf.extractfile(names[f"camera/split_{i}.json"]))
                for i in range(len(info["split"])) if f"camera/split_{i}.json" in names}
        segs = segments(info, cams, base)
        kept = {f for _, _, k, _ in segs for f in k}
        masks = {}
        for f in kept:
            n = names.get(f"gdino_mask/{f:06d}.png")
            if n is not None:
                masks[f] = np.array(Image.open(tf.extractfile(n)).convert("L"))
    if not segs:
        shutil.rmtree(tmp)
        return 0
    rgb_tars = sorted((src / "videos" / SUB / uid).glob(f"{uid}_rgb_*.tar.gz"))
    crop, saved = None, set()
    for hit, data in _stream(rgb_tars, _IMG):
        f = int(hit[2])
        if f not in kept:
            continue
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if crop is None:
            crop = cover_crop(im.size[0], im.size[1], H, W) + (im.size,)
        s, x0, y0, _ = crop
        im.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + W / s, y0 + H / s)).save(
            tmp / "rgb" / f"{f}.jpg", quality=args.quality)
        saved.add(f)
    if crop is None:
        shutil.rmtree(tmp)
        return 0
    s, x0, y0, (sw, sh) = crop
    xs, ys = aux_centres(h, w)                                      # packed px
    X0, Y0 = (xs + 0.5) / s + x0 - 0.5, (ys + 0.5) / s + y0 - 0.5     # source px
    ix = np.clip(np.round(X0).astype(np.int64), 0, sw - 1)
    iy = np.clip(np.round(Y0).astype(np.int64), 0, sh - 1)
    depth = {}
    for hit, data in _stream(sorted(ann.glob(f"{uid}_depth_*.tar.gz")), _IMG):
        f = int(hit[2])
        if f in kept:
            depth[f] = decode_depth(np.array(Image.open(io.BytesIO(data))), scale)[iy, ix].astype(np.float16)
    dyn = {f: np.asarray(Image.fromarray(m).resize((w, h), Image.BOX, box=(x0, y0, x0 + W / s, y0 + H / s)),
                         np.uint8) for f, m in masks.items()}
    # flow: chain raw t -> t + 1 flows from each kept frame to the next one, at the aux points (source px)
    owner = {}                                                      # raw frame -> kept frame whose chain uses it
    for _, _, k, last in segs:
        for a, b in zip(k, k[1:]):
            for r in range(a, b):
                owner[r] = a
    state = {a: [X0.copy(), Y0.copy(), np.ones(X0.shape, bool), a] for a in set(owner.values())}
    pending, flow = {}, {}
    for hit, data in _stream(sorted(ann.glob(f"{uid}_flow_*.tar.gz")), _FLOW):
        r = int(hit[1])
        a = owner.get(r)
        if a is None:
            continue
        pending.setdefault(r, {})[hit[2]] = np.array(Image.open(io.BytesIO(data)))
        st = state[a]
        while a not in flow and len(pending.get(st[3], ())) == 2:  # apply this chain's flows in order
            uv = pending.pop(st[3])
            d = bilinear(decode_flow(uv["u"], uv["v"]), st[0], st[1])
            st[0], st[1] = st[0] + d[..., 0], st[1] + d[..., 1]
            st[2] &= (st[0] >= -0.5) & (st[0] <= sw - 0.5) & (st[1] >= -0.5) & (st[1] <= sh - 0.5)
            st[3] += 1
            if owner.get(st[3]) != a:                               # reached the next kept frame
                disp = np.stack([(st[0] - X0) * s, (st[1] - Y0) * s], -1)
                flow[a] = np.where(st[2][..., None], disp, np.nan).astype(np.float16)
    Kraw = {i: split_cameras(c, scale) for i, c in cams.items()}
    meta_segs, n_written = [], 0
    for name, i, k, _ in segs:
        k = [f for f in k if f in saved and f in depth]
        if len(k) < 2:
            continue
        frames = list(info["split"][i])
        pos = [frames.index(f) for f in k]
        K = adjust_intrinsics(Kraw[i][0][pos], s, x0, y0)
        c2w = Kraw[i][1][pos]
        D = np.stack([depth[f] for f in k])
        M = np.stack([dyn.get(f, np.zeros((h, w), np.uint8)) for f in k])
        Fl = np.stack([flow[f] if f in flow and f + base == g else np.full((h, w, 2), np.nan, np.float16)
                       for f, g in zip(k, k[1:] + [None])])
        absrel = np.array([depth_absrel(D[j].astype(np.float32), K[j], c2w[j], D[j + 1].astype(np.float32), K[j + 1],
                                        c2w[j + 1]) for j in range(len(k) - 1)], np.float32)
        epe = np.array([rigid_flow_epe(D[j].astype(np.float32), M[j], Fl[j].astype(np.float32), K[j], c2w[j],
                                       K[j + 1], c2w[j + 1]) for j in range(len(k) - 1)], np.float32)
        d = tmp / name
        d.mkdir()
        np.savez(d / "meta.npz", frames=np.asarray(k, np.int64), K=K.astype(np.float32), c2w=c2w,
                 pair_absrel=absrel, pair_epe=epe)
        np.save(d / "depth.npy", D)
        np.save(d / "dyn.npy", M)
        np.save(d / "flow.npy", Fl)
        meta_segs.append(dict(name=name, frames=len(k), absrel_median=float(np.nanmedian(absrel)),
                              epe_median=float(np.nanmedian(epe)) if np.isfinite(epe).any() else None))
        n_written += len(k)
    (tmp / "scene.json").write_text(json.dumps(dict(uid=uid, fps=fps, metric_scale=scale, base_stride=base,
                                                    has_flow=bool(flow), segments=meta_segs)))
    tmp.rename(final)
    return n_written


def _settings(args) -> dict:
    return dict(source="omniworld-game", image_height=args.image_height, image_width=args.image_width, aux=AUX,
                base_stride=args.base_stride, val_frac=args.val_frac, quality=args.quality)


def _write_info(d: Path, args, extra=None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    path, info = d / "pack.json", _settings(args)
    if path.exists():
        old = json.loads(path.read_text())
        bad = {k: (old.get(k), v) for k, v in info.items() if old.get(k) != v}
        if bad:
            raise SystemExit(f"{d} holds a pack with other settings (packed, now): {bad}")
        info = {**old, **info}
    path.write_text(json.dumps({**info, **(extra or {})}, indent=1))


def _job(job):
    src, uid, fps, scale, dst, args = job
    t0 = time.time()
    try:
        return uid, pack_scene(src, uid, fps, scale, dst, args), time.time() - t0, None
    except Exception as ex:                                         # noqa: BLE001 - report and go on
        return uid, 0, time.time() - t0, f"{type(ex).__name__}: {ex}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="OmniWorld root (metadata/, videos/, annotations/)")
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--scenes", default="", help="comma-separated uids (default: every scene with an others tar)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--image-height", type=int, default=256)
    p.add_argument("--image-width", type=int, default=448)
    p.add_argument("--base-stride", type=int, default=2, help="keep every N-th raw frame")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=8, help="0: in this process")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    src, out = Path(args.data_root).expanduser(), Path(args.out).expanduser()
    with open(src / "metadata" / "omniworld_game_metadata.csv", newline="") as f:
        meta = {r["UID"]: r for r in csv.DictReader(f)}
    root = src / "annotations" / SUB
    uids = [u.strip() for u in args.scenes.split(",") if u.strip()] or sorted(
        p.name for p in root.iterdir() if (p / f"{p.name}_others.tar.gz").exists())
    uids = [u for u in uids if u in meta and list((src / "videos" / SUB / u).glob(f"{u}_rgb_*.tar.gz"))]
    if args.limit:
        uids = uids[:args.limit]
    for split in ("train", "val"):
        _write_info(out / split, args)
    jobs = [(src, u, float(meta[u]["FPS"]), float(meta[u]["Metric Scale"]),
             out / ("val" if split_bucket(u) < args.val_frac else "train"), args) for u in uids]
    results = []
    it = mp.Pool(args.workers).imap_unordered(_job, jobs) if args.workers > 0 else map(_job, jobs)
    for uid, n, dt, err in it:
        results.append((uid, n, err))
        print(json.dumps(dict(scene=uid, frames=n, s=round(dt, 1), error=err)), flush=True)
    for split in ("train", "val"):
        d = out / split
        scenes = [json.loads(p.read_text()) for p in sorted(d.glob("*/scene.json"))]
        _write_info(d, args, dict(scenes=len(scenes), frames=sum(g["frames"] for sc in scenes for g in sc["segments"]),
                                  scenes_with_flow=sum(sc["has_flow"] for sc in scenes)))
    return results


if __name__ == "__main__":
    main()
