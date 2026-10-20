"""Pack OmniWorld-Game scenes for v6 training (layout and conventions: data_omniworld.py).

    PYTHONPATH=src python -m mavt4d.v6.pack_omniworld --data-root ~/data/omniworld --out ~/data/omniworld-game-v6 \
        --workers 32 [--scenes uid,uid | --limit N] [--check]

Each scene's tars are streamed once; only frames a v6 clip can use are written: anchors every --step-dt on each split
and their tube frames, when rgb, depth and a valid camera (not in invalid_frame) all exist. Scenes go to train or val
by sha1 of 'mavt-split-v1' + 'omniworld-game:<uid>' (--val-frac of the scenes), not by OmniWorld's in-scene test
splits, so no scene is on both sides. A scene is written to <uid>.tmp and renamed when complete: a rerun skips it.
--check prints, per scene, the median depth reprojection AbsRel from step 0 to step 1 of its first clip.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import re
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np

from ..eval.fixed_suite import bucket
from .data_av2 import PACK_INFO, adjust_intrinsics, load_image
from .data_omniworld import (MAX_DEPTH, SOURCE, OmniWorldClipDataset, anchors, decode_depth, depth_consistency,
                             reprojection_absrel)

SUB = 'OmniWorld-Game'
_FRAME = re.compile(r'(?:^|/)(color|depth)/(\d+)\.png$')


def scene_split(uid, val_frac):
    return 'val' if bucket(f'{SOURCE}:{uid}') < val_frac else 'train'


def _quat_matrix(q):
    w, x, y, z = (np.asarray(q, np.float64) / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1).reshape(-1, 3, 3)


def _members(paths, want):
    """Stream .tar.gz files, yield (kind, frame, bytes) for color/depth PNGs whose frame is in want."""
    for path in paths:
        with tarfile.open(path, 'r|gz') as tf:
            for m in tf:
                hit = _FRAME.search(m.name) if m.isfile() else None
                if hit and int(hit[2]) in want:
                    yield hit[1], int(hit[2]), tf.extractfile(m).read()


def _depth_sampler(crop, src_hw, height, width):
    """Nearest source pixel (flat index) for every target pixel; centres at integers as adjust_intrinsics."""
    s, x0, y0 = crop
    u = np.clip(np.floor((np.arange(width) + .5) / s + x0).astype(np.int64), 0, src_hw[1] - 1)
    v = np.clip(np.floor((np.arange(height) + .5) / s + y0).astype(np.int64), 0, src_hw[0] - 1)
    return (v[:, None] * src_hw[1] + u[None]).reshape(-1)


def pack_scene(src, uid, fps, scale, dst, args):
    """Write scene uid under dst/<uid>; returns the number of frames written (0 when it already exists)."""
    from PIL import Image
    final, tmp = dst / uid, dst / f'{uid}.tmp'
    if final.exists():
        return 0
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / 'rgb').mkdir(parents=True)
    (tmp / 'depth').mkdir()
    ann = src / 'annotations' / SUB / uid
    with tarfile.open(ann / f'{uid}_others.tar.gz') as tf:
        names = {re.sub(r'^\./', '', n): n for n in tf.getnames()}
        info = json.load(tf.extractfile(names['split_info.json']))
        cams = {i: json.load(tf.extractfile(names[f'camera/split_{i}.json']))
                for i in range(len(info['split'])) if f'camera/split_{i}.json' in names}
    wanted, plans = set(), {}
    for i, cam in cams.items():
        frames = list(info['split'][i])
        bad = {frames[p] for p in cam.get('invalid_frame', []) if 0 <= p < len(frames)}
        ok = set(frames) - bad
        grid = anchors(frames[0], frames[-1], fps, args.step_dt, args.tube_dt, args.tubelet)
        keep = {f for _, tube in grid if set(tube) <= ok for f in tube}
        if keep:
            plans[i] = (frames, keep)
            wanted |= keep
    ext = 'png' if args.png else 'jpg'
    got, crop, sampler = {'color': set(), 'depth': set()}, None, None
    rgb_tars = sorted((src / 'videos' / SUB / uid).glob(f'{uid}_rgb_*.tar.gz'))
    depth_tars = sorted(ann.glob(f'{uid}_depth_*.tar.gz'))
    for kind, frame, data in _members(rgb_tars, wanted):
        img, this = load_image(io.BytesIO(data), args.image_height, args.image_width)
        if crop is not None and this != crop:
            raise ValueError(f'{uid}: image size changes within the scene')
        crop = this
        Image.fromarray(img).save(tmp / 'rgb' / f'{frame}.{ext}', **({} if args.png else {'quality': args.quality}))
        got[kind].add(frame)
    for kind, frame, data in _members(depth_tars, wanted & got['color']):
        raw = np.array(Image.open(io.BytesIO(data)))
        z, _ = decode_depth(raw, scale)
        if sampler is None:
            sampler = _depth_sampler(crop, raw.shape, args.image_height, args.image_width)
        np.save(tmp / 'depth' / f'{frame}.npy',
                z.reshape(-1)[sampler].reshape(args.image_height, args.image_width).astype(np.float16))
        got[kind].add(frame)
    both = got['color'] & got['depth']
    for frame in got['color'] - both:
        (tmp / 'rgb' / f'{frame}.{ext}').unlink()
    for i, (frames, keep) in plans.items():
        mine = sorted(keep & both)
        if not mine:
            continue
        cam = cams[i]
        pos = [frames.index(f) for f in mine]
        focal = np.broadcast_to(np.asarray(cam['focals'], np.float64), (len(frames),))[pos]
        cx, cy = (float(np.asarray(cam[c], np.float64).ravel()[0]) - .5 for c in ('cx', 'cy'))  # centres to integers
        k = np.stack([adjust_intrinsics([[fc, 0, cx], [0, fc, cy], [0, 0, 1]], *crop) for fc in focal])
        w2c = np.tile(np.eye(4), (len(pos), 1, 1))
        w2c[:, :3, :3] = _quat_matrix(np.asarray(cam['quats'], np.float64)[pos])
        w2c[:, :3, 3] = np.asarray(cam['trans'], np.float64)[pos]
        c2w = np.linalg.inv(w2c)
        c2w[:, :3, 3] *= scale
        row = {f: j for j, f in enumerate(mine)}
        steps = [a for a, _ in anchors(frames[0], frames[-1], fps, args.step_dt, args.tube_dt, args.tubelet)]
        pairs = [(a, b) for a, b in zip(steps, steps[1:]) if a in row and b in row]
        depth = lambda f: np.load(tmp / 'depth' / f'{f}.npy')   # noqa: E731
        rel = [depth_consistency(depth(a), k[row[a]], c2w[row[a]], depth(b), k[row[b]], c2w[row[b]])[0]
               for a, b in pairs]
        np.savez(tmp / f'split_{i}.npz', frames=np.asarray(mine, np.int64), origin=np.int64(frames[0]), K=k, c2w=c2w,
                 pair_from=np.asarray([a for a, _ in pairs], np.int64), pair_absrel=np.asarray(rel, np.float64))
    (tmp / 'scene.json').write_text(json.dumps({'uid': uid, 'fps': fps, 'metric_scale': scale}))
    tmp.rename(final)
    return len(both)


def _settings(args):
    return dict(source=SOURCE, image_height=args.image_height, image_width=args.image_width, tubelet=args.tubelet,
                step_dt=args.step_dt, tube_dt=args.tube_dt, image_ext='png' if args.png else 'jpg',
                val_frac=args.val_frac, max_depth=MAX_DEPTH)


def _write_info(d, args, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    path, info = d / PACK_INFO, _settings(args)
    if path.exists():
        old = json.loads(path.read_text())
        bad = {key: (old.get(key), val) for key, val in info.items() if old.get(key) != val}
        if bad:
            raise SystemExit(f'{d} holds a pack with other settings (packed, now): {bad}')
        info = {**old, **info}
    path.write_text(json.dumps({**info, **(extra or {})}, indent=1))


def _job(job):
    src, uid, fps, scale, dst, args = job
    t0 = time.time()
    n = pack_scene(src, uid, fps, scale, dst, args)
    return uid, n, time.time() - t0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Pack OmniWorld-Game scenes for v6.')
    p.add_argument('--data-root', required=True, help='OmniWorld root (metadata/, videos/, annotations/)')
    p.add_argument('--out', required=True)
    p.add_argument('--val-frac', type=float, default=.1, help='share of scenes in val (by uid hash)')
    p.add_argument('--scenes', default='', help='comma-separated uids (default: every scene with an others tar)')
    p.add_argument('--limit', type=int, default=0, help='first N scenes only (pilot)')
    p.add_argument('--image-height', type=int, default=192)
    p.add_argument('--image-width', type=int, default=256)
    p.add_argument('--tubelet', type=int, default=2)
    p.add_argument('--step-dt', type=float, default=.5)
    p.add_argument('--tube-dt', type=float, default=.1)
    p.add_argument('--png', action='store_true', help='lossless images (default JPEG)')
    p.add_argument('--quality', type=int, default=95)
    p.add_argument('--workers', type=int, default=8, help='0: in this process')
    p.add_argument('--check', action='store_true', help='print the reprojection check per packed scene')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    src, out = Path(args.data_root).expanduser(), Path(args.out).expanduser()
    with open(src / 'metadata' / 'omniworld_game_metadata.csv', newline='') as f:
        meta = {r['UID']: r for r in csv.DictReader(f)}
    uids = [u.strip() for u in args.scenes.split(',') if u.strip()] or sorted(
        p.name for p in (src / 'annotations' / SUB).iterdir() if (p / f'{p.name}_others.tar.gz').exists())
    uids = [u for u in uids if u in meta and list((src / 'videos' / SUB / u).glob(f'{u}_rgb_*.tar.gz'))]
    if args.limit:
        uids = uids[:args.limit]
    for split in ('train', 'val'):
        _write_info(out / split, args)
    jobs = [(src, u, float(meta[u]['FPS']), float(meta[u]['Metric Scale']), out / scene_split(u, args.val_frac), args)
            for u in uids]
    done = []
    if args.workers > 0:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(_job, jobs):
                done.append(r)
                print(json.dumps({'scene': r[0], 'frames': r[1], 's': round(r[2], 1)}), flush=True)
    else:
        done = [_job(j) for j in jobs]
    for split in ('train', 'val'):
        d = out / split
        scenes = sorted(p.parent.name for p in d.glob('*/scene.json'))
        _write_info(d, args, {'scenes': len(scenes), 'frames': sum(len(list((d / u / 'depth').iterdir())) for u in scenes)})
    if args.check:
        cfg = _cfg(args)
        for split in ('train', 'val'):
            ds = OmniWorldClipDataset(out, split, cfg, step_dt=args.step_dt, tube_dt=args.tube_dt)
            first = {}
            for i, (si, _, _) in enumerate(ds.index):
                first.setdefault(si, i)
            for si, i in first.items():
                rel, n = reprojection_absrel(ds[i], 0, 1)
                print(json.dumps({'check': ds.scenes[si]['uid'], 'split': split, 'absrel_median': rel, 'pixels': n}),
                      flush=True)
    return done


def _cfg(args):
    from .config import V6Config
    return V6Config(image_height=args.image_height, image_width=args.image_width, tubelet=args.tubelet)


if __name__ == '__main__':
    main()
