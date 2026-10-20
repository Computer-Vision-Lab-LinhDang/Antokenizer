"""Pack an AV2 Sensor split for v6 training on a machine without the raw data or pyarrow.

    PYTHONPATH=src python -m mavt4d.v6.pack_av2 --data-root ~/data/argoverse2/sensor --split train \
        --out ~/data/av2-v6-pack --workers 32

Only what the split's clips use is written: for each sweep a clip reads, the tube frames of every ring camera,
resized and cropped to the target size, and the sweep's LiDAR already projected into each of those cameras (nearest
point per pixel, ego-motion compensated, as Av2ClipDataset does it). Calibration, poses and annotations become
.npz. Read it back with PackedAv2ClipDataset (or data_av2.open_av2, which picks it from pack.json); the clips and
tensors match Av2ClipDataset on the source, except that JPEG re-encoding changes pixel values slightly (--png
keeps them exact). Logs are written to <log>.tmp and renamed when complete, so a rerun skips finished logs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from .config import V6Config
from .data_av2 import PACK_INFO, Av2ClipDataset, _se3, adjust_intrinsics

TABLES = ('calibration/intrinsics', 'calibration/egovehicle_SE3_sensor', 'city_SE3_egovehicle', 'annotations')

_DS = None   # the source dataset, inherited by forked workers
_ARGS = None


def _save_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {}
    for name, col in table.items():
        col = np.asarray(col)
        cols[name] = col.astype(str) if col.dtype == object else col   # .npz without pickles
    np.savez(path, **cols)


def pack_log(ds, li, sweeps, out, image_ext='jpg', quality=95):
    """Write log li of ds (only the sweep indices `sweeps`) under out/<log_id>. Returns the number of images."""
    log_id = ds.log_ids[li]
    src, dst, tmp = ds.dir / log_id, out / log_id, out / f'{log_id}.tmp'
    if dst.exists():
        return 0
    shutil.rmtree(tmp, ignore_errors=True)
    for rel in TABLES:
        table = ds._table(src / f'{rel}.feather')
        if table is not None:
            _save_table(tmp / f'{rel}.npz', table)
    meta = ds._log_meta(li)
    lidar_ts = ds.lidar_ts[li][np.asarray(sweeps, dtype=np.int64)]
    city_T_ego = _se3(*ds._pose(meta, lidar_ts))
    crops, written = {}, set()
    (tmp / 'sensors' / 'lidar').mkdir(parents=True)
    for cam in meta['cams']:
        (tmp / 'sensors' / 'cameras' / cam).mkdir(parents=True)
        np.save(tmp / 'sensors' / 'cameras' / cam / 'ts.npy', meta['cam_ts'][cam])
    for s, lidar_t in enumerate(lidar_ts.tolist()):
        sweep, proj = {}, {}
        for cam in meta['cams']:
            frames = ds._camera_frames(meta, cam, lidar_t)
            if frames is None:
                continue
            d = tmp / 'sensors' / 'cameras' / cam
            for ts in frames:
                img, crop = ds._image(src, cam, ts)
                if crops.setdefault(cam, crop) != crop:
                    raise ValueError(f'{log_id} {cam}: image size changes within the log')
                if (cam, ts) not in written:
                    from PIL import Image
                    opts = dict(quality=quality) if image_ext == 'jpg' else {}
                    Image.fromarray(img).save(d / f'{ts}.{image_ext}', **opts)
                    written.add((cam, ts))
            k = adjust_intrinsics(meta['k'][cam], *crops[cam])
            flat, z = ds._lidar_in_camera(src, lidar_t, cam, ds._cam_T_lidar(meta, cam, frames[-1], city_T_ego[s]),
                                          k, sweep)
            proj[f'{cam}_flat'], proj[f'{cam}_z'] = flat.astype(np.int32), z.astype(np.float32)
        np.savez(tmp / 'sensors' / 'lidar' / f'{lidar_t}.npz', **proj)
    for cam, crop in crops.items():
        np.save(tmp / 'sensors' / 'cameras' / cam / 'crop.npy', np.asarray(crop, dtype=np.float64))
    np.save(tmp / 'lidar_ts.npy', ds.lidar_ts[li])   # every sweep, so the clip index is rebuilt unchanged
    tmp.rename(dst)
    return len(written)


def _work(job):
    li, sweeps = job
    t0 = time.time()
    n = pack_log(_DS, li, sweeps, _ARGS['out'], _ARGS['image_ext'], _ARGS['quality'])
    return _DS.log_ids[li], n, time.time() - t0


def pack_split(root, split, out, cfg, steps=6, step_dt=0.5, tube_dt=0.1, workers=8, image_ext='jpg', quality=95,
               log_fraction=1.0, max_logs=0):
    """Pack root/split into out/split. Returns {log_id: images written} for the logs written by this call."""
    global _DS, _ARGS
    ds = Av2ClipDataset(root, split, cfg, steps=steps, step_dt=step_dt, tube_dt=tube_dt, log_fraction=log_fraction)
    sweeps = {}
    for li, j in ds.index:
        sweeps.setdefault(li, set()).update(int(x) for x in j)
    jobs = [(li, sorted(js)) for li, js in sorted(sweeps.items())]
    if max_logs:
        jobs = jobs[:max_logs]
    dest = Path(out) / split if split else Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    info = dict(format=1, image_height=cfg.image_height, image_width=cfg.image_width, tubelet=cfg.tubelet,
                steps=int(steps), step_dt=float(step_dt), tube_dt=float(tube_dt), image_ext=image_ext)
    old = dest / PACK_INFO
    if old.exists() and json.loads(old.read_text()) != info:
        raise ValueError(f'{dest} holds a pack with other settings: {old.read_text()}')
    _DS, _ARGS = ds, dict(out=dest, image_ext=image_ext, quality=quality)
    done = {}
    if workers > 1:
        with mp.get_context('fork').Pool(workers) as pool:
            results = pool.imap_unordered(_work, jobs)
            for log_id, n, sec in results:
                done[log_id] = n
                print(json.dumps({'log': log_id, 'images': n, 's': round(sec, 1), 'done': len(done),
                                  'of': len(jobs)}), flush=True)
    else:
        for job in jobs:
            log_id, n, _ = _work(job)
            done[log_id] = n
    old.write_text(json.dumps(info, indent=1))   # last: a pack without it is never read
    return done


def main(argv=None):
    p = argparse.ArgumentParser(description='Pack an AV2 Sensor split for v6 (small images, projected LiDAR).')
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=6, help='must match training and evaluation')
    p.add_argument('--step-dt', type=float, default=.5)
    p.add_argument('--tube-dt', type=float, default=.1)
    p.add_argument('--config', default='{}', help='JSON V6Config overrides (image_height, image_width, tubelet)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--png', action='store_true', help='lossless images (about 3x larger)')
    p.add_argument('--quality', type=int, default=95, help='JPEG quality')
    p.add_argument('--log-fraction', type=float, default=1.0)
    p.add_argument('--max-logs', type=int, default=0, help='stop after this many logs (0: all), for a trial')
    args = p.parse_args(argv)
    cfg = V6Config(**json.loads(args.config))
    t0 = time.time()
    done = pack_split(Path(args.data_root).expanduser(), args.split, Path(args.out).expanduser(), cfg,
                      steps=args.steps, step_dt=args.step_dt, tube_dt=args.tube_dt, workers=args.workers,
                      image_ext='png' if args.png else 'jpg', quality=args.quality, log_fraction=args.log_fraction,
                      max_logs=args.max_logs)
    print(json.dumps({'logs_written': len(done), 'images': sum(done.values()), 'minutes': (time.time() - t0) / 60}))


if __name__ == '__main__':
    main()
