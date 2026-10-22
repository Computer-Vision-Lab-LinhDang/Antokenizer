#!/usr/bin/env python3
"""Evaluate a MAVT-on-DINOv3 v0 checkpoint: reconstruction (ImageNet val, OpenVid held-out tail) and
understanding (kNN + linear probe on ImageNet for z_s, pooled z_r(C) and the frozen DINOv3 CLS).

Example (alpha, one GPU):
  PYTHONPATH=src python3 scripts/eval_dino_v0.py --ckpt runs/dino_v0/last.pt --device cuda --amp \
      --imagenet-root /scratch/vunguyen13/data/imagenet-1k \
      --openvid-manifest /scratch/vunguyen13/data/dino_v0_data/openvid1m --video-limit 500 --video-frames 16 \
      --out runs/dino_v0/eval

Writes <out>/eval.json and <out>/eval.md. SSv2 / Kinetics probes run only when --ssv2 / --kinetics are given.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mavt.dino_v0 import evaluate as E  # noqa: E402

try:
    from mavt.dino_v0.data import ImageNetParquet, OpenVidClips  # noqa: E402
except ImportError as _ex:  # the data module is written in parallel; fail only when it is actually needed
    ImageNetParquet = OpenVidClips = None
    _DATA_IMPORT_ERROR = _ex
else:
    _DATA_IMPORT_ERROR = None


def _need_data():
    if _DATA_IMPORT_ERROR is not None:
        raise SystemExit(f"mavt.dino_v0.data is not importable ({_DATA_IMPORT_ERROR}); cannot read datasets")


def imagenet_labels(ds, root: str, split: str):
    """Labels of every item of ds without decoding images: ds.labels, else the parquet label column."""
    for attr in ("labels", "targets"):
        lab = getattr(ds, attr, None)
        if lab is not None:
            return torch.as_tensor(list(lab) if not torch.is_tensor(lab) else lab).long()
    files = sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet"))) or \
        sorted(glob.glob(os.path.join(root, f"{split}-*.parquet")))
    if files:
        import pyarrow.parquet as pq
        lab = torch.cat([torch.as_tensor(pq.read_table(f, columns=["label"]).column("label").to_numpy()) for f in files])
        if len(lab) == len(ds):
            return lab.long()
    return None


def group_ids(ds, indices):
    """Parquet row group of each index of an ImageNetParquet (falls back to one group per 1000 rows)."""
    starts = getattr(ds, "_starts", None)
    if starts is None:
        return [i // 1000 for i in indices]
    return [bisect.bisect_right(starts, i) - 1 for i in indices]


def grouped_loader(ds, indices, batch_size, workers):
    batches = E.grouped_batches(indices, group_ids(ds, indices), batch_size, workers)
    if hasattr(ds, "cache_groups"):
        ds.cache_groups = max(ds.cache_groups, 2)
    return torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=workers,
                                       pin_memory=torch.cuda.is_available())


def seeded_clips(manifest, args, size):
    ds = OpenVidClips(manifest, args.video_frames, args.frame_stride, size, train=False)
    ds._rng = random.Random(args.seed)                   # clip start is drawn from ds._rng: make it repeatable
    return ds


def openvid_heldout(manifest: str, n_tail: int, out_dir: str) -> str:
    """The last n_tail lines of the OpenVid manifest (dir -> its last *.jsonl by name) as a new jsonl."""
    if os.path.isdir(manifest):
        files = sorted(glob.glob(os.path.join(manifest, "*.jsonl")))
        if not files:
            raise SystemExit(f"no *.jsonl manifest in {manifest}")
        held = [f for f in files if any(s in os.path.basename(f) for s in ("val", "test", "heldout"))]
        manifest = (held or files)[-1]
    with open(manifest) as f:
        lines = [l for l in f if l.strip()]
    tail = lines[-n_tail:] if n_tail > 0 else lines
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "openvid_heldout.jsonl")
    with open(path, "w") as f:
        f.writelines(tail)
    print(f"[data] OpenVid held-out: last {len(tail)} of {len(lines)} lines of {manifest} -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")
    ap.add_argument("--out", required=True, help="output directory (eval.json, eval.md)")
    ap.add_argument("--stem", default="eval")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--video-batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    # images
    ap.add_argument("--imagenet-root", default=None)
    ap.add_argument("--limit", type=int, default=50000, help="ImageNet val images (recon + probe val)")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--no-understanding", action="store_true")
    ap.add_argument("--knn-train", type=int, default=100, help="train images per class for kNN / linear probe")
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-t", type=float, default=0.07)
    ap.add_argument("--probe-epochs", type=int, default=10)
    ap.add_argument("--probe-lrs", default="1e-3,3e-3,1e-2,3e-2")
    # video
    ap.add_argument("--openvid-manifest", default=None, help="jsonl or a directory holding it")
    ap.add_argument("--video-tail", type=int, default=1000, help="held-out manifest tail size")
    ap.add_argument("--video-limit", type=int, default=500)
    ap.add_argument("--video-frames", type=int, default=16, choices=[8, 16])
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--ssv2", default=None, help="SSv2 manifest jsonl (OpenVid format); skipped if absent")
    ap.add_argument("--kinetics", default=None, help="Kinetics manifest jsonl (OpenVid format); skipped if absent")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, cfg, step = E.load_model(args.ckpt, args.device)
    print(f"[ckpt] {args.ckpt} step={step} image_size={cfg.image_size} budget(image)={cfg.budget('image')} "
          f"budget(video seg)={cfg.budget('video')}", flush=True)
    size = cfg.image_size

    image_val = image_train = video = None
    extra = {}
    if args.imagenet_root:
        _need_data()
        val_ds = ImageNetParquet(args.imagenet_root, "validation", size, train=False)
        n_val = min(args.limit or len(val_ds), len(val_ds))
        val_idx = list(range(n_val))                     # HF val shards are class-mixed; first n rows
        image_val = grouped_loader(val_ds, val_idx, args.batch_size, args.workers)
        if not args.no_understanding:
            train_ds = ImageNetParquet(args.imagenet_root, "train", size, train=False)
            lab = imagenet_labels(train_ds, args.imagenet_root, "train")
            if lab is not None:
                idx = E.balanced_by_groups(lab, group_ids(train_ds, range(len(train_ds))), args.knn_train, args.seed)
            else:
                print("[data] train labels unavailable; using a random (not class-balanced) subset", flush=True)
                g = torch.Generator().manual_seed(args.seed)
                idx = torch.randperm(len(train_ds), generator=g)[: args.knn_train * 1000].sort().values
            image_train = grouped_loader(train_ds, idx.tolist(), args.batch_size, args.workers)
            print(f"[data] probe train subset: {len(idx)} images from "
                  f"{len(set(group_ids(train_ds, idx.tolist())))} row groups", flush=True)
    else:
        print("[data] --imagenet-root not given: image reconstruction and understanding skipped", flush=True)
    if args.openvid_manifest:
        _need_data()
        held = openvid_heldout(args.openvid_manifest, args.video_tail, args.out)
        video = seeded_clips(held, args, size)
    for name, path in (("ssv2", args.ssv2), ("kinetics", args.kinetics)):
        if path:
            _need_data()
            extra[name] = seeded_clips(path, args, size)
        else:
            extra[name] = None

    res = E.run_eval(model, args.device, image_val=image_val, image_train=image_train, video=video,
                     batch_size=args.batch_size, video_batch_size=args.video_batch_size, num_workers=args.workers,
                     amp=args.amp, limit=args.limit, video_limit=args.video_limit, knn_k=args.knn_k,
                     knn_t=args.knn_t, num_classes=1000 if image_train is not None else None,
                     probe_epochs=args.probe_epochs, probe_lrs=[float(x) for x in args.probe_lrs.split(",")],
                     use_lpips=not args.no_lpips, use_fid=not args.no_fid, extra_video=extra)
    res["meta"] = dict(ckpt=os.path.abspath(args.ckpt), step=step, amp=args.amp, device=args.device,
                       video_frames=args.video_frames, frame_stride=args.frame_stride, cfg=cfg.__dict__)
    jp, mp = E.write_results(res, args.out, args.stem)
    print(E.to_markdown({k: v for k, v in res.items() if k != "meta"}))
    print(f"[done] {jp} {mp}", flush=True)


if __name__ == "__main__":
    main()
