#!/usr/bin/env python3
"""Depth of the C-D 3.2 tokenizer: how well rho (log inverse depth per cell of C) matches real depth.

  python scripts/eval_cd32_depth.py --ckpt <run>/last.pt --depth <val manifests> [--probe <probe.pt>] \\
      [--objaverse <render root> --n-objects 200] --out <json>
  python scripts/eval_cd32_depth.py --dry-run --out /tmp/x.json        # CPU, untrained tiny model

Images (depth manifests, src/mavt/cd32/depth_data.py): rho of the image's C, median-scaled per image (still
images have no scale), per cell and per pixel (bilinear), next to the frozen probe and a constant depth.
Posed views (Objaverse hold-out, rendered depth): rho of view 0, median-scaled and as is (poses fix the
scale, so rho should be metric there). Metrics: AbsRel, SqRel, RMSE, RMSE log, SILog, delta < 1.25^k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mavt.cd32.depth import (apply_probe, cell_log_inv_depth, depth_metrics, load_probe, mean_metrics,  # noqa: E402
                             merge_metrics, upsample_rho)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="cd32 checkpoint (last.pt); without it only the probe is scored")
    p.add_argument("--probe", default=None, help="depth probe file (scripts/train_depth_probe.py)")
    p.add_argument("--dino-repo", default=None, help="override the checkpoint's dino_repo (local copy)")
    p.add_argument("--depth", default=None, help="comma-separated depth manifests")
    p.add_argument("--limit", type=int, default=0, help="first N rows per manifest (0: all)")
    p.add_argument("--objaverse", default=None, help="Objaverse render root with *_depth.png")
    p.add_argument("--objaverse-holdout", type=int, default=500)
    p.add_argument("--n-objects", type=int, default=200)
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


class _Acc:
    def __init__(self):
        self.d = {}

    def add(self, name: str, m: dict) -> None:
        self.d[name] = merge_metrics(self.d.get(name), m)

    def result(self) -> dict:
        return {k: mean_metrics(v) for k, v in self.d.items()}


def _autocast(dev, on: bool):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=on)


@torch.no_grad()
def eval_images(model, probe, loader, cfg, dev, amp: bool) -> dict:
    acc, G, S = _Acc(), cfg.grid, cfg.image_size
    for b in loader:
        x, depth = b["data"].to(dev), b["depth"].to(dev)
        tgt, mask = cell_log_inv_depth(depth, depth > 0, G, b["min_cell_frac"].float())
        cell_d = torch.where(mask, torch.exp(-tgt), torch.zeros_like(tgt))
        valid = depth > 0
        preds = {"const": torch.zeros(x.shape[0], G * G, device=dev)}
        with _autocast(dev, amp):
            if model is not None:
                preds["rho"] = model(x, "image", return_outputs=True)["outputs"]["rho"].float()
            if probe is not None:
                _, layers, _ = (model.encoder.dino if model is not None else probe["dino"])(x)
                preds["probe"] = apply_probe([layers[i] for i in probe["idx"]], probe["weight"],
                                             probe["bias"]).float()
        for k, r in preds.items():
            acc.add(f"{k}_cell", depth_metrics(r, cell_d, mask))
            acc.add(f"{k}_pixel", depth_metrics(upsample_rho(r, G, S), depth, valid))
    return acc.result()


@torch.no_grad()
def eval_objaverse(model, loader, cfg, dev, amp: bool, n_objects: int) -> dict:
    acc, G, S, seen = _Acc(), cfg.grid, cfg.image_size, 0
    for b in loader:
        x = b["data"].to(dev)
        cam = dict(c2w=b["c2w"].to(dev), K=b["K"].to(dev), depth=b["depth"].to(dev))
        depth = cam["depth"][:, 0]                                    # view 0 stays first in view_order
        tgt, mask = cell_log_inv_depth(depth, depth > 0, G)
        cell_d = torch.where(mask, torch.exp(-tgt), torch.zeros_like(tgt))
        with _autocast(dev, amp):
            rho = model(x, "threed", cam, return_outputs=True)["outputs"]["rho"].float()
        for align in ("median", "none"):
            acc.add(f"rho_cell_{align}", depth_metrics(rho, cell_d, mask, align=align))
            acc.add(f"rho_pixel_{align}", depth_metrics(upsample_rho(rho, G, S), depth, depth > 0, align=align))
        seen += x.shape[0]
        if seen >= n_objects:
            break
    return acc.result()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dev = torch.device("cpu" if args.dry_run else args.device)
    amp = args.bf16 and not args.dry_run
    model, cfg, step = None, None, -1
    if args.dry_run:
        from mavt.cd32.config import tiny_config
        from mavt.cd32.model import MAVTCD32
        cfg = tiny_config()
        model = MAVTCD32(cfg).eval()
        sys.path.insert(0, HERE)
        from train_depth_probe import make_synthetic
        args.depth = make_synthetic(os.path.join(os.path.dirname(os.path.abspath(args.out)), "depth_eval_syn"), 4,
                                    cfg.image_size)
        args.num_workers, args.batch = 0, 2
    elif args.ckpt:
        from mavt.cd32.model import load_model
        kw = {"dino_repo": args.dino_repo} if args.dino_repo else {}
        model, cfg, step = load_model(args.ckpt, dev, **kw)
    probe = None
    if args.probe:
        ck = load_probe(args.probe)
        if model is None:
            from mavt.cd32.config import CD32Config
            from mavt.cd32.encoder import MultiLayerDINO
            cfg = CD32Config(dino_repo=args.dino_repo or ck["dino_repo"], dino_layers=tuple(ck["layers"]),
                             image_size=int(ck["image_size"]), patch=int(ck["patch"]))
            probe = dict(dino=MultiLayerDINO(cfg).to(dev))
            idx = list(range(len(ck["layers"])))
        else:
            idx = [list(cfg.dino_layers).index(int(l)) for l in ck["layers"]]
            probe = {}
        probe.update(idx=idx, weight=ck["weight"].float().to(dev), bias=ck["bias"].float().to(dev))
    if model is None and probe is None:
        raise SystemExit("give --ckpt and/or --probe (or --dry-run)")
    res = dict(ckpt=args.ckpt, step=step, probe=args.probe)
    if args.depth:
        from mavt.cd32.depth_data import depth_datasets
        ds = depth_datasets(args.depth.split(","), cfg.image_size, train=False, limit=args.limit)
        ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        res["images"] = eval_images(model, probe, ld, cfg, dev, amp)
        res["images"]["n_images"] = len(ds)
    if args.objaverse and model is not None:
        from mavt.dino_v0.data import ObjaverseViews, collate
        ds = ObjaverseViews(args.objaverse, n_views=args.n_views, size=cfg.image_size, train=False,
                            holdout=args.objaverse_holdout, depth=True)
        ld = DataLoader(ds, batch_size=min(args.batch, 16), shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate)
        res["objaverse"] = eval_objaverse(model, ld, cfg, dev, amp, args.n_objects)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
