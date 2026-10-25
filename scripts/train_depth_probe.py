#!/usr/bin/env python3
"""Train the depth probe of the C-D 3.2 tokenizer: a linear map on frozen DINOv3 patch features to the log
inverse depth of each cell, on real depth labels (bản 3.2, L_depth_prior). One GPU is plenty.

  python scripts/train_depth_probe.py --dry-run --out /tmp/probe                      # CPU, stub DINO, synthetic
  python scripts/train_depth_probe.py --train <nyu_train.jsonl>,<kitti_train.jsonl> \\
      --val <nyu_test.jsonl> --out <dir> [--dino-repo <hf id or local copy>] [--steps 6000]

Manifests are described in src/mavt/cd32/depth_data.py. The probe sees exactly what the tokenizer sees:
256^2 inputs, the 16 x 16 cell grid, DINOv3 layers 6, 12, 18 and 24 (CD32Config.dino_layers). Loss on
cells: SILog (lambda 0.85) + w_ordinal x pairwise order loss. Writes <out>/probe.pt (pass it as the
tokenizer's depth_probe), <out>/log.jsonl and <out>/val.json (AbsRel, delta1, ... after median scaling,
per cell and per pixel, against a constant-depth baseline).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mavt.cd32.config import CD32Config  # noqa: E402
from mavt.cd32.depth import (DepthProbe, cell_log_inv_depth, cosine_lr, depth_metrics, mean_metrics,  # noqa: E402
                             merge_metrics, ordinal, silog, upsample_rho)
from mavt.cd32.depth_data import depth_datasets, worker_init  # noqa: E402
from mavt.cd32.encoder import MultiLayerDINO  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=None, help="comma-separated depth manifests (jsonl)")
    p.add_argument("--val", default=None, help="comma-separated depth manifests for validation")
    p.add_argument("--out", required=True)
    p.add_argument("--dino-repo", default=CD32Config.dino_repo)
    p.add_argument("--layers", default="6,12,18,24", help="DINOv3 blocks read by the probe (1-based)")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--registers", type=int, default=CD32Config.dino_registers)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--silog-lam", type=float, default=0.85)
    p.add_argument("--w-ordinal", type=float, default=0.1)
    p.add_argument("--min-cell-frac", type=float, default=0.5,
                   help="valid pixel share for a cell to count (a manifest row's min_cell_frac wins)")
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-limit", type=int, default=0, help="first N rows of each val manifest (0: all)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True, help="DINOv3 forward in bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="CPU, stub DINO, synthetic manifests, 6 steps")
    return p


def make_synthetic(root: str, n: int, size: int = 64, seed: int = 0) -> str:
    """Tiny depth set for --dry-run and tests: a tilted plane with a box; colour follows depth."""
    from PIL import Image
    os.makedirs(os.path.join(root, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(root, "depth"), exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    rows = []
    for i in range(n):
        d = 1.0 + 3.0 * yy[::-1] * rng.uniform(0.5, 1.5) + 0.5 * xx
        x0, y0 = rng.integers(0, size // 2, 2)
        d[y0:y0 + size // 4, x0:x0 + size // 4] *= 0.4
        d[: size // 10] = 0                                      # some invalid pixels (sky)
        g = np.clip(255 / np.maximum(d, 0.5), 0, 255).astype(np.uint8)
        Image.fromarray(np.stack([g, 255 - g, g // 2], -1)).save(os.path.join(root, "rgb", f"{i:04d}.png"))
        Image.fromarray((d * 1000).astype(np.uint16)).save(os.path.join(root, "depth", f"{i:04d}.png"))
        rows.append(dict(image=f"rgb/{i:04d}.png", depth=f"depth/{i:04d}.png", scale=0.001))
    path = os.path.join(root, "manifest.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def dino_config(args) -> CD32Config:
    layers = tuple(int(l) for l in args.layers.split(","))
    if args.dry_run:
        return CD32Config(dino_repo=None, image_size=args.size, patch=args.patch, stub_dino_dim=32,
                          stub_dino_blocks=max(layers), dino_layers=layers, dino_registers=args.registers)
    return CD32Config(dino_repo=args.dino_repo, image_size=args.size, patch=args.patch, dino_layers=layers,
                      dino_registers=args.registers)


def cell_targets(b: dict, depth: torch.Tensor, grid: int):
    return cell_log_inv_depth(depth, depth > 0, grid, b["min_cell_frac"].float())


@torch.no_grad()
def evaluate(dino, probe, loader, grid: int, size: int, dev, amp: bool) -> dict:
    probe.eval()
    acc = {k: None for k in ("probe_cell", "probe_pixel", "const_pixel")}
    for b in loader:
        x, depth = b["data"].to(dev, non_blocking=True), b["depth"].to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            _, layers, _ = dino(x)
        rho = probe(layers)
        tgt, mask = cell_targets(b, depth, grid)
        cell_depth = torch.where(mask, torch.exp(-tgt), torch.zeros_like(tgt))
        acc["probe_cell"] = merge_metrics(acc["probe_cell"], depth_metrics(rho, cell_depth, mask))
        pix = upsample_rho(rho, grid, size)
        acc["probe_pixel"] = merge_metrics(acc["probe_pixel"], depth_metrics(pix, depth, depth > 0))
        acc["const_pixel"] = merge_metrics(acc["const_pixel"],
                                           depth_metrics(torch.zeros_like(pix), depth, depth > 0))
    probe.train()
    return {k: mean_metrics(v) for k, v in acc.items() if v is not None}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.dry_run:
        args.device, args.size, args.steps, args.batch, args.val_every = "cpu", 64, 6, 4, 3
        args.num_workers, args.bf16, args.warmup = 0, False, 2
        syn = os.path.join(args.out, "synthetic")
        args.train = make_synthetic(os.path.join(syn, "train"), 12, args.size, 0)
        args.val = make_synthetic(os.path.join(syn, "val"), 4, args.size, 1)
    if not args.train:
        raise SystemExit("--train is required (or --dry-run)")
    dev = torch.device(args.device)
    cfg = dino_config(args)
    grid = args.size // args.patch
    dino = MultiLayerDINO(cfg).to(dev)
    probe = DepthProbe(dino.out_dim, cfg.dino_layers).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, args.steps, args.warmup))

    train_set = depth_datasets(args.train.split(","), args.size, train=True, min_cell_frac=args.min_cell_frac)
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=len(train_set) >= args.batch,
                        num_workers=args.num_workers, pin_memory=dev.type == "cuda", worker_init_fn=worker_init,
                        persistent_workers=args.num_workers > 0)
    val_loader = None
    if args.val:
        val_set = depth_datasets(args.val.split(","), args.size, train=False, limit=args.val_limit,
                                 min_cell_frac=args.min_cell_frac)
        val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                                pin_memory=dev.type == "cuda")
    meta = dict(dino_repo=cfg.dino_repo, image_size=args.size, patch=args.patch, train_sets=args.train,
                val_sets=args.val, silog_lam=args.silog_lam, w_ordinal=args.w_ordinal)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"depth probe: {len(train_set)} train images, DINO layers {cfg.dino_layers}, dim {dino.out_dim}, "
          f"{args.steps} steps on {dev}", flush=True)
    step, best, t0 = 0, math.inf, time.time()
    results = {}
    it = iter(loader)
    while step < args.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        x, depth = b["data"].to(dev, non_blocking=True), b["depth"].to(dev, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=args.bf16):
            _, layers, _ = dino(x)
        tgt, mask = cell_targets(b, depth, grid)
        rho = probe(layers)
        ls = silog(rho, tgt, mask, args.silog_lam)
        lo = ordinal(rho, tgt, mask) if args.w_ordinal > 0 else torch.zeros((), device=dev)
        loss = ls + args.w_ordinal * lo
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0 or step == 1 or args.dry_run:
            rec = dict(step=step, loss=loss.item(), silog=ls.item(), ordinal=lo.item(), lr=sched.get_last_lr()[0],
                       valid_cells=mask.float().mean().item(), sec=round(time.time() - t0, 1))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if val_loader is not None and (step % args.val_every == 0 or step == args.steps):
            results = evaluate(dino, probe, val_loader, grid, args.size, dev, args.bf16)
            results["step"] = step
            log.write(json.dumps(dict(step=step, val=results)) + "\n")
            log.flush()
            print("val", json.dumps(results), flush=True)
            if results["probe_pixel"]["abs_rel"] < best:
                best = results["probe_pixel"]["abs_rel"]
                torch.save(probe.export(step=step, val=results, **meta), os.path.join(args.out, "probe_best.pt"))
    torch.save(probe.export(step=step, val=results, **meta), os.path.join(args.out, "probe.pt"))
    with open(os.path.join(args.out, "val.json"), "w") as f:
        json.dump(results, f, indent=1)
    log.close()
    print(f"saved {os.path.join(args.out, 'probe.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
