#!/usr/bin/env python3
"""Prepare immutable CPU factorial source caches; safe beside a live DataLoader."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time
import uuid

import torch

from mavt4d.object_state.data import CachedFactorialDataset, FactorialSceneConfig


def _initialize_worker(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def _prepare_one(task):
    data_dir, count, config, split, seed, index = task
    dataset = CachedFactorialDataset(data_dir, count, FactorialSceneConfig(**config), split=split, base_seed=seed)
    path = dataset.cache_path(index)
    existed = path.exists()
    started = time.monotonic()
    dataset[index]  # validates existing payloads or publishes a complete new one
    return {"split": split, "index": index, "cached_before": existed,
            "bytes": path.stat().st_size, "seconds": time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--train-sources", type=int, default=4096)
    parser.add_argument("--dev-sources", type=int, default=64)
    parser.add_argument("--test-sources", type=int, default=64)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), default=["train", "dev", "test"])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=17)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--prefix-frames", type=int, default=6)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--observed-views", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--dt", type=float, default=.12)
    parser.add_argument("--gauge", choices=("first_camera", "world"), default="first_camera")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)
    if min(args.workers, args.torch_threads, args.progress_every) < 1:
        parser.error("workers, threads and progress interval must be positive")
    if len(set(args.splits)) != len(args.splits):
        parser.error("duplicate split names are not permitted")
    # This is CPU data preparation. Reduce priority, including on a shared headnode.
    if hasattr(os, "nice"):
        os.nice(max(0, 10 - os.nice(0)))
    config = FactorialSceneConfig(frames=args.frames, prefix_frames=args.prefix_frames,
                                  height=args.height, width=args.width, views=args.views,
                                  observed_views=args.observed_views, max_objects=args.max_objects, dt=args.dt,
                                  gauge=args.gauge)
    counts = {split: getattr(args, split + "_sources") for split in args.splits}
    if min(counts.values()) < 1:
        parser.error("each selected split needs a positive source count")
    configs = {split: CachedFactorialDataset(args.data_dir, count, config, split=split,
                                            base_seed=args.base_seed).fingerprint for split, count in counts.items()}
    tasks = [(str(args.data_dir), count, asdict(config), split, args.base_seed, index)
             for split, count in counts.items() for index in range(count)]
    started = time.monotonic()
    print(json.dumps({"event": "start", "sources": counts, "config": asdict(config),
                      "fingerprints": configs, "workers": args.workers, "torch_threads": args.torch_threads}), flush=True)
    total_bytes, reused = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_worker, initargs=(args.torch_threads,)) as pool:
        for number, result in enumerate(pool.map(_prepare_one, tasks, chunksize=1), 1):
            total_bytes += result["bytes"]
            reused += int(result["cached_before"])
            if number == 1 or number % args.progress_every == 0 or number == len(tasks):
                print(json.dumps({"event": "progress", "complete": number, "total": len(tasks),
                                  "bytes": total_bytes, "elapsed_seconds": time.monotonic() - started,
                                  "last": result}), flush=True)
    receipt = {"schema": "object-state-cache-preparation-v1", "finished_at": datetime.now(timezone.utc).isoformat(),
               "sources": counts, "config": asdict(config), "fingerprints": configs, "total_bytes": total_bytes,
               "reused_sources": reused, "elapsed_seconds": time.monotonic() - started}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.data_dir / ("preparation-" + uuid.uuid4().hex + ".json")
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"event": "complete", "receipt": str(receipt_path), **receipt}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
