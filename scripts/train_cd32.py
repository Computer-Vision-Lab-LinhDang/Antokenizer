#!/usr/bin/env python3
"""Train the C-D 3.2 tokenizer (src/mavt/cd32) with the v0 trainer: same flags, data, DDP, checkpoints.

  python scripts/train_cd32.py --dry-run --out /tmp/x                        # CPU, tiny config, 3 steps
  torchrun --nproc_per_node 2 scripts/train_cd32.py --dry-run --mix-modalities --out /tmp/x
  CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node N scripts/train_cd32.py \\
      --config configs/cd32/base.yaml --imagenet <parquet root> --openvid <manifest> --out <run dir>

The yaml holds CD32Config fields at top level and trainer defaults under `train:` (CLI wins). Video clips
must hold a multiple of keyframe_every frames (--n-frames 8 = 2 segments of K = 4). Staged training (design
option A): first images only (block C + decoder), then add video.
"""
from __future__ import annotations

import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import train_dino_v0 as base  # noqa: E402

from mavt.cd32.config import CD32Config, tiny_config  # noqa: E402


def make_config(model_cfg: dict, dry: bool) -> CD32Config:
    if dry:
        return tiny_config()
    fields = {f.name for f in dataclasses.fields(CD32Config)}
    bad = [k for k in model_cfg if k not in fields]
    if bad:
        raise SystemExit(f"unknown CD32Config fields: {bad}")
    for k in ("dino_layers", "rope_dims", "conv_channels"):        # yaml lists -> tuples
        if k in model_cfg and isinstance(model_cfg[k], list):
            model_cfg[k] = tuple(model_cfg[k])
    return CD32Config(**model_cfg)


def make_model(cfg: CD32Config, args):
    from mavt.cd32.model import MAVTCD32
    return MAVTCD32(cfg, use_checkpoint=args.use_checkpoint)


def main(argv=None) -> int:
    return base.main(argv, config_fn=make_config, model_fn=make_model)


if __name__ == "__main__":
    sys.exit(main())
