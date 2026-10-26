#!/usr/bin/env python3
"""Train the C-D 3.2 tokenizer (src/mavt/cd32) with the v0 trainer: same flags, data, DDP, checkpoints.

  python scripts/train_cd32.py --dry-run --out /tmp/x                        # CPU, tiny config, 3 steps
  torchrun --nproc_per_node 2 scripts/train_cd32.py --dry-run --mix-modalities --out /tmp/x
  CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node N scripts/train_cd32.py \\
      --config configs/cd32/base.yaml --imagenet <parquet root> --openvid <manifest> --out <run dir>

The yaml holds CD32Config fields at top level and trainer defaults under `train:` (CLI wins). Video clips
must hold a multiple of keyframe_every frames (--n-frames 8 = 2 segments of K = 4). Staged training (design
option A): first images only (block C + decoder), then add video.

--set KEY=VALUE (repeatable) overrides a CD32Config field after the yaml, also in --dry-run, e.g.
  --set depth_probe=<probe.pt from scripts/train_depth_probe.py> --set w_depth_prior=0.05
so machine paths stay out of the committed yaml. Values are parsed as yaml (numbers, lists, null).
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


TUPLE_FIELDS = ("dino_layers", "rope_dims", "conv_channels", "depth_prior_on")


def parse_sets(items) -> dict:
    import yaml
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set needs KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def make_config(model_cfg: dict, dry: bool, sets: dict = None) -> CD32Config:
    fields = {f.name for f in dataclasses.fields(CD32Config)}
    bad = [k for k in list(model_cfg) + list(sets or {}) if k not in fields]
    if bad:
        raise SystemExit(f"unknown CD32Config fields: {bad}")
    if dry:
        return tiny_config(**_tuples(dict(sets or {})))
    return CD32Config(**_tuples({**model_cfg, **(sets or {})}))


def _tuples(d: dict) -> dict:
    for k in TUPLE_FIELDS:                                          # yaml lists -> tuples
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return d


def make_model(cfg: CD32Config, args):
    from mavt.cd32.model import MAVTCD32
    return MAVTCD32(cfg, use_checkpoint=args.use_checkpoint)


def main(argv=None) -> int:
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ours, rest = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    sets = parse_sets(ours.set)
    return base.main(rest, config_fn=lambda m, dry: make_config(m, dry, sets), model_fn=make_model)


if __name__ == "__main__":
    sys.exit(main())
