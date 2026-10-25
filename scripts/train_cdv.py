#!/usr/bin/env python3
"""Train the vanilla C-D model (src/mavt/cd32/vanilla.py) with the v0 trainer: same flags, data, DDP, checkpoints.

  python scripts/train_cdv.py --dry-run --out /tmp/x                         # CPU, tiny config, 3 steps
  torchrun --nproc_per_node 2 scripts/train_cdv.py --dry-run --mix-modalities --out /tmp/x
  torchrun --standalone --nproc_per_node N scripts/train_cdv.py --config configs/cd32/vanilla.yaml \\
      --imagenet <parquet root> --out <stage 1 dir>                                      # images
  ... --imagenet <root> --openvid <manifest> --init <stage 1 dir>/last.pt --out <stage 2 dir>   # + video
  ... --objaverse <render root> --init <stage 2 dir>/last.pt --out <stage 3 dir>         # + posed views

The yaml holds CDVConfig fields at top level and trainer defaults under `train:` (CLI wins). --init loads
the model weights of another run (optimizer and schedule start fresh); --resume continues a run in place.
--set KEY=VALUE (repeatable) overrides a CDVConfig field, e.g. --set geometry=false (ablation),
--set w_lpips=0.5, --set depth_probe=<probe.pt> --set w_depth_prior=0.05.
"""
from __future__ import annotations

import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import train_dino_v0 as base  # noqa: E402
from train_cd32 import TUPLE_FIELDS, _tuples, parse_sets  # noqa: E402

from mavt.cd32.vanilla import CDVConfig, tiny_cdv_config  # noqa: E402


def make_config(model_cfg: dict, dry: bool, sets: dict = None) -> CDVConfig:
    fields = {f.name for f in dataclasses.fields(CDVConfig)}
    bad = [k for k in list(model_cfg) + list(sets or {}) if k not in fields]
    if bad:
        raise SystemExit(f"unknown CDVConfig fields: {bad}")
    if dry:
        return tiny_cdv_config(**_tuples(dict(sets or {})))
    return CDVConfig(**_tuples({**model_cfg, **(sets or {})}))


def make_model(cfg: CDVConfig, args):
    from mavt.cd32.vanilla_model import MAVTCDV
    return MAVTCDV(cfg, use_checkpoint=args.use_checkpoint)


def main(argv=None) -> int:
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ours, rest = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    sets = parse_sets(ours.set)
    return base.main(rest, config_fn=lambda m, dry: make_config(m, dry, sets), model_fn=make_model)


if __name__ == "__main__":
    sys.exit(main())
