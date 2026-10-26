#!/usr/bin/env python3
"""Train the C-D 4.1 tokenizer (src/mavt/cd41) with the v0 trainer: same flags, data, DDP, checkpoints.

  python scripts/train_cd41.py --dry-run --out /tmp/x                        # CPU, tiny config, 3 steps
  torchrun --nproc_per_node 2 scripts/train_cd41.py --dry-run --mix-modalities --out /tmp/x
  torchrun --standalone --nproc_per_node N scripts/train_cd41.py --config configs/cd41/base.yaml \\
      --imagenet <parquet root> --openvid <manifest> --objaverse <render root> \\
      --omniworld <pack> --omniworld-modality world --omniworld-scenes 30 --out <run>

The yaml holds CD41Config fields at top level and trainer defaults under `train:` (CLI wins).
--set KEY=VALUE (repeatable) overrides a CD41Config field, e.g. the arm without Δ-Tucker at about the same
payload: --set tucker=false --set n_slots=81 (12 170 scalars per segment against 12 194).
"""
from __future__ import annotations

import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import train_dino_v0 as base  # noqa: E402
from train_cd32 import parse_sets  # noqa: E402

from mavt.cd41.config import CD41Config, config_from_dict, tiny_cd41_config  # noqa: E402


def make_config(model_cfg: dict, dry: bool, sets: dict = None) -> CD41Config:
    fields = {f.name for f in dataclasses.fields(CD41Config)}
    bad = [k for k in list(model_cfg) + list(sets or {}) if k not in fields]
    if bad:
        raise SystemExit(f"unknown CD41Config fields: {bad}")
    if dry:
        return tiny_cd41_config(**{k: tuple(v) if isinstance(v, list) else v for k, v in (sets or {}).items()})
    return config_from_dict({**model_cfg, **(sets or {})})


def make_model(cfg: CD41Config, args):
    from mavt.cd41.model import MAVTCD41
    return MAVTCD41(cfg, use_checkpoint=args.use_checkpoint)


def main(argv=None) -> int:
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ours, rest = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    sets = parse_sets(ours.set)
    return base.main(rest, config_fn=lambda m, dry: make_config(m, dry, sets), model_fn=make_model)


if __name__ == "__main__":
    sys.exit(main())
