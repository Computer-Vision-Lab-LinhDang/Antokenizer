#!/usr/bin/env python3
"""Train the Grain-Prior C-D v1 tokenizer (src/mavt/grain_prior) with the v0 trainer: same flags, data, DDP,
checkpoints.

  python scripts/train_grain_prior.py --dry-run --mix-modalities --out /tmp/x           # CPU, tiny config
  torchrun --standalone --nproc_per_node N scripts/train_grain_prior.py --config configs/grain_prior/base.yaml \\
      --imagenet <parquet root> --openvid <manifest> --objaverse <render root> --objaverse-depth \\
      --omniworld <pack> --omniworld-modality world --omniworld-scenes 30 --mix-modalities --out <run>

The yaml holds GrainPriorConfig fields at top level and trainer defaults under `train:` (CLI wins).
--set KEY=VALUE (repeatable) overrides a GrainPriorConfig field, e.g. the arms of section 15.1:
--set geom_bias=false (grain read without the geometry bias), --set query_init=uncond (learned queries),
--set memory=false (z_i = c_0). Phase T3 (causal prior, tokenizer frozen): --set phase=prior --init <ckpt>.
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

from mavt.grain_prior.config import GrainPriorConfig, config_from_dict, tiny_config  # noqa: E402


def make_config(model_cfg: dict, dry: bool, sets: dict = None) -> GrainPriorConfig:
    fields = {f.name for f in dataclasses.fields(GrainPriorConfig)}
    bad = [k for k in list(model_cfg) + list(sets or {}) if k not in fields]
    if bad:
        raise SystemExit(f"unknown GrainPriorConfig fields: {bad}")
    if dry:
        return tiny_config(**{k: tuple(v) if isinstance(v, list) else v for k, v in (sets or {}).items()})
    return config_from_dict({**model_cfg, **(sets or {})})


def make_model(cfg: GrainPriorConfig, args):
    from mavt.grain_prior.model import MAVTGrainPrior
    return MAVTGrainPrior(cfg, use_checkpoint=args.use_checkpoint)


def main(argv=None) -> int:
    import argparse
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ours, rest = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    sets = parse_sets(ours.set)
    return base.main(rest, config_fn=lambda m, dry: make_config(m, dry, sets), model_fn=make_model)


if __name__ == "__main__":
    sys.exit(main())
