#!/usr/bin/env python3
"""Profile one mixed (image + video) training step of the Grain-Prior model on ONE GPU: synthetic data, no
loader, no DDP. Prints forward time per top-level module (CUDA events), the fwd / bwd / optimizer split, the
peak memory, and the top GPU ops over fwd + bwd + step. Read-only: never touches a run.

  HIP_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/profile_grain_prior.py --config configs/grain_prior/final.yaml \\
      --set d_f=32 --set rgb_decoder=splat --batch-image 32 --batch-video 8 --compile --steps 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import yaml  # noqa: E402

from train_cd32 import parse_sets  # noqa: E402
from train_dino_v0 import MixedStep, apply_compile  # noqa: E402
from train_grain_prior import make_config  # noqa: E402


def hook_children(model: torch.nn.Module, dev) -> tuple:
    """Forward time per direct child of the model (and the frozen DINO inside the stems) via CUDA events."""
    acc, pending = defaultdict(float), {}
    names = {m: n for n, m in model.named_children()}
    dino = getattr(getattr(model, "stems", None), "dino", None)
    if dino is not None:
        names[dino] = "stems.dino"

    def pre(m, a, kw=None):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        pending.setdefault(names[m], []).append(e)

    def post(m, a, o):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        pending[names[m]].append(e)

    hs = []
    for m in names:
        hs += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    # the readout is driven through methods, not forward: time those the same way
    ro = getattr(model, "readout", None)
    for meth in ("init", "base", "innovation"):
        if ro is not None and hasattr(ro, meth):
            orig = getattr(ro, meth)

            def timed(*a, _orig=orig, _name=f"readout.{meth}", **kw):
                e0 = torch.cuda.Event(enable_timing=True); e0.record()
                out = _orig(*a, **kw)
                e1 = torch.cuda.Event(enable_timing=True); e1.record()
                pending.setdefault(_name, []).extend([e0, e1])
                return out
            setattr(ro, meth, timed)

    def flush():
        torch.cuda.synchronize()
        for n, evs in pending.items():
            for s, e in zip(evs[0::2], evs[1::2]):
                acc[n] += s.elapsed_time(e)
        pending.clear()
    return acc, flush, hs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--batch-image", type=int, default=32)
    p.add_argument("--batch-video", type=int, default=8, help="0: image only")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--regions", default=None, help="comma list of compile region class names (default: the model's)")
    p.add_argument("--use-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--steps", type=int, default=5, help="profiled steps after 3 warm-up steps")
    p.add_argument("--trace", default=None, help="dir for a chrome trace (optional)")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args(argv)
    dev = torch.device("cuda")
    model_cfg = yaml.safe_load(open(args.config)) or {}
    model_cfg.pop("train", None)                           # the trainer's own section
    cfg = make_config(model_cfg, False, parse_sets(args.set))
    from mavt.grain_prior.model import MAVTGrainPrior
    model = MAVTGrainPrior(cfg, use_checkpoint=args.use_checkpoint).to(dev).train()
    weights = {"image": 0.5, "video": 0.5} if args.batch_video else {"image": 1.0}
    step_mod = MixedStep(model, weights)
    if args.compile:
        regions = tuple(r for r in args.regions.split(",") if r) if args.regions else None
        apply_compile(model, regions=regions)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, fused=True)
    g = torch.Generator().manual_seed(0)
    s = cfg.image_size

    def batches():
        b = {"image": (torch.rand(args.batch_image, 3, s, s, generator=g).mul_(2).sub_(1).to(dev), None)}
        if args.batch_video:
            b["video"] = (torch.rand(args.batch_video, 3, args.n_frames, s, s, generator=g).mul_(2).sub_(1).to(dev), None)
        return b

    acc, flush, hs = hook_children(model, dev)

    def one_step(profile=None):
        bt = batches()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = step_mod(bt)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        out["loss"].backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2

    for _ in range(3):                                   # warm-up: compile, autotune, allocator
        one_step()
    flush(); acc.clear()
    torch.cuda.reset_peak_memory_stats()
    fwd = bwd = optm = 0.0
    prof = None
    if args.trace:
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                      record_shapes=False)
        prof.__enter__()
    for _ in range(args.steps):
        a, b, c = one_step()
        fwd, bwd, optm = fwd + a, bwd + b, optm + c
    flush()
    if prof is not None:
        prof.__exit__(None, None, None)
    n = args.steps
    tot = (fwd + bwd + optm) / n
    print(f"\n== {args.batch_image} images + {args.batch_video} x {args.n_frames}-frame clips per step, compile={args.compile}, "
          f"use_checkpoint={args.use_checkpoint}, sets={args.set}")
    print(f"step {tot * 1000:.0f} ms  ({1 / tot:.3f} it/s)  fwd {fwd / n * 1000:.0f}  bwd {bwd / n * 1000:.0f}  optim {optm / n * 1000:.0f} ms"
          f"  peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
    print("forward time per module (ms / step, CUDA events; overlapping children: stems contains stems.dino):")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v / n:8.1f}")
    if prof is not None:
        os.makedirs(args.trace, exist_ok=True)
        prof.export_chrome_trace(os.path.join(args.trace, "trace.json"))
        print(f"\ntop {args.top} ops by GPU self time (fwd + bwd + optim, {n} steps):")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=args.top, max_name_column_width=60))
    for h in hs:
        h.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
