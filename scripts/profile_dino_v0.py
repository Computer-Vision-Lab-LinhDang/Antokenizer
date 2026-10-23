#!/usr/bin/env python3
"""Profile one training step of MAVT on DINOv3 v0 on ONE GPU (synthetic data, no loader, no DDP).

Prints (1) time per top-level module (encoder.dino / encoder rest / trunk / cd / decoder, fwd only,
CUDA events) and (2) the top ops by GPU self time over fwd+bwd+optim. Read-only: never touches a run.

  HIP_VISIBLE_DEVICES=0 python scripts/profile_dino_v0.py --config configs/dino_v0/v0_alpha.yaml \
      --modality image --batch 256 --queries 256 --steps 6 --trace /tmp/mavt_prof
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from mavt.dino_v0.config import DinoV0Config  # noqa: E402
from mavt.dino_v0.model import MAVTDinoV0  # noqa: E402


def module_timers(model: torch.nn.Module) -> dict:
    """Forward-time per module via CUDA events (pre/post hooks)."""
    names = {"encoder.dino": model.encoder.dino, "encoder.stem": model.encoder.stem, "trunk": model.trunk,
             "cd": model.cd, "decoder": model.decoder, "latent": model.latent, "pool": model.pool}
    acc, ev = defaultdict(float), {}

    def pre(name):
        def f(m, a, kw=None):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev.setdefault(name, []).append([e, None])
        return f

    def post(name):
        def f(m, a, o):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[name][-1][1] = e
        return f

    for n, m in names.items():
        m.register_forward_pre_hook(pre(n))
        m.register_forward_hook(post(n))

    def collect():
        torch.cuda.synchronize()
        for n, lst in ev.items():
            acc[n] += sum(a.elapsed_time(b) for a, b in lst if b is not None)
        ev.clear()
        return acc
    return collect


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--modality", default="image", choices=["image", "video", "threed"])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--queries", type=int, default=None, help="override train_queries_per_frame")
    p.add_argument("--steps", type=int, default=6, help="profiled steps (after 3 warm-up steps)")
    p.add_argument("--trace", default=None, help="dir for a chrome trace (optional)")
    a = p.parse_args()

    y = yaml.safe_load(open(a.config))
    y.pop("train", None)
    if a.queries:
        y["train_queries_per_frame"] = a.queries
    cfg = DinoV0Config(**y)
    dev = torch.device("cuda")
    model = MAVTDinoV0(cfg).to(dev).train()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-4, fused=True)
    s = cfg.image_size
    shape = (a.batch, 3, s, s) if a.modality == "image" else (a.batch, 3, a.n_frames, s, s)
    x = torch.rand(shape, device=dev) * 2 - 1
    cam = None
    if a.modality == "threed":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_dino_v0 import orbit_cams
        cam = {k: v.to(dev) for k, v in orbit_cams(a.batch, a.n_frames, s).items()}

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, a.modality, cam)
        out["loss"].backward()
        opt.step()

    for _ in range(3):
        step()
    collect = module_timers(model)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts, record_shapes=False) as prof:
        t0.record()
        for _ in range(a.steps):
            step()
        t1.record()
        torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / a.steps
    fwd = collect()
    print(f"\n{a.modality} batch {a.batch} queries {cfg.train_queries_per_frame}: {ms:.0f} ms/step "
          f"({1000 / ms:.2f} it/s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GB)")
    print("forward time per module (ms/step):")
    for n, v in sorted(fwd.items(), key=lambda kv: -kv[1]):
        print(f"  {n:14s} {v / a.steps:8.1f}")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30))
    if a.trace:
        os.makedirs(a.trace, exist_ok=True)
        prof.export_chrome_trace(os.path.join(a.trace, f"{a.modality}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
