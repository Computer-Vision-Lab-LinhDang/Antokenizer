#!/usr/bin/env python3
"""Profile one training step of MAVT on DINOv3 v0 on ONE GPU (synthetic data, no loader, no DDP).

Prints (1) time per top-level module (encoder.dino / encoder rest / trunk / cd / decoder, fwd only,
CUDA events) and (2) the top ops by GPU self time over fwd+bwd+optim. Read-only: never touches a run.

  HIP_VISIBLE_DEVICES=0 python scripts/profile_dino_v0.py --config configs/dino_v0/v0_alpha.yaml \
      --modality image --batch 256 --queries 256 --steps 6 --trace /tmp/mavt_prof
  # the trainer's --mix-modalities step (image + video + 3D in one fwd / bwd), v1 speed flags:
  HIP_VISIBLE_DEVICES=0 python scripts/profile_dino_v0.py --config <v1 yaml> --modality mixed \
      --batch-image 256 --batch-video 32 --batch-3d 32 --compile --use-checkpoint

The yaml's model fields (top level, v1 ones such as rgb_head: conv included) go to DinoV0Config as
they are; its `train:` section is ignored (pass batch sizes etc. on the command line).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import yaml  # noqa: E402

from mavt.dino_v0.config import DinoV0Config  # noqa: E402
from mavt.dino_v0.model import MAVTDinoV0  # noqa: E402
from train_dino_v0 import (MODALITIES, MixedStep, apply_compile, mixed_weights, orbit_cams,  # noqa: E402
                           probe_unused)


class _CpuEvent:
    """torch.cuda.Event stand-in for --device cpu (smoke tests): wall clock in ms."""

    def record(self):
        self.t = time.perf_counter()

    def elapsed_time(self, other) -> float:
        return (other.t - self.t) * 1e3


def _event(dev: torch.device):
    return torch.cuda.Event(enable_timing=True) if dev.type == "cuda" else _CpuEvent()


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def module_timers(model: torch.nn.Module, dev: torch.device) -> dict:
    """Forward-time per module via CUDA events (pre/post hooks)."""
    names = {"encoder.dino": model.encoder.dino, "encoder.stem": model.encoder.stem, "trunk": model.trunk,
             "cd": model.cd, "decoder": model.decoder, "latent": model.latent, "pool": model.pool}
    acc, ev = defaultdict(float), {}

    def pre(name):
        def f(m, a, kw=None):
            e = _event(dev)
            e.record()
            ev.setdefault(name, []).append([e, None])
        return f

    def post(name):
        def f(m, a, o):
            e = _event(dev)
            e.record()
            ev[name][-1][1] = e
        return f

    for n, m in names.items():
        m.register_forward_pre_hook(pre(n))
        m.register_forward_hook(post(n))

    def collect():
        _sync(dev)
        for n, lst in ev.items():
            acc[n] += sum(a.elapsed_time(b) for a, b in lst if b is not None)
        ev.clear()
        return acc
    return collect


def model_config(path: str, queries) -> DinoV0Config:
    """Top-level yaml keys -> DinoV0Config (train: dropped). Keys this checkout's config does not know are
    reported and dropped, so a yaml written for a newer branch still profiles."""
    y = yaml.safe_load(open(path)) or {}
    y.pop("train", None)
    if queries:
        y["train_queries_per_frame"] = queries
    known = {f.name for f in dataclasses.fields(DinoV0Config)}
    unknown = sorted(set(y) - known)
    if unknown:
        print(f"warning: {path}: fields unknown to this DinoV0Config, ignored: {unknown}")
    return DinoV0Config(**{k: v for k, v in y.items() if k in known})


def synthetic(modality: str, B: int, T: int, size: int, dev) -> tuple:
    """(x, cam) on dev: image [B,3,S,S], video / threed [B,3,T,S,S] in [-1, 1], orbit cameras for threed."""
    shape = (B, 3, size, size) if modality == "image" else (B, 3, T, size, size)
    x = torch.rand(shape, device=dev) * 2 - 1
    cam = None
    if modality == "threed":
        cam = {k: v.to(dev) for k, v in orbit_cams(B, T, size).items()}
    return x, cam


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--modality", default="image", choices=["image", "video", "threed", "mixed"],
                   help="mixed: the trainer's --mix-modalities step (MixedStep over image + video + threed)")
    p.add_argument("--batch", type=int, default=256, help="batch of a one-modality step")
    p.add_argument("--batch-image", type=int, default=256, help="mixed step")
    p.add_argument("--batch-video", type=int, default=32, help="mixed step")
    p.add_argument("--batch-3d", type=int, default=32, help="mixed step")
    p.add_argument("--modality-loss-weights", default=None, help="mixed step, as in the trainer (default equal)")
    p.add_argument("--n-frames", type=int, default=8, help="frames per video clip")
    p.add_argument("--n-views", type=int, default=4, help="posed views per 3D object")
    p.add_argument("--queries", type=int, default=None, help="override train_queries_per_frame")
    p.add_argument("--use-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="regional torch.compile as the trainer's --compile")
    p.add_argument("--compile-backend", default="inductor")
    p.add_argument("--compile-conv-head", action=argparse.BooleanOptionalAction, default=True,
                   help="with --compile: also compile decoder.rgb_conv (the trainer always does); "
                        "--no-compile-conv-head profiles the conv head eagerly (e.g. channels_last on ROCm with "
                        "PYTORCH_MIOPEN_SUGGEST_NHWC=1, which inductor's layout choices otherwise undo)")
    p.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--steps", type=int, default=6, help="profiled steps (after 3 warm-up steps)")
    p.add_argument("--trace", default=None, help="dir for a chrome trace (optional)")
    p.add_argument("--device", default="cuda", help="cuda (ROCm too); cpu only to smoke-test the script")
    a = p.parse_args(argv)

    cfg = model_config(a.config, a.queries)
    dev = torch.device(a.device)
    gpu = dev.type == "cuda"
    model = MAVTDinoV0(cfg, use_checkpoint=a.use_checkpoint).to(dev).train()
    s = cfg.image_size
    frames = dict(image=1, video=a.n_frames, threed=a.n_views)
    if a.modality == "mixed":
        step_mod = MixedStep(model, mixed_weights(a.modality_loss_weights, MODALITIES))
        pa = argparse.Namespace(n_frames=a.n_frames, n_views=a.n_views, seed=0)
        unused = probe_unused(step_mod, cfg, pa, dev, use_amp=gpu)
        print(f"mixed step, loss weights {step_mod.weights}; params without a gradient: {unused or 'none'}")
        bsz = dict(image=a.batch_image, video=a.batch_video, threed=a.batch_3d)
        batches = {m: synthetic(m, bsz[m], frames[m], s, dev) for m in step_mod.weights}
        call = lambda: step_mod(batches)
        desc = " + ".join(f"{m} {bsz[m]}" for m in step_mod.weights)
    else:
        x, cam = synthetic(a.modality, a.batch, frames[a.modality], s, dev)
        call = lambda: model(x, a.modality, cam)
        desc = f"{a.modality} batch {a.batch}"
    if a.compile:
        apply_compile(model, a.compile_backend, conv_head=a.compile_conv_head)
    params = [q for q in model.parameters() if q.requires_grad]
    fused = a.fused_adamw and gpu
    opt = torch.optim.AdamW(params, lr=1e-4, fused=True) if fused else torch.optim.AdamW(params, lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=gpu):
            out = call()
        out["loss"].backward()
        opt.step()

    for _ in range(3):                                         # warm-up (and compile)
        step()
    collect = module_timers(model, dev)
    _sync(dev)
    t0, t1 = _event(dev), _event(dev)
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if gpu else [])
    with torch.profiler.profile(activities=acts, record_shapes=False) as prof:
        t0.record()
        for _ in range(a.steps):
            step()
        _sync(dev)
        t1.record()
    ms = t0.elapsed_time(t1) / a.steps
    fwd = collect()
    mem = f", peak mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GB" if gpu else ""
    print(f"\n{desc} queries {cfg.train_queries_per_frame} compile={a.compile} checkpoint={a.use_checkpoint} "
          f"fused_adamw={fused}: {ms:.0f} ms/step ({1000 / ms:.2f} it/s{mem})")
    print("forward time per module (ms/step):")
    for n, v in sorted(fwd.items(), key=lambda kv: -kv[1]):
        print(f"  {n:14s} {v / a.steps:8.1f}")
    print(prof.key_averages().table(sort_by="self_device_time_total" if gpu else "self_cpu_time_total", row_limit=30))
    if a.trace:
        os.makedirs(a.trace, exist_ok=True)
        prof.export_chrome_trace(os.path.join(a.trace, f"{a.modality}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
