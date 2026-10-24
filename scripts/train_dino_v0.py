#!/usr/bin/env python3
"""Train MAVT on DINOv3 v0 (torchrun DDP; image / video / posed-3D-view steps).

  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 3 scripts/train_dino_v0.py \
      --config configs/dino_v0/v0_alpha.yaml --out /scratch/.../runs/muse-spark-mavt
  python scripts/train_dino_v0.py --dry-run --out /tmp/x          # CPU, tiny config, 3 steps
  torchrun --nproc_per_node 2 scripts/train_dino_v0.py --dry-run --mix-modalities --out /tmp/x   # DDP on CPU (gloo)
  python scripts/train_dino_v0.py --config ... --bench-data 50     # loader throughput only

Steps: one modality per step (v0 default; DDP with find_unused_parameters=True), or with
--mix-modalities (train: mix_modalities: true) every available modality in one step: MixedStep runs
model(x, m, cam) per modality and returns the weighted loss sum, so each step has one backward and
one all-reduce, and DDP runs without find_unused_parameters (static graph, grads as bucket views).
--compile: regional torch.compile of the repeated blocks. AdamW is fused on GPU.

The yaml holds DinoV0Config fields at top level and trainer defaults under `train:` (CLI wins).
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
import random
import signal
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from mavt.dino_v0.config import DinoV0Config, tiny_config  # noqa: E402

FROZEN_PREFIX = "encoder.dino.model."          # frozen DINOv3 weights: reloaded from dino_repo, not saved
NEW_HEADS = ("decoder.depth_head.",)            # --init-from may lack these (w_depth > 0 creates them)


# ---------------------------------------------------------------------------- args / config
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="yaml: DinoV0Config fields + optional `train:` defaults")
    p.add_argument("--out", required=False, default=None)
    p.add_argument("--imagenet", default=None, help="ImageNet-1k HF parquet root")
    p.add_argument("--openvid", default=None, help="OpenVid jsonl manifest")
    p.add_argument("--objaverse", default=None, help="Objaverse render root (manifest.lvis_full.jsonl + renders/)")
    p.add_argument("--omniworld", default=None,
                   help="OmniWorld-Game pack (scripts/pack_omniworld_dino.py; its train/ split): modality 'world', "
                        "posed dynamic clips with geometry targets")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.05, help="cosine floor as a fraction of --lr")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--batch-image", type=int, default=32, help="per GPU")
    p.add_argument("--batch-video", type=int, default=4, help="per GPU")
    p.add_argument("--batch-3d", type=int, default=4, help="per GPU, objects of --n-views views")
    p.add_argument("--batch-world", type=int, default=4, help="per GPU, OmniWorld clips of --n-frames frames")
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--video-tail", type=int, default=1000, help="last N OpenVid manifest lines held out for eval")
    p.add_argument("--objaverse-holdout", type=int, default=500, help="last N non-blank uids held out for eval")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--frame-stride", default="2",
                   help="int, or a list like 2,4,8: each training clip draws one stride uniformly "
                        "(the largest one that fits when the video is short)")
    p.add_argument("--ssv2-train", default=None,
                   help="SSv2 train manifest jsonl (OpenVid format), mixed into the video stream as "
                        "unlabeled clips (labels are never read)")
    p.add_argument("--ssv2-weight", type=float, default=0.3,
                   help="probability that a video sample comes from --ssv2-train (with --openvid)")
    p.add_argument("--modality-weights", default=None,
                   help="e.g. image:0.4,video:0.4,threed:0.2 (seeded per step, same on all ranks); "
                        "default: --video-every alternation, or 0.4/0.4/0.2 when --objaverse is given")
    p.add_argument("--video-every", type=int, default=2, help="1 video step every N steps (0: never); "
                   "used only without --modality-weights and --objaverse")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", default="auto", help="auto (out/last.pt if present) | none | <path>")
    p.add_argument("--init-from", default=None,
                   help="checkpoint whose model weights start a NEW run (fresh optimizer and schedule at step 0); "
                        "ignored once the run has its own checkpoint to resume. Heads of new flags (NEW_HEADS) may "
                        "be missing from it and keep their init")
    p.add_argument("--wandb", default=None, help="wandb project (offline unless WANDB_MODE says otherwise)")
    p.add_argument("--use-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    # ---- speed: mixed-modality steps, DDP, compile
    p.add_argument("--mix-modalities", action=argparse.BooleanOptionalAction, default=False,
                   help="every step runs all available modalities (one backward, one all-reduce, DDP without "
                        "find_unused_parameters); per-modality batch sizes from --batch-*. Off: one modality "
                        "per step (v0), --modality-weights / --video-every pick it")
    p.add_argument("--modality-loss-weights", default=None,
                   help="with --mix-modalities, e.g. image:1,video:1,threed:0.5; normalised to sum 1 over the "
                        "available modalities (a modality left out or at 0 is not run). Default: equal")
    p.add_argument("--static-graph", action=argparse.BooleanOptionalAction, default=True,
                   help="DDP static_graph=True for --mix-modalities (only with --grad-accum 1: its first "
                        "iteration cannot run under no_sync); --no-static-graph if a backend objects")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                   help="regional torch.compile (dynamic=False) of each TrunkLayer, CBlock, DBlock, DecoderBlock "
                        "and decoder.rgb_conv; a region that fails to compile runs eagerly")
    p.add_argument("--compile-backend", default="inductor", help="torch.compile backend for --compile")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="CPU, tiny_config, synthetic data, 3 steps; DDP (gloo) only when launched by torchrun")
    p.add_argument("--bench-data", type=int, default=0, metavar="N", help="iterate N batches per loader, print clips/s")
    return p


def parse_args(argv=None) -> tuple[argparse.Namespace, dict]:
    p = build_parser()
    pre, _ = p.parse_known_args(argv)
    model_cfg = {}
    if pre.config:
        import yaml
        with open(pre.config) as f:
            y = yaml.safe_load(f) or {}
        train = y.pop("train", None) or {}
        known = {a.dest for a in p._actions}
        bad = [k for k in train if k.replace("-", "_") not in known]
        if bad:
            raise SystemExit(f"unknown train keys in {pre.config}: {bad}")
        p.set_defaults(**{k.replace("-", "_"): v for k, v in train.items()})
        model_cfg = y
    args = p.parse_args(argv)
    if not args.out:
        p.error("--out is required")
    return args, model_cfg


def make_config(model_cfg: dict, dry: bool) -> DinoV0Config:
    if dry:
        return tiny_config()
    fields = {f.name for f in dataclasses.fields(DinoV0Config)}
    bad = [k for k in model_cfg if k not in fields]
    if bad:
        raise SystemExit(f"unknown DinoV0Config fields: {bad}")
    return DinoV0Config(**model_cfg)


# ---------------------------------------------------------------------------- dist helpers
def setup_dist(dry: bool):
    """(rank, world, local_rank, device). Distributed only when launched by torchrun (RANK set); a dry run
    under torchrun uses gloo on CPU, so the DDP setup can be exercised without GPUs."""
    if "RANK" not in os.environ:
        dev = torch.device("cuda" if torch.cuda.is_available() and not dry else "cpu")
        return 0, 1, 0, dev
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available() and not dry:
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
        try:
            dist.init_process_group("nccl", device_id=dev)    # nccl == rccl on ROCm
        except TypeError:                                     # older torch: no device_id kwarg
            dist.init_process_group("nccl")
    else:
        dev = torch.device("cpu")
        dist.init_process_group("gloo")
    return rank, world, local, dev


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_mean(t: torch.Tensor) -> torch.Tensor:
    if is_dist():
        t = t.clone()
        dist.all_reduce(t)
        t /= dist.get_world_size()
    return t


# ---------------------------------------------------------------------------- optim
def build_optim(model: torch.nn.Module, args, device=None):
    """AdamW (decay on >= 2-D weights only) + warmup / cosine LambdaLR. Fused AdamW (one kernel set for all
    params) when device is a GPU, the default implementation otherwise or if fused is unavailable."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    groups = lambda: [dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)]
    kw = dict(lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    opt = None
    if device is not None and torch.device(device).type == "cuda":
        try:
            opt = torch.optim.AdamW(groups(), fused=True, **kw)
        except (RuntimeError, TypeError, ValueError) as ex:
            print(f"fused AdamW unavailable ({ex}); using the default AdamW", flush=True)
    if opt is None:
        opt = torch.optim.AdamW(groups(), **kw)

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def set_fused(opt: torch.optim.Optimizer, fused: bool) -> None:
    """Restore the fused choice after a resume: load_state_dict keeps the checkpoint's group options (a v0
    checkpoint says fused=None) and leaves its step counters on the CPU, where fused AdamW cannot use them."""
    if not fused:
        return
    for g in opt.param_groups:
        g["fused"], g["foreach"] = True, None
        for p in g["params"]:
            st = opt.state.get(p, {})
            if torch.is_tensor(st.get("step")):
                st["step"] = st["step"].to(device=p.device, dtype=torch.float32)


# ---------------------------------------------------------------------------- checkpoint
def trainable_state(model: torch.nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if not k.startswith(FROZEN_PREFIX)}


def save_ckpt(path: str, model, opt, sched, step: int, cfg: DinoV0Config, args) -> None:
    tmp = path + ".tmp"
    torch.save(dict(model=trainable_state(model), optim=opt.state_dict(), sched=sched.state_dict(),
                    step=step, cfg=dataclasses.asdict(cfg), args=vars(args)), tmp)
    os.replace(tmp, path)


def load_ckpt(path: str, model, opt, sched, lr: Optional[float] = None) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    opt.load_state_dict(ck["optim"])
    sched.load_state_dict(ck["sched"])
    if lr is not None:                          # --lr wins over the checkpoint's peak lr (schedule follows --steps)
        sched.base_lrs = [lr] * len(opt.param_groups)
        for g, f in zip(opt.param_groups, sched.lr_lambdas):
            g["initial_lr"] = lr
            g["lr"] = lr * f(sched.last_epoch)
    return int(ck["step"])


def load_weights(path: str, model) -> tuple:
    """--init-from: model weights only, strict except the frozen encoder and NEW_HEADS. -> (source step, missing)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if not k.startswith(FROZEN_PREFIX)]
    bad = [k for k in missing if not k.startswith(NEW_HEADS)]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
    return int(ck["step"]), missing


# ---------------------------------------------------------------------------- data
def orbit_cams(B: int, T: int, size: int) -> dict:
    """Synthetic OpenCV cameras on a circle of radius 2 looking at the origin (dry-run)."""
    c2w = torch.zeros(B, T, 4, 4)
    for t in range(T):
        a = 2 * math.pi * t / T
        o = torch.tensor([2 * math.cos(a), 2 * math.sin(a), 0.3])
        z = -o / o.norm()
        x = torch.linalg.cross(z, torch.tensor([0.0, 0.0, -1.0]))
        x = x / x.norm()
        y = torch.linalg.cross(z, x)
        c2w[:, t, :3, 0], c2w[:, t, :3, 1], c2w[:, t, :3, 2], c2w[:, t, :3, 3] = x, y, z, o
        c2w[:, t, 3, 3] = 1
    f = size / 2 / math.tan(0.3456)
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]]).expand(B, T, 3, 3).clone()
    return dict(c2w=c2w, K=K)


def synthetic_geo(batch: int, cfg: DinoV0Config, n_frames: int, g: torch.Generator) -> dict:
    """Random geometry targets shaped like OmniWorldClips' (dry run, probe of unused parameters)."""
    G, nK, h = cfg.grid, -(-n_frames // cfg.keyframe_every), cfg.grid * (cfg.patch // 4)
    return dict(geo_corr=(torch.rand(batch, nK, n_frames, G, G, 2, generator=g) - 0.5) * 0.1,
                geo_valid=torch.rand(batch, nK, n_frames, G, G, generator=g) > 0.3,
                geo_dyncell=torch.rand(batch, n_frames, G, G, generator=g) > 0.7,
                geo_depth=torch.rand(batch, n_frames, h, h, generator=g) * 10)


def synthetic_iter(modality: str, batch: int, cfg: DinoV0Config, n_frames: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    s = cfg.image_size
    shape = (batch, 3, s, s) if modality == "image" else (batch, 3, n_frames, s, s)
    while True:
        b = dict(data=torch.rand(shape, generator=g) * 2 - 1, modality=modality)
        if modality in ("threed", "world"):
            b.update(orbit_cams(batch, n_frames, s))
        if modality == "world":
            b.update(synthetic_geo(batch, cfg, n_frames, g))
        yield b


def build_data(args, cfg: DinoV0Config, rank: int, world: int, seed: int) -> dict:
    from mavt.dino_v0.data import infinite, mixed_loaders
    lds = mixed_loaders(args.imagenet, args.openvid, args.objaverse, batch_image=args.batch_image,
                        batch_video=args.batch_video, batch_3d=args.batch_3d, size=cfg.image_size,
                        omniworld=getattr(args, "omniworld", None), batch_world=getattr(args, "batch_world", 4),
                        cell=cfg.patch, keyframe_every=cfg.keyframe_every,
                        n_frames=args.n_frames, frame_stride=args.frame_stride, n_views=args.n_views,
                        video_tail=args.video_tail, objaverse_holdout=args.objaverse_holdout,
                        num_workers=args.num_workers, rank=rank, world=world, seed=seed,
                        ssv2=getattr(args, "ssv2_train", None), ssv2_weight=getattr(args, "ssv2_weight", 0.3))
    return {m: infinite(ld, smp) for m, (ld, smp) in lds.items()}


def write_heldout(args) -> None:
    """Record the eval hold-outs excluded from training (rank 0)."""
    from mavt.dino_v0.data import ObjaverseViews, read_manifest, split_tail
    if args.objaverse and args.objaverse_holdout > 0:
        uids = ObjaverseViews.heldout_uids(args.objaverse, args.objaverse_holdout)
        with open(os.path.join(args.out, "objaverse_heldout.txt"), "w") as f:
            f.write("".join(u + "\n" for u in uids))
    if args.openvid and args.video_tail > 0:
        rows = split_tail(read_manifest(args.openvid), args.video_tail, "heldout")
        with open(os.path.join(args.out, "openvid_heldout.txt"), "w") as f:
            f.write("".join(r["path"] + "\n" for r in rows))


def parse_weights(spec: str) -> dict:
    w = {}
    for part in spec.split(","):
        if part.strip():
            k, v = part.split(":")
            w[k.strip()] = float(v)
    bad = set(w) - set(ALL_MODALITIES)
    if bad:
        raise SystemExit(f"unknown modalities in --modality-weights: {sorted(bad)}")
    return w


def bench_data(args, cfg, rank, world) -> None:
    its = build_data(args, cfg, rank, world, args.seed)
    if not its:
        raise SystemExit("--bench-data needs --imagenet, --openvid, --ssv2-train and/or --objaverse")
    for m, it in its.items():
        b = next(it)                                           # warm-up: worker start + first batch
        t0, n = time.time(), 0
        strides: dict = {}
        for _ in range(args.bench_data):
            b = next(it)
            n += b["data"].shape[0]
            for s in b.get("stride", torch.empty(0)).tolist():
                strides[s] = strides.get(s, 0) + 1
        dt = time.time() - t0
        hist = (" strides " + ",".join(f"{s}:{c}" for s, c in sorted(strides.items()))) if strides else ""
        print(f"[bench rank{rank}] {m}: {args.bench_data} batches, {n} clips in {dt:.1f}s -> "
              f"{n / dt:.1f} clips/s/rank, {args.bench_data / dt:.2f} batches/s; batch shape "
              f"{tuple(b['data'].shape)} range [{b['data'].min():.2f}, {b['data'].max():.2f}]{hist}", flush=True)


# ---------------------------------------------------------------------------- mixed step / compile
MODALITIES = ("image", "video", "threed")
ALL_MODALITIES = MODALITIES + ("world",)        # + OmniWorld clips (--omniworld): posed dynamic video with geometry
COMPILE_REGIONS = ("TrunkLayer", "CBlock", "DBlock", "DecoderBlock")   # repeated blocks, by class name


class MixedStep(torch.nn.Module):
    """One training step over several modalities as ONE module call, so DDP sees one forward, one backward
    and one all-reduce per step and every parameter is used each step (no find_unused_parameters).

    forward(batches {modality: (x, cam)}) runs model(x, modality, cam) for each modality present (fixed
    order image, video, threed, world) and returns loss = sum_m weights[m] * loss_m, parts {'<m>/loss': loss_m,
    '<m>/loss_<k>': ...} and diag {'<m>/diag_<k>': ...}. No parameters of its own: checkpoints hold the
    inner model, keys unchanged."""

    def __init__(self, model: torch.nn.Module, weights: dict):
        super().__init__()
        self.model, self.weights = model, dict(weights)

    def forward(self, batches: dict) -> dict:
        loss, parts, diag = 0.0, {}, {}
        for m in ALL_MODALITIES:
            if m not in batches:
                continue
            x, cam, *geo = batches[m]
            out = self.model(x, m, cam, *geo)
            loss = loss + self.weights[m] * out["loss"]
            parts[f"{m}/loss"] = out["loss"].detach()
            parts.update({f"{m}/loss_{k}": v for k, v in out["parts"].items()})
            diag.update({f"{m}/diag_{k}": v for k, v in out["diag"].items()})
        return dict(loss=loss, parts=parts, diag=diag)


def mixed_weights(spec: Optional[str], available) -> dict:
    """Loss weights of a mixed step over the available modalities, normalised to sum 1 so the step loss
    stays on the scale of a one-modality step (--grad-clip keeps its meaning). spec None: equal; a
    modality missing from spec or at 0 is not run."""
    w = parse_weights(spec) if spec else {m: 1.0 for m in ALL_MODALITIES}
    w = {m: w.get(m, 0.0) for m in ALL_MODALITIES if m in available and w.get(m, 0.0) > 0}
    if not w:
        raise SystemExit("--modality-loss-weights selects no available modality")
    tot = sum(w.values())
    return {m: v / tot for m, v in w.items()}


def to_inputs(b: dict, modality: str, dev) -> tuple:
    """Loader batch -> (x, cam) on dev; cam = dict(c2w, K) for posed views and world clips, else None.
    World clips add a third item, geo = dict(corr, valid, dyncell, depth) (the batch's geo_* tensors)."""
    x = b["data"].to(dev, non_blocking=True)
    cam = None
    if modality in ("threed", "world"):
        cam = dict(c2w=b["c2w"].to(dev, non_blocking=True), K=b["K"].to(dev, non_blocking=True))
    if modality == "world":
        geo = {k[4:]: v.to(dev, non_blocking=True) for k, v in b.items() if k.startswith("geo_")}
        return x, cam, geo
    return x, cam


def probe_unused(step_mod: MixedStep, cfg: DinoV0Config, args, dev, use_amp: bool) -> list:
    """Names of trainable parameters that get no gradient from one mixed step (synthetic batch of 1 per
    modality: which parameters are used depends on cfg and the modalities, not on the data). Max over
    ranks. DDP ignores them: without find_unused_parameters it would wait for their gradients forever."""
    batches = {}
    for m in step_mod.weights:
        T = args.n_views if m == "threed" else args.n_frames
        batches[m] = to_inputs(next(synthetic_iter(m, 1, cfg, T, args.seed)), m, dev)
    step_mod.train().zero_grad(set_to_none=True)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
        out = step_mod(batches)
    out["loss"].backward()
    named = [(n, p) for n, p in step_mod.named_parameters() if p.requires_grad]
    used = torch.tensor([float(p.grad is not None) for _, p in named], device=dev)
    step_mod.zero_grad(set_to_none=True)
    if is_dist():
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
    return [n for (n, _), u in zip(named, used.tolist()) if u == 0]


def needs_buffer_sync(model: torch.nn.Module) -> bool:
    """True if a module keeps running statistics (BatchNorm-like) that DDP must broadcast. The v0 buffers
    (RoPE frequencies, ImageNet mean / std, the C neighbourhood mask, LPIPS scaling) are constants."""
    return any(isinstance(m, torch.nn.modules.batchnorm._NormBase) and m.track_running_stats
               for m in model.modules())


def mixed_ddp(step_mod: MixedStep, unused: list, device_ids, static_graph: bool) -> tuple:
    """DDP for the mixed step: find_unused_parameters=False (every parameter is used each step; the ones
    probe_unused found are ignored), static_graph, grads as views of the all-reduce buckets (no copy), and
    no buffer sync unless a module keeps running statistics. Returns (ddp, description)."""
    DDP = torch.nn.parallel.DistributedDataParallel
    if unused:
        DDP._set_params_and_buffers_to_ignore_for_model(step_mod, unused)
    bufs = needs_buffer_sync(step_mod)                         # False for v0: no buffer changes in training
    kw = dict(forward_sync_buffers=bufs) if "forward_sync_buffers" in inspect.signature(DDP).parameters \
        else dict(broadcast_buffers=bufs)                      # newer torch renamed broadcast_buffers
    ddp = DDP(step_mod, device_ids=device_ids, find_unused_parameters=False, static_graph=static_graph,
              gradient_as_bucket_view=True, **kw)
    return ddp, (f"DDP: find_unused_parameters=False static_graph={static_graph} gradient_as_bucket_view=True "
                 f"buffer sync={bufs} ignored params={len(unused)}")


def apply_compile(model: torch.nn.Module, backend: str = "inductor", log=print, conv_head: bool = True) -> list:
    """Regional torch.compile (dynamic=False) of the repeated blocks: every TrunkLayer, CBlock, DBlock and
    DecoderBlock, plus decoder.rgb_conv when present (conv_head=False leaves it eager: a profiling knob,
    see scripts/profile_dino_v0.py). Each region compiles in place (Module.compile) in try / except; a region
    that fails, here or at its first call (dynamo suppress_errors), runs eagerly. Python loops over frames /
    blocks stay eager. Returns the compiled module names."""
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)   # shapes per modality
    targets = [(n, m) for n, m in model.named_modules() if type(m).__name__ in COMPILE_REGIONS]
    conv = getattr(getattr(model, "decoder", None), "rgb_conv", None)
    if conv_head and isinstance(conv, torch.nn.Module):
        targets.append(("decoder.rgb_conv", conv))
    done = []
    for n, m in targets:
        try:
            m.compile(backend=backend, dynamic=False)
            done.append(n)
        except Exception as ex:                                 # noqa: BLE001 - any failure: stay eager
            log(f"compile: {n} left eager ({type(ex).__name__}: {ex})")
    kinds = sorted({type(m).__name__ for n, m in targets if n in done})
    log(f"compile: {len(done)}/{len(targets)} regions, backend={backend}, dynamic=False ({', '.join(kinds)})")
    return done


# ---------------------------------------------------------------------------- main loop
def main(argv=None) -> int:
    args, model_cfg = parse_args(argv)
    dry = args.dry_run
    if dry:
        args.steps, args.batch_image, args.batch_video, args.batch_3d, args.n_frames = 3, 2, 2, 2, 4
        args.warmup, args.log_every, args.ckpt_every, args.bf16 = 1, 1, 2, False
    cfg = make_config(model_cfg, dry)
    rank, world, local, dev = setup_dist(dry)
    os.makedirs(args.out, exist_ok=True)
    seed = args.seed + 1000 * rank
    random.seed(seed)
    torch.manual_seed(seed)

    if args.bench_data:
        bench_data(args, cfg, rank, world)
        if is_dist():
            dist.destroy_process_group()
        return 0

    from mavt.dino_v0.model import MAVTDinoV0
    log = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)
    torch.manual_seed(args.seed)                               # identical init on every rank
    model = MAVTDinoV0(cfg, use_checkpoint=args.use_checkpoint).to(dev)
    opt, sched = build_optim(model, args, dev)
    fused = bool(opt.param_groups[0].get("fused"))
    step = 0
    resume = None
    if args.resume == "auto":
        cand = os.path.join(args.out, "last.pt")
        resume = cand if os.path.exists(cand) else None
    elif args.resume and args.resume != "none":
        resume = args.resume
    if resume:
        step = load_ckpt(resume, model, opt, sched, lr=args.lr)
        set_fused(opt, fused)
        log(f"resumed from {resume} at step {step}")
    elif args.init_from:
        src_step, new = load_weights(args.init_from, model)
        log(f"weights from {args.init_from} (step {src_step}); optimizer and schedule start at step 0"
            + (f"; new heads keep their init: {sorted({k.rsplit('.', 1)[0] for k in new})}" if new else ""))
    use_amp = args.bf16 and dev.type == "cuda"
    mixed = args.mix_modalities

    if dry:
        bsz = dict(image=args.batch_image, video=args.batch_video, threed=args.batch_3d)
        its = {m: synthetic_iter(m, bsz[m], cfg, args.n_frames, seed) for m in MODALITIES}
    else:
        # sampler seed: the same on every rank, so the rank shards are disjoint (it was per rank in v0).
        # Augmentation stays per rank: worker seeds come from each rank's own torch stream (data._worker_init).
        its = build_data(args, cfg, rank, world, args.seed + step)
    if not its:
        raise SystemExit("no data: pass --imagenet and/or --openvid (or --dry-run)")

    step_mod, unused = None, []
    if mixed:
        step_mod = MixedStep(model, mixed_weights(args.modality_loss_weights, its))
        unused = probe_unused(step_mod, cfg, args, dev, use_amp)
        log(f"mixed step: loss weights {step_mod.weights}; trainable parameters without a gradient "
            f"(DDP ignores them): {unused or 'none'}")
    if args.compile:                                           # after the probe: compile for the real shapes only
        apply_compile(model, args.compile_backend, log)
    torch.manual_seed(seed + step)                             # per-rank stream (queries, data aug)
    random.seed(seed + step)

    ddp = step_mod if mixed else model
    if is_dist():
        DDP, ids = torch.nn.parallel.DistributedDataParallel, [dev.index] if dev.type == "cuda" else None
        if mixed:
            # static_graph trips an internal reducer assert when its first iteration runs under no_sync()
            static = args.static_graph and args.grad_accum == 1
            ddp, desc = mixed_ddp(step_mod, unused, ids, static)
            log(desc + (" (static_graph off: --grad-accum > 1)" if args.static_graph and not static else ""))
        else:                                                  # one modality per step: some params unused
            ddp = DDP(model, device_ids=ids, find_unused_parameters=True)

    weights = None
    if args.modality_weights:
        weights = parse_weights(args.modality_weights)
    elif "threed" in its:
        weights = dict(image=0.4, video=0.4, threed=0.2)
    if weights is not None:
        weights = {m: w for m, w in weights.items() if m in its and w > 0}
        if not weights:
            raise SystemExit("--modality-weights selects no available modality")

    def modality_for(s: int) -> str:
        if dry:
            return ("image", "video", "threed")[s % 3]
        if weights is not None:                                  # same draw on every rank
            r = random.Random(args.seed * 1_000_003 + s)
            return r.choices(list(weights), weights=list(weights.values()))[0]
        if "video" not in its:
            return next(iter(its))
        if "image" not in its:
            return "video"
        return "video" if args.video_every > 0 and s % args.video_every == args.video_every - 1 else "image"

    stop = {"flag": False}

    def on_term(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_term)
    if rank == 0:
        signal.signal(signal.SIGINT, on_term)

    wb = None
    if args.wandb and rank == 0:
        try:
            import wandb
            os.environ.setdefault("WANDB_MODE", "offline")
            wb = wandb.init(project=args.wandb, dir=args.out, name=os.path.basename(os.path.normpath(args.out)),
                            config=dict(cfg=dataclasses.asdict(cfg), args=vars(args)), resume="allow")
        except Exception as ex:
            print(f"wandb disabled: {ex}", flush=True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"world={world} device={dev} trainable={n_train / 1e6:.1f}M start_step={step} "
              f"data={sorted(its)} steps={'mixed' if mixed else 'one modality'} fused_adamw={fused}", flush=True)
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(dict(cfg=dataclasses.asdict(cfg), args=vars(args), modality_weights=weights,
                           mixed_loss_weights=step_mod.weights if mixed else None), f, indent=1)
        if not dry:
            write_heldout(args)
    log_f = open(os.path.join(args.out, "log.jsonl"), "a") if rank == 0 else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t_last, steps_since, clips_since = time.time(), 0, 0
    model.train()

    def fetch(m: str) -> dict:
        try:
            return next(its[m])
        except Exception:
            if stop["flag"] and rank == 0:                      # loader torn down by the signal
                save_ckpt(os.path.join(args.out, "last.pt"), model, opt, sched, step, cfg, args)
                print(f"signal received: saved last.pt at step {step}, exiting", flush=True)
            raise

    while step < args.steps:
        modality = "mixed" if mixed else modality_for(step)
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            if mixed:                                           # one batch of every modality, one module call
                batches = {m: to_inputs(fetch(m), m, dev) for m in step_mod.weights}
                inputs, n_clips = (batches,), sum(v[0].shape[0] for v in batches.values())
            else:
                x, cam, *geo = to_inputs(fetch(modality), modality, dev)
                inputs, n_clips = (x, modality, cam, *geo), x.shape[0]
            sync = micro == args.grad_accum - 1
            ctx = ddp.no_sync() if (is_dist() and not sync) else _null()
            with ctx:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = ddp(*inputs)
                (out["loss"] / args.grad_accum).backward()
            clips_since += n_clips
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               args.grad_clip if args.grad_clip > 0 else float("inf"))
        finite = bool(torch.isfinite(gnorm))
        if finite:                                              # grads are all-reduced: same on all ranks
            opt.step()
        sched.step()
        step += 1
        steps_since += 1

        if step % args.log_every == 0 or step == args.steps:
            # one modality: loss_<k>, diag_<k>; mixed: keys already carry the modality (image/loss_rgb, ...)
            pl, pd = ("", "") if mixed else ("loss_", "diag_")
            named = ([(pl + k, out["parts"][k]) for k in sorted(out["parts"])]
                     + [(pd + k, out["diag"][k]) for k in sorted(out["diag"])])
            vec = all_mean(torch.stack([out["loss"].detach().float()] + [v.float() for _, v in named])).tolist()
            dt = time.time() - t_last
            rec = dict(step=step, modality=modality, loss=vec[0], **{k: v for (k, _), v in zip(named, vec[1:])},
                       grad_norm=float(gnorm), skipped=not finite, lr=sched.get_last_lr()[0],
                       it_s=steps_since / dt, clips_s=clips_since * world / dt, time=time.time())
            if dev.type == "cuda":
                rec["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
            if rank == 0:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()
                               if k != "time"), flush=True)
                if wb is not None:
                    wb.log({k if "/" in k else f"{modality}/{k}" if k.startswith(("loss", "diag")) else k: v
                            for k, v in rec.items() if isinstance(v, (int, float))}, step=step)
            t_last, steps_since, clips_since = time.time(), 0, 0
        if not finite and rank == 0:
            print(f"step {step}: non-finite grad norm, update skipped", flush=True)

        if rank == 0 and (step % args.ckpt_every == 0 or step == args.steps):
            save_ckpt(os.path.join(args.out, "last.pt"), model, opt, sched, step, cfg, args)
            if step % (args.ckpt_every * 5) == 0:
                save_ckpt(os.path.join(args.out, f"step_{step:07d}.pt"), model, opt, sched, step, cfg, args)

        flag = torch.tensor([float(stop["flag"])], device=dev)
        if is_dist():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            if rank == 0:
                save_ckpt(os.path.join(args.out, "last.pt"), model, opt, sched, step, cfg, args)
                print(f"signal received: saved last.pt at step {step}, exiting", flush=True)
            break

    if log_f:
        log_f.close()
    if wb is not None:
        wb.finish()
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    sys.exit(main())
