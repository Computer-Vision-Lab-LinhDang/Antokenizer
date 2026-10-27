"""Query-readout pilot trainer: ImageNet reconstruction through pilot_step for any readout (legacy,
shared_ray_v1 with or without the CRSR flags, shared_field_v1). Same loss for every arm, so arms compare.

  torchrun --standalone --nproc_per_node 4 scripts/train_query_pilot.py --config configs/grain_prior/base.yaml \
    --imagenet <parquet root> --out <run> --set readout_version=shared_ray_v1 --set read_sink=true ...

The target is the source image seen by an identity camera at fov_init: per-ray RGB L1 on --rays sampled pixels.
Eval (rank 0) decodes every pixel of --eval-images validation images: PSNR, grain radius, sink share.
lr: linear warmup, then constant or cosine to --min-lr-ratio. --skip-gn: drop an update whose grad norm (before
clipping) exceeds it (LVSM). --ema: eval also with an EMA of the weights. --dry-run: tiny config, CPU, synthetic.
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import train_dino_v0 as base  # noqa: E402
from train_cd32 import parse_sets  # noqa: E402
from train_grain_prior import make_config  # noqa: E402

from mavt.grain_prior import geometry as g  # noqa: E402
from mavt.grain_prior.evaluate_shared import legacy_queries  # noqa: E402
from mavt.grain_prior.model import MAVTGrainPrior  # noqa: E402
from mavt.grain_prior.pilot import configure_pilot, pilot_step, warm_start_query_encoder  # noqa: E402
from mavt.grain_prior.query import Camera, RayQuery  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/grain_prior/base.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--imagenet", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None, help="legacy checkpoint: copy its encoder, fresh reader")
    p.add_argument("--frozen-encoder", action="store_true")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine: final lr / peak lr")
    p.add_argument("--skip-gn", type=float, default=0.0, help="skip the update when grad norm > this (0: off)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights for eval (0: off)")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=256, help="global images per step")
    p.add_argument("--rays", type=int, default=1024, help="sampled target pixels per image")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-images", type=int, default=32)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def load_model_cfg(path):
    import yaml
    with open(path) as fh:
        y = yaml.safe_load(fh) or {}
    return {k: v for k, v in y.items() if k != "train"}


def identity_camera(cfg, B, dev):
    f0 = 0.5 / math.tan(math.radians(cfg.fov_init) / 2)
    kappa = torch.tensor([f0, f0, 0.5, 0.5], device=dev).expand(B, -1).contiguous()
    return kappa, Camera(kappa, torch.eye(3, device=dev).expand(B, -1, -1).contiguous(), torch.zeros(B, 3, device=dev))


def reader_layers(model):
    if model.cfg.readout_version == "legacy":
        return list(model.readout.layers)
    qd = model.query_decoder
    return list((qd.ray if hasattr(qd, "ray") else qd.field).layers)


@torch.no_grad()
def evaluate(model, x, use_amp):
    """PSNR over every pixel, mean grain radius and sink share of the reader layers."""
    model.eval()
    cfg, B, dev = model.cfg, x.shape[0], x.device
    kappa, cam = identity_camera(cfg, B, dev)
    layers = [l for l in reader_layers(model) if hasattr(l, "sink")]
    for l in layers:
        l.track_sink = True
    psnr, sig, share = [], [], []
    for i in range(B):
        xi, ki = x[i:i + 1], kappa[i:i + 1]
        ci = Camera(cam.kappa[i:i + 1], cam.R[i:i + 1], cam.t[i:i + 1])
        uv = g.cell_centres(cfg.image_size, dev)[None]
        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            S = model.encode_source(xi, ki)
            if cfg.readout_version == "legacy":
                rgb = legacy_queries(model, S, ci, uv)["rgb"]
            else:
                q = RayQuery(ci, uv, cfg.render_near, cfg.render_far, cfg.render_samples)
                rgb = model.decode_packet(model.read_rays(model.prepare_state(S, mode="novel_view"), q), ("rgb",))["rgb"]
        mse = (rgb.float() - ((xi.float() + 1) / 2).flatten(2).transpose(1, 2)).square().mean()
        psnr.append(-10 * math.log10(max(mse.item(), 1e-10)))
        sig.append(S["sigma"].float().mean().item())
        share += [l.last_sink_share for l in layers]
    for l in layers:
        l.track_sink = False
    model.train()
    out = dict(psnr_val=sum(psnr) / len(psnr), sigma_val=sum(sig) / len(sig))
    if share:
        out["sink_share"] = sum(share) / len(share)
    return out


def lr_factor(step, args):
    if step <= args.warmup:
        return step / max(1, args.warmup)
    if args.schedule == "constant":
        return 1.0
    prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


class EMA:
    """fp32 shadow of the trainable weights (rank 0); swap() exchanges it with the live weights for eval."""
    def __init__(self, params, decay):
        self.params, self.decay = params, decay
        self.shadow = [p.detach().float().clone() for p in params]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p.detach().float(), 1 - self.decay)

    @torch.no_grad()
    def swap(self):
        for s, p in zip(self.shadow, self.params):
            tmp = p.detach().clone()
            p.copy_(s.to(p.dtype))
            s.copy_(tmp.float())


def allreduce_grads(params, world):
    """pilot_step calls model methods, not forward: DDP hooks would not fire, so grads are reduced by hand."""
    if world == 1:
        return
    grads = [p.grad for p in params]
    flat = torch.cat([gr.reshape(-1) for gr in grads])
    dist.all_reduce(flat)
    flat /= world
    i = 0
    for gr in grads:
        n = gr.numel()
        gr.copy_(flat[i:i + n].view_as(gr))
        i += n


def main(argv=None):
    args = build_parser().parse_args(argv)
    rank, world, _, dev = base.setup_dist(args.dry_run)
    torch.manual_seed(args.seed)                                   # same init on every rank
    sets = parse_sets(args.set)
    if args.init:
        model, receipt = warm_start_query_encoder(args.init, sets.pop("readout_version", "shared_ray_v1"), **sets)
    else:
        cfg = make_config(load_model_cfg(args.config) if not args.dry_run else {}, args.dry_run, sets)
        cfg.check()
        model, receipt = MAVTGrainPrior(cfg), None
    cfg = model.cfg
    model = model.to(dev)
    if world > 1:                                                  # one set of weights; grads are reduced by hand
        for t in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(t.data, 0)
    torch.manual_seed(args.seed + rank)
    configure_pilot(model, frozen_encoder=args.frozen_encoder)
    use_amp = args.bf16 and dev.type == "cuda"
    per_rank = max(1, args.batch // world)
    os.makedirs(args.out, exist_ok=True)

    if args.dry_run or not args.imagenet:
        if not args.dry_run:
            raise SystemExit("--imagenet is required (or --dry-run)")
        gen = torch.Generator().manual_seed(args.seed)
        data = iter(lambda: dict(data=torch.rand(per_rank, 3, cfg.image_size, cfg.image_size, generator=gen) * 2 - 1), None)
        x_val = torch.rand(2, 3, cfg.image_size, cfg.image_size, generator=gen) * 2 - 1
    else:
        from mavt.dino_v0.data import ImageNetParquet, infinite, mixed_loaders
        ld, smp = mixed_loaders(args.imagenet, batch_image=per_rank, size=cfg.image_size, num_workers=args.num_workers,
                                rank=rank, world=world, seed=args.seed)["image"]
        data = infinite(ld, smp)
        val = ImageNetParquet(args.imagenet, "validation", cfg.image_size, train=False)
        x_val = torch.stack([val[i]["data"] for i in range(args.eval_images)])
    x_val = x_val.to(dev)

    # ---- probe step: freeze what the pilot never reaches (all ranks agree), then the optimiser
    def step_loss(x, seed):
        kappa, cam = identity_camera(cfg, x.shape[0], dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            return pilot_step(model, x, kappa, x, cam, torch.ones(x.shape[0], *x.shape[2:], device=dev),
                              rays=args.rays, seed=seed)

    step_loss(next(data)["data"][:2].to(dev), args.seed)["loss"].backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    used = torch.tensor([float(p.grad is not None) for _, p in trainable], device=dev)
    if world > 1:
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
    for (n, p), u in zip(trainable, used.tolist()):
        p.grad = None
        if not u:
            p.requires_grad_(False)
    params = [p for _, p in trainable if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    if rank == 0:
        n_par = sum(p.numel() for p in params)
        meta = dict(cfg=dataclasses.asdict(cfg), args=vars(args), trainable=n_par, warm_start=receipt)
        with open(os.path.join(args.out, "config.json"), "w") as fh:
            json.dump(meta, fh, indent=1, default=str)
        print(f"trainable {n_par / 1e6:.2f} M, readout {cfg.readout_version}, world {world}, batch/rank {per_rank}", flush=True)

    log = open(os.path.join(args.out, "log.jsonl"), "a") if rank == 0 else None
    ema = EMA(params, args.ema) if args.ema > 0 and rank == 0 else None
    n_skip = 0
    t_last = time.time()
    for step in range(1, args.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = args.lr * lr_factor(step, args)
        x = next(data)["data"].to(dev, non_blocking=True)
        r = step_loss(x, args.seed * 1_000_003 + step * world + rank)
        opt.zero_grad(set_to_none=False)
        r["loss"].backward()
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        allreduce_grads(params, world)
        gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip if args.grad_clip > 0 else float("inf"))
        finite = bool(torch.isfinite(gn))
        ok = finite and not (args.skip_gn > 0 and float(gn) > args.skip_gn)   # same gn on every rank
        if ok:
            opt.step()
            if ema is not None:
                ema.update()
        else:
            n_skip += 1
        if step % args.log_every == 0 or step == 1:
            stats = torch.stack([r["l1"], r["mse"], r["sigma_mean"]]).float()
            stats = base.all_mean(stats)
            rec = dict(step=step, l1=stats[0].item(), psnr=-10 * math.log10(max(stats[1].item(), 1e-10)),
                       sigma=stats[2].item(), grad_norm=float(gn), skipped=not ok, n_skipped=n_skip, lr=opt.param_groups[0]["lr"],
                       it_s=args.log_every / max(1e-6, time.time() - t_last) if step > 1 else 0.0)
            temps = [8 * torch.sigmoid(l.temperature).item() for l in reader_layers(model) if hasattr(l, "temperature")]
            sinks = [l.sink.item() for l in reader_layers(model) if hasattr(l, "sink")]
            if temps:
                rec["temp"] = sum(temps) / len(temps)
            if sinks:
                rec["sink_logit"] = sum(sinks) / len(sinks)
            if rank == 0 and (step % args.eval_every == 0 or step == args.steps):
                rec.update(evaluate(model, x_val, use_amp))
                if ema is not None:
                    ema.swap()
                    rec.update({k + "_ema": v for k, v in evaluate(model, x_val, use_amp).items()})
                    ema.swap()
            if rank == 0:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            t_last = time.time()
        if rank == 0 and (step % args.ckpt_every == 0 or step == args.steps):
            ck = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=dataclasses.asdict(cfg), args=vars(args),
                      ema=None if ema is None else ema.shadow, n_skipped=n_skip)
            torch.save(ck, os.path.join(args.out, "last.pt.tmp"))
            os.replace(os.path.join(args.out, "last.pt.tmp"), os.path.join(args.out, "last.pt"))
    if world > 1:                                                  # the hand-reduced replicas must still agree
        h = torch.stack([p.detach().float().sum() for p in params]).sum().view(1)
        hs = [torch.zeros_like(h) for _ in range(world)]
        dist.all_gather(hs, h)
        if rank == 0 and max(abs((x - hs[0]).item()) for x in hs) > 1e-3 * (1 + abs(hs[0].item())):
            print("WARNING: replicas diverged", [x.item() for x in hs], flush=True)
    if base.is_dist():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
