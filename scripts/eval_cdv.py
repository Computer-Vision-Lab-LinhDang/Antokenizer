#!/usr/bin/env python3
"""Held-out reconstruction of a C-D 3.2 checkpoint (vanilla or two-stream) at the full budget.

  python scripts/eval_cdv.py --ckpt <run>/last.pt --imagenet <parquet root> --openvid <manifest>[,<manifest>] \\
      --objaverse <render root> --objaverse-depth --dino-repo <local copy> --out <run>/eval
  python scripts/eval_cdv.py --dry-run [--ckpt <dry-run last.pt>] --out /tmp/cdv_eval      # CPU, tiny model

Images (ImageNet validation, --image-limit random images): PSNR, SSIM, LPIPS and rFID of the reconstruction
(the `lpips` / `torchmetrics` packages are optional). Video (the held-out tail of each manifest, clips of
--n-frames frames at --frame-stride): per-frame PSNR of the key frames (C only) and of the frames after them
(C + D), the copy-the-keyframe baseline on the same frames (no model: is the geometry-only decode better than
doing nothing?), and the PSNR-vs-L curve: every frame after a keyframe decoded with L innovation slots for
each L of --L-list (0 = geometry only: C + pose + flow; max = L_max), with the numbers a segment then costs.
Diagnostics from the same pass: xi_rot_deg, xi_trans, flow_l1, rho_std, fov_deg. Posed views (Objaverse
hold-out, view 0 = keyframe, --n-views views): the same curve; with --objaverse-depth also rho_gt_corr, the
Pearson correlation of rho with the rendered depth. Outputs are clamped to [-1, 1] before scoring.
The checkpoint kind is read from its config (`dec_blocks` = vanilla). Writes <out>/<stem>.json and .md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from mavt.dino_v0 import evaluate as E  # noqa: E402

DIAG_KEYS = ("L_mean", "xi_rot_deg", "xi_trans", "fov_deg", "rho_std", "rho_gt_corr", "rho_probe_corr",
             "psnr_geo", "lam_d", "lam_z")
PART_KEYS = ("rec_rgb", "rec_feat", "geo_rgb", "geo_feat", "flow_l1", "smooth", "depth_gt", "kl_c", "kl_d")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="checkpoint of scripts/train_cdv.py or scripts/train_cd32.py")
    p.add_argument("--dino-repo", default=None, help="override the checkpoint's dino_repo (local copy)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    p.add_argument("--imagenet", default=None, help="ImageNet parquet root (validation split)")
    p.add_argument("--image-limit", type=int, default=5000, help="random validation images (0: all)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--openvid", default=None, help="comma-separated video manifests (OpenVid, SSv2 ...)")
    p.add_argument("--video-tail", type=int, default=1000, help="held-out manifest tail (as in training)")
    p.add_argument("--video-limit", type=int, default=200, help="clips per manifest")
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--batch-video", type=int, default=16)
    p.add_argument("--objaverse", default=None, help="Objaverse render root")
    p.add_argument("--objaverse-holdout", type=int, default=500)
    p.add_argument("--objaverse-depth", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--objaverse-limit", type=int, default=200, help="objects")
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--batch-3d", type=int, default=16)
    p.add_argument("--L-list", default="0,1,2,4,8,16,32,max", help="innovation lengths of the curve")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--stem", default="eval")
    p.add_argument("--dry-run", action="store_true", help="CPU, synthetic data, tiny model unless --ckpt")
    return p


# ---------------------------------------------------------------------------------------------- model
def checkpoint_kind(path: str) -> str:
    """'vanilla' (scripts/train_cdv.py, config has dec_blocks) or 'two-stream' (scripts/train_cd32.py)."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001  (legacy format)
        ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("cfg", {}))
    del ck
    return "vanilla" if "dec_blocks" in cfg else "two-stream"


def load_any(path: str, dev, **overrides):
    kind = checkpoint_kind(path)
    if kind == "vanilla":
        from mavt.cd32.vanilla_model import load_cdv as loader
    else:
        from mavt.cd32.model import load_model as loader
    model, cfg, step = loader(path, dev, **overrides)
    return model, cfg, step, kind


def parse_L(spec: str, L_max: int) -> list:
    out = []
    for s in spec.split(","):
        s = s.strip()
        if s:
            v = min(L_max if s == "max" else int(s), L_max)
            if v not in out:
                out.append(v)
    return sorted(out) or [L_max]


# ---------------------------------------------------------------------------------------------- metrics
class Mean:
    def __init__(self):
        self.s, self.n = 0.0, 0

    def add(self, v: float, n: int = 1):
        if n > 0:
            self.s, self.n = self.s + float(v) * n, self.n + n

    def add_t(self, t: torch.Tensor):
        self.add(t.float().mean().item(), t.numel())

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else float("nan")


class DiagMean:
    """Weighted means of the scalar diag / parts entries of the model output."""

    def __init__(self):
        self.d = {}

    def add(self, out: dict, n: int):
        for src, keys in ((out["diag"], DIAG_KEYS), (out["parts"], PART_KEYS)):
            for k in keys:
                v = src.get(k)
                if v is not None and torch.is_tensor(v) and v.numel() == 1 and torch.isfinite(v):
                    self.d.setdefault(k, Mean()).add(v.item(), n)

    def result(self) -> dict:
        return {k: m.mean for k, m in self.d.items()}


def _autocast(dev, on: bool):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=on)


def _frames(t: torch.Tensor) -> torch.Tensor:
    return t.flatten(0, 1) if t.dim() == 5 else t


@torch.no_grad()
def eval_images(model, loader, dev, amp: bool, limit: int, lp, fid) -> dict:
    m = {k: Mean() for k in ("psnr", "ssim", "lpips")}
    dg, seen, t0, out = DiagMean(), 0, time.time(), None
    for b in loader:
        x = b["data"].to(dev, non_blocking=True)
        with _autocast(dev, amp):
            out = model(x, "image", return_outputs=True)
        rgb = _frames(out["outputs"]["rgb"]).float().clamp(-1, 1)
        m["psnr"].add_t(E.psnr(rgb, x))
        m["ssim"].add_t(E.ssim(rgb, x))
        if lp.available:
            m["lpips"].add_t(lp(rgb, x))
        if fid.available:
            fid.update(x, rgb)
        dg.add(out, x.shape[0])
        seen += x.shape[0]
        if limit and seen >= limit:
            break
    res = {k: v.mean for k, v in m.items() if v.n}
    res.update(n_images=seen, numbers=float(out["diag"]["numbers"]) if out else None, diag=dg.result(),
               seconds=time.time() - t0)
    if fid.available and seen >= 2:
        res["rfid"] = fid.compute()
    return res


@torch.no_grad()
def eval_clips(model, loader, modality: str, dev, amp: bool, limit: int, L_list: list, lp) -> dict:
    """Key frames vs the frames after them, per L of the curve (threed: view 0 is the keyframe)."""
    K = model.cfg.keyframe_every
    curve = {L: {k: Mean() for k in ("psnr", "ssim", "lpips", "numbers")} for L in L_list}
    key = {k: Mean() for k in ("psnr", "ssim", "lpips")}
    copy = {k: Mean() for k in ("psnr", "ssim", "lpips")}                # baseline: the keyframe as frame t
    dg, seen, T, t0 = DiagMean(), 0, 0, time.time()
    for b in loader:
        x = b["data"].to(dev, non_blocking=True)                       # [B, 3, T, H, W]
        B, T = x.shape[0], x.shape[2]
        cam = None
        t = torch.arange(T, device=dev)
        if modality == "threed":
            cam = {k: b[k].to(dev) for k in ("c2w", "K", "depth") if k in b}
            key_of = torch.zeros_like(t)
            L_shape = (B, T - 1)
        else:
            key_of = t - t % K
            L_shape = (B * (T // K), K - 1)
        is_key = key_of == t
        frames = x.permute(0, 2, 1, 3, 4)
        for L in L_list:
            Lt = torch.full(L_shape, L, dtype=torch.long, device=dev)
            with _autocast(dev, amp):
                out = model(x, modality, cam, L=Lt, return_outputs=True)
            o = out["outputs"]
            tgt = frames
            if "order" in o:                                            # posed views come back reordered
                tgt = torch.gather(frames, 1, o["order"].view(B, T, 1, 1, 1).expand_as(frames))
            rgb = o["rgb"].float().clamp(-1, 1)
            p = E.psnr(rgb.flatten(0, 1), tgt.flatten(0, 1)).view(B, T)
            s = E.ssim(rgb.flatten(0, 1), tgt.flatten(0, 1)).view(B, T)
            nk = ~is_key
            c = curve[L]
            c["psnr"].add_t(p[:, nk])
            c["ssim"].add_t(s[:, nk])
            if lp.available:
                c["lpips"].add_t(lp(rgb[:, nk].flatten(0, 1), tgt[:, nk].flatten(0, 1)))
            c["numbers"].add(float(out["diag"]["numbers"]), B)
            if L == L_list[-1]:                                          # key frames do not depend on L
                key["psnr"].add_t(p[:, is_key])
                key["ssim"].add_t(s[:, is_key])
                if lp.available:
                    key["lpips"].add_t(lp(rgb[:, is_key].flatten(0, 1), tgt[:, is_key].flatten(0, 1)))
                src, dst = tgt[:, key_of[nk]].flatten(0, 1), tgt[:, nk].flatten(0, 1)
                copy["psnr"].add_t(E.psnr(src, dst))
                copy["ssim"].add_t(E.ssim(src, dst))
                if lp.available:
                    copy["lpips"].add_t(lp(src, dst))
                dg.add(out, B)
        seen += B
        if limit and seen >= limit:
            break
    rows = [dict(L=L, **{k: m.mean for k, m in c.items() if m.n}) for L, c in curve.items()]
    return dict(n_clips=seen, n_frames=T, key={k: m.mean for k, m in key.items() if m.n},
                copy_key={k: m.mean for k, m in copy.items() if m.n}, curve=rows, diag=dg.result(),
                seconds=time.time() - t0)


# ---------------------------------------------------------------------------------------------- data
def imagenet_loader(root: str, size: int, batch: int, workers: int, limit: int, seed: int):
    from mavt.dino_v0.data import ImageNetParquet
    ds = ImageNetParquet(root, "validation", size, train=False)
    if limit and limit < len(ds):
        idx = sorted(random.Random(seed).sample(range(len(ds)), limit))   # sorted: row-group locality
        ds = torch.utils.data.Subset(ds, idx)
    return E.make_loader(ds, batch, workers)


def video_loader(manifest: str, args, size: int):
    from mavt.dino_v0.data import OpenVidClips, collate
    ds = OpenVidClips(manifest, args.n_frames, args.frame_stride, size, train=False, tail=args.video_tail,
                      split="heldout", eval_stride=args.frame_stride)
    ds._rng = random.Random(args.seed)                                   # repeatable clip starts
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_video, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=collate, pin_memory=torch.cuda.is_available())


def objaverse_loader(args, size: int):
    from mavt.dino_v0.data import ObjaverseViews, collate
    ds = ObjaverseViews(args.objaverse, n_views=args.n_views, size=size, train=False,
                        holdout=args.objaverse_holdout, depth=args.objaverse_depth)
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_3d, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=collate, pin_memory=torch.cuda.is_available())


def synthetic_loader(modality: str, cfg, n_frames: int, batch: int, n_batches: int, seed: int):
    from train_dino_v0 import synthetic_iter
    return list(itertools.islice(synthetic_iter(modality, batch, cfg, n_frames, seed, depth=True), n_batches))


# ---------------------------------------------------------------------------------------------- report
def _f(v, nd=2) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _diag_line(d: dict) -> str:
    keys = [k for k in DIAG_KEYS + PART_KEYS if k in d and k not in ("psnr_geo",)]
    return ", ".join(f"{k} {_f(d[k], 3)}" for k in keys)


def _curve_md(v: dict) -> list:
    lines = ["", "| L | numbers | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for r in v["curve"]:
        lines.append(f"| {r['L']} | {_f(r.get('numbers'), 0)} | {_f(r.get('psnr'))} | {_f(r.get('ssim'), 3)} | "
                     f"{_f(r.get('lpips'), 3)} |")
    k, c = v["key"], v["copy_key"]
    lines.append("")
    lines.append(f"Key frames (C only): PSNR {_f(k.get('psnr'))}, SSIM {_f(k.get('ssim'), 3)}, "
                 f"LPIPS {_f(k.get('lpips'), 3)}. Frames after the keyframe at L = max minus L = 0: "
                 f"{_f(v['curve'][-1].get('psnr', 0) - v['curve'][0].get('psnr', 0))} dB.")
    lines.append(f"Baseline, the keyframe copied as the frame (no model): PSNR {_f(c.get('psnr'))}, "
                 f"SSIM {_f(c.get('ssim'), 3)}, LPIPS {_f(c.get('lpips'), 3)}; L = 0 above it by "
                 f"{_f(v['curve'][0].get('psnr', 0) - c.get('psnr', 0))} dB.")
    if v["diag"]:
        lines.append("")
        lines.append(_diag_line(v["diag"]))
    return lines


def to_markdown(res: dict) -> str:
    L = [f"# Held-out reconstruction: {res['kind']} model, step {res['step']}", "",
         f"checkpoint `{os.path.basename(res['ckpt']) if res['ckpt'] else 'none (untrained tiny model)'}`, "
         f"L_max {res['L_max']}, keyframe every "
         f"{res['keyframe_every']}, budget {res['budget']} numbers per segment"]
    if "images" in res:
        v = res["images"]
        L += ["", f"## Images (ImageNet validation, n = {v['n_images']}, {_f(v['numbers'], 0)} numbers)", "",
              "| PSNR | SSIM | LPIPS | rFID |", "|---|---|---|---|",
              f"| {_f(v.get('psnr'))} | {_f(v.get('ssim'), 3)} | {_f(v.get('lpips'), 3)} | {_f(v.get('rfid'))} |"]
        if v["diag"]:
            L += ["", _diag_line(v["diag"])]
    for name, v in res.get("video", {}).items():
        L += ["", f"## Video: {name} (n = {v['n_clips']} clips of {v['n_frames']} frames, stride "
                  f"{res['frame_stride']}; PSNR / SSIM / LPIPS on the frames after the keyframe)"]
        L += _curve_md(v)
    if "threed" in res:
        v = res["threed"]
        L += ["", f"## Posed views: Objaverse hold-out (n = {v['n_clips']} objects of {v['n_frames']} views; "
                  f"view 0 is the keyframe)"]
        L += _curve_md(v)
    notes = [n for n in res.get("notes", []) if n]
    if notes:
        L += ["", *[f"- {n}" for n in notes]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ckpt and not args.dry_run:
        raise SystemExit("give --ckpt (or --dry-run)")
    dev = torch.device("cpu" if args.dry_run else args.device)
    amp = args.bf16 and not args.dry_run
    torch.manual_seed(args.seed)
    overrides = {}
    if args.set:
        from train_cd32 import _tuples, parse_sets
        overrides = _tuples(parse_sets(args.set))
    if args.dino_repo:
        overrides["dino_repo"] = args.dino_repo
    if args.ckpt:
        model, cfg, step, kind = load_any(args.ckpt, dev, **overrides)
    else:
        from mavt.cd32.vanilla import tiny_cdv_config
        from mavt.cd32.vanilla_model import MAVTCDV
        cfg = tiny_cdv_config(**overrides)
        model, step, kind = MAVTCDV(cfg).to(dev).eval(), -1, "vanilla"
    L_list = parse_L(args.L_list, cfg.L_max)
    lp, fid = E.OptionalLPIPS(dev), E.OptionalFID(dev)
    res = dict(ckpt=args.ckpt, step=step, kind=kind, L_max=cfg.L_max, keyframe_every=cfg.keyframe_every,
               budget=cfg.budget, L_list=L_list, frame_stride=args.frame_stride, notes=[lp.reason, fid.reason])
    if args.dry_run:
        n, bs = 2, 2
        res["images"] = eval_images(model, synthetic_loader("image", cfg, 1, bs, n, args.seed), dev, amp, 0, lp, fid)
        res["video"] = {"synthetic": eval_clips(model, synthetic_loader("video", cfg, cfg.keyframe_every, bs, n,
                                                                         args.seed), "video", dev, amp, 0, L_list, lp)}
        res["threed"] = eval_clips(model, synthetic_loader("threed", cfg, args.n_views, bs, n, args.seed), "threed",
                                   dev, amp, 0, L_list, lp)
    else:
        if args.imagenet:
            ld = imagenet_loader(args.imagenet, cfg.image_size, args.batch, args.num_workers, args.image_limit, args.seed)
            res["images"] = eval_images(model, ld, dev, amp, args.image_limit, lp, fid)
            print(f"images: {json.dumps({k: v for k, v in res['images'].items() if k != 'diag'})}", flush=True)
        if args.openvid:
            res["video"] = {}
            for man in [m for m in args.openvid.split(",") if m.strip()]:
                name = os.path.basename(man.rstrip("/")).split(".")[0]
                res["video"][name] = eval_clips(model, video_loader(man, args, cfg.image_size), "video", dev, amp,
                                                args.video_limit, L_list, lp)
                print(f"video {name}: {json.dumps(res['video'][name]['curve'])}", flush=True)
        if args.objaverse:
            res["threed"] = eval_clips(model, objaverse_loader(args, cfg.image_size), "threed", dev, amp,
                                       args.objaverse_limit, L_list, lp)
            print(f"threed: {json.dumps(res['threed']['curve'])}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, args.stem + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    md = to_markdown(res)
    with open(os.path.join(args.out, args.stem + ".md"), "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
