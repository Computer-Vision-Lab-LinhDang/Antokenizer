#!/usr/bin/env python3
"""Held-out reconstruction of a C-D 4.1 checkpoint (scripts/train_cd41.py): what the transport, Δ-Tucker and the
innovation slots each carry, at the payload actually serialized.

  python scripts/eval_cd41.py --ckpt <run>/last.pt --imagenet <root> --openvid <manifest> --objaverse <root> \\
      --objaverse-depth --omniworld <pack> --omniworld-modality world --dino-repo <local copy> --out <run>/eval
  python scripts/eval_cd41.py --dry-run [--ckpt <dry-run last.pt>] --out /tmp/cd41_eval     # CPU, tiny model

Same data flags as scripts/eval_cdv.py. Images: PSNR, SSIM, LPIPS, rFID. Clips (video, posed views, world
clips), per observation kind: the anchor, and the targets decoded from the serialized state Z
  full       Z as encoded (L = n_slots)
  L curve    Z with the innovation cut to L slots for each L of --L-list (0 = base only: transport + Δ-Tucker)
  no Δ       Z with a_i = 0 at decode (how much of the target the Tucker term carries, given the same slots)
  copy       the anchor frame as the target (no model)
plus the payload counted by serializing Z, and the model diagnostics (support, cosine of H^base on cells with
and without support, tucker_rel, camera error against the given cameras, rho_gt_corr, flow on moving cells).
Outputs are clamped to [-1, 1]. Writes <out>/<stem>.json and .md.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import eval_cdv as ev  # noqa: E402
from mavt.dino_v0 import evaluate as E  # noqa: E402

DIAG = ("support_frac", "base_cos_sup", "base_cos_new", "warp_cos_sup", "feat_cos", "tucker_rel", "a_norm",
        "psnr_base", "xi_rot_deg", "xi_trans", "fov_deg", "rho_std", "rho_gt_corr", "cam_rot_err_deg",
        "cam_trans_err", "cam_trans_gt", "flow_dyn", "flow_static", "dyn_frac")


def build_parser():
    p = ev.build_parser()
    p.description = __doc__
    p.set_defaults(L_list="0,8,16,32,64,max")
    return p


def _scores(rgb, tgt, lp) -> dict:
    rgb, tgt = rgb.float().clamp(-1, 1).flatten(0, 1), tgt.float().flatten(0, 1)
    out = dict(psnr=E.psnr(rgb, tgt), ssim=E.ssim(rgb, tgt))
    if lp.available:
        out["lpips"] = lp(rgb, tgt)
    return out


class Acc:
    def __init__(self):
        self.m = {}

    def add(self, name: str, scores: dict):
        for k, v in scores.items():
            self.m.setdefault(name, {}).setdefault(k, ev.Mean()).add_t(v)

    def result(self) -> dict:
        return {n: {k: m.mean for k, m in d.items() if m.n} for n, d in self.m.items()}


@torch.no_grad()
def eval_clips(model, loader, modality: str, dev, amp: bool, limit: int, L_list: list, lp) -> dict:
    from mavt.cd41 import state as st
    cfg = model.cfg
    acc, dg, pay = Acc(), {}, ev.Mean()
    seen, t0, T = 0, time.time(), 0
    for b in loader:
        x = b["data"].to(dev, non_blocking=True)                                   # [B, 3, T, H, W]
        B, T = x.shape[0], x.shape[2]
        cam = {k: b[k].to(dev) for k in ("c2w", "K", "depth", "dyn") if k in b} if modality != "video" else None
        with ev._autocast(dev, amp):
            out = model(x, modality, cam, return_outputs=True)
        o = out["outputs"]
        Z = o["Z"]
        n = Z["xi"].shape[1] + 1
        frames = x.permute(0, 2, 1, 3, 4).reshape(-1, n, *x.shape[1:2], *x.shape[3:])  # [B', n, 3, H, W]
        rgb = o["rgb"].reshape(frames.shape)
        acc.add("anchor", _scores(rgb[:, :1], frames[:, :1], lp))
        acc.add("target", _scores(rgb[:, 1:], frames[:, 1:], lp))
        acc.add("copy", _scores(frames[:, :1].expand_as(frames[:, 1:]), frames[:, 1:], lp))
        flats = st.serialize(cfg, Z)
        pay.add(sum(f.numel() for f in flats) / len(flats), len(flats))
        for L in L_list:
            Zl = dict(Z, L=torch.full_like(Z["L"], L), r=Z["r"].clone())
            Zl["r"][:, :, L:] = 0
            with ev._autocast(dev, amp):
                d = model.decode(Zl)
            acc.add(f"L={L}", _scores(d["rgb"][:, 1:], frames[:, 1:], lp))
            acc.add(f"L={L}/numbers", {"n": st.payload_numbers(cfg, Zl)})
        if Z.get("a") is not None:
            with ev._autocast(dev, amp):
                d = model.decode(dict(Z, a=torch.zeros_like(Z["a"])))
            acc.add("no_tucker", _scores(d["rgb"][:, 1:], frames[:, 1:], lp))
        for k in DIAG:
            v = out["diag"].get(k)
            if v is not None and torch.isfinite(v):
                dg.setdefault(k, ev.Mean()).add(v.item(), B)
        seen += B
        if limit and seen >= limit:
            break
    res = acc.result()
    curve = [dict(L=L, numbers=res[f"L={L}/numbers"]["n"], **res[f"L={L}"]) for L in L_list]
    return dict(n_clips=seen, n_frames=T, anchor=res["anchor"], target=res["target"], copy=res["copy"],
                no_tucker=res.get("no_tucker"), curve=curve, payload=pay.mean,
                diag={k: m.mean for k, m in dg.items()}, seconds=time.time() - t0)


def _f(v, nd=2) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def to_markdown(res: dict) -> str:
    lines = [f"# C-D 4.1 eval, step {res['step']}", "",
             f"tucker={res['tucker']} n_slots={res['n_slots']} segment ledger={res['segment_numbers']} "
             f"image ledger={res['image_numbers']}", ""]
    if "images" in res:
        im = res["images"]
        lines += ["## Images", "", "| n | PSNR | SSIM | LPIPS | rFID | numbers |", "|---|---|---|---|---|---|",
                  f"| {im['n_images']} | {_f(im.get('psnr'))} | {_f(im.get('ssim'), 3)} | {_f(im.get('lpips'), 3)} "
                  f"| {_f(im.get('rfid'))} | {_f(im.get('numbers'), 0)} |", ""]
    for name, v in res.get("clips", {}).items():
        lines += [f"## {name} ({v['n_clips']} clips, payload {_f(v['payload'], 0)})", "",
                  "| decode | PSNR | SSIM | LPIPS | numbers |", "|---|---|---|---|---|"]
        rows = [("anchor", v["anchor"], None), ("target, full Z", v["target"], v["payload"])]
        rows += [(f"target, L = {r['L']}", r, r["numbers"]) for r in v["curve"]]
        if v.get("no_tucker"):
            rows.append(("target, a = 0 at decode", v["no_tucker"], None))
        rows.append(("copy anchor (no model)", v["copy"], None))
        for label, r, num in rows:
            lines.append(f"| {label} | {_f(r.get('psnr'))} | {_f(r.get('ssim'), 3)} | {_f(r.get('lpips'), 3)} "
                         f"| {_f(num, 0)} |")
        d = v["diag"]
        lines += ["", " ".join(f"{k}={_f(d[k], 3)}" for k in DIAG if k in d), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ckpt and not args.dry_run:
        raise SystemExit("give --ckpt (or --dry-run)")
    from mavt.cd41 import MAVTCD41, load_cd41, tiny_cd41_config
    dev = torch.device("cpu" if args.dry_run else args.device)
    amp = args.bf16 and not args.dry_run
    torch.manual_seed(args.seed)
    overrides = {}
    if args.set:
        from train_cd32 import parse_sets
        overrides = parse_sets(args.set)
    if args.dino_repo:
        overrides["dino_repo"] = args.dino_repo
    if args.ckpt:
        model, cfg, step = load_cd41(args.ckpt, dev, **overrides)
    else:
        cfg = tiny_cd41_config(**overrides)
        model, step = MAVTCD41(cfg).to(dev).eval(), -1
    L_list = ev.parse_L(args.L_list, cfg.n_slots)
    lp, fid = E.OptionalLPIPS(dev), E.OptionalFID(dev)
    res = dict(ckpt=args.ckpt, step=step, tucker=cfg.tucker, n_slots=cfg.n_slots, L_list=L_list,
               segment_numbers=cfg.segment_numbers("video"), image_numbers=cfg.segment_numbers("image"),
               notes=[lp.reason, fid.reason], clips={})
    K = cfg.keyframe_every
    if args.dry_run:
        n, bs = 2, 2
        res["images"] = ev.eval_images(model, ev.synthetic_loader("image", cfg, 1, bs, n, args.seed), dev, amp, 0, lp, fid)
        for m in ("video", "threed", "world"):
            ld = ev.synthetic_loader(m, cfg, K, bs, n, args.seed)
            res["clips"][m] = eval_clips(model, ld, m, dev, amp, 0, L_list, lp)
        if args.omniworld:                                                         # a tiny pack (tests)
            args.num_workers = 0
            res["clips"]["omniworld_" + args.omniworld_modality] = eval_clips(
                model, ev.omniworld_loader(args, cfg.image_size), args.omniworld_modality, dev, amp,
                args.omniworld_limit, L_list, lp)
    else:
        if args.imagenet:
            ld = ev.imagenet_loader(args.imagenet, cfg.image_size, args.batch, args.num_workers, args.image_limit,
                                    args.seed)
            res["images"] = ev.eval_images(model, ld, dev, amp, args.image_limit, lp, fid)
            print(f"images: {json.dumps({k: v for k, v in res['images'].items() if k != 'diag'})}", flush=True)
        if args.openvid:
            for man in [m for m in args.openvid.split(",") if m.strip()]:
                name = "video " + os.path.basename(man.rstrip("/")).split(".")[0]
                res["clips"][name] = eval_clips(model, ev.video_loader(man, args, cfg.image_size), "video", dev, amp,
                                                args.video_limit, L_list, lp)
                print(f"{name}: {json.dumps(res['clips'][name]['target'])}", flush=True)
        if args.objaverse:
            res["clips"]["objaverse static_views"] = eval_clips(model, ev.objaverse_loader(args, cfg.image_size),
                                                                "threed", dev, amp, args.objaverse_limit, L_list, lp)
        if args.omniworld:
            name = "omniworld " + args.omniworld_modality
            res["clips"][name] = eval_clips(model, ev.omniworld_loader(args, cfg.image_size), args.omniworld_modality,
                                            dev, amp, args.omniworld_limit, L_list, lp)
            res["omniworld_stride"] = args.omniworld_stride
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
