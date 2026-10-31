#!/usr/bin/env python3
"""Held-out reconstruction of a Grain-Prior C-D v1 checkpoint (scripts/train_grain_prior.py), decoded from the
serialized state alone, with the same data flags and scores as scripts/eval_cd41.py.

  python scripts/eval_grain_prior.py --ckpt <run>/last.pt --imagenet <root> --openvid <manifest> \\
      --objaverse <root> --objaverse-depth --omniworld <pack> --omniworld-modality world --dino-repo <local copy> \\
      --out <run>/eval
  python scripts/eval_grain_prior.py --dry-run [--ckpt <dry-run last.pt>] --out /tmp/gp_eval     # CPU, tiny model

Images: PSNR, SSIM, LPIPS, rFID at the image ledger. Clips (video, posed views, world clips), per observation:
the anchor, and the targets of the closed state after a serialize / deserialize round trip,
  full       the state as encoded (L = n_slots)
  L curve    the state with every record cut to L innovation slots (0 = base only: grains + memory + camera)
  copy       the anchor frame as the target (no model)
  copy last  the previous frame as the target (no model)
  novel view grains + c_0 + the record's camera / motion only: no memory update, no innovation (eval_baselines)
  by record  the full-state score of each record of the queue (distance from the anchor)
  gates      the two learning gates, in dB: L0_minus_copy_last_db (video: does the prediction branch beat carrying the
             last frame forward?) and nvs_minus_copy_db (posed views: does the geometry alone beat copying the anchor?)
plus the payload counted by serializing the state, and the model diagnostics (base cosine per refinement
pass, initializer cosine, support, depth, cameras). World clips are scored twice: with the given cameras
copied (calibrated video) and with estimated cameras (--world-camera).
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
from mavt.grain_prior.eval_baselines import copy_last, no_memory, strip_target_posterior  # noqa: E402

DIAG = ("init_cos", "base_cos_pass0", "base_cos_pass1", "base_cos_final", "feat_cos", "support_frac", "psnr_base",
        "xi_rot_deg", "xi_trans", "fov_deg", "rho_std", "rho_gt_corr", "depth_grain_err", "depth_dense_err",
        "cam_rot_err_deg", "cam_trans_err", "mem_delta", "sigma_mean", "psnr_copy_last", "splat_dropped_frac")


def build_parser():
    p = ev.build_parser()
    p.description = __doc__
    p.add_argument("--world-camera", choices=("both", "given", "estimated"), default="both")
    p.set_defaults(L_list="0,8,16,32,max")
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
    from mavt.grain_prior import schema as sc
    cfg = model.cfg
    acc, dg, pay = Acc(), {}, ev.Mean()
    seen, t0, T = 0, time.time(), 0
    for b in loader:
        x = b["data"].to(dev, non_blocking=True)                                   # [B, 3, T, H, W]
        B, T = x.shape[0], x.shape[2]
        cam = {k: b[k].to(dev) for k in ("c2w", "K", "depth", "dyn") if k in b} if modality != "video" else None
        with ev._autocast(dev, amp):
            out = model(x, modality, cam, return_outputs=True)
        S = out["outputs"]["S"]
        frames = x.permute(0, 2, 1, 3, 4)                                          # [B, T, 3, H, W]
        q = len(S["recs"])
        tgt = frames[:, T - q:]
        flats = sc.serialize(cfg, S)
        pay.add(sum(f.numel() for f in flats) / len(flats), len(flats))
        S = sc.deserialize(cfg, [f.to(dev) for f in flats])                         # the decoder sees only this
        with ev._autocast(dev, amp):
            d = model.decode(S)
        acc.add("anchor", _scores(d["rgb"][:, :1], frames[:, :1], lp))
        acc.add("target", _scores(d["rgb"][:, 1:], tgt, lp))
        acc.add("copy", _scores(frames[:, :1].expand_as(tgt), tgt, lp))
        acc.add("copy_last", _scores(copy_last(frames, q), tgt, lp))
        for j in range(q):
            acc.add(f"rec{j}", _scores(d["rgb"][:, 1 + j:2 + j], tgt[:, j:j + 1], lp))
        with ev._autocast(dev, amp), no_memory(model):
            dn = model.decode(strip_target_posterior(cfg, S))
        acc.add("nvs", _scores(dn["rgb"][:, 1:], tgt, lp))
        for L in L_list:
            recs = []
            for rec in S["recs"]:
                Lr = torch.minimum(rec["L"], torch.full_like(rec["L"], L))
                r = rec["r"].clone()
                r[:, L:] = 0
                recs.append(dict(rec, L=Lr, r=r))
            Sl = dict(S, recs=recs, h=S["h"].clone())
            for j, rec in enumerate(recs):
                Sl["h"][:, sc.H_LEN + j] = rec["L"].float()
            with ev._autocast(dev, amp):
                dl = model.decode(Sl)
            acc.add(f"L={L}", _scores(dl["rgb"][:, 1:], tgt, lp))
            acc.add(f"L={L}/numbers", {"n": sc.payload(cfg, Sl).float().mean()})
            if L == 0 and cfg.grain_df_k:                      # 3.13: the same without the grain feature update
                Sn = dict(Sl, recs=[dict(r, dfc=torch.zeros_like(r["dfc"])) for r in recs])
                with ev._autocast(dev, amp):
                    acc.add("L=0/no_df", _scores(model.decode(Sn)["rgb"][:, 1:], tgt, lp))
        for k in DIAG:
            v = out["diag"].get(k)
            if v is not None and torch.isfinite(v):
                dg.setdefault(k, ev.Mean()).add(v.item(), B)
        seen += B
        if limit and seen >= limit:
            break
    res = acc.result()
    curve = [dict(L=L, numbers=res[f"L={L}/numbers"]["n"], **res[f"L={L}"]) for L in L_list]
    q = len([k for k in res if k.startswith("rec")])
    L0 = res.get("L=0")
    gates = dict(L0_minus_copy_last_db=(L0["psnr"] - res["copy_last"]["psnr"]) if L0 else None,
                 nvs_minus_copy_db=res["nvs"]["psnr"] - res["copy"]["psnr"])
    if "L=0/no_df" in res:                                     # 3.13: gate 1 read without the grain feature update
        gates["L0_no_df_minus_copy_last_db"] = res["L=0/no_df"]["psnr"] - res["copy_last"]["psnr"]
    return dict(n_clips=seen, n_frames=T, anchor=res["anchor"], target=res["target"], copy=res["copy"],
                copy_last=res["copy_last"], nvs=res["nvs"], target_by_record=[res[f"rec{j}"] for j in range(q)],
                gates=gates, curve=curve, payload=pay.mean, diag={k: m.mean for k, m in dg.items()},
                seconds=time.time() - t0)


def _f(v, nd=2) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _nvs_numbers(v: dict):
    """Payload of the novel-view decode: the L = 0 entry of the curve when present."""
    for r in v.get("curve", []):
        if r["L"] == 0:
            return r["numbers"]
    return None


def to_markdown(res: dict) -> str:
    lines = [f"# Grain-Prior C-D v1 eval, step {res['step']}", "",
             f"query_init={res['query_init']} geom_bias={res['geom_bias']} memory={res['memory']} "
             f"image ledger={res['image_numbers']} full state={res['state_numbers']}", ""]
    if "images" in res:
        im = res["images"]
        lines += ["## Images", "", "| n | PSNR | SSIM | LPIPS | rFID | numbers |", "|---|---|---|---|---|---|",
                  f"| {im['n_images']} | {_f(im.get('psnr'))} | {_f(im.get('ssim'), 3)} | {_f(im.get('lpips'), 3)} "
                  f"| {_f(im.get('rfid'))} | {_f(im.get('numbers'), 0)} |", ""]
    for name, v in res.get("clips", {}).items():
        lines += [f"## {name} ({v['n_clips']} clips, payload {_f(v['payload'], 0)})", "",
                  "| decode | PSNR | SSIM | LPIPS | numbers |", "|---|---|---|---|---|"]
        rows = [("anchor", v["anchor"], None), ("target, full state", v["target"], v["payload"])]
        rows += [(f"target, L = {r['L']}", r, r["numbers"]) for r in v["curve"]]
        if "nvs" in v:
            rows.append(("novel view: grains + camera only (z = c_0, L = 0)", v["nvs"], _nvs_numbers(v)))
        rows.append(("copy anchor (no model)", v["copy"], None))
        if "copy_last" in v:
            rows.append(("copy last frame (no model)", v["copy_last"], None))
        for label, r, num in rows:
            lines.append(f"| {label} | {_f(r.get('psnr'))} | {_f(r.get('ssim'), 3)} | {_f(r.get('lpips'), 3)} "
                         f"| {_f(num, 0)} |")
        if v.get("target_by_record"):
            lines.append("")
            lines.append("target by record (full state), PSNR: " + " / ".join(_f(r.get("psnr")) for r in v["target_by_record"]))
        if v.get("gates"):
            g_ = v["gates"]
            lines.append("")
            lines.append(f"gates: L0 - copy-last = {_f(g_.get('L0_minus_copy_last_db'))} dB; "
                         f"novel view - copy anchor = {_f(g_.get('nvs_minus_copy_db'))} dB")
        d = v["diag"]
        lines += ["", " ".join(f"{k}={_f(d[k], 3)}" for k in DIAG if k in d), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ckpt and not args.dry_run:
        raise SystemExit("give --ckpt (or --dry-run)")
    from mavt.grain_prior import MAVTGrainPrior, load_grain_prior, tiny_config
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
        model, cfg, step = load_grain_prior(args.ckpt, dev, **overrides)
    else:
        cfg = tiny_config(**overrides)
        model, step = MAVTGrainPrior(cfg).to(dev).eval(), -1
    L_list = ev.parse_L(args.L_list, cfg.n_slots)
    lp, fid = E.OptionalLPIPS(dev), E.OptionalFID(dev)
    res = dict(ckpt=args.ckpt, step=step, query_init=cfg.query_init, geom_bias=cfg.geom_bias, memory=cfg.memory,
               n_slots=cfg.n_slots, L_list=L_list, image_numbers=cfg.state_numbers([]),
               state_numbers=cfg.state_numbers([cfg.n_slots] * cfg.queue), notes=[lp.reason, fid.reason], clips={})
    worlds = {"both": ("given", "estimated"), "given": ("given",), "estimated": ("estimated",)}[args.world_camera]

    def world_runs(loader_fn, prefix, modality):
        for wc in worlds:
            model.cfg.world_given_p = 1.0 if wc == "given" else 0.0
            res["clips"][f"{prefix} ({wc} cameras)"] = eval_clips(model, loader_fn(), modality, dev, amp,
                                                                  args.omniworld_limit, L_list, lp)

    n_obs = cfg.queue + 1
    if args.dry_run:
        n, bs = 2, 2
        res["images"] = ev.eval_images(model, ev.synthetic_loader("image", cfg, 1, bs, n, args.seed), dev, amp, 0, lp, fid)
        for m in ("video", "threed"):
            res["clips"][m] = eval_clips(model, ev.synthetic_loader(m, cfg, n_obs, bs, n, args.seed), m, dev, amp, 0,
                                         L_list, lp)
        world_runs(lambda: ev.synthetic_loader("world", cfg, n_obs, bs, n, args.seed), "world", "world")
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
            if args.omniworld_modality == "world":
                world_runs(lambda: ev.omniworld_loader(args, cfg.image_size), name, "world")
            else:
                res["clips"][name] = eval_clips(model, ev.omniworld_loader(args, cfg.image_size), "threed", dev, amp,
                                                args.omniworld_limit, L_list, lp)
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
