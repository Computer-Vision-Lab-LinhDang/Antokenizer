"""3.2 mechanism checks of a trained Grain-Prior checkpoint (no training): what C and D carry and how the read uses
them. Every decode reads the serialized state alone (serialize -> deserialize, as scripts/eval_grain_prior.py).

  CD1  D off at decode: each target decoded from C only (z = c_0, identity camera, M = 0, no innovation)
  CD3  curve over the record: C -> + xi, kappa -> + M and memory z_i -> + r(L) for L in L_list (PSNR, feature cos)
  CD5  c_0 of another clip in the batch at the anchor decode: band fractions (low / mid / high) of the change
  CD6  anchor decoded from the batch-mean state and from a random state of the batch statistics
  AT4  the k grains of lowest / highest attention mass removed at the anchor decode: PSNR drop
  AT2  (read_coarse_kv) Spearman between the per-cell fine / coarse / sink share and the local high-frequency energy
  AT3  effective support 1 / sum a^2 of the anchor read

  python -m mavt.grain_prior.evaluate_mechanism --ckpt <run>/last.pt --openvid <manifest> --out <run>/mech
  python -m mavt.grain_prior.evaluate_mechanism --dry-run --out /tmp/gp_mech           # CPU, tiny model
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from ..cd32.losses import psnr
from . import schema as sc
from .losses import band_fractions
from .readout import pyr_down, pyr_up


def _rgb(model, o: dict) -> torch.Tensor:
    return model.rgb_of(o["F_hat"], o["H_hat"], o["a0"]) * 2 - 1


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float(), b.float(), -1).mean().item()


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation per row [B, Q], mean over rows."""
    r = lambda x: x.float().argsort(-1).argsort(-1).float()
    ra, rb = r(a), r(b)
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return ((ra * rb).sum(-1) / ((ra.pow(2).sum(-1) * rb.pow(2).sum(-1)).sqrt().clamp_min(1e-12))).mean().item()


def _targets(model, S, der, zs, stage: str, L: Optional[int] = None) -> list:
    """Decoded targets of the queue for one CD3 stage (previous-frame read chained as in decode)."""
    cfg = model.cfg
    B, dev = S["c0"].shape[0], S["c0"].device
    c0 = S["c0"].float()
    zero_xi, zero_M = torch.zeros(B, 6, device=dev), torch.zeros(B, cfg.n_motion, 3, device=dev)
    prev = model._prev_of(model.decode_anchor(S, der)) if cfg.read_prev_frame else None
    outs = []
    for rec, z in zip(S["recs"], zs):
        if stage == "C":
            args = (c0, zero_xi, S["kappa0"], zero_M, None, None)
        elif stage == "C+xi":
            args = (c0, rec["xi"], rec["kappa"], zero_M, None, None)
        elif stage == "C+xi+M":
            args = (z, rec["xi"], rec["kappa"], rec["M"], None, None)
        else:
            r = rec["r"].clone()
            r[:, L:] = 0
            args = (z, rec["xi"], rec["kappa"], rec["M"], r, torch.minimum(rec["L"], torch.full_like(rec["L"], L)))
        o = model.decode_obs(S, der, *args, prev=prev)
        prev = model._prev_of(o) if cfg.read_prev_frame else None
        outs.append(o)
    return outs


@torch.no_grad()
def mechanism_batch(model, batch: dict, modality: str, L_list: list, k_list: list) -> dict:
    """One batch -> flat dict of scalar floats (keys 'group/name')."""
    cfg = model.cfg
    x = batch["data"]
    dev = x.device
    cam = {k: batch[k].to(dev) for k in ("c2w", "K", "depth") if k in batch} if modality in ("threed", "world") else None
    o = model(x, modality, cam, return_outputs=True)["outputs"]
    S = sc.deserialize(cfg, [f.to(dev) for f in sc.serialize(cfg, o["S"])])
    frames = x.unsqueeze(1) if modality == "image" else x.permute(0, 2, 1, 3, 4)
    B, n = frames.shape[:2]
    q = len(S["recs"])
    gt = torch.cat([frames[:, :1], frames[:, n - q:]], 1)                                # [B, 1 + q, 3, H, W]
    Ft = model.stems(gt.flatten(0, 1))["F"].view(B, 1 + q, cfg.n_cells, -1)
    der, zs = model.derived(S), model.replay(S)
    res = {}
    # ---- full decode of the state
    a0 = model.decode_anchor(S, der)
    rgb_a = _rgb(model, a0)
    res["full/psnr_anchor"] = psnr(rgb_a, gt[:, 0]).item()
    if q:
        full = model.decode(S)
        res["full/psnr_target"] = psnr(full["rgb"][:, 1:].flatten(0, 1), gt[:, 1:].flatten(0, 1)).item()
        res["full/feat_cos_target"] = _cos(full["F"][:, 1:], Ft[:, 1:])
        # ---- CD3 curve (its first stage is CD1)
        stages = ["C", "C+xi", "C+xi+M"] + [f"C+xi+M+r(L={L})" for L in L_list]
        for st in stages:
            L = int(st.split("L=")[1][:-1]) if "L=" in st else None
            outs = _targets(model, S, der, zs, st, L)
            rgb = torch.stack([_rgb(model, t) for t in outs], 1)
            res[f"cd3/{st}/psnr_target"] = psnr(rgb.flatten(0, 1), gt[:, 1:].flatten(0, 1)).item()
            res[f"cd3/{st}/feat_cos"] = _cos(torch.stack([t["F_hat"] for t in outs], 1), Ft[:, 1:])
        res["cd1/psnr_target_D_off"] = res["cd3/C/psnr_target"]
        res["cd1/drop_vs_full"] = res["full/psnr_target"] - res["cd3/C/psnr_target"]
    # ---- CD5: c_0 of the next clip in the batch
    if B > 1:
        Sw = dict(S, c0=S["c0"].roll(1, 0))
        rgb_w = _rgb(model, model.decode_anchor(Sw, model.derived(Sw)))
        lo, mid, hi = band_fractions(rgb_w - rgb_a)
        res.update({"cd5/low": lo.item(), "cd5/mid": mid.item(), "cd5/high": hi.item(),
                    "cd5/psnr_anchor_swapped": psnr(rgb_w, gt[:, 0]).item()})
        # ---- CD6: batch-mean and random states
        keys = ("u", "rho", "sigma", "f", "c0", "kappa0")
        mean = {k: S[k].float().mean(0, keepdim=True).expand_as(S[k]) for k in keys}
        rand = {k: mean[k] + S[k].float().std(0, keepdim=True) * torch.randn_like(S[k].float()) for k in keys}
        rand["u"], rand["sigma"] = rand["u"].clamp(0, 1), rand["sigma"].abs().clamp_min(1e-3)
        rand["kappa0"] = S["kappa0"]
        for name, sub in (("mean_state", mean), ("random_state", rand)):
            Sx = dict(S, **sub)
            res[f"cd6/psnr_{name}"] = psnr(_rgb(model, model.decode_anchor(Sx, model.derived(Sx))), gt[:, 0]).item()
        res["cd6/psnr_anchor"] = res["full/psnr_anchor"]
    # ---- AT4: grains of lowest / highest read mass removed at the anchor
    mass, G = a0["diag"]["grain_mass"], cfg.n_grains
    zero_xi, zero_M = torch.zeros(B, 6, device=dev), torch.zeros(B, cfg.n_motion, 3, device=dev)
    for k in [k for k in k_list if k < G]:
        for name, largest in (("drop_low", False), ("drop_high", True)):
            keep = torch.ones(B, G, dtype=torch.bool, device=dev)
            keep.scatter_(-1, mass.topk(k, -1, largest=largest).indices, False)
            ok = model.decode_obs(S, der, S["c0"].float(), zero_xi, S["kappa0"], zero_M, grain_keep=keep)
            res[f"at4/{name}/{k}"] = res["full/psnr_anchor"] - psnr(_rgb(model, ok), gt[:, 0]).item()
    # ---- AT2 / AT3 on the anchor read
    d = a0["diag"]
    if cfg.read_coarse_kv:
        X = (gt[:, 0].float() + 1) / 2
        hf = (X - pyr_up(pyr_down(X, 1), 1)).pow(2).mean(1, keepdim=True)
        cell = F.adaptive_avg_pool2d(hf, cfg.grid).flatten(1)                          # [B, Q]
        for s in ("fine", "coarse", "sink"):
            res[f"at2/{s}_hf_spearman"] = _spearman(d[f"{s}_share_q"], cell)
    es = d["eff_support_q"].float().flatten()
    res["at3/eff_support_mean"] = es.mean().item()
    res["at3/eff_support_p10"], res["at3/eff_support_p90"] = [v.item() for v in es.quantile(torch.tensor([0.1, 0.9]))]
    return res


def _nest(flat: dict) -> dict:
    out = {}
    for key, v in flat.items():
        head, *rest = key.split("/")
        if head == "cd3":
            curve = out.setdefault("cd3_curve", [])
            if not curve or curve[-1]["stage"] != rest[0]:
                curve.append(dict(stage=rest[0]))
            curve[-1][rest[1]] = v
        elif len(rest) == 2:
            out.setdefault(head, {}).setdefault(rest[0], {})[rest[1]] = v
        else:
            out.setdefault(head, {})[rest[0]] = v
    return out


def evaluate_mechanism(model, batches: Iterable[dict], modality: str, L_list: list, k_list: list,
                       limit: int = 0, autocast=None) -> dict:
    """Mean over the clips of every check (batches: dict(data [B, 3, T, H, W], c2w, K, depth) as the eval loaders)."""
    model.eval()
    dev = next(model.parameters()).device
    tot, seen = {}, 0
    for b in batches:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with autocast() if autocast is not None else _null():
            r = mechanism_batch(model, b, modality, L_list, k_list)
        B = b["data"].shape[0]
        for k, v in r.items():
            tot[k] = tot.get(k, 0.0) + v * B
        seen += B
        if limit and seen >= limit:
            break
    res = _nest({k: v / max(seen, 1) for k, v in tot.items()})
    res["n_clips"] = seen
    return res


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def to_markdown(res: dict) -> str:
    f = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["# Mechanism checks", "", f"checkpoint: {res.get('ckpt')}  step: {res.get('step')}", ""]
    for name, r in res["clips"].items():
        lines += [f"## {name} ({r['n_clips']} clips)", ""]
        if "full" in r:
            lines.append("full: " + ", ".join(f"{k} {f(v)}" for k, v in r["full"].items()))
        if "cd3_curve" in r:
            lines += ["", "| CD3 stage | PSNR target | feature cos |", "|---|---:|---:|"]
            lines += [f"| {c['stage']} | {f(c['psnr_target'])} | {f(c['feat_cos'])} |" for c in r["cd3_curve"]]
            lines.append("")
        for key in ("cd1", "cd5", "cd6", "at2", "at3"):
            if key in r:
                lines.append(f"{key.upper()}: " + ", ".join(f"{k} {f(v)}" for k, v in r[key].items()))
        if "at4" in r:
            for name4, dd in r["at4"].items():
                lines.append(f"AT4 {name4} (dB): " + ", ".join(f"k={k} {f(v)}" for k, v in dd.items()))
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, os.path.join(root, "scripts"))
    import eval_cdv as ev                                                            # loaders and flags of the evals
    from . import MAVTGrainPrior, load_grain_prior, tiny_config
    p = ev.build_parser()
    p.description = __doc__
    p.add_argument("--k-list", default="1,4,16", help="AT4: grains removed")
    p.set_defaults(L_list="0,8,16,32,max", stem="mechanism")
    args = p.parse_args(argv)
    if not args.ckpt and not args.dry_run:
        raise SystemExit("give --ckpt (or --dry-run)")
    dev = torch.device("cpu" if args.dry_run else args.device)
    amp = args.bf16 and not args.dry_run
    torch.manual_seed(args.seed)
    over = {"dino_repo": args.dino_repo} if args.dino_repo else {}
    if args.ckpt:
        model, cfg, step = load_grain_prior(args.ckpt, dev, **over)
    else:
        cfg = tiny_config(read_coarse_kv=True, readout_version="legacy", read_strict=True, read_sink=True)
        model, step = MAVTGrainPrior(cfg).to(dev).eval(), -1
    L_list = ev.parse_L(args.L_list, cfg.n_slots)
    k_list = [int(k) for k in args.k_list.split(",") if k.strip()]
    ac = (lambda: ev._autocast(dev, amp)) if amp else None
    run = lambda ld, mod, lim: evaluate_mechanism(model, ld, mod, L_list, k_list, lim, ac)
    res = dict(ckpt=args.ckpt, step=step, L_list=L_list, k_list=k_list, clips={})
    if args.dry_run:
        for mod in ("video", "threed", "world"):
            res["clips"][mod] = run(ev.synthetic_loader(mod, cfg, cfg.queue + 1, 2, 2, args.seed), mod, 0)
    else:
        for man in [m for m in (args.openvid or "").split(",") if m.strip()]:
            name = "video " + os.path.basename(man.rstrip("/")).split(".")[0]
            res["clips"][name] = run(ev.video_loader(man, args, cfg.image_size), "video", args.video_limit)
        if args.objaverse:
            res["clips"]["objaverse static_views"] = run(ev.objaverse_loader(args, cfg.image_size), "threed",
                                                         args.objaverse_limit)
        if args.omniworld:
            res["clips"]["omniworld " + args.omniworld_modality] = run(
                ev.omniworld_loader(args, cfg.image_size), args.omniworld_modality, args.omniworld_limit)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, args.stem + ".json"), "w") as fh:
        json.dump(res, fh, indent=1)
    md = to_markdown(res)
    with open(os.path.join(args.out, args.stem + ".md"), "w") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
