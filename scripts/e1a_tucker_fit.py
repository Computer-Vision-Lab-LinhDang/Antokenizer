#!/usr/bin/env python3
"""E1a of the C-D 4.1 design (section 12): relaxed capacity of Δ-Tucker on real teacher features after an
oracle warp, before any tokenizer is trained.

  python scripts/e1a_tucker_fit.py --omniworld <pack> --dino-repo <local copy> --stride 8 --limit 64 --out <dir>
  python scripts/e1a_tucker_fit.py --dry-run --out /tmp/e1a                     # CPU, stub DINO, synthetic pack

Segments of 4 frames of the OmniWorld-Game val split (anchor + 3 targets, raw frame stride --stride). Teacher
features F_i: LN of the centred last DINOv3 layer (the target of the tokenizer). Oracle warp W*_i: the transport
of src/mavt/cd41/transport.py with the measured anchor depth (cell log inverse depth), the given relative pose
and intrinsics and M = 0. It is an oracle only for the static part, so the evaluation mask Omega_i keeps the
target cells with support, outside moving objects, and the anchor cells that are read must have valid depth and
be static. On Omega:

  E_i = F_i - W*_i F_0           residual after transport      (none: F_i - F_0, the copy residual)
  min_{U, B_i, Q} sum_i || Omega_i (E_i - W*_i U B_i Q) ||^2   U [N, r], B_i [r, r], Q [r, d] free per segment
                                                                (none: the same fit with W = I)

E1a drops the constraints of the model (U generated from the anchor, shared core G, a_i of 32 numbers), so it
is a loosened ceiling, and Adam finds a solution, not necessarily the global one. Reported per rank r: the
share of the residual energy captured and the remaining energy as a share of ||F||^2, with and without the
warp, plus the singular spectrum (channel unfolding) of the warped residual. Writes <out>/e1a.json and .md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omniworld", default=None, help="OmniWorld-Game pack root")
    p.add_argument("--split", default="val")
    p.add_argument("--stride", type=int, default=8, help="raw frame stride between the 4 observations")
    p.add_argument("--limit", type=int, default=64, help="segments")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ranks", default="4,8,16,32", help="r = R_xy = R_c of the fit")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--r-cut", type=float, default=1.5)
    p.add_argument("--sigma-u", type=float, default=0.35)
    p.add_argument("--tau-z", type=float, default=0.1)
    p.add_argument("--dino-repo", default=None, help="DINOv3 (HF id or local copy); default: the CD41Config one")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true", help="CPU, stub DINO, a synthetic pack under --out")
    return p


# ---------------------------------------------------------------------------------------------- pieces
def teacher(dino, frames: torch.Tensor) -> torch.Tensor:
    """frames [B, n, 3, H, W] -> F [B, n, N, d]: LN(last - mean over cells), no affine (as FrameEncoder)."""
    B, n = frames.shape[:2]
    last, _, _ = dino(frames.flatten(0, 1))
    c = last.float() - last.float().mean(1, keepdim=True)
    f = F.layer_norm(c, c.shape[-1:])
    return f.view(B, n, *f.shape[1:])


def oracle_warp(b: dict, grid: int, size: int, r_cut: float, sigma_u: float, tau_z: float, dev) -> tuple:
    """Loader batch (world clip with depth, dyn) -> (W [B, T, N, N], omega [B, T, N] bool)."""
    from mavt.cd32 import geometry as geo
    from mavt.cd32.depth import cell_log_inv_depth
    from mavt.cd41 import transport as tr
    depth, dyn = b["depth"].to(dev).float(), b["dyn"].to(dev).float()
    c2w, K = b["c2w"].to(dev).float(), b["K"].to(dev).float()
    B, n = depth.shape[:2]
    T = n - 1
    rho, valid = cell_log_inv_depth(depth[:, 0], depth[:, 0] > 0, grid)
    share = F.adaptive_avg_pool2d(dyn.flatten(0, 1).unsqueeze(1), grid).flatten(1).view(B, n, -1)   # [B, n, N]
    static = share < 0.5
    rho = torch.where(valid & static[:, 0], rho, torch.full_like(rho, float("nan")))   # unreadable sources
    kappa = tr.kappa_from_K(K, size, size)
    R, t = geo.pose_matrix(tr.se3_log(tr.relative_pose(c2w))[:, 1:].flatten(0, 1))
    rep = lambda v: v.unsqueeze(1).expand(B, T, *v.shape[1:]).flatten(0, 1)
    out = tr.transport(rep(rho), rep(kappa[:, 0]), R, t, None, kappa[:, 1:].flatten(0, 1), grid, 9, r_cut,
                       sigma_u, tau_z)
    N = rho.shape[1]
    W, sup = out["W"].view(B, T, N, N), out["support"].view(B, T, N)
    return W, sup & static[:, 1:]


def fit_relaxed(E: torch.Tensor, W, omega: torch.Tensor, rank: int, steps: int, lr: float, seed: int = 0) -> dict:
    """min over U [B, N, r], B_i [B, T, r, r], Q [B, r, d] of sum omega ||E - W U B_i Q||^2 (W None: identity).
    E [B, T, N, d], omega [B, T, N] -> dict(captured = 1 - residual / ||omega E||^2, residual energy)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, T, N, d = E.shape
    dev = E.device
    init = lambda *s: (torch.randn(*s, generator=g) * 0.1).to(dev).requires_grad_(True)
    U, Bm, Q = init(B, N, rank), init(B, T, rank, rank), init(B, rank, d)
    w = omega.float().unsqueeze(-1)
    energy = (w * E.pow(2)).sum()
    opt = torch.optim.Adam([U, Bm, Q], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def resid():
        V = torch.einsum("bnp,btpq,bqd->btnd", U, Bm, Q)                          # canonical variation
        P = V if W is None else torch.einsum("btij,btjd->btid", W, V)
        return (w * (E - P).pow(2)).sum()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = resid() / energy.clamp_min(1e-12)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        r = resid()
    return dict(captured=float(1 - r / energy.clamp_min(1e-12)), residual=float(r), energy=float(energy))


def spectrum(E: torch.Tensor, omega: torch.Tensor, k: int = 64) -> dict:
    """Singular values of the channel unfolding [cells in omega, d] of E: energy share of the first k and the
    effective rank at 90 % of the energy."""
    X = E[omega].float()
    if X.shape[0] < 2:
        return dict(rank90=None, top=[])
    s = torch.linalg.svdvals(X - X.mean(0, keepdim=True))
    e = s.pow(2) / s.pow(2).sum().clamp_min(1e-12)
    cum = e.cumsum(0)
    return dict(rank90=int((cum < 0.9).sum().item()) + 1, top=[round(float(v), 4) for v in cum[:k:8]])


# ---------------------------------------------------------------------------------------------- data
def make_loader(args, size: int):
    from mavt.dino_v0.data import OmniWorldViews, collate
    ds = OmniWorldViews(args.omniworld, args.split, n_views=4, frame_stride=str(args.stride), size=size, train=False,
                        depth=True, eval_stride=args.stride, modality="world")
    return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=collate)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from mavt.cd32.encoder import MultiLayerDINO
    from mavt.cd41 import CD41Config, tiny_cd41_config
    if args.dry_run:
        sys.path.insert(0, os.path.join(HERE, "..", "tests", "cd32"))
        from test_omniworld_views import write_pack
        args.omniworld = write_pack(os.path.join(args.out, "pack"), args.split, n_scenes=1)
        args.device, args.num_workers, args.stride, args.steps = "cpu", 0, 2, min(args.steps, 60)
        cfg = tiny_cd41_config()
    else:
        if not args.omniworld:
            raise SystemExit("give --omniworld (or --dry-run)")
        cfg = CD41Config(**({"dino_repo": args.dino_repo} if args.dino_repo else {}))
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    dino = MultiLayerDINO(cfg).to(dev)
    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]
    acc = {"warp": {r: [0.0, 0.0] for r in ranks}, "none": {r: [0.0, 0.0] for r in ranks}}
    tot = dict(feat=0.0, warp=0.0, none=0.0, cells=0.0, omega=0.0)
    Ew_all, Om_all = [], []
    seen, t0 = 0, time.time()
    for b in make_loader(args, cfg.image_size):
        with torch.no_grad():
            Fs = teacher(dino, b["data"].to(dev).permute(0, 2, 1, 3, 4))
            W, om = oracle_warp(b, cfg.grid, cfg.image_size, args.r_cut, args.sigma_u, args.tau_z, dev)
        Ft = Fs[:, 1:]
        Ew = Ft - torch.einsum("btij,bjd->btid", W, Fs[:, 0])
        En = Ft - Fs[:, :1]
        w = om.float().unsqueeze(-1)
        tot["feat"] += float((w * Ft.pow(2)).sum())
        tot["warp"] += float((w * Ew.pow(2)).sum())
        tot["none"] += float((w * En.pow(2)).sum())
        tot["omega"] += float(om.sum())
        tot["cells"] += om.numel()
        for r in ranks:
            fw = fit_relaxed(Ew, W, om, r, args.steps, args.lr, args.seed)
            fn = fit_relaxed(En, None, om, r, args.steps, args.lr, args.seed)
            acc["warp"][r][0] += fw["residual"]
            acc["none"][r][0] += fn["residual"]
        Ew_all.append(Ew.detach().cpu())
        Om_all.append(om.cpu())
        seen += Fs.shape[0]
        print(f"{seen} segments, {time.time() - t0:.0f}s", flush=True)
        if args.limit and seen >= args.limit:
            break
    Ew_all, Om_all = torch.cat(Ew_all), torch.cat(Om_all)
    rows = []
    for r in ranks:
        rw, rn = acc["warp"][r][0], acc["none"][r][0]
        rows.append(dict(rank=r, captured_warp=1 - rw / max(tot["warp"], 1e-12),
                         captured_none=1 - rn / max(tot["none"], 1e-12),
                         left_warp=rw / max(tot["feat"], 1e-12), left_none=rn / max(tot["feat"], 1e-12),
                         scalars_per_target=r * r))
    res = dict(segments=seen, stride=args.stride, omega_frac=tot["omega"] / max(tot["cells"], 1.0),
               residual_share_warp=tot["warp"] / max(tot["feat"], 1e-12),
               residual_share_none=tot["none"] / max(tot["feat"], 1e-12),
               warp_gain=1 - tot["warp"] / max(tot["none"], 1e-12), fits=rows,
               spectrum_warp=spectrum(Ew_all, Om_all), steps=args.steps, transport=dict(r_cut=args.r_cut,
               sigma_u=args.sigma_u, tau_z=args.tau_z), seconds=time.time() - t0)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e1a.json"), "w") as f:
        json.dump(res, f, indent=1)
    md = [f"# E1a: relaxed Δ-Tucker fit after an oracle warp ({seen} segments, stride {args.stride})", "",
          f"Omega keeps {res['omega_frac']:.2f} of the target cells. Residual energy as a share of ||F||^2 on "
          f"Omega: copy {res['residual_share_none']:.3f}, after the warp {res['residual_share_warp']:.3f} "
          f"(the warp removes {res['warp_gain']:.2f} of the copy residual).", "",
          "| rank r | captured, warp | captured, no warp | left of ||F||^2, warp | left, no warp | B_i scalars |",
          "|---|---|---|---|---|---|"]
    md += [f"| {x['rank']} | {x['captured_warp']:.3f} | {x['captured_none']:.3f} | {x['left_warp']:.3f} "
           f"| {x['left_none']:.3f} | {x['scalars_per_target']} |" for x in rows]
    sp = res["spectrum_warp"]
    md += ["", f"Warped residual, channel unfolding: effective rank at 90 % energy = {sp['rank90']}; cumulative "
               f"energy at 1, 9, 17, ... components: {sp['top']}", ""]
    with open(os.path.join(args.out, "e1a.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
