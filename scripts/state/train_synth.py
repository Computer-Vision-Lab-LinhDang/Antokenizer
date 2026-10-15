"""Train the time-indexed state model on synthetic scenes and measure whether the state remembers.

The central check of the state direction: with reconstruction of what was just seen as the only signal, the
state can be overwritten every step. Both arms see the same random dropped steps and hidden views; they differ
only in what is supervised:
  --memory-signal on   every (view, step) cell, seen or not: unseen cells must be read out of the state
  --memory-signal off  only the cells the update saw

Scenes use a per-scene random texture offset (--texture-shift): with the default fixed background the readout
learns the background from the camera rays alone and ignores the observations.

Eval on held-out seeds (never trained on). Ground and sky still dominate the pixels, so the headline is on box
pixels (GT instance > 0) of the cells of interest, next to three references on the same pixels:
  seen       the same model with every view of every step visible (upper bound)
  copy_prev  the same view's previous frame
  prior      per-pixel median image over train scenes (no state, no images: the "no memory" floor)
Conditions (cells of interest in brackets):
  all [every cell]  drop_last [last step]  drop_last2 [last two steps]
  hide_view [view t mod V, t >= 1]  hide_edge [view 0 for every t >= 1]

    PYTHONPATH=src python scripts/state/train_synth.py --memory-signal on --out results/state_synth/v2_on
    PYTHONPATH=src python scripts/state/train_synth.py --eval-ckpt results/state_synth/v2_on/last.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from mavt4d.data.synthetic import SyntheticSceneConfig, make_synthetic_scene
from mavt4d.model.state import StateConfig, StateModel, state_loss

VAL_SEED0 = 10_000_000  # train seeds are 0 .. steps * batch - 1, checked to stay below this
N_PRIOR = 64            # train scenes in the per-pixel median prior


class Scenes(Dataset):
    def __init__(self, seeds: range, cfg: SyntheticSceneConfig) -> None:
        self.seeds, self.cfg = seeds, cfg

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s, g = make_synthetic_scene(self.cfg, self.seeds[i])
        return {"images": s.images, "K": s.intrinsics, "c2w": s.cam_to_world, "times": s.frame_times[0].float(),
                "depth": g.depth, "valid": g.valid, "instance": g.instance}


def sample_visible(B: int, V: int, T: int, p_drop: float, p_hide: float, g: torch.Generator) -> torch.Tensor:
    """(B, V, T) bool. Step 0 is always fully observed; an observed step keeps at least one view."""
    hide = torch.rand(B, V, T, generator=g) < p_hide
    hide[:, :, 0] = False
    vis = ~hide
    vis[:, 0] |= ~vis.any(1)
    drop = torch.rand(B, T, generator=g) < p_drop
    drop[:, 0] = False
    return vis & ~drop[:, None]


def eval_masks(B: int, V: int, T: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """condition -> (visible (B, V, T), cells of interest (B, V, T))."""
    full = torch.ones(B, V, T, dtype=torch.bool)
    out = {"all": (full, full)}
    for name, steps in (("drop_last", [T - 1]), ("drop_last2", [T - 2, T - 1])):
        vis = full.clone()
        vis[:, :, steps] = False
        out[name] = (vis, ~vis)
    rot, edge = full.clone(), full.clone()
    for t in range(1, T):
        rot[:, t % V, t] = False
        edge[:, 0, t] = False
    out["hide_view"], out["hide_edge"] = (rot, ~rot), (edge, ~edge)
    return out


class Pool:
    """Pooled sums so that every pixel counts once, whatever the batch sizes."""

    def __init__(self) -> None:
        self.s: dict[str, list[float]] = {}

    def add(self, key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        acc = self.s.setdefault(key, [0.0, 0])
        acc[0] += values[mask].sum().item()
        acc[1] += int(mask.sum().item())

    def mean(self, key: str) -> float:
        total, n = self.s.get(key, [0.0, 0])
        return total / n if n else float("nan")


def _psnr(mse: float) -> float:
    return -10 * math.log10(max(mse, 1e-10))


@torch.no_grad()
def evaluate(model: StateModel, batches: list[dict], prior: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    pool = Pool()
    for b in batches:
        b = {k: v.to(device) for k, v in b.items()}
        B, V, T = b["images"].shape[:3]
        img, box = b["images"], b["instance"] > 0                         # box: (B, V, T, H, W)
        prev = torch.cat([img[:, :, :1], img[:, :, :-1]], 2)               # same view, previous frame
        se = lambda pred: (((pred - img) / 2) ** 2).mean(3)               # (B, V, T, H, W), images in [0, 1]
        rel = lambda ld: (ld[:, :, :, 0].exp() - b["depth"]).abs() / b["depth"].clamp(min=1e-3)
        seen = model(img, b["K"], b["c2w"], b["times"], torch.ones(B, V, T, dtype=torch.bool, device=device))
        for name, (vis, cells) in eval_masks(B, V, T).items():
            vis, cells = vis.to(device), cells.to(device)
            out = seen if name == "all" else model(img, b["K"], b["c2w"], b["times"], vis)
            on_cells = cells[..., None, None].expand_as(box)
            pool.add(f"{name}/img", se(out.rgb), on_cells)
            for src, pred in (("model", out.rgb), ("seen", seen.rgb), ("copy_prev", prev), ("prior", prior)):
                pool.add(f"{name}/{src}/box", se(pred.expand_as(img)), on_cells & box)
            for src, o in (("model", out), ("seen", seen)):
                pool.add(f"{name}/{src}/absrel", rel(o.log_depth), on_cells & box & b["valid"])
    model.train()
    res = {}
    for name in eval_masks(1, 1, 2):
        r = {"psnr_img": _psnr(pool.mean(f"{name}/img"))}
        for src in ("model", "seen", "copy_prev", "prior"):
            r[f"psnr_box_{src}"] = _psnr(pool.mean(f"{name}/{src}/box"))
        for src in ("model", "seen"):
            r[f"absrel_box_{src}"] = pool.mean(f"{name}/{src}/absrel")
        res[name] = {k: round(v, 4) for k, v in r.items()}
    return res


def median_prior(scene: SyntheticSceneConfig, device: torch.device) -> torch.Tensor:
    """(1, V, T, 3, H, W) per-pixel median image over N_PRIOR train scenes."""
    imgs = torch.stack([make_synthetic_scene(scene, s)[0].images for s in range(N_PRIOR)])
    return imgs.median(0).values[None].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-signal", choices=["on", "off"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--eval-ckpt", type=Path, help="only evaluate this checkpoint and print the result")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--texture-shift", type=float, default=50.0,
                    help="per-scene random texture offset (m); 0 = the fixed background of the default scenes")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p-drop", type=float, default=0.25)
    ap.add_argument("--p-hide", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    scene = SyntheticSceneConfig(T=args.T, texture_shift=args.texture_shift)
    val = list(DataLoader(Scenes(range(VAL_SEED0, VAL_SEED0 + args.n_val), scene), batch_size=args.batch))
    prior = median_prior(scene, device)

    if args.eval_ckpt:
        ckpt = torch.load(args.eval_ckpt, map_location="cpu")
        model = StateModel(StateConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        print(json.dumps(evaluate(model.to(device), val, prior, device), indent=1))
        return
    if args.memory_signal is None or args.out is None:
        ap.error("--memory-signal and --out are required for training")
    if args.steps * args.batch > VAL_SEED0:
        ap.error("train seeds would overlap the validation seeds")

    torch.manual_seed(args.seed)
    cfg = StateConfig(dim=args.dim, depth=args.depth, n_state=args.K)
    model = StateModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    train = DataLoader(Scenes(range(args.steps * args.batch), scene), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    g = torch.Generator().manual_seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "device": str(device),
            "params": sum(p.numel() for p in model.parameters()),
            "patches_per_step": scene.V * (scene.H // scene.patch) * (scene.W // scene.patch),
            "z_floats_per_step": args.K * cfg.z_dim, "input_floats_per_step": scene.V * 3 * scene.H * scene.W}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    log = (args.out / "log.jsonl").open("a")
    print(json.dumps(meta), flush=True)

    t0, step = time.time(), 0
    for b in train:
        b = {k: v.to(device) for k, v in b.items()}
        B, V, T = b["images"].shape[:3]
        vis = sample_visible(B, V, T, args.p_drop, args.p_hide, g).to(device)
        cells = None if args.memory_signal == "on" else vis
        losses = state_loss(model(b["images"], b["K"], b["c2w"], b["times"], vis), b["images"], b["depth"],
                            b["valid"], cells)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        do_eval = step % args.eval_every == 0 or step == args.steps
        if step % 50 == 0 or do_eval:
            rec = {"step": step, "sec": round(time.time() - t0, 1), **{k: round(v.item(), 5) for k, v in losses.items()}}
            if do_eval:
                rec["eval"] = evaluate(model, val, prior, device)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if step >= args.steps:
            break
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "meta": meta}, args.out / "last.pt")


if __name__ == "__main__":
    main()
