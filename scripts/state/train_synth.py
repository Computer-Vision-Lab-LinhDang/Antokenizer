"""Train the time-indexed state model on synthetic scenes and measure whether the state remembers.

The central check of the state direction: with reconstruction of what was just seen as the only signal, the
state can be overwritten every step. ``--memory-signal on`` drops whole steps and hides single views during
training, so their readout must come from the state; ``off`` keeps every view of every step visible.

Eval on held-out seeds (never used in training), PSNR (dB) and depth abs-rel on the cells of interest:
  all        every view of every step visible; all cells (reconstruction guard)
  drop_last  last step unobserved; its cells (memory + predict)
  hide_view  from step 1, view (t mod V) hidden; the hidden cells (fusion with other views + memory)
``copy_prev`` is the reference of copying the same view's previous frame into the cells of interest.

    PYTHONPATH=src python scripts/state/train_synth.py --memory-signal on --out runs/state_synth/mem_on
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

VAL_SEED0 = 10_000_000  # train seeds are 0 .. steps * batch - 1, far below this


class Scenes(Dataset):
    def __init__(self, seeds: range, cfg: SyntheticSceneConfig) -> None:
        self.seeds, self.cfg = seeds, cfg

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s, g = make_synthetic_scene(self.cfg, self.seeds[i])
        return {"images": s.images, "K": s.intrinsics, "c2w": s.cam_to_world, "times": s.frame_times[0].float(),
                "depth": g.depth, "valid": g.valid}


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
    drop = full.clone()
    drop[:, :, T - 1] = False
    hide = full.clone()
    for t in range(1, T):
        hide[:, t % V, t] = False
    return {"all": (full, full), "drop_last": (drop, ~drop), "hide_view": (hide, ~hide)}


def cell_metrics(rgb: torch.Tensor, log_depth: torch.Tensor, b: dict, cells: torch.Tensor) -> dict[str, float]:
    """Mean PSNR over cells (images in [-1, 1]) and depth abs-rel over valid pixels of those cells."""
    mse = (((rgb - b["images"]) / 2) ** 2).mean((-3, -2, -1))           # (B, V, T)
    psnr = (-10 * torch.log10(mse.clamp(min=1e-10)))[cells].mean().item()
    m = b["valid"] & cells[..., None, None]
    rel = (log_depth[:, :, :, 0].exp() - b["depth"]).abs() / b["depth"].clamp(min=1e-3)
    return {"psnr": psnr, "absrel": rel[m].mean().item() if m.any() else float("nan")}


@torch.no_grad()
def evaluate(model: StateModel, batches: list[dict], device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    acc: dict[str, list[dict[str, float]]] = {}
    for b in batches:
        b = {k: v.to(device) for k, v in b.items()}
        B, V, T = b["images"].shape[:3]
        prev = torch.cat([b["images"][:, :, :1], b["images"][:, :, :-1]], 2)  # same view, previous frame
        for name, (vis, cells) in eval_masks(B, V, T).items():
            vis, cells = vis.to(device), cells.to(device)
            out = model(b["images"], b["K"], b["c2w"], b["times"], vis)
            acc.setdefault(name, []).append(cell_metrics(out.rgb, out.log_depth, b, cells))
            if name != "all":
                acc.setdefault(f"{name}/copy_prev", []).append(
                    {"psnr": cell_metrics(prev, out.log_depth, b, cells)["psnr"]})
    model.train()
    return {k: {m: sum(r[m] for r in v) / len(v) for m in v[0]} for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-signal", choices=["on", "off"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=4)
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

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    scene = SyntheticSceneConfig(T=args.T)
    cfg = StateConfig(dim=args.dim, depth=args.depth, n_state=args.K)
    model = StateModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    train = DataLoader(Scenes(range(args.steps * args.batch), scene), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    val = list(DataLoader(Scenes(range(VAL_SEED0, VAL_SEED0 + args.n_val), scene), batch_size=args.batch))
    p_drop, p_hide = (args.p_drop, args.p_hide) if args.memory_signal == "on" else (0.0, 0.0)
    g = torch.Generator().manual_seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "device": str(device), "params": n_params,
            "patches_per_step": scene.V * (scene.H // scene.patch) * (scene.W // scene.patch),
            "z_floats_per_step": args.K * cfg.z_dim, "input_floats_per_step": scene.V * 3 * scene.H * scene.W}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    log = (args.out / "log.jsonl").open("a")
    print(json.dumps(meta), flush=True)

    t0, step = time.time(), 0
    for b in train:
        b = {k: v.to(device) for k, v in b.items()}
        B, V, T = b["images"].shape[:3]
        vis = sample_visible(B, V, T, p_drop, p_hide, g).to(device)
        losses = state_loss(model(b["images"], b["K"], b["c2w"], b["times"], vis), b["images"], b["depth"],
                            b["valid"])
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0 or step == args.steps:
            rec = {"step": step, "sec": round(time.time() - t0, 1), **{k: round(v.item(), 5) for k, v in losses.items()}}
            if step % args.eval_every == 0 or step == args.steps:
                rec["eval"] = evaluate(model, val, device)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if step >= args.steps:
            break
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "meta": meta}, args.out / "last.pt")


if __name__ == "__main__":
    main()
