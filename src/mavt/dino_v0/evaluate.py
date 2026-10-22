"""Evaluation library for MAVT on DINOv3, v0.

Reconstruction
  images : PSNR, Gaussian SSIM, LPIPS-VGG (if `lpips`), rFID (if `torchmetrics` + inception weights).
  video  : per-frame PSNR / SSIM, PSNR split keyframe vs non-keyframe, swap diagnostics
           (frame t decoded with D_t zeroed / without D vs real D_t), mean |delta| in cells, mean gate.
Understanding (ImageNet)
  features: z_s, mean-pooled z_r of C, frozen DINOv3 CLS (reference).
  kNN top-1 (k=20, cosine, temperature-weighted vote as in DINO) and a linear probe
  (logistic regression on L2-normalised features, AdamW, lr chosen on a held-out split).

All functions take plain datasets / tensors so the tests can run them on CPU with tiny_config().
Images in the model's layout: [B, 3, H, W] in [-1, 1]; video [B, 3, T, H, W].
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DinoV0Config
from .decoder import patches_to_image

PSNR_CAP = 100.0      # dB; PSNR of identical images is capped here instead of inf


# ====================================================================== checkpoint
def load_model(ckpt_path: str, device="cpu", strict: bool = True):
    """Checkpoint dict {'model': state_dict, 'cfg': dict of DinoV0Config fields, 'step'} -> (model, cfg, step)."""
    from .model import MAVTDinoV0
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fields = DinoV0Config.__dataclass_fields__
    cfg_d = {k: v for k, v in dict(ck.get("cfg", {})).items() if k in fields}
    cfg = DinoV0Config(**cfg_d)
    model = MAVTDinoV0(cfg)
    sd = ck["model"] if "model" in ck else ck
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the frozen DINOv3 may be left out of the checkpoint (it is re-loaded from the hub)
    missing = [k for k in missing if not k.startswith("encoder.dino.")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return model.to(device).eval(), cfg, int(ck.get("step", -1))


def autocast(device, amp: bool):
    dev = torch.device(device).type
    if not amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type=dev, dtype=torch.bfloat16)


class Logger:
    def __init__(self, every: float = 30.0, fn: Callable[[str], None] = print):
        self.every, self.fn, self.t0, self.last = every, fn, time.time(), 0.0

    def __call__(self, msg: str, force: bool = False):
        now = time.time()
        if force or now - self.last >= self.every:
            self.last = now
            self.fn(f"[eval {now - self.t0:7.1f}s] {msg}")


# ====================================================================== pixel metrics
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Per-sample PSNR in dB, [B, ...] -> [B]; capped at PSNR_CAP (identical images)."""
    mse = (x.float() - y.float()).pow(2).flatten(1).mean(1)
    out = 10 * torch.log10(data_range ** 2 / mse.clamp_min(1e-20))
    return out.clamp(max=PSNR_CAP)


def _gauss_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    g = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0, win: int = 11,
         sigma: float = 1.5) -> torch.Tensor:
    """Gaussian SSIM (Wang et al. 2004), per channel, valid conv, averaged -> [B]. x, y [B, C, H, W]."""
    x, y = x.float(), y.float()
    C = x.shape[1]
    win = min(win, x.shape[-1], x.shape[-2])
    w = _gauss_window(win, sigma, x.device, x.dtype).expand(C, 1, win, win).contiguous()
    f = lambda t: F.conv2d(t, w, groups=C)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = f(x), f(y)
    sxx = f(x * x) - mx * mx
    syy = f(y * y) - my * my
    sxy = f(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


class OptionalLPIPS:
    """LPIPS-VGG if the `lpips` package is installed; `.available` / `.reason` otherwise."""

    def __init__(self, device):
        self.model, self.reason = None, ""
        try:
            import lpips  # noqa: F401
            self.model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        except Exception as ex:  # noqa: BLE001
            self.reason = f"LPIPS skipped: {type(ex).__name__}: {ex}"

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(x.float(), y.float()).flatten()


class OptionalFID:
    """rFID via torchmetrics FrechetInceptionDistance (2048-d pool features)."""

    def __init__(self, device):
        self.metric, self.reason = None, ""
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        except Exception as ex:  # noqa: BLE001
            self.reason = f"rFID skipped: {type(ex).__name__}: {ex}"

    @property
    def available(self) -> bool:
        return self.metric is not None

    @torch.no_grad()
    def update(self, real: torch.Tensor, fake: torch.Tensor):
        to01 = lambda t: ((t.float() + 1) / 2).clamp(0, 1)
        self.metric.update(to01(real), real=True)
        self.metric.update(to01(fake), real=False)

    def compute(self) -> float:
        return float(self.metric.compute())


# ====================================================================== loaders
def make_loader(ds, batch_size: int, num_workers: int = 0, shuffle: bool = False):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=torch.cuda.is_available(), drop_last=False,
                                       persistent_workers=False)


def as_loader(ds_or_loader, batch_size: int, num_workers: int = 0):
    """A DataLoader is used as given (e.g. with a grouped batch_sampler); a Dataset gets a plain loader."""
    if isinstance(ds_or_loader, torch.utils.data.DataLoader):
        return ds_or_loader
    return make_loader(ds_or_loader, batch_size, num_workers)


def grouped_batches(indices: Sequence[int], group_ids: Sequence[int], batch_size: int,
                    num_workers: int) -> List[List[int]]:
    """Batches for a row-group-backed dataset (parquet): each DataLoader worker gets whole groups.

    Groups are dealt round-robin to max(1, num_workers) streams; batches are emitted round-robin over
    the streams, matching DataLoader's dispatch order, so every row group is decoded by one worker only.
    """
    by_group: Dict[int, List[int]] = {}
    for i, g in zip(indices, group_ids):
        by_group.setdefault(int(g), []).append(int(i))
    gs = sorted(by_group)
    W = max(1, num_workers)
    streams = []
    for w in range(W):
        rows = [i for g in gs[w::W] for i in by_group[g]]
        streams.append([rows[b:b + batch_size] for b in range(0, len(rows), batch_size)])
    out, r = [], 0
    while any(r < len(s) for s in streams):
        out += [s[r] for s in streams if r < len(s)]
        r += 1
    return out


def balanced_by_groups(labels: torch.Tensor, group_ids: torch.Tensor, per_class: int, seed: int = 0) -> torch.Tensor:
    """Up to per_class indices per class, reading as few row groups as possible: visit groups in a random
    order and keep each row whose class is not yet full; stop when every class is full. Sorted indices."""
    labels, group_ids = torch.as_tensor(labels).long(), torch.as_tensor(group_ids).long()
    C = int(labels.max()) + 1
    need = torch.full((C,), per_class, dtype=torch.long)
    need[torch.bincount(labels[labels >= 0], minlength=C) == 0] = 0
    g = torch.Generator().manual_seed(seed)
    order = group_ids.unique()[torch.randperm(group_ids.unique().numel(), generator=g)]
    sort = torch.argsort(group_ids, stable=True)
    gs = group_ids[sort]
    out = []
    for grp in order.tolist():
        lo, hi = torch.searchsorted(gs, grp).item(), torch.searchsorted(gs, grp, right=True).item()
        for i in sort[lo:hi].tolist():
            c = int(labels[i])
            if c >= 0 and need[c] > 0:
                need[c] -= 1
                out.append(i)
        if int(need.sum()) == 0:
            break
    return torch.tensor(sorted(out), dtype=torch.long)


def _take(batch, key="data"):
    return batch[key] if isinstance(batch, dict) else batch[0]


# ====================================================================== image pass
@torch.no_grad()
def image_features(model, e: dict) -> Dict[str, torch.Tensor]:
    """Three understanding features of an encoded image batch, each [B, d] float32."""
    zs = model.semantic(e)[:, 0]
    zr = e["zc"][:, 0].mean(1)                 # mean over the N cells -> [B, zr_c_image]
    cls = e["cls"][:, 0]
    return dict(zs=zs.float(), zr_c=zr.float(), cls=cls.float())


@torch.no_grad()
def decode_image(model, e: dict) -> torch.Tensor:
    cfg = model.cfg
    o = model.decode_frame(e["zc"][:, 0], None, 0.0, "image")
    return patches_to_image(o["rgb"], cfg.grid, cfg.patch).clamp(-1, 1)


@torch.no_grad()
def image_pass(model, loader: Iterable, device, *, amp: bool = False, limit: Optional[int] = None,
               recon: bool = True, feats: bool = True, lpips: Optional[OptionalLPIPS] = None,
               fid: Optional[OptionalFID] = None, log: Optional[Logger] = None) -> dict:
    """One pass over an image loader: encode once, collect features / labels and reconstruction metrics."""
    model.eval()
    sums = dict(psnr=0.0, ssim=0.0, lpips=0.0)
    n = 0
    fbuf: Dict[str, List[torch.Tensor]] = dict(zs=[], zr_c=[], cls=[])
    labels: List[torch.Tensor] = []
    log = log or Logger()
    for batch in loader:
        x = _take(batch).to(device, non_blocking=True)
        if limit is not None:
            if n >= limit:
                break
            x = x[: limit - n]
        B = x.shape[0]
        with autocast(device, amp):
            e = model.encode(x, "image")
            if feats:
                for k, v in image_features(model, e).items():
                    fbuf[k].append(v.cpu())
            rec = decode_image(model, e) if recon else None
        if feats and isinstance(batch, dict) and "label" in batch:
            labels.append(torch.as_tensor(batch["label"])[:B].long().cpu())
        if recon:
            rec = rec.float()
            xf = x.float()
            sums["psnr"] += psnr(rec, xf).sum().item()
            sums["ssim"] += ssim(rec, xf).sum().item()
            if lpips is not None and lpips.available:
                sums["lpips"] += lpips(rec, xf).sum().item()
            if fid is not None and fid.available:
                fid.update(xf, rec)
        n += B
        log(f"image pass: {n}" + (f"/{limit}" if limit else "") +
            (f"  psnr={sums['psnr'] / n:.2f}" if recon else ""))
    out: dict = dict(n=n)
    if recon and n:
        out["psnr"] = sums["psnr"] / n
        out["ssim"] = sums["ssim"] / n
        if lpips is not None and lpips.available:
            out["lpips_vgg"] = sums["lpips"] / n
        if fid is not None and fid.available:
            try:
                out["rfid"] = fid.compute()
            except Exception as ex:  # noqa: BLE001
                out["rfid_error"] = f"{type(ex).__name__}: {ex}"
    if feats:
        out["features"] = {k: torch.cat(v) if v else torch.empty(0) for k, v in fbuf.items()}
        out["labels"] = torch.cat(labels) if labels else None
    return out


# ====================================================================== video pass
@torch.no_grad()
def video_pass(model, loader: Iterable, device, *, amp: bool = False, limit: Optional[int] = None,
               log: Optional[Logger] = None) -> dict:
    """Per-frame reconstruction + C/D swap diagnostics on video clips [B, 3, T, H, W]."""
    model.eval()
    cfg = model.cfg
    K = cfg.keyframe_every
    log = log or Logger()
    acc: Dict[str, List[float]] = {}

    def add(key, vals):
        acc.setdefault(key, []).extend(vals.flatten().tolist())

    n = 0
    T_seen = None
    for batch in loader:
        x = _take(batch).to(device, non_blocking=True)
        if limit is not None:
            if n >= limit:
                break
            x = x[: limit - n]
        B, _, T = x.shape[:3]
        T_seen = T
        with autocast(device, amp):
            e = model.encode(x, "video")
            seg = e["state"]["seg"]
            for t in range(T):
                zc = e["zc"][:, seg[t]]
                zd = e["zd"][:, t] if "zd" in e else None
                tgt = e["frames"][:, t].float()
                o = model.decode_frame(zc, zd, float(t), "video")
                rec = patches_to_image(o["rgb"], cfg.grid, cfg.patch).clamp(-1, 1).float()
                p, s = psnr(rec, tgt), ssim(rec, tgt)
                kind = "key" if t % K == 0 else "nonkey"
                add("psnr", p); add("ssim", s); add(f"psnr_{kind}", p); add(f"ssim_{kind}", s)
                if o["gate"] is not None:
                    add("gate", o["gate"].float().mean(dim=(1, 2)))
                    add(f"gate_{kind}", o["gate"].float().mean(dim=(1, 2)))
                add("delta_abs_cells", o["delta"].float().abs().mean(dim=(1, 2)) * cfg.grid)
                if zd is not None:
                    # swap diagnostics: what D_t contributes on top of C_k
                    o0 = model.decode_frame(zc, torch.zeros_like(zd), float(t), "video")
                    r0 = patches_to_image(o0["rgb"], cfg.grid, cfg.patch).clamp(-1, 1).float()
                    p0 = psnr(r0, tgt)
                    add("psnr_zeroD", p0); add(f"psnr_zeroD_{kind}", p0)
                    add("gap_zeroD", p - p0); add(f"gap_zeroD_{kind}", p - p0)
                    o1 = model.decode_frame(zc, None, float(t), "video")
                    r1 = patches_to_image(o1["rgb"], cfg.grid, cfg.patch).clamp(-1, 1).float()
                    p1 = psnr(r1, tgt)
                    add("psnr_noD", p1); add("gap_noD", p - p1); add(f"gap_noD_{kind}", p - p1)
        n += B
        log(f"video pass: {n}" + (f"/{limit}" if limit else "") + f"  psnr={sum(acc['psnr']) / len(acc['psnr']):.2f}")
    out = {k: sum(v) / len(v) for k, v in sorted(acc.items()) if v}
    out["n_clips"] = n
    out["n_frames"] = T_seen
    out["keyframe_every"] = K
    out["budget_per_segment"] = cfg.budget("video")
    out["budget_per_frame"] = cfg.budget("video") / K
    return out


# ====================================================================== kNN / linear probe
def l2n(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


@torch.no_grad()
def knn_classify(train_f: torch.Tensor, train_y: torch.Tensor, test_f: torch.Tensor, test_y: torch.Tensor,
                 k: int = 20, temperature: float = 0.07, num_classes: Optional[int] = None,
                 device="cpu", chunk: int = 1024) -> float:
    """DINO-style weighted kNN top-1 accuracy (cosine similarity, weights exp(sim / T))."""
    C = int(num_classes or (max(int(train_y.max()), int(test_y.max())) + 1))
    tr = l2n(train_f).to(device)
    ty = train_y.to(device)
    k = min(k, tr.shape[0])
    correct = 0
    for i in range(0, test_f.shape[0], chunk):
        q = l2n(test_f[i:i + chunk]).to(device)
        sim, idx = (q @ tr.T).topk(k, dim=1)
        w = (sim / temperature).exp()
        votes = torch.zeros(q.shape[0], C, device=device)
        votes.scatter_add_(1, ty[idx], w)
        correct += (votes.argmax(1).cpu() == test_y[i:i + chunk]).sum().item()
    return correct / max(1, test_f.shape[0])


def _train_logreg(x: torch.Tensor, y: torch.Tensor, C: int, lr: float, epochs: int, wd: float,
                  batch: int, device, seed: int = 0) -> nn.Linear:
    g = torch.Generator().manual_seed(seed)
    lin = nn.Linear(x.shape[1], C).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(x.shape[0] / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps))
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], generator=g)
        for i in range(0, x.shape[0], batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(lin(x[b.to(x.device)]), y[b.to(y.device)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return lin


@torch.no_grad()
def _acc(lin: nn.Linear, x: torch.Tensor, y: torch.Tensor, chunk: int = 8192) -> float:
    c = 0
    for i in range(0, x.shape[0], chunk):
        c += (lin(x[i:i + chunk]).argmax(1) == y[i:i + chunk]).sum().item()
    return c / max(1, x.shape[0])


def linear_probe(train_f: torch.Tensor, train_y: torch.Tensor, test_f: torch.Tensor, test_y: torch.Tensor,
                 num_classes: Optional[int] = None, lrs: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2),
                 epochs: int = 10, wd: float = 1e-4, batch: int = 1024, holdout: float = 0.1,
                 device="cpu", seed: int = 0) -> dict:
    """Logistic regression on L2-normalised (then x sqrt(d)-scaled) features.

    lr chosen on a `holdout` split of the train set; the final probe is re-trained on all train data.
    """
    C = int(num_classes or (max(int(train_y.max()), int(test_y.max())) + 1))
    scale = math.sqrt(train_f.shape[1])
    xtr, ytr = (l2n(train_f) * scale).to(device), train_y.long().to(device)
    xte, yte = (l2n(test_f) * scale).to(device), test_y.long().to(device)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(xtr.shape[0], generator=g).to(device)
    n_ho = max(1, int(holdout * xtr.shape[0]))
    ho, fit = perm[:n_ho], perm[n_ho:]
    grid = {}
    with torch.enable_grad():
        for lr in lrs:
            lin = _train_logreg(xtr[fit], ytr[fit], C, lr, epochs, wd, batch, device, seed)
            grid[lr] = _acc(lin, xtr[ho], ytr[ho])
        best = max(grid, key=lambda k: grid[k])
        lin = _train_logreg(xtr, ytr, C, best, epochs, wd, batch, device, seed)
    return dict(top1=_acc(lin, xte, yte), lr=best, holdout_acc={str(k): v for k, v in grid.items()})


def class_balanced_indices(labels: torch.Tensor, per_class: int, seed: int = 0) -> torch.Tensor:
    """Up to `per_class` random indices of each class, sorted."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.as_tensor(labels).long()
    out = []
    for c in labels.unique():
        idx = (labels == c).nonzero().flatten()
        out.append(idx[torch.randperm(idx.numel(), generator=g)[:per_class]])
    return torch.cat(out).sort().values


def understanding(train: dict, val: dict, *, k: int = 20, temperature: float = 0.07,
                  num_classes: Optional[int] = None, probe_epochs: int = 10,
                  probe_lrs: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2), device="cpu",
                  log: Optional[Logger] = None) -> dict:
    """kNN + linear probe for each feature in train['features']; gaps of zs / zr_c to the CLS baseline."""
    log = log or Logger()
    res: dict = {}
    for name in train["features"]:
        ftr, fva = train["features"][name], val["features"][name]
        knn = knn_classify(ftr, train["labels"], fva, val["labels"], k=k, temperature=temperature,
                           num_classes=num_classes, device=device)
        lp = linear_probe(ftr, train["labels"], fva, val["labels"], num_classes=num_classes, lrs=probe_lrs,
                          epochs=probe_epochs, device=device)
        res[name] = dict(dim=int(ftr.shape[1]), knn_top1=knn, linear_top1=lp["top1"], linear_lr=lp["lr"],
                         linear_holdout=lp["holdout_acc"])
        log(f"{name}: knn={knn:.4f} linear={lp['top1']:.4f}", force=True)
    if "cls" in res:
        for name in ("zs", "zr_c"):
            if name in res:
                res[name]["knn_gap_to_cls"] = res[name]["knn_top1"] - res["cls"]["knn_top1"]
                res[name]["linear_gap_to_cls"] = res[name]["linear_top1"] - res["cls"]["linear_top1"]
    res["n_train"] = int(train["labels"].numel())
    res["n_val"] = int(val["labels"].numel())
    return res


# ====================================================================== top level
def run_eval(model, device, *, image_val=None, image_train=None, video=None, batch_size: int = 32,
             video_batch_size: int = 4, num_workers: int = 0, amp: bool = False, limit: Optional[int] = None,
             video_limit: Optional[int] = None, knn_k: int = 20, knn_t: float = 0.07,
             num_classes: Optional[int] = None, probe_epochs: int = 10,
             probe_lrs: Sequence[float] = (1e-3, 3e-3, 1e-2, 3e-2), use_lpips: bool = True,
             use_fid: bool = True, extra_video: Optional[dict] = None, log: Optional[Logger] = None) -> dict:
    """Run every eval whose dataset (or ready DataLoader) is given. extra_video: {name: dataset | None}."""
    log = log or Logger()
    cfg: DinoV0Config = model.cfg
    res: dict = dict(budget=dict(image=cfg.budget("image"), video_segment=cfg.budget("video"),
                                 video_per_frame=cfg.budget("video") / cfg.keyframe_every),
                     notes=[])
    val_feats = None
    if image_val is not None:
        lp = OptionalLPIPS(device) if use_lpips else None
        fid = OptionalFID(device) if use_fid else None
        for opt in (lp, fid):
            if opt is not None and not opt.available:
                res["notes"].append(opt.reason)
                log(opt.reason, force=True)
        out = image_pass(model, as_loader(image_val, batch_size, num_workers), device, amp=amp, limit=limit,
                         recon=True, feats=image_train is not None, lpips=lp, fid=fid, log=log)
        val_feats = dict(features=out.pop("features", None), labels=out.pop("labels", None))
        out["budget"] = cfg.budget("image")
        res["image_recon"] = out
        log(f"image recon: {json.dumps(out)}", force=True)
    if image_train is not None and val_feats is not None and val_feats["labels"] is not None:
        tr = image_pass(model, as_loader(image_train, batch_size, num_workers), device, amp=amp,
                        recon=False, feats=True, log=log)
        res["understanding"] = understanding(tr, val_feats, k=knn_k, temperature=knn_t, num_classes=num_classes,
                                             probe_epochs=probe_epochs, probe_lrs=probe_lrs, device=device,
                                             log=log)
    elif image_train is not None:
        res["notes"].append("understanding skipped: image val set has no labels or was not given")
    if video is not None:
        res["video_recon"] = video_pass(model, as_loader(video, video_batch_size, num_workers), device,
                                        amp=amp, limit=video_limit, log=log)
        log(f"video recon: {json.dumps(res['video_recon'])}", force=True)
    for name, ds in (extra_video or {}).items():
        if ds is None:
            msg = f"{name} probe skipped: no dataset path given"
            res["notes"].append(msg)
            log(msg, force=True)
            continue
        res[f"video_{name}"] = video_pass(model, as_loader(ds, video_batch_size, num_workers), device,
                                          amp=amp, limit=video_limit, log=log)
    return res


def _f(v, fmt="{:.4f}"):
    return "-" if v is None else fmt.format(v)


def to_markdown(res: dict) -> str:
    lines = [f"# MAVT DINOv3 v0 eval", ""]
    if "meta" in res:
        lines += [f"- {k}: {v}" for k, v in res["meta"].items() if k != "cfg"] + [""]
    b = res.get("budget", {})
    lines += [f"Latent budget: image {b.get('image')} values; video {b.get('video_segment')} per "
              f"K-frame segment ({_f(b.get('video_per_frame'), '{:.0f}')} per frame)", ""]
    ir = res.get("image_recon")
    if ir:
        lines += ["## Image reconstruction", "", "| n | PSNR | SSIM | LPIPS-VGG | rFID |", "|---|---|---|---|---|",
                  f"| {ir['n']} | {_f(ir.get('psnr'), '{:.2f}')} | {_f(ir.get('ssim'))} | "
                  f"{_f(ir.get('lpips_vgg'))} | {_f(ir.get('rfid'), '{:.2f}')} |", ""]
    for key in [k for k in res if k.startswith("video_") and isinstance(res[k], dict)]:
        v = res[key]
        lines += [f"## {key.replace('_', ' ')}", "",
                  "| clips x T | PSNR | SSIM | PSNR key | PSNR non-key | PSNR D=0 | gap D=0 (key / non-key) "
                  "| gap no-D | mean gate | mean abs delta (cells) |",
                  "|---|---|---|---|---|---|---|---|---|---|",
                  f"| {v['n_clips']} x {v['n_frames']} | {_f(v.get('psnr'), '{:.2f}')} | {_f(v.get('ssim'))} | "
                  f"{_f(v.get('psnr_key'), '{:.2f}')} | {_f(v.get('psnr_nonkey'), '{:.2f}')} | "
                  f"{_f(v.get('psnr_zeroD'), '{:.2f}')} | {_f(v.get('gap_zeroD'), '{:.2f}')} "
                  f"({_f(v.get('gap_zeroD_key'), '{:.2f}')} / {_f(v.get('gap_zeroD_nonkey'), '{:.2f}')}) | "
                  f"{_f(v.get('gap_noD'), '{:.2f}')} | {_f(v.get('gate'), '{:.3f}')} | "
                  f"{_f(v.get('delta_abs_cells'), '{:.3f}')} |", ""]
    u = res.get("understanding")
    if u:
        lines += [f"## Understanding (ImageNet, train {u['n_train']} / val {u['n_val']})", "",
                  "| feature | dim | kNN top-1 | gap to CLS | linear top-1 | gap to CLS |", "|---|---|---|---|---|---|"]
        names = {"zs": "z_s", "zr_c": "mean-pooled z_r(C)", "cls": "DINOv3 CLS (frozen)"}
        for k in ("zs", "zr_c", "cls"):
            if k in u:
                r = u[k]
                lines.append(f"| {names[k]} | {r['dim']} | {_f(r['knn_top1'])} | {_f(r.get('knn_gap_to_cls'), '{:+.4f}')} "
                             f"| {_f(r['linear_top1'])} | {_f(r.get('linear_gap_to_cls'), '{:+.4f}')} |")
        lines.append("")
    if res.get("notes"):
        lines += ["## Notes", ""] + [f"- {n}" for n in res["notes"]] + [""]
    return "\n".join(lines)


def write_results(res: dict, out_dir: str, stem: str = "eval") -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    jp, mp = os.path.join(out_dir, f"{stem}.json"), os.path.join(out_dir, f"{stem}.md")
    with open(jp, "w") as f:
        json.dump(res, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(mp, "w") as f:
        f.write(to_markdown(res))
    return jp, mp
