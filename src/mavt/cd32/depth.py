"""Depth for the C-D 3.2 tokenizer: the depth probe on frozen DINOv3, depth losses and depth metrics.

rho is the log inverse depth of a cell. On monocular video and still images it is known only up to an
additive constant (the global scale of the scene), so the losses below compare log depths after taking
out that constant (shift-invariant in log depth = scale-invariant in depth) unless they say otherwise.

Probe (bản 3.2, L_depth_prior). A linear map on frozen DINOv3 patch features,

    rho_probe(u) = w . [ LN(F_l(u) - mean_u F_l(u)) for l in layers ] + b          (LN without affine)

trained once on real depth labels (scripts/train_depth_probe.py) and then frozen. It is the only
external label of the framework. During tokenizer training it is a soft anchor for rho on images and
monocular video, where the decoder alone cannot tell a moving object from a nearer static one.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

PRIOR_KINDS = ("silog", "ssi", "ordinal")


# ---------------------------------------------------------------------------------- labels
def cell_log_inv_depth(depth: torch.Tensor, valid: torch.Tensor, grid: int,
                       min_frac=0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel depth [B, H, W] (> 0 where valid) -> per-cell target and mask on the grid x grid cells.

    The target of a cell is the mean of -log depth over its valid pixels (log of the geometric mean of
    inverse depth). A cell counts only when at least `min_frac` (a float, or one per image [B]) of its
    pixels are valid, and always when it has at least one valid pixel and min_frac is 0.
    Returns (target [B, grid^2] float32, mask [B, grid^2] bool)."""
    B, H, W = depth.shape
    v = valid & (depth > 0) & torch.isfinite(depth)
    lid = torch.where(v, -torch.log(depth.float().clamp_min(1e-6)), torch.zeros((), device=depth.device))
    s = F.adaptive_avg_pool2d(lid.unsqueeze(1), grid).flatten(1)
    n = F.adaptive_avg_pool2d(v.float().unsqueeze(1), grid).flatten(1)
    if torch.is_tensor(min_frac):
        min_frac = min_frac.to(n.device, n.dtype).view(-1, 1)
    mask = (n >= min_frac) & (n > 0)
    return torch.where(mask, s / n.clamp_min(1e-6), torch.zeros((), device=depth.device)), mask


def _mask_like(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return torch.ones_like(x, dtype=torch.bool) if mask is None else mask.bool()


def masked_median(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Median over the last dim of the masked entries: [B, N] -> [B, 1] (0 for rows with no entry)."""
    m = torch.nanmedian(torch.where(mask, x.float(), torch.full((), float("nan"), device=x.device)), -1,
                        keepdim=True).values
    return torch.nan_to_num(m, nan=0.0)


# ---------------------------------------------------------------------------------- losses
def silog(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None,
          lam: float = 0.85) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al. 2014), per image then averaged:
    sqrt(mean d^2 - lam (mean d)^2), d = pred - target in log depth, over the masked cells.
    lam = 1 is fully shift-invariant in log depth (the std of d); lam < 1 also asks for the right scale."""
    mask = _mask_like(pred, mask)
    w = mask.float()
    n = w.sum(-1).clamp_min(1.0)
    d = (pred.float() - target.float()) * w
    m1, m2 = d.sum(-1) / n, (d * d).sum(-1) / n
    var = (m2 - lam * m1 * m1).clamp_min(1e-8)
    keep = (w.sum(-1) >= 2).float()
    return (var.sqrt() * keep).sum() / keep.sum().clamp_min(1.0)


def ssi(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scale- and shift-invariant L1 (MiDaS-style alignment by median and mean absolute deviation),
    applied to log depth: also forgives a power law between the two depths, so it is the weakest prior."""
    mask = _mask_like(pred, mask)

    def norm(x):
        x = x.float()
        c = x - masked_median(x, mask)
        s = (c.abs() * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
        return c / s.clamp_min(1e-6)
    w = mask.float()
    keep = (w.sum(-1) >= 2).float()
    per = ((norm(pred) - norm(target)).abs() * w).sum(-1) / w.sum(-1).clamp_min(1.0)
    return (per * keep).sum() / keep.sum().clamp_min(1.0)


def ordinal(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None,
            n_pairs: int = 512, tau: float = 0.03, temp: float = 0.1,
            generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Pairwise depth-order loss (Chen et al. 2016) on random pairs of masked cells per image:
    softplus(-r (p_i - p_j) / temp) where the targets differ by more than tau in log depth (r their sign),
    (p_i - p_j)^2 / temp where they are about equal. Shift- and scale-free: it only asks for the order."""
    mask = _mask_like(pred, mask)
    B, N = pred.shape
    w = mask.float() + 1e-6                                       # rows with no valid cell stay finite
    i = torch.multinomial(w, n_pairs, replacement=True, generator=generator)
    j = torch.multinomial(w, n_pairs, replacement=True, generator=generator)
    ok = (torch.gather(mask, 1, i) & torch.gather(mask, 1, j) & (i != j)).float()
    dp = torch.gather(pred.float(), 1, i) - torch.gather(pred.float(), 1, j)
    dt = torch.gather(target.float(), 1, i) - torch.gather(target.float(), 1, j)
    far = (dt.abs() > tau).float()
    lo = far * F.softplus(-torch.sign(dt) * dp / temp) + (1 - far) * dp.pow(2) / temp
    return (lo * ok).sum() / ok.sum().clamp_min(1.0)


def prior_loss(kind: str, rho: torch.Tensor, target: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L_depth_prior between rho [B, N] and the (detached) probe output [B, N]."""
    if kind == "silog":
        return silog(rho, target, mask, lam=1.0)
    if kind == "ssi":
        return ssi(rho, target, mask)
    if kind == "ordinal":
        return ordinal(rho, target, mask)
    raise ValueError(f"depth prior kind must be one of {PRIOR_KINDS}, got {kind!r}")


# ---------------------------------------------------------------------------------- metrics
METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "silog", "d1", "d2", "d3")


def upsample_rho(rho: torch.Tensor, grid: int, size: int) -> torch.Tensor:
    """Cell rho [B, grid^2] -> pixel log inverse depth [B, size, size] (bilinear)."""
    r = rho.float().view(-1, 1, grid, grid)
    return F.interpolate(r, size=(size, size), mode="bilinear", align_corners=False)[:, 0]


@torch.no_grad()
def depth_metrics(pred_lid: torch.Tensor, gt_depth: torch.Tensor, valid: torch.Tensor,
                  align: str = "median") -> dict:
    """Standard monocular depth errors, summed over images (divide by 'n' to average).

    pred_lid [B, ...] predicted log inverse depth, gt_depth [B, ...] (> 0), valid [B, ...] bool, on the
    same grid (cells or pixels). align = 'median': scale the prediction per image so that its median
    depth equals the ground truth's (the usual protocol for depth known up to scale); 'none': as is."""
    B = pred_lid.shape[0]
    p = torch.exp(-pred_lid.float()).reshape(B, -1)
    g = gt_depth.float().reshape(B, -1)
    v = (valid.reshape(B, -1) & (g > 0) & torch.isfinite(g) & torch.isfinite(p))
    out = {k: 0.0 for k in METRICS}
    out["n"] = 0
    for b in range(B):
        pb, gb = p[b][v[b]], g[b][v[b]]
        if gb.numel() < 2:
            continue
        if align == "median":
            pb = pb * (gb.median() / pb.median().clamp_min(1e-12))
        elif align != "none":
            raise ValueError(f"align must be 'median' or 'none', got {align!r}")
        pb = pb.clamp_min(1e-6)
        ratio = torch.maximum(pb / gb, gb / pb)
        dl = torch.log(pb) - torch.log(gb)
        out["abs_rel"] += ((pb - gb).abs() / gb).mean().item()
        out["sq_rel"] += ((pb - gb).pow(2) / gb).mean().item()
        out["rmse"] += (pb - gb).pow(2).mean().sqrt().item()
        out["rmse_log"] += dl.pow(2).mean().sqrt().item()
        out["silog"] += 100 * (dl.pow(2).mean() - dl.mean().pow(2)).clamp_min(0).sqrt().item()
        for k, t in (("d1", 1.25), ("d2", 1.25 ** 2), ("d3", 1.25 ** 3)):
            out[k] += (ratio < t).float().mean().item()
        out["n"] += 1
    return out


def merge_metrics(acc: Optional[dict], new: dict) -> dict:
    if acc is None:
        return dict(new)
    return {k: acc[k] + new[k] for k in acc}


def mean_metrics(acc: dict) -> dict:
    n = max(1, acc.get("n", 0))
    return {k: (v / n if k != "n" else v) for k, v in acc.items()}


# ---------------------------------------------------------------------------------- probe
def probe_features(layers: Sequence[torch.Tensor]) -> torch.Tensor:
    """[B, N, D] DINOv3 patch features per layer -> [B, N, len * D]: centred over the cells, then LN
    without affine, so the probe on top is linear in the features up to this fixed normalisation."""
    out = []
    for f in layers:
        f = f.float()
        f = f - f.mean(1, keepdim=True)
        out.append(F.layer_norm(f, f.shape[-1:]))
    return torch.cat(out, -1)


class DepthProbe(nn.Module):
    """rho_probe = Linear(probe_features(DINOv3 layers)) -> log inverse depth per cell (up to a shift).

    `layers` are DINOv3 block indices (1-based, as in CD32Config.dino_layers)."""

    def __init__(self, dim: int, layers: Sequence[int]):
        super().__init__()
        self.layers = tuple(int(l) for l in layers)
        self.linear = nn.Linear(dim * len(self.layers), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, layers: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.linear(probe_features(layers)).squeeze(-1)

    def export(self, **meta) -> dict:
        """Checkpoint read by MAVTCD32 (cfg.depth_probe) and scripts/eval_cd32_depth.py."""
        return dict(kind="cd32_depth_probe", weight=self.linear.weight.detach().cpu().view(-1).clone(),
                    bias=self.linear.bias.detach().cpu().view(()).clone(), layers=list(self.layers),
                    dim=self.linear.in_features // len(self.layers), **meta)

    @classmethod
    def from_export(cls, ck: dict) -> "DepthProbe":
        p = cls(int(ck["dim"]), ck["layers"])
        with torch.no_grad():
            p.linear.weight.copy_(ck["weight"].view(1, -1))
            p.linear.bias.copy_(ck["bias"].view(1))
        return p


def load_probe(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or ck.get("kind") != "cd32_depth_probe":
        raise ValueError(f"{path} is not a depth probe written by scripts/train_depth_probe.py")
    return ck


def apply_probe(layers: Sequence[torch.Tensor], weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Functional DepthProbe (the tokenizer keeps the probe as non-saved buffers)."""
    return probe_features(layers) @ weight.float() + bias.float()


def cosine_lr(step: int, total: int, warmup: int, floor: float = 0.05) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
