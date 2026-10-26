"""Objectives of section 14: every term is a mean over the valid queries / examples of its own source.

    L_F = mean_q [1 - <F/|F|, F^/|F^|> + w_logn (log |F^| - log |F|)^2]
    L_X = mean |X - X^| + w_lpips LPIPS(X, X^)            (X in [0, 1]; LPIPS gets [-1, 1])
    L_depth = Huber_delta(log d^ - log d)                  (metric: no per-frame median alignment)
    L_pose  = |log(R_gt^T R)| + |t - t_gt|_1 / s_scene
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from . import geometry as g

EPS = 1e-6


def feat_loss(pred: torch.Tensor, target: torch.Tensor, w_logn: float, mask: Optional[torch.Tensor] = None):
    """pred / target [..., D] -> scalar (mean over the rows of mask, or all rows)."""
    p, t = pred.float(), target.float()
    pn, tn = p.norm(dim=-1), t.norm(dim=-1)
    cos = (p * t).sum(-1) / (pn.clamp_min(EPS) * tn.clamp_min(EPS))
    v = 1 - cos + w_logn * (torch.log(pn + EPS) - torch.log(tn + EPS)) ** 2
    if mask is None:
        return v.mean()
    m = mask.float()
    return (v * m).sum() / m.sum().clamp_min(1)


def huber_log_depth(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    """pred / gt depth [...] (> 0 where mask) -> Huber of the log difference over the mask (0 if empty)."""
    m = mask & (gt > 0) & torch.isfinite(gt)
    if not m.any():
        return pred.sum() * 0
    d = torch.log(pred.float().clamp_min(EPS)) - torch.log(gt.float().clamp_min(EPS))
    return F.huber_loss(d[m], torch.zeros_like(d[m]), delta=delta)


def pose_loss(R: torch.Tensor, t: torch.Tensor, R_gt: torch.Tensor, t_gt: torch.Tensor, s_scene: torch.Tensor):
    """[B, 3, 3], [B, 3] each; s_scene [B] -> scalar."""
    w = g.se3_log(R_gt.transpose(-1, -2) @ R, torch.zeros_like(t))[..., 3:]
    return (w.norm(dim=-1) + (t - t_gt).abs().sum(-1) / s_scene.clamp_min(EPS)).mean()


def sample_cells(x: torch.Tensor, u: torch.Tensor, grid: int) -> torch.Tensor:
    """Bilinear read of a per-cell map x [B, G*G] at normalised positions u [B, M, 2] -> [B, M]."""
    B = x.shape[0]
    img = x.float().view(B, 1, grid, grid)
    out = F.grid_sample(img, (u.float() * 2 - 1).unsqueeze(1), mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.view(B, -1)
