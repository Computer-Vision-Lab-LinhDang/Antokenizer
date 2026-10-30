"""Objectives of section 14: every term is a mean over the valid queries / examples of its own source.

    L_F = mean_q [1 - <F/|F|, F^/|F^|> + w_logn (log |F^| - log |F|)^2]
    L_X = mean |X - X^| + w_lpips LPIPS(X, X^)            (X in [0, 1]; LPIPS gets [-1, 1])
    L_depth = Huber_delta(log d^ - log d)                  (metric: no per-frame median alignment)
    L_pose  = |log(R_gt^T R)| + |t - t_gt|_1 / s_scene
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
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
    return g.read_coarse(x.float().unsqueeze(-1), u, grid).squeeze(-1)


class PatchDisc(nn.Module):
    """PatchGAN discriminator (Isola et al. 2017, as in VQGAN) on images in [-1, 1]: conv 4 x 4 stride 2 per layer,
    GroupNorm (no batch statistics under DDP), LeakyReLU 0.2 -> a map of patch logits."""

    def __init__(self, channels: int = 64, layers: int = 3):
        super().__init__()
        mods, c = [nn.Conv2d(3, channels, 4, 2, 1), nn.LeakyReLU(0.2)], channels
        for i in range(1, layers + 1):
            c2 = channels * min(2 ** i, 8)
            groups = max(g for g in range(1, 33) if c2 % g == 0 and g <= max(1, c2 // 4))   # divides c2
            mods += [nn.Conv2d(c, c2, 4, 2 if i < layers else 1, 1), nn.GroupNorm(groups, c2),
                     nn.LeakyReLU(0.2)]
            c = c2
        mods.append(nn.Conv2d(c, 1, 4, 1, 1))
        self.net = nn.Sequential(*mods)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class StubSemanticTeacher:
    """2.5 stub teacher (tests, dry runs): a fixed random projection of the 4 x 4 average-pooled image [B, 48] to
    [B, dim]. Not a module: no parameters, nothing in the state_dict, never trained."""

    def __init__(self, dim: int, seed: int = 0):
        self.W = torch.randn(48, dim, generator=torch.Generator().manual_seed(seed)) / 48 ** 0.5

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(frames.float(), 4).flatten(1)
        return x @ self.W.to(x.device)


def semantic_teacher(name: str, dim: int):
    """2.5: the built-in teacher of sem_target, or None for the external ones (dinov3txt, siglip2), which need their
    weights: attach with `model.set_sem_teacher(fn)`, fn(frames [B, 3, H, W] in [-1, 1]) -> [B, sem_dim] (plain
    assignment of an nn.Module would register it in the state_dict)."""
    return StubSemanticTeacher(dim) if name == "stub" else None


def sem_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos per sample, mean."""
    return (1 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()


def band_fractions(err: torch.Tensor) -> tuple:
    """Energy fractions of err [..., S, S] in the low (< 1/32), mid and high (>= 1/8 cycles / px) radial bands."""
    E = torch.fft.rfft2(err.float()).abs().pow(2)
    S = err.shape[-1]
    fy = torch.fft.fftfreq(S, device=err.device).view(-1, 1)
    fx = torch.fft.rfftfreq(S, device=err.device).view(1, -1)
    fr = (fx ** 2 + fy ** 2).sqrt()
    tot = E.sum().clamp_min(1e-12)
    return E[..., fr < 1 / 32].sum() / tot, E[..., (fr >= 1 / 32) & (fr < 1 / 8)].sum() / tot, E[..., fr >= 1 / 8].sum() / tot


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """3.8: mean |Sobel(pred) - Sobel(target)| over both directions, channels and pixels (reflect padding).
    pred / target [N, C, H, W] in any common range; the /8 normalisation makes a unit step edge give |dx| = 1 on
    the two columns beside it."""
    p, t = pred.float(), target.float()
    C = p.shape[1]
    kx = _SOBEL_X.to(p.device, p.dtype).view(1, 1, 3, 3).expand(C, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    d = p - t
    d = F.pad(d, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(d, kx, groups=C)
    gy = F.conv2d(d, ky, groups=C)
    return (gx.abs().mean() + gy.abs().mean())
