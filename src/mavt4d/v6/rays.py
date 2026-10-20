"""Patch rays, Plücker lines, Fourier(log z) and per-patch measured-depth tags. All fp32.

Pixel convention as geometry/camera.py: pixel i covers [i, i + 1), patch (r, c) centre is
((c + 0.5)·p, (r + 0.5)·p); patches are row-major, n = r·Wp + c.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from torch import Tensor


def no_autocast(t: Tensor):
    """Disable autocast locally so geometry stays fp32."""
    if t.device.type in ('cpu', 'cuda', 'xpu', 'mps'):
        return torch.autocast(t.device.type, enabled=False)
    return contextlib.nullcontext()


def patch_rays(intrinsics: Tensor, cfg) -> Tensor:
    """Unit camera-frame rays [B,V,N,3] through the patch centres of a cfg.image_height x cfg.image_width image."""
    with no_autocast(intrinsics):
        K = intrinsics.float()
        p = cfg.patch_size
        hp, wp = cfg.image_height // p, cfg.image_width // p
        r, c = torch.meshgrid(torch.arange(hp, device=K.device, dtype=torch.float32),
                              torch.arange(wp, device=K.device, dtype=torch.float32), indexing='ij')
        u, v = ((c + 0.5) * p).reshape(-1), ((r + 0.5) * p).reshape(-1)
        # Closed-form K^-1 for an upper-triangular pinhole K; zero focal (dead camera rows) -> 1, no NaN.
        fx, fy = K[..., 0, 0, None], K[..., 1, 1, None]
        fx = torch.where(fx.abs() > 1e-6, fx, torch.ones_like(fx))
        fy = torch.where(fy.abs() > 1e-6, fy, torch.ones_like(fy))
        y = (v - K[..., 1, 2, None]) / fy
        x = (u - K[..., 0, 2, None] - K[..., 0, 1, None] * y) / fx
        return F.normalize(torch.stack((x, y, torch.ones_like(x)), -1), dim=-1)


def plucker(origin: Tensor, direction: Tensor) -> Tensor:
    """Plücker line [...,6]: unit direction d and moment m = o x d, rescaled to m/|m| · log(1+|m|)."""
    with no_autocast(direction):
        d = F.normalize(direction.float(), dim=-1)
        o = origin.float().expand_as(d)
        m = torch.linalg.cross(o, d, dim=-1)
        n = m.norm(dim=-1, keepdim=True)
        # log1p(n)/n -> 1 - n/2 near 0
        scale = torch.where(n > 1e-6, torch.log1p(n) / n.clamp_min(1e-6), 1 - n / 2)
        return torch.cat((d, m * scale), -1)


def fourier_log_depth(log_z: Tensor, bands: int) -> Tensor:
    """sin/cos features [...,2*bands] of log z. Frequencies pi/8·2^k: the lowest band has period 16 in log z
    (injective over 1..120 m and over log dt), the highest (bands=6, 4·pi) turns 0.25 rad per 2 % of depth."""
    with no_autocast(log_z):
        x = log_z.float()[..., None]
        freqs = (math.pi / 8) * 2.0 ** torch.arange(bands, device=x.device, dtype=torch.float32)
        angle = x * freqs
        return torch.cat((angle.sin(), angle.cos()), -1)


@torch.no_grad()
def patch_depth_stats(depth: Tensor, depth_valid: Tensor, cfg) -> tuple[Tensor, Tensor]:
    """Measured-depth tag per patch: ([B,V,N,2] = (median log z of valid pixels, log1p(count)), [B,V,N] count > 0).

    Computed outside the network and without gradient, so nothing can copy measured depth into predictions.
    """
    with no_autocast(depth):
        z = depth.detach().float()
        b, v, h, w = z.shape
        p = cfg.patch_size
        if h % p or w % p:
            raise ValueError('depth size must be a multiple of the patch size')
        ok = depth_valid.bool() & torch.isfinite(z) & (z > 0)
        z = z.masked_fill(~ok, float('nan'))

        def tiles(x):
            return x.reshape(b, v, h // p, p, w // p, p).permute(0, 1, 2, 4, 3, 5).reshape(b, v, -1, p * p)

        count = tiles(ok).sum(-1)
        valid = count > 0
        median = torch.nanmedian(tiles(z), dim=-1).values
        log_z = torch.where(valid, torch.where(valid, median, 1.).clamp_min(1e-6).log(), 0.)
        return torch.stack((log_z, torch.log1p(count.float())), -1), valid
