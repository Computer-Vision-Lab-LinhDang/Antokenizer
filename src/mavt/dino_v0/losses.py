"""Losses of MAVT on DINOv3. Plain functions, each returns a scalar tensor; LPIPSLoss also gives the
Gram loss from the same VGG forward."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def feat_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(1 - cosine).mean() + mse over the last dim."""
    cos = F.cosine_similarity(pred.float(), target.float(), dim=-1)
    return (1 - cos).mean() + F.mse_loss(pred.float(), target.float())


def kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, 1)), mean over all elements."""
    mu, logvar = mu.float(), logvar.float()
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def var_cov(C: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg on C [B, N, D] over the B*N tokens: hinge std >= 1 per channel + off-diag cov^2 mean."""
    x = C.float().reshape(-1, C.shape[-1])
    n, d = x.shape
    x = x - x.mean(0)
    std = (x.var(0) + eps).sqrt()
    var = F.relu(1 - std).mean()
    cov = x.T @ x / max(n - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    cov_l = off.pow(2).sum() / max(d * (d - 1), 1)
    return var + cov_l


def sem_loss(zs: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(zs.float(), cls.float(), dim=-1)).mean()


def gate_reg(gate: torch.Tensor) -> torch.Tensor:
    """v0: (1 - o).mean(); pushes every gate to 1 (the decoder ends up reading only C)."""
    return (1 - gate.float()).mean()


def geo_delta(delta: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, grid: int,
              beta: float = 1.0) -> torch.Tensor:
    """Huber (smooth L1, transition beta) of the decoder shift against its target, both [R, N, 2] in (u, v)
    units, measured in cells and summed over (u, v); mean over the valid [R, N] queries."""
    err = F.smooth_l1_loss(delta.float() * grid, target.float() * grid, beta=beta, reduction="none").sum(-1)
    w = valid.float()
    return (err * w).sum() / w.sum().clamp_min(1.0)


def silog(pred_log: torch.Tensor, depth: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
    """Scale-invariant log-depth loss: pred_log [R, P] log depth, depth [R, P] (0 = invalid); per row
    mean(d^2) - lam mean(d)^2 with d = pred_log - log depth, mean over rows with a valid pixel."""
    m = depth > 0
    d = torch.where(m, pred_log.float() - depth.float().clamp_min(1e-6).log(), torch.zeros_like(pred_log.float()))
    n = m.sum(-1)
    k = n.clamp_min(1)
    loss = (d * d).sum(-1) / k - lam * (d.sum(-1) / k) ** 2
    ok = n > 0
    return loss[ok].mean() if bool(ok.any()) else pred_log.float().sum() * 0


def gate_hinge(gate: torch.Tensor, floor: float) -> torch.Tensor:
    """v1: relu(floor - mean(o)); zero while C carries at least `floor` of the read on average,
    so it penalises only D taking over, not D being used."""
    return F.relu(floor - gate.float().mean())


def gram(f: torch.Tensor) -> torch.Tensor:
    """Feature map [B, C, H, W] -> Gram matrix [B, C, C] / (C * H * W), float32 (autocast off)."""
    B, C, H, W = f.shape
    with torch.autocast(device_type=f.device.type, enabled=False):
        x = f.float().reshape(B, C, H * W)
        return x @ x.transpose(1, 2) / (C * H * W)


def _unit(f: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Unit norm over channels at each position (lpips.normalize_tensor)."""
    return f / (f.pow(2).sum(1, keepdim=True).sqrt() + eps)


class LPIPSLoss(nn.Module):
    """Frozen LPIPS (VGG) on images in [-1, 1]; the `lpips` package is imported lazily.
    pretrained_trunk=False: random VGG trunk, no weight download (tests only)."""

    def __init__(self, net: str = "vgg", pretrained_trunk: bool = True):
        super().__init__()
        try:
            import lpips
        except ImportError as e:
            raise ImportError("LPIPSLoss needs the `lpips` package: pip install lpips "
                              "(or set w_lpips = 0 and w_gram = 0)") from e
        self.model = lpips.LPIPS(net=net, pnet_rand=not pretrained_trunk, verbose=False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def state_dict(self, *args, destination=None, prefix: str = "", keep_vars: bool = False):
        """Frozen and re-built from the lpips package, so never saved (keeps ~60 MB of VGG out of every
        checkpoint); the owner ignores the then-missing keys on load (see ignore_lpips_keys)."""
        return {} if destination is None else destination

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lpips_gram(pred, target)[0]

    def lpips_gram(self, pred: torch.Tensor, target: torch.Tensor, with_gram: bool = False,
                   dist: str = "l1") -> tuple:
        """pred / target [B, 3, H, W] in [-1, 1] -> (LPIPS mean, Gram loss | None) from ONE VGG forward
        over [pred ; target]. LPIPS as lpips.LPIPS.forward (scaling layer, unit-normed features, linear
        heads, spatial mean, summed over layers). Gram: per VGG layer, dist ("l1" | "mse") between the
        Gram matrices (normalised by C * H * W) of the raw feature maps, mean over layers."""
        m = self.model
        assert m.lpips and not m.spatial, "LPIPSLoss expects lpips.LPIPS(lpips=True, spatial=False)"
        B = pred.shape[0]
        x = torch.cat([pred.float(), target.float()])
        if m.version == "0.1":
            x = m.scaling_layer(x)
        lp, gl = 0.0, []
        for kk, f in enumerate(m.net(x)):
            f0, f1 = f[:B], f[B:]
            lp = lp + m.lins[kk]((_unit(f0) - _unit(f1)) ** 2).mean((2, 3), keepdim=True)
            if with_gram:
                d = gram(f0) - gram(f1.detach())
                gl.append(d.abs().mean() if dist == "l1" else d.pow(2).mean())
        return lp.mean(), (torch.stack(gl).mean() if with_gram else None)


def ignore_lpips_keys(module: nn.Module, incompatible) -> None:
    """load_state_dict post-hook for a module that owns an LPIPSLoss: its keys are never saved, so they
    are not 'missing'; LPIPS keys of an older checkpoint load into it or are dropped, never fatal."""
    for keys in (incompatible.missing_keys, incompatible.unexpected_keys):
        keys[:] = [k for k in keys if "lpips" not in k.split(".")]
