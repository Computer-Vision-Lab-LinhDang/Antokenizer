"""Regularisers of C shared by the two-stream model and the vanilla one: KL of c, edge-aware smoothness of
rho, the frozen depth probe (L_depth_prior), rendered depth (L_depth_gt) and rho self-distillation."""
from __future__ import annotations

from typing import Optional

import torch

from . import geometry as geo
from .depth import apply_probe, cell_log_inv_depth, load_probe, prior_loss, silog
from .losses import edge_aware_smooth, kl


def _pearson(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-row correlation of [B, N] tensors over the masked entries (how well rho follows a depth target)."""
    w = torch.ones_like(a, dtype=torch.float32) if mask is None else mask.float()
    n = w.sum(-1, keepdim=True).clamp_min(1.0)
    a = (a.float() - (a.float() * w).sum(-1, keepdim=True) / n) * w
    b = (b.float() - (b.float() * w).sum(-1, keepdim=True) / n) * w
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-6)


class DepthRegs:
    """Mixin for an nn.Module with `cfg`: call init_probe() in __init__, then _c_regs() per keyframe batch."""

    def init_probe(self, cfg, feat_dim: int) -> None:
        """Frozen depth probe (L_depth_prior) as non-persistent buffers: it is not part of the tokenizer, so
        checkpoints neither carry it nor need it on resume."""
        self.probe_idx = None
        if cfg.w_depth_prior > 0 and not cfg.depth_probe:
            raise ValueError("w_depth_prior > 0 needs depth_probe: a probe file written by "
                             "scripts/train_depth_probe.py (linear probe on real depth labels)")
        if cfg.depth_probe:
            ck = load_probe(cfg.depth_probe)
            if int(ck["dim"]) != feat_dim:
                raise ValueError(f"depth probe expects {ck['dim']}-dim DINO features, the encoder has {feat_dim}")
            if any(int(l) not in cfg.dino_layers for l in ck["layers"]):
                raise ValueError(f"depth probe layers {ck['layers']} must be among dino_layers {cfg.dino_layers}")
            self.probe_idx = [list(cfg.dino_layers).index(int(l)) for l in ck["layers"]]
            self.register_buffer("probe_weight", ck["weight"].float(), persistent=False)
            self.register_buffer("probe_bias", ck["bias"].float(), persistent=False)

    def probe_rho(self, layers: list) -> torch.Tensor:
        """Frozen probe on the keyframe's DINO layers (list over cfg.dino_layers of [B, N, D]) -> [B, N]."""
        with geo.fp32(layers[0].device):
            return apply_probe([layers[i] for i in self.probe_idx], self.probe_weight, self.probe_bias).detach()

    def _c_regs(self, ch, rho, feat_key, parts, diag, layers_key=None, modality=None, depth_gt=None):
        cfg = self.cfg
        loss = 0.0
        k = kl(ch["mu"], ch["logvar"])
        parts["kl_c"] = k.detach()
        loss = loss + cfg.w_kl_c * k
        if cfg.w_smooth > 0:
            s = edge_aware_smooth(rho, feat_key, cfg.grid)
            parts["smooth"] = s.detach()
            loss = loss + cfg.w_smooth * s
        if self.probe_idx is not None and layers_key is not None:
            target = self.probe_rho(layers_key)
            dp = prior_loss(cfg.depth_prior_kind, rho, target)
            parts["depth_prior"] = dp.detach()
            diag["rho_probe_corr"] = _pearson(rho.detach(), target).mean()
            if cfg.w_depth_prior > 0 and modality in cfg.depth_prior_on:
                loss = loss + cfg.w_depth_prior * dp
        mask = None
        if depth_gt is not None:
            tgt, mask = cell_log_inv_depth(depth_gt, depth_gt > 0, cfg.grid)
            dg = silog(rho, tgt, mask, lam=cfg.depth_gt_lam)
            parts["depth_gt"] = dg.detach()
            diag["rho_gt_corr"] = _pearson(rho.detach(), tgt, mask).mean()
            if cfg.w_depth_gt > 0:
                loss = loss + cfg.w_depth_gt * dg
        if "rho_mono" in ch:
            # the single-view part of rho learns what the cross-view refinement adds; 0 while the gate is
            # closed (tanh(alpha) = 0 at init); only on cells with rendered depth when the batch has it
            # (background cells are not constrained by any view)
            w = mask.float() if mask is not None else torch.ones_like(rho)
            dd = ((ch["rho_mono"] - rho.detach()).abs() * w).sum() / w.sum().clamp_min(1.0)
            parts["rho_distill"] = dd.detach()
            if cfg.w_rho_distill > 0:
                loss = loss + cfg.w_rho_distill * dd
        diag["rho_std"] = rho.detach().std(-1).mean()
        return loss
