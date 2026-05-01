"""Stage 6: Matryoshka-aware MAVT loss.

For every active prefix d_k the loss accumulates:

    L_k = mod_w · ( w_l1·L1(recon_k, x) + w_lpips·LPIPS(recon_k, x) )
        + β_k     · KL_k                                           (β_k = w_kl · d_k / D_max)
        + w_sem   · cosine_distill(sem_k, teacher)

    L_total = Σ_k α_k · L_k  +  w_aux · slot_diversity

Per-prefix recon and semantic losses are logged individually so curriculum
weighting and posterior-collapse monitoring can be done from W&B.
"""

from __future__ import annotations
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  LPIPS wrapper                                                                #
# --------------------------------------------------------------------------- #

class LPIPSLoss(nn.Module):
    def __init__(self, net: str = 'vgg'):
        super().__init__()
        try:
            import lpips
            self._lpips = lpips.LPIPS(net=net)
            self._available = True
        except ImportError as exc:
            import warnings
            warnings.warn(
                f"[MAVTLoss] `lpips` package not installed ({exc}); "
                "perceptual loss disabled. Install with `pip install lpips`. "
                "Without LPIPS, image reconstructions will collapse to gray (L1-only failure mode).",
                RuntimeWarning,
                stacklevel=2,
            )
            self._available = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._available:
            return torch.tensor(0.0, device=pred.device)
        if pred.ndim == 5:
            B, C, N, H, W = pred.shape
            pred   = pred.permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)
            target = target.permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)
        return self._lpips(pred.clamp(-1, 1), target.clamp(-1, 1)).mean()


# --------------------------------------------------------------------------- #
#  InfoNCE (kept for optional text/retrieval alignment)                        #
# --------------------------------------------------------------------------- #

def infonce_loss(
    embeds_a: torch.Tensor,
    embeds_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    a = F.normalize(embeds_a, dim=-1)
    b = F.normalize(embeds_b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(logits.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# --------------------------------------------------------------------------- #
#  Cosine distillation                                                          #
# --------------------------------------------------------------------------- #

def cosine_distill_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """1 - mean cosine similarity. Teacher gradient is detached."""
    s = F.normalize(student, dim=-1)
    t = F.normalize(teacher.detach(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


# --------------------------------------------------------------------------- #
#  Slot diversity auxiliary                                                     #
# --------------------------------------------------------------------------- #

def slot_diversity_loss(slot_diversity_metric: torch.Tensor) -> torch.Tensor:
    return F.relu(slot_diversity_metric)


# --------------------------------------------------------------------------- #
#  Per-modality EMA weighter                                                    #
# --------------------------------------------------------------------------- #

class ModalityEMAWeighter(nn.Module):
    """Maintain a running EMA of L_recon per active modality.

    The returned weight is proportional to the modality EMA and normalised so
    the active-modality mean is 1. This gives harder modalities more gradient
    while keeping single-modality training unchanged.
    """

    def __init__(self, modalities=('image', 'video', 'threed'), momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.modalities = tuple(modalities)
        self.active_modalities = tuple(modalities)
        for m in modalities:
            self.register_buffer(f'ema_{m}', torch.tensor(1.0))

    def set_active_modalities(self, modalities: Sequence[str]) -> None:
        active = tuple(m for m in modalities if m in self.modalities)
        self.active_modalities = active or self.modalities

    def update(self, modality: str, loss_recon: torch.Tensor) -> None:
        attr = f'ema_{modality}'
        if hasattr(self, attr):
            ema = getattr(self, attr)
            setattr(self, attr, ema * self.momentum + loss_recon.detach() * (1 - self.momentum))

    def weight(self, modality: str) -> torch.Tensor:
        attr = f'ema_{modality}'
        if not hasattr(self, attr):
            return torch.tensor(1.0)
        active = self.active_modalities or self.modalities
        values = [getattr(self, f'ema_{m}') for m in active]
        normalizer = torch.stack(values).mean()
        return getattr(self, attr) / (normalizer + 1e-8)


# --------------------------------------------------------------------------- #
#  Combined MAVT loss                                                           #
# --------------------------------------------------------------------------- #

class MAVTLoss(nn.Module):
    """Matryoshka-aware multi-task loss.

    The model emits per-prefix dicts (``recon_per_prefix``, ``kl_per_prefix``,
    etc.). For every active prefix this loss applies the per-task weights,
    rescales KL by ``d_k / D_max`` to avoid posterior collapse on small
    prefixes, and combines them with optional Matryoshka coefficients
    ``alphas[d]``.
    """

    def __init__(
        self,
        w_l1: float    = 1.0,
        w_lpips: float = 0.1,
        w_kl: float    = 1.0,
        w_sem: float   = 0.5,
        w_aux: float   = 0.01,
        w_kl_spatial: float = 1e-4,
        use_lpips: bool = True,
        matryoshka_alphas: Optional[Mapping[int, float]] = None,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_lpips = w_lpips
        self.w_kl = w_kl
        self.w_sem = w_sem
        self.w_aux = w_aux
        self.w_kl_spatial = w_kl_spatial
        self.alphas: Dict[int, float] = (
            {int(k): float(v) for k, v in matryoshka_alphas.items()}
            if matryoshka_alphas else {}
        )

        self.lpips = LPIPSLoss() if use_lpips else None
        self.ema_weighter = ModalityEMAWeighter()

    def forward(
        self,
        target: torch.Tensor,
        modality: str,
        recon_per_prefix: Mapping[int, torch.Tensor],
        kl_per_prefix:    Mapping[int, torch.Tensor],
        sem_per_prefix:   Mapping[int, torch.Tensor],
        all_prefixes:     Sequence[int],
        slot_diversity:   torch.Tensor,
        teacher_embed:    Optional[torch.Tensor] = None,
        kl_spatial:       Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        device = target.device
        d_max = max(int(p) for p in all_prefixes)
        zero = torch.tensor(0.0, device=device)

        # Modality EMA scaling driven by recon at the largest prefix.
        if d_max in recon_per_prefix:
            l1_ref = F.l1_loss(recon_per_prefix[d_max], target)
            lp_ref = (self.lpips(recon_per_prefix[d_max], target)
                      if self.lpips is not None else zero)
            ref_recon = self.w_l1 * l1_ref + self.w_lpips * lp_ref
        else:
            ref_recon = zero
        mod_w = self.ema_weighter.weight(modality).to(device)
        if isinstance(ref_recon, torch.Tensor) and ref_recon.requires_grad:
            self.ema_weighter.update(modality, ref_recon)

        l_recon_sum = zero.clone()
        l_l1_sum    = zero.clone()
        l_lp_sum    = zero.clone()
        l_kl_sum    = zero.clone()
        l_sem_sum   = zero.clone()

        per_prefix_logs: Dict[str, torch.Tensor] = {}

        for d in all_prefixes:
            d = int(d)
            alpha = self.alphas.get(d, 1.0)

            if d in recon_per_prefix:
                pred = recon_per_prefix[d]
                l1_d = F.l1_loss(pred, target)
                lp_d = (self.lpips(pred, target)
                        if self.lpips is not None else zero)
                recon_d = self.w_l1 * l1_d + self.w_lpips * lp_d
                l_l1_sum    = l_l1_sum    + alpha * l1_d
                l_lp_sum    = l_lp_sum    + alpha * lp_d
                l_recon_sum = l_recon_sum + alpha * recon_d
                per_prefix_logs[f'loss_l1_{d}'] = l1_d.detach()
                if isinstance(lp_d, torch.Tensor):
                    per_prefix_logs[f'loss_lpips_{d}'] = lp_d.detach()

            beta_k = self.w_kl * (d / d_max)
            kl_d = kl_per_prefix[d]
            l_kl_sum = l_kl_sum + alpha * beta_k * kl_d
            per_prefix_logs[f'loss_kl_{d}'] = kl_d.detach()

            if self.w_sem > 0.0 and teacher_embed is not None and d in sem_per_prefix:
                sem_d = cosine_distill_loss(sem_per_prefix[d], teacher_embed)
                l_sem_sum = l_sem_sum + alpha * sem_d
                per_prefix_logs[f'loss_sem_{d}'] = sem_d.detach()

        l_div = slot_diversity_loss(slot_diversity)

        l_kl_spatial = (self.w_kl_spatial * kl_spatial
                        if kl_spatial is not None
                        else zero)

        total = (
            mod_w * l_recon_sum
            + l_kl_sum
            + self.w_sem * l_sem_sum
            + self.w_aux * l_div
            + l_kl_spatial
        )

        losses: Dict[str, torch.Tensor] = {
            'loss':       total,
            'loss_recon': l_recon_sum,
            'loss_recon_weighted': mod_w * l_recon_sum,
            'loss_l1':    l_l1_sum,
            'loss_lpips': l_lp_sum,
            'loss_kl':    l_kl_sum,
            'loss_sem':   l_sem_sum,
            'loss_sem_weighted': self.w_sem * l_sem_sum,
            'loss_div':   l_div,
            'modality_weight': mod_w.detach(),
            'loss_kl_spatial': l_kl_spatial.detach() if isinstance(l_kl_spatial, torch.Tensor) else l_kl_spatial,
        }
        losses.update(per_prefix_logs)
        return losses
