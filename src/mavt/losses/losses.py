"""Stage 6: Loss functions for MAVT training.

L_total = w_recon·(w_l1·L1 + w_lpips·LPIPS) + w_kl·KL + w_clip·CLIP + w_aux·SlotDiv

Dynamic per-modality weighting: scale by 1/EMA(L_recon) so harder modalities
receive proportionally more gradient.
"""

from __future__ import annotations
from typing import Dict, Optional

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
        """pred, target: (B, 3, H, W) or (B, 3, N, H, W) in [-1, 1] or [0, 1].

        5-D inputs (video frames / triplane planes) are merged into the batch
        dimension so LPIPS always sees (B', 3, H, W).
        """
        if not self._available:
            return torch.tensor(0.0, device=pred.device)
        if pred.ndim == 5:
            # (B, 3, N, H, W) → (B*N, 3, H, W)
            B, C, N, H, W = pred.shape
            pred   = pred.permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)
            target = target.permute(0, 2, 1, 3, 4).reshape(B * N, C, H, W)
        pred_n   = pred.clamp(-1, 1)
        target_n = target.clamp(-1, 1)
        return self._lpips(pred_n, target_n).mean()


# --------------------------------------------------------------------------- #
#  InfoNCE / SigLIP-style contrastive loss (kept for optional text alignment)  #
# --------------------------------------------------------------------------- #

def infonce_loss(
    embeds_a: torch.Tensor,   # (B, D) — visual
    embeds_b: torch.Tensor,   # (B, D) — text or another view
    temperature: float = 0.07,
) -> torch.Tensor:
    a = F.normalize(embeds_a, dim=-1)
    b = F.normalize(embeds_b, dim=-1)
    logits = a @ b.T / temperature                   # (B, B)
    labels = torch.arange(logits.size(0), device=a.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss


# --------------------------------------------------------------------------- #
#  Vision-vision cosine distillation loss                                       #
# --------------------------------------------------------------------------- #

def cosine_distill_loss(
    student: torch.Tensor,   # (B, D)
    teacher: torch.Tensor,   # (B, D) — frozen vision teacher (e.g. SigLIP2)
) -> torch.Tensor:
    """1 - mean cosine similarity. Teacher is detached (no gradient flows back)."""
    s = F.normalize(student, dim=-1)
    t = F.normalize(teacher.detach(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


# --------------------------------------------------------------------------- #
#  Slot diversity auxiliary loss                                                #
# --------------------------------------------------------------------------- #

def slot_diversity_loss(slot_diversity_metric: torch.Tensor) -> torch.Tensor:
    """Penalise high cosine similarity between content slots (collapse prevention).

    The metric is already mean pairwise cosine sim; penalise if > 0.
    """
    return F.relu(slot_diversity_metric)


# --------------------------------------------------------------------------- #
#  Dynamic per-modality loss weighting                                          #
# --------------------------------------------------------------------------- #

class ModalityEMAWeighter(nn.Module):
    """Maintains a running EMA of L_recon per modality for inverse scaling.

    Normalizes the *coefficients* across modalities to sum to ``num_modalities``
    (i.e. mean weight = 1.0), instead of normalizing each modality's loss to 1.
    This keeps relative balancing between modalities while preserving the
    absolute magnitude of l_recon — so a decreasing l_recon shows up in l_total.
    """

    def __init__(self, modalities=('image', 'video', 'threed'), momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.modalities = tuple(modalities)
        for m in modalities:
            self.register_buffer(f'ema_{m}', torch.tensor(1.0))

    def update(self, modality: str, loss_recon: torch.Tensor) -> None:
        attr = f'ema_{modality}'
        if hasattr(self, attr):
            ema = getattr(self, attr)
            setattr(self, attr, ema * self.momentum + loss_recon.detach() * (1 - self.momentum))

    def weight(self, modality: str) -> torch.Tensor:
        attr = f'ema_{modality}'
        if not hasattr(self, attr):
            return torch.tensor(1.0)

        inv = [1.0 / (getattr(self, f'ema_{m}') + 1e-8) for m in self.modalities]
        inv_stack = torch.stack(inv)
        normalizer = inv_stack.sum() / len(self.modalities)  # mean(inv)
        return (1.0 / (getattr(self, attr) + 1e-8)) / (normalizer + 1e-8)


# --------------------------------------------------------------------------- #
#  Combined MAVT loss                                                           #
# --------------------------------------------------------------------------- #

class MAVTLoss(nn.Module):
    """Combines all losses with configurable weights.

    Default training signal for unified tokenizer:
        recon (L1 + LPIPS) + KL + vision-vision distillation + slot diversity.

    InfoNCE/text branch is retained but disabled by default (use_clip=False).
    """

    def __init__(
        self,
        w_l1: float   = 1.0,
        w_lpips: float = 0.1,
        w_kl: float   = 1e-4,   # already baked into VAEHead; set 1.0 here
        w_clip: float = 0.0,    # InfoNCE(visual, text) — off by default
        w_sem: float  = 0.5,    # cosine distill from frozen vision teacher
        w_aux: float  = 0.01,
        use_lpips: bool = True,
        use_clip: bool  = False,  # requires text embeddings
    ):
        super().__init__()
        self.w_l1   = w_l1
        self.w_lpips = w_lpips
        self.w_kl   = w_kl
        self.w_clip = w_clip
        self.w_sem  = w_sem
        self.w_aux  = w_aux
        self.use_clip = use_clip

        self.lpips = LPIPSLoss() if use_lpips else None
        self.ema_weighter = ModalityEMAWeighter()

    def forward(
        self,
        pred: torch.Tensor,           # reconstructed, pixel-space
        target: torch.Tensor,         # ground-truth, pixel-space
        loss_kl: torch.Tensor,
        slot_diversity: torch.Tensor,
        modality: str,
        semantic_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        teacher_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # Reconstruction
        l1   = F.l1_loss(pred, target)
        lpips_val = self.lpips(pred, target) if self.lpips is not None else torch.tensor(0.0, device=pred.device)
        l_recon = self.w_l1 * l1 + self.w_lpips * lpips_val

        # Modality inverse-EMA scaling
        mod_w = self.ema_weighter.weight(modality).to(pred.device)
        self.ema_weighter.update(modality, l_recon)

        # CLIP contrastive (legacy text path — off by default)
        l_clip = torch.tensor(0.0, device=pred.device)
        if self.use_clip and semantic_embed is not None and text_embed is not None:
            l_clip = infonce_loss(semantic_embed, text_embed)

        # Vision-vision distillation (default semantic supervision)
        l_sem = torch.tensor(0.0, device=pred.device)
        if self.w_sem > 0.0 and semantic_embed is not None and teacher_embed is not None:
            l_sem = cosine_distill_loss(semantic_embed, teacher_embed)

        # Slot diversity (auxiliary)
        l_div = slot_diversity_loss(slot_diversity)

        total = (
            mod_w * l_recon
            + self.w_kl * loss_kl
            + self.w_clip * l_clip
            + self.w_sem * l_sem
            + self.w_aux * l_div
        )

        return {
            'loss':       total,
            'loss_recon': l_recon,
            'loss_l1':    l1,
            'loss_lpips': lpips_val,
            'loss_kl':    loss_kl,
            'loss_clip':  l_clip,
            'loss_sem':   l_sem,
            'loss_div':   l_div,
        }
