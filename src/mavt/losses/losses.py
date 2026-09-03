"""Stage 6: Loss functions for MAVT training.

L_total = w_recon·(w_l1·L1 + w_lpips·LPIPS) + w_kl·KL + w_clip·CLIP + w_aux·SlotDiv

Dynamic per-modality weighting: scale by 1/EMA(L_recon) so harder modalities
receive proportionally more gradient.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

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


def pool_teacher_tokens(hidden: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
    """Resample teacher patch tokens (B, Gt*Gt, D) onto the student grid (Hp, Wp) → (B, Hp*Wp, D).

    SigLIP2-SO400M at 384/16 gives a 24×24 map; the student works on 16×16 at 256 px.
    Bilinear (antialiased) resampling of the token map keeps each student token aligned
    with the teacher's features at the same image location.
    """
    B, N, D = hidden.shape
    g = int(round(N ** 0.5))
    if g * g != N:
        raise ValueError(f"teacher tokens must form a square grid, got N={N}")
    Hp, Wp = grid
    if (g, g) == (Hp, Wp):
        return hidden
    m = hidden.transpose(1, 2).reshape(B, D, g, g)
    m = F.interpolate(m.float(), size=(Hp, Wp), mode="bilinear", align_corners=False, antialias=True)
    return m.reshape(B, D, Hp * Wp).transpose(1, 2).to(hidden.dtype)


def dense_distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Per-token cosine distillation (REPA-style): 1 - mean_{b,n} cos(student[b,n], teacher[b,n]).

    student : (B, N, D) backbone tokens (after the learnable dense projection)
    teacher : (B, N, D) frozen teacher patch tokens resampled to the same grid (detached)
    Supervises every token of the trunk directly — N× more signal per image than the
    pooled cosine target, which the 2026-09-02 probe showed cannot make the trunk semantic.
    """
    if student.shape != teacher.shape:
        raise ValueError(f"dense distill shape mismatch: student {tuple(student.shape)} vs teacher {tuple(teacher.shape)}")
    cos = F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1)
    return 1.0 - cos.mean()


def cosine_distill_loss(
    student: torch.Tensor,   # (B, D)
    teacher: torch.Tensor,   # (B, D) — frozen vision teacher (e.g. SigLIP2)
    center: bool = False,
) -> torch.Tensor:
    """1 - mean cosine similarity. Teacher is detached (no gradient flows back).

    center=True subtracts each side's batch mean first, so the loss rewards
    matching the per-sample (discriminative) deviation rather than the teacher's
    shared mean direction — which plain cosine would otherwise fit first and
    collapse onto (pair_cos 0.95 in the pilot).
    """
    teacher = teacher.detach()
    if center and student.shape[0] > 1:
        student = student - student.mean(0, keepdim=True)
        teacher = teacher - teacher.mean(0, keepdim=True)
    s = F.normalize(student, dim=-1)
    t = F.normalize(teacher, dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


def vicreg_loss(e: torch.Tensor, std_target: float = 1.0, w_var: float = 1.0,
                w_cov: float = 0.04) -> torch.Tensor:
    """Variance + covariance regulariser (VICReg) on a (B, D) embedding.

    var: hinge pushing every dimension's std above `std_target`;
    cov: off-diagonal covariance pushed to zero (de-correlate dimensions).
    Prevents the pooled embedding from collapsing while distillation pulls it.
    """
    if e.shape[0] < 2:
        return e.new_zeros(())
    e = e.float()
    std = torch.sqrt(e.var(dim=0, unbiased=False) + 1e-4)
    var_loss = F.relu(std_target - std).mean()
    c = e - e.mean(0, keepdim=True)
    cov = (c.T @ c) / (e.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / e.shape[1]
    return w_var * var_loss + w_cov * cov_loss


# --------------------------------------------------------------------------- #
#  Slot diversity auxiliary loss                                                #
# --------------------------------------------------------------------------- #

def slot_diversity_loss(slot_diversity_metric: torch.Tensor) -> torch.Tensor:
    """Penalise high cosine similarity between content slots (collapse prevention).

    The metric is already mean pairwise cosine sim; penalise if > 0.
    """
    return F.relu(slot_diversity_metric)


# --------------------------------------------------------------------------- #
#  Temporal consistency loss (video)                                            #
# --------------------------------------------------------------------------- #

def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match frame-to-frame motion between pred and target.

    pred, target: (B, 3, T, H, W)
    Computes L1 between temporal gradients so the model learns to reproduce
    motion patterns, not just per-frame appearance.
    """
    pred_diff   = pred[:, :, 1:] - pred[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]
    return F.l1_loss(pred_diff, target_diff)


# --------------------------------------------------------------------------- #
#  Dynamic per-modality loss weighting                                          #
# --------------------------------------------------------------------------- #

class ModalityEMAWeighter(nn.Module):
    """Maintains a running EMA of L_recon per modality for proportional scaling.

    weight(modality) = ema_modality / mean(ema_active). Modalities with HIGHER
    recon loss get LARGER weight so cross-stage transfer (e.g. image already
    converged from stage 1, video starts random in stage 2) doesn't starve the
    untrained branch of gradient. Mean weight = 1.0, so total loss magnitude is
    preserved.

    All three EMA buffers are always registered regardless of active_modalities
    so that checkpoints load cleanly across stages (stage1→stage2→stage3 resume).
    active_modalities controls only which set is used for weight normalization
    and can be updated at runtime (e.g. synced from DataModule in setup()).
    """

    _ALL_MODALITIES = ('image', 'video', 'threed')

    def __init__(self, modalities=('image', 'video', 'threed'), momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.active_modalities = tuple(modalities)  # plain attr — mutable, not a buffer
        # Always register all three so stage1 ckpt loads into stage2 without missing keys
        for m in self._ALL_MODALITIES:
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

        # Proportional: high-loss modality gets boosted weight. Normalize only
        # over active modalities so inactive ones (e.g. threed in stage2,
        # frozen at the init value 1.0) don't distort the mean.
        vals = [getattr(self, f'ema_{m}') for m in self.active_modalities]
        val_stack = torch.stack(vals)
        normalizer = val_stack.sum() / len(self.active_modalities)
        return (getattr(self, attr) + 1e-8) / (normalizer + 1e-8)


# --------------------------------------------------------------------------- #
#  Combined MAVT loss                                                           #
# --------------------------------------------------------------------------- #

class MAVTLoss(nn.Module):
    """Combines all losses with configurable weights.

    Default training signal for unified tokenizer:
        recon (L1 + LPIPS) + KL + vision-vision distillation + slot diversity.

    InfoNCE/text branch is retained but disabled by default (use_clip=False).

    Note on w_kl: loss_kl arrives pre-scaled by VAEHead.kl_weight already.
    Keep w_kl=1.0 (default) so no double-scaling occurs. Setting both
    w_kl=1e-4 and kl_weight=1e-4 makes KL ~1e-8 of total loss → posterior collapse.
    """

    def __init__(
        self,
        w_l1: float   = 1.0,
        w_lpips: float = 0.1,
        w_kl: float   = 1.0,    # passthrough — KL already scaled by VAEHead.kl_weight
        w_clip: float = 0.0,    # InfoNCE(visual, text) — off by default
        w_sem: float  = 0.5,    # cosine distill from frozen vision teacher
        w_aux: float  = 0.01,
        w_temp: float = 0.0,    # temporal consistency (video only); 0 = disabled
        w_vic: float  = 0.0,    # VICReg anti-collapse on the pooled semantic embedding
        w_dense: float = 0.0,   # per-token cosine distillation of backbone tokens to teacher patch tokens
        distill_center: bool = False,  # centered cosine distillation
        use_lpips: bool = True,
        use_clip: bool  = False,  # requires text embeddings
        active_modalities: tuple = ('image', 'video', 'threed'),
    ):
        super().__init__()
        self.w_l1   = w_l1
        self.w_lpips = w_lpips
        self.w_kl   = w_kl
        self.w_clip = w_clip
        self.w_sem  = w_sem
        self.w_aux  = w_aux
        self.w_temp = w_temp
        self.w_vic = w_vic
        self.w_dense = w_dense
        self.distill_center = distill_center
        self.use_clip = use_clip

        self.lpips = LPIPSLoss() if use_lpips else None
        self.ema_weighter = ModalityEMAWeighter(modalities=active_modalities)

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
        dense_student: Optional[torch.Tensor] = None,   # (B, N, D) backbone tokens (projected)
        dense_teacher: Optional[torch.Tensor] = None,   # (B, N, D) teacher patch tokens on the student grid
    ) -> Dict[str, torch.Tensor]:

        # Reconstruction
        if pred.shape != target.shape:
            raise ValueError(
                f"MAVTLoss: pred/target shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)} "
                "(modality={modality}). Decoders must return the full target shape — no silent "
                "truncation/alignment here."
            )
        l1   = F.l1_loss(pred, target)
        lpips_val = self.lpips(pred, target) if self.lpips is not None else torch.tensor(0.0, device=pred.device)
        l_recon = self.w_l1 * l1 + self.w_lpips * lpips_val

        # Modality inverse-EMA scaling
        mod_w = self.ema_weighter.weight(modality).to(pred.device)
        if self.training:  # don't let validation batches poison training weights
            self.ema_weighter.update(modality, l_recon)

        # CLIP contrastive (legacy text path — off by default)
        l_clip = torch.tensor(0.0, device=pred.device)
        if self.use_clip and semantic_embed is not None and text_embed is not None:
            l_clip = infonce_loss(semantic_embed, text_embed)

        # Vision-vision distillation (default semantic supervision)
        l_sem = torch.tensor(0.0, device=pred.device)
        if self.w_sem > 0.0 and semantic_embed is not None and teacher_embed is not None:
            l_sem = cosine_distill_loss(semantic_embed, teacher_embed, center=self.distill_center)

        # VICReg anti-collapse on the pooled embedding
        l_vic = torch.tensor(0.0, device=pred.device)
        if self.w_vic > 0.0 and semantic_embed is not None:
            l_vic = vicreg_loss(semantic_embed)

        l_dense = torch.tensor(0.0, device=pred.device)
        if self.w_dense > 0.0 and dense_student is not None and dense_teacher is not None:
            l_dense = dense_distill_loss(dense_student, dense_teacher)

        # Slot diversity (auxiliary)
        l_div = slot_diversity_loss(slot_diversity)

        # Temporal consistency (video only — 5-D pred with T > 1)
        l_temp = torch.tensor(0.0, device=pred.device)
        if self.w_temp > 0.0 and pred.ndim == 5 and pred.shape[2] > 1:
            l_temp = temporal_consistency_loss(pred, target)

        total = (
            mod_w * l_recon
            + self.w_kl * loss_kl
            + self.w_clip * l_clip
            + self.w_sem * l_sem
            + self.w_aux * l_div
            + self.w_temp * l_temp
            + self.w_vic * l_vic
            + self.w_dense * l_dense
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
            'loss_temp':  l_temp,
            'loss_vic':   l_vic,
            'loss_dense': l_dense,
        }
