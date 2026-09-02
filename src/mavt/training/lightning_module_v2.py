"""PyTorch Lightning Module for MAVT v2 training.

MAVT v2 with Two-Axis Decomposition:
- Semantic loss on z_inv ONLY
- Reconstruction loss on z_var ONLY
- No gradient conflict between objectives
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities import grad_norm

from mavt.model.mavt_v2 import MAVT


# ============================================================================
# Loss Functions
# ============================================================================

class MAVTv2Loss(nn.Module):
    """Loss function for MAVT v2.

    Separated losses for semantic and reconstruction:
    - Semantic loss: ON z_inv ONLY
    - Reconstruction loss: ON z_var ONLY
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_kl: float = 1.0,
        w_sem: float = 0.1,
        w_aux: float = 0.01,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_kl = w_kl
        self.w_sem = w_sem
        self.w_aux = w_aux

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        loss_kl: torch.Tensor,
        semantic: torch.Tensor,
        teacher_emb: Optional[torch.Tensor] = None,
        two_axis_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            recon: (B, C, H, W) reconstruction
            target: (B, C, H, W) ground truth
            loss_kl: scalar KL loss from VAE
            semantic: (B, D) semantic features from z_inv
            teacher_emb: (B, D) optional frozen teacher embeddings
            two_axis_metrics: dict with energy_ratio, gini

        Returns:
            dict of losses
        """
        losses = {}

        # Reconstruction loss (L1)
        loss_l1 = F.l1_loss(recon, target)
        losses['loss_l1'] = loss_l1

        # KL loss
        losses['loss_kl'] = loss_kl

        # Semantic loss
        if teacher_emb is not None:
            # Cosine distance with teacher
            semantic_norm = F.normalize(semantic, dim=-1)
            teacher_norm = F.normalize(teacher_emb, dim=-1)
            loss_sem = 1 - (semantic_norm * teacher_norm).sum(dim=-1).mean()
        else:
            loss_sem = torch.tensor(0.0, device=recon.device)
        losses['loss_sem'] = loss_sem

        # Two-axis monitoring
        if two_axis_metrics:
            for k, v in two_axis_metrics.items():
                if isinstance(v, float):
                    losses[f'two_axis_{k}'] = torch.tensor(v, device=recon.device)

        # Total loss
        loss_total = (
            self.w_l1 * loss_l1 +
            self.w_kl * loss_kl +
            self.w_sem * loss_sem
        )
        losses['loss'] = loss_total

        return losses


# ============================================================================
# Lightning Module
# ============================================================================

class MAVTv2LightningModule(L.LightningModule):
    """Lightning module for MAVT v2 training."""

    def __init__(
        self,
        # Model
        embed_dim: int = 768,
        num_heads: int = 12,
        num_blocks: int = 12,
        patch_size: int = 16,
        t_patch: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # Scale Pyramid
        num_scales: int = 4,
        # VAE
        latent_dim: int = 32,
        kl_weight: float = 1e-4,
        # Semantic
        semantic_dim: int = 768,
        # Decoder
        dec_dim: int = 512,
        num_dec_attn_blocks: int = 4,
        # Loss
        w_l1: float = 1.0,
        w_kl: float = 1.0,
        w_sem: float = 0.1,
        w_aux: float = 0.01,
        # Semantic teacher
        use_semantic_distill: bool = False,
        siglip2_model_name: str = "google/siglip2-base-patch16-224",
        init_siglip2: bool = True,
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        total_steps: int = 200_000,
        # Init
        init_from_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = MAVT(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            patch_size=patch_size,
            t_patch=t_patch,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_scales=num_scales,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
            semantic_dim=semantic_dim,
            dec_dim=dec_dim,
            num_dec_attn_blocks=num_dec_attn_blocks,
        )

        # Loss
        self.loss_fn = MAVTv2Loss(
            w_l1=w_l1,
            w_kl=w_kl,
            w_sem=w_sem,
            w_aux=w_aux,
        )

        # Teacher
        self.semantic_teacher: Optional[nn.Module] = None
        self._teacher_image_size: int = 224

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def setup(self, stage: str) -> None:
        if stage != 'fit':
            return

        # Load SigLIP2 teacher if needed
        if self.hparams.init_siglip2 or self.hparams.use_semantic_distill:
            self._load_semantic_teacher(self.hparams.siglip2_model_name)

        # Load checkpoint if specified
        if self.hparams.init_from_ckpt:
            self._load_weights_from_ckpt(self.hparams.init_from_ckpt)

    def _load_weights_from_ckpt(self, path: str) -> None:
        """Load weights from checkpoint."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[init_from_ckpt] loaded {path}")
        print(f"[init_from_ckpt] missing: {len(missing)}, unexpected: {len(unexpected)}")

    def _load_semantic_teacher(self, model_name: str) -> None:
        """Load frozen SigLIP2 as teacher."""
        try:
            from transformers import AutoModel
            siglip = AutoModel.from_pretrained(model_name)
            teacher = siglip.vision_model
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()
            self.semantic_teacher = teacher
            try:
                self._teacher_image_size = int(siglip.config.vision_config.image_size)
            except AttributeError:
                self._teacher_image_size = 224
            print(f"[lightning_module] loaded semantic teacher: {model_name}")
        except Exception as exc:
            print(f"[lightning_module] failed to load teacher ({exc}); disabled")
            self.semantic_teacher = None

    def train(self, mode: bool = True):
        """Keep teacher in eval mode."""
        super().train(mode)
        if self.semantic_teacher is not None:
            self.semantic_teacher.eval()
        return self

    # ------------------------------------------------------------------ #
    #  Training Step                                                       #
    # ------------------------------------------------------------------ #

    def _step(self, batch: Dict, log_prefix: str) -> torch.Tensor:
        x = batch['data']
        modality = batch['modality']

        # Forward
        out = self.model(x, modality, decode=True)

        # Target
        if modality == 'video':
            t_patch = self.hparams.t_patch
            target = x[:, :, ::t_patch]
        else:
            target = x

        # Teacher embedding
        teacher_emb = None
        if self.semantic_teacher is not None:
            with torch.no_grad():
                proxy = self._make_teacher_input(x, modality, self._teacher_image_size)
                teacher_emb = self.semantic_teacher(pixel_values=proxy).pooler_output

        # Compute losses
        losses = self.loss_fn(
            recon=out.reconstruction,
            target=target,
            loss_kl=out.loss_kl,
            semantic=out.semantic,
            teacher_emb=teacher_emb,
            two_axis_metrics=out.two_axis_metrics,
        )

        # Logging
        for k, v in losses.items():
            self.log(f'{log_prefix}/{k}', v, on_step=True, on_epoch=True,
                     prog_bar=(k == 'loss'), sync_dist=True)

        # Per-modality
        for k in ('loss', 'loss_l1', 'loss_kl', 'loss_sem'):
            if k in losses:
                self.log(f'{log_prefix}/{k}_{modality}', losses[k],
                        on_step=True, on_epoch=True, sync_dist=True)

        return losses['loss']

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        with torch.no_grad():
            self._step(batch, 'val')
            if batch_idx == 0:
                self._log_images(batch)

    # ------------------------------------------------------------------ #
    #  Teacher Input                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_teacher_input(x: torch.Tensor, modality: str,
                           target_size: int) -> torch.Tensor:
        """Project multi-modal input to single image for teacher."""
        if modality == 'image':
            proxy = x
        elif modality == 'video':
            T = x.shape[2]
            proxy = x[:, :, T // 2]  # Middle frame
        elif modality == 'threed':
            proxy = x[:, 0]  # XY plane
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if proxy.shape[-1] != target_size or proxy.shape[-2] != target_size:
            proxy = F.interpolate(proxy, size=(target_size, target_size),
                                 mode='bilinear', align_corners=False)
        return proxy

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #

    def _log_images(self, batch: Dict, n: int = 4) -> None:
        """Log sample reconstructions."""
        try:
            x = batch['data'][:n]
            modality = batch['modality']
            out = self.model(x, modality, decode=True)

            if modality == 'image':
                grid_in = self._make_grid(x)
                grid_out = self._make_grid(out.reconstruction)
            elif modality == 'video':
                B, C, T, H, W = x.shape
                t_idx = torch.linspace(0, T - 1, 4).long()
                in_strip = x[:, :, t_idx].permute(0, 2, 1, 3, 4).reshape(B * 4, C, H, W)
                out_strip = out.reconstruction[:, :, t_idx].permute(0, 2, 1, 3, 4).reshape(B * 4, C, H, W)
                grid_in = self._make_grid(in_strip, nrow=4)
                grid_out = self._make_grid(out_strip, nrow=4)
            elif modality == 'threed':
                grid_in = self._make_grid(x[:, 0])
                grid_out = self._make_grid(out.reconstruction[:, 0])
            else:
                return

            loggers = self.loggers if isinstance(self.loggers, (list, tuple)) else [self.loggers]
            for logger in loggers:
                if hasattr(logger, 'log_image'):
                    logger.log_image(key=f'val/{modality}_input', images=[grid_in])
                    logger.log_image(key=f'val/{modality}_recon', images=[grid_out])
        except Exception:
            pass

    @staticmethod
    def _make_grid(x: torch.Tensor, nrow: int = 4):
        """Convert tensor to PIL Image grid."""
        try:
            from torchvision.utils import make_grid
            from PIL import Image
            import numpy as np
            x = x.detach().cpu().float().clamp(-1, 1)
            x = (x + 1) / 2
            grid = make_grid(x, nrow=nrow, normalize=False)
            arr = (grid.permute(1, 2, 0).numpy() * 255).astype('uint8')
            return Image.fromarray(arr)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Optimizer                                                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        hp = self.hparams

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Linear warmup + cosine decay
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return step / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        self.clip_gradients(optimizer, gradient_clip_val=self.hparams.grad_clip,
                           gradient_clip_algorithm='norm')


# ============================================================================
# CLI Config
# ============================================================================

from dataclasses import dataclass, field


@dataclass
class MAVTv2Config:
    """Configuration for MAVT v2."""

    # Model
    embed_dim: int = 768
    num_heads: int = 12
    num_blocks: int = 12
    patch_size: int = 16
    t_patch: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Scale Pyramid
    num_scales: int = 4

    # VAE
    latent_dim: int = 32
    kl_weight: float = 1e-4

    # Semantic
    semantic_dim: int = 768

    # Decoder
    dec_dim: int = 512
    num_dec_attn_blocks: int = 4

    # Loss
    w_l1: float = 1.0
    w_kl: float = 1.0
    w_sem: float = 0.1
    w_aux: float = 0.01

    # Training
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    total_steps: int = 200_000

    # Teacher
    use_semantic_distill: bool = False
    siglip2_model_name: str = "google/siglip2-base-patch16-224"
    init_siglip2: bool = True

    # Init
    init_from_ckpt: Optional[str] = None
