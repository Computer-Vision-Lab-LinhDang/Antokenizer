"""PyTorch Lightning Module for MAVT training.

Supports 3-stage curriculum via `training_stage` parameter:
  1 — image only,  SigLIP2 fully frozen,  LR = 1e-4
  2 — +video,      SigLIP2 last 4 unfrozen, LR = 5e-5
  3 — +3D,         SigLIP2 fully unfrozen, LR = 2e-5

To move between stages: start training with the next stage config and pass
`--ckpt_path <prev_stage_checkpoint>` to LightningCLI.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.utilities import grad_norm

from mavt.model.mavt import MAVT
from mavt.losses.losses import MAVTLoss


_STAGE_LR = {1: 1e-4, 2: 5e-5, 3: 2e-5}
_STAGE_SIGLIP2_FROZEN_BLOCKS = {1: 10, 2: 6, 3: 0}  # number of transformer blocks frozen


class MAVTLightningModule(L.LightningModule):
    """Lightning module for end-to-end MAVT training."""

    def __init__(
        self,
        # Model
        embed_dim: int = 1152,
        num_heads: int = 16,
        num_blocks: int = 12,
        patch_size: int = 16,
        t_patch: int = 2,
        latent_dim: int = 32,
        kl_weight: float = 1e-4,
        semantic_dim: int = 768,
        dec_dim: int = 768,
        num_dec_attn_blocks: int = 4,
        r_s: int = 2,
        r_t: int = 1,
        use_gradient_checkpointing: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # Loss
        w_l1: float = 1.0,
        w_lpips: float = 0.1,
        w_clip: float = 0.1,
        w_aux: float = 0.01,
        use_lpips: bool = True,
        use_clip: bool = False,
        # Curriculum
        training_stage: int = 1,
        siglip2_model_name: str = "google/siglip2-base-patch16-224",
        init_siglip2: bool = False,
        # Optimiser
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        total_steps: int = 200_000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MAVT(
            embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks,
            patch_size=patch_size, t_patch=t_patch,
            latent_dim=latent_dim, kl_weight=kl_weight,
            semantic_dim=semantic_dim, dec_dim=dec_dim,
            num_dec_attn_blocks=num_dec_attn_blocks, r_s=r_s, r_t=r_t,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

        self.loss_fn = MAVTLoss(
            w_l1=w_l1, w_lpips=w_lpips, w_kl=1.0,
            w_clip=w_clip, w_aux=w_aux,
            use_lpips=use_lpips, use_clip=use_clip,
        )

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def setup(self, stage: str) -> None:
        hp = self.hparams
        if hp.init_siglip2 and stage == 'fit':
            frozen = _STAGE_SIGLIP2_FROZEN_BLOCKS[hp.training_stage]
            self.model.load_siglip2_weights(hp.siglip2_model_name, frozen)

    # ------------------------------------------------------------------ #
    #  Training step                                                       #
    # ------------------------------------------------------------------ #

    def _step(self, batch: Dict, log_prefix: str) -> torch.Tensor:
        x       = batch['data']
        modality = batch['modality']

        out = self.model(x, modality, decode=True)

        # For video: decoder reconstructs in patch-grid temporal space (Tp = T//t_patch).
        # Target must match. Use first frame of each temporal patch group.
        if modality == 'video':
            t_patch = self.hparams.t_patch
            target = x[:, :, ::t_patch]  # (B, 3, Tp, H, W)
        else:
            target = x  # image and threed pass through as-is

        losses = self.loss_fn(
            pred=out.reconstruction,
            target=target,
            loss_kl=out.loss_kl,
            slot_diversity=out.cd_metrics['slot_diversity'],
            modality=modality,
        )

        # Logging
        for k, v in losses.items():
            self.log(f'{log_prefix}/{k}', v, on_step=True, on_epoch=True,
                     prog_bar=(k == 'loss'), sync_dist=True)
        for k, v in out.cd_metrics.items():
            self.log(f'{log_prefix}/cd_{k}', v, on_step=False, on_epoch=True,
                     sync_dist=True)
        self.log(f'{log_prefix}/modality_{modality}', 1.0,
                 on_step=False, on_epoch=True, sync_dist=True)

        return losses['loss']

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        with torch.no_grad():
            self._step(batch, 'val')
            # Log sample reconstructions to wandb/tensorboard every N steps
            if batch_idx == 0:
                self._log_images(batch)

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def _log_images(self, batch: Dict, n: int = 4) -> None:
        try:
            x       = batch['data'][:n]
            modality = batch['modality']
            out = self.model(x, modality, decode=True)

            if modality == 'image':
                grid_in  = _to_grid(x)
                grid_out = _to_grid(out.reconstruction)
            elif modality == 'video':
                # Log first frame of each clip
                grid_in  = _to_grid(x[:, :, 0])
                grid_out = _to_grid(out.reconstruction[:, :, 0])
            elif modality == 'threed':
                # Log XY plane (plane index 0)
                grid_in  = _to_grid(x[:, 0])
                grid_out = _to_grid(out.reconstruction[:, 0])
            else:
                return

            loggers = self.loggers if isinstance(self.loggers, (list, tuple)) else [self.loggers]
            for logger in loggers:
                if hasattr(logger, 'log_image'):
                    logger.log_image(key=f'val/{modality}_input',  images=[grid_in])
                    logger.log_image(key=f'val/{modality}_recon',  images=[grid_out])
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    #  Optimiser                                                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        hp = self.hparams
        lr = _STAGE_LR[hp.training_stage]

        # Separate RGAT params for potential different LR (currently same LR)
        rgat_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'rgat' in name.lower() or 'rgat4d' in name.lower():
                rgat_params.append(p)
            else:
                other_params.append(p)

        param_groups = [{'params': other_params, 'lr': lr}]
        if rgat_params:
            param_groups.append({'params': rgat_params, 'lr': lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

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


# --------------------------------------------------------------------------- #
#  Visualisation helper                                                         #
# --------------------------------------------------------------------------- #

def _to_grid(x: torch.Tensor, nrow: int = 4) -> Any:
    """Convert (B, 3, H, W) tensor to a PIL Image grid for logging."""
    try:
        from torchvision.utils import make_grid
        from PIL import Image
        import numpy as np
        x = x.detach().cpu().float().clamp(-1, 1)
        x = (x + 1) / 2                         # [0, 1]
        grid = make_grid(x, nrow=nrow, normalize=False)
        arr = (grid.permute(1, 2, 0).numpy() * 255).astype('uint8')
        return Image.fromarray(arr)
    except Exception:  # noqa: BLE001
        return None
