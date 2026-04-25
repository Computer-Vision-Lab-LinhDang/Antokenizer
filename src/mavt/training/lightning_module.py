"""PyTorch Lightning Module for MAVT (Matryoshka edition).

Supports a 3-stage curriculum via ``training_stage``:
  1 — image only,  SigLIP2 fully frozen,  LR = 1e-4
  2 — +video,      SigLIP2 last 4 unfrozen, LR = 5e-5
  3 — +3D,         SigLIP2 fully unfrozen, LR = 2e-5

Matryoshka-specific knobs:
  matryoshka_dims      — nested prefix widths exposed by the head.
  matryoshka_alphas    — optional dict {d: α_d} per-prefix coefficients.
  recon_prefix_policy  — 'all' (default) decodes every prefix every step;
                         'random' samples ONE prefix per step (MRL-E from
                         Kusupati et al., reduces decoder cost K×);
                         'largest' decodes only the full prefix.
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from mavt.model.mavt import MAVT
from mavt.losses.losses import MAVTLoss


_STAGE_LR = {1: 1e-4, 2: 5e-5, 3: 2e-5}
_STAGE_SIGLIP2_FROZEN_BLOCKS = {1: 10, 2: 6, 3: 0}
_STAGE_W_SEM = {1: 0.5, 2: 0.3, 3: 0.2}


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
        kl_weight: float = 1e-4,           # legacy: KL weighting now lives in MAVTLoss
        semantic_dim: int = 768,
        num_classes: Optional[int] = None,
        matryoshka_dims: Optional[List[int]] = None,
        matryoshka_alphas: Optional[Dict[int, float]] = None,
        recon_prefix_policy: str = 'all',  # {'all', 'random', 'largest'}
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
        w_kl: float = 1.0,
        w_sem: float = 0.0,
        w_aux: float = 0.01,
        use_lpips: bool = True,
        # Curriculum
        training_stage: int = 1,
        siglip2_model_name: str = "google/siglip2-base-patch16-224",
        init_siglip2: bool = True,
        use_semantic_distill: bool = False,
        # Optimiser
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        total_steps: int = 200_000,
    ):
        super().__init__()
        self.save_hyperparameters()
        del kl_weight  # silences "unused var" lint; the value is preserved in self.hparams

        if recon_prefix_policy not in ('all', 'random', 'largest'):
            raise ValueError(
                f"recon_prefix_policy must be one of {{'all','random','largest'}}, "
                f"got {recon_prefix_policy!r}"
            )

        self.model = MAVT(
            embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks,
            patch_size=patch_size, t_patch=t_patch,
            matryoshka_dims=matryoshka_dims,
            latent_dim=latent_dim,
            semantic_dim=semantic_dim, num_classes=num_classes,
            dec_dim=dec_dim, num_dec_attn_blocks=num_dec_attn_blocks,
            r_s=r_s, r_t=r_t,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )

        self.loss_fn = MAVTLoss(
            w_l1=w_l1, w_lpips=w_lpips, w_kl=w_kl,
            w_sem=w_sem, w_aux=w_aux,
            use_lpips=use_lpips,
            matryoshka_alphas=matryoshka_alphas,
        )

        # Frozen vision teacher (loaded lazily in setup() to keep __init__ light)
        self.semantic_teacher: Optional[nn.Module] = None
        self._teacher_image_size: int = 224

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def setup(self, stage: str) -> None:
        hp = self.hparams
        if stage != 'fit':
            return
        self._prepare_cd_split_poolers()
        if hp.init_siglip2:
            frozen = _STAGE_SIGLIP2_FROZEN_BLOCKS[hp.training_stage]
            self.model.load_siglip2_weights(hp.siglip2_model_name, frozen)
        if hp.use_semantic_distill and self.semantic_teacher is None:
            self._load_semantic_teacher(hp.siglip2_model_name)

    def _prepare_cd_split_poolers(self) -> None:
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is None or not hasattr(dm, 'hparams'):
            return
        dm_hp = dm.hparams
        active = getattr(dm_hp, 'active_modalities', None) or []
        specs = []
        for modality in active:
            if modality == 'image':
                specs.append({
                    'modality': 'image',
                    'resolution': dm_hp.image_resolution,
                })
            elif modality == 'video':
                specs.append({
                    'modality': 'video',
                    'resolution': dm_hp.video_resolution,
                    'frames':     dm_hp.video_frames,
                    't_patch':    self.hparams.t_patch,
                })
            elif modality == 'threed':
                specs.append({
                    'modality': 'threed',
                    'resolution': dm_hp.triplane_res,
                })
        if specs:
            self.model.prepare_for_modalities(specs)

    def _load_semantic_teacher(self, model_name: str) -> None:
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
        except Exception as exc:  # noqa: BLE001
            print(f"[lightning_module] semantic teacher load failed ({exc}); "
                  f"distillation disabled this run")
            self.semantic_teacher = None

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.semantic_teacher is not None:
            self.semantic_teacher.eval()
        return self

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state = checkpoint.get('state_dict', {})
        for k in list(state.keys()):
            if k.startswith('semantic_teacher.'):
                del state[k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.semantic_teacher is None:
            return
        sd = checkpoint.setdefault('state_dict', {})
        for k, v in self.semantic_teacher.state_dict().items():
            sd.setdefault(f'semantic_teacher.{k}', v)

    # ------------------------------------------------------------------ #
    #  Prefix sampling                                                    #
    # ------------------------------------------------------------------ #

    def _select_recon_prefixes(self) -> Optional[Sequence[int]]:
        """Return the prefixes to decode this step, per ``recon_prefix_policy``."""
        policy = self.hparams.recon_prefix_policy
        all_dims = self.model.matryoshka_dims
        if policy == 'all':
            return None  # MAVT.forward defaults to all
        if policy == 'largest':
            return (all_dims[-1],)
        return (random.choice(all_dims),)  # 'random' (MRL-E)

    # ------------------------------------------------------------------ #
    #  Training step                                                       #
    # ------------------------------------------------------------------ #

    def _step(self, batch: Dict, log_prefix: str) -> torch.Tensor:
        x       = batch['data']
        modality = batch['modality']

        out = self.model(
            x, modality,
            decode=True,
            recon_prefixes=self._select_recon_prefixes(),
        )

        # For video: decoder reconstructs in patch-grid temporal space.
        if modality == 'video':
            t_patch = self.hparams.t_patch
            target = x[:, :, ::t_patch]
        else:
            target = x

        teacher_embed: Optional[torch.Tensor] = None
        if self.semantic_teacher is not None:
            with torch.no_grad():
                proxy = self._make_teacher_input(x, modality, self._teacher_image_size)
                teacher_embed = self.semantic_teacher(pixel_values=proxy).pooler_output

        losses = self.loss_fn(
            target=target,
            modality=modality,
            recon_per_prefix=out.reconstruction,
            kl_per_prefix=out.loss_kl,
            sem_per_prefix=out.semantic,
            all_prefixes=self.model.matryoshka_dims,
            slot_diversity=out.cd_metrics['slot_diversity'],
            teacher_embed=teacher_embed,
        )

        for k, v in losses.items():
            self.log(f'{log_prefix}/{k}', v, on_step=True, on_epoch=True,
                     prog_bar=(k == 'loss'), sync_dist=True)
        for k, v in out.cd_metrics.items():
            self.log(f'{log_prefix}/cd_{k}', v, on_step=False, on_epoch=True,
                     sync_dist=True)
        self.log(f'{log_prefix}/modality_{modality}', 1.0,
                 on_step=False, on_epoch=True, sync_dist=False)

        return losses['loss']

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        with torch.no_grad():
            self._step(batch, 'val')
            if batch_idx == 0:
                self._log_images(batch)

    # ------------------------------------------------------------------ #
    #  Teacher input proxy                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_teacher_input(x: torch.Tensor, modality: str,
                            target_size: int) -> torch.Tensor:
        if modality == 'image':
            proxy = x
        elif modality == 'video':
            T = x.shape[2]
            proxy = x[:, :, T // 2]
        elif modality == 'threed':
            proxy = x[:, 0]
        else:
            raise ValueError(f"Unknown modality: {modality}")

        if proxy.shape[-1] != target_size or proxy.shape[-2] != target_size:
            proxy = F.interpolate(
                proxy, size=(target_size, target_size),
                mode='bilinear', align_corners=False,
            )
        return proxy

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def _log_images(self, batch: Dict, n: int = 4) -> None:
        try:
            x       = batch['data'][:n]
            modality = batch['modality']
            d_max = self.model.matryoshka_dims[-1]
            out = self.model(x, modality, decode=True, recon_prefixes=(d_max,))
            recon = out.reconstruction[d_max]

            if modality == 'image':
                grid_in  = _to_grid(x)
                grid_out = _to_grid(recon)
            elif modality == 'video':
                grid_in  = _to_grid(x[:, :, 0])
                grid_out = _to_grid(recon[:, :, 0])
            elif modality == 'threed':
                grid_in  = _to_grid(x[:, 0])
                grid_out = _to_grid(recon[:, 0])
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
        x = x.detach().cpu().float().clamp(-1, 1)
        x = (x + 1) / 2
        grid = make_grid(x, nrow=nrow, normalize=False)
        arr = (grid.permute(1, 2, 0).numpy() * 255).astype('uint8')
        return Image.fromarray(arr)
    except Exception:  # noqa: BLE001
        return None
