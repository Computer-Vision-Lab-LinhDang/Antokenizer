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
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities import grad_norm

from mavt.model.mavt import MAVT
from mavt.losses.losses import MAVTLoss, pool_teacher_tokens


_STAGE_LR = {1: 1e-4, 2: 5e-5, 3: 2e-5}
_STAGE_SIGLIP2_FROZEN_BLOCKS = {1: 10, 2: 6, 3: 0}  # number of transformer blocks frozen
_STAGE_W_SEM = {1: 0.5, 2: 0.3, 3: 0.2}             # default cosine-distill weight per stage


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
        rgat_impl: str = "dense",             # dense | sparse | flex (graph attention)
        edge_plane_local: bool = False,
        edge_cross_mode: str = "shared_axis",
        use_gradient_checkpointing: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # C-D split (cd-split commit 6368dfb)
        local_detail_window_size: int = 1,
        local_detail_temporal_window_size: int = 1,
        # Loss
        w_l1: float = 1.0,
        w_lpips: float = 0.1,
        w_kl: float = 1.0,      # passthrough — KL pre-scaled by VAEHead.kl_weight
        w_clip: float = 0.0,
        w_sem: float = 0.0,
        w_aux: float = 0.01,
        w_temp: float = 0.0,    # temporal consistency weight (video only)
        w_vic: float = 0.0,     # VICReg anti-collapse on pooled embedding
        w_dense: float = 0.0,   # per-token distillation of backbone tokens → teacher patch tokens (image)
        w_content: float = 0.0, # force content slots to reconstruct the backbone features (real residual split)
        distill_center: bool = False,  # centered cosine distillation
        use_lpips: bool = True,
        use_clip: bool = False,
        active_modalities: list = None,  # e.g. ['image', 'video']; None → all three
        # Curriculum
        training_stage: int = 1,
        siglip2_model_name: str = "google/siglip2-so400m-patch16-384",
        init_siglip2: bool = True,
        init_siglip2_patchify: bool = False,   # inherit Conv2d kernel + 24x24 pos table
        use_semantic_distill: bool = True,
        semantic_content_only: bool = False,  # understanding head reads content slots only
        # Cross-stage weight transfer (loads model weights only, NOT optimizer
        # / scheduler / step state — use --ckpt_path for true resume instead).
        init_from_ckpt: Optional[str] = None,
        # Optimiser
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        total_steps: int = 200_000,
        lr: Optional[float] = None,          # override the per-stage default LR
    ):
        super().__init__()
        # active_modalities is owned by the DataModule (single source of truth);
        # keeping it out of the model hparams avoids Lightning's duplicate-key
        # merge error when both modules would log it with different values.
        self.save_hyperparameters(ignore=["active_modalities"])
        self._active_modalities_cfg = tuple(active_modalities) if active_modalities else None

        self.model = MAVT(
            embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks,
            patch_size=patch_size, t_patch=t_patch,
            latent_dim=latent_dim, kl_weight=kl_weight,
            semantic_dim=semantic_dim, dec_dim=dec_dim,
            semantic_content_only=semantic_content_only,
            num_dec_attn_blocks=num_dec_attn_blocks, r_s=r_s, r_t=r_t,
            rgat_impl=rgat_impl, edge_plane_local=edge_plane_local,
            edge_cross_mode=edge_cross_mode,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mlp_ratio=mlp_ratio, dropout=dropout,
            local_detail_window_size=local_detail_window_size,
            local_detail_temporal_window_size=local_detail_temporal_window_size,
        )

        _active_mods = tuple(active_modalities) if active_modalities else ('image', 'video', 'threed')
        self.loss_fn = MAVTLoss(
            w_l1=w_l1, w_lpips=w_lpips, w_kl=w_kl,
            w_clip=w_clip, w_sem=w_sem, w_aux=w_aux,
            w_temp=w_temp, w_vic=w_vic, w_dense=w_dense, w_content=w_content,
            distill_center=distill_center,
            use_lpips=use_lpips, use_clip=use_clip,
            active_modalities=_active_mods,
        )

        # Learnable projection for dense distillation (identity init; lets the trunk keep its
        # own basis while matching the teacher's patch features up to a linear map).
        self.dense_proj: Optional[nn.Linear] = None
        if w_dense > 0.0:
            self.dense_proj = nn.Linear(embed_dim, embed_dim)
            with torch.no_grad():
                self.dense_proj.weight.copy_(torch.eye(embed_dim)); self.dense_proj.bias.zero_()

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
        # Eager slot pooler creation — must run BEFORE configure_optimizers
        # so the new params land in the optimizer's param_groups.
        self._prepare_cd_split_poolers()
        # Sync EMA modalities from DataModule — single source of truth
        self._sync_ema_modalities()
        if hp.init_siglip2:
            frozen = _STAGE_SIGLIP2_FROZEN_BLOCKS[hp.training_stage]
            self.model.load_siglip2_weights(hp.siglip2_model_name, frozen, strict=True,
                                            init_patchify=hp.init_siglip2_patchify)
        if hp.use_semantic_distill and self.semantic_teacher is None:
            self._load_semantic_teacher(hp.siglip2_model_name)
        # Cross-stage weight transfer (after siglip2 / teacher are in place so
        # they get overwritten by ckpt values when present).
        if hp.init_from_ckpt:
            self._load_weights_from_ckpt(hp.init_from_ckpt)

    def _load_weights_from_ckpt(self, path: str) -> None:
        # weights_only=False is required because Lightning ckpts contain a
        # full pickle (state_dict + hparams + callbacks). Source is our own
        # filesystem so untrusted-pickle risk is N/A.
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        kept_missing = [
            k for k in missing
            if not k.startswith('semantic_teacher.')
            and not k.startswith('model.cd_split._content_poolers.')
            and not k.startswith('model.cd_split._detail_poolers.')
        ]
        print(f"[init_from_ckpt] loaded {path}")
        print(f"[init_from_ckpt] missing (kept random init): {len(kept_missing)} keys "
              f"+ {len(missing) - len(kept_missing)} expected (teacher/new poolers)")
        if unexpected:
            print(f"[init_from_ckpt] unexpected (dropped): {len(unexpected)} keys")
        print("[init_from_ckpt] NOTE: optimizer state / LR scheduler / global_step are NOT "
              "restored — this is a soft restart. For exact resume of the SAME stage, "
              "use Lightning --ckpt_path instead.")

    def _prepare_cd_split_poolers(self) -> None:
        """Read active modality + resolution from the attached DataModule and
        eagerly create every SlotPooler the trainer will need."""
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

    def _sync_ema_modalities(self) -> None:
        """Use DataModule as single source of truth for active_modalities.

        Overrides whatever was set in model config so data.active_modalities
        and the EMA weighter never drift apart.
        """
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is not None and hasattr(dm, 'hparams'):
            active = getattr(dm.hparams, 'active_modalities', None)
            if active:
                self.loss_fn.ema_weighter.active_modalities = tuple(active)
                return
        # Fallback: use model config param if DataModule not available
        fallback = self._active_modalities_cfg
        if fallback:
            self.loss_fn.ema_weighter.active_modalities = tuple(fallback)

    def _load_semantic_teacher(self, model_name: str) -> None:
        """Load frozen SigLIP2 vision tower as teacher for cosine distillation."""
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
            # use_semantic_distill was requested: failing loud beats silently training
            # without the objective the whole design depends on.
            raise RuntimeError(f"semantic teacher {model_name!r} failed to load: {exc}") from exc

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep frozen teacher in eval mode regardless of train()/eval() calls."""
        super().train(mode)
        if self.semantic_teacher is not None:
            self.semantic_teacher.eval()
        return self

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Strip frozen teacher weights from checkpoints to keep them small."""
        state = checkpoint.get('state_dict', {})
        for k in list(state.keys()):
            if k.startswith('semantic_teacher.'):
                del state[k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Inject the freshly-loaded teacher's params back into the checkpoint
        state_dict so Lightning's strict load does not fail with missing keys.

        Order at training-resume time:
          1. setup('fit')               → teacher loaded from HF (deterministic)
          2. on_load_checkpoint(ckpt)   → we inject teacher keys here
          3. self.load_state_dict(ckpt) → strict=True now sees a complete dict

        Teacher weights are deterministic from the HF model name, so injecting
        the current values is equivalent to whatever the checkpoint would have
        contained — no behavior change, only avoids the strict-load error.
        """
        if self.semantic_teacher is None:
            return
        sd = checkpoint.setdefault('state_dict', {})
        for k, v in self.semantic_teacher.state_dict().items():
            sd.setdefault(f'semantic_teacher.{k}', v)

    # ------------------------------------------------------------------ #
    #  Training step                                                       #
    # ------------------------------------------------------------------ #

    def _step(self, batch: Dict, log_prefix: str) -> torch.Tensor:
        x       = batch['data']
        modality = batch['modality']

        out = self.model(x, modality, decode=True)

        # The decoder now returns the full input shape for every modality (video:
        # every frame via temporal expansion; 3D: all three planes). Any mismatch
        # is a bug and MAVTLoss raises instead of aligning silently.
        target = x

        # Vision-vision distillation: forward x_proxy through frozen teacher.
        teacher_embed: Optional[torch.Tensor] = None
        dense_student = dense_teacher = None
        if self.semantic_teacher is not None:
            with torch.no_grad():
                proxy = self._make_teacher_input(x, modality, self._teacher_image_size)
                t_out = self.semantic_teacher(pixel_values=proxy)
                teacher_embed = t_out.pooler_output
                if self.dense_proj is not None and modality == 'image':
                    grid = self.model._grid_shape(modality, x)
                    dense_teacher = pool_teacher_tokens(t_out.last_hidden_state, grid)
            if dense_teacher is not None:
                dense_student = self.dense_proj(out.backbone_tokens)

        losses = self.loss_fn(
            pred=out.reconstruction,
            target=target,
            loss_kl=out.loss_kl,
            slot_diversity=out.cd_metrics['slot_diversity'],
            modality=modality,
            semantic_embed=out.semantic,
            teacher_embed=teacher_embed,
            dense_student=dense_student,
            dense_teacher=dense_teacher,
            content_recon_error=out.cd_metrics.get('content_recon_error'),
        )

        self._log_losses(losses, out.cd_metrics, modality, log_prefix)
        return losses['loss']

    def _log_losses(self, losses: Dict[str, torch.Tensor], cd_metrics: Dict[str, torch.Tensor],
                    modality: str, log_prefix: str) -> None:
        """Log aggregate metrics with a cross-rank reduction and per-modality metrics rank-locally.

        Under DDP each rank trains its own single-modality batch, so a key such as
        ``loss_l1_video`` exists on some ranks and not others at the same step. With
        ``sync_dist=True`` every rank issues one all-reduce per logged key; different key
        sets → the collectives pair up wrongly (silently mixed numbers) or their counts
        differ (validation with a modality missing on a rank) → **deadlock**. The 3-modality
        Stage-3 run hung at its first validation exactly this way (2026-09-02). Only keys that
        every rank logs on every step may use sync_dist=True.
        """
        for k, v in losses.items():                       # same key set on every rank
            self.log(f'{log_prefix}/{k}', v, on_step=True, on_epoch=True,
                     prog_bar=(k == 'loss'), sync_dist=True)
        for k in ('loss', 'loss_recon', 'loss_l1', 'loss_kl'):   # modality-specific → rank-local
            if k in losses:
                self.log(f'{log_prefix}/{k}_{modality}', losses[k],
                         on_step=True, on_epoch=True, sync_dist=False)
        for k, v in cd_metrics.items():                   # values depend on the modality → rank-local
            self.log(f'{log_prefix}/cd_{k}', v, on_step=False, on_epoch=True, sync_dist=False)
        self.log(f'{log_prefix}/modality_{modality}', 1.0,
                 on_step=False, on_epoch=True, sync_dist=False)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        with torch.no_grad():
            self._step(batch, 'val')
            # Log sample reconstructions to wandb/tensorboard every N steps
            if batch_idx == 0:
                self._log_images(batch)

    # ------------------------------------------------------------------ #
    #  Teacher input proxy                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_teacher_input(x: torch.Tensor, modality: str,
                            target_size: int) -> torch.Tensor:
        """Project the multi-modal input down to a single (B, 3, S, S) image
        the SigLIP2 vision teacher can consume.

        image  : x as-is.
        video  : middle frame.
        threed : XY plane (front view, closest to natural-image distribution).
        """
        if modality == 'image':
            proxy = x
        elif modality == 'video':
            T = x.shape[2]
            proxy = x[:, :, T // 2]                # (B, 3, H, W)
        elif modality == 'threed':
            proxy = x[:, 0]                         # (B, 3, H, W) plane XY
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
            out = self.model(x, modality, decode=True)

            if modality == 'image':
                grid_in  = _to_grid(x)
                grid_out = _to_grid(out.reconstruction)
            elif modality == 'video':
                # Log a temporal strip: 4 evenly-spaced frames per clip stacked
                # horizontally so reviewers can spot temporal coherence (vs.
                # only a single first frame).
                B, C, T, H, W = x.shape
                Tp = out.reconstruction.shape[2]
                t_in = torch.linspace(0, T - 1, 4).long()
                t_out = torch.linspace(0, Tp - 1, 4).long()
                in_strip  = x[:, :, t_in].permute(0, 2, 1, 3, 4).reshape(B * 4, C, H, W)
                out_strip = out.reconstruction[:, :, t_out].permute(0, 2, 1, 3, 4).reshape(B * 4, C, H, W)
                grid_in  = _to_grid(in_strip,  nrow=4)
                grid_out = _to_grid(out_strip, nrow=4)
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
        lr = hp.lr if hp.lr is not None else _STAGE_LR[hp.training_stage]

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

        # Linear warmup + cosine decay over the *actual* run length. `trainer.max_steps`
        # is what this invocation asked for; `hp.total_steps` can be stale when a run is
        # resumed with --ckpt_path (hparams come back from the checkpoint), which once
        # continued a 5k run to 10k at lr ~1e-9.
        total_steps = hp.total_steps
        max_steps = getattr(getattr(self, "_trainer", None), "max_steps", -1)
        if isinstance(max_steps, int) and max_steps > 0:
            total_steps = max_steps
        if total_steps != hp.total_steps:
            print(f"[configure_optimizers] LR schedule length = trainer.max_steps={total_steps} "
                  f"(hparam total_steps={hp.total_steps} ignored)", flush=True)

        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return step / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, total_steps - hp.warmup_steps)
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