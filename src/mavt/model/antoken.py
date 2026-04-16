from __future__ import annotations

import torch

from mavt.compat import LightningModule
from mavt.decoder.image import ImageDecoder
from mavt.decoder.video import VideoDecoder
from mavt.encoder.unified_encoder import UnifiedEncoder
from mavt.latent.router import LatentRouter
from mavt.losses.clip_contrastive import CLIPContrastiveLoss
from mavt.losses.fsq_vq_aux import DiscreteAuxLoss
from mavt.losses.kl import KLRegLoss
from mavt.losses.recon import ReconstructionLoss
from mavt.losses.temporal import OpticalFlowConsistencyLoss
from mavt.model.stage_manager import StageManager


class AToken(LightningModule):
    def __init__(
        self,
        encoder_cfg: dict | None = None,
        latent_cfg: dict | None = None,
        decoder_cfg: dict | None = None,
        loss_cfg: dict | None = None,
        *,
        stage: int = 1,
        optimizer_cfg: dict | None = None,
        scheduler_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        encoder_cfg = encoder_cfg or {}
        latent_cfg = latent_cfg or {}
        decoder_cfg = decoder_cfg or {}
        loss_cfg = loss_cfg or {}
        optimizer_cfg = optimizer_cfg or {"lr": 1e-4, "weight_decay": 0.05}
        scheduler_cfg = scheduler_cfg or {"t_max": 10000, "eta_min": 1e-6}

        embed_dim = encoder_cfg.get("embed_dim", 256)
        patch_size = encoder_cfg.get("patch_size", 16)
        temporal_patch_size = encoder_cfg.get("temporal_patch_size", 2)
        latent_dim = latent_cfg.get("latent_dim", 32)
        semantic_dim = latent_cfg.get("semantic_dim", 256)

        self.stage = stage
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        router_cfg = dict(latent_cfg)
        router_cfg.pop("stage2_latent_dim", None)

        self.encoder = UnifiedEncoder(**encoder_cfg)
        self.router = LatentRouter(embed_dim=embed_dim, **router_cfg)
        self.image_decoder = ImageDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            patch_size=decoder_cfg.get("patch_size", patch_size),
            depth=decoder_cfg.get("depth", 4),
            num_heads=decoder_cfg.get("num_heads", encoder_cfg.get("num_heads", 8)),
        )
        self.video_decoder = VideoDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            patch_size=decoder_cfg.get("patch_size", patch_size),
            temporal_patch_size=decoder_cfg.get("temporal_patch_size", temporal_patch_size),
            depth=decoder_cfg.get("depth", 4),
            num_heads=decoder_cfg.get("num_heads", encoder_cfg.get("num_heads", 8)),
        )

        self.recon_loss = ReconstructionLoss(
            l1_weight=loss_cfg.get("l1_weight", 1.0),
            lpips_weight=loss_cfg.get("lpips_weight", 0.0),
            gram_weight=loss_cfg.get("gram_weight", 0.0),
        )
        self.temporal_loss = OpticalFlowConsistencyLoss(weight=loss_cfg.get("temporal_weight", 0.0))
        self.semantic_loss = CLIPContrastiveLoss(
            semantic_dim,
            text_model_name=loss_cfg.get("text_model_name"),
            weight=loss_cfg.get("semantic_weight", 0.0),
            temperature=loss_cfg.get("temperature", 0.07),
        )
        self.kl_loss = KLRegLoss(weight=loss_cfg.get("kl_weight", 1e-5))
        self.discrete_aux_loss = DiscreteAuxLoss(
            commitment_weight=loss_cfg.get("commitment_weight", 1.0)
        )
        self.stage_manager = StageManager(stage=stage, stage2_latent_dim=latent_cfg.get("stage2_latent_dim", 48))

    def on_fit_start(self) -> None:
        self.stage_manager.apply(self, stage=self.stage)

    def _extract_modality(self, batch: dict) -> str:
        modality = batch["modality"]
        if isinstance(modality, (list, tuple)):
            return modality[0]
        return modality

    def _decode(self, modality: str, latents: torch.Tensor, batch: dict) -> torch.Tensor:
        if modality == "image":
            return self.image_decoder(latents, batch)
        if modality == "video":
            return self.video_decoder(latents, batch)
        raise ValueError(f"Unsupported modality: {modality}")

    def forward(self, batch: dict) -> dict:
        enc_out = self.encoder(batch)
        latent_out = self.router(enc_out, stage=self.stage)
        reconstruction = self._decode(self._extract_modality(batch), latent_out["continuous"]["sample"], batch)
        return {
            "encoder": enc_out,
            "latents": latent_out,
            "reconstruction": reconstruction,
        }

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        outputs = self.forward(batch)
        modality = self._extract_modality(batch)
        target = batch[modality]
        reconstruction = outputs["reconstruction"]

        recon_losses = self.recon_loss(target, reconstruction)
        kl = self.kl_loss(
            outputs["latents"]["continuous"]["mu"],
            outputs["latents"]["continuous"]["logvar"],
        )
        temporal = self.temporal_loss(target, reconstruction) if modality == "video" else target.new_zeros(())
        semantic = self.semantic_loss(outputs["latents"]["semantic"], list(batch["caption"]))
        discrete_loss, usage = self.discrete_aux_loss(outputs["latents"].get("discrete"))

        total = recon_losses["total"] + kl + temporal + semantic + discrete_loss
        metrics = {
            f"{prefix}/loss": total,
            f"{prefix}/recon_l1": recon_losses["l1"],
            f"{prefix}/recon_lpips": recon_losses["lpips"],
            f"{prefix}/recon_gram": recon_losses["gram"],
            f"{prefix}/kl": kl,
            f"{prefix}/temporal": temporal,
            f"{prefix}/semantic": semantic,
            f"{prefix}/discrete": discrete_loss,
            f"{prefix}/code_usage": usage,
        }
        self.log_dict(metrics, prog_bar=(prefix == "train"), sync_dist=False)
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.stage_manager.trainable_params(self), **self.optimizer_cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.scheduler_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
