"""Stage-to-stage training orchestrator for MAVT.

Implements progressive training:
    Stage 1: Image Foundation (200k steps)
    Stage 2: Video Dynamics (200k steps)
    Stage 3: 3D Geometry (50k steps)

Each stage:
- Loads datasets with appropriate modalities
- Configures model freezing/unfreezing
- Sets up optimizer with per-component learning rates
- Runs training loop with logging and checkpointing
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .converter_4d import Unified4DConverter
from .curriculum import StageConfig, apply_stage
from .packing_enhanced import EnhancedNaViTPacker
from .sampler_weighted import ModalityWeightedSampler, StageDatasetFactory

logger = logging.getLogger(__name__)


class StageTrainer:
    """Complete stage-to-stage training orchestrator."""

    def __init__(
        self,
        model: nn.Module,
        data_roots: Dict[str, Optional[str]],
        output_dir: str = "./checkpoints",
        log_every: int = 100,
        val_every: int = 5000,
        save_every: int = 10000,
    ):
        """Initialize trainer.

        Args:
            model: MAVTokenizer or MAVTTask model
            data_roots: Dict mapping modality -> data path
                e.g., {"image": "/data/images", "video": "/data/videos", "3d": "/data/3d"}
            output_dir: Directory for checkpoints
            log_every: Log every N steps
            val_every: Validate every N steps
            save_every: Save checkpoint every N steps
        """
        self.model = model
        self.data_roots = data_roots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_every = log_every
        self.val_every = val_every
        self.save_every = save_every

        # Setup converter and packer
        self.converter = Unified4DConverter(
            patch_size=16,
            temporal_patch=2,
        )
        self.packer = EnhancedNaViTPacker(
            max_seq_len=4096,
            max_samples=32,
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"StageTrainer initialized on {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

    def train_all_stages(
        self,
        stages: list[StageConfig],
        start_stage: int = 0,
    ):
        """Train through all stages sequentially.

        Args:
            stages: List of StageConfig objects
            start_stage: Stage index to start from (0-indexed)
        """
        for stage_idx, stage_config in enumerate(stages):
            if stage_idx < start_stage:
                logger.info(f"Skipping Stage {stage_idx + 1}: {stage_config.name}")
                continue

            logger.info(
                f"\n{'='*70}\n"
                f"Starting Stage {stage_idx + 1}/{len(stages)}: {stage_config.name}\n"
                f"Steps: {stage_config.n_steps}\n"
                f"Modalities: {stage_config.modalities}\n"
                f"Task ratios: {stage_config.task_ratios}\n"
                f"{'='*70}\n"
            )

            # Load checkpoint from previous stage if available
            if stage_idx > 0:
                prev_ckpt = self.output_dir / f"stage{stage_idx}_final.pt"
                if prev_ckpt.exists():
                    logger.info(f"Loading from previous stage: {prev_ckpt}")
                    self.load_checkpoint(prev_ckpt)

            # Train this stage
            self.train_stage(stage_config, stage_idx + 1)

            # Save final checkpoint
            final_ckpt = self.output_dir / f"stage{stage_idx + 1}_final.pt"
            self.save_checkpoint(final_ckpt, stage_idx + 1, stage_config.n_steps)
            logger.info(f"Stage {stage_idx + 1} complete. Saved: {final_ckpt}")

    def train_stage(self, stage_config: StageConfig, stage_num: int):
        """Train a single stage.

        Args:
            stage_config: Configuration for this stage
            stage_num: Stage number (1, 2, or 3)
        """
        # Apply stage configuration to model
        apply_stage(self.model, stage_config)

        # Create datasets
        datasets = StageDatasetFactory.create_stage_datasets(
            stage_config, self.data_roots
        )

        # Create weighted sampler
        sampler = StageDatasetFactory.create_sampler(
            stage_config, datasets, num_workers=4
        )

        # Setup optimizer
        optimizer = self._create_optimizer(stage_config)
        scheduler = self._create_scheduler(optimizer, stage_config)

        # Training loop
        self.model.train()
        global_step = 0

        logger.info(f"Starting training loop for {stage_config.n_steps} steps...")

        for step in range(stage_config.n_steps):
            # ── Sample raw data ──
            raw_samples = sampler.sample(n=64)  # Sample 64 items per step

            # ── Convert to 4D tokens ──
            converted = []
            for sample in raw_samples:
                try:
                    conv = self.converter.convert(sample)
                    converted.append(conv)
                except Exception as e:
                    logger.warning(f"Failed to convert sample: {e}")

            if not converted:
                logger.warning("No valid samples in this step, skipping...")
                continue

            # ── Pack into sequences ──
            packs = self.packer.pack_batch(converted, batch_size=8)

            # ── Create batch tensors with masks ──
            batch = self.packer.create_batch_tensors(packs, self.device)

            # ── Forward pass ──
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output = self.model(batch)

                    # Extract loss
                    if isinstance(output, dict) and "loss" in output:
                        loss = output["loss"]
                        logs = output.get("logs", {})
                    else:
                        loss = output
                        logs = {}

                # ── Backward ──
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=stage_config.grad_clip
                )

                optimizer.step()
                scheduler.step()

                # ── Logging ──
                if step % self.log_every == 0:
                    self._log_step(step, stage_num, loss, logs, batch, optimizer)

                # ── Validation ──
                if step % self.val_every == 0 and step > 0:
                    val_loss = self._validate(sampler)
                    logger.info(
                        f"[Stage {stage_num}] Step {step}: val_loss={val_loss:.4f}"
                    )
                    self.model.train()

                # ── Checkpointing ──
                if step % self.save_every == 0 and step > 0:
                    ckpt_path = self.output_dir / f"stage{stage_num}_step{step}.pt"
                    self.save_checkpoint(ckpt_path, stage_num, step)

            except Exception as e:
                logger.error(f"Error in training step {step}: {e}", exc_info=True)
                continue

            global_step += 1

    def _create_optimizer(self, stage_config: StageConfig) -> AdamW:
        """Create optimizer with per-component learning rates.

        Different LR multipliers for different components:
        - SigLIP2: 0.0 (frozen) → 0.1 (partial) → 0.3 (unfrozen)
        - STF + Graph + Spectral: 1.0
        - Encoder: 0.8
        - Decoder: 1.0
        - Latent proj: 1.0
        """
        param_groups = []
        base_lr = stage_config.lr

        # Get model components
        model = getattr(self.model, "tokenizer", self.model)

        # SigLIP2 (if unfrozen)
        if hasattr(model, "patchify") and not stage_config.freeze_siglip2:
            siglip_params = [
                p for p in model.patchify.parameters() if p.requires_grad
            ]
            if siglip_params:
                lr_mult = 0.1 if stage_config.siglip2_unfreeze_last_n > 0 else 0.3
                param_groups.append({
                    "params": siglip_params,
                    "lr": base_lr * lr_mult,
                    "name": "siglip2",
                })

        # STF + Graph + Spectral (if available)
        for name in ["stf", "graph_builder", "spectral_pe"]:
            if hasattr(model, name):
                module = getattr(model, name)
                params = [p for p in module.parameters() if p.requires_grad]
                if params:
                    param_groups.append({
                        "params": params,
                        "lr": base_lr * 1.0,
                        "name": name,
                    })

        # Encoder
        if hasattr(model, "encoder"):
            params = [p for p in model.encoder.parameters() if p.requires_grad]
            if params:
                param_groups.append({
                    "params": params,
                    "lr": base_lr * 0.8,
                    "name": "encoder",
                })

        # Decoder
        if hasattr(model, "decoder"):
            params = [p for p in model.decoder.parameters() if p.requires_grad]
            if params:
                param_groups.append({
                    "params": params,
                    "lr": base_lr * 1.0,
                    "name": "decoder",
                })

        # Latent projection
        if hasattr(model, "latent_proj"):
            params = [p for p in model.latent_proj.parameters() if p.requires_grad]
            if params:
                param_groups.append({
                    "params": params,
                    "lr": base_lr * 1.0,
                    "name": "latent_proj",
                })

        # Fallback: all trainable params
        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": base_lr}]

        optimizer = AdamW(
            param_groups,
            betas=(0.9, 0.95),
            weight_decay=0.05,
        )

        logger.info(f"Created optimizer with {len(param_groups)} parameter groups")
        for pg in param_groups:
            n_params = sum(p.numel() for p in pg["params"])
            logger.info(
                f"  {pg.get('name', 'default')}: "
                f"{n_params/1e6:.1f}M params, lr={pg['lr']:.2e}"
            )

        return optimizer

    def _create_scheduler(
        self, optimizer: AdamW, stage_config: StageConfig
    ) -> LambdaLR:
        """Create learning rate scheduler with warmup + cosine decay."""
        warmup = stage_config.warmup_steps
        total = stage_config.n_steps

        def lr_lambda(step):
            if step < warmup:
                return step / warmup  # Linear warmup
            else:
                # Cosine decay
                progress = (step - warmup) / (total - warmup)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    def _log_step(
        self,
        step: int,
        stage_num: int,
        loss: torch.Tensor,
        logs: dict,
        batch,
        optimizer: AdamW,
    ):
        """Log training metrics."""
        # Compute average packing efficiency
        avg_efficiency = sum(batch.packing_efficiency) / len(batch.packing_efficiency)
        avg_n_samples = sum(batch.n_samples) / len(batch.n_samples)

        # Get current LR
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[Stage {stage_num}] Step {step}: "
            f"loss={loss.item():.4f}, "
            f"lr={lr:.2e}, "
            f"pack_eff={avg_efficiency:.2%}, "
            f"n_samples={avg_n_samples:.1f}"
        )

        # Log detailed metrics if available
        if logs:
            log_str = ", ".join([f"{k}={v.item():.4f}" for k, v in logs.items()])
            logger.info(f"  Metrics: {log_str}")

    @torch.no_grad()
    def _validate(self, sampler: ModalityWeightedSampler) -> float:
        """Run validation."""
        self.model.eval()

        val_losses = []

        for _ in range(10):  # 10 validation batches
            raw_samples = sampler.sample(n=16)
            converted = [self.converter.convert(s) for s in raw_samples]
            packs = self.packer.pack_batch(converted, batch_size=4)
            batch = self.packer.create_batch_tensors(packs, self.device)

            try:
                output = self.model(batch)
                if isinstance(output, dict) and "loss" in output:
                    val_losses.append(output["loss"].item())
                else:
                    val_losses.append(output.item())
            except Exception as e:
                logger.warning(f"Validation error: {e}")

        return sum(val_losses) / len(val_losses) if val_losses else 0.0

    def save_checkpoint(self, path: Path, stage_num: int, step: int):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "stage": stage_num,
            "step": step,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(
            f"Loaded checkpoint from {path} "
            f"(stage {checkpoint.get('stage', '?')}, step {checkpoint.get('step', '?')})"
        )


__all__ = ["StageTrainer"]
