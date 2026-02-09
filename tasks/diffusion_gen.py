"""Training task for the diffusion generator.

Orchestrates training of the dual-branch diffusion model with:
- VQ loss for codebook learning
- Cross-entropy loss for semantic branch (masked modeling)
- D3PM loss for detail branch (discrete diffusion)
- Optional reconstruction loss for end-to-end supervision
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from atoken.diffusion.generator import DiffusionGenerator
from atoken.losses.recon import ReconstructionLoss


class DiffusionGenerationTask(nn.Module):
    """Training task for the diffusion generator.

    Combines multiple loss terms:
    - Semantic loss: Cross-entropy on masked token prediction
    - Detail loss: D3PM variational bound + auxiliary CE
    - VQ loss: Codebook learning (commitment + embedding)
    - Reconstruction loss: Optional pixel-level supervision

    Example:
        >>> generator = DiffusionGenerator(...)
        >>> task = DiffusionGenerationTask(generator)
        >>> optimizer = torch.optim.AdamW(task.parameters(), lr=1e-4)
        >>> for batch in dataloader:
        ...     outputs = task(batch)
        ...     loss = outputs["loss"]
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        generator: DiffusionGenerator,
        recon_loss: Optional[ReconstructionLoss] = None,
        lambda_semantic: float = 1.0,
        lambda_detail: float = 1.0,
        lambda_vq: float = 1.0,
        lambda_recon: float = 0.0,
        use_reconstruction_loss: bool = False,
    ) -> None:
        """Initialize training task.

        Args:
            generator: The diffusion generator model.
            recon_loss: Reconstruction loss module (optional).
            lambda_semantic: Weight for semantic branch loss.
            lambda_detail: Weight for detail branch loss.
            lambda_vq: Weight for VQ loss.
            lambda_recon: Weight for reconstruction loss.
            use_reconstruction_loss: Whether to compute reconstruction loss.
        """
        super().__init__()

        self.generator = generator
        self.lambda_semantic = lambda_semantic
        self.lambda_detail = lambda_detail
        self.lambda_vq = lambda_vq
        self.lambda_recon = lambda_recon
        self.use_reconstruction_loss = use_reconstruction_loss

        if use_reconstruction_loss and recon_loss is None:
            self.recon_loss = ReconstructionLoss(
                lambda_l1=1.0,
                lambda_lpips=0.0,  # Disable LPIPS by default for speed
                lambda_gram=0.0,
                lambda_clip=0.0,
            )
        else:
            self.recon_loss = recon_loss

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Training forward pass.

        Args:
            batch: Dictionary containing:
                - "clean": Clean target tensor (B, C, [T], H, W)
                - "degraded": Degraded input tensor
                - "degradation_params": Degradation parameter vector (B, D)

        Returns:
            Dictionary with:
                - "loss": Total training loss
                - "logs": Dictionary of individual loss components
        """
        x_clean = batch["clean"]
        y_degraded = batch["degraded"]
        deg_params = batch["degradation_params"]

        # Forward through generator (training mode)
        gen_out = self.generator(x_clean, y_degraded, deg_params)

        # Compute weighted loss
        loss = (
            self.lambda_semantic * gen_out["semantic_loss"]
            + self.lambda_detail * gen_out["detail_loss"]
            + self.lambda_vq * gen_out["vq_loss"]
        )

        logs = {
            "semantic_loss": gen_out["semantic_loss"].detach(),
            "detail_loss": gen_out["detail_loss"].detach(),
            "detail_ce_loss": gen_out["detail_ce_loss"].detach(),
            "detail_vb_loss": gen_out["detail_vb_loss"].detach(),
            "vq_loss": torch.as_tensor(gen_out["vq_loss"]).detach(),
        }

        # Optional reconstruction loss
        if self.use_reconstruction_loss and self.lambda_recon > 0:
            # Generate reconstruction for loss computation
            with torch.no_grad():
                recon_out = self.generator.generate(
                    y_degraded, deg_params,
                    semantic_steps=5,  # Fewer steps during training
                    detail_steps=10,
                )
            reconstruction = recon_out["reconstruction"]

            # Compute reconstruction loss
            recon_loss_val, recon_logs = self.recon_loss(reconstruction, x_clean)
            loss = loss + self.lambda_recon * recon_loss_val
            logs.update({f"recon_{k}": v for k, v in recon_logs.items()})

        logs["total_loss"] = loss.detach()

        return {"loss": loss, "logs": logs}

    @torch.no_grad()
    def validate(
        self,
        batch: Dict[str, torch.Tensor],
        semantic_steps: int = 10,
        detail_steps: int = 25,
    ) -> Dict[str, Any]:
        """Validation forward pass with generation.

        Args:
            batch: Validation batch.
            semantic_steps: Steps for semantic sampling.
            detail_steps: Steps for detail sampling.

        Returns:
            Dictionary with losses and reconstructions.
        """
        x_clean = batch["clean"]
        y_degraded = batch["degraded"]
        deg_params = batch["degradation_params"]

        # Compute training losses
        gen_out = self.generator(x_clean, y_degraded, deg_params)

        # Generate reconstruction
        recon_out = self.generator.generate(
            y_degraded, deg_params,
            semantic_steps=semantic_steps,
            detail_steps=detail_steps,
        )

        # Compute reconstruction metrics
        reconstruction = recon_out["reconstruction"]

        # L1/L2 metrics
        l1_error = (reconstruction - x_clean).abs().mean()
        l2_error = ((reconstruction - x_clean) ** 2).mean().sqrt()

        # PSNR
        mse = ((reconstruction - x_clean) ** 2).mean()
        psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))

        logs = {
            "val_semantic_loss": gen_out["semantic_loss"].detach(),
            "val_detail_loss": gen_out["detail_loss"].detach(),
            "val_l1_error": l1_error,
            "val_l2_error": l2_error,
            "val_psnr": psnr,
        }

        return {
            "loss": gen_out["semantic_loss"] + gen_out["detail_loss"],
            "logs": logs,
            "reconstruction": reconstruction,
            "clean": x_clean,
            "degraded": y_degraded,
        }


class DiffusionGenerationTaskV2(nn.Module):
    """Alternative training task with staged training support.

    Supports multiple training phases:
    1. VQ pretraining: Train quantizers only
    2. Semantic pretraining: Train semantic branch only
    3. Detail pretraining: Train detail branch only
    4. Joint training: Train all components together
    """

    def __init__(
        self,
        generator: DiffusionGenerator,
        recon_loss: Optional[ReconstructionLoss] = None,
        training_stage: str = "joint",
    ) -> None:
        """Initialize task.

        Args:
            generator: Diffusion generator.
            recon_loss: Reconstruction loss module.
            training_stage: One of "vq", "semantic", "detail", "joint".
        """
        super().__init__()

        self.generator = generator
        self.recon_loss = recon_loss
        self.training_stage = training_stage

        # Configure which components to train
        self._configure_training_stage(training_stage)

    def _configure_training_stage(self, stage: str) -> None:
        """Configure which parameters to train based on stage."""
        # First, freeze everything
        for param in self.generator.parameters():
            param.requires_grad = False

        if stage == "vq":
            # Train only quantizers and encoder projections
            for name, param in self.generator.named_parameters():
                if "quantizer" in name or "to_semantic" in name or "to_detail" in name:
                    param.requires_grad = True

        elif stage == "semantic":
            # Train semantic branch and related components
            for name, param in self.generator.named_parameters():
                if "semantic" in name or "shallow_encoder" in name:
                    param.requires_grad = True

        elif stage == "detail":
            # Train detail branch
            for name, param in self.generator.named_parameters():
                if "detail" in name or "degradation_encoder" in name:
                    param.requires_grad = True

        elif stage == "joint":
            # Train everything
            for param in self.generator.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def set_training_stage(self, stage: str) -> None:
        """Change training stage."""
        self.training_stage = stage
        self._configure_training_stage(stage)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Training forward pass based on current stage."""
        x_clean = batch["clean"]
        y_degraded = batch["degraded"]
        deg_params = batch["degradation_params"]

        if self.training_stage == "vq":
            return self._forward_vq(x_clean)
        elif self.training_stage == "semantic":
            return self._forward_semantic(x_clean, y_degraded, deg_params)
        elif self.training_stage == "detail":
            return self._forward_detail(x_clean, y_degraded, deg_params)
        else:  # joint
            return self._forward_joint(x_clean, y_degraded, deg_params)

    def _forward_vq(self, x_clean: torch.Tensor) -> Dict[str, Any]:
        """VQ pretraining: only codebook loss."""
        enc = self.generator.encode(x_clean, return_quantized=True)
        loss = enc["vq_loss"]
        return {
            "loss": loss,
            "logs": {"vq_loss": torch.as_tensor(loss).detach()},
        }

    def _forward_semantic(
        self,
        x_clean: torch.Tensor,
        y_degraded: torch.Tensor,
        deg_params: torch.Tensor,
    ) -> Dict[str, Any]:
        """Semantic branch pretraining."""
        clean_enc = self.generator.encode(x_clean, return_quantized=True)
        y_features = self.generator.shallow_encoder(y_degraded)
        deg_enc = self.generator.encode(y_degraded, return_quantized=False)
        zA = deg_enc["z_artifact"]

        semantic_loss, _ = self.generator.semantic_branch(
            zC_tokens=clean_enc["zC_tokens"],
            zA_embeddings=zA,
            y_features=y_features,
            positions=clean_enc.get("positions"),
        )

        return {
            "loss": semantic_loss,
            "logs": {"semantic_loss": semantic_loss.detach()},
        }

    def _forward_detail(
        self,
        x_clean: torch.Tensor,
        y_degraded: torch.Tensor,
        deg_params: torch.Tensor,
    ) -> Dict[str, Any]:
        """Detail branch pretraining."""
        clean_enc = self.generator.encode(x_clean, return_quantized=True)
        y_features = self.generator.shallow_encoder(y_degraded)
        deg_enc = self.generator.encode(y_degraded, return_quantized=False)
        zA = deg_enc["z_artifact"]
        deg_embed = self.generator.degradation_encoder(deg_params)

        # Use ground truth semantic tokens
        zC_embed = clean_enc["zC_quantized"]

        detail_out = self.generator.detail_branch(
            zD_tokens=clean_enc["zD_tokens"],
            zC_completed=zC_embed,
            zA_embeddings=zA,
            y_features=y_features,
            degradation_params=deg_embed,
            positions=clean_enc.get("positions"),
        )

        return {
            "loss": detail_out["loss"],
            "logs": {
                "detail_loss": detail_out["loss"].detach(),
                "detail_ce_loss": detail_out["ce_loss"].detach(),
                "detail_vb_loss": detail_out["vb_loss"].detach(),
            },
        }

    def _forward_joint(
        self,
        x_clean: torch.Tensor,
        y_degraded: torch.Tensor,
        deg_params: torch.Tensor,
    ) -> Dict[str, Any]:
        """Joint training of all components."""
        gen_out = self.generator(x_clean, y_degraded, deg_params)

        return {
            "loss": gen_out["loss"],
            "logs": gen_out["logs"],
        }


__all__ = ["DiffusionGenerationTask", "DiffusionGenerationTaskV2"]
