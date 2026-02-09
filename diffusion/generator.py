"""Unified diffusion generator combining semantic and detail branches.

The generator orchestrates:
1. Encoding input to discrete tokens (zC, zD, zA)
2. Semantic branch: Masked modeling on zC
3. Detail branch: D3PM discrete diffusion on zD
4. Decoding tokens back to pixels
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from atoken.core.sparse_tensor import SparseTensor4D
from atoken.model.encoder import ATokenEncoder
from atoken.model.decoder import ATokenDecoder

from .detail_branch import DetailBranch
from .quantizer import VectorQuantizer, EMAVectorQuantizer
from .semantic_branch import SemanticBranch
from .shallow_encoder import DegradationEncoder, ShallowFeatureEncoder


class DiffusionGenerator(nn.Module):
    """Unified diffusion generator for image and video synthesis.

    Combines:
    - ATokenEncoder: Encodes input to continuous embeddings
    - VectorQuantizers: Converts embeddings to discrete tokens
    - SemanticBranch: Masked modeling on semantic tokens
    - DetailBranch: D3PM discrete diffusion on detail tokens
    - ATokenDecoder: Reconstructs pixels from tokens

    Training:
    1. Encode clean input to get ground truth tokens
    2. Encode degraded input for conditioning (zA, y_features)
    3. Train semantic branch on masked prediction of zC
    4. Train detail branch on diffusion denoising of zD

    Generation:
    1. Extract features from degraded/noise input
    2. Generate zC via MaskGIT-style parallel decoding
    3. Generate zD via D3PM reverse diffusion
    4. Decode combined tokens to pixels
    """

    def __init__(
        self,
        # Encoder/Decoder (can be provided or created)
        encoder: Optional[ATokenEncoder] = None,
        decoder: Optional[ATokenDecoder] = None,
        # Encoder config (if not provided)
        in_channels: int = 3,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        d_model: int = 768,
        encoder_depth: int = 12,
        decoder_depth: int = 8,
        num_heads: int = 12,
        # Quantization
        semantic_vocab_size: int = 8192,
        detail_vocab_size: int = 16384,
        use_ema_quantizer: bool = True,
        # Semantic branch
        semantic_depth: int = 6,
        semantic_mask_ratio: float = 0.15,
        # Detail branch
        detail_depth: int = 8,
        detail_timesteps: int = 50,
        detail_transition_type: str = "absorbing",
        detail_schedule_type: str = "cosine",
        # Shallow encoder
        shallow_layers: int = 4,
        shallow_base_channels: int = 64,
        # Degradation
        degradation_dim: int = 32,
        # Shared
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        """Initialize diffusion generator.

        Args:
            encoder: Pre-built ATokenEncoder (optional).
            decoder: Pre-built ATokenDecoder (optional).
            in_channels: Input channels.
            patch_size: Patch size for tokenization.
            d_model: Model dimension.
            encoder_depth: Encoder transformer depth.
            decoder_depth: Decoder transformer depth.
            num_heads: Number of attention heads.
            semantic_vocab_size: Semantic codebook size.
            detail_vocab_size: Detail codebook size.
            use_ema_quantizer: Use EMA quantizer.
            semantic_depth: Semantic branch depth.
            semantic_mask_ratio: Training mask ratio.
            detail_depth: Detail branch depth.
            detail_timesteps: D3PM timesteps.
            detail_transition_type: D3PM transition type.
            detail_schedule_type: D3PM schedule type.
            shallow_layers: Shallow encoder layers.
            shallow_base_channels: Shallow encoder base channels.
            degradation_dim: Degradation parameter dimension.
            mlp_ratio: MLP ratio.
            dropout: Dropout rate.
            attention_dropout: Attention dropout rate.
        """
        super().__init__()

        self.d_model = d_model
        self.semantic_vocab_size = semantic_vocab_size
        self.detail_vocab_size = detail_vocab_size

        # Encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ATokenEncoder(
                in_channels=in_channels,
                patch_size=patch_size,
                d_model=d_model,
                depth=encoder_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )

        # Decoder
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = ATokenDecoder(
                out_channels=in_channels,
                patch_size=patch_size,
                d_model=d_model,
                depth=decoder_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )

        # Vector quantizers
        VQClass = EMAVectorQuantizer if use_ema_quantizer else VectorQuantizer

        self.semantic_quantizer = VQClass(
            num_embeddings=semantic_vocab_size,
            embedding_dim=d_model,
        )
        self.detail_quantizer = VQClass(
            num_embeddings=detail_vocab_size,
            embedding_dim=d_model,
        )

        # Projection heads for different token types
        # Split encoder output into semantic and detail components
        self.to_semantic = nn.Linear(d_model, d_model)
        self.to_detail = nn.Linear(d_model, d_model)
        self.to_artifact = nn.Linear(d_model, d_model)

        # Shallow feature encoder
        self.shallow_encoder = ShallowFeatureEncoder(
            in_channels=in_channels,
            out_dim=d_model,
            num_layers=shallow_layers,
            base_channels=shallow_base_channels,
        )

        # Degradation encoder
        self.degradation_encoder = DegradationEncoder(
            input_dim=degradation_dim,
            out_dim=d_model,
        )

        # Semantic branch
        self.semantic_branch = SemanticBranch(
            d_model=d_model,
            depth=semantic_depth,
            num_heads=num_heads,
            vocab_size=semantic_vocab_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            mask_ratio=semantic_mask_ratio,
            context_dim=d_model,
        )

        # Detail branch
        self.detail_branch = DetailBranch(
            d_model=d_model,
            depth=detail_depth,
            num_heads=num_heads,
            vocab_size=detail_vocab_size,
            num_timesteps=detail_timesteps,
            transition_type=detail_transition_type,
            schedule_type=detail_schedule_type,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            context_dim=d_model,
        )

        # Token combiner for decoder
        self.token_combiner = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def encode(
        self,
        x: torch.Tensor,
        return_quantized: bool = True,
    ) -> Dict[str, Any]:
        """Encode input to tokens.

        Args:
            x: Input tensor (B, C, T, H, W) or (B, C, H, W).
            return_quantized: Whether to quantize to discrete tokens.

        Returns:
            Dictionary with tokens, embeddings, and metadata.
        """
        # Ensure 5D input
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, C, 1, H, W)

        # Encode through AToken encoder
        enc_out = self.encoder(x, return_sparse=True)
        sequence = enc_out["sequence"]  # (B, N, D)
        sparse = enc_out["sparse"]

        # Split into semantic, detail, artifact
        z_semantic = self.to_semantic(sequence)
        z_detail = self.to_detail(sequence)
        z_artifact = self.to_artifact(sequence)

        result = {
            "z_semantic": z_semantic,
            "z_detail": z_detail,
            "z_artifact": z_artifact,
            "sparse": sparse,
            "positions": sparse.positions,
            "mask": sparse.mask,
        }

        if return_quantized:
            # Quantize to discrete tokens
            zC_tokens, zC_quantized, vq_loss_c = self.semantic_quantizer(z_semantic)
            zD_tokens, zD_quantized, vq_loss_d = self.detail_quantizer(z_detail)

            result.update({
                "zC_tokens": zC_tokens,
                "zC_quantized": zC_quantized,
                "zD_tokens": zD_tokens,
                "zD_quantized": zD_quantized,
                "vq_loss": (vq_loss_c or 0) + (vq_loss_d or 0),
            })

        return result

    def forward(
        self,
        x_clean: torch.Tensor,
        y_degraded: torch.Tensor,
        degradation_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            x_clean: Clean target (B, C, T, H, W) or (B, C, H, W).
            y_degraded: Degraded input.
            degradation_params: Degradation parameter vector (B, deg_dim).

        Returns:
            Dictionary with losses and intermediate outputs.
        """
        # Encode clean input for ground truth tokens
        clean_enc = self.encode(x_clean, return_quantized=True)

        # Extract features from degraded input
        y_features = self.shallow_encoder(y_degraded)

        # Encode degraded input for artifact tokens
        deg_enc = self.encode(y_degraded, return_quantized=False)
        zA_embeddings = deg_enc["z_artifact"]

        # Encode degradation parameters
        deg_embed = self.degradation_encoder(degradation_params)

        # Get positions for transformer
        positions = clean_enc.get("positions")
        mask = clean_enc.get("mask")

        # === Semantic Branch ===
        semantic_loss, zC_completed = self.semantic_branch(
            zC_tokens=clean_enc["zC_tokens"],
            zA_embeddings=zA_embeddings,
            y_features=y_features,
            positions=positions,
            mask=mask,
        )

        # === Detail Branch ===
        detail_out = self.detail_branch(
            zD_tokens=clean_enc["zD_tokens"],
            zC_completed=zC_completed,
            zA_embeddings=zA_embeddings,
            y_features=y_features,
            degradation_params=deg_embed,
            positions=positions,
            mask=mask,
        )

        # Total loss
        vq_loss = clean_enc.get("vq_loss", 0)
        total_loss = semantic_loss + detail_out["loss"] + vq_loss

        return {
            "loss": total_loss,
            "semantic_loss": semantic_loss,
            "detail_loss": detail_out["loss"],
            "detail_ce_loss": detail_out["ce_loss"],
            "detail_vb_loss": detail_out["vb_loss"],
            "vq_loss": vq_loss,
            "logs": {
                "semantic_loss": semantic_loss.detach(),
                "detail_loss": detail_out["loss"].detach(),
                "vq_loss": torch.as_tensor(vq_loss).detach(),
            },
        }

    @torch.no_grad()
    def generate(
        self,
        y_degraded: torch.Tensor,
        degradation_params: torch.Tensor,
        semantic_steps: int = 10,
        detail_steps: Optional[int] = None,
        temperature: float = 1.0,
        progress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate clean output from degraded input.

        Args:
            y_degraded: Degraded input (B, C, T, H, W) or (B, C, H, W).
            degradation_params: Degradation parameters (B, deg_dim).
            semantic_steps: MaskGIT steps for semantic branch.
            detail_steps: D3PM steps (None = use default).
            temperature: Sampling temperature.
            progress: Show progress bar.

        Returns:
            Dictionary with reconstruction and intermediate outputs.
        """
        # Ensure 5D input
        if y_degraded.dim() == 4:
            y_degraded = y_degraded.unsqueeze(2)

        # Extract conditioning
        y_features = self.shallow_encoder(y_degraded)
        deg_enc = self.encode(y_degraded, return_quantized=False)
        zA_embeddings = deg_enc["z_artifact"]
        deg_embed = self.degradation_encoder(degradation_params)
        positions = deg_enc.get("positions")
        num_tokens = deg_enc["z_semantic"].size(1)

        # Step 1: Generate semantic tokens
        zC_tokens, zC_embed = self.semantic_branch.sample(
            zA_embeddings=zA_embeddings,
            y_features=y_features,
            num_tokens=num_tokens,
            positions=positions,
            num_steps=semantic_steps,
            temperature=temperature,
        )

        # Step 2: Generate detail tokens
        if detail_steps is not None and detail_steps < self.detail_branch.num_timesteps:
            zD_tokens, zD_embed = self.detail_branch.sample_fast(
                zC_completed=zC_embed,
                zA_embeddings=zA_embeddings,
                y_features=y_features,
                num_tokens=num_tokens,
                degradation_params=deg_embed,
                positions=positions,
                num_steps=detail_steps,
                temperature=temperature,
            )
        else:
            zD_tokens, zD_embed = self.detail_branch.sample(
                zC_completed=zC_embed,
                zA_embeddings=zA_embeddings,
                y_features=y_features,
                num_tokens=num_tokens,
                degradation_params=deg_embed,
                positions=positions,
                temperature=temperature,
                progress=progress,
            )

        # Step 3: Combine and decode
        combined = self._combine_tokens(zC_embed, zD_embed, zA_embeddings, deg_enc["sparse"])
        dec_out = self.decoder(combined)

        reconstruction = dec_out["reconstruction"]

        # Remove temporal dimension if input was 4D
        if reconstruction.size(2) == 1:
            reconstruction = reconstruction.squeeze(2)

        return {
            "reconstruction": reconstruction,
            "zC_tokens": zC_tokens,
            "zD_tokens": zD_tokens,
            "zC_embed": zC_embed,
            "zD_embed": zD_embed,
        }

    def _combine_tokens(
        self,
        zC_embed: torch.Tensor,
        zD_embed: torch.Tensor,
        zA_embed: torch.Tensor,
        sparse_template: SparseTensor4D,
    ) -> SparseTensor4D:
        """Combine token embeddings for decoder.

        Args:
            zC_embed: Semantic embeddings (B, N, D).
            zD_embed: Detail embeddings (B, N, D).
            zA_embed: Artifact embeddings (B, N, D).
            sparse_template: Template for sparse tensor structure.

        Returns:
            Combined SparseTensor4D for decoder.
        """
        # Combine through learned projection
        combined_input = torch.cat([zC_embed, zD_embed, zA_embed], dim=-1)
        combined = self.token_combiner(combined_input)

        return SparseTensor4D(
            tokens=combined,
            positions=sparse_template.positions,
            mask=sparse_template.mask,
            metadata=sparse_template.metadata,
            weights=sparse_template.weights,
        )

    @torch.no_grad()
    def generate_unconditional(
        self,
        batch_size: int,
        resolution: Tuple[int, int] = (256, 256),
        num_frames: int = 1,
        device: Optional[torch.device] = None,
        semantic_steps: int = 10,
        detail_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate from scratch (unconditional).

        Args:
            batch_size: Number of samples.
            resolution: (H, W) resolution.
            num_frames: Number of frames (1 for image).
            device: Target device.
            semantic_steps: MaskGIT steps.
            detail_steps: D3PM steps.
            temperature: Sampling temperature.

        Returns:
            Generated tensor (B, C, H, W) or (B, C, T, H, W).
        """
        device = device or next(self.parameters()).device

        # Create noise input
        C = 3  # Assume RGB
        if num_frames > 1:
            y = torch.randn(batch_size, C, num_frames, *resolution, device=device)
        else:
            y = torch.randn(batch_size, C, *resolution, device=device)

        # Zero degradation params for unconditional
        deg_params = torch.zeros(batch_size, 32, device=device)

        result = self.generate(
            y_degraded=y,
            degradation_params=deg_params,
            semantic_steps=semantic_steps,
            detail_steps=detail_steps,
            temperature=temperature,
        )

        return result["reconstruction"]


__all__ = ["DiffusionGenerator"]
