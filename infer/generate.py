"""Inference pipeline for the diffusion generator.

Provides a high-level interface for image and video generation/restoration.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from atoken.diffusion.generator import DiffusionGenerator
from atoken.train.degradation import DegradationPipeline


class DiffusionInference:
    """Inference wrapper for the diffusion generator.

    Provides convenient methods for:
    - Image/video restoration (deblurring, denoising, super-resolution)
    - Unconditional generation
    - Controllable generation with degradation parameters
    """

    def __init__(
        self,
        generator: DiffusionGenerator,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        compile_model: bool = False,
    ) -> None:
        """Initialize inference wrapper.

        Args:
            generator: Trained DiffusionGenerator model.
            device: Target device.
            dtype: Data type for inference.
            compile_model: Whether to compile with torch.compile.
        """
        self.generator = generator
        self.device = device or next(generator.parameters()).device
        self.dtype = dtype

        self.generator = self.generator.to(self.device, dtype=self.dtype)
        self.generator.eval()

        if compile_model and hasattr(torch, "compile"):
            self.generator = torch.compile(self.generator)

        # Initialize degradation estimator
        self.degradation_pipeline = DegradationPipeline()

    @torch.no_grad()
    def restore(
        self,
        degraded: torch.Tensor,
        degradation_type: Literal["auto", "blur", "noise", "jpeg", "sr", "mixed"] = "auto",
        degradation_strength: float = 0.5,
        semantic_steps: int = 10,
        detail_steps: int = 50,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Restore degraded image or video.

        Args:
            degraded: Degraded input (B, C, H, W) or (B, C, T, H, W) in [0, 1].
            degradation_type: Type of degradation or "auto" to estimate.
            degradation_strength: Strength for non-auto mode.
            semantic_steps: MaskGIT decoding steps.
            detail_steps: D3PM sampling steps.
            temperature: Sampling temperature.
            batch_size: Process in batches if input is large.

        Returns:
            Restored tensor with same shape as input.
        """
        degraded = degraded.to(self.device, dtype=self.dtype)

        # Get degradation parameters
        if degradation_type == "auto":
            deg_params = self._estimate_degradation(degraded)
        elif degradation_type == "mixed":
            deg_params = torch.zeros(
                degraded.size(0), 32, device=self.device, dtype=self.dtype
            )
        else:
            _, deg_params = self.degradation_pipeline.apply_specific(
                degraded, degradation_type, degradation_strength
            )
            # Reset to degraded (we just want params)
            deg_params = deg_params.to(self.device, dtype=self.dtype)

        # Process in batches if needed
        if batch_size is not None and degraded.size(0) > batch_size:
            return self._batch_process(
                degraded, deg_params,
                semantic_steps, detail_steps, temperature,
                batch_size,
            )

        # Generate
        result = self.generator.generate(
            y_degraded=degraded,
            degradation_params=deg_params,
            semantic_steps=semantic_steps,
            detail_steps=detail_steps,
            temperature=temperature,
        )

        return result["reconstruction"]

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        resolution: Tuple[int, int] = (256, 256),
        num_frames: int = 1,
        semantic_steps: int = 10,
        detail_steps: int = 50,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate images or videos unconditionally.

        Args:
            batch_size: Number of samples to generate.
            resolution: (H, W) resolution.
            num_frames: Number of frames (1 for image).
            semantic_steps: MaskGIT steps.
            detail_steps: D3PM steps.
            temperature: Sampling temperature.
            guidance_scale: Classifier-free guidance scale.

        Returns:
            Generated tensor (B, C, H, W) or (B, C, T, H, W).
        """
        return self.generator.generate_unconditional(
            batch_size=batch_size,
            resolution=resolution,
            num_frames=num_frames,
            device=self.device,
            semantic_steps=semantic_steps,
            detail_steps=detail_steps,
            temperature=temperature,
        )

    @torch.no_grad()
    def super_resolve(
        self,
        low_res: torch.Tensor,
        scale_factor: int = 4,
        semantic_steps: int = 10,
        detail_steps: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Super-resolve low-resolution input.

        Args:
            low_res: Low-resolution input (B, C, H, W) or (B, C, T, H, W).
            scale_factor: Upscaling factor.
            semantic_steps: MaskGIT steps.
            detail_steps: D3PM steps.
            temperature: Sampling temperature.

        Returns:
            High-resolution output.
        """
        # Upsample to target resolution
        if low_res.dim() == 4:
            target_size = (low_res.size(2) * scale_factor, low_res.size(3) * scale_factor)
            upsampled = F.interpolate(low_res, size=target_size, mode="bilinear", align_corners=False)
        else:
            B, C, T, H, W = low_res.shape
            target_size = (H * scale_factor, W * scale_factor)
            low_res_2d = low_res.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            upsampled = F.interpolate(low_res_2d, size=target_size, mode="bilinear", align_corners=False)
            upsampled = upsampled.reshape(B, T, C, *target_size).permute(0, 2, 1, 3, 4)

        # Set degradation params for super-resolution
        deg_params = torch.zeros(low_res.size(0), 32, device=self.device, dtype=self.dtype)
        deg_params[:, 7] = 1.0  # resize enabled
        deg_params[:, 8] = 1.0 - 1.0 / scale_factor  # scale factor

        return self.restore(
            upsampled.to(self.device, dtype=self.dtype),
            degradation_type="mixed",  # Use provided params
            semantic_steps=semantic_steps,
            detail_steps=detail_steps,
            temperature=temperature,
        )

    @torch.no_grad()
    def denoise(
        self,
        noisy: torch.Tensor,
        noise_level: Optional[float] = None,
        semantic_steps: int = 10,
        detail_steps: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Denoise noisy input.

        Args:
            noisy: Noisy input tensor.
            noise_level: Estimated noise level (0-1), auto-estimated if None.
            semantic_steps: MaskGIT steps.
            detail_steps: D3PM steps.
            temperature: Sampling temperature.

        Returns:
            Denoised tensor.
        """
        if noise_level is None:
            # Estimate noise level from input
            noise_level = self._estimate_noise_level(noisy)

        deg_params = torch.zeros(noisy.size(0), 32, device=self.device, dtype=self.dtype)
        deg_params[:, 3] = 1.0  # noise enabled
        deg_params[:, 4] = noise_level  # noise level

        return self.restore(
            noisy,
            degradation_type="mixed",
            semantic_steps=semantic_steps,
            detail_steps=detail_steps,
            temperature=temperature,
        )

    @torch.no_grad()
    def deblur(
        self,
        blurry: torch.Tensor,
        blur_strength: Optional[float] = None,
        semantic_steps: int = 10,
        detail_steps: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Deblur blurry input.

        Args:
            blurry: Blurry input tensor.
            blur_strength: Estimated blur strength (0-1), auto-estimated if None.
            semantic_steps: MaskGIT steps.
            detail_steps: D3PM steps.
            temperature: Sampling temperature.

        Returns:
            Deblurred tensor.
        """
        if blur_strength is None:
            blur_strength = 0.5  # Default mid-level blur

        deg_params = torch.zeros(blurry.size(0), 32, device=self.device, dtype=self.dtype)
        deg_params[:, 0] = 1.0  # blur enabled
        deg_params[:, 1] = blur_strength  # kernel size
        deg_params[:, 2] = blur_strength  # sigma

        return self.restore(
            blurry,
            degradation_type="mixed",
            semantic_steps=semantic_steps,
            detail_steps=detail_steps,
            temperature=temperature,
        )

    def _estimate_degradation(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate degradation parameters from input.

        Simple heuristics for degradation estimation.
        In practice, could use a learned estimator network.
        """
        B = x.size(0)
        params = torch.zeros(B, 32, device=self.device, dtype=self.dtype)

        # Estimate blur from high-frequency content
        blur_level = self._estimate_blur_level(x)
        params[:, 0] = (blur_level > 0.1).float()
        params[:, 1] = blur_level
        params[:, 2] = blur_level

        # Estimate noise from local variance
        noise_level = self._estimate_noise_level(x)
        params[:, 3] = (noise_level > 0.05).float()
        params[:, 4] = noise_level

        return params

    def _estimate_blur_level(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate blur level from high-frequency content."""
        if x.dim() == 5:
            x = x[:, :, 0]  # Use first frame

        # Laplacian variance as blur metric
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)

        # Apply to grayscale
        gray = x.mean(dim=1, keepdim=True)
        laplacian = F.conv2d(gray, laplacian_kernel, padding=1)
        variance = laplacian.var(dim=(-2, -1)).squeeze()

        # Normalize: higher variance = less blur
        # Invert to get blur level
        blur_level = 1.0 - torch.clamp(variance / 0.1, 0, 1)

        return blur_level

    def _estimate_noise_level(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate noise level from local variance."""
        if x.dim() == 5:
            x = x[:, :, 0]

        # Use median absolute deviation for robust noise estimation
        # Simplified version using small patch variance
        B, C, H, W = x.shape

        # Unfold into patches
        patch_size = 8
        patches = F.unfold(x, patch_size, stride=patch_size)  # (B, C*64, num_patches)

        # Compute variance of each patch
        patch_var = patches.var(dim=1)  # (B, num_patches)

        # Use minimum variance patches (likely flat regions)
        k = max(1, patch_var.size(1) // 10)
        min_vars, _ = patch_var.topk(k, dim=1, largest=False)

        # Average of minimum variances as noise estimate
        noise_level = min_vars.mean(dim=1).sqrt()

        # Normalize
        noise_level = torch.clamp(noise_level / 0.1, 0, 1)

        return noise_level

    def _batch_process(
        self,
        degraded: torch.Tensor,
        deg_params: torch.Tensor,
        semantic_steps: int,
        detail_steps: int,
        temperature: float,
        batch_size: int,
    ) -> torch.Tensor:
        """Process large input in batches."""
        results = []
        total = degraded.size(0)

        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch_deg = degraded[i:end]
            batch_params = deg_params[i:end]

            result = self.generator.generate(
                y_degraded=batch_deg,
                degradation_params=batch_params,
                semantic_steps=semantic_steps,
                detail_steps=detail_steps,
                temperature=temperature,
            )
            results.append(result["reconstruction"])

        return torch.cat(results, dim=0)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        if "generator" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator"])
        elif "model" in checkpoint:
            self.generator.load_state_dict(checkpoint["model"])
        elif "task" in checkpoint:
            # Handle task checkpoint format
            state_dict = checkpoint["task"]
            # Remove 'generator.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("generator."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            self.generator.load_state_dict(new_state_dict, strict=False)
        else:
            self.generator.load_state_dict(checkpoint)

        self.generator.eval()


class VideoInference(DiffusionInference):
    """Extended inference for video with temporal consistency."""

    @torch.no_grad()
    def restore_video(
        self,
        video: torch.Tensor,
        degradation_type: str = "auto",
        semantic_steps: int = 10,
        detail_steps: int = 50,
        temperature: float = 1.0,
        overlap: int = 4,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Restore long video with chunked processing.

        Args:
            video: Input video (B, C, T, H, W).
            degradation_type: Degradation type.
            semantic_steps: MaskGIT steps.
            detail_steps: D3PM steps.
            temperature: Sampling temperature.
            overlap: Overlap between chunks for blending.
            chunk_size: Number of frames per chunk.

        Returns:
            Restored video.
        """
        B, C, T, H, W = video.shape
        video = video.to(self.device, dtype=self.dtype)

        if T <= chunk_size:
            return self.restore(
                video, degradation_type,
                semantic_steps=semantic_steps,
                detail_steps=detail_steps,
                temperature=temperature,
            )

        # Process in overlapping chunks
        results = []
        weights = []

        for start in range(0, T, chunk_size - overlap):
            end = min(start + chunk_size, T)
            chunk = video[:, :, start:end]

            # Restore chunk
            restored_chunk = self.restore(
                chunk, degradation_type,
                semantic_steps=semantic_steps,
                detail_steps=detail_steps,
                temperature=temperature,
            )

            results.append((start, end, restored_chunk))

        # Blend overlapping regions
        output = torch.zeros_like(video)
        weight_sum = torch.zeros(1, 1, T, 1, 1, device=self.device, dtype=self.dtype)

        for start, end, chunk in results:
            chunk_len = end - start

            # Create blending weights (linear ramp at edges)
            w = torch.ones(chunk_len, device=self.device, dtype=self.dtype)
            if start > 0:
                w[:overlap] = torch.linspace(0, 1, overlap, device=self.device)
            if end < T:
                w[-overlap:] = torch.linspace(1, 0, overlap, device=self.device)

            w = w.view(1, 1, chunk_len, 1, 1)
            output[:, :, start:end] += chunk * w
            weight_sum[:, :, start:end] += w

        output = output / weight_sum.clamp(min=1e-8)

        return output


__all__ = ["DiffusionInference", "VideoInference"]
