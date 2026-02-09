"""Degradation pipeline for creating training pairs.

Applies various degradations to clean images/videos to create
degraded inputs for training the restoration model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass
class DegradationConfig:
    """Configuration for degradation pipeline."""

    # Blur
    blur_enabled: bool = True
    blur_kernel_range: Tuple[int, int] = (3, 21)
    blur_sigma_range: Tuple[float, float] = (0.1, 3.0)
    blur_prob: float = 0.5

    # Noise
    noise_enabled: bool = True
    noise_sigma_range: Tuple[float, float] = (0, 50)
    noise_prob: float = 0.5

    # JPEG compression
    jpeg_enabled: bool = True
    jpeg_quality_range: Tuple[int, int] = (10, 95)
    jpeg_prob: float = 0.3

    # Downsampling
    resize_enabled: bool = True
    resize_range: Tuple[float, float] = (0.25, 1.0)
    resize_prob: float = 0.4

    # Color jitter
    color_jitter_enabled: bool = True
    brightness_range: Tuple[float, float] = (-0.2, 0.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    color_prob: float = 0.3

    # Degradation param dimension
    param_dim: int = 32


def gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 2D Gaussian kernel."""
    x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = gauss[:, None] * gauss[None, :]
    return kernel / kernel.sum()


def apply_gaussian_blur(
    x: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """Apply Gaussian blur to tensor.

    Args:
        x: Input tensor (B, C, H, W) or (B, C, T, H, W).
        kernel_size: Blur kernel size (must be odd).
        sigma: Blur sigma.

    Returns:
        Blurred tensor.
    """
    is_video = x.dim() == 5

    if is_video:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    # Ensure odd kernel size
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    kernel = gaussian_kernel_2d(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.expand(x.size(1), 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    x_blurred = F.conv2d(x, kernel, padding=padding, groups=x.size(1))

    if is_video:
        x_blurred = x_blurred.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    return x_blurred


def apply_gaussian_noise(
    x: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Add Gaussian noise.

    Args:
        x: Input tensor (assumed to be in [0, 1]).
        sigma: Noise sigma (in [0, 255] scale).

    Returns:
        Noisy tensor.
    """
    noise = torch.randn_like(x) * (sigma / 255.0)
    return (x + noise).clamp(0, 1)


def apply_jpeg_compression(
    x: torch.Tensor,
    quality: int,
) -> torch.Tensor:
    """Simulate JPEG compression artifacts.

    This is a simplified differentiable approximation.
    For exact JPEG, use PIL/torchvision with actual encoding.

    Args:
        x: Input tensor (B, C, H, W) in [0, 1].
        quality: JPEG quality (0-100).

    Returns:
        Compressed tensor.
    """
    # Simplified JPEG simulation via quantization
    # Lower quality = more quantization = more artifacts

    # Scale factor based on quality
    if quality >= 90:
        factor = 1
    elif quality >= 70:
        factor = 2
    elif quality >= 50:
        factor = 4
    elif quality >= 30:
        factor = 8
    else:
        factor = 16

    # Quantize and dequantize
    levels = 256 // factor
    x_quantized = (x * levels).round() / levels

    # Add slight blur to simulate block artifacts
    if quality < 50:
        kernel = torch.ones(3, 1, 3, 3, device=x.device, dtype=x.dtype) / 9
        padding = 1
        x_quantized = F.conv2d(x_quantized, kernel, padding=padding, groups=3)

    return x_quantized.clamp(0, 1)


def apply_resize(
    x: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Downsample and upsample to original size.

    Args:
        x: Input tensor (B, C, H, W) or (B, C, T, H, W).
        scale: Downsampling scale (0, 1].

    Returns:
        Degraded tensor at original resolution.
    """
    if scale >= 1.0:
        return x

    is_video = x.dim() == 5

    if is_video:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        original_size = (H, W)
    else:
        original_size = x.shape[-2:]

    # Downsample
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    x_down = F.interpolate(x, size=new_size, mode="bilinear", align_corners=False)

    # Upsample back
    x_up = F.interpolate(x_down, size=original_size, mode="bilinear", align_corners=False)

    if is_video:
        x_up = x_up.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    return x_up


def apply_color_jitter(
    x: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
) -> torch.Tensor:
    """Apply color jittering.

    Args:
        x: Input tensor (B, C, H, W) in [0, 1].
        brightness: Brightness adjustment (-1, 1).
        contrast: Contrast multiplier (0, 2).
        saturation: Saturation multiplier (0, 2).

    Returns:
        Color-jittered tensor.
    """
    # Brightness
    x = x + brightness

    # Contrast
    mean = x.mean(dim=(-2, -1), keepdim=True)
    x = (x - mean) * contrast + mean

    # Saturation (simplified)
    if x.size(1) == 3:
        gray = x.mean(dim=1, keepdim=True)
        x = gray + (x - gray) * saturation

    return x.clamp(0, 1)


class DegradationPipeline:
    """Pipeline for applying degradations to create training pairs.

    Applies a random combination of degradations and returns
    both the degraded image and a parameter vector encoding
    the degradation settings.
    """

    def __init__(self, config: Optional[DegradationConfig] = None) -> None:
        """Initialize pipeline.

        Args:
            config: Degradation configuration.
        """
        self.config = config or DegradationConfig()

    def __call__(
        self,
        x_clean: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply degradations.

        Args:
            x_clean: Clean input tensor (B, C, H, W) or (B, C, T, H, W) in [0, 1].

        Returns:
            y_degraded: Degraded tensor.
            params: Degradation parameter vector (B, param_dim).
        """
        cfg = self.config
        B = x_clean.size(0)
        device = x_clean.device

        y = x_clean.clone()
        params_list = []

        # Blur
        if cfg.blur_enabled and random.random() < cfg.blur_prob:
            kernel_size = random.randint(*cfg.blur_kernel_range)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            sigma = random.uniform(*cfg.blur_sigma_range)
            y = apply_gaussian_blur(y, kernel_size, sigma)
            params_list.extend([
                1.0,  # blur enabled
                kernel_size / cfg.blur_kernel_range[1],
                sigma / cfg.blur_sigma_range[1],
            ])
        else:
            params_list.extend([0.0, 0.0, 0.0])

        # Noise
        if cfg.noise_enabled and random.random() < cfg.noise_prob:
            sigma = random.uniform(*cfg.noise_sigma_range)
            y = apply_gaussian_noise(y, sigma)
            params_list.extend([
                1.0,  # noise enabled
                sigma / cfg.noise_sigma_range[1],
            ])
        else:
            params_list.extend([0.0, 0.0])

        # JPEG
        if cfg.jpeg_enabled and random.random() < cfg.jpeg_prob:
            quality = random.randint(*cfg.jpeg_quality_range)
            # Handle video by processing frame-by-frame
            if y.dim() == 5:
                B, C, T, H, W = y.shape
                y = y.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                y = apply_jpeg_compression(y, quality)
                y = y.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            else:
                y = apply_jpeg_compression(y, quality)
            params_list.extend([
                1.0,  # jpeg enabled
                1.0 - quality / 100,  # Inverse quality (higher = more degradation)
            ])
        else:
            params_list.extend([0.0, 0.0])

        # Resize
        if cfg.resize_enabled and random.random() < cfg.resize_prob:
            scale = random.uniform(*cfg.resize_range)
            y = apply_resize(y, scale)
            params_list.extend([
                1.0,  # resize enabled
                1.0 - scale,  # Inverse scale
            ])
        else:
            params_list.extend([0.0, 0.0])

        # Color jitter
        if cfg.color_jitter_enabled and random.random() < cfg.color_prob:
            brightness = random.uniform(*cfg.brightness_range)
            contrast = random.uniform(*cfg.contrast_range)
            saturation = random.uniform(*cfg.saturation_range)
            # Handle video
            if y.dim() == 5:
                B, C, T, H, W = y.shape
                y = y.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                y = apply_color_jitter(y, brightness, contrast, saturation)
                y = y.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            else:
                y = apply_color_jitter(y, brightness, contrast, saturation)
            params_list.extend([
                1.0,  # color jitter enabled
                (brightness - cfg.brightness_range[0]) / (cfg.brightness_range[1] - cfg.brightness_range[0]),
                (contrast - cfg.contrast_range[0]) / (cfg.contrast_range[1] - cfg.contrast_range[0]),
                (saturation - cfg.saturation_range[0]) / (cfg.saturation_range[1] - cfg.saturation_range[0]),
            ])
        else:
            params_list.extend([0.0, 0.0, 0.0, 0.0])

        # Pad or truncate params to param_dim
        while len(params_list) < cfg.param_dim:
            params_list.append(0.0)
        params_list = params_list[:cfg.param_dim]

        # Create parameter tensor (same for all batch elements)
        params = torch.tensor(params_list, device=device, dtype=y.dtype)
        params = params.unsqueeze(0).expand(B, -1)

        return y, params

    def apply_specific(
        self,
        x_clean: torch.Tensor,
        degradation_type: str,
        strength: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply specific degradation type.

        Args:
            x_clean: Clean input.
            degradation_type: One of "blur", "noise", "jpeg", "resize", "sr" (super-res).
            strength: Degradation strength (0-1).

        Returns:
            Degraded tensor and parameters.
        """
        cfg = self.config
        B = x_clean.size(0)
        device = x_clean.device

        y = x_clean.clone()
        params_list = [0.0] * cfg.param_dim

        if degradation_type == "blur":
            kernel_size = int(cfg.blur_kernel_range[0] + strength * (cfg.blur_kernel_range[1] - cfg.blur_kernel_range[0]))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            sigma = cfg.blur_sigma_range[0] + strength * (cfg.blur_sigma_range[1] - cfg.blur_sigma_range[0])
            y = apply_gaussian_blur(y, kernel_size, sigma)
            params_list[0:3] = [1.0, kernel_size / cfg.blur_kernel_range[1], sigma / cfg.blur_sigma_range[1]]

        elif degradation_type == "noise":
            sigma = cfg.noise_sigma_range[0] + strength * (cfg.noise_sigma_range[1] - cfg.noise_sigma_range[0])
            y = apply_gaussian_noise(y, sigma)
            params_list[3:5] = [1.0, sigma / cfg.noise_sigma_range[1]]

        elif degradation_type == "jpeg":
            quality = int(cfg.jpeg_quality_range[1] - strength * (cfg.jpeg_quality_range[1] - cfg.jpeg_quality_range[0]))
            if y.dim() == 5:
                B, C, T, H, W = y.shape
                y = y.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                y = apply_jpeg_compression(y, quality)
                y = y.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            else:
                y = apply_jpeg_compression(y, quality)
            params_list[5:7] = [1.0, 1.0 - quality / 100]

        elif degradation_type in ["resize", "sr"]:
            scale = cfg.resize_range[1] - strength * (cfg.resize_range[1] - cfg.resize_range[0])
            y = apply_resize(y, scale)
            params_list[7:9] = [1.0, 1.0 - scale]

        params = torch.tensor(params_list, device=device, dtype=y.dtype)
        params = params.unsqueeze(0).expand(B, -1)

        return y, params


class VideoTemporalDegradation:
    """Additional temporal degradations for video."""

    @staticmethod
    def apply_frame_drop(
        x: torch.Tensor,
        drop_prob: float = 0.1,
    ) -> torch.Tensor:
        """Randomly drop frames and interpolate.

        Args:
            x: Video tensor (B, C, T, H, W).
            drop_prob: Probability of dropping each frame.

        Returns:
            Video with interpolated dropped frames.
        """
        B, C, T, H, W = x.shape

        # Create drop mask
        drop_mask = torch.rand(B, T, device=x.device) < drop_prob

        # Don't drop first or last frame
        drop_mask[:, 0] = False
        drop_mask[:, -1] = False

        result = x.clone()

        for b in range(B):
            for t in range(1, T - 1):
                if drop_mask[b, t]:
                    # Linear interpolation from neighbors
                    result[b, :, t] = (x[b, :, t - 1] + x[b, :, t + 1]) / 2

        return result

    @staticmethod
    def apply_temporal_noise(
        x: torch.Tensor,
        sigma: float = 10.0,
        correlation: float = 0.5,
    ) -> torch.Tensor:
        """Add temporally correlated noise.

        Args:
            x: Video tensor (B, C, T, H, W).
            sigma: Noise sigma.
            correlation: Temporal correlation factor.

        Returns:
            Noisy video.
        """
        B, C, T, H, W = x.shape

        # Generate base noise
        noise = torch.randn_like(x) * (sigma / 255.0)

        # Apply temporal correlation
        for t in range(1, T):
            noise[:, :, t] = correlation * noise[:, :, t - 1] + (1 - correlation) * noise[:, :, t]

        return (x + noise).clamp(0, 1)


__all__ = [
    "DegradationConfig",
    "DegradationPipeline",
    "VideoTemporalDegradation",
    "apply_gaussian_blur",
    "apply_gaussian_noise",
    "apply_jpeg_compression",
    "apply_resize",
    "apply_color_jitter",
]
