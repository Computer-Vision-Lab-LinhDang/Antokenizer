"""Shallow feature encoder for extracting continuous features from degraded input.

Provides spatial anchoring for the diffusion process by encoding
low-level features from the degraded image/video.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: str = "batch",
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=norm == "none",
        )

        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "layer":
            self.norm = nn.GroupNorm(1, out_channels)
        elif norm == "group":
            self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        else:
            self.norm = nn.Identity()

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for shallow encoder."""

    def __init__(
        self,
        channels: int,
        norm: str = "batch",
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.conv1 = ConvBlock(channels, channels, norm=norm, activation=activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels) if norm == "batch" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm(self.conv2(x))
        return F.gelu(x + residual)


class ShallowFeatureEncoder(nn.Module):
    """Lightweight CNN encoder for extracting continuous features from degraded input.

    Provides spatial features that anchor the reconstruction process.
    The features capture low-level information (edges, textures) from
    the degraded input that complements the semantic tokens.

    Architecture:
    - Series of strided convolutions for downsampling
    - Optional residual blocks
    - Final projection to model dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 768,
        num_layers: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        use_residual: bool = True,
        norm: str = "batch",
        activation: str = "gelu",
    ) -> None:
        """Initialize shallow encoder.

        Args:
            in_channels: Input channels (3 for RGB).
            out_dim: Output embedding dimension.
            num_layers: Number of downsampling layers.
            base_channels: Initial channel count.
            max_channels: Maximum channel count.
            use_residual: Whether to use residual blocks.
            norm: Normalization type ("batch", "layer", "group", "none").
            activation: Activation function.
        """
        super().__init__()

        self.out_dim = out_dim

        # Build downsampling layers
        layers = []
        ch_in = in_channels

        for i in range(num_layers):
            ch_out = min(base_channels * (2 ** i), max_channels)

            # Strided conv for downsampling
            layers.append(
                ConvBlock(
                    ch_in, ch_out,
                    kernel_size=4 if i == 0 else 3,
                    stride=2,
                    padding=1,
                    norm=norm,
                    activation=activation,
                )
            )

            # Optional residual block
            if use_residual and i > 0:
                layers.append(ResidualBlock(ch_out, norm=norm, activation=activation))

            ch_in = ch_out

        self.backbone = nn.Sequential(*layers)

        # Final projection to model dimension
        self.proj = nn.Conv2d(ch_in, out_dim, kernel_size=1)

        # Layer norm on output
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Extract features from degraded input.

        Args:
            y: Degraded input.
               - Image: (B, C, H, W)
               - Video: (B, C, T, H, W)

        Returns:
            Spatial features (B, N, out_dim) where N = h * w (* T for video).
        """
        is_video = y.dim() == 5

        if is_video:
            B, C, T, H, W = y.shape
            # Reshape to process frames as batch
            y = y.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            B = y.size(0)
            T = 1

        # Extract features
        features = self.backbone(y)  # (B*T, ch, h, w)
        features = self.proj(features)  # (B*T, out_dim, h, w)

        _, D, h, w = features.shape

        if is_video:
            # Reshape back to video
            features = features.view(B, T, D, h, w)
            features = features.permute(0, 1, 3, 4, 2)  # (B, T, h, w, D)
            features = features.reshape(B, T * h * w, D)
        else:
            features = features.permute(0, 2, 3, 1)  # (B, h, w, D)
            features = features.reshape(B, h * w, D)

        # Normalize
        features = self.norm(features)

        return features

    def get_output_spatial_size(
        self,
        input_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Compute output spatial size for given input.

        Args:
            input_size: (H, W) of input.

        Returns:
            (h, w) of output features.
        """
        h, w = input_size
        for _ in range(len([m for m in self.backbone if isinstance(m, ConvBlock)])):
            h = (h + 1) // 2
            w = (w + 1) // 2
        return h, w


class ShallowViTEncoder(nn.Module):
    """Shallow ViT-style encoder as alternative to CNN.

    Uses non-overlapping patches with transformer layers
    for capturing spatial features.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 768,
        patch_size: int = 16,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize shallow ViT encoder.

        Args:
            in_channels: Input channels.
            out_dim: Output dimension.
            patch_size: Patch size for tokenization.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
            dropout: Dropout rate.
        """
        super().__init__()

        self.out_dim = out_dim
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, out_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=num_heads,
            dim_feedforward=int(out_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output norm
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Args:
            y: Input (B, C, H, W) or (B, C, T, H, W).

        Returns:
            Features (B, N, out_dim).
        """
        is_video = y.dim() == 5

        if is_video:
            B, C, T, H, W = y.shape
            y = y.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        else:
            B = y.size(0)
            T = 1

        # Patch embed
        x = self.patch_embed(y)  # (B*T, D, h, w)
        _, D, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)  # (B*T, h*w, D)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        if is_video:
            x = x.view(B, T, h * w, D)
            x = x.reshape(B, T * h * w, D)

        return x


class DegradationEncoder(nn.Module):
    """Encoder for degradation parameters.

    Encodes degradation type and parameters into a conditioning vector
    for the diffusion process.
    """

    def __init__(
        self,
        input_dim: int = 32,
        out_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        """Initialize degradation encoder.

        Args:
            input_dim: Dimension of degradation parameter vector.
            out_dim: Output embedding dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of MLP layers.
        """
        super().__init__()

        layers = []
        dim_in = input_dim

        for i in range(num_layers):
            dim_out = hidden_dim if i < num_layers - 1 else out_dim
            layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.SiLU() if i < num_layers - 1 else nn.Identity(),
            ])
            dim_in = dim_out

        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Encode degradation parameters.

        Args:
            params: Degradation parameter vector (B, input_dim).

        Returns:
            Degradation embedding (B, out_dim).
        """
        return self.net(params)


__all__ = [
    "ShallowFeatureEncoder",
    "ShallowViTEncoder",
    "DegradationEncoder",
]
