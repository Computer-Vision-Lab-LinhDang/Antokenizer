"""Reconstruction Head for MAVT v2.

MAVT v2: Reconstruction branch operates on z_var ONLY.

Key changes from v1:
- Receives ONLY the variant component (z_var)
- z_var contains detail/motion with 2% of the energy
- Semantic gradients don't contaminate reconstruction
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    """Reconstruction head that operates on z_var only.

    This head receives ONLY the variant component (z_var),
    which contains detail/motion with ~2% of the energy.

    Benefits:
    - Reconstruction gradients don't mix with semantic
    - Clean gradient flow for pixel prediction
    """

    def __init__(
        self,
        latent_dim: int = 32,
        dec_dim: int = 768,
        patch_size: int = 16,
        num_heads: int = 16,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dec_dim = dec_dim
        self.patch_size = patch_size
        self.num_layers = num_layers

        # Project z_var to decoder dimension
        self.z_var_proj = nn.Linear(latent_dim, dec_dim)

        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(dec_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Output projection to pixels
        self.to_pixels = nn.Linear(dec_dim, patch_size * patch_size * 3)

    def forward(
        self,
        z_var: torch.Tensor,
        positions: torch.Tensor,
        grid_shape: Tuple[int, ...],
        latent_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_var: (B, N, K, D) or (B, N, D) variant features
            positions: (N, 4) or (B, N, 4) position embeddings
            grid_shape: (H, W) or (T, H, W) spatial grid
            latent_positions: optional latent positions for positional decoding

        Returns:
            reconstruction: (B, C, H, W) or (B, C, T, H, W)
        """
        # Handle different input shapes
        if len(z_var.shape) == 4:
            # (B, N, K, D) - with K dimension
            B, N, K, D = z_var.shape
            # Pool across K if needed
            z = z_var.mean(dim=2)  # (B, N, D)
        else:
            # (B, N, D)
            B, N, D = z_var.shape
            z = z_var

        # Project to decoder dimension
        z = self.z_var_proj(z)  # (B, N, dec_dim)

        # Run decoder blocks
        for block in self.blocks:
            z = block(z, positions)

        # Project to pixels
        pixels = self.to_pixels(z)  # (B, N, patch_size^2 * 3)

        # Reshape to image
        reconstruction = self._reshape_to_image(pixels, grid_shape)

        return reconstruction

    def _reshape_to_image(
        self,
        tokens: torch.Tensor,
        grid_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Reshape token predictions to image/video tensor.

        Args:
            tokens: (B, N, P) where P = patch_size^2 * 3
            grid_shape: (H, W) or (T, H, W)

        Returns:
            image: (B, 3, H*patch, W*patch) or (B, 3, T, H*patch, W*patch)
        """
        B = tokens.shape[0]
        patch_size_sq_3 = tokens.shape[-1]
        patch_pix = int(patch_pix ** 0.5) if (patch_size_sq_3 % 3 == 0) else int((patch_size_sq_3 // 3) ** 0.5)
        patch_pix = int((patch_size_sq_3 // 3) ** 0.5)
        p2 = patch_pix

        if len(grid_shape) == 2:
            H, W = grid_shape
            # Reshape: (B, N, P) -> (B, 3, H*p, W*p)
            tokens = tokens.reshape(B, H, W, 3, p2, p2)
            tokens = tokens.permute(0, 3, 1, 4, 2, 5)  # (B, 3, H, p, W, p)
            tokens = tokens.reshape(B, 3, H * p2, W * p2)
        elif len(grid_shape) == 3:
            T, H, W = grid_shape
            # Reshape: (B, N, P) -> (B, 3, T, H*p, W*p)
            N = T * H * W
            if N != tokens.shape[1]:
                # Assume N = H * W (for single-frame)
                tokens = tokens.reshape(B, H, W, 3, p2, p2)
                tokens = tokens.permute(0, 3, 1, 4, 2, 5)
                tokens = tokens.reshape(B, 3, H * p2, W * p2)
            else:
                tokens = tokens.reshape(B, T, H, W, 3, p2, p2)
                tokens = tokens.permute(0, 4, 1, 2, 5, 3, 6)  # (B, 3, T, H, p, W, p)
                tokens = tokens.reshape(B, 3, T, H * p2, W * p2)

        return tokens


class DecoderBlock(nn.Module):
    """Decoder block with cross-attention to latent positions.

    This block can optionally attend to content/detail tokens
    for better reconstruction.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        use_cross_attention: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) decoder input
            context: (B, M, D) context (from content/detail tokens)
            use_cross_attention: whether to use cross-attention

        Returns:
            output: (B, N, D)
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Cross-attention (if context provided)
        if use_cross_attention and context is not None:
            x = x + self.cross_attn(
                self.norm2(x),
                self.norm2(context),
                self.norm2(context),
            )[0]

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x


class AsymmetricDecoderV2(nn.Module):
    """Asymmetric decoder for MAVT v2.

    Decodes from z_var to pixel space with optional content guidance.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        dec_dim: int = 768,
        num_dec_attn_blocks: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dec_dim = dec_dim

        # Upsampling from latent dimension
        self.up = nn.Sequential(
            nn.Linear(latent_dim, dec_dim * 4),
            nn.GELU(),
            nn.Linear(dec_dim * 4, dec_dim),
        )

        # Decoder blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dec_dim,
                nhead=num_heads,
                dim_feedforward=int(dec_dim * mlp_ratio),
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_dec_attn_blocks)
        ])

        # To pixels
        self.to_pixel = nn.Linear(dec_dim, 16 * 16 * 3)

    def forward(
        self,
        z_var: torch.Tensor,
        z_inv: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_var: (B, N, K, D) or (B, N, D) variant features
            z_inv: (B, N, D) optional invariant for guidance
            grid_shape: (H, W) output grid shape

        Returns:
            reconstruction: (B, 3, H*16, W*16)
        """
        # Handle K dimension
        if len(z_var.shape) == 4:
            z_var = z_var.mean(dim=2)  # (B, N, D)

        # Upsample
        z = self.up(z_var)  # (B, N, dec_dim)

        # Optionally combine with z_inv
        if z_inv is not None:
            z = z + z_inv

        # Decoder blocks
        for block in self.blocks:
            z = block(z)

        # To pixels
        pixels = self.to_pixel(z)  # (B, N, 768)

        # Reshape
        B, N, P = pixels.shape
        p = int((P // 3) ** 0.5)
        H = W = int(N ** 0.5)

        pixels = pixels.reshape(B, H, W, 3, p, p)
        pixels = pixels.permute(0, 3, 1, 4, 2, 5)
        pixels = pixels.reshape(B, 3, H * p, W * p)

        return pixels


# ============================================================================
# Loss Functions
# ============================================================================

def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'mse',
) -> torch.Tensor:
    """Compute reconstruction loss.

    Args:
        recon: (B, C, H, W) or (B, C, T, H, W) predicted
        target: (B, C, H, W) or (B, C, T, H, W) ground truth
        loss_type: 'mse' | 'l1' | 'perceptual'

    Returns:
        loss: scalar
    """
    if loss_type == 'mse':
        return F.mse_loss(recon, target)

    elif loss_type == 'l1':
        return F.l1_loss(recon, target)

    elif loss_type == 'huber':
        return F.smooth_l1_loss(recon, target)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def combined_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    z_inv: torch.Tensor,
    alpha: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Combined reconstruction loss with detail preservation.

    Args:
        recon: (B, C, H, W) reconstruction
        target: (B, C, H, W) ground truth
        z_inv: (B, N, D) invariant features (for detail loss)
        alpha: weight for detail preservation

    Returns:
        loss: scalar
        metrics: dict with per-component losses
    """
    # Base reconstruction loss
    recon_loss = F.mse_loss(recon, target)

    # High-frequency detail loss (Sobel filter)
    def sobel(x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
        sobel_x = sobel_x.reshape(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        return grad_x.abs() + grad_y.abs()

    recon_edges = sobel(recon)
    target_edges = sobel(target)
    detail_loss = F.l1_loss(recon_edges, target_edges)

    # Total
    loss = recon_loss + alpha * detail_loss

    metrics = {
        'recon_loss': recon_loss.item(),
        'detail_loss': detail_loss.item(),
        'total_loss': loss.item(),
    }

    return loss, metrics
