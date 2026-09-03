"""Semantic Head for MAVT v2.

MAVT v2: Semantic branch operates on z_inv ONLY.

Key changes from v1:
- No gradient contamination from reconstruction loss
- JEPA-style masked prediction (optional)
- SigLIP2 alignment (optional)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticHead(nn.Module):
    """Semantic head that operates on z_inv only.

    This head receives ONLY the invariant component (z_inv),
    which contains semantic content with 98% of the energy.

    Benefits:
    - Semantic gradients don't mix with reconstruction
    - JEPA-style masked prediction is natural here
    """

    def __init__(
        self,
        latent_dim: int = 32,
        semantic_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.semantic_dim = semantic_dim

        # Project z_inv to semantic space
        self.proj = nn.Linear(latent_dim, semantic_dim)

        # Optional: JEPA-style predictor for masked prediction
        self.predictor = JEPAPredictor(
            dim=semantic_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        z_inv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_jepa: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            z_inv: (B, N, D) or (B, D) invariant features
            mask: (B, N) optional mask for JEPA-style prediction
            use_jepa: whether to use JEPA predictor

        Returns:
            semantic: (B, semantic_dim) semantic representation
        """
        # Handle different input shapes
        if len(z_inv.shape) == 2:
            # (B, D) - global features
            z = z_inv
        elif len(z_inv.shape) == 3:
            # (B, N, D) - patch features, pool to global
            z = z_inv.mean(dim=1)  # (B, D)
        else:
            raise ValueError(f"Unexpected z_inv shape: {z_inv.shape}")

        # Project to semantic space
        semantic = self.proj(z)  # (B, semantic_dim)

        if use_jepa and mask is not None:
            # JEPA-style: predict masked regions
            semantic = self.predictor(semantic, z_inv, mask)

        return semantic


class JEPAPredictor(nn.Module):
    """JEPA-style predictor for masked prediction.

    From V-JEPA and UniJEPA:
    - Predict representations for masked regions
    - Only from visible (unmasked) regions
    - In latent space (not pixel space)
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        mask_token_dim: int = 32,
    ):
        super().__init__()
        self.dim = dim

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Predictor blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        visible_emb: torch.Tensor,
        all_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visible_emb: (B, D) global embedding from visible patches
            all_emb: (B, N, D) all patch embeddings
            mask: (B, N) boolean mask (True = masked)

        Returns:
            predicted: (B, N, D) predicted representations for masked patches
        """
        B, N, D = all_emb.shape

        # Create masked input
        masked_emb = all_emb.clone()
        masked_emb[mask] = self.mask_token

        # Run predictor blocks
        x = masked_emb
        for block in self.blocks:
            x = block(x)

        # Predict only masked regions
        predicted = self.norm(x)

        return predicted


class TransformerBlock(nn.Module):
    """Simple transformer block for JEPA predictor."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SigLIPAlignment(nn.Module):
    """Optional: Align semantic representations with SigLIP2.

    This can be used to align the learned semantic space with
    a pre-trained SigLIP2 model for better zero-shot transfer.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        teacher_dim: int = 768,
    ):
        super().__init__()
        # Project from latent space to teacher space
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, teacher_dim),
        )

        # Temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

    def forward(
        self,
        z_inv: torch.Tensor,
        teacher_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_inv: (B, latent_dim) learned invariant features
            teacher_emb: (B, teacher_dim) frozen teacher features

        Returns:
            loss: contrastive loss
            z_proj: projected features
            logits: similarity logits
        """
        # Project to teacher space
        z_proj = self.proj(z_inv)  # (B, teacher_dim)

        # Normalize
        z_norm = F.normalize(z_proj, dim=-1)
        t_norm = F.normalize(teacher_emb, dim=-1)

        # Cosine similarity
        logits = (z_norm @ t_norm.T) * self.logit_scale.exp()  # (B, B)

        # Labels (diagonal)
        labels = torch.arange(len(logits), device=logits.device)

        # Contrastive loss
        loss = F.cross_entropy(logits, labels)

        return loss, z_proj, logits


# ============================================================================
# Loss Functions
# ============================================================================

def semantic_loss(
    z_inv: torch.Tensor,
    teacher_emb: torch.Tensor,
    loss_type: str = 'mse',
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute semantic loss from z_inv.

    Args:
        z_inv: (B, D) or (B, N, D) invariant features
        teacher_emb: (B, D) frozen teacher embeddings
        loss_type: 'mse' | 'cosine' | 'contrastive'
        temperature: for contrastive loss

    Returns:
        loss: scalar loss
    """
    # Pool if needed
    if len(z_inv.shape) == 3:
        z = z_inv.mean(dim=1)  # (B, D)
    else:
        z = z_inv

    if loss_type == 'mse':
        return F.mse_loss(z, teacher_emb)

    elif loss_type == 'cosine':
        # Negative cosine similarity
        z_norm = F.normalize(z, dim=-1)
        t_norm = F.normalize(teacher_emb, dim=-1)
        return 1 - (z_norm * t_norm).sum(dim=-1).mean()

    elif loss_type == 'contrastive':
        z_norm = F.normalize(z, dim=-1)
        t_norm = F.normalize(teacher_emb, dim=-1)
        logits = z_norm @ t_norm.T / temperature
        labels = torch.arange(len(logits), device=logits.device)
        return F.cross_entropy(logits, labels)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
