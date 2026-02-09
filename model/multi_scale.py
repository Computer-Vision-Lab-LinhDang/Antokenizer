"""Multi-scale autoregressive generation with D2O token merging.

Implements TAO's coarse-to-fine generation strategy:
    Scale 0: 2×2   (4 tokens)    - Global structure
    Scale 1: 4×4   (16 tokens)   - Layout
    Scale 2: 8×8   (64 tokens)   - Objects
    Scale 3: 16×16 (256 tokens)  - Details
    Scale 4: 32×32 (1024 tokens) - Fine details

At each scale transition, D2OTokenMerger decides whether to merge
cross-scale tokens based on information criteria (IC).

References:
    - TAO: "Token-All-Objects" multi-scale autoregression
    - D2O: Dynamic token merging (ICLR 2025)
    - dHT: Information criteria for tokenization (NeurIPS 2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from atoken.core.sparse_tensor import SparseTensor4D
from atoken.d2o import D2OTokenMerger, ScaleState, MultiScaleState, create_scale_state
from atoken.d2o.similarity import build_spatial_mapping

from .blocks import TransformerBlock


__all__ = [
    'ScaleConfig',
    'MultiScaleConfig',
    'ScaleEmbedding',
    'ScaleUpsampler',
    'MultiScaleARHead',
    'MultiScaleEncoder',
    'generate_4d_positions',  # Utility for position generation
]


# =============================================================================
# Utility Functions
# =============================================================================

def generate_4d_positions(
    B: int,
    H: int,
    W: int,
    device: torch.device,
    t: int = 0,
    z: int = 0,
) -> Tensor:
    """Generate 4D position tensor for RoPE.

    Creates positions in (t, z, y, x) format for RoPE4D.
    This is a shared utility to avoid code duplication.

    Args:
        B: Batch size
        H: Grid height
        W: Grid width
        device: Target device
        t: Temporal coordinate (default: 0 for images)
        z: Depth coordinate (default: 0 for 2D)

    Returns:
        Position tensor, shape [B, H*W, 4]
    """
    y = torch.arange(H, device=device)
    x = torch.arange(W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    positions = torch.stack([
        torch.full_like(yy, t),
        torch.full_like(yy, z),
        yy,
        xx,
    ], dim=-1)  # [H, W, 4]

    return positions.view(1, H * W, 4).expand(B, -1, -1)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ScaleConfig:
    """Configuration for a single scale level.

    Attributes:
        scale_idx: Index of this scale (0=coarsest, 4=finest)
        grid_size: (H, W) grid dimensions at this scale
        num_tokens: Total tokens (H * W)
        parent_scale: Index of parent scale (None for root)
    """
    scale_idx: int
    grid_size: Tuple[int, int]
    num_tokens: int
    parent_scale: Optional[int] = None

    @property
    def H(self) -> int:
        return self.grid_size[0]

    @property
    def W(self) -> int:
        return self.grid_size[1]


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale tokenization.

    Default configuration for 512×512 images with 16×16 patches:
        - Base grid: 32×32 tokens (finest scale)
        - 5 scales with 2× downsampling between each
    """
    scales: List[ScaleConfig]
    d_model: int = 1152
    num_heads: int = 16
    depth_per_scale: int = 2

    @classmethod
    def default_5_scales(cls, d_model: int = 1152) -> 'MultiScaleConfig':
        """Create default 5-scale configuration.

        Scales:
            0: 2×2   → 4 tokens
            1: 4×4   → 16 tokens
            2: 8×8   → 64 tokens
            3: 16×16 → 256 tokens
            4: 32×32 → 1024 tokens
        """
        scales = [
            ScaleConfig(0, (2, 2), 4, None),
            ScaleConfig(1, (4, 4), 16, 0),
            ScaleConfig(2, (8, 8), 64, 1),
            ScaleConfig(3, (16, 16), 256, 2),
            ScaleConfig(4, (32, 32), 1024, 3),
        ]
        return cls(scales=scales, d_model=d_model)

    @classmethod
    def for_resolution(
        cls,
        image_size: int,
        patch_size: int = 16,
        num_scales: int = 5,
        d_model: int = 1152,
    ) -> 'MultiScaleConfig':
        """Create configuration for arbitrary resolution.

        Args:
            image_size: Input image size (assumes square)
            patch_size: Patch size for finest scale
            num_scales: Number of scales
            d_model: Model dimension
        """
        finest_grid = image_size // patch_size
        scales = []

        for i in range(num_scales):
            # Compute grid size for this scale (from fine to coarse)
            scale_from_finest = num_scales - 1 - i
            grid_size = finest_grid // (2 ** scale_from_finest)
            grid_size = max(2, grid_size)  # Minimum 2×2

            parent = i - 1 if i > 0 else None
            scales.append(ScaleConfig(
                scale_idx=i,
                grid_size=(grid_size, grid_size),
                num_tokens=grid_size * grid_size,
                parent_scale=parent,
            ))

        return cls(scales=scales, d_model=d_model)

    @property
    def num_scales(self) -> int:
        return len(self.scales)

    @property
    def total_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scales)

    def get_scale(self, idx: int) -> ScaleConfig:
        return self.scales[idx]


class ScaleEmbedding(nn.Module):
    """Learnable scale embeddings for multi-scale tokens.

    Adds scale-specific information to token representations.
    """

    def __init__(
        self,
        num_scales: int,
        d_model: int,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.d_model = d_model

        if learnable:
            self.scale_embed = nn.Parameter(torch.zeros(num_scales, d_model))
            nn.init.normal_(self.scale_embed, std=0.02)
        else:
            # Fixed sinusoidal embeddings
            scale_embed = torch.zeros(num_scales, d_model)
            position = torch.arange(num_scales).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
            )
            scale_embed[:, 0::2] = torch.sin(position * div_term)
            scale_embed[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('scale_embed', scale_embed)

    def forward(self, x: Tensor, scale_idx: int) -> Tensor:
        """Add scale embedding to tokens.

        Args:
            x: Token features, shape [B, N, C] or [N, C]
            scale_idx: Scale index

        Returns:
            Tokens with scale embedding added
        """
        return x + self.scale_embed[scale_idx]


class ScaleUpsampler(nn.Module):
    """Upsample tokens from coarse to fine scale.

    Uses learned upsampling with optional nearest-neighbor initialization.
    Each coarse token expands to 4 fine tokens (2×2 spatial expansion).
    """

    def __init__(
        self,
        d_model: int,
        upsample_factor: int = 2,
        use_conv: bool = True,
    ) -> None:
        """Initialize upsampler.

        Args:
            d_model: Feature dimension
            upsample_factor: Spatial upsample factor (2 for 2×2)
            use_conv: Use conv-based upsampling (vs linear)
        """
        super().__init__()
        self.d_model = d_model
        self.upsample_factor = upsample_factor
        self.expansion = upsample_factor ** 2  # 4 for 2×2

        if use_conv:
            # Transposed conv for learnable upsampling
            self.upsample = nn.Sequential(
                nn.Linear(d_model, d_model * self.expansion),
                nn.GELU(),
                nn.Linear(d_model * self.expansion, d_model * self.expansion),
            )
        else:
            # Simple linear expansion
            self.upsample = nn.Linear(d_model, d_model * self.expansion)

        # Optional refinement after spatial rearrangement
        self.refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        x: Tensor,
        H_coarse: int,
        W_coarse: int,
    ) -> Tuple[Tensor, int, int]:
        """Upsample coarse tokens to fine resolution.

        Args:
            x: Coarse tokens, shape [B, N_coarse, C] where N_coarse = H_coarse * W_coarse
            H_coarse: Coarse grid height
            W_coarse: Coarse grid width

        Returns:
            fine_tokens: Shape [B, N_fine, C] where N_fine = N_coarse * 4
            H_fine: Fine grid height (H_coarse * 2)
            W_fine: Fine grid width (W_coarse * 2)
        """
        B, N, C = x.shape
        assert N == H_coarse * W_coarse, f"Token count {N} != grid {H_coarse}×{W_coarse}"

        # Expand each token to 4 tokens
        expanded = self.upsample(x)  # [B, N, C*4]

        # Reshape to spatial grid with expansion
        expanded = expanded.view(B, H_coarse, W_coarse, self.upsample_factor, self.upsample_factor, C)

        # Rearrange: [B, H, W, 2, 2, C] → [B, H*2, W*2, C]
        H_fine = H_coarse * self.upsample_factor
        W_fine = W_coarse * self.upsample_factor
        fine_tokens = expanded.permute(0, 1, 3, 2, 4, 5).reshape(B, H_fine * W_fine, C)

        # Refine
        fine_tokens = fine_tokens + self.refine(fine_tokens)

        return fine_tokens, H_fine, W_fine


class ScaleTransition(nn.Module):
    """Handles transition between scales with D2O merging.

    At each scale transition:
    1. Upsample coarse tokens to fine resolution
    2. Optionally merge with existing fine tokens using D2O
    3. Process through transformer blocks
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_d2o: bool = True,
        d2o_criterion: Literal['aic', 'bic', 'aicc', 'cic'] = 'bic',
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_d2o = use_d2o

        # Upsampler
        self.upsampler = ScaleUpsampler(d_model)

        # D2O merger
        if use_d2o:
            self.merger = D2OTokenMerger(
                d_model=d_model,
                num_scales=5,
                use_infocrit=True,
                criterion=d2o_criterion,
            )
        else:
            self.merger = None

        # Transformer blocks for processing at this scale
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        coarse_tokens: Tensor,
        coarse_state: Optional[ScaleState],
        fine_tokens: Optional[Tensor],
        fine_state: Optional[ScaleState],
        H_coarse: int,
        W_coarse: int,
        positions: Tensor,
        mask: Optional[Tensor] = None,
        scale_idx: int = 0,
    ) -> Tuple[Tensor, ScaleState, Dict]:
        """Transition from coarse to fine scale.

        Args:
            coarse_tokens: Coarse scale tokens [B, N_coarse, C]
            coarse_state: D2O state for coarse scale
            fine_tokens: Optional existing fine tokens (for merging)
            fine_state: D2O state for fine scale
            H_coarse, W_coarse: Coarse grid dimensions
            positions: Position encodings for transformer
            mask: Attention mask
            scale_idx: Current scale index

        Returns:
            output_tokens: Processed tokens at fine scale
            output_state: Updated D2O state
            info: Dict with merge statistics
        """
        B = coarse_tokens.shape[0]

        # Step 1: Upsample coarse to fine
        upsampled, H_fine, W_fine = self.upsampler(coarse_tokens, H_coarse, W_coarse)

        # Step 2: D2O merging (if enabled and fine tokens exist)
        merge_info = {'did_merge': False, 'ic_improvement': 0.0}

        if self.use_d2o and self.merger is not None and fine_tokens is not None:
            # Create states if not provided
            if coarse_state is None:
                coarse_state = create_scale_state(
                    coarse_tokens.view(-1, self.d_model),
                    H_coarse, W_coarse, scale_idx - 1,
                )
            if fine_state is None:
                fine_state = create_scale_state(
                    fine_tokens.view(-1, self.d_model),
                    H_fine, W_fine, scale_idx,
                )

            # Build mapping from fine to coarse
            mapping = build_spatial_mapping(
                H_fine, W_fine, H_coarse, W_coarse,
                device=coarse_tokens.device,
            )

            # Merge decision via D2O
            result = self.merger(fine_state, coarse_state, mapping)

            if result.did_merge:
                # Use merged features
                output_tokens = result.merged_state.features.view(B, -1, self.d_model)
                output_state = result.merged_state
            else:
                # Keep upsampled (refined coarse)
                output_tokens = upsampled
                output_state = create_scale_state(
                    upsampled.view(-1, self.d_model),
                    H_fine, W_fine, scale_idx,
                )

            merge_info = {
                'did_merge': result.did_merge,
                'ic_improvement': result.ic_improvement,
                'similarity_mean': result.similarity_mean,
            }
        else:
            # No merging, just use upsampled
            output_tokens = upsampled
            output_state = create_scale_state(
                upsampled.view(-1, self.d_model),
                H_fine, W_fine, scale_idx,
            )

        # Step 3: Process through transformer blocks
        # Need to ensure positions match fine scale token count
        N_fine = H_fine * W_fine
        if positions.shape[1] != N_fine:
            # Generate positions for fine scale
            positions = self._generate_positions(B, H_fine, W_fine, output_tokens.device)

        for block in self.blocks:
            output_tokens = block(output_tokens, positions=positions, mask=mask)

        output_tokens = self.norm(output_tokens)

        return output_tokens, output_state, merge_info

    def _generate_positions(
        self,
        B: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> Tensor:
        """Generate 4D position tensor for RoPE using shared utility."""
        return generate_4d_positions(B, H, W, device)


class MultiScaleARHead(nn.Module):
    """Multi-scale autoregressive generation head.

    Generates tokens from coarse to fine using TAO's scale autoregression:
    1. Start with global tokens (2×2)
    2. Progressively upsample and refine (4×4 → 8×8 → 16×16 → 32×32)
    3. At each transition, D2O decides whether to merge cross-scale tokens

    This implements TAO's "next-scale prediction" for visual generation.
    """

    def __init__(
        self,
        config: Optional[MultiScaleConfig] = None,
        d_model: int = 1152,
        num_heads: int = 16,
        depth_per_scale: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_d2o: bool = True,
        d2o_criterion: Literal['aic', 'bic', 'aicc', 'cic'] = 'bic',
    ) -> None:
        """Initialize multi-scale AR head.

        Args:
            config: Multi-scale configuration (default: 5 scales)
            d_model: Model dimension
            num_heads: Attention heads per scale
            depth_per_scale: Transformer depth per scale
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            use_d2o: Enable D2O token merging
            d2o_criterion: IC criterion for merging
        """
        super().__init__()

        self.config = config or MultiScaleConfig.default_5_scales(d_model)
        self.d_model = d_model
        self.use_d2o = use_d2o

        # Scale embeddings
        self.scale_embed = ScaleEmbedding(self.config.num_scales, d_model)

        # Initial coarse generation (from noise/conditioning)
        self.coarse_init = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * self.config.scales[0].num_tokens),
        )

        # Scale transitions (coarse → fine)
        self.transitions = nn.ModuleList([
            ScaleTransition(
                d_model=d_model,
                num_heads=num_heads,
                depth=depth_per_scale,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_d2o=use_d2o,
                d2o_criterion=d2o_criterion,
            )
            for _ in range(self.config.num_scales - 1)  # n-1 transitions for n scales
        ])

        # Final output projection
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        conditioning: Tensor,
        target_tokens: Optional[List[Tensor]] = None,
        positions_list: Optional[List[Tensor]] = None,
    ) -> Dict[str, Union[Tensor, List]]:
        """Forward pass for training.

        Args:
            conditioning: Global conditioning [B, C] or [B, N, C]
            target_tokens: Optional list of target tokens per scale (for supervision)
            positions_list: Optional list of positions per scale

        Returns:
            Dictionary with:
                - scale_outputs: List of outputs per scale
                - multi_state: Final MultiScaleState
                - merge_history: List of merge info per transition
        """
        B = conditioning.shape[0]
        device = conditioning.device

        # Handle different conditioning shapes
        if conditioning.dim() == 2:
            cond = conditioning  # [B, C]
        else:
            cond = conditioning.mean(dim=1)  # Pool to [B, C]

        # Initialize coarse scale
        coarse_tokens = self.coarse_init(cond)  # [B, num_tokens * C]
        scale_0 = self.config.scales[0]
        coarse_tokens = coarse_tokens.view(B, scale_0.num_tokens, self.d_model)

        # Add scale embedding
        coarse_tokens = self.scale_embed(coarse_tokens, 0)

        # Track outputs and states
        scale_outputs = [coarse_tokens]
        scale_states = []
        merge_history = []

        # Create initial state
        current_state = create_scale_state(
            coarse_tokens.view(-1, self.d_model),
            scale_0.H, scale_0.W, 0,
        )
        scale_states.append(current_state)

        current_tokens = coarse_tokens
        H_current, W_current = scale_0.H, scale_0.W

        # Progressive refinement through scales
        for i, transition in enumerate(self.transitions):
            scale_idx = i + 1
            scale_config = self.config.scales[scale_idx]

            # Get target tokens for this scale (if available)
            fine_target = None
            fine_state = None
            if target_tokens is not None and scale_idx < len(target_tokens):
                fine_target = target_tokens[scale_idx]
                fine_state = create_scale_state(
                    fine_target.view(-1, self.d_model),
                    scale_config.H, scale_config.W, scale_idx,
                )

            # Generate positions for this scale
            positions = self._generate_positions(B, scale_config.H, scale_config.W, device)
            if positions_list is not None and scale_idx < len(positions_list):
                positions = positions_list[scale_idx]

            # Transition to next scale
            output_tokens, output_state, merge_info = transition(
                coarse_tokens=current_tokens,
                coarse_state=current_state,
                fine_tokens=fine_target,
                fine_state=fine_state,
                H_coarse=H_current,
                W_coarse=W_current,
                positions=positions,
                scale_idx=scale_idx,
            )

            # Add scale embedding
            output_tokens = self.scale_embed(output_tokens, scale_idx)

            # Update tracking
            scale_outputs.append(output_tokens)
            scale_states.append(output_state)
            merge_history.append(merge_info)

            # Update for next iteration
            current_tokens = output_tokens
            current_state = output_state
            H_current, W_current = scale_config.H, scale_config.W

        # Final normalization
        final_output = self.output_norm(current_tokens)
        scale_outputs[-1] = final_output

        # Build multi-scale state
        multi_state = MultiScaleState(scales=scale_states)
        multi_state.merge_history = merge_history

        return {
            'scale_outputs': scale_outputs,
            'final_output': final_output,
            'multi_state': multi_state,
            'merge_history': merge_history,
        }

    @torch.no_grad()
    def generate(
        self,
        conditioning: Tensor,
        temperature: float = 1.0,
        return_all_scales: bool = False,
    ) -> Dict[str, Union[Tensor, List]]:
        """Generate tokens autoregressively from coarse to fine.

        Args:
            conditioning: Global conditioning [B, C]
            temperature: Sampling temperature (not used in deterministic mode)
            return_all_scales: Return outputs for all scales

        Returns:
            Dictionary with final tokens and optionally all scale outputs
        """
        result = self.forward(conditioning, target_tokens=None)

        output = {
            'tokens': result['final_output'],
            'merge_history': result['merge_history'],
        }

        if return_all_scales:
            output['scale_outputs'] = result['scale_outputs']
            output['multi_state'] = result['multi_state']

        return output

    def _generate_positions(
        self,
        B: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> Tensor:
        """Generate 4D position tensor using shared utility."""
        return generate_4d_positions(B, H, W, device)

    def get_d2o_stats(self) -> Dict:
        """Get D2O merging statistics across all transitions."""
        stats = {
            'total_merges': 0,
            'total_ic_improvement': 0.0,
            'per_transition': [],
        }

        for i, transition in enumerate(self.transitions):
            if transition.merger is not None:
                t_stats = transition.merger.get_stats()
                stats['total_merges'] += t_stats['merge_count']
                stats['total_ic_improvement'] += t_stats['total_ic_improvement']
                stats['per_transition'].append({
                    'transition': i,
                    **t_stats
                })

        return stats


class MultiScaleEncoder(nn.Module):
    """Encoder wrapper for multi-scale token extraction.

    Extracts tokens at multiple scales from input image/video.
    Used during training to get target tokens for each scale.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        config: Optional[MultiScaleConfig] = None,
        d_model: int = 1152,
    ) -> None:
        """Initialize multi-scale encoder.

        Args:
            base_encoder: Base ATokenEncoder
            config: Multi-scale configuration
            d_model: Model dimension
        """
        super().__init__()
        self.encoder = base_encoder
        self.config = config or MultiScaleConfig.default_5_scales(d_model)
        self.d_model = d_model

        # Pooling layers for each scale
        self.scale_poolers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((s.H, s.W))
            for s in self.config.scales
        ])

        # Projection to match encoder output
        self.scale_projs = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in self.config.scales
        ])

    def forward(
        self,
        x: Tensor,
        spatial_dims: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Extract multi-scale tokens.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, T, H, W]
            spatial_dims: Optional (H, W) tuple for explicit spatial dimensions.
                         If not provided, inferred from finest scale config or sqrt(N).

        Returns:
            Dictionary with:
                - encoder_output: Full encoder output
                - scale_tokens: List of tokens per scale
                - spatial_dims: (H, W) tuple of inferred/provided spatial dimensions
        """
        # Get full-resolution encoding
        enc_out = self.encoder(x, return_sparse=True)
        sequence = enc_out['sequence']  # [B, N, C]

        B, N, C = sequence.shape

        # Infer spatial dimensions with multiple strategies
        if spatial_dims is not None:
            # Strategy 1: Explicit dimensions provided
            H_full, W_full = spatial_dims
        elif hasattr(enc_out.get('sparse'), 'metadata') and enc_out['sparse'] is not None:
            # Strategy 2: Get from sparse tensor metadata
            metadata = enc_out['sparse'].metadata or {}
            H_full = metadata.get('H', None)
            W_full = metadata.get('W', None)
            if H_full is None or W_full is None:
                H_full = W_full = None
        else:
            H_full = W_full = None

        if H_full is None or W_full is None:
            # Strategy 3: Use finest scale from config as hint
            finest_scale = self.config.scales[-1]
            if N == finest_scale.num_tokens:
                H_full, W_full = finest_scale.H, finest_scale.W
            else:
                # Strategy 4: Assume square grid
                H_full = W_full = int(N ** 0.5)

        # Validate dimensions
        if H_full * W_full != N:
            raise ValueError(
                f"Spatial dimensions {H_full}×{W_full}={H_full*W_full} don't match "
                f"token count {N}. Provide explicit spatial_dims parameter."
            )

        # Reshape to spatial
        spatial = sequence.view(B, H_full, W_full, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Extract multi-scale tokens
        scale_tokens = []
        for i, (pooler, proj) in enumerate(zip(self.scale_poolers, self.scale_projs)):
            # Pool to scale resolution
            pooled = pooler(spatial)  # [B, C, H_s, W_s]

            # Reshape to sequence
            H_s, W_s = pooled.shape[2:]
            tokens = pooled.permute(0, 2, 3, 1).reshape(B, H_s * W_s, C)

            # Project
            tokens = proj(tokens)
            scale_tokens.append(tokens)

        return {
            'encoder_output': enc_out,
            'scale_tokens': scale_tokens,
            'sequence': sequence,
            'spatial_dims': (H_full, W_full),
        }
