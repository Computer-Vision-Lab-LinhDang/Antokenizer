from __future__ import annotations

"""Multi-scale state management for TAO token merging.

Adapted from dHT's TokenizerState with extensions for multi-scale AR generation.

Key concepts:
    - ScaleState: State for a single scale (features, variance, sizes, IC)
    - MultiScaleState: Aggregated state across all scales
    - Dual tracking: Current state + Optimal state (from dHT)

References:
    - dHT state.py: Dual state tracking for hierarchical tokenization
"""
import torch
from torch import Tensor
from typing import NamedTuple, Optional, Literal
from dataclasses import dataclass

__all__ = [
    'ScaleState',
    'MultiScaleState',
    'create_scale_state',
    'update_scale_state_after_merge',
    'estimate_token_variance',
    'update_params',
]


class ScaleState(NamedTuple):
    """State for a single scale in multi-scale tokenization.

    All tensors are on the same device and dtype.

    Attributes:
        features: Token features, shape [N, C]
        variance: Token variances, shape [N, C]
        sizes: Token sizes (pixel counts), shape [N]
        info: Information criteria scores, shape [N]
        H: Height of this scale's grid
        W: Width of this scale's grid
        scale_idx: Index of this scale (0=coarsest, 4=finest for TAO)
    """
    features: Tensor
    variance: Tensor
    sizes: Tensor
    info: Tensor
    H: int
    W: int
    scale_idx: int

    @property
    def n_tokens(self) -> int:
        """Number of tokens at this scale."""
        return self.features.shape[0]

    @property
    def dim(self) -> int:
        """Feature dimension."""
        return self.features.shape[1]

    def mean_ic(self) -> float:
        """Mean information criteria for this scale."""
        return self.info.mean().item()


@dataclass
class MultiScaleState:
    """State for all scales in TAO.

    Mutable container for tracking multi-scale tokenization progress.

    Attributes:
        scales: List of ScaleState for each scale
        cross_scale_similarity: Similarity between adjacent scales
        optimal_info: Best IC scores seen (for dual tracking)
        merge_history: History of merge decisions (for debugging)
    """
    scales: list[ScaleState]
    cross_scale_similarity: Optional[list[Tensor]] = None
    optimal_info: Optional[list[Tensor]] = None
    merge_history: Optional[list[dict]] = None

    def __post_init__(self):
        """Initialize optimal_info from current scales if not provided."""
        if self.optimal_info is None:
            self.optimal_info = [s.info.clone() for s in self.scales]

    @property
    def num_scales(self) -> int:
        """Number of scales."""
        return len(self.scales)

    def get_scale(self, idx: int) -> ScaleState:
        """Get state for a specific scale."""
        return self.scales[idx]

    def update_scale(self, idx: int, new_state: ScaleState) -> None:
        """Update state for a specific scale."""
        self.scales[idx] = new_state

    def total_tokens(self) -> int:
        """Total tokens across all scales."""
        return sum(s.n_tokens for s in self.scales)

    def total_ic(self) -> float:
        """Sum of mean IC across all scales."""
        return sum(s.mean_ic() for s in self.scales)


def create_scale_state(
    features: Tensor,
    H: int,
    W: int,
    scale_idx: int,
    criterion: Literal['aic', 'bic', 'aicc', 'cic'] = 'bic',
    iota: float = 1.0,
    eps: float = 1e-8,
    initial_variance: float = 0.01,
) -> ScaleState:
    """Initialize state for a single scale.

    Args:
        features: Token features, shape [N, C]
        H, W: Grid dimensions for this scale
        scale_idx: Scale index
        criterion: IC mode for computing info
        iota, eps: Stability constants
        initial_variance: Initial variance value (conservative default)

    Returns:
        Initialized ScaleState
    """
    from .infocrit import infocrit

    N, C = features.shape

    # Initialize variance to small positive value
    # Will be updated after first merge with actual computed variance
    variance = torch.full_like(features, initial_variance)

    # Assume uniform token sizes initially (each token = H*W/N pixels)
    sizes = torch.full(
        (N,), float(H * W) / N,
        dtype=features.dtype, device=features.device
    )

    # Compute initial IC
    info = infocrit(variance, sizes, H, W, criterion, iota, eps)

    return ScaleState(features, variance, sizes, info, H, W, scale_idx)


def estimate_token_variance(
    features: Tensor,
    method: Literal['uniform', 'feature_std', 'learnable'] = 'uniform',
    variance_proj: Optional[torch.nn.Module] = None,
    base_variance: float = 0.01,
) -> Tensor:
    """Estimate variance for discrete VQ tokens.

    Args:
        features: Token features, shape [N, C]
        method: Estimation method:
            - 'uniform': Constant variance (conservative)
            - 'feature_std': Use feature standard deviation
            - 'learnable': Use variance projection network
        variance_proj: Optional learned variance projection (for 'learnable')
        base_variance: Base variance value (for 'uniform')

    Returns:
        Estimated variance, shape [N, C]

    Note:
        For discrete VQ tokens, true variance depends on:
        - Distance to codebook centroid
        - Spatial variance within token's region
        - We use conservative estimates for numerical stability
    """
    N, C = features.shape

    if method == 'uniform':
        return torch.full_like(features, base_variance)

    elif method == 'feature_std':
        # Use channel-wise std as proxy for variance
        channel_std = features.std(dim=0, keepdim=True)  # [1, C]
        return channel_std.expand(N, C).contiguous() * base_variance

    elif method == 'learnable' and variance_proj is not None:
        # Use learned projection: features → variance
        # variance_proj: [N, C] → [N, C] or [N, 1]
        projected = variance_proj(features)
        # Ensure positive with softplus
        return torch.nn.functional.softplus(projected)

    else:
        # Fallback to uniform
        return torch.full_like(features, base_variance)


def update_params(
    cc: Tensor,
    fV: Tensor,
    s2: Tensor,
    mV: Tensor,
    nV: int,
    wV: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Update feature vectors, variances, and sizes after merging.

    Ported from dHT. This is the core parameter aggregation function.

    Args:
        cc: Connected component labels, shape [old_nV]
            cc[i] = new vertex index for old vertex i
        fV: Old vertex features, shape [old_nV, C]
        s2: Old vertex variances, shape [old_nV, C]
        mV: Old vertex sizes, shape [old_nV]
        nV: Number of new vertices after merge
        wV: Optional weights (from argmax edge similarities), shape [old_nV]
            If None, uses size-based weighting (theoretically correct)
            If provided, weights by sqrt(similarity)*size (gradient-aware)

    Returns:
        fVp: New features, shape [nV, C]
        s2p: New variances, shape [nV, C]
        mVp: New sizes, shape [nV]

    Formula (with wV):
        w_i = sqrt(wV_i) * mV_i
        μ_new = Σ(w_i * μ_i) / Σ(w_i)
        σ²_new = Σ(w_i * (σ²_i + (μ_i - μ_new)²)) / Σ(w_i)

    Note:
        The sqrt(wV) ensures balanced weighting between size and similarity.
        This reduces influence of outliers while maintaining gradient flow.
    """
    # New sizes: sum of merged region sizes
    mVp = _scatter_sum_2d(mV, cc, nV)

    # Compute weights
    if wV is not None:
        # Weight by sqrt(similarity) * size
        # sqrt provides balanced contribution
        num_w = wV.view(-1, 1).sqrt() * mV.view(-1, 1)
        den_w = _scatter_sum_2d(num_w, cc, nV)
    else:
        # Weight by size only (theoretically correct for equal features)
        num_w = mV.view(-1, 1)
        den_w = mVp.view(-1, 1)

    # Weighted mean: μ_new = Σ(w * μ) / Σ(w)
    fVp = _scatter_sum_2d(num_w * fV, cc, nV) / den_w

    # Weighted variance with mean shift
    # σ²_new = Σ(w * (σ² + (μ - μ_new)²)) / Σ(w)
    # The (μ - μ_new)² term accounts for variance introduced by merging
    s2p = _scatter_sum_2d(
        num_w * (s2 + (fV - fVp[cc]).pow(2)),
        cc, nV
    ) / den_w

    return fVp, s2p, mVp


def _scatter_sum_2d(
    src: Tensor,
    index: Tensor,
    dim_size: int,
) -> Tensor:
    """Scatter sum for 2D tensors along dimension 0.

    Args:
        src: Source tensor, shape [N, C] or [N]
        index: Index tensor, shape [N]
        dim_size: Output size

    Returns:
        Summed tensor, shape [dim_size, C] or [dim_size]
    """
    if src.dim() == 1:
        out = src.new_zeros(dim_size)
        return out.scatter_add_(0, index, src)
    else:
        C = src.shape[1]
        out = src.new_zeros(dim_size, C)
        index_expanded = index.unsqueeze(1).expand(-1, C)
        return out.scatter_add_(0, index_expanded, src)


def update_scale_state_after_merge(
    state: ScaleState,
    merged_features: Tensor,
    merged_sizes: Tensor,
    criterion: Literal['aic', 'bic', 'aicc', 'cic'] = 'bic',
    iota: float = 1.0,
    eps: float = 1e-8,
    variance_method: Literal['uniform', 'feature_std', 'learnable'] = 'uniform',
    variance_proj: Optional[torch.nn.Module] = None,
) -> ScaleState:
    """Update scale state after token merging.

    Args:
        state: Current scale state
        merged_features: New token features after merge
        merged_sizes: New token sizes
        criterion, iota, eps: IC parameters
        variance_method: How to estimate new variance
        variance_proj: Optional learned variance projection

    Returns:
        Updated ScaleState
    """
    from .infocrit import infocrit

    # Estimate new variance
    merged_variance = estimate_token_variance(
        merged_features, variance_method, variance_proj
    )

    # Compute new IC
    merged_info = infocrit(
        merged_variance, merged_sizes,
        state.H, state.W,
        criterion, iota, eps
    )

    return ScaleState(
        features=merged_features,
        variance=merged_variance,
        sizes=merged_sizes,
        info=merged_info,
        H=state.H,
        W=state.W,
        scale_idx=state.scale_idx,
    )


def compute_merged_statistics(
    fine_state: ScaleState,
    coarse_state: ScaleState,
    fine_to_coarse_map: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute hypothetical merged statistics for IC comparison.

    Args:
        fine_state: Fine scale state
        coarse_state: Coarse scale state
        fine_to_coarse_map: Mapping from fine to coarse tokens

    Returns:
        merged_features: Shape [N_fine, C]
        merged_variance: Shape [N_fine, C]
        merged_sizes: Shape [N_fine]

    Note:
        This computes what the statistics WOULD be if we merged,
        without actually performing the merge. Used for IC comparison.
    """
    # Get coarse parents
    coarse_parents = coarse_state.features[fine_to_coarse_map]
    coarse_sizes = coarse_state.sizes[fine_to_coarse_map]
    coarse_var = coarse_state.variance[fine_to_coarse_map]

    # Compute weights
    w_fine = fine_state.sizes.view(-1, 1)
    w_coarse = coarse_sizes.view(-1, 1)
    total_w = w_fine + w_coarse

    # Weighted mean
    merged_features = (w_fine * fine_state.features + w_coarse * coarse_parents) / total_w

    # Weighted variance with mean shift
    fine_diff_sq = (fine_state.features - merged_features).pow(2)
    coarse_diff_sq = (coarse_parents - merged_features).pow(2)
    merged_variance = (
        w_fine * (fine_state.variance + fine_diff_sq)
        + w_coarse * (coarse_var + coarse_diff_sq)
    ) / total_w

    # Merged sizes
    merged_sizes = fine_state.sizes + coarse_sizes

    return merged_features, merged_variance, merged_sizes
