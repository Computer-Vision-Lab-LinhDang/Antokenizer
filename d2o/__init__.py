"""D2O: Dynamic Discriminative Operations for TAO Token Optimization.

This module implements IC-guided token merging for TAO's multi-scale
autoregressive generation, integrating:

    - dHT (NeurIPS 2025): Information criteria for merge quality assessment
    - D2O (ICLR 2025): Cross-scale KV cache optimization
    - TAO: Multi-scale unified tokenization

Core Components:
    - infocrit: Information criteria (AIC, BIC, AICc, CIC)
    - similarity: Gaussian, Cosine, and Hybrid similarity functions
    - state: Multi-scale state management
    - merger: IC-guided token merger

Quick Start:
    >>> from atoken.d2o import D2OTokenMerger, ScaleState, create_scale_state
    >>>
    >>> # Create merger
    >>> merger = D2OTokenMerger(
    ...     d_model=1152,
    ...     use_infocrit=True,
    ...     criterion='bic',
    ... )
    >>>
    >>> # Create scale states
    >>> fine_state = create_scale_state(fine_features, H=16, W=16, scale_idx=2)
    >>> coarse_state = create_scale_state(coarse_features, H=8, W=8, scale_idx=1)
    >>>
    >>> # Build mapping
    >>> from atoken.d2o.similarity import build_spatial_mapping
    >>> mapping = build_spatial_mapping(16, 16, 8, 8)
    >>>
    >>> # Merge
    >>> result = merger(fine_state, coarse_state, mapping)
    >>> if result.did_merge:
    ...     print(f"IC improved by {result.ic_improvement:.4f}")

Key Formulas:
    Information Criteria:
        IC = -NLL - penalty(k)
        NLL = n * log(2π * Π_c σ²_c + ε)
        k = 1 / (2n - 2√n)  # Degrees of freedom

    Hybrid Similarity:
        sim = α * exp(-||f_i - f_j||²) + β * cos(f_i, f_j)

    Merge Decision:
        Merge IF: IC(merged) > IC(fine) + IC(coarse)

References:
    - dHT Paper: "Hierarchical Token: A New Approach to Visual Tokenization"
    - D2O Paper: "D2O: Dynamic Discriminative Operations for Efficient Long-Context LLM Inference"
    - TAO Plan: .claude/plan/tao-unified-tokenizer.md
"""

# Information Criteria
from .infocrit import (
    gaussian_2nll,
    infocrit,
    infocrit_multi_scale,
    compare_ic_for_merge,
)

# Similarity Functions
from .similarity import (
    gaussian_similarity,
    cosine_similarity,
    hybrid_similarity,
    get_similarity_fn,
    compute_edge_features,
    compute_cross_scale_similarity,
    build_spatial_mapping,
)

# State Management
from .state import (
    ScaleState,
    MultiScaleState,
    create_scale_state,
    update_scale_state_after_merge,
    estimate_token_variance,
    update_params,
    compute_merged_statistics,
)

# Token Merger
from .merger import (
    D2OTokenMerger,
    MergeResult,
)

__all__ = [
    # infocrit
    'gaussian_2nll',
    'infocrit',
    'infocrit_multi_scale',
    'compare_ic_for_merge',
    # similarity
    'gaussian_similarity',
    'cosine_similarity',
    'hybrid_similarity',
    'get_similarity_fn',
    'compute_edge_features',
    'compute_cross_scale_similarity',
    'build_spatial_mapping',
    # state
    'ScaleState',
    'MultiScaleState',
    'create_scale_state',
    'update_scale_state_after_merge',
    'estimate_token_variance',
    'update_params',
    'compute_merged_statistics',
    # merger
    'D2OTokenMerger',
    'MergeResult',
]

__version__ = '1.0.0'
