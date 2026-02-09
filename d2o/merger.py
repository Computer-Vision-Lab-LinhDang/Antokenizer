from __future__ import annotations

"""IC-guided token merger for TAO D2O optimization.

Integrates:
    - dHT: Information criteria for merge quality assessment
    - D2O: Cross-scale KV cache optimization
    - TAO: Multi-scale autoregressive generation

Core Logic:
    Merge tokens IF: IC(merged) > IC(fine) + IC(coarse)

    This ensures merging only occurs when it improves
    the overall information representation.

References:
    - dHT (NeurIPS 2025): Information criteria-based tokenization
    - D2O (ICLR 2025): Dynamic KV cache optimization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal, NamedTuple

from .infocrit import infocrit, compare_ic_for_merge
from .similarity import compute_cross_scale_similarity, build_spatial_mapping
from .state import (
    ScaleState, MultiScaleState,
    create_scale_state, update_scale_state_after_merge,
    compute_merged_statistics, estimate_token_variance,
)

__all__ = [
    'D2OTokenMerger',
    'MergeResult',
]


class MergeResult(NamedTuple):
    """Result of a merge operation.

    Attributes:
        merged_state: Updated scale state after merge (or original if no merge)
        did_merge: Whether merging occurred
        ic_improvement: IC improvement from merging (negative if no improvement)
        similarity_mean: Mean similarity between merged tokens
    """
    merged_state: ScaleState
    did_merge: bool
    ic_improvement: float
    similarity_mean: float


class D2OTokenMerger(nn.Module):
    """Cross-scale token merging with dHT-style information criteria.

    This module implements IC-guided token merging for TAO's multi-scale
    autoregressive generation. It combines:
        - Gaussian similarity (content-based, from dHT)
        - Cosine similarity (semantic-based, from D2O)
        - Information criteria (merge quality assessment)

    Key Features:
        1. IC-guided merge decisions (provable BIC consistency)
        2. Hybrid similarity (Gaussian + Cosine)
        3. Learnable variance projection (per-scale)
        4. Configurable fallback to threshold-based merging

    Example:
        >>> merger = D2OTokenMerger(
        ...     d_model=1152,
        ...     use_infocrit=True,
        ...     criterion='bic',
        ... )
        >>> result = merger(fine_state, coarse_state, mapping)
        >>> if result.did_merge:
        ...     fine_state = result.merged_state
    """

    def __init__(
        self,
        d_model: int = 1152,
        num_scales: int = 5,
        similarity_threshold: float = 0.75,
        merge_ratio: float = 0.9999,
        use_infocrit: bool = True,
        criterion: Literal['aic', 'bic', 'aicc', 'cic'] = 'bic',
        alpha: float = 0.5,
        beta: float = 0.5,
        iota: float = 1.0,
        eps: float = 1e-8,
        use_variance_proj: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            d_model: Feature dimension (default 1152 for AToken)
            num_scales: Number of scales (default 5 for TAO)
            similarity_threshold: Min similarity for threshold-based merging
            merge_ratio: Target merge ratio (1.0 = no merging)
            use_infocrit: Use IC-guided merging (True) vs threshold (False)
            criterion: IC mode ('aic', 'bic', 'aicc', 'cic')
            alpha: Weight for Gaussian similarity
            beta: Weight for Cosine similarity
            iota: NLL stability constant
            eps: Division stability constant
            use_variance_proj: Use learnable variance projection
            debug: Enable debug logging
        """
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.similarity_threshold = similarity_threshold
        self.merge_ratio = merge_ratio
        self.use_infocrit = use_infocrit
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta
        self.iota = iota
        self.eps = eps
        self.debug = debug

        # Learnable variance projection (per-scale)
        # Maps hidden features → merge likelihood for self-edges
        if use_variance_proj:
            self.variance_proj = nn.ModuleList([
                nn.Linear(d_model, 1) for _ in range(num_scales)
            ])
            # Initialize with small weights, negative bias
            # Negative bias → resist merging by default
            for proj in self.variance_proj:
                proj.weight.data = torch.randn(1, d_model) * 1e-4
                proj.bias.data = -torch.ones(1) * 2 * torch.pi
        else:
            self.variance_proj = None

        # Statistics tracking
        self.register_buffer('merge_count', torch.tensor(0))
        self.register_buffer('total_ic_improvement', torch.tensor(0.0))

    def get_variance_proj(self, scale_idx: int) -> Optional[nn.Module]:
        """Get variance projection for a specific scale."""
        if self.variance_proj is None:
            return None
        return self.variance_proj[min(scale_idx, len(self.variance_proj) - 1)]

    def compute_hybrid_similarity(
        self,
        fine_features: Tensor,
        coarse_features: Tensor,
        fine_to_coarse_map: Tensor,
    ) -> Tensor:
        """Compute hybrid Gaussian + Cosine similarity.

        Args:
            fine_features: Shape [N_fine, C]
            coarse_features: Shape [N_coarse, C]
            fine_to_coarse_map: Shape [N_fine]

        Returns:
            Similarity scores, shape [N_fine]
        """
        return compute_cross_scale_similarity(
            fine_features,
            coarse_features,
            fine_to_coarse_map,
            similarity='hybrid',
            alpha=self.alpha,
            beta=self.beta,
        )

    def compute_merge_ic(
        self,
        fine_state: ScaleState,
        coarse_state: ScaleState,
        fine_to_coarse_map: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute IC for separate vs merged states.

        Args:
            fine_state: Fine scale state
            coarse_state: Coarse scale state
            fine_to_coarse_map: Mapping from fine to coarse tokens

        Returns:
            ic_separate: IC for keeping separate (scalar)
            ic_merged: IC for merged state (scalar)
            should_merge: Boolean decision
            merged_features: Merged features (cached for reuse)
            merged_variance: Merged variance (cached for reuse)
            merged_sizes: Merged sizes (cached for reuse)
            merged_info: Merged IC per token (cached for reuse)
        """
        # Separate IC: mean of individual ICs
        ic_fine = fine_state.info.mean()
        ic_coarse = coarse_state.info.mean()
        ic_separate = ic_fine + ic_coarse

        # Compute hypothetical merged statistics (computed once, reused if merge)
        merged_features, merged_variance, merged_sizes = compute_merged_statistics(
            fine_state, coarse_state, fine_to_coarse_map
        )

        # Compute IC for merged tokens (computed once, reused if merge)
        merged_info = infocrit(
            merged_variance,
            merged_sizes,
            fine_state.H, fine_state.W,
            self.criterion, self.iota, self.eps
        )
        ic_merged = merged_info.mean()

        # Decision: merge if merged IC > separate IC
        should_merge = ic_merged > ic_separate

        if self.debug:
            print(
                f"Scale {fine_state.scale_idx}: "
                f"IC separate={ic_separate:.4f}, merged={ic_merged:.4f}, "
                f"improvement={ic_merged - ic_separate:.4f}, "
                f"merge={should_merge.item()}"
            )

        return ic_separate, ic_merged, should_merge, merged_features, merged_variance, merged_sizes, merged_info

    def forward(
        self,
        fine_state: ScaleState,
        coarse_state: ScaleState,
        fine_to_coarse_map: Tensor,
    ) -> MergeResult:
        """Perform cross-scale token merging.

        Main entry point for token merging.

        Args:
            fine_state: Fine scale state
            coarse_state: Coarse scale state
            fine_to_coarse_map: Mapping from fine tokens to coarse tokens

        Returns:
            MergeResult containing:
                - merged_state: Updated or original fine state
                - did_merge: Whether merging occurred
                - ic_improvement: IC improvement
                - similarity_mean: Mean similarity
        """
        # Compute similarity for logging
        similarity = self.compute_hybrid_similarity(
            fine_state.features,
            coarse_state.features,
            fine_to_coarse_map,
        )
        sim_mean = similarity.mean().item()

        if self.use_infocrit:
            result = self._merge_by_infocrit(
                fine_state, coarse_state, fine_to_coarse_map
            )
        else:
            result = self._merge_by_threshold(
                fine_state, coarse_state, fine_to_coarse_map, similarity
            )

        # Update result with similarity
        return MergeResult(
            merged_state=result.merged_state,
            did_merge=result.did_merge,
            ic_improvement=result.ic_improvement,
            similarity_mean=sim_mean,
        )

    def _merge_by_infocrit(
        self,
        fine_state: ScaleState,
        coarse_state: ScaleState,
        fine_to_coarse_map: Tensor,
    ) -> MergeResult:
        """IC-guided merging (dHT-style)."""
        # Compute IC and merged statistics (computed once, reused if merge)
        (ic_separate, ic_merged, should_merge,
         merged_features, merged_variance, merged_sizes, merged_info) = self.compute_merge_ic(
            fine_state, coarse_state, fine_to_coarse_map
        )

        ic_improvement = (ic_merged - ic_separate).item()

        if not should_merge:
            # Keep fine tokens separate
            return MergeResult(
                merged_state=fine_state,
                did_merge=False,
                ic_improvement=ic_improvement,
                similarity_mean=0.0,
            )

        # Create new state (reuse all cached values - no recomputation)
        merged_state = ScaleState(
            features=merged_features,
            variance=merged_variance,
            sizes=merged_sizes,
            info=merged_info,
            H=fine_state.H,
            W=fine_state.W,
            scale_idx=fine_state.scale_idx,
        )

        # Update statistics
        self.merge_count += 1
        self.total_ic_improvement += ic_improvement

        return MergeResult(
            merged_state=merged_state,
            did_merge=True,
            ic_improvement=ic_improvement,
            similarity_mean=0.0,
        )

    def _merge_by_threshold(
        self,
        fine_state: ScaleState,
        coarse_state: ScaleState,
        fine_to_coarse_map: Tensor,
        similarity: Tensor,
    ) -> MergeResult:
        """Threshold-based merging (baseline D2O-style)."""
        # Merge tokens with high similarity
        merge_mask = similarity > self.similarity_threshold
        n_merge = merge_mask.sum().item()

        if n_merge == 0:
            return MergeResult(
                merged_state=fine_state,
                did_merge=False,
                ic_improvement=0.0,
                similarity_mean=similarity.mean().item(),
            )

        # Get coarse parents
        coarse_parents = coarse_state.features[fine_to_coarse_map]

        # Merge high-similarity tokens: replace fine with coarse
        merged_features = torch.where(
            merge_mask.view(-1, 1),
            coarse_parents,
            fine_state.features,
        )

        # Estimate new variance
        merged_variance = estimate_token_variance(merged_features, 'uniform')

        # Sizes remain unchanged (not a weighted merge)
        merged_sizes = fine_state.sizes

        # Compute new IC
        merged_info = infocrit(
            merged_variance, merged_sizes,
            fine_state.H, fine_state.W,
            self.criterion, self.iota, self.eps
        )

        merged_state = ScaleState(
            features=merged_features,
            variance=merged_variance,
            sizes=merged_sizes,
            info=merged_info,
            H=fine_state.H,
            W=fine_state.W,
            scale_idx=fine_state.scale_idx,
        )

        # Compute IC improvement
        ic_improvement = (merged_info.mean() - fine_state.info.mean()).item()

        return MergeResult(
            merged_state=merged_state,
            did_merge=True,
            ic_improvement=ic_improvement,
            similarity_mean=similarity.mean().item(),
        )

    def merge_all_scales(
        self,
        multi_state: MultiScaleState,
    ) -> MultiScaleState:
        """Merge tokens across all adjacent scale pairs.

        Convenience method for multi-scale processing.

        Args:
            multi_state: State for all scales

        Returns:
            Updated MultiScaleState with merged tokens

        Note:
            Processes scales from fine to coarse (5→4→3→2→1).
            This order ensures finer details are preserved when possible.
        """
        if multi_state.num_scales < 2:
            return multi_state

        merge_history = []

        # Process from finest to coarsest
        for i in range(multi_state.num_scales - 1, 0, -1):
            fine_state = multi_state.get_scale(i)
            coarse_state = multi_state.get_scale(i - 1)

            # Build mapping
            mapping = build_spatial_mapping(
                fine_state.H, fine_state.W,
                coarse_state.H, coarse_state.W,
                device=fine_state.features.device,
            )

            # Merge
            result = self.forward(fine_state, coarse_state, mapping)

            # Update state
            multi_state.update_scale(i, result.merged_state)

            # Log
            merge_history.append({
                'scale_pair': (i, i - 1),
                'did_merge': result.did_merge,
                'ic_improvement': result.ic_improvement,
                'similarity_mean': result.similarity_mean,
            })

        multi_state.merge_history = merge_history
        return multi_state

    def get_stats(self) -> dict:
        """Get merging statistics."""
        return {
            'merge_count': self.merge_count.item(),
            'total_ic_improvement': self.total_ic_improvement.item(),
            'avg_ic_improvement': (
                self.total_ic_improvement.item() / max(1, self.merge_count.item())
            ),
        }

    def reset_stats(self) -> None:
        """Reset merging statistics."""
        self.merge_count.zero_()
        self.total_ic_improvement.zero_()

    def extra_repr(self) -> str:
        """Extra representation for print()."""
        return (
            f"d_model={self.d_model}, "
            f"use_infocrit={self.use_infocrit}, "
            f"criterion={self.criterion}, "
            f"alpha={self.alpha}, beta={self.beta}"
        )
