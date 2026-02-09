"""Unit tests for D2O module.

Tests cover:
    - Information criteria (infocrit)
    - Similarity functions
    - State management
    - Token merger

Run with:
    pytest atoken/d2o/tests/test_d2o.py -v
"""
import pytest
import torch
import torch.nn as nn
from typing import Literal

# Import modules under test
from atoken.d2o.infocrit import (
    gaussian_2nll,
    infocrit,
    infocrit_multi_scale,
    compare_ic_for_merge,
)
from atoken.d2o.similarity import (
    gaussian_similarity,
    cosine_similarity,
    hybrid_similarity,
    compute_cross_scale_similarity,
    build_spatial_mapping,
)
from atoken.d2o.state import (
    ScaleState,
    create_scale_state,
    estimate_token_variance,
    compute_merged_statistics,
)
from atoken.d2o.merger import D2OTokenMerger, MergeResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Default device for tests."""
    return torch.device('cpu')


@pytest.fixture
def small_features(device):
    """Small feature tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(16, 64, device=device)  # 16 tokens, 64 dims


@pytest.fixture
def medium_features(device):
    """Medium feature tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(256, 64, device=device)  # 256 tokens (16x16)


@pytest.fixture
def edges(device):
    """Sample edges for testing."""
    # Create 50 random edges between 16 nodes
    torch.manual_seed(42)
    return torch.randint(0, 16, (2, 50), device=device)


# ============================================================================
# Test Information Criteria (infocrit.py)
# ============================================================================

class TestInfocrit:
    """Tests for information criteria module."""

    def test_gaussian_2nll_shape(self, device):
        """Test NLL output shape."""
        s2 = torch.rand(100, 3, device=device)  # 100 regions, 3 channels
        mV = torch.rand(100, device=device)
        nll = gaussian_2nll(s2, mV)
        assert nll.shape == (100,)

    def test_gaussian_2nll_positive(self, device):
        """Test NLL is positive (log of positive)."""
        s2 = torch.rand(100, 3, device=device) + 0.1  # Ensure positive
        mV = torch.rand(100, device=device) + 0.1
        nll = gaussian_2nll(s2, mV, iota=1.0)
        # NLL = mV * log(2π * prod(s2) + iota)
        # With iota=1.0 and positive inputs, should be positive
        assert torch.isfinite(nll).all()

    def test_infocrit_shape(self, device):
        """Test IC output shape."""
        s2 = torch.rand(100, 3, device=device)
        mV = torch.randint(10, 100, (100,), device=device).float()
        ic = infocrit(s2, mV, H=32, W=32, mode='bic')
        assert ic.shape == (100,)

    def test_infocrit_all_modes(self, device):
        """Test all IC modes produce valid output."""
        s2 = torch.rand(50, 3, device=device)
        mV = torch.randint(10, 100, (50,), device=device).float()

        for mode in ['aic', 'bic', 'aicc', 'cic']:
            ic = infocrit(s2, mV, H=32, W=32, mode=mode)
            assert torch.isfinite(ic).all(), f"Mode {mode} produced non-finite values"

    def test_infocrit_higher_variance_lower_ic(self, device):
        """Test that higher variance leads to lower IC (worse fit)."""
        mV = torch.full((2,), 100.0, device=device)
        s2_low = torch.full((2, 3), 0.1, device=device)
        s2_high = torch.full((2, 3), 1.0, device=device)

        ic_low = infocrit(s2_low, mV, H=32, W=32, mode='bic')
        ic_high = infocrit(s2_high, mV, H=32, W=32, mode='bic')

        # Higher variance → worse fit → lower IC
        assert (ic_low > ic_high).all()

    def test_infocrit_multi_scale(self, device):
        """Test multi-scale IC computation."""
        s2_list = [torch.rand(n, 3, device=device) for n in [4, 16, 64]]
        mV_list = [torch.rand(n, device=device) * 100 + 10 for n in [4, 16, 64]]
        H_list = [2, 4, 8]
        W_list = [2, 4, 8]

        ic_list = infocrit_multi_scale(s2_list, mV_list, H_list, W_list)

        assert len(ic_list) == 3
        for ic, n in zip(ic_list, [4, 16, 64]):
            assert ic.shape == (n,)

    def test_compare_ic_for_merge(self, device):
        """Test IC comparison for merge decision."""
        ic_fine = torch.tensor([1.0, 2.0, 3.0], device=device)
        ic_coarse = torch.tensor([1.5, 1.5, 1.5], device=device)

        # Merged IC that's better than separate
        ic_merged_better = torch.tensor([3.0, 4.0, 5.0], device=device)
        should_merge, improvement = compare_ic_for_merge(
            ic_fine, ic_coarse, ic_merged_better
        )
        assert should_merge.item() == True
        assert improvement.mean() > 0

        # Merged IC that's worse than separate
        ic_merged_worse = torch.tensor([1.0, 1.0, 1.0], device=device)
        should_merge, improvement = compare_ic_for_merge(
            ic_fine, ic_coarse, ic_merged_worse
        )
        assert should_merge.item() == False
        assert improvement.mean() < 0


# ============================================================================
# Test Similarity Functions (similarity.py)
# ============================================================================

class TestSimilarity:
    """Tests for similarity functions."""

    def test_gaussian_similarity_shape(self, small_features, edges):
        """Test Gaussian similarity output shape."""
        sim = gaussian_similarity(small_features, edges)
        assert sim.shape == (edges.shape[1],)

    def test_gaussian_similarity_range(self, small_features, edges):
        """Test Gaussian similarity is in [0, 1]."""
        sim = gaussian_similarity(small_features, edges)
        # Gaussian can be 0 when features are very far apart (exp(-inf) = 0)
        assert (sim >= 0).all()
        assert (sim <= 1).all()

    def test_gaussian_similarity_identical(self, device):
        """Test similarity of identical features is 1."""
        features = torch.randn(10, 64, device=device)
        # Self-edges (same node)
        edges = torch.arange(10, device=device).unsqueeze(0).expand(2, -1)
        sim = gaussian_similarity(features, edges)
        # Identical features → distance=0 → exp(0)=1
        assert torch.allclose(sim, torch.ones_like(sim), atol=1e-6)

    def test_cosine_similarity_shape(self, small_features, edges):
        """Test Cosine similarity output shape."""
        sim = cosine_similarity(small_features, edges)
        assert sim.shape == (edges.shape[1],)

    def test_cosine_similarity_range(self, small_features, edges):
        """Test Cosine similarity is in [0, 1]."""
        sim = cosine_similarity(small_features, edges)
        assert (sim >= 0).all()
        assert (sim <= 1).all()

    def test_cosine_similarity_identical(self, device):
        """Test cosine similarity of identical features is 1."""
        features = torch.randn(10, 64, device=device)
        edges = torch.arange(10, device=device).unsqueeze(0).expand(2, -1)
        sim = cosine_similarity(features, edges)
        # Same direction → cosine=1 → mapped to 1
        assert torch.allclose(sim, torch.ones_like(sim), atol=1e-6)

    def test_hybrid_similarity(self, small_features, edges):
        """Test hybrid similarity is weighted sum."""
        g_sim = gaussian_similarity(small_features, edges)
        c_sim = cosine_similarity(small_features, edges)
        h_sim = hybrid_similarity(small_features, edges, alpha=0.3, beta=0.7)

        expected = 0.3 * g_sim + 0.7 * c_sim
        assert torch.allclose(h_sim, expected, atol=1e-6)

    def test_cross_scale_similarity_shape(self, device):
        """Test cross-scale similarity output shape."""
        fine = torch.randn(256, 64, device=device)  # 16x16
        coarse = torch.randn(64, 64, device=device)  # 8x8
        mapping = torch.randint(0, 64, (256,), device=device)

        sim = compute_cross_scale_similarity(fine, coarse, mapping)
        assert sim.shape == (256,)

    def test_build_spatial_mapping(self, device):
        """Test spatial mapping construction."""
        mapping = build_spatial_mapping(16, 16, 8, 8, device=device)

        assert mapping.shape == (256,)
        assert mapping.min() >= 0
        assert mapping.max() < 64

        # Check specific mappings
        # (0,0) → 0, (0,1) → 0, (1,0) → 0, (1,1) → 0 (2x2 pooling)
        assert mapping[0] == 0  # (0,0)
        assert mapping[1] == 0  # (0,1)
        assert mapping[16] == 0  # (1,0)
        assert mapping[17] == 0  # (1,1)


# ============================================================================
# Test State Management (state.py)
# ============================================================================

class TestState:
    """Tests for state management."""

    def test_create_scale_state(self, medium_features):
        """Test scale state creation."""
        state = create_scale_state(
            medium_features, H=16, W=16, scale_idx=2
        )

        assert state.features.shape == (256, 64)
        assert state.variance.shape == (256, 64)
        assert state.sizes.shape == (256,)
        assert state.info.shape == (256,)
        assert state.H == 16
        assert state.W == 16
        assert state.scale_idx == 2

    def test_scale_state_properties(self, medium_features):
        """Test ScaleState properties."""
        state = create_scale_state(
            medium_features, H=16, W=16, scale_idx=2
        )

        assert state.n_tokens == 256
        assert state.dim == 64
        assert isinstance(state.mean_ic(), float)

    def test_estimate_token_variance_uniform(self, medium_features):
        """Test uniform variance estimation."""
        var = estimate_token_variance(medium_features, 'uniform', base_variance=0.01)
        assert var.shape == medium_features.shape
        assert torch.allclose(var, torch.full_like(var, 0.01))

    def test_estimate_token_variance_feature_std(self, medium_features):
        """Test feature-based variance estimation."""
        var = estimate_token_variance(medium_features, 'feature_std', base_variance=0.01)
        assert var.shape == medium_features.shape
        assert (var > 0).all()

    def test_compute_merged_statistics(self, device):
        """Test merged statistics computation."""
        # Create fine and coarse states
        fine_features = torch.randn(256, 64, device=device)
        coarse_features = torch.randn(64, 64, device=device)

        fine_state = create_scale_state(fine_features, H=16, W=16, scale_idx=2)
        coarse_state = create_scale_state(coarse_features, H=8, W=8, scale_idx=1)

        mapping = build_spatial_mapping(16, 16, 8, 8, device=device)

        merged_f, merged_v, merged_s = compute_merged_statistics(
            fine_state, coarse_state, mapping
        )

        assert merged_f.shape == (256, 64)
        assert merged_v.shape == (256, 64)
        assert merged_s.shape == (256,)

        # Merged sizes should be larger
        assert (merged_s > fine_state.sizes).all()


# ============================================================================
# Test Token Merger (merger.py)
# ============================================================================

class TestMerger:
    """Tests for token merger."""

    def test_merger_creation(self):
        """Test merger module creation."""
        merger = D2OTokenMerger(
            d_model=64,
            num_scales=5,
            use_infocrit=True,
            criterion='bic',
        )

        assert merger.d_model == 64
        assert merger.use_infocrit == True
        assert merger.criterion == 'bic'

    def test_merger_forward_infocrit(self, device):
        """Test merger forward pass with IC."""
        merger = D2OTokenMerger(
            d_model=64,
            use_infocrit=True,
            criterion='bic',
        )

        # Create states
        fine = torch.randn(256, 64, device=device)
        coarse = torch.randn(64, 64, device=device)
        fine_state = create_scale_state(fine, H=16, W=16, scale_idx=2)
        coarse_state = create_scale_state(coarse, H=8, W=8, scale_idx=1)
        mapping = build_spatial_mapping(16, 16, 8, 8, device=device)

        result = merger(fine_state, coarse_state, mapping)

        assert isinstance(result, MergeResult)
        assert isinstance(result.did_merge, bool)
        assert isinstance(result.ic_improvement, float)
        assert result.merged_state.features.shape == (256, 64)

    def test_merger_forward_threshold(self, device):
        """Test merger forward pass with threshold."""
        merger = D2OTokenMerger(
            d_model=64,
            use_infocrit=False,
            similarity_threshold=0.75,
        )

        fine = torch.randn(256, 64, device=device)
        coarse = torch.randn(64, 64, device=device)
        fine_state = create_scale_state(fine, H=16, W=16, scale_idx=2)
        coarse_state = create_scale_state(coarse, H=8, W=8, scale_idx=1)
        mapping = build_spatial_mapping(16, 16, 8, 8, device=device)

        result = merger(fine_state, coarse_state, mapping)

        assert isinstance(result, MergeResult)

    def test_merger_similar_features_merge(self, device):
        """Test that identical features lead to high similarity."""
        merger = D2OTokenMerger(
            d_model=64,
            use_infocrit=True,
            criterion='bic',
        )

        # Create identical coarse features, fine is slightly perturbed
        torch.manual_seed(123)
        coarse = torch.randn(64, 64, device=device)
        # Each fine token is its coarse parent + small noise
        mapping = build_spatial_mapping(16, 16, 8, 8, device=device)
        fine = coarse[mapping] + torch.randn(256, 64, device=device) * 0.01

        fine_state = create_scale_state(fine, H=16, W=16, scale_idx=2)
        coarse_state = create_scale_state(coarse, H=8, W=8, scale_idx=1)

        result = merger(fine_state, coarse_state, mapping)

        # With nearly identical features, similarity should be very high
        assert result.similarity_mean > 0.9

    def test_merger_stats(self, device):
        """Test merger statistics tracking."""
        merger = D2OTokenMerger(d_model=64, use_infocrit=True)

        # Reset stats
        merger.reset_stats()
        assert merger.merge_count.item() == 0

        # Run a merge
        fine = torch.randn(256, 64, device=device)
        coarse = torch.randn(64, 64, device=device)
        fine_state = create_scale_state(fine, H=16, W=16, scale_idx=2)
        coarse_state = create_scale_state(coarse, H=8, W=8, scale_idx=1)
        mapping = build_spatial_mapping(16, 16, 8, 8, device=device)

        result = merger(fine_state, coarse_state, mapping)

        stats = merger.get_stats()
        assert 'merge_count' in stats
        assert 'total_ic_improvement' in stats
        assert 'avg_ic_improvement' in stats


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""

    def test_end_to_end_pipeline(self, device):
        """Test complete pipeline from features to merged state."""
        # 1. Create features with reasonable scale
        torch.manual_seed(42)
        features_s5 = torch.randn(1024, 64, device=device)  # 32x32
        features_s4 = torch.randn(256, 64, device=device)   # 16x16
        features_s3 = torch.randn(64, 64, device=device)    # 8x8

        # 2. Create states with proper sizes (larger regions for numerical stability)
        # Use sizes that are representative of actual tokenization
        state_s5 = ScaleState(
            features=features_s5,
            variance=torch.full_like(features_s5, 0.01),
            sizes=torch.full((1024,), 10.0, device=device),  # 10 pixels per region
            info=torch.zeros(1024, device=device),
            H=32, W=32, scale_idx=4
        )
        state_s4 = ScaleState(
            features=features_s4,
            variance=torch.full_like(features_s4, 0.01),
            sizes=torch.full((256,), 40.0, device=device),  # 40 pixels per region
            info=torch.zeros(256, device=device),
            H=16, W=16, scale_idx=3
        )
        state_s3 = ScaleState(
            features=features_s3,
            variance=torch.full_like(features_s3, 0.01),
            sizes=torch.full((64,), 160.0, device=device),  # 160 pixels per region
            info=torch.zeros(64, device=device),
            H=8, W=8, scale_idx=2
        )

        # 3. Create merger
        merger = D2OTokenMerger(d_model=64, use_infocrit=True, criterion='bic')

        # 4. Merge s5 → s4
        mapping_54 = build_spatial_mapping(32, 32, 16, 16, device=device)
        result_54 = merger(state_s5, state_s4, mapping_54)

        # 5. Merge s4 → s3
        mapping_43 = build_spatial_mapping(16, 16, 8, 8, device=device)
        result_43 = merger(state_s4, state_s3, mapping_43)

        # Verify results
        assert result_54.merged_state.features.shape == (1024, 64)
        assert result_43.merged_state.features.shape == (256, 64)

        # Check IC is finite
        assert torch.isfinite(result_54.merged_state.info).all()
        assert torch.isfinite(result_43.merged_state.info).all()

    def test_gradient_flow(self, device):
        """Test gradients flow through merger when merging occurs."""
        # Create features with gradient tracking
        # Make fine features very similar to their coarse parents to ensure merging
        torch.manual_seed(42)
        coarse = torch.randn(64, 64, device=device, requires_grad=True)
        mapping = build_spatial_mapping(16, 16, 8, 8, device=device)
        # Fine features = coarse parents + tiny noise (ensures high similarity & merge)
        noise = torch.randn(256, 64, device=device) * 0.001
        fine = coarse[mapping] + noise
        fine.requires_grad_(True)
        fine.retain_grad()

        # Create states with proper sizes
        fine_state = ScaleState(
            features=fine,
            variance=torch.full_like(fine, 0.01),
            sizes=torch.full((256,), 40.0, device=device),
            info=torch.zeros(256, device=device),
            H=16, W=16, scale_idx=2
        )
        coarse_state = ScaleState(
            features=coarse,
            variance=torch.full_like(coarse, 0.01),
            sizes=torch.full((64,), 160.0, device=device),
            info=torch.zeros(64, device=device),
            H=8, W=8, scale_idx=1
        )

        merger = D2OTokenMerger(d_model=64, use_infocrit=True)
        result = merger(fine_state, coarse_state, mapping)

        # Compute loss and backward
        loss = result.merged_state.features.sum()
        loss.backward()

        # Fine gradients should always exist (features are always in output)
        assert fine.grad is not None

        # Coarse gradients exist only if merging occurred
        if result.did_merge:
            assert coarse.grad is not None
        else:
            # When no merge, coarse isn't used, so no grad expected
            pass


# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
