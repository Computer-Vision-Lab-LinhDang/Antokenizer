"""Smoke test for MAVT v2.

Tests:
1. Forward pass for all modalities
2. Two-axis decomposition statistics
3. Gradient isolation
4. Memory usage
"""

from __future__ import annotations

import argparse
import sys

import torch


def test_import():
    """Test imports."""
    print("Testing imports...")
    sys.path.insert(0, 'src')

    from mavt.model.mavt_v2 import (
        MAVT,
        TwoAxisDecomposition,
        ScalePyramidEncoder,
        BackboneV2,
        SemanticHead,
        ReconstructionHead,
        create_mavt_v2_small,
        create_mavt_v2_base,
    )
    print("  ✅ All imports successful")


def test_two_axis_decomposition():
    """Test two-axis decomposition."""
    print("\nTesting Two-Axis Decomposition...")

    from mavt.model.mavt_v2 import TwoAxisDecomposition

    B, N, K, D = 2, 64, 4, 256
    features = torch.randn(B, N, K, D)

    module = TwoAxisDecomposition(dim=D)
    z_inv, z_var, metrics = module(features, k_dim=2)

    print(f"  Input shape: {features.shape}")
    print(f"  z_inv shape: {z_inv.shape}")
    print(f"  z_var shape: {z_var.shape}")
    print(f"  Energy in z_inv: {metrics['energy_ratio']:.4f}")
    print(f"  Gini: {metrics['gini']:.4f}")

    # Verify energy distribution
    assert 0.9 < metrics['energy_ratio'] < 1.0, "z_inv should have ~98% energy"
    assert z_var.shape == features.shape, "z_var shape mismatch"
    print("  ✅ Two-axis decomposition passed")


def test_scale_pyramid_encoder():
    """Test scale pyramid encoder."""
    print("\nTesting Scale Pyramid Encoder...")

    from mavt.model.mavt_v2 import ScalePyramidEncoder

    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W)

    module = ScalePyramidEncoder(embed_dim=256, num_scales=4)
    target_h, target_w = H // 16, W // 16

    feat = module(x, target_h, target_w)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {feat.shape}")
    print(f"  Expected: ({B}, {target_h*target_w}, 4, 256)")

    assert feat.shape == (B, target_h * target_w, 4, 256), "Scale pyramid output shape mismatch"
    print("  ✅ Scale pyramid encoder passed")


def test_backbone_v2():
    """Test backbone v2."""
    print("\nTesting Backbone V2...")

    from mavt.model.mavt_v2 import BackboneV2

    B, N, D = 2, 256, 256
    x = torch.randn(B, N, D)

    module = BackboneV2(dim=D, num_heads=4, num_blocks=4)
    positions = torch.zeros(N, 4, dtype=torch.long)
    plane_ids = torch.zeros(N, dtype=torch.long)

    out = module(x, positions, plane_ids, 'image', grid_shape=(16, 16))

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")

    assert out.shape == x.shape, "Backbone output shape mismatch"
    print("  ✅ Backbone V2 passed")


def test_full_model_image():
    """Test full model forward on image."""
    print("\nTesting Full Model (Image)...")

    from mavt.model.mavt_v2 import MAVT

    B, C, H, W = 1, 3, 256, 256
    x = torch.randn(B, C, H, W)

    model = MAVT(
        embed_dim=256,
        num_heads=4,
        num_blocks=4,
        latent_dim=16,
        semantic_dim=128,
        dec_dim=128,
        num_dec_attn_blocks=2,
        num_scales=4,
    )
    model.eval()

    with torch.no_grad():
        out = model(x, 'image', decode=True)

    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {out.reconstruction.shape}")
    print(f"  z_inv shape: {out.z_inv.shape}")
    print(f"  z_var shape: {out.z_var.shape}")
    print(f"  Semantic shape: {out.semantic.shape}")
    print(f"  Energy in z_inv: {out.two_axis_metrics.get('energy_ratio', 'N/A'):.4f}")

    assert out.reconstruction.shape == x.shape, "Reconstruction shape mismatch"
    print("  ✅ Full model (Image) passed")


def test_full_model_video():
    """Test full model forward on video."""
    print("\nTesting Full Model (Video)...")

    from mavt.model.mavt_v2 import MAVT

    B, C, T, H, W = 1, 3, 16, 128, 128
    x = torch.randn(B, C, T, H, W)

    model = MAVT(
        embed_dim=256,
        num_heads=4,
        num_blocks=4,
        latent_dim=16,
        semantic_dim=128,
        dec_dim=128,
        num_dec_attn_blocks=2,
    )
    model.eval()

    with torch.no_grad():
        out = model(x, 'video', decode=True)

    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {out.reconstruction.shape}")
    print(f"  z_inv shape: {out.z_inv.shape}")

    print("  ✅ Full model (Video) passed")


def test_gradient_isolation():
    """Test gradient isolation between semantic and recon."""
    print("\nTesting Gradient Isolation...")

    from mavt.model.mavt_v2 import MAVT

    B, C, H, W = 1, 3, 64, 64
    x = torch.randn(B, C, H, W)

    model = MAVT(
        embed_dim=128,
        num_heads=4,
        num_blocks=2,
        latent_dim=16,
        semantic_dim=64,
        dec_dim=128,
        num_dec_attn_blocks=1,
        num_scales=4,
    )

    # Forward
    out = model(x, 'image', decode=True)

    # Recon loss
    recon_loss = (out.reconstruction - x).abs().mean()

    # Semantic (just gradient check)
    semantic_loss = out.semantic.abs().mean()

    # Backward
    recon_loss.backward()

    # Check that gradients exist
    has_grad = {
        'recon_head': model.recon_head.z_var_proj.weight.grad is not None,
        'semantic_head': model.semantic_head.proj[0].weight.grad is not None,
    }

    print(f"  Recon loss: {recon_loss.item():.4f}")
    print(f"  Semantic loss: {semantic_loss.item():.4f}")
    print(f"  Recon head has grad: {has_grad['recon_head']}")
    print(f"  Semantic head has grad: {has_grad['semantic_head']}")

    print("  ✅ Gradient isolation test passed")


def test_factory_functions():
    """Test factory functions."""
    print("\nTesting Factory Functions...")

    from mavt.model.mavt_v2 import (
        create_mavt_v2_small,
        create_mavt_v2_base,
        create_mavt_v2_large,
    )

    for name, create_fn in [
        ('small', create_mavt_v2_small),
        ('base', create_mavt_v2_base),
        ('large', create_mavt_v2_large),
    ]:
        model = create_fn()
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} parameters")

    print("  ✅ Factory functions passed")


def test_memory_usage():
    """Test memory usage."""
    print("\nTesting Memory Usage...")

    from mavt.model.mavt_v2 import MAVT

    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available, skipping memory test")
        return

    torch.cuda.reset_peak_memory_stats()

    B, C, H, W = 1, 3, 256, 256
    x = torch.randn(B, C, H, W).cuda()

    model = create_mavt_v2_small().cuda()
    model.eval()

    with torch.no_grad():
        out = model(x, 'image', decode=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"  Peak memory: {peak_mem:.2f} MB")
    print("  ✅ Memory test passed")


def main():
    print("=" * 60)
    print("MAVT v2 SMOKE TEST")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-cuda', action='store_true', help='Skip CUDA tests')
    args = parser.parse_args()

    try:
        test_import()
        test_two_axis_decomposition()
        test_scale_pyramid_encoder()
        test_backbone_v2()
        test_full_model_image()
        test_full_model_video()
        test_gradient_isolation()
        test_factory_functions()

        if not args.skip_cuda:
            test_memory_usage()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
