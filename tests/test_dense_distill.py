"""Semantic-head improvements (2026-09-02 analysis): dense token distillation onto the
backbone tokens + a content-only understanding head.

Probe showed: content slots kNN 0.386 > z-head 0.335 (head loses info by pooling over 256
non-semantic detail tokens), trunk ceiling ~0.4 vs teacher 0.79 (one pooled target per image
is too weak). Dense per-token alignment to teacher patch tokens hits the trunk directly."""
from __future__ import annotations
import pytest
import torch
from mavt.model.mavt import MAVT
from mavt.losses.losses import MAVTLoss, dense_distill_loss, pool_teacher_tokens

TINY = dict(embed_dim=64, num_heads=4, num_blocks=2, patch_size=16, t_patch=2, latent_dim=8,
            semantic_dim=16, dec_dim=64, num_dec_attn_blocks=1, use_gradient_checkpointing=False)


def test_pool_teacher_tokens_maps_teacher_grid_to_student_grid():
    B, D = 2, 8
    hidden = torch.randn(B, 24 * 24, D)                      # SigLIP2 @384/16
    out = pool_teacher_tokens(hidden, (16, 16))
    assert out.shape == (B, 256, D)
    same = pool_teacher_tokens(torch.randn(B, 16 * 16, D), (16, 16))
    assert same.shape == (B, 256, D)
    # a constant map stays constant (no edge artefacts)
    const = pool_teacher_tokens(torch.ones(B, 24 * 24, D), (16, 16))
    assert torch.allclose(const, torch.ones(B, 256, D), atol=1e-5)


def test_dense_distill_loss_is_zero_for_aligned_tokens_and_positive_otherwise():
    s = torch.randn(2, 10, 8)
    assert dense_distill_loss(s, s * 3.0) < 1e-6              # scale-invariant (cosine)
    assert dense_distill_loss(s, -s) > 1.9
    with pytest.raises(ValueError):
        dense_distill_loss(s, torch.randn(2, 9, 8))          # token count must match


def test_mavt_loss_exposes_loss_dense():
    lf = MAVTLoss(w_dense=0.5, use_lpips=False)
    pred = torch.rand(2, 3, 32, 32); tgt = torch.rand(2, 3, 32, 32)
    st = torch.randn(2, 4, 8); te = torch.randn(2, 4, 8)
    out = lf(pred=pred, target=tgt, loss_kl=torch.tensor(0.0), slot_diversity=torch.tensor(0.0),
             modality="image", dense_student=st, dense_teacher=te)
    assert "loss_dense" in out and out["loss_dense"] > 0
    base = lf(pred=pred, target=tgt, loss_kl=torch.tensor(0.0), slot_diversity=torch.tensor(0.0), modality="image")
    assert out["loss"] > base["loss"], "dense term must contribute to the total"


def test_mavt_output_exposes_backbone_tokens():
    torch.manual_seed(0)
    m = MAVT(**TINY); m.prepare_for_modalities([{"modality": "image", "resolution": 64}]); m.eval()
    out = m(torch.randn(2, 3, 64, 64), "image", decode=False)
    assert out.backbone_tokens.shape == (2, 16, TINY["embed_dim"])   # (64/16)^2 = 16 tokens, pre-split


def test_content_only_semantic_head_ignores_detail_tokens():
    torch.manual_seed(0)
    m = MAVT(**TINY, semantic_content_only=True)
    m.prepare_for_modalities([{"modality": "image", "resolution": 64}]); m.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = m(x, "image", decode=False)
        types = out.latent_token_types
        assert (types == 1).any() and (types == 0).any()
        z_perturbed = out.z.clone(); z_perturbed[:, types == 1] += torch.randn_like(z_perturbed[:, types == 1]) * 5
        a = m.understanding_decoder(out.z, types); b = m.understanding_decoder(z_perturbed, types)
    assert torch.allclose(a, b, atol=1e-5), "content-only head must not depend on detail tokens"
    # default (all tokens) behaviour still reachable
    m2 = MAVT(**TINY); m2.prepare_for_modalities([{"modality": "image", "resolution": 64}]); m2.eval()
    with torch.no_grad():
        o2 = m2(x, "image", decode=False); zp = o2.z.clone(); zp[:, o2.latent_token_types == 1] += 5
        assert not torch.allclose(m2.understanding_decoder(o2.z, o2.latent_token_types), m2.understanding_decoder(zp, o2.latent_token_types))
