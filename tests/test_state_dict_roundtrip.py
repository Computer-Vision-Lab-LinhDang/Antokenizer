"""Dynamically-created parameters (SigLIP2-inherited pos2d, lazily-built content poolers)
must survive save → fresh-model load. The 2026-09-02 gate evals silently dropped
`patchify.pos2d` (strict=False, unexpected=1) and evaluated a model without its
learned 2D position table → 12 dB PSNR while training/validation showed ~22 dB."""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from mavt.model.mavt import MAVT

TINY = dict(embed_dim=64, num_heads=4, num_blocks=2, patch_size=16, t_patch=2, latent_dim=8,
            semantic_dim=16, dec_dim=64, num_dec_attn_blocks=1, use_gradient_checkpointing=False)


def _trained_like_model() -> MAVT:
    torch.manual_seed(0)
    m = MAVT(**TINY)
    m.prepare_for_modalities([{"modality": "image", "resolution": 64}])
    m.patchify.set_pos2d(torch.randn(1, TINY["embed_dim"], 3, 3))       # what init_from_siglip2 does
    return m.eval()


def test_state_dict_contains_dynamic_params():
    m = _trained_like_model()
    keys = m.state_dict().keys()
    assert "patchify.pos2d" in keys
    assert any(k.startswith("cd_split._content_poolers.") for k in keys)


def test_fresh_model_loads_strict_and_reproduces_output():
    m = _trained_like_model()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref = m(x, "image", decode=True).reconstruction
    fresh = MAVT(**TINY)                       # no prepare_for_modalities, no pos2d
    missing, unexpected = fresh.load_state_dict(m.state_dict(), strict=True)
    assert not missing and not unexpected
    fresh.eval()
    with torch.no_grad():
        out = fresh(x, "image", decode=True).reconstruction
    assert torch.allclose(out, ref, atol=1e-5), "loaded model must reproduce the saved model's output"


def test_load_under_parent_prefix_like_lightning():
    """Lightning loads the *parent* module's state_dict (keys prefixed 'model.'); the
    materialisation must work through the recursive load path, not only MAVT.load_state_dict."""
    m = _trained_like_model()
    wrapper = nn.Module(); wrapper.model = m
    sd = wrapper.state_dict()
    fresh = nn.Module(); fresh.model = MAVT(**TINY)
    missing, unexpected = fresh.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    assert fresh.model.patchify.pos2d is not None and fresh.model.patchify.pos2d.shape == (1, TINY["embed_dim"], 3, 3)


def test_pos2d_is_actually_used_in_forward():
    m = _trained_like_model()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = m(x, "image", decode=True).reconstruction
        m.patchify.pos2d.zero_()
        b = m(x, "image", decode=True).reconstruction
    assert not torch.allclose(a, b), "zeroing pos2d must change the output (otherwise it is dead)"


def test_eval_forward_is_deterministic():
    """VAEHead must decode from mu in eval mode — sampling there made every eval number noisy."""
    m = _trained_like_model()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a = m(x, "image", decode=True); b = m(x, "image", decode=True)
    assert torch.equal(a.z, b.z) and torch.equal(a.reconstruction, b.reconstruction)
    m.train()
    with torch.no_grad():
        c = m(x, "image", decode=False); d = m(x, "image", decode=False)
    assert not torch.equal(c.z, d.z), "training mode must still sample z"
