"""Availability-aware patch innovation, change pooling and observer patches."""

from dataclasses import replace

import pytest
import torch

from mavt4d.contracts import ObservationStep
from mavt4d.hybrid.config import HybridV3Config
from mavt4d.hybrid.innovation import InnovationPool, patch_innovation
from mavt4d.hybrid.observation import HybridObservationEncoder

T, B, V, HP, WP, S = 4, 2, 1, 3, 4, 24


def cfg(**kw):
    return HybridV3Config.tiny(innovation=True, **kw)


def run(patches, coverage, *, appearance_on=False, first=None):
    t, b = patches.shape[:2]
    if first is None:
        first = torch.zeros(t, b, dtype=torch.bool)
        first[0] = True
    prev = torch.zeros(b, *patches.shape[2:])
    seen = torch.zeros(coverage.shape[1:2] + coverage.shape[2:], dtype=torch.bool)
    return patch_innovation(patches, coverage, prev, seen, first, appearance_on=appearance_on)


def static_patches(seed=0):
    x = torch.randn(1, B, V, HP, WP, S, generator=torch.Generator().manual_seed(seed))
    return x.expand(T, -1, -1, -1, -1, -1).clone()


def make_pool(**kw):
    torch.manual_seed(3)
    return InnovationPool(cfg(**kw))


def test_static_clip_gives_zero_after_first_step():
    x, cov = static_patches(), torch.ones(T, B, V, HP, WP)
    delta, _, _ = run(x, cov)
    assert torch.count_nonzero(delta[1:]) == 0
    pool = make_pool()
    evidence = torch.randn(B, 2, 16)
    for t in range(1, T):
        out = pool(delta[t], cov[t] >= 1, evidence)
        assert torch.count_nonzero(out) == 0 and torch.isfinite(out).all()


def test_mask_on_off_is_not_change_synthetic():
    x, cov = static_patches(1), torch.ones(T, B, V, HP, WP)
    cov[1:3, :, :, :2, :2] = 0
    delta, prev, seen = run(x, cov)
    assert torch.count_nonzero(delta[1:]) == 0
    assert torch.equal(prev, x[0]) and seen.all()


def test_first_seen_patch_is_new_evidence():
    x, cov = torch.randn(T, B, V, HP, WP, S), torch.ones(T, B, V, HP, WP)
    cov[:2, :, :, 0, 0] = 0
    delta, _, _ = run(x, cov, appearance_on=True)
    assert torch.equal(delta[2, :, :, 0, 0], x[2, :, :, 0, 0])
    assert torch.count_nonzero(delta[0, :, :, 0, 0]) == 0 and torch.count_nonzero(delta[1, :, :, 0, 0]) == 0


def test_first_step_zero_with_appearance():
    x, cov = torch.randn(T, B, V, HP, WP, S), torch.ones(T, B, V, HP, WP)
    delta, prev, seen = run(x, cov, appearance_on=True)
    assert torch.count_nonzero(delta[0]) == 0 and seen.all()
    assert torch.equal(prev, x[-1])
    delta_off, _, _ = run(x, cov, appearance_on=False)
    assert torch.equal(delta_off[0], x[0])


def test_chunked_equals_full():
    x, cov = torch.randn(T, B, V, HP, WP, S), (torch.rand(T, B, V, HP, WP) > .3).float()
    first = torch.zeros(T, B, dtype=torch.bool)
    first[0] = True
    zeros = (torch.zeros(B, V, HP, WP, S), torch.zeros(B, V, HP, WP, dtype=torch.bool))
    full, p, s = patch_innovation(x, cov, *zeros, first, appearance_on=True)
    a, p1, s1 = patch_innovation(x[:2], cov[:2], *zeros, first[:2], appearance_on=True)
    b, p2, s2 = patch_innovation(x[2:], cov[2:], p1, s1, first[2:], appearance_on=True)
    assert torch.equal(torch.cat((a, b)), full) and torch.equal(p2, p) and torch.equal(s2, s)


def test_floor_bounds_small_changes():
    pool = make_pool()
    evidence = torch.randn(B, 2, 16)
    valid = torch.ones(B, V, HP, WP, dtype=torch.bool)
    captured = []
    pool.output.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    unit = torch.randn(B, V, HP, WP, S)
    pool(unit * 1e-9, valid, evidence)  # linear regime: pooled is proportional to delta
    rms_e = evidence.square().mean((1, 2), keepdim=True).sqrt()
    peak = captured[0].square().mean(-1).sqrt().amax((1,), keepdim=True)  # [B,1]
    scale = (1e-6 * rms_e.flatten() / peak.flatten()).reshape(B, 1, 1, 1, 1) * 1e-9
    out = pool(unit * scale, valid, evidence)
    pooled_rms = captured[1].square().mean(-1).sqrt().amax(1)
    torch.testing.assert_close(pooled_rms, 1e-6 * rms_e.flatten(), rtol=1e-3, atol=0)
    assert out.square().mean(-1).sqrt().max() <= 1e-3 + 1e-9


def test_zero_pool_backward_is_finite():
    for valid_all in (True, False):
        pool = make_pool()
        valid = torch.full((B, V, HP, WP), valid_all, dtype=torch.bool)
        out = pool(torch.zeros(B, V, HP, WP, S), valid, torch.randn(B, 2, 16))
        assert torch.count_nonzero(out) == 0
        out.sum().backward()
        for name, p in pool.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (valid_all, name)


def test_missing_step_floor_is_zero():
    pool = make_pool()
    out = pool(torch.zeros(B, V, HP, WP, S), torch.zeros(B, V, HP, WP, dtype=torch.bool), torch.zeros(B, 2, 16))
    assert torch.count_nonzero(out) == 0 and torch.isfinite(out).all()


def obs(images, observed, views=1, frames=2):
    times = torch.arange(1, frames + 1, dtype=torch.float64)[None].expand(views, -1).clone()
    return ObservationStep(images.clone(), observed.clone(), times)


def scene(seed=0, frames=2, h=6, w=8):
    img = torch.rand(1, 1, 3, h, w, generator=torch.Generator().manual_seed(seed)).expand(1, frames, 3, h, w)
    return img.clone(), torch.ones(1, frames, h, w, dtype=torch.bool)


def test_mask_on_off_is_not_change_real_encoder():
    torch.manual_seed(5)
    config = cfg(reference_views=1)
    encoder = HybridObservationEncoder(config).eval()
    pool = InnovationPool(config)
    img, ok = scene()
    steps = []
    for t in range(4):
        mask = ok.clone()
        if t in (1, 2):
            mask[:, :, 0:2, 0:2] = False  # one patch, shares window with observed neighbours
        steps.append(obs(img, mask))
    with torch.no_grad():
        outs = [encoder.forward_batch([s, s], modality="camera", return_patches=True) for s in steps]
        ev = torch.stack([o[0] for o in outs])
        patches = torch.stack([o[2] for o in outs])
        cov = torch.stack([o[3] for o in outs])
        assert patches.dtype == torch.float32 and cov[1, 0, 0, 0, 0] == 0
        first = torch.zeros(4, 2, dtype=torch.bool)
        first[0] = True
        delta, _, _ = patch_innovation(patches, cov, torch.zeros(2, 1, 3, 4, config.spatial_width),
                                       torch.zeros(2, 1, 3, 4, dtype=torch.bool), first, appearance_on=False)
        assert torch.count_nonzero(delta[1:]) == 0
        for t in range(1, 4):
            assert torch.count_nonzero(pool(delta[t], cov[t] >= 1, ev[t])) == 0


def test_innovation_patches_fp32_under_spatial_autocast():
    torch.manual_seed(6)
    encoder = HybridObservationEncoder(cfg()).eval()
    img, ok = scene(2)
    got = []
    for flag in (False, True):
        encoder.spatial_autocast = flag
        with torch.no_grad():
            got.append(encoder.forward_batch([obs(img, ok)], modality="camera", return_patches=True)[2])
    assert got[0].dtype == got[1].dtype == torch.float32
    assert torch.equal(got[0], got[1])


def test_return_patches_requires_single_shape():
    encoder = HybridObservationEncoder(cfg()).eval()
    a, b = scene(0), scene(1, h=8, w=8)
    with torch.no_grad(), pytest.raises(ValueError):
        encoder.forward_batch([obs(*a), obs(*b)], modality="camera", return_patches=True)


def test_return_patches_false_is_unchanged():
    torch.manual_seed(7)
    encoder = HybridObservationEncoder(cfg()).eval()
    img, ok = scene(3)
    items = [obs(img, ok), obs(img, torch.zeros_like(ok))]
    with torch.no_grad():
        plain = encoder.forward_batch(items, modality="camera")
        full = encoder.forward_batch(items, modality="camera", return_patches=True)
    assert len(plain) == 2 and len(full) == 4
    assert torch.equal(plain[0], full[0]) and torch.equal(plain[1], full[1])
    assert full[1].tolist() == [True, False]
    assert full[2].shape == (2, 1, 3, 4, 24) and torch.count_nonzero(full[2][1]) == 0
    assert full[3].shape == (2, 1, 3, 4) and torch.count_nonzero(full[3][1]) == 0


def test_return_patches_all_missing_step_is_zero():
    encoder = HybridObservationEncoder(cfg()).eval()
    img, ok = scene(4)
    with torch.no_grad():
        ev, avail, patches, cov = encoder.forward_batch([obs(img, torch.zeros_like(ok))], modality="camera",
                                                        return_patches=True)
    assert not avail.any() and patches.shape == (1, 1, 3, 4, 24)
    assert torch.count_nonzero(patches) == 0 and torch.count_nonzero(cov) == 0


def test_return_patches_evidence_bitwise_equal():
    torch.manual_seed(8)
    encoder = HybridObservationEncoder(cfg()).eval()
    img, ok = scene(5)
    for flag in (False, True):
        encoder.spatial_autocast = flag
        with torch.no_grad():
            plain = encoder.forward_batch([obs(img, ok)], modality="camera")[0]
            full = encoder.forward_batch([obs(img, ok)], modality="camera", return_patches=True)[0]
        assert torch.equal(plain, full)


def test_missing_row_zero_shape_comes_from_available_group():
    encoder = HybridObservationEncoder(cfg()).eval()
    img, ok = scene(0, h=8, w=8)
    small_img, small_ok = scene(1)
    with torch.no_grad():
        out = encoder.forward_batch([obs(small_img, torch.zeros_like(small_ok)), obs(img, ok)],
                                    modality="camera", return_patches=True)
    assert out[2].shape == (2, 1, 4, 4, 24) and torch.count_nonzero(out[2][0]) == 0
