"""Time-indexed state model: causal by construction, readout only through z_t, hidden evidence is ignored."""
from __future__ import annotations

import pytest
import torch

from mavt4d.data.synthetic import SyntheticSceneConfig, make_synthetic_scene
from mavt4d.model.state import StateConfig, StateModel, state_loss

SCENE = SyntheticSceneConfig(V=2, T=3, H=32, W=48)
CFG = StateConfig(dim=64, heads=4, depth=2, n_state=8, z_dim=8, readout_depth=1)


@pytest.fixture(scope="module")
def batch():
    samples = [make_synthetic_scene(SCENE, seed=s) for s in (0, 1)]
    images = torch.stack([s.images for s, _ in samples])                # (B, V, T, 3, H, W)
    K = torch.stack([s.intrinsics for s, _ in samples])                 # (B, V, 3, 3)
    c2w = torch.stack([s.cam_to_world for s, _ in samples])             # (B, V, T, 4, 4)
    times = torch.stack([s.frame_times[0] for s, _ in samples]).float()  # (B, T)
    depth = torch.stack([g.depth for _, g in samples])                  # (B, V, T, H, W)
    valid = torch.stack([g.valid for _, g in samples])
    return images, K, c2w, times, depth, valid


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return StateModel(CFG).eval()


def _all_visible(images):
    B, V, T = images.shape[:3]
    return torch.ones(B, V, T, dtype=torch.bool)


def test_shapes(model, batch):
    images, K, c2w, times, *_ = batch
    out = model(images, K, c2w, times, _all_visible(images))
    B, V, T, _, H, W = images.shape
    assert out.rgb.shape == (B, V, T, 3, H, W)
    assert out.log_depth.shape == (B, V, T, 1, H, W)
    assert out.states.shape == (B, T, CFG.n_state, CFG.dim)
    assert out.z.shape == (B, T, CFG.n_state, CFG.z_dim)
    assert out.rgb.abs().max() <= 1.0


def test_future_does_not_change_past(model, batch):
    images, K, c2w, times, *_ = batch
    vis = _all_visible(images)
    a = model(images, K, c2w, times, vis)
    future = images.clone()
    future[:, :, 2:] = torch.rand_like(future[:, :, 2:]) * 2 - 1
    b = model(future, K, c2w, times, vis)
    assert torch.equal(a.states[:, :2], b.states[:, :2])
    assert torch.equal(a.rgb[:, :, :2], b.rgb[:, :, :2])
    assert not torch.equal(a.states[:, 2], b.states[:, 2])


def test_dropped_step_ignores_its_images(model, batch):
    images, K, c2w, times, *_ = batch
    vis = _all_visible(images)
    vis[:, :, 1] = False
    a = model(images, K, c2w, times, vis)
    other = images.clone()
    other[:, :, 1] = -other[:, :, 1]
    b = model(other, K, c2w, times, vis)
    assert torch.equal(a.rgb, b.rgb) and torch.equal(a.states, b.states)


def test_hidden_view_ignores_its_image(model, batch):
    images, K, c2w, times, *_ = batch
    vis = _all_visible(images)
    vis[:, 1, 1] = False
    a = model(images, K, c2w, times, vis)
    other = images.clone()
    other[:, 1, 1] = -other[:, 1, 1]
    b = model(other, K, c2w, times, vis)
    assert torch.equal(a.rgb, b.rgb)


def test_readout_only_through_z(model, batch):
    images, K, c2w, times, *_ = batch
    out = model(images, K, c2w, times, _all_visible(images))
    H, W = images.shape[-2:]
    for t in range(images.shape[2]):
        rgb, log_depth = model.readout(out.z[:, t], K, c2w[:, :, t], H, W)
        assert torch.allclose(rgb, out.rgb[:, :, t]) and torch.allclose(log_depth, out.log_depth[:, :, t])


def test_predict_is_identity_at_init(model, batch):
    images, K, c2w, times, *_ = batch
    vis = torch.zeros_like(_all_visible(images))
    vis[:, :, 0] = True                      # observe only the first step
    out = model(images, K, c2w, times, vis)
    assert torch.equal(out.states[:, 1], out.states[:, 0])
    assert torch.equal(out.states[:, 2], out.states[:, 0])


def test_predict_uses_time_gap(batch):
    images, K, c2w, times, *_ = batch
    torch.manual_seed(0)
    m = StateModel(CFG).eval()
    for p in m.predict.parameters():
        torch.nn.init.normal_(p, std=0.1)
    vis = torch.zeros_like(_all_visible(images))
    vis[:, :, 0] = True
    a = m(images, K, c2w, times, vis)
    b = m(images, K, c2w, times * 3, vis)
    assert torch.equal(a.states[:, 0], b.states[:, 0])
    assert not torch.allclose(a.states[:, 1], b.states[:, 1])


def test_loss_backward_reaches_stem_and_state_init(batch):
    images, K, c2w, times, depth, valid = batch
    torch.manual_seed(0)
    m = StateModel(CFG).train()
    out = m(images, K, c2w, times, _all_visible(images))
    losses = state_loss(out, images, depth, valid)
    assert set(losses) >= {"loss", "rgb", "depth", "kl"}
    assert all(torch.isfinite(v) for v in losses.values())
    losses["loss"].backward()
    assert m.stem.proj.weight.grad.abs().sum() > 0
    assert m.state_init.grad.abs().sum() > 0
