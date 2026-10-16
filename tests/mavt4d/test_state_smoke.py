"""State model on small random inputs (no synthetic scene): shapes, missing views, seeding, gradients."""
from __future__ import annotations

import pytest
import torch

from mavt4d.model.state import StateConfig, StateModel, state_loss

CFG = StateConfig(dim=32, heads=4, depth=1, n_state=4, z_dim=8, readout_depth=1)
B, V, T, H, W = 2, 3, 3, 32, 48


def _rotation(g: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g))
    q = q * torch.sign(torch.diagonal(r))
    return q * torch.det(q).sign()


def random_inputs(seed: int = 0, b: int = B, v: int = V, t: int = T, h: int = H, w: int = W) -> dict:
    g = torch.Generator().manual_seed(seed)
    K = torch.tensor([[40.0, 0, w / 2], [0, 40.0, h / 2], [0, 0, 1]]).expand(b, v, 3, 3).clone()
    K[..., :2, :2] *= 1 + 0.2 * torch.rand(b, v, 1, 1, generator=g)
    c2w = torch.eye(4).repeat(b, v, t, 1, 1)
    for i in range(b):
        for j in range(v):
            for k in range(t):
                c2w[i, j, k, :3, :3] = _rotation(g)
    c2w[..., :3, 3] = torch.randn(b, v, t, 3, generator=g) * 5
    return {
        "images": torch.rand(b, v, t, 3, h, w, generator=g) * 2 - 1,
        "K": K,
        "c2w": c2w,
        "times": torch.cumsum(0.05 + 0.5 * torch.rand(b, t, generator=g), 1),
        "visible": torch.ones(b, v, t, dtype=torch.bool),
        "depth": 1 + 50 * torch.rand(b, v, t, h, w, generator=g),
        "valid": torch.rand(b, v, t, h, w, generator=g) > 0.2,
    }


def make_model(seed: int = 0, train: bool = False) -> StateModel:
    torch.manual_seed(seed)
    return StateModel(CFG).train(train)


def run(model: StateModel, x: dict, **override):
    x = {**x, **override}
    return model(x["images"], x["K"], x["c2w"], x["times"], x["visible"])


# ---------------------------------------------------------------- shapes


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("b,v,t", [(1, 1, 1), (2, 3, 3)])
def test_forward_shapes_and_finite(train, b, v, t):
    x = random_inputs(b=b, v=v, t=t)
    out = run(make_model(train=train), x)
    assert out.rgb.shape == (b, v, t, 3, H, W)
    assert out.log_depth.shape == (b, v, t, 1, H, W)
    assert out.states.shape == (b, t, CFG.n_state, CFG.dim)
    assert out.z.shape == (b, t, CFG.n_state, CFG.z_dim)
    assert out.kl.shape == ()
    for name in ("rgb", "log_depth", "states", "z", "kl"):
        assert torch.isfinite(getattr(out, name)).all(), name
    assert out.rgb.abs().max() <= 1.0


def test_image_size_must_be_patch_multiple():
    x = random_inputs(h=40)
    with pytest.raises(ValueError, match="multiple of patch"):
        run(make_model(), x)


# ---------------------------------------------------------------- masked / missing views


def test_nothing_visible_keeps_initial_state():
    """No evidence at any step: the state stays at state_init (predict is the identity at init)."""
    x = random_inputs()
    m = make_model()
    out = run(m, x, visible=torch.zeros(B, V, T, dtype=torch.bool))
    init = m.state_init.expand(B, T, -1, -1)
    assert torch.equal(out.states, init)
    assert torch.isfinite(out.rgb).all() and torch.isfinite(out.log_depth).all()


def test_batch_elements_are_independent_under_mixed_visibility():
    """Sample 0 sees everything, sample 1 nothing at step 1: each matches running it alone."""
    x = random_inputs()
    vis = torch.ones(B, V, T, dtype=torch.bool)
    vis[1, :, 1] = False
    vis[1, 0, 2] = False
    m = make_model()
    out = run(m, x, visible=vis)
    for i in range(B):
        one = {k: v[i:i + 1] for k, v in x.items()}
        alone = run(m, one, visible=vis[i:i + 1])
        torch.testing.assert_close(out.states[i:i + 1], alone.states)
        torch.testing.assert_close(out.rgb[i:i + 1], alone.rgb)
    assert torch.equal(out.states[1, 1], out.states[1, 0])      # fully hidden step: carried over
    assert not torch.equal(out.states[0, 1], out.states[0, 0])  # observed step: updated


def test_hidden_view_content_never_reaches_outputs():
    """Hidden views may carry arbitrary (even non-finite) pixels, as a missing camera would."""
    x = random_inputs()
    vis = torch.ones(B, V, T, dtype=torch.bool)
    vis[:, 2] = False
    vis[0, 1, 0] = False
    m = make_model()
    a = run(m, x, visible=vis)
    junk = x["images"].clone()
    junk[~vis] = float("nan")
    b = run(m, x, images=junk, visible=vis)
    assert torch.equal(a.states, b.states) and torch.equal(a.rgb, b.rgb)


def test_hidden_view_gets_no_gradient():
    x = random_inputs()
    vis = torch.ones(B, V, T, dtype=torch.bool)
    vis[:, 1, 1] = False
    images = x["images"].clone().requires_grad_(True)
    out = run(make_model(train=True), x, images=images, visible=vis)
    state_loss(out, x["images"], x["depth"], x["valid"])["loss"].backward()
    assert torch.equal(images.grad[:, 1, 1], torch.zeros_like(images.grad[:, 1, 1]))
    assert images.grad[:, 0, 1].abs().sum() > 0


# ---------------------------------------------------------------- determinism


@pytest.mark.parametrize("train", [False, True])
def test_same_seed_same_outputs(train):
    x = random_inputs()
    outs = []
    for _ in range(2):
        m = make_model(seed=3, train=train)
        torch.manual_seed(7)  # VAE sampling in train mode
        outs.append(run(m, x))
    for name in ("rgb", "log_depth", "states", "z", "kl"):
        assert torch.equal(getattr(outs[0], name), getattr(outs[1], name)), name


def test_eval_is_deterministic_without_reseeding():
    x = random_inputs()
    m = make_model()
    a, b = run(m, x), run(m, x)
    assert torch.equal(a.rgb, b.rgb) and torch.equal(a.z, b.z)


def test_train_mode_samples_z():
    x = random_inputs()
    m = make_model(train=True)
    a, b = run(m, x), run(m, x)
    assert torch.equal(a.states, b.states)
    assert not torch.equal(a.z, b.z)


def test_different_seeds_different_init():
    x = random_inputs()
    assert not torch.equal(run(make_model(0), x).states, run(make_model(1), x).states)


# ---------------------------------------------------------------- gradients


def _backward(vis: torch.Tensor, cells: torch.Tensor | None = None) -> StateModel:
    x = random_inputs()
    m = make_model(train=True)
    out = run(m, x, visible=vis)
    losses = state_loss(out, x["images"], x["depth"], x["valid"], cells)
    assert all(torch.isfinite(v) for v in losses.values())
    losses["loss"].backward()
    return m


def test_backward_all_gradients_finite():
    vis = torch.ones(B, V, T, dtype=torch.bool)
    vis[:, 2, 1:] = False
    vis[1, :, 2] = False
    m = _backward(vis)
    # stem.registers is (0, dim): the stem is built with n_registers=0 and its registers are never read.
    missing = [n for n, p in m.named_parameters() if p.grad is None and p.numel()]
    assert not missing, missing
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, bad
    for name in ("stem.proj.weight", "state_init", "mask_token", "predict.2.weight", "obs_geometry.0.weight",
                 "query_geometry.0.weight", "vae.proj.weight", "to_pixels.weight", "blocks.0.attn.log_scale"):
        assert dict(m.named_parameters())[name].grad.abs().sum() > 0, name


def test_backward_finite_with_nothing_visible_and_partial_supervision():
    cells = torch.zeros(B, V, T, dtype=torch.bool)
    cells[0, 0, 0] = True
    m = _backward(torch.zeros(B, V, T, dtype=torch.bool), cells)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.state_init.grad.abs().sum() > 0


def test_backward_finite_with_no_supervised_cells():
    m = _backward(torch.ones(B, V, T, dtype=torch.bool), torch.zeros(B, V, T, dtype=torch.bool))
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
