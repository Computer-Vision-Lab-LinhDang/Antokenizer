"""3.10 photometric warp loss on the transported grains (self-supervised camera / motion / parallax signal), tiny
config, CPU. Default off (w_warp = 0): no new parameter, the default loss unchanged."""
import math

import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.grain_prior import geometry as g
from mavt.grain_prior.losses import warp_loss

from .test_grain_prior import _inputs, _wake

GRID = 4


def _model(**kw):
    torch.manual_seed(0)
    return MAVTGrainPrior(tiny_config(**kw))


def _translated_clip(shift_cells=1, D=8, f=1.0):
    """Anchor grains at the cell centres of a GRID x GRID feature map at depth 1; the target camera is translated so
    every grain lands exactly one cell to the right: F_t is F0 rolled by one column."""
    torch.manual_seed(0)
    F0 = torch.randn(1, GRID * GRID, D)
    F_t = F0.view(1, GRID, GRID, D).roll(shift_cells, dims=2).reshape(1, -1, D)
    u0 = g.cell_centres(GRID).unsqueeze(0)
    kappa = torch.tensor([[f, f, 0.5, 0.5]])
    P = g.unproject(torch.ones(1, GRID * GRID), kappa, u0)
    xi_true = torch.tensor([[shift_cells / GRID / f, 0, 0, 0, 0, 0.0]])        # +x in the target camera, z = 1
    return F0, F_t, u0, P, kappa, xi_true


def _warp_at(xi, F0, F_t, u0, P, kappa):
    R, t = g.se3_exp(xi)
    G = P.shape[1]
    tr = g.transport_grains(P, torch.zeros(1, G, 1), torch.zeros(1, 1, 3), R, t, kappa, 1e-2)
    return warp_loss(F0, F_t, u0, tr["u"], tr["valid"])


# ------------------------------------------------------------------------------------------------ 3.10
def test_warp_loss_is_small_at_the_true_camera_and_large_at_identity():
    F0, F_t, u0, P, kappa, xi_true = _translated_clip()
    at_true = _warp_at(xi_true, F0, F_t, u0, P, kappa)
    at_zero = _warp_at(torch.zeros(1, 6), F0, F_t, u0, P, kappa)
    assert at_true < 1e-4 < 0.5 < at_zero


def test_warp_loss_gradient_reaches_the_camera_and_vanishes_at_the_true_camera():
    F0, F_t, u0, P, kappa, xi_true = _translated_clip()
    xi = torch.zeros(1, 6, requires_grad=True)
    _warp_at(xi, F0, F_t, u0, P, kappa).backward()
    assert xi.grad.abs().sum() > 0 and torch.isfinite(xi.grad).all()
    xi2 = xi_true.clone().requires_grad_(True)
    _warp_at(xi2, F0, F_t, u0, P, kappa).backward()
    assert xi2.grad.abs().max() < 1e-3 * xi.grad.abs().max()


def test_warp_loss_ignores_grains_that_leave_the_image_or_are_invalid():
    F0, F_t, u0, P, kappa, xi_true = _translated_clip()
    R, t = g.se3_exp(xi_true)
    G = P.shape[1]
    tr = g.transport_grains(P, torch.zeros(1, G, 1), torch.zeros(1, 1, 3), R, t, kappa, 1e-2)
    off = tr["u"][..., 0] > 1
    assert off.sum() == GRID                                            # the last column left the image
    full = warp_loss(F0, F_t, u0, tr["u"], tr["valid"])
    masked = warp_loss(F0, F_t, u0, tr["u"], tr["valid"] & ~off)
    assert torch.allclose(full, masked) and full < 1e-4                 # their border read (a mismatch) is not counted
    valid = tr["valid"].clone()
    valid[:, 0] = False
    F_bad = F_t.clone()
    F_bad[:, 1] = -F_bad[:, 1]                                          # only grain 0 lands in cell 1: invalid, unread
    assert warp_loss(F0, F_bad, u0, tr["u"], valid) < 1e-4
    assert warp_loss(F0, F_t, u0, tr["u"], torch.zeros_like(valid)) == 0


@pytest.mark.parametrize("modality", ["video", "threed"])
def test_w_warp_trains_the_transport_and_reports_warp_err(modality):
    m = _wake(_model(w_warp=1.0, w_motion=0.0)).train()
    x, cam = _inputs(modality, B=2, T=2)
    out = m(x, modality, cam)
    assert torch.isfinite(out["loss"]) and "warp" in out["parts"] and torch.isfinite(out["diag"]["warp_err"])
    out["loss"].backward()
    pre = "anchor.w_rho" if modality == "threed" else "upd."           # posed views: depth; video: camera update
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith(pre))


def test_w_warp_off_by_default_and_leaves_the_loss_alone():
    cfg = tiny_config()
    assert cfg.w_warp == 0.0 and cfg.w_motion == 1e-3
    m = _wake(_model())
    x, cam = _inputs("video", B=2, T=2)
    out = m(x, "video", cam)
    assert "warp" not in out["parts"] and "warp_err" not in out["diag"]
    with pytest.raises(ValueError):
        tiny_config(w_warp=-1.0).check()
