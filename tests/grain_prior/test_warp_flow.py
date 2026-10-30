"""3.12 correspondence target for the transport (w_warp_flow): each anchor grain's DINO feature is matched against the
target feature grid inside a window of +-warp_radius cells around its anchor position (soft-argmax at temperature
warp_tau, detached); the transported position is pulled to that match with a Huber on the cell displacement,
weighted by the match confidence. Unlike the bilinear warp (3.10) the pull has a constant size and a capture range of
warp_radius cells. Tiny config, CPU; default off."""
import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.grain_prior import geometry as g
from mavt.grain_prior.losses import warp_flow_loss, warp_match

from .test_grain_prior import _inputs, _wake

GRID = 8


def _shifted(shift=(2, 0), D=16, seed=0):
    """F_t = F0 rolled by shift (cells along x, y); grains at the cell centres."""
    torch.manual_seed(seed)
    F0 = torch.randn(1, GRID * GRID, D)
    F_t = F0.view(1, GRID, GRID, D).roll(shifts=(shift[1], shift[0]), dims=(1, 2)).reshape(1, -1, D)
    u0 = g.cell_centres(GRID).unsqueeze(0)
    return F0, F_t, u0


def test_match_finds_a_two_cell_shift():
    F0, F_t, u0 = _shifted((2, 0))
    u_star, conf = warp_match(F0, F_t, u0, radius=3, tau=0.05)
    inner = (u0[..., 0] < 1 - 2.5 / GRID)                                  # grains whose match stays in the image
    d = (u_star - u0) * GRID
    assert torch.allclose(d[inner][:, 0], torch.full_like(d[inner][:, 0], 2.0), atol=0.05)
    assert torch.allclose(d[inner][:, 1], torch.zeros_like(d[inner][:, 1]), atol=0.05)
    assert (conf[inner] > 0.9).all()
    assert not u_star.requires_grad and not conf.requires_grad


def test_flow_loss_pulls_the_transport_toward_the_match_with_a_constant_size():
    """At identity (u_t = u0) the bilinear warp has no useful gradient for a 2-cell shift; the flow loss does, and it
    points toward the match."""
    F0, F_t, u0 = _shifted((2, 0))
    u_t = u0.clone().requires_grad_(True)
    valid = torch.ones(1, GRID * GRID, dtype=torch.bool)
    warp_flow_loss(F0, F_t, u0, u_t, valid, radius=3, tau=0.05).backward()
    inner = (u0[..., 0] < 1 - 2.5 / GRID) & (u0[..., 0] > 0.5 / GRID)
    gx = u_t.grad[..., 0][inner]
    assert (gx < 0).all()                                                  # gradient descent moves u_t to +x
    assert gx.abs().min() > 0.2 * gx.abs().max()                          # comparable pull on every grain


def test_flow_loss_is_zero_at_the_match_and_ignores_invalid_grains():
    F0, F_t, u0 = _shifted((1, 1))
    u_star, _ = warp_match(F0, F_t, u0, radius=3, tau=0.05)
    valid = torch.ones(1, GRID * GRID, dtype=torch.bool)
    assert warp_flow_loss(F0, F_t, u0, u_star, valid, radius=3, tau=0.05) < 1e-6
    far = u0 + 0.3
    assert warp_flow_loss(F0, F_t, u0, far, torch.zeros_like(valid), radius=3, tau=0.05) == 0


def test_w_warp_flow_is_off_by_default_and_validated():
    cfg = tiny_config()
    assert cfg.w_warp_flow == 0.0
    m = _wake(MAVTGrainPrior(cfg))
    x, cam = _inputs("video", B=2, T=2)
    out = m(x, "video", cam)
    assert "warp_flow" not in out["parts"]
    with pytest.raises(ValueError):
        tiny_config(w_warp_flow=-1.0).check()
    with pytest.raises(ValueError):
        tiny_config(warp_radius=0).check()


def test_w_warp_flow_trains_the_camera_update_and_reports_the_grain_shift():
    torch.manual_seed(0)
    m = _wake(MAVTGrainPrior(tiny_config(w_warp_flow=1.0, w_motion=0.0))).train()
    x, cam = _inputs("video", B=2, T=2)
    out = m(x, "video", cam)
    assert torch.isfinite(out["loss"]) and "warp_flow" in out["parts"]
    for k in ("warp_flow_err", "grain_shift_cells", "match_shift_cells"):
        assert torch.isfinite(out["diag"][k])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("upd."))
