"""3.11 relative depth loss for posed views (scale-invariant log depth or depth order instead of the absolute Huber)
and the far-depth pin of grains without a depth label (background), tiny config, CPU. Defaults unchanged."""
import math

import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.grain_prior.losses import bg_pin_loss, depth_loss

from .test_grain_prior import _inputs, _orbit, _wake


def _model(**kw):
    torch.manual_seed(0)
    return MAVTGrainPrior(tiny_config(**kw))


def _structured_depth(B=1, N=64):
    """An object at distance 2 with a thickness of +-0.25 (the Objaverse case) and a flat prediction at 2."""
    torch.manual_seed(0)
    gt = 2.0 + 0.25 * torch.sin(torch.linspace(0, 6 * math.pi, N)).expand(B, N)
    return torch.full((B, N), 2.0), gt, torch.ones(B, N, dtype=torch.bool)


def test_relative_depth_loss_rejects_flat_depth_where_gt_has_structure():
    flat, gt, m = _structured_depth()
    huber = depth_loss(flat, gt, m, "huber", 0.1)
    silog = depth_loss(flat, gt, m, "silog", 0.1)
    ordinal = depth_loss(flat, gt, m, "ordinal", 0.1)
    assert huber < 0.01                                                  # the absolute loss is at its floor
    assert silog > 10 * huber and ordinal > 10 * huber
    assert depth_loss(gt, gt, m, "silog", 0.1) < 2e-4                       # silog floors at sqrt(1e-8)
    assert depth_loss(gt, gt, m, "ordinal", 0.1) < 0.5 * ordinal          # soft margin: lower, not 0, at the truth
    assert depth_loss(3 * gt, gt, m, "silog", 0.1) < 0.01 * silog < 0.1 < depth_loss(3 * gt, gt, m, "huber", 0.1)   # scale-free


def test_unlabelled_grains_do_not_enter_the_relative_depth_loss():
    flat, gt, m = _structured_depth()
    m[:, ::2] = False                                                    # background: no label
    pred = (flat + 0.1 * torch.randn_like(flat)).requires_grad_(True)
    for kind in ("silog", "huber"):
        depth_loss(pred, gt, m, kind, 0.1).backward()
        assert pred.grad[:, ::2].abs().sum() == 0 and pred.grad[:, 1::2].abs().sum() > 0
        pred.grad = None
    moved = pred.detach().clone()
    moved[:, ::2] = 50.0
    assert torch.allclose(depth_loss(moved, gt, m, "silog", 0.1), depth_loss(pred.detach(), gt, m, "silog", 0.1))


def test_bg_pin_pulls_only_unlabelled_grains_to_the_far_depth():
    d = torch.tensor([[1.0, 4.0, 1.0, 4.0]], requires_grad=True)
    labelled = torch.tensor([[True, True, False, False]])
    bg_pin_loss(d, labelled, 4.0, 0.1).backward()
    assert d.grad[0, 0] == 0 and d.grad[0, 1] == 0                        # labelled: untouched
    assert d.grad[0, 2] < 0 and d.grad[0, 3] == 0                         # unlabelled: pulled to 4, at rest at 4
    assert bg_pin_loss(d.detach(), torch.ones_like(labelled), 4.0, 0.1) == 0


def _posed_views_with_background(B=2, V=2):
    cam = _orbit(B, V)
    S_ = cam["depth"].shape[-1]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, S_), torch.linspace(-1, 1, S_), indexing="ij")
    r2 = xx ** 2 + yy ** 2
    cam["depth"] = torch.where(r2 < 0.5, 2.0 - 0.25 * (1 - 2 * r2), torch.zeros(())).expand(B, V, -1, -1).clone()
    return cam


@pytest.mark.parametrize("kind", ["silog", "ordinal"])
def test_relative_depth_on_posed_views_trains_the_grain_depth_and_reports_the_correlation(kind):
    m = _wake(_model(depth_loss=kind, depth_bg_far=4.0)).train()
    x, _ = _inputs("threed", B=2, T=2)
    cam = _posed_views_with_background()
    out = m(x, "threed", cam)
    assert torch.isfinite(out["loss"]) and "geom" in out["parts"] and "bg_pin" in out["parts"]
    assert torch.isfinite(out["diag"]["rho_gt_corr"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("anchor.w_rho"))


def test_depth_loss_defaults_and_checks():
    cfg = tiny_config()
    assert cfg.depth_loss == "huber" and cfg.depth_bg_far == 0.0
    with pytest.raises(ValueError):
        tiny_config(depth_loss="l2").check()
    with pytest.raises(ValueError):
        tiny_config(depth_bg_far=-1.0).check()
    m = _wake(_model())
    x, _ = _inputs("threed", B=2, T=2)
    out = m(x, "threed", _posed_views_with_background())
    assert "bg_pin" not in out["parts"]


def test_both_geometry_signals_report_rho_gt_corr_and_warp_err_together():
    """Item 3: with the warp loss and the relative depth loss on, diag carries both geometry readings."""
    m = _wake(_model(w_warp=0.1, w_motion=0.0, depth_loss="silog", depth_bg_far=4.0)).train()
    x, _ = _inputs("threed", B=2, T=2)
    out = m(x, "threed", _posed_views_with_background())
    assert torch.isfinite(out["diag"]["rho_gt_corr"]) and torch.isfinite(out["diag"]["warp_err"])
    assert {"geom", "bg_pin", "warp"} <= set(out["parts"])
