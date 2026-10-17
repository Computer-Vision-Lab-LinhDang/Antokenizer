"""The C write of a v3 cell reads only the change signal, never the evidence."""
import pytest
import torch

from mavt4d.hybrid.cells import HybridCoupledCell, dc_feedback_ratio
from mavt4d.hybrid.config import HybridConfig, HybridV3Config

T, B = 3, 2


def make(innovation=True, seed=0):
    torch.manual_seed(seed)
    cfg = HybridV3Config.tiny(innovation=True) if innovation else HybridConfig.tiny()
    return HybridCoupledCell(cfg), cfg


def inputs(cfg, seed=1, observed=None):
    g = torch.Generator().manual_seed(seed)
    shape = (T, B, cfg.summary_tokens, cfg.width)
    ev = torch.randn(shape, generator=g)
    inn = torch.randn(shape, generator=g)
    obs = torch.ones(T, B, dtype=torch.bool) if observed is None else observed
    dt = torch.full((T, B), 0.5, dtype=torch.float64)
    c0 = torch.randn(B, cfg.state_tokens, cfg.state_dim, generator=g)
    d0 = torch.randn(B, cfg.state_tokens, cfg.state_dim, generator=g)
    return ev, inn, obs, dt, c0, d0


def coef(cell, ev, obs, dt, inn):
    return cell.coefficients(ev.flatten(0, 1), obs.flatten(), dt.flatten(), innovation=inn.flatten(0, 1))


def test_v2_cell_has_no_new_parameters():
    cell, _ = make(innovation=False)
    assert not any(k.startswith("innovation_") for k in cell.state_dict())
    v3, _ = make(innovation=True)
    assert {k for k in v3.state_dict()} - {k for k in cell.state_dict()} == {
        "innovation_attention.in_proj_weight", "innovation_attention.out_proj.weight",
        "innovation_head.weight"}


def test_v2_cell_state_dict_unchanged():
    cell, cfg = make(innovation=False)
    keys = list(cell.state_dict())
    assert keys == ["write_queries", "missing_rotation", "missing_damping",
                    "evidence_norm.weight", "evidence_norm.bias",
                    "write_attention.in_proj_weight", "write_attention.in_proj_bias",
                    "write_attention.out_proj.weight", "write_attention.out_proj.bias",
                    "time_projection.weight", "time_projection.bias",
                    "write_head.0.weight", "write_head.0.bias", "write_head.2.weight", "write_head.2.bias"]
    ev, _, obs, dt, c0, d0 = inputs(cfg)
    c, d = cell.scan(c0, d0, ev, obs, dt)
    assert c.shape == (T, B, cfg.state_tokens, cfg.state_dim)


def test_c_write_ignores_evidence_given_innovation():
    cell, cfg = make()
    ev, inn, obs, dt, _, _ = inputs(cfg)
    ev2 = ev + torch.randn_like(ev)
    a1, b1 = coef(cell, ev, obs, dt, inn)
    a2, b2 = coef(cell, ev2, obs, dt, inn)
    # alpha is the only evidence-dependent factor of the C write: recover it from the D write path
    ratio1 = _alpha(cell, ev, obs, dt)
    ratio2 = _alpha(cell, ev2, obs, dt)
    assert torch.allclose(b1[..., 0] / ratio1, b2[..., 0] / ratio2, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(b1[..., 1], b2[..., 1])


def _alpha(cell, ev, obs, dt):
    cfg = cell.config
    flat = ev.flatten(0, 1)
    active = cell.evidence_norm(flat)
    q = cell.write_queries[None].expand(len(flat), -1, -1)
    ctx, _ = cell.write_attention(q, active, active, need_weights=False)
    tc = cell.time_projection(torch.log1p(dt.flatten()).float()[:, None])
    raw = cell.write_head(ctx + tc[:, None]).reshape(len(flat), cfg.state_tokens, cfg.state_dim, 7)
    return torch.sigmoid(raw[..., 6])


def test_zero_innovation_gives_zero_c_write():
    cell, cfg = make()
    ev, inn, obs, dt, _, _ = inputs(cfg)
    _, b = coef(cell, ev, obs, dt, torch.zeros_like(inn))
    assert bool((b[..., 0] == 0).all())
    assert bool((b[..., 1] != 0).any())


def test_innovation_required_when_enabled():
    cell, cfg = make()
    ev, inn, obs, dt, _, _ = inputs(cfg)
    with pytest.raises(ValueError):
        cell.coefficients(ev.flatten(0, 1), obs.flatten(), dt.flatten())
    with pytest.raises(ValueError):
        cell.coefficients(ev.flatten(0, 1), obs.flatten(), dt.flatten(), innovation=inn.flatten(0, 1)[:-1])
    with pytest.raises(ValueError):
        cell.coefficients(ev.flatten(0, 1), obs.flatten(), dt.flatten(), innovation=inn.flatten(0, 1).double())
    bad = inn.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        coef(cell, ev, obs, dt, bad)


def test_innovation_rejected_when_disabled():
    for cell in (make(innovation=False)[0], HybridCoupledCell(HybridV3Config.tiny(appearance_tokens=2))):
        cfg = cell.config
        ev, inn, obs, dt, _, _ = inputs(cfg)
        with pytest.raises(ValueError):
            coef(cell, ev, obs, dt, inn)


def test_missing_rows_ignore_innovation():
    cell, cfg = make()
    obs = torch.tensor([[True, False], [False, True], [True, True]])
    ev, inn, _, dt, _, _ = inputs(cfg)
    a1, b1 = coef(cell, ev, obs, dt, inn)
    a2, b2 = coef(cell, ev, obs, dt, inn * 7 + 3)
    miss = ~obs.flatten()
    assert bool((b1[miss] == 0).all()) and bool((b2[miss] == 0).all())
    assert torch.equal(a1[miss], a2[miss])
    # nonfinite innovation on a missing row is ignored
    bad = inn.clone()
    bad[0, 1] = float("nan")
    coef(cell, ev, obs, dt, bad)


@pytest.mark.parametrize("pattern", ["all", "mixed"])
def test_scan_parity_with_innovation(pattern):
    cell, cfg = make()
    obs = None if pattern == "all" else torch.tensor([[True, False], [True, True], [False, True]])
    ev, inn, obs, dt, c0, d0 = inputs(cfg, observed=obs)
    outs, grads = [], []
    for method in ("sequential", "associative"):
        cell.zero_grad()
        e = ev.clone().requires_grad_()
        i = inn.clone().requires_grad_()
        c, d = cell.scan(c0, d0, e, obs, dt, innovation=i, method=method)
        (c.square().sum() + d.sum()).backward()
        outs.append((c, d))
        grads.append([e.grad, i.grad] + [p.grad.clone() for p in cell.parameters()])
    for x, y in zip(outs[0], outs[1]):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
    for x, y in zip(*grads):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)


def test_dc_feedback_ratio_matches_definition():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(5, 3, 2, 2, generator=g)
    prev = torch.randn(5, 3, 2, generator=g)
    b = torch.randn(5, 3, 2, generator=g)
    num = (a[..., 0, 1] * prev[..., 1]).square().mean().sqrt()
    den = b[..., 0].square().mean().sqrt()
    torch.testing.assert_close(dc_feedback_ratio(a, prev, b), num / den)
    assert torch.isfinite(dc_feedback_ratio(a, prev, torch.zeros_like(b)))


def test_scan_diagnostics_dc_feedback_ratio():
    cell, cfg = make()
    obs = torch.tensor([[True, False], [True, True], [False, True]])
    ev, inn, obs, dt, c0, d0 = inputs(cfg, observed=obs)
    base = cell.scan(c0, d0, ev, obs, dt, innovation=inn)
    ratios = []
    for method in ("sequential", "associative"):
        diag = {}
        out = cell.scan(c0, d0, ev, obs, dt, innovation=inn, method=method, diagnostics=diag)
        torch.testing.assert_close(out, base, rtol=2e-5, atol=2e-6)
        r = diag["dc_feedback_ratio"]
        assert r.ndim == 0 and torch.isfinite(r) and not r.requires_grad
        ratios.append(r)
    torch.testing.assert_close(ratios[0], ratios[1], rtol=1e-4, atol=1e-6)
    # equals the helper on the observed rows
    a, b = coef(cell, ev, obs, dt, inn)
    a, b = a.reshape(T, B, *a.shape[1:]), b.reshape(T, B, *b.shape[1:])
    c, d = base
    vals = torch.stack((c, d), dim=-1)
    prev = torch.cat((torch.stack((c0, d0), dim=-1)[None], vals[:-1]))
    m = obs
    torch.testing.assert_close(ratios[0], dc_feedback_ratio(a[m], prev[m], b[m]), rtol=1e-4, atol=1e-6)
    diag = {}
    cell.scan(c0, d0, ev, torch.zeros_like(obs), dt, innovation=inn, diagnostics=diag)
    assert "dc_feedback_ratio" not in diag
