"""splat_index = projected: the splat window of an RGB query is built from the strata the grains project INTO on the
decoded observation (bucketed, `splat_cap` grains per stratum), not from the stratum each grain came from on the
anchor. Identical to the anchor indexing when nothing moved; follows transported grains otherwise. Tiny config, CPU."""
import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.grain_prior import geometry as g
from mavt.grain_prior.splat import GrainSplat

from .test_grain_prior import _inputs, _wake
from .test_splat_decoder import SPLAT

PROJ = dict(SPLAT, stratum=1, splat_window=1, splat_index="projected", splat_cap=4)   # grid 4: 16 grains, 16 strata


def _splat(**kw):
    torch.manual_seed(0)
    return GrainSplat(tiny_config(**{**PROJ, **kw}))


def _grains(cfg, B=2, shift=0.0):
    """One grain per stratum at the anchor cell centres, shifted by `shift` strata along x on the observation."""
    G = cfg.n_grains
    u = g.cell_centres(cfg.grid).unsqueeze(0).expand(B, -1, -1).clone()
    u[..., 0] = u[..., 0] + shift / cfg.n_strata
    sigma = torch.full((B, G), 0.3)
    depth = 1 + 0.1 * torch.rand(B, G)
    valid = torch.ones(B, G, dtype=torch.bool)
    return u, sigma, depth, valid


def test_default_index_is_anchor_and_adds_no_parameters():
    cfg = tiny_config(**SPLAT)
    assert cfg.splat_index == "anchor"
    a, p = _splat(splat_index="anchor"), _splat()
    assert set(a.state_dict()) == set(p.state_dict())


def test_projected_equals_anchor_indexing_when_grains_stay_in_their_stratum():
    a, p = _splat(splat_index="anchor"), _splat()
    p.load_state_dict(a.state_dict())
    cfg = p.cfg
    u, sigma, depth, valid = _grains(cfg)
    B, G = u.shape[:2]
    f = torch.randn(B, G, cfg.d_f)
    c0 = z = torch.randn(B, cfg.n_coarse, cfg.d_coarse)
    scale = torch.ones(B)
    uq = g.cell_centres(cfg.query_grid).unsqueeze(0).expand(B, -1, -1)
    with torch.no_grad():
        xa, _ = a(f, u, depth, sigma, valid, scale, c0, z, uq)
        xp, _ = p(f, u, depth, sigma, valid, scale, c0, z, uq)
    assert torch.allclose(xa, xp, atol=1e-5)


def test_projected_window_follows_a_transported_grain():
    """Every grain moves 2 strata to the right (w = 1). Anchor indexing: no grain of a query's window projects into
    the query's own stratum. Projected indexing: the heaviest grain of every query is the one projecting into it."""
    cfg = tiny_config(**PROJ)
    a, p = _splat(splat_index="anchor"), _splat()
    u, sigma, depth, valid = _grains(cfg, shift=2.0)
    ns = cfg.n_strata
    uq = g.cell_centres(cfg.query_grid).unsqueeze(0).expand(2, -1, -1)
    q_s = (uq[..., 1] * ns).floor().long() * ns + (uq[..., 0] * ns).floor().long()        # stratum of each query
    g_s = (u[..., 1] * ns).floor().long() * ns + (u[..., 0] * ns).floor().long()   # off-screen grains: x >= ns, no query
    wa, ia, oka = a.weights(uq, u, sigma, depth, valid)
    wp, ip, okp = p.weights(uq, u, sigma, depth, valid)
    # queries in strata with x >= 2 have exactly one grain projecting into their stratum
    has = (uq[..., 0] * ns).floor() >= 2
    best_a = torch.gather(ia, 2, wa.argmax(-1, keepdim=True)).squeeze(-1)
    best_p = torch.gather(ip, 2, wp.argmax(-1, keepdim=True)).squeeze(-1)
    same_a = torch.gather(g_s, 1, best_a) == q_s
    same_p = torch.gather(g_s, 1, best_p) == q_s
    assert not same_a[has].any()
    assert same_p[has].all()
    assert torch.allclose(wp.sum(-1)[has], torch.ones_like(wp.sum(-1)[has]), atol=1e-5)


def test_projected_index_caps_collisions_and_reports_the_dropped_share():
    cfg = tiny_config(**{**PROJ, "splat_cap": 2})
    p = _splat(splat_cap=2)
    u, sigma, depth, valid = _grains(cfg)
    u[:] = u[:, :1]                                                  # every grain lands in stratum 0
    uq = g.cell_centres(cfg.query_grid).unsqueeze(0).expand(2, -1, -1)
    w, idx, ok = p.weights(uq, u, sigma, depth, valid)
    assert ok.sum(-1).max() == 2                                     # at most splat_cap grains per window stratum
    assert 0 < p.last_dropped_frac < 1


def test_grains_behind_the_camera_or_off_screen_enter_no_window():
    cfg = tiny_config(**PROJ)
    p = _splat()
    u, sigma, depth, valid = _grains(cfg)
    valid[:, :4] = False
    u[:, 4:8, 0] = 1.5                                                # off screen
    uq = g.cell_centres(cfg.query_grid).unsqueeze(0).expand(2, -1, -1)
    w, idx, ok = p.weights(uq, u, sigma, depth, valid)
    used = idx[ok]
    assert not torch.isin(used, torch.arange(8)).any()


@pytest.mark.parametrize("modality", ["image", "video", "threed"])
def test_model_decodes_and_learns_with_the_projected_index(modality):
    torch.manual_seed(0)
    m = _wake(MAVTGrainPrior(tiny_config(**PROJ)))
    x, cam = _inputs(modality, B=2, T=2)
    out = m(x, modality, cam)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith("splat.amp"))
    assert torch.isfinite(out["diag"]["splat_dropped_frac"])
