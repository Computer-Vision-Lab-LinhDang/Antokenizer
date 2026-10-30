"""rgb_decoder = splat: grain context transformer (geometry-biased self-attention over grains + coarse slots), LTE-style
frequency atoms per grain, depth-aware Gaussian splat of the decoded grain features onto the RGB query grid (a
window of neighbouring strata), then the existing RGB head. Tiny config, CPU."""
import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.grain_prior.readout import geom_bias
from mavt.grain_prior.splat import GrainSplat, window_index

from .test_crsr import LEGACY_CRSR
from .test_grain_prior import _inputs, _wake

SPLAT = dict(LEGACY_CRSR, query_grid=16, rgb_head="pointwise", rgb_decoder="splat", splat_layers=2, splat_feat=8, splat_atoms=3,
             splat_window=1)


def _model(**kw):
    torch.manual_seed(0)
    return MAVTGrainPrior(tiny_config(**{**SPLAT, **kw}))


def test_default_read_decoder_has_no_splat_module():
    assert not hasattr(MAVTGrainPrior(tiny_config(**LEGACY_CRSR, query_grid=16)), "splat")


def test_window_index_covers_the_neighbouring_strata():
    idx, ok = window_index(n_side=4, w=1)                        # [16 strata, 9] grain ids, valid
    assert idx.shape == (16, 9) and ok.shape == (16, 9)
    assert set(idx[5][ok[5]].tolist()) == {0, 1, 2, 4, 5, 6, 8, 9, 10}  # stratum (1, 1): its 3 x 3 neighbourhood
    assert ok[0].sum() == 4                                       # corner: 2 x 2 inside the image


def test_splat_weights_match_the_dense_geometry_bias_inside_the_window():
    """With one grain per stratum at the cell centres and a window covering every grain, the splat weights are the
    softmax of the read's log nu (geom_bias) over the valid grains."""
    torch.manual_seed(0)
    cfg = tiny_config(**{**SPLAT, "stratum": 1, "splat_window": 4})
    sp = GrainSplat(cfg)
    B, G = 2, cfg.n_grains
    u = torch.rand(B, G, 2)
    sigma, depth = 0.3 + torch.rand(B, G), torch.rand(B, G)
    valid = torch.ones(B, G, dtype=torch.bool)
    uq = torch.rand(B, cfg.query_grid ** 2, 2)
    w, idx, ok = sp.weights(uq, u, sigma, depth, valid)          # [B, Q, W] each
    dense = geom_bias(uq, u, sigma, depth, valid, cfg.grid, cfg.tau_d).softmax(-1)
    ref = torch.gather(dense, 2, idx)
    assert torch.allclose(w[ok], ref[ok], atol=1e-5) and (w[~ok] == 0).all()
    assert torch.allclose(w.sum(-1), torch.ones(B, uq.shape[1]), atol=1e-5)


def test_atoms_start_at_zero_and_the_output_has_the_rgb_state_shape():
    m = _model()
    S = m(*_inputs("image", B=2)[:1], "image", return_outputs=True)["outputs"]["S"]
    o = m.decode_anchor(S, m.derived(S))
    assert o["a_rgb"].shape == (2, 16 * 16, m.cfg.dim)
    assert all((p == 0).all() for n, p in m.splat.named_parameters() if n.startswith("amp."))


@pytest.mark.parametrize("modality", ["image", "video"])
def test_gradients_reach_the_grains_and_the_splat_decoder(modality):
    m = _wake(_model())
    x, cam = _inputs(modality, B=2, T=2)
    loss = m(x, modality, cam)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    gs = lambda pre: [p.grad for n, p in m.named_parameters() if n.startswith(pre)]
    assert any(g is not None and g.abs().sum() > 0 for g in gs("anchor.w_f"))
    assert any(g is not None and g.abs().sum() > 0 for g in gs("splat.blocks"))
    assert any(g is not None and g.abs().sum() > 0 for g in gs("splat.amp"))       # atoms learn from step 0


# ---------------------------------------------------------------- 3.7 pixel atoms (splat_pix_feat) and pixel ResBlocks
PIX = dict(SPLAT, splat_pix_feat=4, splat_pix_atoms=2, rgb_pix_blocks=2, rgb_pix_width=8)


def _load_from(src, dst):
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected
    return missing


def _rgb(m, x):
    o = m(x, "image", return_outputs=True)["outputs"]
    return o["rgb"]


def test_pixel_atoms_and_pixel_blocks_are_off_by_default():
    m = _model()
    assert not any(n.startswith(("splat.amp_pix", "splat.freq_pix", "rgb.pw.pix_in", "rgb.pw.pix_blocks"))
                   for n, _ in m.named_parameters())


def test_pixel_atoms_and_pixel_blocks_preserve_the_function_at_init():
    """Zero-init pixel amplitudes (so the pixel map is 0) and zero-init ResBlock outputs: loading a checkpoint
    trained without them reproduces its images exactly."""
    base, m = _model(), _model(**PIX)
    missing = _load_from(base, m)
    assert missing and all(k.startswith(("splat.amp_pix", "splat.freq_pix", "rgb.pw.pix_in", "rgb.pw.pix_blocks"))
                           for k in missing)
    x, _ = _inputs("image", B=2)
    with torch.no_grad():
        assert torch.allclose(_rgb(base, x), _rgb(m, x), atol=1e-6)


def test_pixel_atoms_map_has_the_image_resolution_and_learns():
    m = _wake(_model(**PIX))
    S = m(*_inputs("image", B=2)[:1], "image", return_outputs=True)["outputs"]["S"]
    o = m.decode_anchor(S, m.derived(S))
    assert o["pix"].shape == (2, 4, m.cfg.image_size, m.cfg.image_size)
    x, cam = _inputs("image", B=2)
    m(x, "image", cam)["loss"].backward()
    gs = lambda pre: [p.grad for n, p in m.named_parameters() if n.startswith(pre)]
    assert any(g is not None and g.abs().sum() > 0 for g in gs("splat.amp_pix"))
    assert any(g is not None and g.abs().sum() > 0 for g in gs("rgb.pw.pix_blocks"))


def test_pixel_atoms_need_the_pointwise_head():
    with pytest.raises(ValueError):
        tiny_config(**{**PIX, "rgb_head": "legacy", "rgb_decoder": "read"}).check()
