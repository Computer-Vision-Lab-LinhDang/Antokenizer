"""Spike: D tokens anchored to nodes of the C grid (d_anchor, CPU). Token k sits on node k of an a x a grid over
the cells; a query reads only the tokens of its node's neighbourhood, D-D attention runs along node edges and a
token reads X_t / C_k only near its node. d_drop "bernoulli" keeps each token with a per-clip rate. d_anchor 0
(default) keeps the phase-3 model (the rest of tests/dino_v0 covers that)."""
import torch

from mavt.dino_v0.cd_state import DState, anchor_cells, anchor_neighbours
from mavt.dino_v0.config import DinoV0Config
from mavt.dino_v0.model import MAVTDinoV0
from .test_nested_d import _calls, _flat_inputs, _model, _nonzero_delta, _v1, _video

GRAPH = dict(d_nested=True, d_drop="bernoulli", d_anchor=2, d_anchor_radius=0)   # tiny: 4 x 4 cells, 2 x 2 nodes


def _node_of_cell(cfg):
    a, g = cfg.d_anchor, cfg.grid
    r = torch.arange(g) // (g // a)
    return r.repeat_interleave(g) * a + r.repeat(g)


def test_masks_of_the_alpha_graph():
    cfg = DinoV0Config(dino_repo=None, d_tokens=64, d_anchor=8, d_anchor_radius=1)
    nb, cells = anchor_neighbours(cfg), anchor_cells(cfg)
    assert nb.shape == (64, 64) and cells.shape == (64, 256)
    assert nb.diagonal().all() and torch.equal(nb, nb.T)
    assert nb.sum(1).tolist()[:2] == [4, 6] and int(nb[9].sum()) == 9      # corner, edge, interior node
    assert int(cells[9].sum()) == 36 and int(cells[0].sum()) == 16         # 3 x 3 nodes of 2 x 2 cells


def test_bad_anchor_is_refused():
    try:
        MAVTDinoV0(_v1(d_nested=True, d_anchor=3))
    except ValueError as e:
        assert "d_anchor" in str(e)
    else:
        raise AssertionError("d_anchor 3 with 4 tokens must be refused")


def test_graph_model_has_the_same_weights_as_the_plain_one():
    plain = _model(_v1(d_nested=True)).state_dict()
    graph = _model(_v1(**GRAPH)).state_dict()
    assert plain.keys() == graph.keys()                     # masks are not saved: v1c checkpoints load as-is


def test_keep_mask_equal_to_a_prefix_decodes_like_the_prefix():
    cfg = _v1(d_nested=True)
    m = _model(cfg).double().eval()
    _nonzero_delta(m.decoder)
    zc, zd, qc, qm = _flat_inputs(cfg, 2, 2, dtype=torch.float64)
    keep = torch.tensor([True, True, False, False]).expand(4, -1)
    a = m.decoder(zc, zd, qc, qm, "video", zd_len=torch.full((4,), 2))
    b = m.decoder(zc, zd, qc, qm, "video", zd_len=keep)
    torch.testing.assert_close(a["image"], b["image"])


def test_delta_of_a_query_depends_only_on_its_node_token():
    cfg = _v1(**GRAPH)
    m = _model(cfg).double().eval()
    _nonzero_delta(m.decoder)
    zc, zd, qc, qm = _flat_inputs(cfg, 1, 2, dtype=torch.float64)
    node = _node_of_cell(cfg)
    base = m.decoder(zc, zd, qc, qm, "video")["delta"]
    for k in range(cfg.d_tokens):
        zd2 = zd.clone()
        zd2[:, k] += 1.0
        diff = (m.decoder(zc, zd2, qc, qm, "video")["delta"] - base).abs().amax((0, 2))   # [N]
        assert (diff[node == k] > 1e-6).all() and (diff[node != k] == 0).all()


def test_token_reaches_only_its_node_cells_through_every_decoder_block():
    cfg = _v1(**GRAPH)
    m = _model(cfg).double().eval()
    _nonzero_delta(m.decoder)
    zc, zd, qc, qm = _flat_inputs(cfg, 1, 2, dtype=torch.float64)
    node = _node_of_cell(cfg)
    base = m.decoder(zc, zd, qc, qm, "video")["feat"]                        # per query: no conv-head mixing
    for k in range(cfg.d_tokens):
        zd2 = zd.clone()
        zd2[:, k] += torch.randn_like(zd2[:, k])
        diff = (m.decoder(zc, zd2, qc, qm, "video")["feat"] - base).abs().amax((0, 2))   # [N]
        assert (diff[node == k] > 1e-6).all() and (diff[node != k] == 0).all()


def test_images_decode_as_in_the_plain_model():
    plain, graph = _model(_v1(d_nested=True)).double().eval(), _model(_v1(**GRAPH)).double().eval()
    _, _, qc, qm = _flat_inputs(plain.cfg, 1, 2, dtype=torch.float64)
    zc = torch.randn(2, plain.cfg.n_patches, plain.cfg.zr_c_image, dtype=torch.float64)
    torch.testing.assert_close(plain.decoder(zc, None, qc, qm, "image")["image"],
                               graph.decoder(zc, None, qc, qm, "image")["image"], atol=0, rtol=0)


def test_dropping_every_token_reads_the_null_token_only():
    cfg = _v1(**GRAPH)
    m = _model(cfg).double().eval()
    _nonzero_delta(m.decoder)
    zc, zd, qc, qm = _flat_inputs(cfg, 1, 2, dtype=torch.float64)
    none = torch.zeros(2, cfg.d_tokens, dtype=torch.bool)
    a = m.decoder(zc, zd, qc, qm, "video", zd_len=none)
    b = m.decoder(zc, zd + 1.0, qc, qm, "video", zd_len=none)
    assert a["image"].isfinite().all()
    torch.testing.assert_close(a["image"], b["image"])


def test_d_state_token_reads_only_its_node():
    cfg = _v1(**GRAPH)
    torch.manual_seed(0)
    ds = DState(cfg).double().eval()
    with torch.no_grad():
        ds.fut_out.weight.normal_(std=0.3)                                    # zero-init residual: copy-last at init
    B, N = 2, cfg.n_patches
    x, c = torch.randn(B, N, cfg.dim, dtype=torch.float64), torch.randn(B, N, cfg.dim, dtype=torch.float64)
    D_hat = ds.predict(None, c)
    base = ds.correct(D_hat, x, c)
    node = _node_of_cell(cfg)
    for k in range(cfg.d_tokens):
        x2 = x.clone()
        x2[:, node == k] += torch.randn_like(x2[:, node == k])   # random: LayerNorm erases a constant shift
        diff = (ds.correct(D_hat, x2, c) - base).abs().amax((0, 2))           # [M]
        assert diff[k] > 1e-6 and (diff[torch.arange(cfg.d_tokens) != k] == 0).all()
        c2 = c.clone()
        c2[:, node == k] += torch.randn_like(c2[:, node == k])
        diff = (ds.predict(None, c2) - D_hat).abs().amax((0, 2))
        assert diff[k] > 1e-6 and (diff[torch.arange(cfg.d_tokens) != k] == 0).all()


def test_training_draws_one_keep_mask_per_clip():
    cfg = _v1(**GRAPH)
    m = _model(cfg).train()
    calls = _calls(m)
    torch.manual_seed(3)
    out = m(_video(B=2, T=6), "video")
    assert out["loss"].isfinite()
    out["loss"].backward()
    masks = []
    for rows, keep in calls:
        assert keep.dtype == torch.bool and tuple(keep.shape) == (rows, cfg.d_tokens)
        keep = keep.view(2, -1, cfg.d_tokens)                  # rows are clip-major
        assert (keep == keep[:, :1]).all()                      # one mask per clip, every frame
        masks.append(keep[:, 0])
    assert all(torch.equal(masks[0], k) for k in masks[1:])    # the same draw for the whole step
