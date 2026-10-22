"""R2 learned relations in the trunk (cfg.rgat_relations > 0); R0 must stay unchanged when off."""
import torch
import torch.nn.functional as F

from mavt.dino_v0.config import tiny_config
from mavt.dino_v0.model import MAVTDinoV0
from mavt.dino_v0.rope import grid_coords, modality_mask
from mavt.dino_v0.trunk import Trunk, _shift


def _coords(cfg, T, modality="video"):
    c = torch.stack([grid_coords(cfg.grid, t=float(t)) for t in range(T)])
    m = torch.stack([modality_mask(modality, cfg.n_patches) for _ in range(T)])
    return c, m


def _ref_r0(trunk, x, c, m):
    """The R0 trunk as written before R2: SDPA over [x_t ; x_{t-1}] with the grid mask."""
    B, T, N, D = x.shape
    ang = trunk.layers[0].rope.angles(c, m)
    ang_prev = _shift(ang, 0)
    am = trunk.attn_mask(T, N)
    for L in trunk.layers:
        h = L.norm1(x)
        q, k, v = L.qkv(h).view(B, T, N, 3, L.h, L.hd).permute(3, 0, 1, 4, 2, 5)
        rot = lambda z, a: L.rope.rotate(z.reshape(B * T, L.h, N, L.hd), a.repeat(B, 1, 1)).view(B, T, L.h, N, L.hd)
        k = torch.cat([rot(k, ang), rot(_shift(k, 1), ang_prev)], 3)
        v = torch.cat([v, _shift(v, 1)], 3)
        o = F.scaled_dot_product_attention(rot(q, ang), k, v, attn_mask=am)
        x = x + L.out(o.transpose(2, 3).reshape(B, T, N, D))
        x = x + L.mlp(L.norm2(x))
    return trunk.norm(x)


def _pair(T=3, **kw):
    torch.manual_seed(0)
    off = Trunk(tiny_config(**kw)).eval()
    on = Trunk(tiny_config(rgat_relations=4, rgat_rank=8, **kw)).eval()
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not unexpected and all(".rel" in k or "_bias" in k for k in missing)
    torch.manual_seed(1)
    cfg = tiny_config()
    x = torch.randn(2, T, cfg.n_patches, cfg.dim)
    return off, on, x


def test_off_mode_unchanged():
    torch.manual_seed(0)
    cfg = tiny_config()
    a = Trunk(cfg).eval()
    torch.manual_seed(0)
    b = Trunk(tiny_config(rgat_relations=0)).eval()
    assert not any("rel" in n or "type_bias" in n for n, _ in b.named_parameters())
    c, m = _coords(cfg, 3)
    x = torch.randn(2, 3, cfg.n_patches, cfg.dim)
    ya, yb = a(x, c, m), b(x, c, m)
    assert torch.equal(ya, yb)
    assert torch.allclose(ya, _ref_r0(a, x, c, m), atol=1e-6)
    assert a.relation_stats() is None


def test_on_mode_init_equals_off():
    off, on, x = _pair()
    c, m = _coords(tiny_config(), 3)
    assert torch.allclose(off(x, c, m), on(x, c, m), atol=1e-5)
    on.train()                                   # Gumbel noise changes r, but U = 0 and biases = 0
    assert torch.allclose(off(x, c, m), on(x, c, m), atol=1e-5)
    st = on.relation_stats()
    assert st["usage"].shape == (4,) and abs(st["usage"].sum().item() - 1) < 1e-5
    assert 0 <= st["marginal_entropy"].item() <= torch.log(torch.tensor(4.0)).item() + 1e-5
    assert 0 <= st["edge_entropy"].item() <= st["marginal_entropy"].item() + 1e-5


def test_grad_reaches_relation_mlp():
    torch.manual_seed(0)
    cfg = tiny_config(rgat_relations=4, rgat_rank=8)
    trunk = Trunk(cfg).train()
    with torch.no_grad():
        for L in trunk.layers:
            L.rel_U.normal_(0, 0.1)
    c, m = _coords(cfg, 2)
    x = torch.randn(2, 2, cfg.n_patches, cfg.dim)
    trunk(x, c, m).pow(2).mean().backward()
    for L in trunk.layers:
        g = L.rel.mlp[0].weight.grad
        assert g is not None and g.abs().sum() > 0
        assert L.rel_V.grad.abs().sum() > 0


def test_entropy_grad_alone_reaches_mlp():
    torch.manual_seed(0)
    cfg = tiny_config(rgat_relations=4, rgat_rank=8)
    trunk = Trunk(cfg).train()
    c, m = _coords(cfg, 2)
    trunk(torch.randn(1, 2, cfg.n_patches, cfg.dim), c, m)
    (-trunk.relation_stats()["marginal_entropy"]).backward()
    assert trunk.layers[0].rel.mlp[2].weight.grad.abs().sum() > 0


def _perturbed(cfg):
    torch.manual_seed(0)
    trunk = Trunk(cfg).eval()
    with torch.no_grad():
        for L in trunk.layers:
            L.rel_U.normal_(0, 0.3)
            L.rel_bias.normal_(0, 0.5)
            L.type_bias.normal_(0, 0.5)
    return trunk


def test_on_mode_causal():
    cfg = tiny_config(rgat_relations=4, rgat_rank=8)
    trunk = _perturbed(cfg)
    c, m = _coords(cfg, 4)
    x = torch.randn(1, 4, cfg.n_patches, cfg.dim)
    y = trunk(x, c, m)
    x2 = x.clone()
    x2[:, 2:] += torch.randn_like(x2[:, 2:])
    y2 = trunk(x2, c, m)
    assert torch.allclose(y[:, :2], y2[:, :2], atol=1e-5)
    assert not torch.allclose(y[:, 2:], y2[:, 2:], atol=1e-3)


def test_on_mode_window_locality():
    cfg = tiny_config(trunk_layers=1, temporal_window=1, rgat_relations=4, rgat_rank=8)
    trunk = _perturbed(cfg)
    c, m = _coords(cfg, 2)
    x = torch.randn(1, 2, cfg.n_patches, cfg.dim)
    y = trunk(x, c, m)
    q = 0
    for j, inside in ((5, True), (1, True), (2, False), (15, False)):
        x2 = x.clone()
        x2[0, 0, j] += torch.randn(cfg.dim)
        y2 = trunk(x2, c, m)
        changed = not torch.allclose(y[0, 1, q], y2[0, 1, q], atol=1e-5)
        assert changed == inside, (j, inside)


def test_on_mode_eval_deterministic():
    cfg = tiny_config(rgat_relations=4, rgat_rank=8)
    trunk = _perturbed(cfg)
    c, m = _coords(cfg, 3)
    x = torch.randn(2, 3, cfg.n_patches, cfg.dim)
    y1 = trunk(x, c, m)
    torch.manual_seed(123)
    y2 = trunk(x, c, m)
    assert torch.equal(y1, y2)


def test_on_mode_checkpoint():
    cfg = tiny_config(rgat_relations=4, rgat_rank=8)
    trunk = Trunk(cfg, use_checkpoint=True).train()
    x = torch.randn(1, 3, cfg.n_patches, cfg.dim, requires_grad=True)
    y = trunk(x, *_coords(cfg, 3))
    (y.sum() - trunk.relation_stats()["marginal_entropy"]).backward()
    assert x.grad is not None and trunk.layers[0].rel.mlp[0].weight.grad is not None


def test_full_model_rgat():
    torch.manual_seed(0)
    cfg = tiny_config(rgat_relations=4)
    model = MAVTDinoV0(cfg).train()
    for modality, data in (("image", torch.rand(2, 3, 64, 64) * 2 - 1),
                           ("video", torch.rand(2, 3, 3, 64, 64) * 2 - 1)):
        model.zero_grad()
        out = model(data, modality)
        assert torch.isfinite(out["loss"])
        assert "rel" in out["parts"] and "rel_marginal_entropy" in out["diag"]
        expect = -cfg.rgat_entropy_w * out["diag"]["rel_marginal_entropy"]
        assert torch.allclose(out["parts"]["rel"], expect)
        out["loss"].backward()
        assert model.trunk.layers[0].rel.mlp[0].weight.grad is not None
    off = MAVTDinoV0(tiny_config()).train()(torch.rand(1, 3, 64, 64) * 2 - 1, "image")
    assert "rel" not in off["parts"] and "rel_marginal_entropy" not in off["diag"]
