"""Decoder fine-tuning for FID (plan 2026-10-28 A): PatchGAN hinge loss (w_gan, gan_start), a frozen encoder
(train_scope=decoder) and wide-receptive-field ConvNeXt blocks before the RGB head (rgb_ctx_blocks)."""
import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config

from .test_crsr import LEGACY_CRSR
from .test_grain_prior import _inputs, _script, _wake

QG = dict(LEGACY_CRSR, query_grid=16, rgb_head="pointwise")


def _model(**kw):
    torch.manual_seed(0)
    return MAVTGrainPrior(tiny_config(**kw))


def _grads(m, prefix):
    return [p.grad for n, p in m.named_parameters() if n.startswith(prefix)]


def _nonzero(gs):
    return any(g is not None and g.abs().sum() > 0 for g in gs)


@pytest.mark.parametrize("bad", [dict(w_gan=-1.0), dict(train_scope="encoder"), dict(rgb_ctx_blocks=-1),
                                 dict(gan_start=-5)])
def test_config_checks(bad):
    with pytest.raises(ValueError):
        tiny_config(**bad).check()


def test_defaults_add_nothing():
    keys = set(_model(**QG).state_dict())
    assert not any(k.startswith("disc.") or ".ctx." in k for k in keys)


# ------------------------------------------------------------------------------------------------ GAN
def test_gan_terms_reach_only_their_own_side():
    m = _model(w_gan=0.5, **QG).train()
    fake = (torch.rand(2, 3, 64, 64) * 2 - 1).requires_grad_()
    real = torch.rand(2, 3, 64, 64) * 2 - 1
    g, d = m._gan_terms(fake, real)
    g.backward(retain_graph=True)
    assert fake.grad.abs().sum() > 0 and not _nonzero(_grads(m, "disc."))     # generator term: no D grads
    fake.grad = None
    d.backward()
    assert fake.grad is None or fake.grad.abs().sum() == 0                    # D term: fake detached
    assert _nonzero(_grads(m, "disc."))


def test_gan_weight_starts_at_gan_start_but_the_discriminator_trains_from_step_0():
    m = _wake(_model(w_gan=0.5, gan_start=100, **QG)).train()
    x, _ = _inputs("image")
    m.set_step(0)
    out = m(x, "image")
    assert out["parts"]["gan_w"] == 0 and "gan_d" in out["parts"]
    out["loss"].backward()
    assert _nonzero(_grads(m, "disc."))                                        # DDP sees D grads every step
    m.zero_grad()
    m.set_step(100)
    assert m(x, "image")["parts"]["gan_w"] == pytest.approx(0.5)


def test_gan_only_in_training():
    m = _model(w_gan=0.5, **QG).eval()
    with torch.no_grad():
        out = m(_inputs("image")[0], "image")
    assert "gan_d" not in out["parts"]


def test_gan_on_clips():
    m = _wake(_model(w_gan=0.5, **QG)).train()
    x, cam = _inputs("video", T=3)
    out = m(x, "video", cam)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and _nonzero(_grads(m, "disc."))


# ------------------------------------------------------------------------------------------------ scope
def test_train_scope_decoder_freezes_the_encoder():
    m = _wake(_model(train_scope="decoder", w_gan=0.5, rgb_ctx_blocks=1, **QG)).train()
    train = {n for n, p in m.named_parameters() if p.requires_grad}
    assert train and all(n.startswith(("readout.", "rgb.", "disc.")) for n in train)
    assert any(n.startswith("readout.") for n in train) and any(n.startswith("disc.") for n in train)
    x, _ = _inputs("image")
    out = m(x, "image")
    out["loss"].backward()
    assert all(p.grad is None for n, p in m.named_parameters() if not p.requires_grad)
    assert _nonzero(_grads(m, "rgb.")) and _nonzero(_grads(m, "readout."))


# ------------------------------------------------------------------------------------------------ ctx head
@pytest.mark.parametrize("head", ["pointwise", "legacy"])
def test_ctx_blocks_are_the_identity_at_init(head):
    kw = dict(QG, rgb_head=head)
    on, off = _model(rgb_ctx_blocks=2, **kw).eval(), _model(**kw).eval()
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert not unexpected and missing and all(".ctx." in k for k in missing)
    x, _ = _inputs("image")
    with torch.no_grad():
        assert torch.allclose(on(x, "image", return_outputs=True)["outputs"]["rgb"],
                              off(x, "image", return_outputs=True)["outputs"]["rgb"], atol=1e-6)


def test_ctx_blocks_widen_the_receptive_field():
    m = _model(rgb_ctx_blocks=2, **QG)
    torch.manual_seed(1)
    with torch.no_grad():
        for blk in m.rgb.ctx:
            blk.fc2.weight.normal_(0, 0.1)
            m.rgb.pw.final.weight.normal_(0, 0.1)
    G = m.cfg.query_grid
    H = torch.randn(1, G * G, m.cfg.dim, requires_grad=True)
    y = m.rgb.residual(H)
    y[0, :, 32, 32].sum().backward()
    cells = (H.grad[0].abs().sum(-1) > 0).nonzero().squeeze(-1)
    rows, cols = cells // G, cells % G
    span_px = 4 * (max(rows.max() - rows.min(), cols.max() - cols.min()) + 1)
    assert span_px > 16                                                        # pointwise alone: <= 16 px


def test_optim_groups_and_new_params():
    m = _model(w_gan=0.5, rgb_ctx_blocks=1, **QG)
    kind = {r["name"]: r["kind"] for r in m.optim_groups()}
    assert kind["rgb.ctx.0.fc1.weight"] == "muon" and kind["rgb.ctx.0.fc2.weight"] == "muon"
    assert kind["rgb.ctx.0.dw.weight"] == "io"
    assert all(k == "io" or k == "scalar" for n, k in kind.items() if n.startswith("disc."))
    new = set(m.state_dict()) - set(_model(**QG).state_dict())
    pre = m.new_param_prefixes()
    assert new and all(k.startswith(pre) for k in new)


def test_trainer_decoder_finetune_from_a_checkpoint(tmp_path):
    tr = _script("train_grain_prior")
    base = tmp_path / "base"
    assert tr.main(["--dry-run", "--mix-modalities", "--out", str(base)]) == 0
    ft = tmp_path / "ft"
    assert tr.main(["--dry-run", "--mix-modalities", "--out", str(ft), "--init", str(base / "last.pt"),
                    "--init-new-ok", "--set", "w_gan=0.1", "--set", "rgb_ctx_blocks=1",
                    "--set", "train_scope=decoder"]) == 0
    ck = torch.load(ft / "last.pt", weights_only=False)
    b = torch.load(base / "last.pt", weights_only=False)["model"]
    assert torch.equal(ck["model"]["anchor.w_c.weight"], b["anchor.w_c.weight"])          # encoder frozen
    assert any(k.startswith("disc.") for k in ck["model"])
