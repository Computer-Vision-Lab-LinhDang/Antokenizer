"""MAVT on DINOv3, v1 phase 1 at the model level (tiny config, CPU): batched decodes, forecasts,
warp-only and gate losses, LPIPS / Gram, and that default flags keep v0 (keys and losses)."""
import functools
import math
import os

import pytest
import torch

from mavt.dino_v0 import losses as L
from mavt.dino_v0.cd_state import CDState
from mavt.dino_v0.config import DinoV0Config, tiny_config
from mavt.dino_v0.decoder import image_to_patches
from mavt.dino_v0.model import MAVTDinoV0
from mavt.dino_v0.rope import grid_coords, modality_mask

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V0_PARTS_VIDEO = {"rgb", "feat", "fut", "kl", "var", "sem", "gate"}
V0_PARTS_IMAGE = {"rgb", "feat", "kl", "var", "sem"}


def _v1(**kw):
    """v1_alpha flags on the tiny config (small conv head: 4 x2 stages from the 4 x 4 grid to 64^2)."""
    base = dict(delta_mode="query", rgb_head="conv", conv_channels=[16, 16, 8, 8, 8], conv_res_blocks=1,
                gate_loss="hinge", gate_floor=0.5, gate_init_bias=1.0, train_queries_per_frame=16,
                w_warp=0.5, forecast_horizons=[1, 4], forecast_residual=True, w_fut_rgb=0.5)
    base.update(kw)
    return tiny_config(**base)


def _model(cfg):
    torch.manual_seed(0)
    return MAVTDinoV0(cfg)


def _data(modality, B=2, T=6):
    g = torch.Generator().manual_seed(1)
    if modality == "image":
        return torch.rand(B, 3, 64, 64, generator=g) * 2 - 1, None
    x = torch.rand(B, 3, T, 64, 64, generator=g) * 2 - 1
    return x, (_orbit_cams(B, T) if modality == "threed" else None)


def _orbit_cams(B, T, size=64):
    """Cameras on a circle of radius 2 looking at the origin, OpenCV axes."""
    c2w = torch.zeros(B, T, 4, 4)
    for t in range(T):
        a = 2 * math.pi * t / T
        o = torch.tensor([2 * math.cos(a), 2 * math.sin(a), 0.3])
        z = -o / o.norm()
        x = torch.linalg.cross(z, torch.tensor([0.0, 0.0, -1.0]))
        x = x / x.norm()
        y = torch.linalg.cross(z, x)
        c2w[:, t, :3, 0], c2w[:, t, :3, 1], c2w[:, t, :3, 2], c2w[:, t, :3, 3] = x, y, z, o
        c2w[:, t, 3, 3] = 1
    f = size / 2 / math.tan(0.3456)
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]]).expand(B, T, 3, 3).clone()
    return dict(c2w=c2w, K=K)


# ---------------------------------------------------------------------------- default flags = v0
def test_default_flags_keep_v0_state_dict_keys():
    keys = set(_model(tiny_config()).state_dict())
    assert len(keys) == 220                                         # v0 (stub DINO included)
    d_state = {k.split(".")[2] for k in keys if k.startswith("cd.d_state.")}
    assert d_state == {"d0", "pred", "corr", "up_proj", "pose_proj"}
    assert "decoder.offset.2.weight" in keys and "decoder.rgb_head.weight" in keys
    v1_names = ("h_emb", "fut_out", "delta_", "rgb_conv", "lpips")
    assert not [k for k in keys if any(n in k for n in v1_names)]


def _v0_reference(m, x, modality, cam=None):
    """The v0 per-frame loop of forward() (one decoder call per frame), all N queries."""
    e = m.encode(x, modality, cam)
    seg, T = e["state"]["seg"], e["frames"].shape[1]
    rays = lambda t: None if e["rays"] is None else e["rays"][:, t]
    rgb, feat, gates, fut = [], [], [], []
    for t in range(T):
        o = m.decode_frame(e["zc"][:, seg[t]], e["zd"][:, t], float(t), modality, rays=rays(t))
        rgb.append(L.rgb_l1(o["rgb"], image_to_patches(e["frames"][:, t], m.cfg.patch)))
        feat.append(L.feat_loss(o["feat"], e["feat_target"][:, t]))
        gates.append(o["gate"].flatten())
    for t in range(T - 1):
        o = m.decode_frame(e["zc"][:, seg[t]], e["zd_next"][:, t], float(t + 1), modality, rays=rays(t + 1))
        fut.append(L.feat_loss(o["feat"], e["feat_target"][:, t + 1]))
    return dict(rgb=torch.stack(rgb).mean(), feat=torch.stack(feat).mean(), fut=torch.stack(fut).mean(),
                gate=L.gate_reg(torch.cat(gates)))


@pytest.mark.parametrize("modality", ["video", "threed"])
def test_default_flags_batched_losses_equal_v0_loop(modality):
    m = _model(tiny_config()).eval()                               # eval: all N queries, z = mu
    x, cam = _data(modality, T=5)
    with torch.no_grad():
        out = m(x, modality, cam)
        ref = _v0_reference(m, x, modality, cam)
    assert set(out["parts"]) == V0_PARTS_VIDEO
    for k, v in ref.items():
        assert torch.allclose(out["parts"][k], v, atol=1e-5), k


@pytest.mark.parametrize("modality", ["image", "video", "threed"])
def test_default_flags_train_step_parts(modality):
    m = _model(tiny_config()).train()
    x, cam = _data(modality, T=5)
    out = m(x, modality, cam)
    assert torch.isfinite(out["loss"])
    assert set(out["parts"]) == (V0_PARTS_IMAGE if modality == "image" else V0_PARTS_VIDEO)
    assert "psnr_queries" in out["diag"] and "psnr" not in out["diag"]   # query subset: no full frames
    out["loss"].backward()


def test_in_loop_prediction_is_the_one_step_forecast():
    torch.manual_seed(0)
    cfg = tiny_config()
    st = CDState(cfg)(*_cd_inputs(cfg, 2, 6))
    assert torch.equal(st["D_hat"][:, 1:], st["D_next"][:, :-1])
    assert list(st["D_fut"]) == [1] and torch.equal(st["D_fut"][1], st["D_next"][:, :-1])


# ---------------------------------------------------------------------------- decode_frame
@pytest.mark.parametrize("rgb_head", ["linear", "conv"])
def test_decode_rows_with_own_times_equal_per_frame_decodes(rgb_head):
    cfg = _v1(rgb_head=rgb_head)
    m = _model(cfg).eval()
    x, _ = _data("video", B=2, T=5)
    with torch.no_grad():
        e = m.encode(x, "video")
        seg = e["state"]["seg"]
        B, T = 2, 5
        tt = torch.arange(T)
        o = m.decode_frame(e["zc"][:, seg].flatten(0, 1), e["zd"].flatten(0, 1), tt.float().repeat(B), "video")
        assert o["image"].shape == (B * T, 3, 64, 64)
        img = o["image"].unflatten(0, (B, T))
        for t in range(T):
            ot = m.decode_frame(e["zc"][:, seg[t]], e["zd"][:, t], float(t), "video")
            assert torch.allclose(img[:, t], ot["image"], atol=1e-5), t
        rec = m.reconstruct(x, "video")
    assert torch.allclose(rec, img.clamp(-1, 1).permute(0, 2, 1, 3, 4), atol=1e-5)


def test_conv_head_refuses_query_subsets_and_warp_only_locks_gate():
    m = _model(_v1()).eval()
    x, _ = _data("video", T=3)
    with torch.no_grad():
        e = m.encode(x, "video")
        with pytest.raises(ValueError, match="full grid"):
            m.decode_frame(e["zc"][:, 0], e["zd"][:, 1], 1.0, "video", idx=torch.zeros(2, 4, dtype=torch.long))
        o = m.decode_frame(e["zc"][:, 0], e["zd"][:, 1], 1.0, "video", warp_only=True)
    assert torch.all(o["gate"] == 1)


# ---------------------------------------------------------------------------- v1 flags
@pytest.mark.parametrize("modality", ["image", "video", "threed"])
def test_v1_flags_step(modality):
    m = _model(_v1()).train()
    T = 6 if modality == "video" else 5
    x, cam = _data(modality, T=T)
    out = m(x, modality, cam)
    assert torch.isfinite(out["loss"])
    assert all(torch.isfinite(v) for v in out["parts"].values())
    assert all(torch.isfinite(v) for v in out["diag"].values())
    d = out["diag"]
    assert {"psnr", "psnr_queries", "gate_mean", "delta_abs_cells", "mean_energy"} <= set(d)
    if modality == "image":
        assert set(out["parts"]) == V0_PARTS_IMAGE
    else:
        assert set(out["parts"]) == V0_PARTS_VIDEO | {"fut_rgb", "warp"}
        assert {"psnr_warp", "psnr_fut_h1", "psnr_fut_h4", "delta_abs_cells_key", "delta_abs_cells_nonkey"} <= set(d)
        assert out["parts"]["gate"] == 0                          # hinge: o ~ 0.7 at init, above the floor
        assert d["delta_abs_cells"] == 0                          # Δ head starts at zero
    out["loss"].backward()
    if modality != "image":
        ds = m.cd.d_state
        assert ds.fut_out.weight.grad.abs().sum() > 0              # the zero-init residual branch learns
        # (everything behind fut_out, h_emb included, gets gradient once fut_out moves off zero)
    m.eval()
    assert m.reconstruct(x, modality, cam).shape == x.shape


@pytest.mark.parametrize("modality", ["video", "threed"])
def test_horizon_embedding_and_pose_get_the_forecast_loss(modality):
    m = _model(_v1(forecast_residual=False)).train()               # predictor not behind a zero layer
    x, cam = _data(modality, T=6)
    m(x, modality, cam)["loss"].backward()
    ds = m.cd.d_state
    assert ds.h_emb.grad[4].abs().sum() > 0                        # the h = 4 row: only the h = 4 forecasts
    assert ds.h_emb.grad[2:4].abs().sum() == 0
    if modality == "threed":
        assert ds.pose_proj[-1].weight.grad.abs().sum() > 0


def test_v1_linear_head_keeps_query_subsets():
    m = _model(_v1(rgb_head="linear", train_queries_per_frame=8)).train()
    x, _ = _data("video", T=5)
    out = m(x, "video")
    assert torch.isfinite(out["loss"]) and {"warp", "fut_rgb", "fut"} <= set(out["parts"])
    assert "psnr" not in out["diag"]
    out["loss"].backward()


def test_gate_loss_mean_vs_hinge():
    g = torch.full((2, 5, 1), 0.7)
    assert L.gate_hinge(g, 0.5) == 0
    assert torch.allclose(L.gate_hinge(torch.full((2, 5, 1), 0.2), 0.5), torch.tensor(0.3))
    assert torch.allclose(L.gate_reg(g), torch.tensor(0.3))
    x, _ = _data("video", T=5)
    mean = _model(_v1(gate_loss="mean"))(x, "video")["parts"]["gate"]
    assert mean > 0                                               # v0 loss pushes o -> 1 even at o ~ 0.7


# ---------------------------------------------------------------------------- forecasts
def _cd_inputs(cfg, B, T, requires_grad=False):
    N = cfg.n_patches
    X = torch.randn(B, T, N, cfg.dim, requires_grad=requires_grad)
    coords = torch.stack([grid_coords(cfg.grid, t=float(t)) for t in range(T)])
    return X, coords, modality_mask("video", N).expand(T, N, 4)


def test_forecast_residual_starts_as_copy_last():
    torch.manual_seed(0)
    cfg = _v1(forecast_horizons=(1, 2, 4))
    B, T = 2, 6
    st = CDState(cfg)(*_cd_inputs(cfg, B, T))
    D = st["D"]
    assert torch.equal(st["D_next"], D)                           # D_hat(t + 1) = D_t
    assert torch.equal(st["D_hat"][:, 1:], D[:, :-1])             # in-loop prediction: copy D_{t-1}
    assert list(st["D_fut"]) == [1, 2, 4]
    for h, D_h in st["D_fut"].items():
        assert D_h.shape == (B, T - h, cfg.d_tokens, cfg.d_dim)
        assert torch.equal(D_h, D[:, :T - h]), h


def test_zd_fut_shapes_and_copy_last_latents():
    cfg = _v1()
    m = _model(cfg).eval()                                        # z = mu
    B, T = 2, 6
    x, _ = _data("video", B=B, T=T)
    with torch.no_grad():
        e = m.encode(x, "video")
    assert e["zd_next"].shape == (B, T, cfg.d_tokens, cfg.zr_d)
    assert list(e["zd_fut"]) == [1, 4]
    for h, z in e["zd_fut"].items():
        assert z.shape == (B, T - h, cfg.d_tokens, cfg.zr_d)
        assert torch.allclose(z, e["zd"][:, :T - h], atol=1e-6), h
    assert torch.equal(e["zd_fut"][1], e["zd_next"][:, :T - 1])
    # horizons that do not fit the clip are left out
    with torch.no_grad():
        e3 = m.encode(x[:, :, :3], "video")
    assert list(e3["zd_fut"]) == [1]


def test_forecasts_are_causal():
    """D_hat(t + h) made at t reads X up to t only (no residual: the predictor itself reads C)."""
    torch.manual_seed(0)
    cfg = tiny_config(forecast_horizons=(1, 3))
    m = CDState(cfg)
    X, coords, mask = _cd_inputs(cfg, 1, 6, requires_grad=True)
    st = m(X, coords, mask)
    for h, D_h in st["D_fut"].items():
        for t in range(D_h.shape[1]):
            w = torch.randn(D_h[:, t].shape, generator=torch.Generator().manual_seed(t))
            (g,) = torch.autograd.grad((D_h[:, t] * w).sum(), X, retain_graph=True)
            g = g.abs().sum(dim=(0, 2, 3))
            assert g[t] > 0 and torch.all(g[t + 1:] == 0), (h, t)


def test_h_emb_only_when_needed():
    def names(**kw):
        return {k.split(".")[0] for k in CDState(tiny_config(**kw)).d_state.state_dict()}
    assert "h_emb" not in names() and "fut_out" not in names()
    assert "h_emb" in names(forecast_horizons=[1, 4]) and "fut_out" not in names(forecast_horizons=[1, 4])
    assert {"h_emb", "fut_out"} <= names(forecast_residual=True)


# ---------------------------------------------------------------------------- config
def test_v1_alpha_yaml():
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(ROOT, "configs", "dino_v0", "v1_alpha.yaml")) as f:
        y = yaml.safe_load(f)
    train = y.pop("train")
    cfg = DinoV0Config(**y)
    assert cfg.budget("video") == 12288 and cfg.budget("image") == 12288
    assert (cfg.zr_c_video, cfg.zr_d, cfg.zr_c_image) == (32, 64, 48)
    assert (cfg.delta_mode, cfg.rgb_head, cfg.gate_loss) == ("query", "conv", "hinge")
    assert list(cfg.forecast_horizons) == [1, 4] and cfg.forecast_residual
    assert len(cfg.conv_channels) == int(math.log2(cfg.patch)) + 1
    assert cfg.w_lpips > 0 and cfg.w_gram > 0 and cfg.w_warp > 0 and cfg.w_fut_rgb > 0
    assert train["frame_stride"] == "2,4,8" and train["mix_modalities"] is True


# ---------------------------------------------------------------------------- LPIPS / Gram
def test_gram():
    f = torch.ones(1, 2, 3, 4)
    assert torch.allclose(L.gram(f), torch.full((1, 2, 2), 12 / 24))
    x = torch.randn(2, 8, 5, 5)
    G = L.gram(x)
    assert G.shape == (2, 8, 8) and torch.allclose(G, G.transpose(1, 2))


def _lpips_random_vgg(monkeypatch):
    """LPIPS with a random VGG trunk: no weight download in tests."""
    pytest.importorskip("lpips")
    monkeypatch.setattr(L, "LPIPSLoss", functools.partial(L.LPIPSLoss, pretrained_trunk=False))


def test_lpips_gram_shares_one_vgg_forward(monkeypatch):
    _lpips_random_vgg(monkeypatch)
    lp = L.LPIPSLoss()
    a, b = torch.rand(3, 3, 64, 64) * 2 - 1, torch.rand(3, 3, 64, 64) * 2 - 1
    v, g = lp.lpips_gram(a, b, with_gram=True)
    assert torch.allclose(v, lp.model(a, b).mean(), atol=1e-6)    # same value as the lpips package
    assert g > 0 and torch.allclose(lp(a, b), v)
    v0, g0 = lp.lpips_gram(a, a, with_gram=True)
    assert v0.abs() < 1e-6 and g0 == 0


@pytest.mark.parametrize("rgb_head", ["conv", "linear"])
def test_lpips_and_gram_in_the_model(monkeypatch, rgb_head):
    _lpips_random_vgg(monkeypatch)
    cfg = _v1(rgb_head=rgb_head, w_lpips=0.5, w_gram=50.0, lpips_max_frames=3)
    m = _model(cfg).train()
    x, _ = _data("video", T=5)
    out = m(x, "video")
    assert {"lpips", "gram"} <= set(out["parts"]) and torch.isfinite(out["loss"])
    out["loss"].backward()
    # the frozen VGG is never saved; a model with LPIPS loads its own and a v0 state_dict strictly
    sd = m.state_dict()
    assert not [k for k in sd if k.startswith("lpips.")]
    m2 = MAVTDinoV0(cfg)
    m2.load_state_dict(sd, strict=True)
    m3 = MAVTDinoV0(tiny_config(w_lpips=0.5))
    m3.load_state_dict(_model(tiny_config()).state_dict(), strict=True)
