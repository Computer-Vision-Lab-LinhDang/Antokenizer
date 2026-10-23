"""v1 phase-2 speed options against the phase-1 math (CPU, fp32 / fp64): forecast and warp-only decodes on
frame subsets, the conv rgb head in channels_last, the gathered window read. Defaults keep phase 1."""
import math
import os

import pytest
import torch

from mavt.dino_v0.config import DinoV0Config, tiny_config
from mavt.dino_v0.decoder import QueryDecoder, query_coords
from mavt.dino_v0.model import MAVTDinoV0, _subset
from mavt.dino_v0.rope import modality_mask

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEAT = 16
CONV = dict(rgb_head="conv", conv_channels=(32, 16, 16, 8, 8), conv_res_blocks=1)   # 4 x2 stages: 4 -> 64


def _v1(**kw):
    """v1_alpha flags on the tiny config (as tests/dino_v0/test_model_v1.py)."""
    base = dict(delta_mode="query", rgb_head="conv", conv_channels=[16, 16, 8, 8, 8], conv_res_blocks=1,
                gate_loss="hinge", gate_floor=0.5, gate_init_bias=1.0, train_queries_per_frame=16,
                w_warp=0.5, forecast_horizons=[1, 4], forecast_residual=True, w_fut_rgb=0.5)
    base.update(kw)
    return tiny_config(**base)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return MAVTDinoV0(cfg)


def _video(B=2, T=6):
    g = torch.Generator().manual_seed(1)
    return torch.rand(B, 3, T, 64, 64, generator=g) * 2 - 1


def _rows(m):
    """Row count (frames) of every decoder call of m, in call order."""
    rows, fwd = [], m.decoder.forward

    def wrapped(zc, *a, **kw):
        rows.append(int(zc.shape[0]))
        return fwd(zc, *a, **kw)
    m.decoder.forward = wrapped
    return rows


def _flat_inputs(cfg, B, T, seed=0, dtype=torch.float32):
    """B * T flattened video frames: row b * T + t carries frame t's coords."""
    g = torch.Generator().manual_seed(seed)
    N = cfg.n_patches
    zc = torch.randn(B * T, N, cfg.zr_c_video, generator=g).to(dtype)
    zd = torch.randn(B * T, cfg.d_tokens, cfg.zr_d, generator=g).to(dtype)
    qc = torch.stack([query_coords(cfg.grid, float(t), B) for t in range(T)], 1).flatten(0, 1).to(dtype)
    qm = modality_mask("video", N).unsqueeze(0).expand(B * T, -1, -1).to(dtype)
    return zc, zd, qc, qm


def _nonzero_delta(dec, std=0.3):
    last = dec.offset[-1] if dec.cfg.delta_mode == "mean" else dec.delta_mlp[-1]
    with torch.no_grad():
        last.weight.normal_(0, std)


# ---------------------------------------------------------------------------- defaults
def test_phase2_defaults_are_off():
    cfg = DinoV0Config()
    assert (cfg.fut_frames_per_horizon, cfg.warp_frames, cfg.conv_channels_last, cfg.window_read) == (0, 0, False, "dense")
    # the options add no parameters: a phase-1 checkpoint loads strictly either way
    a = set(QueryDecoder(tiny_config(delta_mode="query", **CONV), FEAT).state_dict())
    b = set(QueryDecoder(tiny_config(delta_mode="query", conv_channels_last=True, window_read="gather", **CONV), FEAT).state_dict())
    assert a == b
    with pytest.raises(AssertionError):
        QueryDecoder(tiny_config(window_read="nope"), FEAT)


def test_subset_helper():
    torch.manual_seed(0)
    assert torch.equal(_subset(5, 0, "cpu"), torch.arange(5))
    assert torch.equal(_subset(5, 5, "cpu"), torch.arange(5))
    assert torch.equal(_subset(5, 9, "cpu"), torch.arange(5))
    s = _subset(6, 2, "cpu")
    assert s.shape == (2,) and s[0] < s[1] and 0 <= s[0] and s[1] < 6
    assert torch.equal(_subset(0, 2, "cpu"), torch.arange(0))


# ---------------------------------------------------------------------------- frame subsets
def test_subsets_decode_fewer_rows_with_the_same_loss_parts():
    """T = 6, K = 4: non-key frames 1, 2, 3, 5; horizons 1 and 4 give 5 and 2 forecast pairs. The subset
    options cut the forecast and warp-only decodes to 2 source frames per horizon and 2 non-key frames."""
    B, T = 2, 6
    x = _video(B, T)
    full = _model(_v1()).train()
    r_full = _rows(full)
    out_full = full(x, "video")
    assert r_full == [B * T, B * (5 + 2), B * 4]                 # recon, forecasts (h1 + h4), warp-only

    sub = _model(_v1(fut_frames_per_horizon=2, warp_frames=2)).train()
    r_sub = _rows(sub)
    out_sub = sub(x, "video")
    assert r_sub == [B * T, B * (2 + 2), B * 2]
    assert set(out_sub["parts"]) == set(out_full["parts"]) and set(out_sub["diag"]) == set(out_full["diag"])
    assert all(torch.isfinite(v) for v in out_sub["parts"].values())
    out_sub["loss"].backward()
    assert sub.cd.d_state.fut_out.weight.grad.abs().sum() > 0     # forecasts still train the residual branch
    assert sub.decoder.blocks[0].read_c.q.weight.grad.abs().sum() > 0


def test_subset_larger_than_the_clip_is_bit_identical_to_full():
    x = _video(2, 6)
    torch.manual_seed(3)
    a = _model(_v1()).train()(x, "video")
    torch.manual_seed(3)
    b = _model(_v1(fut_frames_per_horizon=99, warp_frames=99)).train()(x, "video")
    assert torch.equal(a["loss"], b["loss"])
    for k in a["parts"]:
        assert torch.equal(a["parts"][k], b["parts"][k]), k


def test_subset_draw_is_seeded_and_shared_by_the_batch():
    """Same seed -> same frames (same loss). The draw is one per step: a clip repeated twice in the batch gets
    the same frames as the clip alone, so the forecast / warp losses agree (eval mode: z = mu, no sampling
    noise between the two runs; the draw is the only random op)."""
    cfg = _v1(fut_frames_per_horizon=1, warp_frames=1)
    x = _video(2, 6)
    outs = []
    for _ in range(2):
        torch.manual_seed(5)
        outs.append(_model(cfg).train()(x, "video"))
    assert torch.equal(outs[0]["loss"], outs[1]["loss"])
    m = _model(cfg).eval()
    torch.manual_seed(5)
    one = m(x[:1].repeat(2, 1, 1, 1, 1), "video")
    torch.manual_seed(5)
    two = m(x[:1], "video")
    for k in ("fut", "fut_rgb", "warp"):
        torch.testing.assert_close(one["parts"][k], two["parts"][k], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------- channels_last conv head
def test_conv_head_channels_last_matches_nchw():
    cfg = tiny_config(delta_mode="query", **CONV)
    torch.manual_seed(0)
    a = QueryDecoder(cfg, FEAT)
    torch.manual_seed(0)
    b = QueryDecoder(tiny_config(delta_mode="query", conv_channels_last=True, **CONV), FEAT)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)   # same init stream
    conv4d = [p for p in b.rgb_conv.parameters() if p.dim() == 4]
    assert conv4d and all(p.is_contiguous(memory_format=torch.channels_last) for p in conv4d)
    _nonzero_delta(a)
    b.load_state_dict(a.state_dict(), strict=True)                # a phase-1 checkpoint loads; layout kept
    assert all(p.is_contiguous(memory_format=torch.channels_last) for p in conv4d)
    B, T = 2, 3
    zc_a, zd_a, qc, qm = _flat_inputs(cfg, B, T)
    zc_b, zd_b = zc_a.clone(), zd_a.clone()
    zc_a.requires_grad_(True), zc_b.requires_grad_(True)
    oa, ob = a(zc_a, zd_a, qc, qm, "video"), b(zc_b, zd_b, qc, qm, "video")
    assert ob["image"].shape == (B * T, 3, 64, 64) and ob["image"].is_contiguous()
    for k in ("image", "rgb", "feat", "delta", "gate"):
        torch.testing.assert_close(oa[k], ob[k], atol=1e-5, rtol=1e-5)
    w = torch.randn_like(oa["image"])
    (oa["image"] * w).sum().backward()
    (ob["image"] * w).sum().backward()
    torch.testing.assert_close(zc_a.grad, zc_b.grad, atol=1e-5, rtol=1e-4)
    for na, pa in a.rgb_conv.named_parameters():
        pb = dict(b.rgb_conv.named_parameters())[na]
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------- gathered window read
@pytest.mark.parametrize("delta_mode", ["mean", "query"])
def test_window_read_gather_matches_dense(delta_mode):
    torch.manual_seed(0)
    dense = QueryDecoder(tiny_config(delta_mode=delta_mode), FEAT).double()
    gather = QueryDecoder(tiny_config(delta_mode=delta_mode, window_read="gather"), FEAT).double()
    _nonzero_delta(dense, std=0.5)
    gather.load_state_dict(dense.state_dict(), strict=True)
    cfg = dense.cfg
    B, T = 2, 3
    zc, zd, qc, qm = _flat_inputs(cfg, B, T, dtype=torch.float64)
    zc2, zd2 = zc.clone(), zd.clone()
    for z in (zc, zd, zc2, zd2):
        z.requires_grad_(True)
    rays = torch.randn(B * T, cfg.n_patches, 6, dtype=torch.float64)
    for r in (None, rays):
        od, og = dense(zc, zd, qc, qm, "video", q_rays=r), gather(zc2, zd2, qc, qm, "video", q_rays=r)
        assert od["delta"].abs().max() > 0.05
        for k in ("rgb", "feat", "delta", "gate"):
            torch.testing.assert_close(od[k], og[k], atol=1e-6, rtol=1e-5)
    w = torch.randn_like(od["rgb"])
    gd = torch.autograd.grad((od["rgb"] * w).sum(), [zc, zd, dense.blocks[0].read_c.kv.weight])
    gg = torch.autograd.grad((og["rgb"] * w).sum(), [zc2, zd2, gather.blocks[0].read_c.kv.weight])
    for x, y in zip(gd, gg):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)


def test_window_read_gather_in_the_model_step():
    m = _model(_v1(window_read="gather", conv_channels_last=True, fut_frames_per_horizon=2, warp_frames=2)).train()
    out = m(_video(2, 6), "video")
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    m.eval()
    assert m.reconstruct(_video(1, 5), "video").shape == (1, 3, 5, 64, 64)


# ---------------------------------------------------------------------------- config
def test_v1b_alpha_yaml():
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(ROOT, "configs", "dino_v0", "v1b_alpha.yaml")) as f:
        y = yaml.safe_load(f)
    train = y.pop("train")
    cfg = DinoV0Config(**y)
    assert cfg.budget("video") == 12288 and cfg.budget("image") == 12288
    assert (cfg.delta_mode, cfg.rgb_head, cfg.gate_loss) == ("query", "conv", "hinge")   # phase-1 model unchanged
    assert cfg.fut_frames_per_horizon == 2 and cfg.warp_frames == 2 and cfg.conv_channels_last
    assert cfg.window_read in ("dense", "gather")
    assert len(cfg.conv_channels) == int(math.log2(cfg.patch)) + 1
    assert train["mix_modalities"] is True and train["compile"] is True
