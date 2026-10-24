"""Variable-length D by nested dropout (v1 phase 3, CPU). Training keeps a random prefix of the D tokens
per clip and masks the rest in every read of D (content read and Δ read); at inference any prefix length
goes through decode_frame(zd_len=...). d_nested False (default) keeps phase 2 bit-identical."""
import math
import os

import pytest
import torch

from mavt.dino_v0 import evaluate as E
from mavt.dino_v0.config import DinoV0Config, tiny_config
from mavt.dino_v0.decoder import query_coords
from mavt.dino_v0.model import MAVTDinoV0
from mavt.dino_v0.rope import modality_mask

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _v1(**kw):
    """v1_alpha flags on the tiny config (as tests/dino_v0/test_speed_v1b.py)."""
    base = dict(delta_mode="query", rgb_head="conv", conv_channels=[16, 16, 8, 8, 8], conv_res_blocks=1,
                gate_loss="hinge", gate_floor=0.5, gate_init_bias=1.0, train_queries_per_frame=16,
                w_warp=0.5, forecast_horizons=[1, 4], forecast_residual=True, w_fut_rgb=0.5,
                fut_frames_per_horizon=2, warp_frames=2)
    base.update(kw)
    return tiny_config(**base)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return MAVTDinoV0(cfg)


def _video(B=2, T=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, 3, T, 64, 64, generator=g) * 2 - 1


def _flat_inputs(cfg, B, T, seed=0, dtype=torch.float32):
    """B * T flattened video frames: row b * T + t carries frame t's coords."""
    g = torch.Generator().manual_seed(seed)
    N = cfg.n_patches
    zc = torch.randn(B * T, N, cfg.zr_c_video, generator=g).to(dtype)
    zd = torch.randn(B * T, cfg.d_tokens, cfg.zr_d, generator=g).to(dtype)
    qc = torch.stack([query_coords(cfg.grid, float(t), B) for t in range(T)], 1).flatten(0, 1).to(dtype)
    qm = modality_mask("video", N).unsqueeze(0).expand(B * T, -1, -1).to(dtype)
    return zc, zd, qc, qm


def _calls(m):
    """(rows, zd_len) of every decoder call of m, in call order."""
    calls, fwd = [], m.decoder.forward

    def wrapped(zc, zd, qc, qm, mm, *a, **kw):
        calls.append((int(zc.shape[0]), kw.get("zd_len")))
        return fwd(zc, zd, qc, qm, mm, *a, **kw)
    m.decoder.forward = wrapped
    return calls


def _nonzero_delta(dec, std=0.3):
    """The Δ heads are zero-initialised; give them weights so that the D read reaches Δ."""
    last = dec.delta_mlp[-1] if dec.cfg.delta_mode == "query" else dec.offset[-1]
    with torch.no_grad():
        last.weight.normal_(std=std)
        last.bias.normal_(std=std)


# ---------------------------------------------------------------------------- decoder: prefix masks
def test_flag_off_has_no_null_token_and_masks_still_mean_a_token_prefix():
    cfg = _v1()
    m = _model(cfg).double()
    _nonzero_delta(m.decoder)
    assert not hasattr(m.decoder, "d_null")
    B, T = 2, 3
    zc, zd, qc, qm = _flat_inputs(cfg, B, T, dtype=torch.float64)
    a = m.decoder(zc, zd, qc, qm, "video")
    b = m.decoder(zc, zd, qc, qm, "video", zd_len=torch.full((B * T,), cfg.d_tokens))
    c = m.decoder(zc, zd, qc, qm, "video", zd_len=torch.full((B * T,), 2))
    d = m.decoder(zc, zd[:, :2], qc, qm, "video")
    for k in ("image", "delta", "gate"):
        torch.testing.assert_close(a[k], b[k])                 # the full prefix is the usual decode
        torch.testing.assert_close(c[k], d[k])                 # a prefix mask is the sliced tokens
    with pytest.raises(ValueError):                            # no null token: an empty prefix has no key
        m.decoder(zc, zd, qc, qm, "video", zd_len=torch.zeros(B * T, dtype=torch.long))


@pytest.mark.parametrize("delta_mode", ["query", "mean"])
def test_prefix_mask_equals_sliced_tokens_with_the_null_token(delta_mode):
    cfg = _v1(d_nested=True, delta_mode=delta_mode)
    m = _model(cfg).double()
    _nonzero_delta(m.decoder)
    assert tuple(m.decoder.d_null.shape) == (1, 1, cfg.dec_dim)
    with torch.no_grad():
        m.decoder.d_null.normal_()                             # a null token that is not just zeros
    B, T = 2, 3
    zc, zd, qc, qm = _flat_inputs(cfg, B, T, dtype=torch.float64)
    for ell in (1, cfg.d_tokens // 2, cfg.d_tokens):
        a = m.decoder(zc, zd, qc, qm, "video", zd_len=torch.full((B * T,), ell))
        b = m.decoder(zc, zd[:, :ell], qc, qm, "video")
        for k in ("image", "delta", "gate", "feat"):
            torch.testing.assert_close(a[k], b[k], atol=1e-9, rtol=1e-7)


def test_empty_prefix_reads_only_the_null_token():
    cfg = _v1(d_nested=True)
    m = _model(cfg)
    _nonzero_delta(m.decoder)
    B, T = 2, 2
    zc, zd1, qc, qm = _flat_inputs(cfg, B, T, seed=0)
    _, zd2, _, _ = _flat_inputs(cfg, B, T, seed=5)
    zero = torch.zeros(B * T, dtype=torch.long)
    a = m.decoder(zc, zd1, qc, qm, "video", zd_len=zero)
    b = m.decoder(zc, zd2, qc, qm, "video", zd_len=zero)
    torch.testing.assert_close(a["image"], b["image"])         # nothing of zd gets through
    torch.testing.assert_close(a["delta"], b["delta"])
    full = m.decoder(zc, zd1, qc, qm, "video")["image"]
    assert not torch.allclose(a["image"], full, atol=1e-4)     # and the tokens do matter at full length


def test_prefix_lengths_can_differ_per_row():
    cfg = _v1(d_nested=True)
    m = _model(cfg).double()
    _nonzero_delta(m.decoder)
    B, T = 3, 1
    zc, zd, qc, qm = _flat_inputs(cfg, B, T, dtype=torch.float64)
    lens = torch.tensor([0, 1, cfg.d_tokens])
    mixed = m.decoder(zc, zd, qc, qm, "video", zd_len=lens)
    for r in range(B):
        one = m.decoder(zc[r:r + 1], zd[r:r + 1], qc[r:r + 1], qm[r:r + 1], "video", zd_len=lens[r:r + 1])
        torch.testing.assert_close(mixed["image"][r:r + 1], one["image"], atol=1e-9, rtol=1e-7)


# ---------------------------------------------------------------------------- model: the training draw
def test_training_draws_one_prefix_length_per_clip_shared_by_every_decode():
    cfg = _v1(d_nested=True)
    m = _model(cfg).train()
    calls = _calls(m)
    torch.manual_seed(3)
    out = m(_video(B=2, T=6), "video")
    assert out["loss"].isfinite()
    assert len(calls) == 3 and all(l is not None for _, l in calls)   # reconstruction, forecasts, warp-only
    per_clip = []
    for rows, l in calls:
        assert tuple(l.shape) == (rows,)
        l = l.view(2, -1)                                      # rows are clip-major
        assert (l == l[:, :1]).all()                           # one length per clip
        per_clip.append(l[:, 0])
    assert all(torch.equal(per_clip[0], p) for p in per_clip[1:])      # the same draw for the whole step
    assert ((per_clip[0] >= 0) & (per_clip[0] <= cfg.d_tokens)).all()
    assert {"zd_energy_head", "zd_energy_tail"} <= set(out["diag"])  # energy by token order, for the log


def test_training_draw_covers_every_length():
    cfg = _v1(d_nested=True)
    m = _model(cfg).train()
    calls = _calls(m)
    torch.manual_seed(0)
    seen = set()
    for _ in range(12):
        m(_video(B=4, T=4), "video")
        seen |= set(calls[-1][1].tolist())
    assert seen == set(range(cfg.d_tokens + 1))               # 0 .. M, 4 tokens x 48 clips


def test_eval_mode_and_images_decode_with_every_token():
    cfg = _v1(d_nested=True)
    m = _model(cfg).eval()
    calls = _calls(m)
    with torch.no_grad():
        m(_video(B=1, T=4), "video")
    assert calls and all(l is None for _, l in calls)
    m.train()
    calls.clear()
    m(torch.rand(2, 3, 64, 64) * 2 - 1, "image")
    assert calls and all(l is None for _, l in calls)


def test_flag_off_keeps_the_phase_2_step():
    cfg = _v1()
    m = _model(cfg).train()
    calls = _calls(m)
    torch.manual_seed(3)
    out = m(_video(B=2, T=6), "video")
    assert all(l is None for _, l in calls) and "zd_energy_head" not in out["diag"]


# ---------------------------------------------------------------------------- eval: distortion vs prefix length
class _Clips(torch.utils.data.Dataset):
    def __init__(self, n, T=4):
        self.x = torch.rand(n, 3, T, 64, 64, generator=torch.Generator().manual_seed(7)) * 2 - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"data": self.x[i]}


def test_prefix_lengths_are_a_short_ladder():
    assert E.prefix_lengths(4) == [0, 1, 2, 3, 4]
    assert E.prefix_lengths(64) == [0, 8, 16, 32, 48, 64]
    assert E.prefix_lengths(16) == [0, 2, 4, 8, 12, 16]


def test_video_pass_sweeps_the_prefix_length():
    cfg = _v1(d_nested=True)
    m = _model(cfg).eval()
    v = E.video_pass(m, E.make_loader(_Clips(3), 2), "cpu")
    M = cfg.d_tokens
    for ell in E.prefix_lengths(M):
        assert f"psnr_dlen{ell}" in v and f"psnr_dlen{ell}_nonkey" in v
    assert math.isclose(v[f"psnr_dlen{M}"], v["psnr_full"], abs_tol=1e-3)   # the full prefix is the usual decode
    assert "motion_l1" in v["per_clip"][0] and v["per_clip"][0]["motion_l1"] > 0
    md = E.to_markdown({"video_openvid": v})                  # video results sit under video_<name>
    assert "D prefix length" in md and "high motion" in md and f"n = {M}" in md
    plain = E.video_pass(_model(_v1()).eval(), E.make_loader(_Clips(2), 2), "cpu")
    assert not any(k.startswith("psnr_dlen") for k in plain)


def test_video_pass_sweeps_given_prefix_lengths():
    cfg = _v1(d_nested=True)
    m = _model(cfg).eval()
    v = E.video_pass(m, E.make_loader(_Clips(2), 2), "cpu", prefix_lens=[2, 0, 1])
    assert sorted(k for k in v if k.startswith("psnr_dlen") and k[9:].isdigit()) == ["psnr_dlen0", "psnr_dlen1", "psnr_dlen2"]
    with pytest.raises(AssertionError):
        E.video_pass(m, E.make_loader(_Clips(2), 2), "cpu", prefix_lens=[cfg.d_tokens + 1])


# ---------------------------------------------------------------------------- the run config
def test_v1c_alpha_yaml():
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(ROOT, "configs", "dino_v0", "v1c_alpha.yaml")) as f:
        y = yaml.safe_load(f)
    train = y.pop("train")
    cfg = DinoV0Config(**y)
    assert cfg.d_nested and cfg.d_tokens == 64 and cfg.zr_d == 24 and cfg.zr_c_video == 24
    assert cfg.budget("video") == 12288 and cfg.budget("image") == 12288
    assert cfg.fut_frames_per_horizon == 2 and cfg.warp_frames == 2        # phase-2 speed options kept
    assert (cfg.delta_mode, cfg.rgb_head) == ("query", "conv")
    assert train["mix_modalities"] is True and train["compile"] is True
