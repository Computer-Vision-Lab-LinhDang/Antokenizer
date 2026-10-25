"""Vanilla C-D model on the tiny config: shapes, the asymmetric mask, causality, L = 0, budget, the trainer."""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from mavt.cd32.vanilla import CDVConfig, VanillaBackbone, tiny_cdv_config
from mavt.cd32.vanilla_model import MAVTCDV, load_cdv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _model(**kw):
    torch.manual_seed(0)
    return MAVTCDV(tiny_cdv_config(**kw))


def _orbit(B, V, size):
    c2w = torch.zeros(B, V, 4, 4)
    for v in range(V):
        a = 2 * math.pi * v / V
        o = torch.tensor([2 * math.cos(a), 2 * math.sin(a), 0.3])
        z = -o / o.norm()
        x = torch.linalg.cross(z, torch.tensor([0.0, 0.0, -1.0]))
        x = x / x.norm()
        y = torch.linalg.cross(z, x)
        c2w[:, v, :3, 0], c2w[:, v, :3, 1], c2w[:, v, :3, 2], c2w[:, v, :3, 3] = x, y, z, o
        c2w[:, v, 3, 3] = 1
    f = size / 2 / math.tan(0.3456)
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]]).expand(B, V, 3, 3).clone()
    return dict(c2w=c2w, K=K)


def _inputs(modality, B=2, T=8):
    if modality == "image":
        return torch.rand(B, 3, 64, 64) * 2 - 1, None
    T = T if modality == "video" else 4
    return torch.rand(B, 3, T, 64, 64) * 2 - 1, (_orbit(B, 4, 64) if modality == "threed" else None)


@pytest.mark.parametrize("modality", ["image", "video", "threed"])
def test_step_finite_and_backward(modality):
    m = _model()
    x, cam = _inputs(modality)
    out = m(x, modality, cam, return_outputs=True)
    assert torch.isfinite(out["loss"])
    assert all(torch.isfinite(v) for v in out["parts"].values())
    out["loss"].backward()
    o = out["outputs"]
    assert o["rgb"].shape == (x.unsqueeze(1).shape if modality == "image" else x.permute(0, 2, 1, 3, 4).shape)
    assert o["rho"].shape[-1] == 16
    if modality == "image":
        assert o["c"].shape[-1] == 11 and "kl_d" not in out["parts"]
        assert out["diag"]["numbers"].item() == 192
    else:
        assert o["c"].shape[-1] == 5 and "kl_d" in out["parts"] and "geo_rgb" in out["parts"]
        assert out["diag"]["numbers"].item() <= 192
    if modality == "video":
        assert o["xi"].shape == (4, 3, 6) and o["M"].shape == (4, 3, 2, 2, 3) and o["tokens"].shape[1] == 3
        assert "xi_rot_deg" in out["diag"] and "flow_l1" in out["parts"]
    grads = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any(n.startswith("decoder.read.") for n in grads) and any(n.startswith("backbone.blocks.") for n in grads)
    if modality != "image":
        assert "heads.slot_out.weight" in grads


def test_c_reads_only_the_keyframe():
    """The asymmetric mask: C (and rho, fov) of a segment do not depend on the frames after the keyframe."""
    m = _model().eval()
    x, _ = _inputs("video", T=4)
    y = x.clone()
    y[:, :, 1:] = torch.rand_like(y[:, :, 1:]) * 2 - 1
    with torch.no_grad():
        a = m(x, "video", return_outputs=True)["outputs"]
        b = m(y, "video", return_outputs=True)["outputs"]
    assert torch.allclose(a["c"], b["c"], atol=1e-5) and torch.allclose(a["rho"], b["rho"], atol=1e-5)
    assert torch.allclose(a["fov"], b["fov"], atol=1e-5)
    assert not torch.allclose(a["tokens"][:, 1], b["tokens"][:, 1], atol=1e-5)


def test_d_is_sparse_causal():
    """D_1 reads C and itself only, so changing frames 2 and 3 leaves xi_1, M_1 and the tokens of frame 1."""
    m = _model().eval()
    x, _ = _inputs("video", T=4)
    y = x.clone()
    y[:, :, 2:] = torch.rand_like(y[:, :, 2:]) * 2 - 1
    with torch.no_grad():
        a = m(x, "video", return_outputs=True)["outputs"]
        b = m(y, "video", return_outputs=True)["outputs"]
    assert torch.allclose(a["xi"][:, 0], b["xi"][:, 0], atol=1e-5)
    assert torch.allclose(a["tokens"][:, 0], b["tokens"][:, 0], atol=1e-5)
    assert not torch.allclose(a["tokens"][:, 1], b["tokens"][:, 1], atol=1e-5)
    m2 = _model(causal_d=False).eval()
    with torch.no_grad():
        a = m2(x, "video", return_outputs=True)["outputs"]
        b = m2(y, "video", return_outputs=True)["outputs"]
    assert not torch.allclose(a["tokens"][:, 0], b["tokens"][:, 0], atol=1e-5)


def test_mask_shape_and_rows():
    cfg = tiny_cdv_config()
    bb = VanillaBackbone(cfg)
    NC, ND = bb.n_c, bb.n_d
    m = bb.mask(3, "cpu")[0, 0]
    assert m.shape == (NC + 3 * ND, NC + 3 * ND)
    assert (m[:NC, :NC] == 0).all() and torch.isinf(m[:NC, NC:]).all()
    r2 = slice(NC + ND, NC + 2 * ND)
    assert (m[r2, :NC] == 0).all() and (m[r2, NC:NC + ND] == 0).all() and (m[r2, r2] == 0).all()
    assert torch.isinf(m[r2, NC + 2 * ND:]).all()


def test_l_zero_is_geometry_only():
    """With L = 0 the decoder output does not depend on the innovation tokens."""
    m = _model().eval()
    x, _ = _inputs("video", T=4)
    L0 = torch.zeros(2, 3, dtype=torch.long)
    Lm = torch.full((2, 3), m.cfg.L_max, dtype=torch.long)
    with torch.no_grad():
        a = m(x, "video", L=L0, return_outputs=True)
        b = m(x, "video", L=L0, return_outputs=True)
        c = m(x, "video", L=Lm, return_outputs=True)
    assert torch.allclose(a["outputs"]["rgb"], b["outputs"]["rgb"], atol=1e-5)
    assert a["diag"]["L_mean"].item() == 0 and c["diag"]["L_mean"].item() == m.cfg.L_max
    assert (a["outputs"]["tokens"] == 0).all() and not (c["outputs"]["tokens"] == 0).all()
    assert not torch.allclose(a["outputs"]["rgb"], c["outputs"]["rgb"], atol=1e-5)


def test_geometry_ablation_and_conv_head():
    m = _model(geometry=False, rgb_head="conv")
    x, _ = _inputs("video", T=4)
    out = m(x, "video", return_outputs=True)
    assert torch.isfinite(out["loss"]) and out["diag"]["xi_rot_deg"].item() == 0
    out["loss"].backward()
    assert m.heads.pose.weight.grad is None or m.heads.pose.weight.grad.abs().sum() == 0
    assert out["outputs"]["rgb"].shape[-2:] == (64, 64)


def test_budget_and_config():
    cfg = CDVConfig()
    assert cfg.L_max == 82 and cfg.segment_numbers("video") <= cfg.budget and cfg.segment_numbers("image") == 12288
    assert cfg.w_lpips == 0.5 and cfg.rgb_head == "linear"
    with pytest.raises(AssertionError):
        CDVConfig(rgb_head="pixel").check()


def test_too_many_views_for_the_frame_embedding():
    m = _model()
    x = torch.rand(1, 3, 6, 64, 64) * 2 - 1
    with pytest.raises(ValueError):
        m(x, "threed", _orbit(1, 6, 64))


def _trainer():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("train_cdv", os.path.join(ROOT, "scripts", "train_cdv.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_and_staged_init(tmp_path):
    tr = _trainer()
    out = tmp_path / "s1"
    assert tr.main(["--dry-run", "--mix-modalities", "--out", str(out)]) == 0
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert all(r["modality"] == "mixed" and np.isfinite(r["loss"]) for r in recs)
    assert {"image/loss", "video/loss", "threed/loss"} <= set(recs[0])
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 3 and ck["cfg"]["dec_blocks"] == 1
    assert any(k.startswith("backbone.blocks.") for k in ck["model"]) and not any("enc_r" in k for k in ck["model"])
    model, cfg, step = load_cdv(str(out / "last.pt"))
    assert step == 3 and isinstance(cfg, CDVConfig)
    out2 = tmp_path / "s2"
    assert tr.main(["--dry-run", "--init", str(out / "last.pt"), "--set", "geometry=false", "--out", str(out2)]) == 0
    ck2 = torch.load(out2 / "last.pt", map_location="cpu", weights_only=False)
    assert ck2["step"] == 3 and ck2["cfg"]["geometry"] is False
    k = "backbone.pos_emb"
    assert not torch.equal(ck2["model"][k], ck["model"][k])          # trained on from the stage-1 weights


class _CountingLPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen, self.w = [], torch.nn.Parameter(torch.zeros(1))

    def forward(self, pred, target):
        self.seen.append(pred.shape[0])
        return (pred - target).abs().mean() + 0 * self.w.sum()


def test_lpips_on_a_subset_of_frames_and_resume():
    m = _model(lpips_max_frames=2)
    m.lpips = _CountingLPIPS()
    x, _ = _inputs("video", T=4)                                  # 2 clips x 4 frames = 8 decoded frames
    out = m(x, "video")
    assert "lpips" in out["parts"] and m.lpips.seen == [2]
    m.cfg.lpips_max_frames = 0
    m(x, "video")
    assert m.lpips.seen == [2, 8]
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("lpips.")}   # LPIPS is never saved
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "lpips" in k] and not unexpected


def _evaluator():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("eval_cdv", os.path.join(ROOT, "scripts", "eval_cdv.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_dry_run_on_a_trained_checkpoint(tmp_path):
    """scripts/eval_cdv.py: checkpoint kind detection, the PSNR-vs-L curve, key vs non-key frames, depth corr."""
    tr, ev = _trainer(), _evaluator()
    run = tmp_path / "s1"
    assert tr.main(["--dry-run", "--out", str(run)]) == 0
    assert ev.checkpoint_kind(str(run / "last.pt")) == "vanilla"
    out = tmp_path / "ev"
    assert ev.main(["--dry-run", "--ckpt", str(run / "last.pt"), "--L-list", "0,1,max,99", "--out", str(out)]) == 0
    res = json.loads((out / "eval.json").read_text())
    assert res["kind"] == "vanilla" and res["step"] == 3 and res["L_list"] == [0, 1, res["L_max"]]
    im = res["images"]
    assert im["n_images"] == 4 and np.isfinite(im["psnr"]) and 0 <= im["ssim"] <= 1 and im["numbers"] == 192
    for v in (res["video"]["synthetic"], res["threed"]):
        Ls = [r["L"] for r in v["curve"]]
        assert Ls == res["L_list"] and all(np.isfinite(r["psnr"]) for r in v["curve"])
        nums = [r["numbers"] for r in v["curve"]]
        assert nums == sorted(nums) and nums[-1] <= res["budget"] and nums[0] < nums[-1]
        assert np.isfinite(v["key"]["psnr"]) and v["n_clips"] == 4 and v["n_frames"] == 4
        assert v["diag"]["L_mean"] == res["L_max"]
    assert "xi_rot_deg" in res["video"]["synthetic"]["diag"] and "flow_l1" in res["video"]["synthetic"]["diag"]
    assert "rho_gt_corr" in res["threed"]["diag"]
    md = (out / "eval.md").read_text()
    assert "| L | numbers | PSNR |" in md and "Key frames (C only)" in md
    assert ev.parse_L("max,0,0,5", 3) == [0, 3]
