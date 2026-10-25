"""Depth part of the C-D 3.2 tokenizer: labels on cells, invariant losses, metrics, data, probe, prior in the model."""
import importlib.util
import json
import os
import random
import sys

import numpy as np
import pytest
import torch

from mavt.cd32 import MAVTCD32, tiny_config
from mavt.cd32.depth import (DepthProbe, apply_probe, cell_log_inv_depth, depth_metrics, load_probe, ordinal,
                             prior_loss, silog, ssi)
from mavt.cd32.depth_data import DepthManifest, crop_pair, read_depth, resample_depth

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _script(name):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------- labels and losses
def test_cell_targets_average_valid_pixels_and_mask_sparse_cells():
    d = torch.full((2, 8, 8), 2.0)
    d[0, :4, :4] = 0                                   # one cell fully invalid
    d[1, :4, :4] = 0
    d[1, 0, 0] = 4.0                                   # one valid pixel of 16 in that cell
    t, m = cell_log_inv_depth(d, d > 0, grid=2)
    assert torch.allclose(t[0, 1:], torch.full((3,), -np.log(2.0)))
    assert not m[0, 0] and m[0, 1:].all() and not m[1, 0]
    t, m = cell_log_inv_depth(d, d > 0, grid=2, min_frac=torch.tensor([0.5, 0.0]))
    assert m[1, 0] and torch.isclose(t[1, 0], torch.tensor(-np.log(4.0), dtype=torch.float32))
    assert not m[0, 0]                                 # no valid pixel never counts


def test_silog_is_shift_invariant_with_lambda_one_and_ssi_affine_invariant():
    g = torch.Generator().manual_seed(0)
    p, t = torch.randn(3, 64, generator=g), torch.randn(3, 64, generator=g)
    assert torch.isclose(silog(p + 1.7, t, lam=1.0), silog(p, t, lam=1.0), atol=1e-5)
    assert torch.isclose(silog(t + 1.7, t, lam=0.85), torch.tensor(0.15 ** 0.5 * 1.7), atol=1e-4)
    assert silog(t, t, lam=1.0) < 1e-3
    assert torch.isclose(ssi(3 * p - 2, t), ssi(p, t), atol=1e-5)
    mask = torch.rand(3, 64, generator=g) > 0.3
    assert torch.isclose(silog(p + 5, t, mask, lam=1.0), silog(p, t, mask, lam=1.0), atol=1e-5)


def test_ordinal_prefers_the_right_order():
    t = torch.linspace(-1, 1, 64).expand(2, -1)
    gen = torch.Generator().manual_seed(0)
    good = ordinal(2 * t, t, generator=gen)
    gen = torch.Generator().manual_seed(0)
    bad = ordinal(-2 * t, t, generator=gen)
    assert good < bad


def test_metrics_median_alignment_forgives_scale_only():
    gt = torch.rand(2, 100) * 5 + 0.5
    pred_lid = -torch.log(gt * 3.0)                   # right up to a factor 3
    m = depth_metrics(pred_lid, gt, torch.ones_like(gt, dtype=torch.bool))
    assert m["n"] == 2 and m["abs_rel"] / 2 < 1e-5 and m["d1"] / 2 > 0.999
    m = depth_metrics(pred_lid, gt, torch.ones_like(gt, dtype=torch.bool), align="none")
    assert abs(m["abs_rel"] / 2 - 2.0) < 1e-4


def test_prior_loss_kinds():
    p, t = torch.randn(2, 16, requires_grad=True), torch.randn(2, 16)
    for k in ("silog", "ssi", "ordinal"):
        l = prior_loss(k, p, t)
        assert torch.isfinite(l)
    with pytest.raises(ValueError):
        prior_loss("l2", p, t)


# ---------------------------------------------------------------------------------- data
def test_resample_depth_keeps_validity():
    dense = np.full((64, 64), 2.0, np.float32)
    dense[:32] = 8.0
    out = resample_depth(dense, 0, 0, 64, 64, 8)
    assert np.allclose(out[:4], 8.0) and np.allclose(out[4:], 2.0)
    sparse = np.zeros((100, 100), np.float32)
    sparse[55, 22] = 3.0                               # one LiDAR-like point
    out = resample_depth(sparse, 0, 0, 100, 100, 10)
    assert out[5, 2] == pytest.approx(3.0) and (out > 0).sum() == 1
    coarse = np.arange(16, dtype=np.float32).reshape(4, 4) + 1
    out = resample_depth(coarse, 0, 0, 4, 4, 8)        # coarser than the output: nearest neighbour
    assert out.shape == (8, 8) and out[0, 0] == 1 and out[7, 7] == 16


def test_crop_pair_matches_depth_of_other_resolution():
    from PIL import Image
    img = Image.fromarray(np.zeros((40, 80, 3), np.uint8))
    depth = np.tile(np.linspace(1, 2, 20, dtype=np.float32), (10, 1))   # quarter resolution, same aspect
    a, d = crop_pair(img, depth, 16, None)
    assert a.shape == (16, 16, 3) and d.shape == (16, 16)
    assert 1.4 < d.mean() < 1.6                          # the centre crop of a left-to-right ramp
    a, d = crop_pair(img, depth, 16, random.Random(0))
    assert (d > 0).all()


def test_manifest_rows_with_stacked_npy(tmp_path):
    from PIL import Image
    Image.fromarray(np.full((32, 48, 3), 128, np.uint8)).save(tmp_path / "im.jpg")
    np.save(tmp_path / "depth.npy", np.stack([np.full((8, 12), k + 1.0, np.float16) for k in range(3)]))
    rows = [dict(image="im.jpg", depth="depth.npy", index=2, max_depth=10.0, min_cell_frac=0.1)]
    (tmp_path / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = DepthManifest(str(tmp_path / "m.jsonl"), size=16, train=False)
    it = ds[0]
    assert it["data"].shape == (3, 16, 16) and torch.allclose(it["depth"], torch.full((16, 16), 3.0))
    assert it["min_cell_frac"] == 0.1
    assert read_depth(str(tmp_path / "depth.npy"), index=0).shape == (8, 12)


# ---------------------------------------------------------------------------------- probe
def _probe_file(tmp_path, cfg, seed=0):
    torch.manual_seed(seed)
    p = DepthProbe(cfg.stub_dino_dim, cfg.dino_layers)
    with torch.no_grad():
        p.linear.weight.normal_(0, 0.1)
    path = str(tmp_path / "probe.pt")
    torch.save(p.export(dino_repo=None, image_size=cfg.image_size, patch=cfg.patch), path)
    return path, p


def test_probe_export_round_trip(tmp_path):
    cfg = tiny_config()
    path, p = _probe_file(tmp_path, cfg)
    ck = load_probe(path)
    q = DepthProbe.from_export(ck)
    feats = [torch.randn(2, cfg.n_cells, cfg.stub_dino_dim) for _ in cfg.dino_layers]
    assert torch.allclose(p(feats), q(feats), atol=1e-6)
    assert torch.allclose(apply_probe(feats, ck["weight"], ck["bias"]), p(feats), atol=1e-5)
    torch.save({"weight": torch.zeros(1)}, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        load_probe(str(tmp_path / "bad.pt"))


def test_prior_trains_image_rho_and_probe_is_not_saved(tmp_path):
    path, _ = _probe_file(tmp_path, tiny_config())
    torch.manual_seed(0)
    m = MAVTCD32(tiny_config(depth_probe=path, w_depth_prior=1.0, w_smooth=0.0))
    with torch.no_grad():
        m.c_heads.rho_mlp.fc2.weight.normal_(0, 0.1)
    out = m(torch.rand(2, 3, 64, 64) * 2 - 1, "image")
    assert "depth_prior" in out["parts"] and "rho_probe_corr" in out["diag"]
    out["loss"].backward()
    assert m.c_heads.rho_mlp.fc2.weight.grad.abs().sum() > 0          # images: rho learns from the prior only
    assert not any(k.startswith("probe_") for k in m.state_dict())
    plain = MAVTCD32(tiny_config())
    assert set(plain.state_dict()) == set(m.state_dict())            # resume works with or without a probe
    m.load_state_dict(plain.state_dict())


def test_prior_off_for_threed_by_default_and_log_only_without_weight(tmp_path):
    path, _ = _probe_file(tmp_path, tiny_config())
    from .test_model import _orbit
    torch.manual_seed(0)
    m = MAVTCD32(tiny_config(depth_probe=path, w_depth_prior=0.0))
    out = m(torch.rand(1, 3, 8, 64, 64) * 2 - 1, "video")
    assert "depth_prior" in out["parts"]                              # logged, not trained
    m2 = MAVTCD32(tiny_config(depth_probe=path, w_depth_prior=1.0, w_smooth=0.0, w_kl_c=0.0))
    x, cam = torch.rand(1, 3, 4, 64, 64) * 2 - 1, _orbit(1, 4, 64)
    torch.manual_seed(1)
    a = m2(x, "threed", cam)["loss"]
    m2.cfg.depth_prior_on = ("image", "video", "threed")
    torch.manual_seed(1)
    b = m2(x, "threed", cam)["loss"]
    assert not torch.isclose(a, b)


def test_rendered_depth_loss_on_posed_views():
    from .test_model import _orbit
    torch.manual_seed(0)
    m = MAVTCD32(tiny_config(w_depth_gt=1.0))
    cam = _orbit(2, 4, 64)
    cam["depth"] = torch.full((2, 4, 64, 64), 2.0)
    cam["depth"][:, :, :8] = 0                                        # background
    out = m(torch.rand(2, 3, 4, 64, 64) * 2 - 1, "threed", cam)
    assert "depth_gt" in out["parts"] and torch.isfinite(out["parts"]["depth_gt"])
    out["loss"].backward()
    assert m.c_heads.rho_mlp.fc2.bias.grad.abs().sum() > 0           # lambda < 1: the scale is supervised


def test_rho_distill_teaches_the_single_view_part_only():
    from .test_model import _orbit
    torch.manual_seed(0)
    off = dict(w_rgb=0.0, w_feat=0.0, w_geo=0.0, w_pred=0.0, w_flow=0.0, w_smooth=0.0, w_kl_c=0.0, w_kl_d=0.0)
    m = MAVTCD32(tiny_config(w_rho_distill=1.0, **off))                # the distill term is the whole loss
    x, cam = torch.rand(2, 3, 4, 64, 64) * 2 - 1, _orbit(2, 4, 64)
    assert m(x, "threed", cam)["parts"]["rho_distill"] == 0          # gate closed: rho = rho_mono
    with torch.no_grad():
        m.c_heads.alpha.fill_(1.0)
    out = m(x, "threed", cam)
    assert out["parts"]["rho_distill"] > 0 and torch.isclose(out["loss"], out["parts"]["rho_distill"])
    out["loss"].backward()
    assert m.c_heads.rho_mlp.fc2.weight.grad.abs().sum() > 0
    assert m.c_heads.refine_out.weight.grad.abs().sum() == 0           # the teacher is not pulled back
    cam["depth"] = torch.zeros(2, 4, 64, 64)                           # all background: nothing to distill
    assert m(x, "threed", cam)["parts"]["rho_distill"] == 0


def test_model_checks_probe_layers(tmp_path):
    cfg = tiny_config()
    p = DepthProbe(cfg.stub_dino_dim, (1, 3))
    torch.save(p.export(), tmp_path / "p.pt")
    with pytest.raises(ValueError):
        MAVTCD32(tiny_config(depth_probe=str(tmp_path / "p.pt")))
    with pytest.raises(ValueError):
        MAVTCD32(tiny_config(w_depth_prior=0.1))


# ---------------------------------------------------------------------------------- scripts
def test_probe_trainer_and_eval_dry_run(tmp_path):
    tr = _script("train_depth_probe")
    out = tmp_path / "probe"
    assert tr.main(["--dry-run", "--out", str(out)]) == 0
    ck = load_probe(str(out / "probe.pt"))
    val = json.loads((out / "val.json").read_text())
    assert {"probe_cell", "probe_pixel", "const_pixel"} <= set(val) and ck["layers"] == [6, 12, 18, 24]
    ev = _script("eval_cd32_depth")
    assert ev.main(["--dry-run", "--out", str(tmp_path / "ev.json")]) == 0
    res = json.loads((tmp_path / "ev.json").read_text())
    assert {"rho_cell", "rho_pixel", "const_pixel"} <= set(res["images"])


def test_cd32_trainer_set_overrides_with_probe(tmp_path):
    path, _ = _probe_file(tmp_path, tiny_config())
    tr = _script("train_cd32")
    out = tmp_path / "run"
    assert tr.main(["--dry-run", "--out", str(out), "--set", f"depth_probe={path}",
                    "--set", "w_depth_prior=0.1", "--set", "depth_prior_on=[image, video]"]) == 0
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert "loss_depth_prior" in recs[0] and np.isfinite(recs[0]["loss"])
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ck["cfg"]["w_depth_prior"] == 0.1 and not any(k.startswith("probe_") for k in ck["model"])
    with pytest.raises(SystemExit):
        tr.main(["--dry-run", "--out", str(out), "--set", "no_such_field=1"])


def test_objaverse_depth_decoding():
    from mavt.dino_v0.data import objaverse_depth
    raw = np.full((4, 4), 65535, np.uint16)
    raw[1:3, 1:3] = 26214                              # 26214 / 65535 * 5 = 2.0
    alpha = np.zeros((4, 4), np.uint8)
    alpha[1:3, 1:3] = 255
    alpha[1, 1] = 0                                    # rim pixel without alpha
    z = objaverse_depth(raw, alpha, 5.0, 8)
    assert z.shape == (8, 8) and z.max() == pytest.approx(2.0, abs=1e-3)
    assert (z > 0).sum() == 3 * 4 and z[0, 0] == 0


def test_cd32_trainer_with_rendered_depth(tmp_path):
    tr = _script("train_cd32")
    out = tmp_path / "gt"
    assert tr.main(["--dry-run", "--out", str(out), "--objaverse-depth", "--set", "w_depth_gt=0.5"]) == 0
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    threed = [r for r in recs if r["modality"] == "threed"][0]
    assert "loss_depth_gt" in threed and "diag_rho_gt_corr" in threed and np.isfinite(threed["loss"])
