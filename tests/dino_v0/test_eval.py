import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mavt.dino_v0 import evaluate as E
from mavt.dino_v0.config import tiny_config
from mavt.dino_v0.model import MAVTDinoV0

ROOT = Path(__file__).resolve().parents[2]


class _Images(torch.utils.data.Dataset):
    def __init__(self, n, size=64, classes=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, size, size, generator=g) * 2 - 1
        self.y = torch.arange(n) % classes

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"data": self.x[i], "label": int(self.y[i])}


class _Clips(torch.utils.data.Dataset):
    def __init__(self, n, T=8, size=64, seed=1, static=False, stride=None):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, T, size, size, generator=g) * 2 - 1
        if static:                                                   # every frame equals frame 0
            self.x = self.x[:, :, :1].expand_as(self.x).contiguous()
        self.stride = stride

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"data": self.x[i]} if self.stride is None else {"data": self.x[i], "stride": self.stride}


class _V1Like(torch.nn.Module):
    """The tiny v0 model behind the v1 API: encode() adds zd_fut {1, 2}; decode_frame takes warp_only and
    returns 'image' (here: zeros, to check it is preferred over rgb). `drop` removes encode() keys."""

    def __init__(self, base, drop=(), warp=True):
        super().__init__()
        self.base, self.cfg, self.drop, self.warp_calls = base, base.cfg, drop, 0
        if not warp:
            self.decode_frame = self._decode_v0

    def encode(self, data, modality, cam=None):
        e = self.base.encode(data, modality, cam)
        T = e["frames"].shape[1]
        e["zd_fut"] = {1: e["zd_next"][:, :T - 1], "2": e["zd"][:, :T - 2]}
        for k in self.drop:
            e.pop(k, None)
        return e

    def decode_frame(self, zc, zd, t, modality, idx=None, rays=None, warp_only=False):
        self.warp_calls += int(warp_only)
        o = dict(self.base.decode_frame(zc, zd, t, modality, idx, rays))
        o["image"] = torch.zeros(zc.shape[0], 3, self.cfg.image_size, self.cfg.image_size)
        return o

    def _decode_v0(self, zc, zd, t, modality, idx=None, rays=None):
        return self.base.decode_frame(zc, zd, t, modality, idx, rays)


def _separable(n_per=50, classes=5, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(classes, dim, generator=g) * 5
    y = torch.arange(classes).repeat_interleave(n_per)
    x = centres[y] + torch.randn(len(y), dim, generator=g) * 0.3
    return x, y


def test_psnr_ssim_identity():
    x = torch.rand(3, 3, 32, 32) * 2 - 1
    assert torch.all(E.psnr(x, x) == E.PSNR_CAP)
    assert torch.allclose(E.ssim(x, x), torch.ones(3), atol=1e-5)
    y = (x + 0.1 * torch.randn_like(x)).clamp(-1, 1)
    assert torch.all(E.psnr(x, y) < 40) and torch.all(E.ssim(x, y) < 1)


def test_knn_and_linear_probe_separable():
    xtr, ytr = _separable(seed=0)
    xte, yte = _separable(n_per=20, seed=0)
    assert E.knn_classify(xtr, ytr, xte, yte, k=20) >= 0.99
    lp = E.linear_probe(xtr, ytr, xte, yte, epochs=30, batch=64, lrs=(1e-2, 3e-2))
    assert lp["top1"] >= 0.9


def test_class_balanced_indices():
    lab = torch.tensor([0] * 10 + [1] * 3 + [2] * 7)
    idx = E.class_balanced_indices(lab, 5)
    counts = torch.bincount(lab[idx])
    assert counts.tolist() == [5, 3, 5]


def test_full_eval_path(tmp_path):
    torch.manual_seed(0)
    model = MAVTDinoV0(tiny_config()).eval()
    res = E.run_eval(model, "cpu", image_val=_Images(10), image_train=_Images(12, seed=3),
                     video=_Clips(3), batch_size=4, video_batch_size=2, limit=8, video_limit=3,
                     probe_epochs=2, probe_lrs=(1e-2,), extra_video={"ssv2": None, "kinetics": None})
    ir = res["image_recon"]
    assert ir["n"] == 8 and ir["budget"] == model.cfg.budget("image")
    assert 0 < ir["psnr"] < 100 and -1 <= ir["ssim"] <= 1
    u = res["understanding"]
    assert set(u) >= {"zs", "zr_c", "cls"} and u["n_val"] == 8 and u["n_train"] == 12
    assert u["zs"]["dim"] == model.cfg.zs_dim and u["zr_c"]["dim"] == model.cfg.zr_c_image
    assert "knn_gap_to_cls" in u["zs"]
    v = res["video_recon"]
    assert v["n_clips"] == 3 and v["n_frames"] == 8
    for k in ("psnr_key", "psnr_nonkey", "psnr_zeroD", "gap_zeroD", "gap_noD", "gate", "delta_abs_cells"):
        assert k in v
    assert any("ssv2" in n for n in res["notes"])
    jp, mp = E.write_results(res, str(tmp_path))
    assert json.load(open(jp))["video_recon"]["n_clips"] == 3
    assert "Understanding" in open(mp).read()


CD_KEYS = ("psnr_full", "psnr_noD", "psnr_zeroD", "psnr_eraseC", "gap_eraseC", "delta_cells_mean", "delta_cells_p95",
           "gate", "cos_full")
FC_KEYS = ("psnr", "psnr_copy_rec", "psnr_copy_gt", "cos", "cos_copy_rec", "cos_copy_gt")


def test_video_pass_cd_split_and_forecast_v0():
    """A v0 model: C/D split and the h = 1 forecast (from zd_next) are measured, warp is skipped."""
    torch.manual_seed(0)
    model = MAVTDinoV0(tiny_config()).eval()
    v = E.video_pass(model, E.make_loader(_Clips(3, stride=4), 2), "cpu")
    for k in CD_KEYS + tuple(f"forecast_h1_{m}" for m in FC_KEYS):
        for kind in ("", "_key", "_nonkey"):
            assert k + kind in v, k + kind
    assert v["psnr_full"] == pytest.approx(v["psnr"]) and v["psnr_full_key"] == pytest.approx(v["psnr_key"])
    assert "psnr_warp" not in v and any("warp" in s for s in v["skipped"])
    assert v["forecast_horizons"] == [1] and "forecast_h2_psnr" not in v and v["frame_stride"] == 4
    assert v["delta_cells_p95"] >= 0 and 0 < v["gate"] <= 1 and -1 <= v["cos_full"] <= 1
    pc = v["per_clip"]
    assert [r["clip"] for r in pc] == [0, 1, 2] and all(r["stride"] == 4 for r in pc)
    assert all("psnr_full_nonkey" in r and "delta_cells_p95_key" in r and "forecast_h1_psnr" in r for r in pc)
    assert sum(r["psnr_full"] for r in pc) / 3 == pytest.approx(v["psnr_full"], abs=1e-4)


def test_video_pass_v1_api_warp_and_horizons():
    torch.manual_seed(0)
    model = _V1Like(MAVTDinoV0(tiny_config()).eval())
    clips = _Clips(2, static=True)
    v = E.video_pass(model, E.make_loader(clips, 2), "cpu")
    assert model.warp_calls == 8                                    # one warp-only decode per frame
    for kind in ("", "_key", "_nonkey"):
        assert f"psnr_warp{kind}" in v and f"gap_warp{kind}" in v
    assert not any("warp" in s for s in v["skipped"])
    assert v["forecast_horizons"] == [1, 2]
    for h in (1, 2):
        for m in FC_KEYS:
            assert f"forecast_h{h}_{m}" in v
    # static clip: copying ground-truth frame t is perfect at every horizon
    assert v["forecast_h1_psnr_copy_gt"] == E.PSNR_CAP and v["forecast_h2_psnr_copy_gt"] == E.PSNR_CAP
    assert v["forecast_h2_cos_copy_gt"] == pytest.approx(1.0, abs=1e-5)
    # decode_frame's 'image' (zeros) is used instead of rgb
    want = E.psnr(torch.zeros(2, 3, 64, 64), clips.x[:, :, 0]).mean().item()
    assert v["psnr_full"] == pytest.approx(want, abs=1e-4)
    # forecasts are bucketed by target frame: h = 2 never targets frames 0 / 1
    assert all("forecast_h2_psnr" in r for r in v["per_clip"])


def test_video_pass_degrades_without_forecast_or_features():
    torch.manual_seed(0)
    model = _V1Like(MAVTDinoV0(tiny_config()).eval(), drop=("zd_fut", "zd_next", "feat_target"), warp=False)
    res = E.run_eval(model, "cpu", video=_Clips(2), video_batch_size=2)
    v = res["video_recon"]
    assert "psnr_full" in v and "psnr_eraseC" in v and "delta_cells_p95" in v
    assert v["forecast_horizons"] == [] and not any(k.startswith("forecast_") and k != "forecast_horizons" for k in v)
    assert not any(k.startswith("cos") for k in v) and "psnr_warp" not in v
    assert {s.split(" ")[0] for s in v["skipped"]} >= {"forecast", "feature", "psnr_warp"}
    assert any(n.startswith("video_recon: forecast skipped") for n in res["notes"])
    md = E.to_markdown(res)
    assert "C/D split" in md and "Forecast of frame" not in md


def test_markdown_has_cd_and_forecast_tables(tmp_path):
    torch.manual_seed(0)
    model = MAVTDinoV0(tiny_config()).eval()
    res = E.run_eval(model, "cpu", video=_Clips(2, stride=2), video_batch_size=2,
                     extra_video={"ssv2": _Clips(2, T=4, stride=8)})
    md = E.to_markdown(res)
    assert "C/D split" in md and "Forecast of frame" in md and "frame stride 8" in md
    assert res["video_ssv2"]["n_frames"] == 4 and "per_clip" in res["video_ssv2"]
    jp, _ = E.write_results(res, str(tmp_path))
    j = json.load(open(jp))
    assert len(j["video_recon"]["per_clip"]) == 2 and "forecast_h1_psnr_copy_rec" in j["video_recon"]


def test_grouped_batches_and_balanced_selection():
    groups = [i // 10 for i in range(40)]
    b = E.grouped_batches(list(range(40)), groups, batch_size=4, num_workers=2)
    assert sorted(i for x in b for i in x) == list(range(40))
    # worker w gets batches w, w+2, ...: each group must be touched by one worker only
    owner = {}
    for k, batch in enumerate(b):
        for i in batch:
            assert owner.setdefault(groups[i], k % 2) == k % 2
    lab = torch.arange(200) % 5
    idx = E.balanced_by_groups(lab, torch.arange(200) // 20, per_class=6)
    assert torch.bincount(lab[idx]).tolist() == [6] * 5
    assert len(set((idx // 20).tolist())) <= 2            # 20-row groups hold 4 per class: 2 groups suffice


def _write_imagenet(root, n=24, classes=3, size=72):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import io
    import numpy as np
    from PIL import Image
    (root / "data").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split in ("train", "validation"):
        imgs, labs = [], []
        for i in range(n):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 255, (size, size + 8, 3), dtype=np.uint8)).save(buf, "PNG")
            imgs.append({"bytes": buf.getvalue(), "path": f"{i}.png"})
            labs.append(i % classes)
        t = pa.table({"image": imgs, "label": labs})
        pq.write_table(t, root / "data" / f"{split}-00000-of-00001.parquet", row_group_size=8)


def test_script_end_to_end(tmp_path):
    """The CLI loads a trainer-style checkpoint and runs on a synthetic ImageNet parquet."""
    pytest.importorskip("mavt.dino_v0.data")
    torch.manual_seed(0)
    cfg = tiny_config()
    ck = tmp_path / "ck.pt"
    torch.save({"model": MAVTDinoV0(cfg).state_dict(), "cfg": cfg.__dict__, "step": 7}, ck)
    _write_imagenet(tmp_path / "imagenet")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_dino_v0.py"), "--ckpt", str(ck), "--device", "cpu",
                        "--out", str(out), "--workers", "2", "--batch-size", "4", "--limit", "16",
                        "--imagenet-root", str(tmp_path / "imagenet"), "--knn-train", "4", "--knn-k", "3",
                        "--probe-epochs", "1", "--probe-lrs", "1e-2"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.load(open(out / "eval.json"))
    assert res["meta"]["step"] == 7 and res["budget"]["image"] == cfg.budget("image")
    assert res["image_recon"]["n"] == 16
    assert res["understanding"]["n_train"] == 12 and res["understanding"]["n_val"] == 16
    assert any("ssv2" in n for n in res["notes"])
    assert (out / "eval.md").exists()


def _write_mp4(path, n, size=72, fps=24):
    av = pytest.importorskip("av")
    import numpy as np
    with av.open(str(path), "w") as c:
        st = c.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = size, size, "yuv420p"
        for i in range(n):
            arr = np.full((size, size, 3), (i * 8) % 256, dtype=np.uint8)
            for p in st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def test_script_video_at_stride(tmp_path):
    """--frame-stride sets the fixed stride of the video eval; C/D split and forecast reach eval.json / eval.md."""
    pytest.importorskip("mavt.dino_v0.data")
    pytest.importorskip("av")
    torch.manual_seed(0)
    cfg = tiny_config()
    ck = tmp_path / "ck.pt"
    torch.save({"model": MAVTDinoV0(cfg).state_dict(), "cfg": cfg.__dict__, "step": 3}, ck)
    _write_mp4(tmp_path / "long.mp4", 40)
    _write_mp4(tmp_path / "short.mp4", 20)
    man = tmp_path / "ov.jsonl"
    man.write_text("".join(json.dumps(dict(path=str(tmp_path / f), n_frames=n, fps=24)) + "\n"
                           for f, n in (("long.mp4", 40), ("short.mp4", 20))))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_dino_v0.py"), "--ckpt", str(ck), "--device", "cpu",
                        "--out", str(out), "--workers", "0", "--openvid-manifest", str(man), "--video-tail", "2",
                        "--video-limit", "2", "--video-batch-size", "2", "--video-frames", "8",
                        "--frame-stride", "4", "--stem", "eval_s4"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.load(open(out / "eval_s4.json"))
    v = res["video_recon"]
    assert res["meta"]["frame_stride"] == 4 and v["frame_stride"] == 4 and v["n_clips"] == 2
    assert all(k in v for k in ("psnr_full_key", "psnr_eraseC_nonkey", "delta_cells_p95", "forecast_h1_psnr_copy_gt"))
    assert len(v["per_clip"]) == 2 and all(c["stride"] == 4 for c in v["per_clip"])
    assert any("1/2 clips" in n and "29-frame span" in n for n in res["notes"])     # short.mp4 < 29 frames
    assert any("psnr_warp skipped" in n for n in res["notes"])
    assert "Forecast of frame" in (out / "eval_s4.md").read_text()


@pytest.mark.skipif(importlib.util.find_spec("lpips") is None, reason="lpips not installed")
def test_lpips_available():
    assert E.OptionalLPIPS("cpu").available


@pytest.mark.skipif(importlib.util.find_spec("torchmetrics") is None, reason="torchmetrics not installed")
def test_fid_constructs():
    f = E.OptionalFID("cpu")
    assert f.available or f.reason
