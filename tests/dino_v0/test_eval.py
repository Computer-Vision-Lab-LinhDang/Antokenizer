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
    def __init__(self, n, T=8, size=64, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, T, size, size, generator=g) * 2 - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"data": self.x[i]}


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


@pytest.mark.skipif(importlib.util.find_spec("lpips") is None, reason="lpips not installed")
def test_lpips_available():
    assert E.OptionalLPIPS("cpu").available


@pytest.mark.skipif(importlib.util.find_spec("torchmetrics") is None, reason="torchmetrics not installed")
def test_fid_constructs():
    f = E.OptionalFID("cpu")
    assert f.available or f.reason
