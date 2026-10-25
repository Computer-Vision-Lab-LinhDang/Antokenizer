"""scripts/train_cd32.py: the v0 trainer driving the C-D 3.2 model (CPU, tiny config, synthetic data)."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _trainer():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("train_cd32", os.path.join(ROOT, "scripts", "train_cd32.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_one_modality_per_step(tmp_path):
    tr = _trainer()
    out = tmp_path / "run"
    assert tr.main(["--dry-run", "--out", str(out)]) == 0
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert [r["modality"] for r in recs] == ["image", "video", "threed"]
    assert all(np.isfinite(r["loss"]) for r in recs)
    assert "loss_geo_feat" in recs[1] and "diag_xi_rot_deg" in recs[1] and "loss_kl_d" in recs[2]
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 3 and "L_max" not in ck["cfg"] and ck["cfg"]["d_c_video"] == 5
    assert not any(k.startswith("encoder.dino.model.") for k in ck["model"])   # frozen DINO not saved
    assert any(k.startswith("backbone.d_blocks.") for k in ck["model"])


def test_dry_run_mixed_step(tmp_path):
    tr = _trainer()
    out = tmp_path / "mixed"
    assert tr.main(["--dry-run", "--mix-modalities", "--out", str(out)]) == 0
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert all(r["modality"] == "mixed" and np.isfinite(r["loss"]) for r in recs)
    assert {"image/loss", "video/loss", "threed/loss"} <= set(recs[0])


def test_yaml_config_lists_become_tuples(tmp_path):
    tr = _trainer()
    cfg = tr.make_config(dict(dino_repo=None, dino_layers=[1, 2], rope_dims=[16, 24, 24]), dry=False)
    assert cfg.dino_layers == (1, 2) and cfg.rope_dims == (16, 24, 24)
