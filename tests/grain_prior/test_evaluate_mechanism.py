"""3.2 mechanism checks (grain_prior/evaluate_mechanism.py) on the tiny config: keys, consistency with decode(S),
the CLI dry run."""
import json
import math

import pytest
import torch

from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.grain_prior import evaluate_mechanism as EM

from .test_crsr import LEGACY_CRSR
from .test_grain_prior import _inputs, _wake


def _batch(modality, T=3):
    x, cam = _inputs(modality, T=T)
    return dict(data=x, **(cam or {}))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return _wake(MAVTGrainPrior(tiny_config(read_coarse_kv=True, **LEGACY_CRSR))).eval()


@pytest.mark.parametrize("modality", ["video", "threed"])
def test_mechanism_report_keys_and_consistency(model, modality):
    L_max = model.cfg.n_slots
    res = EM.evaluate_mechanism(model, [_batch(modality)], modality, L_list=[0, 2, L_max], k_list=[1, 2])
    curve = {c["stage"]: c for c in res["cd3_curve"]}
    assert list(curve)[:3] == ["C", "C+xi", "C+xi+M"] and f"C+xi+M+r(L={L_max})" in curve
    # the last stage is the state as encoded: decode(S) of the targets
    assert curve[f"C+xi+M+r(L={L_max})"]["psnr_target"] == pytest.approx(res["full"]["psnr_target"], abs=1e-3)
    assert res["cd1"]["psnr_target_D_off"] == pytest.approx(curve["C"]["psnr_target"], abs=1e-6)
    assert sum(res["cd5"][k] for k in ("low", "mid", "high")) == pytest.approx(1.0, abs=1e-4)
    for k in ("psnr_mean_state", "psnr_random_state"):
        assert math.isfinite(res["cd6"][k])
    for name in ("drop_low", "drop_high"):
        assert set(res["at4"][name]) == {"1", "2"}
    assert {"fine_hf_spearman", "coarse_hf_spearman", "sink_hf_spearman"} <= set(res["at2"])
    assert res["at3"]["eff_support_mean"] >= 1.0
    assert res["n_clips"] == 2


def test_mechanism_without_coarse_slots_skips_at2():
    torch.manual_seed(0)
    m = _wake(MAVTGrainPrior(tiny_config())).eval()
    res = EM.evaluate_mechanism(m, [_batch("video")], "video", L_list=[0, m.cfg.n_slots], k_list=[1])
    assert "at2" not in res and "at3" in res


def test_mechanism_cli_dry_run(tmp_path):
    out = tmp_path / "mech"
    assert EM.main(["--dry-run", "--out", str(out), "--L-list", "0,max", "--k-list", "1"]) == 0
    res = json.loads((out / "mechanism.json").read_text())
    assert {"video", "threed", "world"} <= set(res["clips"])
    assert (out / "mechanism.md").read_text().startswith("# Mechanism checks")
