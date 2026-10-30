"""Evaluation baselines the video / posed-view numbers must be read against: copy-last (the previous ground-truth
frame), the novel-view protocol (grains + camera only: no target posterior, z = c_0, no innovation) and the
per-record curve. Tiny config, CPU."""
import json

import torch

from mavt.cd32.losses import psnr
from mavt.grain_prior import MAVTGrainPrior, tiny_config
from mavt.grain_prior import schema as sc
from mavt.grain_prior.eval_baselines import copy_last, no_memory, strip_target_posterior

from .test_crsr import LEGACY_CRSR
from .test_grain_prior import _inputs, _script, _wake


def _state(m, modality, T=4):
    x, cam = _inputs(modality, B=2, T=T)
    with torch.no_grad():
        out = m(x, modality, cam, return_outputs=True)
    return x.permute(0, 2, 1, 3, 4), out["outputs"]["S"]


def test_copy_last_is_the_previous_ground_truth_frame():
    frames = torch.arange(2 * 5).float().view(2, 5, 1, 1, 1).expand(-1, -1, 3, 4, 4)
    cl = copy_last(frames, q=3)                                          # targets = frames 2, 3, 4
    assert cl.shape == (2, 3, 3, 4, 4)
    assert torch.equal(cl[:, :, 0, 0, 0], frames[:, 1:4, 0, 0, 0])


def test_strip_target_posterior_removes_innovation_and_lengths():
    cfg = tiny_config(**LEGACY_CRSR)
    m = _wake(MAVTGrainPrior(cfg))
    _, S = _state(m, "video")
    S0 = strip_target_posterior(cfg, S)
    assert all((r["r"] == 0).all() and (r["L"] == 0).all() for r in S0["recs"])
    assert (S0["h"][:, sc.H_LEN:sc.H_LEN + cfg.queue] == 0).all()
    assert all((r["r"] != 0).any() for r in S["recs"])                    # the input state is untouched


def test_no_memory_decodes_every_record_from_c0():
    cfg = tiny_config(**LEGACY_CRSR)
    m = _wake(MAVTGrainPrior(cfg))
    _, S = _state(m, "threed")
    ref = MAVTGrainPrior(tiny_config(**LEGACY_CRSR, memory=False))
    ref.load_state_dict(m.state_dict())
    with torch.no_grad(), no_memory(m):
        zs = m.replay(S)
        d = m.decode(S)["rgb"]
    assert all(torch.allclose(z, S["c0"].float()) for z in zs)
    with torch.no_grad():
        assert torch.allclose(d, ref.decode(S)["rgb"], atol=1e-5)
    assert m.cfg.memory is True                                           # restored


def test_training_diag_reports_copy_last():
    m = _wake(MAVTGrainPrior(tiny_config(**LEGACY_CRSR)))
    x, _ = _inputs("video", B=2, T=4)
    frames = x.permute(0, 2, 1, 3, 4)
    d = m(x, "video")["diag"]
    assert torch.allclose(d["psnr_copy_last"], psnr(frames[:, :-1].flatten(0, 1), frames[:, 1:].flatten(0, 1)))


def test_eval_dry_run_reports_copy_last_novel_view_and_the_per_record_curve(tmp_path):
    out = tmp_path / "ev"
    assert _script("eval_grain_prior").main(["--dry-run", "--L-list", "0,max", "--out", str(out)]) == 0
    res = json.loads((out / "eval.json").read_text())
    for name in ("video", "threed"):
        v = res["clips"][name]
        assert {"psnr", "ssim"} <= set(v["copy_last"])
        assert {"psnr", "ssim"} <= set(v["nvs"])                              # grains + camera only
        assert len(v["target_by_record"]) == tiny_config().queue
    md = (out / "eval.md").read_text()
    assert "copy last" in md and "novel view" in md


def test_eval_reports_the_two_learning_gates(tmp_path):
    """Gate 1 (video): L = 0 minus copy-last, in dB. Gate 2 (posed views): novel view minus copy-anchor, in dB."""
    out = tmp_path / "ev"
    assert _script("eval_grain_prior").main(["--dry-run", "--L-list", "0,max", "--out", str(out)]) == 0
    res = json.loads((out / "eval.json").read_text())
    v, t = res["clips"]["video"], res["clips"]["threed"]
    L0 = next(c for c in v["curve"] if c["L"] == 0)
    assert abs(v["gates"]["L0_minus_copy_last_db"] - (L0["psnr"] - v["copy_last"]["psnr"])) < 1e-6
    assert abs(t["gates"]["nvs_minus_copy_db"] - (t["nvs"]["psnr"] - t["copy"]["psnr"])) < 1e-6
    md = (out / "eval.md").read_text()
    assert "gates:" in md and "L0 - copy-last" in md and "novel view - copy" in md
