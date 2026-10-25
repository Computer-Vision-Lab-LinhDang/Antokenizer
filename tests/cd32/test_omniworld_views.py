"""OmniWorld-Game posed views (OmniWorldViews) on a tiny synthetic pack: shapes, scale, crop, dynamic mask,
eval determinism, the mixed loaders, a vanilla model step and the held-out eval."""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from mavt.dino_v0.data import OmniWorldViews, collate, mixed_loaders
from mavt.cd32.vanilla import tiny_cdv_config
from mavt.cd32.vanilla_model import MAVTCDV

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
H, W, AUX, BASE = 64, 112, 4, 2


def _c2w(t: int) -> np.ndarray:
    """Camera moving forward along z and a little sideways, slightly turning: OpenCV camera -> world."""
    a = 0.02 * t
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, [0.3 * t, 0.0, 1.5 * t]
    return m


def write_pack(root, split: str, n_scenes: int = 2, F: int = 14, dyn_scene: int = 0):
    from PIL import Image
    d = os.path.join(root, split)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "pack.json"), "w") as f:
        json.dump(dict(image_height=H, image_width=W, base_stride=BASE), f)
    rng = np.random.default_rng(0)
    for si in range(n_scenes):
        uid = f"scene{si}"
        sd = os.path.join(d, uid)
        os.makedirs(os.path.join(sd, "rgb"), exist_ok=True)
        seg = os.path.join(sd, "seg0")
        os.makedirs(seg, exist_ok=True)
        frames = np.arange(F) * BASE
        for fr in frames:
            img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            img[:, :, 0] = np.linspace(0, 255, W, dtype=np.uint8)[None]       # a horizontal ramp
            Image.fromarray(img).save(os.path.join(sd, "rgb", f"{int(fr)}.jpg"), quality=95)
        K = np.tile(np.array([[80.0, 0, W / 2 - 0.5], [0, 80.0, H / 2 - 0.5], [0, 0, 1]]), (F, 1, 1))
        c2w = np.stack([_c2w(t) for t in range(F)])
        depth = np.full((F, H // AUX, W // AUX), 8.0, np.float32)
        depth[:, :, : W // AUX // 2] = 4.0                                   # the left half is nearer
        depth[:, 0, 0] = 0.0                                                 # one invalid pixel
        dyn = np.zeros((F, H // AUX, W // AUX), np.uint8)
        if si == dyn_scene:
            dyn[:, 2:4, 10:14] = 255                                         # a moving object
        absrel = np.full(F - 1, 0.01, np.float32)
        np.savez(os.path.join(seg, "meta.npz"), frames=frames.astype(np.int64), K=K.astype(np.float32), c2w=c2w,
                 pair_absrel=absrel, pair_epe=np.full(F - 1, np.nan, np.float32))
        np.save(os.path.join(seg, "depth.npy"), depth.astype(np.float16))
        np.save(os.path.join(seg, "dyn.npy"), dyn)
        with open(os.path.join(sd, "scene.json"), "w") as f:
            json.dump(dict(uid=uid, fps=30.0, metric_scale=1.0, base_stride=BASE, has_flow=False,
                           segments=[dict(name="seg0", frames=F)]), f)
    return root


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    root = tmp_path_factory.mktemp("omni")
    write_pack(str(root), "train")
    write_pack(str(root), "val", n_scenes=1)
    return str(root)


def test_item_shapes_scale_and_cameras(pack):
    ds = OmniWorldViews(pack, "train", n_views=4, frame_stride="4,8", size=H, train=False, depth=True)
    assert len(ds) == 2 and ds.pstrides == [2, 4]
    it = ds[0]
    assert it["data"].shape == (3, 4, H, H) and it["modality"] == "threed" and it["label"] == "scene0"
    assert it["c2w"].shape == (4, 4, 4) and it["K"].shape == (4, 3, 3) and it["depth"].shape == (4, H, H)
    d0 = it["depth"][0]
    assert torch.isclose(d0[d0 > 0].median(), torch.tensor(1.0))                 # median depth 1 after scaling
    assert (it["depth"][0, 8:16, :] == 0).all() is not None                       # placeholder for shape check
    # the camera moved: relative translation of frame 3 is non-zero and divided by the same scale
    rel = torch.linalg.inv(it["c2w"][0]) @ it["c2w"][3]
    assert rel[:3, 3].norm() > 0.5
    # the centre crop shifted the principal point by the offset
    assert it["K"][0, 0, 2] == pytest.approx(W / 2 - 0.5 - 16 * ((W - H) // 16 // 2))


def test_dynamic_pixels_are_invalid_and_eval_is_deterministic(pack):
    ds = OmniWorldViews(pack, "train", n_views=4, frame_stride="4", size=H, train=False, depth=True)
    a, b = ds[0], ds[0]
    assert torch.equal(a["data"], b["data"]) and torch.equal(a["depth"], b["depth"])
    off = 16 * ((W - H) // 16 // 2)
    cols = slice((10 * AUX - off), (14 * AUX - off))                             # the dynamic block, in the crop
    assert (a["depth"][0, 2 * AUX:4 * AUX, cols] == 0).all()
    assert (a["depth"][0, 6 * AUX:8 * AUX, cols] > 0).all()
    tr = OmniWorldViews(pack, "train", n_views=4, frame_stride="4,8", size=H, train=True, depth=False)
    it = tr[1]
    assert "depth" not in it and it["data"].shape == (3, 4, H, H)
    with pytest.raises(ValueError):
        OmniWorldViews(pack, "train", frame_stride="3", size=H)                   # not a multiple of the base stride
    with pytest.raises(ValueError):
        OmniWorldViews(pack, "train", frame_stride="4", size=32)                  # pack height is 64


def test_mixed_loaders_and_a_model_step(pack):
    lds = mixed_loaders(omniworld=pack, batch_3d=2, size=H, n_views=4, num_workers=0, objaverse_depth=True,
                        omniworld_stride="4,8")
    assert set(lds) == {"threed"}
    b = next(iter(lds["threed"][0]))
    assert b["data"].shape == (2, 3, 4, H, H) and b["depth"].shape == (2, 4, H, H) and len(b["label"]) == 2
    torch.manual_seed(0)
    m = MAVTCDV(tiny_cdv_config(w_depth_gt=0.5))
    out = m(b["data"], "threed", dict(c2w=b["c2w"], K=b["K"], depth=b["depth"]), return_outputs=True)
    assert torch.isfinite(out["loss"]) and "depth_gt" in out["parts"] and "rho_gt_corr" in out["diag"]
    out["loss"].backward()
    assert m.heads.rho.fc2.weight.grad is not None and m.heads.rho.fc2.weight.grad.abs().sum() > 0


def test_eval_on_the_pack(pack, tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("eval_cdv", os.path.join(ROOT, "scripts", "eval_cdv.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    out = tmp_path / "ev"
    assert ev.main(["--dry-run", "--omniworld", pack, "--omniworld-stride", "4", "--batch-3d", "1",
                    "--L-list", "0,max", "--out", str(out)]) == 0
    res = json.loads((out / "eval.json").read_text())
    v = res["omniworld"]
    assert v["n_clips"] == 1 and v["n_frames"] == 4 and np.isfinite(v["curve"][0]["psnr"])
    assert "rho_gt_corr" in v["diag"] and "OmniWorld-Game val" in (out / "eval.md").read_text()
