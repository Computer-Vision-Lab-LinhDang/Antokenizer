"""OmniWorld-Game geometry (omniworld.py, scripts/pack_omniworld_dino.py, model losses w_geo / w_depth), CPU.
Targets on synthetic scenes with known answers, the dataset on a tiny pack written here, the losses and the
trainer's 'world' modality. Defaults (w_geo = w_depth = 0, no --omniworld) leave the model and trainer unchanged."""
import argparse
import json
import os
import random

import numpy as np
import pytest
import torch

from mavt.dino_v0 import omniworld as O
from mavt.dino_v0.model import MAVTDinoV0
from .test_nested_d import _v1
from .test_speed_v1 import _script


def _cam(f=100.0, c=31.5):
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])


def _pose(tx=0.0):
    m = np.eye(4)
    m[0, 3] = tx
    return m


# ---------------------------------------------------------------------------- maths
def test_depth_and_flow_decoding():
    z = O.decode_depth(np.array([[0, 65535, 32768]], np.uint16), 2.0)
    d = 32768 / 65535
    assert z[0, 0] == 0 and z[0, 1] == 0 and z[0, 2] == pytest.approx(d / (1000 - d * 999) / 0.004 * 2, rel=1e-5)
    f = O.decode_flow(np.array([[0, 65535]], np.uint16), np.array([[32767.5, 32767.5]]).astype(np.uint16))
    assert f[0, 0, 0] == pytest.approx(-300) and f[0, 1, 0] == pytest.approx(300) and abs(f[0, 0, 1]) < 0.01


def test_cover_crop_and_intrinsics_match_the_pack():
    s, x0, y0 = O.cover_crop(1280, 720, 256, 448)
    assert s == pytest.approx(256 / 720) and x0 == pytest.approx(10.0) and y0 == pytest.approx(0.0)
    K = O.adjust_intrinsics(np.array([[500.0, 0, 639.5], [0, 500.0, 359.5], [0, 0, 1]]), s, x0, y0)
    assert K[0, 2] == pytest.approx((639.5 + 0.5 - 10) * s - 0.5) and K[1, 2] == pytest.approx(127.5)
    assert K[0, 0] == pytest.approx(500 * s)


def test_chain_flow_samples_each_flow_where_the_point_is():
    h, w = 4, 8
    xs, _ = O.aux_centres(h, w)
    f1 = np.zeros((h, w, 2), np.float32)
    f1[..., 0] = 3.0
    f2 = np.zeros((h, w, 2), np.float32)
    f2[..., 0] = 0.5 * xs                                           # displacement grows with x
    x0, y0 = np.array([5.5]), np.array([5.5])
    x, y, ok = O.chain_flow([f1, f2], x0, y0)
    assert ok.all() and x[0] == pytest.approx(5.5 + 3 + 0.5 * 8.5) and y[0] == pytest.approx(5.5)
    _, _, ok = O.chain_flow([f1 * 100], x0, y0)                     # leaves the frame
    assert not ok.any()
    nan = f1.copy()
    nan[:] = np.nan
    assert not O.chain_flow([nan], x0, y0)[2].any()


def test_static_targets_of_a_camera_pan():
    """Plane at z = 2, camera moves +0.1 in x from k to t: content of t sits 5 px to the right in k."""
    size, cell = 64, 16
    depth = np.full((2, 16, 16), 2.0, np.float32)
    dyn = np.zeros((2, 16, 16), np.float32)
    K = np.stack([_cam(), _cam()])
    c2w = np.stack([_pose(0.0), _pose(0.1)])
    tg = O.corr_targets(depth, dyn, K, c2w, {}, [0], size, cell, 0)
    assert tg["corr"].shape == (1, 2, 4, 4, 2) and tg["valid"].all()
    assert np.allclose(tg["corr"][0, 0], 0)                         # keyframe: no shift
    assert np.allclose(tg["corr"][0, 1, ..., 0], 5 / size, atol=1e-5) and np.allclose(tg["corr"][0, 1, ..., 1], 0)


def test_occluded_and_out_of_crop_pixels_give_no_target():
    size, cell = 64, 16
    flat = np.full((2, 16, 16), 2.0, np.float32)
    wall = flat.copy()
    wall[0, :, :8] = 0.5                                            # in k, a near wall covers the left half
    K = np.stack([_cam(), _cam()])
    tg = O.corr_targets(wall, np.zeros_like(wall), K, np.stack([_pose(), _pose(-0.1)]), {}, [0], size, cell, 0)
    v = tg["valid"][0, 1]                                           # content sits 5 px left in k: behind the wall
    assert not v[:, :2].any() and v[:, 3].all()
    tg = O.corr_targets(flat, np.zeros_like(flat), K, np.stack([_pose(), _pose(1.0)]), {}, [0], size, cell, 0)
    v = tg["valid"][0, 1]                                           # 50 px to the right: off the crop
    assert not v[:, 1:].any() and v[:, 0].all()


def test_dynamic_targets_follow_the_chained_flow():
    """An object 2 cells wide moves 8 px per packed step; t is 2 steps after k, the camera stands still."""
    size, cell = 64, 16
    depth = np.full((2, 16, 16), 2.0, np.float32)
    dyn = np.zeros((2, 16, 16), np.float32)
    dyn[0, 4:8, 0:8] = 1.0                                          # cells (1, 0), (1, 1) in k
    dyn[1, 4:8, 4:12] = 1.0                                         # cells (1, 1), (1, 2) in t
    flow = np.zeros((16, 16, 2), np.float32)
    flow[..., 0] = 8.0
    K = np.stack([_cam(), _cam()])
    tg = O.corr_targets(depth, dyn, K, np.stack([_pose(), _pose()]), {(0, 1): [flow, flow]}, [0], size, cell, 0)
    assert tg["dyncell"][1, 1, 1] and tg["dyncell"][1, 1, 2] and not tg["dyncell"][1, 0, 0]
    for j in (1, 2):
        assert tg["valid"][0, 1, 1, j] and np.allclose(tg["corr"][0, 1, 1, j], [-16 / size, 0], atol=1e-5)
    assert tg["valid"][0, 1, 0, 0] and np.allclose(tg["corr"][0, 1, 0, 0], 0)   # static cells: no shift


# ---------------------------------------------------------------------------- dataset on a tiny pack
def _tiny_pack(root, F=12, H=64, W=96):
    from PIL import Image
    d = os.path.join(root, "train")
    os.makedirs(os.path.join(d, "scn", "rgb"))
    json.dump(dict(image_height=H, image_width=W, aux=4, base_stride=2), open(os.path.join(d, "pack.json"), "w"))
    json.dump(dict(uid="scn", segments=[dict(name="seg", frames=F)]), open(os.path.join(d, "scn", "scene.json"), "w"))
    frames = np.arange(F) * 2
    rng = np.random.default_rng(0)
    for f in frames:
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(os.path.join(d, "scn", "rgb", f"{f}.jpg"))
    seg = os.path.join(d, "scn", "seg")
    os.makedirs(seg)
    K = np.tile(np.array([[100.0, 0, 47.5], [0, 100.0, 31.5], [0, 0, 1]], np.float32), (F, 1, 1))
    c2w = np.stack([_pose(0.02 * j) for j in range(F)])
    c2w[:, :3, 3] += [5.0, -3.0, 7.0]                               # world origin far from the cameras
    absrel = np.full(F - 1, 0.001, np.float32)
    absrel[9] = 0.2                                                 # a broken camera pair near the end
    np.savez(os.path.join(seg, "meta.npz"), frames=frames, K=K, c2w=c2w, pair_absrel=absrel,
             pair_epe=np.full(F - 1, 0.2, np.float32))
    np.save(os.path.join(seg, "depth.npy"), np.full((F, H // 4, W // 4), 2.0, np.float16))
    np.save(os.path.join(seg, "dyn.npy"), np.zeros((F, H // 4, W // 4), np.uint8))
    np.save(os.path.join(seg, "flow.npy"), np.zeros((F, H // 4, W // 4, 2), np.float16))
    return root


def test_dataset_clip_layout(tmp_path):
    root = _tiny_pack(str(tmp_path))
    ds = O.OmniWorldClips(root, "train", n_frames=4, frame_stride="2,4", size=64, cell=16, keyframe_every=2,
                          train=False, chunk=2)
    b = ds[0]
    assert b["modality"] == "world" and b["data"].shape == (3, 4, 64, 64) and b["stride"] == 2
    assert torch.allclose(b["c2w"][0], torch.eye(4), atol=1e-6)     # clip-relative camera
    assert b["c2w"][1, 0, 3] == pytest.approx(0.02 / 2.0)           # translation over the median depth
    assert b["K"][0, 0, 2] == pytest.approx(47.5 - 16)              # centre crop offset 16 px
    assert b["geo_depth"].shape == (4, 16, 16) and b["geo_corr"].shape == (2, 4, 4, 4, 2)
    assert b["geo_valid"][0, :2].any() and not b["geo_valid"][1, :2].any()   # key 2 has no targets before it
    for i in range(len(ds)):                                        # the broken pair (9 -> 10) is never in a clip
        si, j0, ps = ds._draw(i, random.Random(i)) or (None, 0, 1)
        assert si is None or not (j0 <= 9 < j0 + 3 * ps)
    train = O.OmniWorldClips(root, "train", n_frames=4, frame_stride="2,4", size=64, cell=16, keyframe_every=2)
    offs = {int(round(47.5 - float(train[0]["K"][0, 0, 2]))) for _ in range(20)}
    assert offs <= {0, 16, 32} and len(offs) > 1                     # random crops on the cell grid


# ---------------------------------------------------------------------------- model losses
def _geo_cfg(**kw):
    return _v1(keyframe_every=2, forecast_horizons=[1, 2], w_geo=1.0, w_depth=0.5, **kw)


def _world_batch(cfg, B=2, T=4, seed=0):
    tr = _script("train_dino_v0")
    return tr.to_inputs(next(tr.synthetic_iter("world", B, cfg, T, seed)), "world", torch.device("cpu"))


def test_geometry_losses_train_the_shift_and_the_depth_head():
    cfg = _geo_cfg()
    torch.manual_seed(0)
    m = MAVTDinoV0(cfg).train()
    x, cam, geo = _world_batch(cfg)
    out = m(x, "world", cam, geo)
    assert {"geo", "geo_fut", "depth"} <= set(out["parts"]) and torch.isfinite(out["loss"])
    assert {"geo_epe_static", "geo_epe_dyn", "geo_valid", "geo_fut_epe_static"} <= set(out["diag"])
    out["loss"].backward()
    assert m.decoder.depth_head.weight.grad.abs().sum() > 0
    assert m.decoder.delta_mlp[-1].weight.grad.abs().sum() > 0


def test_defaults_add_no_head_and_no_geometry_parts():
    cfg = _v1(keyframe_every=2, forecast_horizons=[1, 2])
    torch.manual_seed(0)
    m = MAVTDinoV0(cfg).train()
    assert not hasattr(m.decoder, "depth_head")
    x, cam, geo = _world_batch(cfg)
    out = m(x, "world", cam)                                        # no geo: exactly the posed-sequence step
    assert not {"geo", "geo_fut", "depth"} & set(out["parts"])
    out = m(x, "world", cam, geo)                                   # geo with w_geo = 0: logged, weight 0
    assert "depth" not in out["parts"] and out["parts"]["geo"] >= 0


def test_mixed_step_with_world_uses_every_parameter():
    tr = _script("train_dino_v0")
    cfg = _geo_cfg()
    torch.manual_seed(0)
    model = MAVTDinoV0(cfg).train()
    w = tr.mixed_weights(None, {"image", "video", "world"})
    assert w == {m: pytest.approx(1 / 3) for m in ("image", "video", "world")}
    step = tr.MixedStep(model, w)
    args = argparse.Namespace(n_frames=4, n_views=2, seed=0)
    assert tr.probe_unused(step, cfg, args, torch.device("cpu"), False) == []
    batches = {m: tr.to_inputs(next(tr.synthetic_iter(m, 2, cfg, 4, 0)), m, torch.device("cpu")) for m in w}
    out = step(batches)
    assert torch.isfinite(out["loss"]) and "world/loss_geo" in out["parts"] and "world/diag_geo_valid" in out["diag"]


def test_init_from_tolerates_only_the_new_heads(tmp_path):
    tr = _script("train_dino_v0")
    torch.manual_seed(0)
    old = MAVTDinoV0(_v1(keyframe_every=2)).eval()
    path = str(tmp_path / "old.pt")
    torch.save(dict(model=tr.trainable_state(old), step=7), path)
    new = MAVTDinoV0(_v1(keyframe_every=2, w_depth=0.5)).eval()
    step, missing = tr.load_weights(path, new)
    assert step == 7 and missing and all(k.startswith("decoder.depth_head.") for k in missing)
    assert torch.equal(new.decoder.feat_head.weight, old.decoder.feat_head.weight)
    sd = tr.trainable_state(old)
    sd.pop("decoder.feat_head.weight")
    torch.save(dict(model=sd, step=7), path)
    with pytest.raises(RuntimeError, match="feat_head"):
        tr.load_weights(path, new)
