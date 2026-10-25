"""Vanilla C-D model on OmniWorld-style posed dynamic video (modality 'world'): the clip loader, the model branch
with known cameras and predicted object flow, and the trainer. CPU, tiny config, a tiny pack written here."""
import json
import os

import numpy as np
import pytest
import torch

from mavt.cd32.vanilla import tiny_cdv_config
from mavt.cd32.vanilla_model import MAVTCDV
from mavt.dino_v0 import omniworld as O
from mavt.dino_v0.data import mixed_loaders

from .test_vanilla import _trainer


def _pose(tx=0.0):
    m = np.eye(4)
    m[0, 3] = tx
    return m


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
    np.savez(os.path.join(seg, "meta.npz"), frames=frames, K=K, c2w=c2w, pair_absrel=np.full(F - 1, 0.001, np.float32),
             pair_epe=np.full(F - 1, 0.2, np.float32))
    np.save(os.path.join(seg, "depth.npy"), np.full((F, H // 4, W // 4), 2.0, np.float16))
    np.save(os.path.join(seg, "dyn.npy"), np.zeros((F, H // 4, W // 4), np.uint8))
    np.save(os.path.join(seg, "flow.npy"), np.zeros((F, H // 4, W // 4, 2), np.float16))
    return root


def test_clips_carry_depth_and_skip_the_geometry_targets(tmp_path):
    root = _tiny_pack(str(tmp_path))
    ds = O.OmniWorldClips(root, "train", n_frames=4, frame_stride="2", size=64, cell=16, keyframe_every=4,
                          train=False, chunk=2, geo_targets=False)
    b = ds[0]
    assert b["modality"] == "world" and b["data"].shape == (3, 4, 64, 64)
    assert b["depth"].shape == (4, 16, 16) and float(b["depth"].max()) == pytest.approx(2.0)
    assert torch.allclose(b["c2w"][0], torch.eye(4), atol=1e-6)
    assert b["c2w"][1, 0, 3] == pytest.approx(0.02 / 2.0)           # translation over the median depth
    assert not any(k.startswith("geo_") for k in b)


def test_mixed_loaders_give_world_batches(tmp_path):
    root = _tiny_pack(str(tmp_path))
    lds = mixed_loaders(omniworld=root, batch_world=1, size=64, n_frames=4, frame_stride="2", num_workers=0,
                        cell=16, keyframe_every=4)
    b = next(iter(lds["world"][0]))
    assert b["modality"] == "world" and b["data"].shape == (1, 3, 4, 64, 64)
    assert b["c2w"].shape == (1, 4, 4, 4) and b["K"].shape == (1, 4, 3, 3) and b["depth"].shape == (1, 4, 16, 16)


def _world(B=2, T=4, tx=0.05):
    c2w = torch.eye(4).repeat(B, T, 1, 1)
    c2w[:, :, 0, 3] = tx * torch.arange(T).float()
    K = torch.tensor([[60.0, 0, 31.5], [0, 60.0, 31.5], [0, 0, 1]]).expand(B, T, 3, 3).clone()
    depth = 1.0 + torch.rand(B, T, 16, 16)
    return torch.rand(B, 3, T, 64, 64) * 2 - 1, dict(c2w=c2w, K=K, depth=depth)


def test_world_step_uses_known_cameras_and_predicted_flow():
    torch.manual_seed(0)
    m = MAVTCDV(tiny_cdv_config())
    x, cam = _world()
    out = m(x, "world", cam, return_outputs=True)
    assert torch.isfinite(out["loss"]) and all(torch.isfinite(v) for v in out["parts"].values())
    assert {"geo_rgb", "flow_l1", "kl_d", "depth_gt"} <= set(out["parts"]) and "rho_gt_corr" in out["diag"]
    out["loss"].backward()
    grads = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert "heads.flow.weight" in grads and "heads.slot_out.weight" in grads
    assert not any(n.startswith(("heads.pose.", "heads.fov.")) for n in grads)   # cameras are given
    assert out["outputs"]["rgb"].shape == x.permute(0, 2, 1, 3, 4).shape
    assert out["diag"]["xi_trans"].item() == pytest.approx(0.05 * 2, rel=1e-4)   # mean over t = 1..3


def test_world_reads_through_the_given_camera():
    torch.manual_seed(0)
    m = MAVTCDV(tiny_cdv_config()).eval()
    x, cam = _world(tx=0.0)
    L = torch.zeros(2, 3, dtype=torch.long)
    with torch.no_grad():
        still = m(x, "world", cam, L=L, return_outputs=True)["outputs"]["rgb"]
        x2, cam2 = _world(tx=0.3)
        moved = m(x, "world", cam2, L=L, return_outputs=True)["outputs"]["rgb"]
    assert torch.allclose(still[:, 0], moved[:, 0], atol=1e-5)      # keyframe unchanged
    assert not torch.allclose(still[:, 2], moved[:, 2], atol=1e-4)  # geometry-only frames move with the camera


def test_world_needs_cameras_and_one_segment():
    m = MAVTCDV(tiny_cdv_config())
    x, cam = _world(T=8)
    with pytest.raises(ValueError):
        m(x, "world", None)
    with pytest.raises(ValueError):
        m(x, "world", cam)


def test_dry_run_mixed_step_includes_world(tmp_path):
    tr = _trainer()
    out = tmp_path / "w"
    assert tr.main(["--dry-run", "--mix-modalities", "--n-frames", "4", "--out", str(out)]) == 0
    recs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert {"world/loss", "world/diag_rho_gt_corr"} <= set(recs[0]) and all(np.isfinite(r["loss"]) for r in recs)
