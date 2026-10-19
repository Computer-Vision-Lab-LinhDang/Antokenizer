"""OmniWorld-Game for v6: pack a fake raw scene layout, read it back in the AV2 batch layout."""
import csv
import io
import json
import math
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image

from mavt4d.eval.fixed_suite import bucket
from mavt4d.v6 import V6Config, data_av2
from mavt4d.v6 import data_omniworld as dow
from mavt4d.v6.data_av2 import adjust_intrinsics, collate_clips, crop_params
from mavt4d.v6.pack_omniworld import main as pack_main
from mavt4d.v6.train import batch_to_sequence

SRC_W, SRC_H = 128, 72
FOCAL, CX, CY = 60., 64., 36.          # OmniWorld K: pixel centres at +0.5 (cx = W / 2)
SPEED, Z0, SKY_ROWS = 2., 10., 6      # unscaled units / s; plane depth at t = 0; rows of sky at the top
FAR_COLS = 8                           # columns whose depth is beyond MAX_DEPTH once scaled
CROP_X = 16                            # 128 x 72 -> 64 x 48 cover-crops 16 source columns on each side
# uid: fps, metric scale, splits (original frame indices), invalid positions per split, frame from which the stored
# camera pose is 5 units off (a broken OmniWorld camera; None: consistent)
SCENES = {
    'aaaa00000001': (24., 3., [list(range(0, 100)), list(range(120, 141))], [[96], []], None),
    'bbbb00000002': (30., 2., [list(range(0, 90))], [[]], None),
    'cccc00000003': (24., 3., [list(range(0, 100))], [[]], 50),
}


def _depth_png(depth_unscaled):
    """Inverse of dow.decode_depth (README section 2.3)."""
    d = 0.004 * dow.FAR * depth_unscaled / (1 + 0.004 * (dow.FAR - dow.NEAR) * depth_unscaled)
    return np.round(d * 65535).astype(np.uint16)


def _png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


def _tar(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w:gz') as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _camera_z(fps, frame):
    return SPEED * frame / fps   # camera moves along its +z (forward), world = first camera


def _write_scene(root, uid, fps, scale, splits, invalid, jump):
    frames = sorted({f for s in splits for f in s})
    color, depth = {}, {}
    for f in frames:
        img = np.zeros((SRC_H, SRC_W, 3), np.uint8)
        img[..., 0] = f % 256
        img[..., 1] = 100
        color[f'color/{f:06d}.png'] = _png_bytes(img)
        raw = _depth_png(np.full((SRC_H, SRC_W), Z0 - _camera_z(fps, f)))
        raw[:SKY_ROWS] = 65535                                                    # sky
        far = slice(SRC_W - CROP_X - FAR_COLS, SRC_W - CROP_X)                    # inside the crop, right edge
        raw[:, far] = _depth_png(np.full((SRC_H, FAR_COLS), 300. / scale))         # 300 m: too far
        depth[f'depth/{f:06d}.png'] = _png_bytes(raw)
    others = {'fps.txt': f'FPS: {fps}\n'.encode(),
              'split_info.json': json.dumps({'scene_name': uid, 'split_num': len(splits), 'split': splits}).encode()}
    for i, (s, inv) in enumerate(zip(splits, invalid)):
        n = len(s)
        trans = [[0., 0., -_camera_z(fps, f) - (5. if jump is not None and f >= jump else 0.)] for f in s]  # w -> c
        cam = {'focals': [FOCAL] * n, 'cx': CX, 'cy': CY, 'quats': [[1e6, 0., 0., 0.]] * n, 'trans': trans,
               'invalid_frame': inv}
        others[f'camera/split_{i}.json'] = json.dumps(cam).encode()
    _tar(root / 'videos' / 'OmniWorld-Game' / uid / f'{uid}_rgb_0000.tar.gz', color)
    _tar(root / 'annotations' / 'OmniWorld-Game' / uid / f'{uid}_depth_0000.tar.gz', depth)
    _tar(root / 'annotations' / 'OmniWorld-Game' / uid / f'{uid}_others.tar.gz', others)
    return dict(UID=uid, FPS=fps, **{'Metric Scale': scale})


@pytest.fixture(scope='module')
def cfg():
    return V6Config(image_height=48, image_width=64)


@pytest.fixture(scope='module')
def raw_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('omniworld')
    rows = [_write_scene(root, uid, *spec) for uid, spec in SCENES.items()]
    (root / 'metadata').mkdir()
    with open(root / 'metadata' / 'omniworld_game_metadata.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['UID', 'FPS', 'Metric Scale'])
        w.writeheader()
        w.writerows(rows)
    return root


@pytest.fixture(scope='module')
def packed(raw_root, tmp_path_factory, cfg):
    out = tmp_path_factory.mktemp('pack')
    pack_main(['--data-root', str(raw_root), '--out', str(out), '--val-frac', '0', '--workers', '0',
               '--image-height', str(cfg.image_height), '--image-width', str(cfg.image_width), '--png'])
    return out


def test_clips_respect_splits_tube_and_invalid_frames(packed, cfg):
    ds = dow.OmniWorldClipDataset(packed, 'train', cfg)
    # scene a (24 fps): anchors 2 + 12 k up to 98 -> 9 anchors; invalid position 96 is the tube frame of the last
    # anchor, so 6-step clips start at anchors 0..2. split_1 (21 frames) is too short. scene b (30 fps): 1 clip.
    assert len(ds) == 4
    logs = sorted(ds[i]['log_id'] for i in range(4))
    assert logs == ['omniworld-game/aaaa00000001/split_0'] * 3 + ['omniworld-game/bbbb00000002/split_0']


def test_clips_with_inconsistent_camera_pairs_are_dropped(packed, cfg):
    ds = dow.OmniWorldClipDataset(packed, 'train', cfg)
    assert not any('cccc00000003' in ds[i]['log_id'] for i in range(len(ds)))
    with np.load(packed / 'train' / 'cccc00000003' / 'split_0.npz') as z:
        rel = dict(zip(z['pair_from'].tolist(), z['pair_absrel'].tolist()))
    assert sorted(rel) == [2, 14, 26, 38, 50, 62, 74, 86]                  # anchor -> next anchor
    assert rel[38] > dow.MAX_PAIR_ABSREL and max(v for a, v in rel.items() if a != 38) < 1e-2
    assert len(dow.OmniWorldClipDataset(packed, 'train', cfg, max_absrel=math.inf)) == len(ds) + 4


def test_sample_layout_matches_av2(packed, cfg):
    ds = dow.OmniWorldClipDataset(packed, 'train', cfg, uint8_images=True)
    x = ds[0]
    T, V, H, W = 6, cfg.max_cameras, cfg.image_height, cfg.image_width
    assert x['images'].shape == (T, V, cfg.tubelet, 3, H, W) and x['images'].dtype == torch.uint8
    assert x['camera_valid'][:, 0].all() and not x['camera_valid'][:, 1:].any()
    assert (x['camera_key'][:, 0] == 0).all()
    assert torch.allclose(x['t'], torch.arange(T, dtype=torch.float64) * .5)
    assert torch.allclose(x['tube_dt'], torch.full((T,), 2 / 24))
    # frame content: anchor 2 + 12 s, its tube frame 12 s
    red = x['images'][:, 0, :, 0, H // 2, W // 2].int()
    want = torch.tensor([[12 * s, 2 + 12 * s] for s in range(T)])
    assert (red - want).abs().max() <= 2
    s, x0, y0 = crop_params(SRC_W, SRC_H, H, W)
    k = adjust_intrinsics([[FOCAL, 0, CX - .5], [0, FOCAL, CY - .5], [0, 0, 1]], s, x0, y0)
    assert torch.allclose(x['intrinsics'][:, 0], torch.tensor(k, dtype=torch.float32).expand(T, 3, 3), atol=1e-4)
    assert x['has_metric_depth'].all() and x['rig_pose_valid'].all()
    assert x['boxes']['center'].shape == (T, 64, 3) and not x['boxes']['mask'].any()


def test_rig_pose_is_forward_motion_in_metres(packed, cfg):
    ds = dow.OmniWorldClipDataset(packed, 'train', cfg)
    x = ds[0]
    scale = SCENES['aaaa00000001'][1]
    c2r = x['cam_to_rig'][0, 0]
    assert torch.allclose(c2r[:3, :3], torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]))
    pose = x['rig_pose']
    assert torch.allclose(pose[:, :3, :3], torch.eye(3).expand(6, 3, 3), atol=1e-5)
    fwd = scale * SPEED * x['t'].float()                 # metres along rig +x
    assert torch.allclose(pose[:, 0, 3], fwd, atol=1e-3) and pose[:, 1:3, 3].abs().max() < 1e-4


def test_metric_depth_split_and_invalid_regions(packed, cfg):
    ds = dow.OmniWorldClipDataset(packed, 'train', cfg)
    x = ds[0]
    scale = SCENES['aaaa00000001'][1]
    inp, tgt = x['depth_valid'][:, 0], x['target_valid'][:, 0]
    assert not (inp & tgt).any()
    valid = inp | tgt
    assert .75 < float(inp.sum() / valid.sum()) < .85
    want = scale * (Z0 - SPEED * (2 + 12 * torch.arange(6)) / 24)        # anchor frames 2 + 12 s
    z = torch.where(inp, x['depth'][:, 0], 0.) + torch.where(tgt, x['target_depth'][:, 0], 0.)
    for s in range(6):
        assert torch.allclose(z[s][valid[s]], want[s].float().expand(int(valid[s].sum())), rtol=2e-3)
    assert not valid[:, :2].any()                                          # sky rows (top of the crop)
    assert not valid[:, :, -2:].any()                                      # beyond MAX_DEPTH
    assert torch.isfinite(x['depth']).all() and x['depth'].max() <= dow.MAX_DEPTH


def test_reprojection_check_is_exact_on_consistent_geometry(packed, cfg):
    ds = dow.OmniWorldClipDataset(packed, 'train', cfg)
    rel, n = dow.reprojection_absrel(ds[0], 0, 1)
    assert n > 100 and rel < 5e-3


def test_batches_go_through_batch_to_sequence(packed, cfg):
    ds = dow.OmniWorldClipDataset(packed, 'train', cfg, uint8_images=True)
    batch = collate_clips([ds[0], ds[3]])
    steps, targets = batch_to_sequence(batch, torch.device('cpu'), heldout=True)
    assert len(steps) == 6 and steps[0].images.shape[:2] == (2, cfg.max_cameras)
    assert 'heldout' not in targets[0]                                     # one camera: nothing to hold out
    assert not targets[0]['box_valid'].any()


def test_scene_split_by_group_hash(raw_root, tmp_path, cfg):
    out = tmp_path / 'pack'
    pack_main(['--data-root', str(raw_root), '--out', str(out), '--val-frac', '0.5', '--workers', '0',
               '--image-height', str(cfg.image_height), '--image-width', str(cfg.image_width)])
    for uid in SCENES:
        split = 'val' if bucket(f'omniworld-game:{uid}') < .5 else 'train'
        assert (out / split / uid / 'scene.json').exists()
        assert not (out / ('train' if split == 'val' else 'val') / uid).exists()
    info = json.loads((out / 'train' / dow.PACK_INFO).read_text())
    assert info['source'] == dow.SOURCE and info['tube_dt'] == .1 and info['step_dt'] == .5


def test_open_clips_dispatch_and_concat(packed, cfg, monkeypatch):
    one = dow.open_clips(str(packed), 'train', cfg)
    assert isinstance(one, dow.OmniWorldClipDataset)
    both = dow.open_clips(f'{packed},{packed}', 'train', cfg)
    assert len(both) == 2 * len(one)
    seen = {}
    monkeypatch.setattr(data_av2, 'open_av2', lambda root, split, cfg, **kw: seen.setdefault('av2', (root, split)))
    dow.open_clips('/no/pack/here', 'val', cfg)
    assert seen['av2'][1] == 'val'


def test_pack_settings_must_match(packed):
    with pytest.raises(ValueError, match='packed for other settings'):
        dow.OmniWorldClipDataset(packed, 'train', V6Config(image_height=48, image_width=64), step_dt=.25)
