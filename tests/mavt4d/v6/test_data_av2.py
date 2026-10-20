import numpy as np
import pytest
import torch

pa = pytest.importorskip('pyarrow')
feather = pytest.importorskip('pyarrow.feather')
from PIL import Image, ImageDraw  # noqa: E402

from mavt4d.v6 import V6Config  # noqa: E402
from mavt4d.v6 import data_av2  # noqa: E402
from mavt4d.v6.data_av2 import (CAMERA_KEY, Av2ClipDataset, adjust_intrinsics, choose_cameras,  # noqa: E402
                                collate_clips, crop_params, rng_for, to_observation)

H, W = 48, 64
T0 = 1_600_000_000_000_000_000
LIDAR = [T0 + k * 100_000_000 for k in range(12)]
CAMS_TS = [T0 + j * 50_000_000 for j in range(-2, 23)]
# name: (w, h, fx, fy, cx, cy, R ego<-cam (columns: cam x, y, z in ego), mount t, dot point in ego)
CAMS = {
    'ring_front_center': (240, 320, 200., 205., 110.2, 170.9, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], [1.5, 0., 1.6],
                          [10., 1., 0.5]),
    'ring_side_left': (320, 240, 210., 200., 150.3, 125.7, [[1, 0, 0], [0, 0, 1], [0, -1, 0]], [1.0, 0.8, 1.6],
                       [1., 10., -0.5]),
}
BOX_CITY = np.array([20., 5., 0.5])
BOX_YAW = 0.3


def _yaw_q(y):
    return np.array([np.cos(y / 2), 0, 0, np.sin(y / 2)])


def _mat_q(r):
    r = np.asarray(r, dtype=np.float64)
    w = np.sqrt(max(1e-12, 1 + np.trace(r))) / 2
    if w > 1e-3:
        return np.array([w, r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / [1, 4 * w, 4 * w, 4 * w]
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    v = np.zeros(3)
    v[i] = np.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) / 2
    v[j] = (r[j, i] + r[i, j]) / (4 * v[i])
    v[k] = (r[k, i] + r[i, k]) / (4 * v[i])
    return np.array([(r[k, j] - r[j, k]) / (4 * v[i]), *v])


def _ego_pose(ts):
    """city <- ego: drive along x at 5 m/s while yawing."""
    t = (ts - T0) * 1e-9
    yaw = 0.1 + 0.2 * t
    m = np.eye(4)
    m[:3, :3] = data_av2._qmat(_yaw_q(yaw))
    m[:3, 3] = [2 + 5 * t, 1., 0.]
    return m, yaw


def _cam_R(name):
    return np.array(CAMS[name][6], dtype=np.float64)


def _project_original(name, p_ego):
    w, h, fx, fy, cx, cy, _, t, _ = CAMS[name]
    pc = _cam_R(name).T @ (np.asarray(p_ego) - np.asarray(t))
    return np.array([fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy]), pc[2]


def _table(path, cols):
    feather.write_feather(pa.table(cols), str(path))


def _write_root(root, cam_ts):
    """One AV2-layout log under root/train; cam_ts: camera name -> its frame timestamps."""
    log = root / 'train' / 'log_a'
    (log / 'calibration').mkdir(parents=True)
    (log / 'sensors' / 'lidar').mkdir(parents=True)
    names = list(CAMS)
    _table(log / 'calibration' / 'intrinsics.feather', dict(
        sensor_name=names, fx_px=[CAMS[n][2] for n in names], fy_px=[CAMS[n][3] for n in names],
        cx_px=[CAMS[n][4] for n in names], cy_px=[CAMS[n][5] for n in names], k1=[0.] * 2, k2=[0.] * 2, k3=[0.] * 2,
        height_px=[CAMS[n][1] for n in names], width_px=[CAMS[n][0] for n in names]))
    qs = [_mat_q(_cam_R(n)) for n in names]
    _table(log / 'calibration' / 'egovehicle_SE3_sensor.feather', dict(
        sensor_name=names, **{c: [q[i] for q in qs] for i, c in enumerate(['qw', 'qx', 'qy', 'qz'])},
        **{c: [CAMS[n][7][i] for n in names] for i, c in enumerate(['tx_m', 'ty_m', 'tz_m'])}))
    poses = [_ego_pose(ts) for ts in CAMS_TS]
    _table(log / 'city_SE3_egovehicle.feather', dict(
        timestamp_ns=pa.array(CAMS_TS, pa.int64()),
        **{c: [_yaw_q(y)[i] for _, y in poses] for i, c in enumerate(['qw', 'qx', 'qy', 'qz'])},
        **{c: [m[i, 3] for m, _ in poses] for i, c in enumerate(['tx_m', 'ty_m', 'tz_m'])}))
    for n in names:
        w, h = CAMS[n][:2]
        (log / 'sensors' / 'cameras' / n).mkdir(parents=True)
        uv, _ = _project_original(n, CAMS[n][8])
        img = Image.new('RGB', (w, h))
        ImageDraw.Draw(img).ellipse([uv[0] - 8, uv[1] - 8, uv[0] + 8, uv[1] + 8], fill=(255, 255, 255))
        for ts in cam_ts[n]:
            img.save(log / 'sensors' / 'cameras' / n / f'{ts}.jpg', quality=95)
    rng = np.random.default_rng(0)
    for ts in LIDAR:
        # far dome of points, the two dot points, and a farther point behind the front dot (z-buffer)
        az, r = rng.uniform(-np.pi, np.pi, 3000), rng.uniform(20, 40, 3000)
        pts = np.stack([r * np.cos(az), r * np.sin(az), rng.uniform(-1, 3, 3000)], -1)
        front = np.array(CAMS['ring_front_center'][8])
        mount = np.array(CAMS['ring_front_center'][7])
        pts = np.concatenate([pts, [front, CAMS['ring_side_left'][8], mount + 1.8 * (front - mount)]])
        _table(log / 'sensors' / 'lidar' / f'{ts}.feather', dict(
            x=pts[:, 0].astype(np.float16), y=pts[:, 1].astype(np.float16), z=pts[:, 2].astype(np.float16),
            intensity=np.zeros(len(pts), np.uint8)))
    rows = []
    for k, ts in enumerate(LIDAR):
        m, yaw = _ego_pose(ts)
        inv = np.linalg.inv(m)
        c = inv[:3, :3] @ BOX_CITY + inv[:3, 3]
        rows.append(('static-uuid', 'REGULAR_VEHICLE', c, _yaw_q(BOX_YAW - yaw), 120, ts))
        if k >= 4:
            rows.append(('ped-uuid', 'PEDESTRIAN', np.array([5., -3., 0.]), _yaw_q(0.), 7, ts))
    _table(log / 'annotations.feather', dict(
        timestamp_ns=pa.array([r[5] for r in rows], pa.int64()), track_uuid=[r[0] for r in rows],
        category=pa.array([r[1] for r in rows]).dictionary_encode(),
        length_m=[4.5 if r[1] == 'REGULAR_VEHICLE' else .6 for r in rows], width_m=[2.] * len(rows),
        height_m=[1.6] * len(rows), **{c: [r[3][i] for r in rows] for i, c in enumerate(['qw', 'qx', 'qy', 'qz'])},
        **{c: [r[2][i] for r in rows] for i, c in enumerate(['tx_m', 'ty_m', 'tz_m'])},
        num_interior_pts=pa.array([r[4] for r in rows], pa.uint32())))
    return root


@pytest.fixture(scope='module')
def av2_root(tmp_path_factory):
    return _write_root(tmp_path_factory.mktemp('av2'), {n: CAMS_TS for n in CAMS})


CFG = V6Config(slots=3, trunk_layers=1, encoder_layers=2, image_height=H, image_width=W, max_cameras=4)


def _ds(root, **kw):
    return Av2ClipDataset(root, 'train', CFG, **{**dict(steps=3, step_dt=0.2, tube_dt=0.1), **kw})


def test_shapes_padding_collate(av2_root):
    ds = _ds(av2_root)
    assert len(ds) == 4   # starts at sweeps 1, 3, 5, 7 (sweep 0 has no earlier tube frame)
    x = ds[0]
    T, V = 3, CFG.max_cameras
    assert x['images'].shape == (T, V, 2, 3, H, W) and x['images'].dtype == torch.float32
    assert 0 <= x['images'].min() and x['images'].max() <= 1
    for k in ('depth', 'depth_valid', 'target_depth', 'target_valid'):
        assert x[k].shape == (T, V, H, W)
    assert x['intrinsics'].shape == (T, V, 3, 3) and x['cam_to_rig'].shape == (T, V, 4, 4)
    assert x['camera_valid'].tolist() == [[True, True, False, False]] * T
    keys = sorted([CAMERA_KEY['ring_front_center'], CAMERA_KEY['ring_side_left']])
    assert x['camera_key'].tolist() == [keys + [-1, -1]] * T
    assert x['t'].dtype == torch.float64 and torch.allclose(x['t'], torch.tensor([0., .2, .4], dtype=torch.float64))
    assert torch.allclose(x['tube_dt'], torch.full((T,), .1))
    assert x['rig_pose_valid'].all() and x['has_metric_depth'].all()
    assert torch.allclose(x['rig_pose'][0], torch.eye(4), atol=1e-6)
    b = collate_clips([x, ds[1]])
    assert b['images'].shape == (2, T, V, 2, 3, H, W) and b['boxes']['center'].shape == (2, T, 64, 3)
    obs = to_observation(b, 1)
    assert obs.images.shape == (2, V, 2, 3, H, W) and obs.t.shape == (2,) and obs.rig_pose.shape == (2, 4, 4)
    assert torch.equal(obs.depth, b['depth'][:, 1])


def test_tube_frame_must_sit_near_its_target(av2_root):
    ds = _ds(av2_root)
    ms = 1_000_000
    gap = {'cam_ts': {'c': np.array([T0, T0 + 50 * ms, T0 + 150 * ms, T0 + 200 * ms])}}   # frame at -100 ms dropped
    assert ds._camera_frames(gap, 'c', T0 + 200 * ms) is None
    jitter = {'cam_ts': {'c': np.array([T0, T0 + 105 * ms, T0 + 150 * ms, T0 + 200 * ms])}}
    assert ds._camera_frames(jitter, 'c', T0 + 200 * ms) == [T0 + 105 * ms, T0 + 200 * ms]


def test_clips_start_once_every_camera_has_its_tube(tmp_path):
    late = {**{n: CAMS_TS for n in CAMS}, 'ring_side_left': CAMS_TS[3:]}   # starts 50 ms after sweep 0
    ds = _ds(_write_root(tmp_path, late))
    assert len(ds) == 3   # sweep 1 has no side_left frame 0.1 s back: starts at sweeps 3, 5, 7
    for x in ds:
        assert torch.allclose(x['tube_dt'], torch.full((3,), .1))
        assert x['camera_valid'][:, :2].all()


def _centroid(g):
    g = np.where(g > 0.3, g, 0)
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
    return np.array([(g * xx).sum(), (g * yy).sum()]) / g.sum()


def test_intrinsics_resize_crop(av2_root):
    x = _ds(av2_root)[0]
    for v, name in enumerate(sorted(CAMS, key=CAMERA_KEY.get)):
        w0, h0 = CAMS[name][:2]
        uv0, z = _project_original(name, CAMS[name][8])
        s, x0, y0 = crop_params(w0, h0, H, W)
        expected = (uv0 + 0.5 - [x0, y0]) * s - 0.5
        k = x['intrinsics'][0, v].double().numpy()
        pc = _cam_R(name).T @ (np.asarray(CAMS[name][8]) - CAMS[name][7])
        uv = (k @ pc)[:2] / pc[2]
        assert np.allclose(uv, expected, atol=1e-4)
        assert np.allclose(x['cam_to_rig'][0, v, :3, 3].numpy(), CAMS[name][7], atol=1e-6)
        # the drawn dot moves with the pixels: output centroid = mapped original centroid
        ts = int(x['timestamp_ns'][0])
        orig = np.asarray(Image.open(av2_root / 'train' / 'log_a' / 'sensors' / 'cameras' / name / f'{ts}.jpg'),
                          dtype=np.float64).mean(-1) / 255
        mapped = (_centroid(orig) + 0.5 - [x0, y0]) * s - 0.5
        out = _centroid(x['images'][0, v, 1].mean(0).double().numpy())
        assert np.allclose(out, mapped, atol=0.15), (name, out, mapped)
        assert np.allclose(out, expected, atol=0.6), (name, out, expected)
    # portrait front center: landscape centre crop of a taller resize
    assert crop_params(240, 320, H, W)[2] > 0 and crop_params(240, 320, H, W)[1] == 0
    k2 = adjust_intrinsics(np.diag([100., 100., 1.]), 0.5, 0., 0.)
    assert k2[0, 2] == -0.25 and k2[0, 0] == 50


def test_depth_projection_and_split(av2_root):
    x = _ds(av2_root)[0]
    front = sorted(CAMS, key=CAMERA_KEY.get).index('ring_front_center')
    s, x0, y0 = crop_params(240, 320, H, W)
    uv0, z = _project_original('ring_front_center', CAMS['ring_front_center'][8])
    uv = (uv0 + 0.5 - [x0, y0]) * s - 0.5
    assert np.all(np.abs(uv - np.floor(uv) - 0.5) > 0.05)  # fixture sanity: not on a pixel boundary
    px, py = np.floor(uv + 0.5).astype(int)
    for step in range(3):
        d, dv = x['depth'][step, front], x['depth_valid'][step, front]
        td, tv = x['target_depth'][step, front], x['target_valid'][step, front]
        assert bool(dv[py, px]) != bool(tv[py, px])
        got = d[py, px] if dv[py, px] else td[py, px]
        assert abs(float(got) - z) < 1e-3   # nearest point kept, not the one behind it
    assert not (x['depth_valid'] & x['target_valid']).any()
    assert ((x['depth'] > 0) == x['depth_valid']).all() and ((x['target_depth'] > 0) == x['target_valid']).all()
    n_in, n_out = int(x['depth_valid'].sum()), int(x['target_valid'].sum())
    assert n_in + n_out > 1000 and abs(n_in / (n_in + n_out) - 0.8) < 0.05
    assert not x['depth_valid'][:, 2:].any()


def test_rig_pose_and_boxes(av2_root):
    ds = _ds(av2_root)
    x = ds[1]   # sweeps 3, 5, 7
    ts = x['timestamp_ns'].tolist()
    m0, yaw0 = _ego_pose(ts[0])
    for s, t in enumerate(ts):
        mt, _ = _ego_pose(t)
        assert np.allclose(x['rig_pose'][s].numpy(), np.linalg.inv(m0) @ mt, atol=1e-5)
    b = x['boxes']
    inv0 = np.linalg.inv(m0)
    world_static = inv0[:3, :3] @ BOX_CITY + inv0[:3, 3]
    assert b['mask'].sum(1).tolist() == [1, 2, 2]
    tracks = {}
    for s in range(3):
        for j in np.nonzero(b['mask'][s].numpy())[0]:
            cat = int(b['category'][s, j])
            tracks.setdefault(cat, set()).add(int(b['track'][s, j]))
            if cat == data_av2.CATEGORY_ID['REGULAR_VEHICLE']:
                assert np.allclose(b['center'][s, j].numpy(), world_static, atol=1e-4)
                assert np.allclose(b['quaternion'][s, j].numpy(), _yaw_q(BOX_YAW - yaw0), atol=1e-5)
                assert np.allclose(b['size'][s, j].numpy(), [4.5, 2., 1.6]) and int(b['num_interior_pts'][s, j]) == 120
    assert set(tracks) == {0, 1} and all(len(v) == 1 for v in tracks.values())
    assert sorted(t for v in tracks.values() for t in v) == [0, 1]
    assert (b['track'][~b['mask']] == -1).all()


def test_rng_streams_independent(av2_root, monkeypatch):
    a = rng_for(0, 5, data_av2.PURPOSE_CAMERA_COUNT).random(4)
    assert not np.allclose(a, rng_for(0, 5, data_av2.PURPOSE_DEPTH_SPLIT).random(4))
    assert np.allclose(a, rng_for(0, 5, data_av2.PURPOSE_CAMERA_COUNT).random(4))
    ds = _ds(av2_root, cameras='random', min_multi_cam=0.5, clip_stride=1)
    base = [ds[i] for i in range(len(ds))]

    def by_key(x):
        return {int(k): x['depth_valid'][:, v] for v, k in enumerate(x['camera_key'][0].tolist()) if k >= 0}

    # new depth-split stream: same cameras, same pixels, different split
    monkeypatch.setattr(data_av2, 'PURPOSE_DEPTH_SPLIT', 77)
    changed = False
    for i, x0 in enumerate(base):
        x1 = ds[i]
        assert torch.equal(x1['camera_key'], x0['camera_key'])
        assert torch.equal(x1['depth_valid'] | x1['target_valid'], x0['depth_valid'] | x0['target_valid'])
        changed |= not torch.equal(x1['depth_valid'], x0['depth_valid'])
    assert changed
    monkeypatch.setattr(data_av2, 'PURPOSE_DEPTH_SPLIT', 3)
    # new camera streams: per-camera depth split unchanged
    changed = False
    for purpose in (50, 51, 52):
        monkeypatch.setattr(data_av2, 'PURPOSE_CAMERA_COUNT', purpose)
        monkeypatch.setattr(data_av2, 'PURPOSE_CAMERA_PICK', purpose + 10)
        for i, x0 in enumerate(base):
            x1 = ds[i]
            changed |= not torch.equal(x1['camera_key'], x0['camera_key'])
            k0, k1 = by_key(x0), by_key(x1)
            for key in set(k0) & set(k1):
                assert torch.equal(k0[key], k1[key])
    assert changed


def test_random_camera_counts():
    names = list(data_av2.RING_CAMERAS)
    counts = [len(choose_cameras(names, 'random', 7, 0.4, 3, i)) for i in range(2000)]
    assert min(counts) == 1 and max(counts) == 7
    assert np.mean(np.array(counts) >= 2) >= 0.4
    two = [len(choose_cameras(names[:2], 'random', 7, 0.2, 3, i)) for i in range(2000)]
    assert np.mean(np.array(two) >= 2) >= 0.2
    assert choose_cameras(names, 3, 7, 0.2, 0, 0) == names[:3]
    assert choose_cameras(names, 'all', 4, 0.2, 0, 0) == names[:4]


def _same(a, b, path=''):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            _same(a[key], b[key], f'{path}.{key}')
    elif torch.is_tensor(a):
        assert a.dtype == b.dtype and torch.equal(a, b), path
    else:
        assert a == b, path


def test_packed_split_gives_the_same_clips(av2_root, tmp_path, monkeypatch):
    from mavt4d.v6.data_av2 import PackedAv2ClipDataset, open_av2
    from mavt4d.v6.pack_av2 import pack_split
    kw = dict(steps=3, step_dt=0.2, tube_dt=0.1)
    assert pack_split(av2_root, 'train', tmp_path / 'png', CFG, workers=1, image_ext='png', **kw)['log_a'] > 0
    assert pack_split(av2_root, 'train', tmp_path / 'png', CFG, workers=1, image_ext='png', **kw) == {'log_a': 0}
    raw = _ds(av2_root)
    want = [raw[i] for i in range(len(raw))]
    monkeypatch.setattr(data_av2, '_read', lambda *a, **k: (_ for _ in ()).throw(AssertionError('raw read')))
    packed = open_av2(tmp_path / 'png', 'train', CFG, **kw)
    assert isinstance(packed, PackedAv2ClipDataset) and len(packed) == len(raw) == 4
    for (li, j), (li2, j2) in zip(raw.index, packed.index):
        assert li == li2 and np.array_equal(j, j2)
    for i, x in enumerate(want):
        _same(x, packed[i])
    with pytest.raises(ValueError, match='other settings'):
        open_av2(tmp_path / 'png', 'train', CFG, steps=4, step_dt=0.2, tube_dt=0.1)


def test_packed_jpeg_images_stay_close(av2_root, tmp_path):
    from mavt4d.v6.data_av2 import open_av2
    from mavt4d.v6.pack_av2 import pack_split
    kw = dict(steps=3, step_dt=0.2, tube_dt=0.1)
    pack_split(av2_root, 'train', tmp_path, CFG, workers=2, **kw)
    a, b = _ds(av2_root)[2], open_av2(tmp_path, 'train', CFG, **kw)[2]
    assert torch.equal(a['target_depth'], b['target_depth']) and torch.equal(a['intrinsics'], b['intrinsics'])
    assert (a['images'] - b['images']).abs().mean() < 2 / 255


def test_log_share_is_nested():
    from mavt4d.v6.data_av2 import keep_share
    logs = [f'log_{i:03d}' for i in range(40)]
    quarter, half = keep_share(logs, .25), keep_share(logs, .5)
    assert len(quarter) == 10 and len(half) == 20 and set(quarter) <= set(half)
    assert quarter == sorted(quarter) and keep_share(logs, 1.) == logs
