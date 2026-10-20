import json
import math
from types import SimpleNamespace

import pytest
import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import ACTIVE, FREE
from mavt4d.v6.dense import DenseTeacher, sample_features, teacher_features
from mavt4d.v6.evaluate import (auc, dense_copy_metrics, forecast_metrics, handoff_tracks, main, masked_psnr, seen_by,
                                spawn_misses, warp_last)
from mavt4d.v6.train import main as train_main

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)
DENSE = dict(dense_dim=8, dense_teacher='dinov3_vits16')
CAM_TO_RIG = torch.tensor([[0., 0., 1., 1.5], [-1., 0., 0., 0.], [0., -1., 0., 1.6], [0., 0., 0., 1.]])  # z forward


def test_auc_counts_ties_half():
    score = torch.tensor([[.9, .1, .5, .5], [1., 1., 1., 1.]])
    pos = torch.tensor([[True, False, True, False], [True, False, False, False]])
    got = auc(score, pos, torch.ones_like(pos))
    assert got.tolist() == [.875, .5]


def test_masked_psnr_scores_each_region_and_is_nan_when_empty():
    target = torch.zeros(1, 4, 3)
    rgb = torch.tensor([[[.1] * 3, [.1] * 3, [.01] * 3, [.01] * 3]])
    first = torch.tensor([[True, True, False, False]])
    got = [masked_psnr(rgb, target, m).item() for m in (first, ~first)]
    assert got == pytest.approx([20., 40.], abs=1e-4)
    assert math.isnan(masked_psnr(rgb, target, torch.zeros_like(first)).item())


def test_evaluate_checkpoint_dry_run(tmp_path):
    ckpt = tmp_path / 'run'
    train_main(['--dry-run', '--max-steps', '1', '--batch-size', '1', '--steps', '2', '--cameras', '2',
                '--rays-per-camera', '16', '--device', 'cpu', '--num-workers', '0', '--warmup', '1',
                '--ckpt-dir', str(ckpt), '--config', json.dumps(TINY)])
    out = tmp_path / 'eval'
    summary = main(['--dry-run', '--checkpoint', str(ckpt / 'last.pt'), '--clips', '3', '--batch-size', '2',
                    '--steps', '3', '--observe', '2', '--stride', '4', '--rays-per-camera', '32', '--vis', '1',
                    '--device', 'cpu', '--out', str(out)])
    assert summary['step'] == 1 and summary['clips'] == 3
    groups = summary['groups']
    assert set(groups) == {'step_0', 'step_1', 'step_2', 'observed', 'rolled_out'}
    for key in ('held_psnr', 'held_psnr_const', 'in_psnr', 'held_depth_cov', 'loss_terms', 'rig_t_err', 'boxes'):
        assert math.isfinite(groups['observed'][key]['mean']), key
    assert groups['rolled_out']['records'] == 3
    assert (out / 'summary.md').read_text().startswith('# v6 eval')
    assert len(list((out / 'vis').glob('*.png'))) == 1
    lines = [json.loads(line) for line in open(out / 'records_rank0.jsonl')]
    assert len(lines) == 9 and {r['horizon'] for r in lines} == {0, 1}
    # the copy-last baseline exists on rolled-out steps only; observability on every step
    assert all(math.isnan(r['in_psnr_copy']) == r['observed'] for r in lines)
    assert all(math.isfinite(r['in_l1_copy']) for r in lines if not r['observed'])
    for key in ('held_seen_now', 'held_seen_any'):
        assert 0 <= groups['observed'][key]['mean'] <= 1 and 0 <= groups['rolled_out'][key]['mean'] <= 1, key
    # held-out PSNR split by what the input showed and by GT boxes (NaN when a split has no LiDAR pixel)
    for key in ('held_psnr_seen', 'held_psnr_unseen', 'held_psnr_obj', 'held_psnr_static'):
        assert all(key in r for r in lines), key
    assert any(math.isfinite(r['held_psnr_seen']) or math.isfinite(r['held_psnr_unseen']) for r in lines)


def _boxes(centers, tracks):
    m = len(centers)
    return {'box_center': torch.tensor([centers], dtype=torch.float), 'box_size': torch.full((1, m, 3), 2.),
            'box_quat': torch.tensor([[[1., 0., 0., 0.]] * m]), 'box_class': torch.zeros(1, m, dtype=torch.long),
            'box_valid': torch.ones(1, m, dtype=torch.bool), 'box_track': torch.tensor([tracks])}


def test_forecast_scores_model_and_hold_on_the_same_slots_without_a_gate():
    tg = _boxes([[0., 0., 0.], [10., 0., 0.]], [3, 7])
    # at hand-off slot 0 sits on track 7's box and slot 1 is 3 m from any box
    then = SimpleNamespace(status=torch.tensor([[ACTIVE, ACTIVE]]), position=torch.tensor([[[9.5, 0., 0.], [3., 3., 3.]]]))
    handoff = handoff_tracks(then, _boxes([[0., 0., 0.], [9., 0., 0.]], [3, 7]))
    assert handoff == [{0: 7}]
    now = SimpleNamespace(status=torch.tensor([[ACTIVE, ACTIVE]]), position=torch.tensor([[[5., 0., 0.], [0., 0., 0.]]]),
                          velocity=torch.tensor([[[2., 0., 0.], [0., 0., 0.]]]), t=torch.tensor([1.]))
    last_pos = torch.zeros(1, 2, 3)
    prev = [{7: (torch.tensor([9., 0., 0.]), None, .5)}]
    got = forecast_metrics(now, tg, handoff, last_pos, prev)
    # the model is 5 m off: the old 4 m match would have dropped this slot, leaving hold-last unscored
    assert got == {'fc_err': [5.], 'fc_err_hold': [10.], 'fc_vel_err': [0.], 'fc_vel_zero': [2.]}


def test_spawn_misses_names_the_gate_that_stopped_the_free_slot():
    tg = _boxes([[0., 0., 0.], [50., 0., 0.]], [1, 2])
    state = SimpleNamespace(status=torch.tensor([[FREE, FREE]]), position=torch.zeros(1, 2, 3))
    meas = SimpleNamespace(
        points=torch.tensor([[[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.], [20., 0., 0.]]]),
        ownership=torch.tensor([[[0., .6, .4], [0., .6, .4], [0., .6, .4], [1., 0., 0.]]]),
        spawn_logit=torch.tensor([[-5., 2.]]),
        anchor_meas_valid=torch.tensor([[[True, True, True, False], [True, False, False, False]]]))
    rig = (torch.eye(3)[None], torch.zeros(1, 3))
    got = spawn_misses(torch.tensor([[True, True]]), state, meas, torch.ones(1, 4, dtype=torch.bool), tg, rig,
                       threshold=.5, min_anchors=3)
    assert got['miss_boxes'] == [2.] and got['miss_no_tokens'] == [.5] and got['miss_tokens'] == [3.]
    # slot 1 owns the box's tokens and wants to spawn, but measures only one valid anchor
    assert got['miss_low_p'] == [0.] and got['miss_few_anchors'] == [1.] and got['miss_spawned_off'] == [0.]
    assert math.isclose(got['miss_trash'][0], .4, rel_tol=1e-6) and math.isclose(got['miss_best_mass'][0], 1.8, rel_tol=1e-6)
    assert math.isclose(got['miss_focus'][0], 1., rel_tol=1e-6) and got['miss_anchors'] == [1.]


def _camera(h, w, f, rig, **kw):
    """One camera (V=1) of focal f px at CAM_TO_RIG, rig -> world rig = (R [1,3,3], t [1,3])."""
    k = torch.tensor([[f, 0., (w - 1) / 2], [0., f, (h - 1) / 2], [0., 0., 1.]])
    return {'intrinsics': k[None, None], 'cam_to_rig': CAM_TO_RIG[None, None], 'rig': rig,
            'camera_valid': torch.ones(1, 1, dtype=torch.bool), **kw}


def _rig(yaw=0., t=(0., 0., 0.)):
    c, s = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]]), torch.tensor([t])


def test_warp_without_ego_motion_is_the_copy():
    h, w = 12, 16
    g = torch.Generator().manual_seed(0)
    image = torch.rand(1, 1, 3, h, w, generator=g)
    rig = _rig(.3, (5., 2., 0.))
    depth = 2 + 30 * torch.rand(1, 1, h, w, generator=g)
    cams = _camera(h, w, 10., rig, depth=depth, depth_valid=torch.ones(1, 1, h, w, dtype=torch.bool))
    rgb, ok = warp_last(_camera(h, w, 10., rig, images=image), cams, torch.arange(h * w)[None, None], rig)
    assert ok.all() and torch.allclose(rgb, image.flatten(-2).transpose(-1, -2), atol=1e-5)


def test_warp_moves_a_pixel_by_the_parallax_of_a_sideways_step():
    h, w, f, d = 12, 16, 10., .75
    u, v = torch.arange(w) / (w - 1.), torch.arange(h) / (h - 1.)
    ramp = torch.stack((u.expand(h, w), v[:, None].expand(h, w), torch.zeros(h, w)))[None, None]   # rgb = (u, v, 0)
    depth = torch.zeros(1, 1, h, w)
    depth[0, 0, 5, 8], depth[0, 0, 7, 10] = 5., 10.
    cams = _camera(h, w, f, None, depth=depth, depth_valid=depth > 0)
    idx = torch.tensor([[[5 * w + 8, 7 * w + 10, 0]]])
    # the rig steps d to its left (camera -x), so a point at depth z moves f * d / z pixels left in the last frame
    rgb, ok = warp_last(_camera(h, w, f, _rig(), images=ramp), cams, idx, _rig(t=(0., d, 0.)))
    assert ok.tolist() == [[[True, True, False]]]
    want = torch.tensor([[8 - f * d / 5, 5., 0.], [10 - f * d / 10, 7., 0.]]) / torch.tensor([w - 1., h - 1., 1.])
    assert torch.allclose(rgb[0, 0, :2], want, atol=1e-5)


def test_seen_by_needs_the_point_in_front_of_an_input_camera():
    points = torch.tensor([[[20., 0., 1.6], [-20., 0., 1.6], [5., 20., 1.6]]])   # ahead, behind, far left
    ahead = _camera(12, 16, 10., _rig())
    assert seen_by(points, [ahead], 12, 16).tolist() == [[True, False, False]]
    back = _camera(12, 16, 10., _rig(math.pi))   # the rig turned around: what was behind is now in view
    assert seen_by(points, [ahead, back], 12, 16).tolist() == [[True, True, False]]


def test_dense_copy_and_warp_are_one_on_a_static_frame():
    h, w, f = 64, 64, 40.
    g = torch.Generator().manual_seed(0)
    image = torch.rand(1, 1, 3, h, w, generator=g)
    feats = teacher_features(DenseTeacher(V6Config(**TINY, **DENSE)), image, torch.ones(1, 1, dtype=torch.bool))
    rig = _rig(.3, (5., 2., 0.))
    depth = 2 + 30 * torch.rand(1, 1, h, w, generator=g)
    cams = _camera(h, w, f, rig, images=image, depth=depth, depth_valid=torch.ones(1, 1, h, w, dtype=torch.bool))
    last = _camera(h, w, f, rig, images=image, dense=feats)
    idx = torch.arange(h * w)[None, None]
    got = dense_copy_metrics(last, cams, idx, rig, feats)
    for key in ('dense_cos_copy', 'dense_cos_warp', 'dense_cos_copy_lidar'):
        assert got[key].item() == pytest.approx(1., abs=1e-4), key
    assert math.isnan(got['dense_cos_lidar'])
    # a model that predicts the target scores 1 on the same pixels
    pixels = torch.stack((idx % w, idx // w), -1).float()
    got = dense_copy_metrics(last, cams, idx, rig, feats, sample_features(feats, pixels, h, w).flatten(1, 2))
    assert got['dense_cos_lidar'].item() == pytest.approx(1., abs=1e-5)


def test_dense_warp_follows_the_parallax_the_copy_misses():
    h, w, f, d, z = 12, 16, 10., .75, 5.
    u = (torch.arange(w) / (w - 1.)).expand(h, w)
    ramp = lambda x: torch.stack((2 * x - 1, torch.full_like(x, .25)))[None, None]  # noqa: E731  linear in u, per pixel
    # a static wall at depth z; the rig steps d to its left, so what the target sees at u sat at u - f d / z before
    cams = _camera(h, w, f, _rig(t=(0., d, 0.)), images=torch.zeros(1, 1, 3, h, w), depth=torch.full((1, 1, h, w), z),
                   depth_valid=torch.ones(1, 1, h, w, dtype=torch.bool))
    last = _camera(h, w, f, _rig(), images=torch.zeros(1, 1, 3, h, w), dense=ramp(u))
    idx = torch.tensor([[[r * w + c for r in range(h) for c in range(2, w)]]])   # warped pixels stay on the image
    got = dense_copy_metrics(last, cams, idx, cams['rig'], ramp(u - f * d / z / (w - 1)))
    assert got['dense_cos_warp'].item() == pytest.approx(1., abs=1e-5)
    assert got['dense_cos_copy_lidar'].item() < .95 and got['dense_cos_copy'].item() < .95


def _dense_records(tmp_path, name, *extra):
    main(['--dry-run', '--clips', '2', '--batch-size', '2', '--steps', '3', '--observe', '2', '--stride', '4',
          '--rays-per-camera', '128', '--vis', '0', '--device', 'cpu', '--out', str(tmp_path / name), *extra])
    return [json.loads(line) for line in open(tmp_path / name / 'records_rank0.jsonl')]


def test_evaluate_dense_dry_run(tmp_path):
    lines = _dense_records(tmp_path, 'on', '--config', json.dumps({**TINY, **DENSE}))
    assert len(lines) == 6
    for r in lines:
        assert math.isfinite(r['in_dense_cos']) and math.isfinite(r['held_dense_cos'])
        # copy / warp predict a rolled-out step from the last observed frame's features only
        for key in ('in_dense_cos_copy', 'in_dense_cos_warp', 'in_dense_cos_copy_lidar', 'in_dense_cos_lidar'):
            assert math.isnan(r[key]) == r['observed'], key
        assert r['term_recon_dense' if r['observed'] else 'term_pred_dense'] > 0
    assert json.loads((tmp_path / 'on' / 'summary.json').read_text())['dense_teacher'] == 'dinov3_vits16'
    with pytest.raises(SystemExit):   # the model renders vits16 features
        _dense_records(tmp_path, 'other', '--config', json.dumps({**TINY, **DENSE}), '--dense-teacher', 'dinov3_vitl16')


def test_evaluate_dense_baselines_without_dense_features(tmp_path):
    lines = _dense_records(tmp_path, 'off', '--config', json.dumps(TINY), '--dense-teacher', 'dinov3_vits16')
    assert not any('in_dense_cos' in r or 'held_dense_cos' in r for r in lines)
    for r in lines:
        for key in ('in_dense_cos_copy', 'in_dense_cos_warp', 'in_dense_cos_copy_lidar'):
            assert math.isnan(r[key]) == r['observed'], key
        assert math.isnan(r['in_dense_cos_lidar']) and r['term_recon_dense'] == r['term_pred_dense'] == 0
    assert not any('in_dense_cos_copy' in r for r in _dense_records(tmp_path, 'none', '--config', json.dumps(TINY)))
