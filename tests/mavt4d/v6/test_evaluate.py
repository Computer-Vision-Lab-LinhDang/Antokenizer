import json
import math
from types import SimpleNamespace

import torch

from mavt4d.v6.contracts import ACTIVE, FREE
from mavt4d.v6.evaluate import (auc, forecast_metrics, handoff_tracks, main, recall_bins, seen_by, spawn_misses,
                                 summarize, warp_last)
from mavt4d.v6.train import main as train_main

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)
CAM_TO_RIG = torch.tensor([[0., 0., 1., 1.5], [-1., 0., 0., 0.], [0., -1., 0., 1.6], [0., 0., 0., 1.]])  # z forward


def test_auc_counts_ties_half():
    score = torch.tensor([[.9, .1, .5, .5], [1., 1., 1., 1.]])
    pos = torch.tensor([[True, False, True, False], [True, False, False, False]])
    got = auc(score, pos, torch.ones_like(pos))
    assert got.tolist() == [.875, .5]


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
    # pooled 2 m recall by box range (every step) and by LiDAR-tagged tokens (observed steps)
    pooled = groups['observed'].get('pooled', {})
    assert {'recall_2m', 'recall_2m_d0_10', 'recall_2m_t0_n', 'class_pairs'} <= set(pooled), pooled
    assert 'recall_2m_t0_n' not in groups['rolled_out'].get('pooled', {})
    # the copy-last baseline exists on rolled-out steps only; observability on every step
    assert all(math.isnan(r['in_psnr_copy']) == r['observed'] for r in lines)
    assert all(math.isfinite(r['in_l1_copy']) for r in lines if not r['observed'])
    for key in ('held_seen_now', 'held_seen_any'):
        assert 0 <= groups['observed'][key]['mean'] <= 1 and 0 <= groups['rolled_out'][key]['mean'] <= 1, key


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


def test_recall_bins_pool_hits_by_range_and_lidar_tokens():
    # 2 m boxes 5, 15, 30 and 50 m ahead; allocated slots on the first and third, a free one on the second
    tg = _boxes([[5., 0., 0.], [15., 0., 0.], [30., 0., 0.], [50., 0., 0.]], [0, 1, 2, 3])
    state = SimpleNamespace(status=torch.tensor([[ACTIVE, ACTIVE, FREE]]),
                            position=torch.tensor([[[5.5, 0., 0.], [30., 1., 0.], [15., 0., 0.]]]))
    # LiDAR tags: 4 in the first box, 1 in the second (inside the teacher's 1.1 x half + .25 m box only),
    # an invalid one in the last
    points = torch.tensor([[[5., 0., 0.]] * 4 + [[16.3, 0., 0.], [50., 0., 0.]]])
    valid = torch.tensor([[True] * 5 + [False]])
    rig = (torch.eye(3)[None], torch.zeros(1, 3))
    got = recall_bins(state, tg, rig, (points, valid))
    want = {'r2m_n': 4, 'r2m_hit': 2,
            'r2m_d0_10_n': 1, 'r2m_d0_10_hit': 1, 'r2m_d10_20_n': 1, 'r2m_d10_20_hit': 0,
            'r2m_d20_40_n': 1, 'r2m_d20_40_hit': 1, 'r2m_d40p_n': 1, 'r2m_d40p_hit': 0,
            'r2m_t0_n': 2, 'r2m_t0_hit': 1, 'r2m_t1_2_n': 1, 'r2m_t1_2_hit': 0,
            'r2m_t3_5_n': 1, 'r2m_t3_5_hit': 1, 'r2m_t6p_n': 0, 'r2m_t6p_hit': 0}
    assert {k: v[0] for k, v in got.items()} == want
    # range is horizontal, from the rig at its metadata pose: with the rig 20 m ahead and 9 m up the boxes are
    # 15, 5, 10 and 30 m away (in 3D the 5 m one would be 10.3 m)
    moved = recall_bins(state, tg, (torch.eye(3)[None], torch.tensor([[20., 0., 9.]])))
    assert [moved[f'r2m_{b}_n'][0] for b in ('d0_10', 'd10_20', 'd20_40', 'd40p')] == [1, 2, 1, 0]
    assert 'r2m_t0_n' not in moved


def test_summary_pools_recall_and_class_majority():
    pool = [{'r2m_n': 3, 'r2m_hit': 3, 'cls': [[0, 1], [0, 1], [2, 0]]}, {'r2m_n': 1, 'r2m_hit': 0, 'cls': [[0, 0]]}]
    records = [{'clip': i, 'step': 0, 'observed': True, 'recall_2m': p['r2m_hit'] / p['r2m_n'], 'pool': p}
               for i, p in enumerate(pool)]
    group = summarize(records, {})['groups']['observed']
    assert group['recall_2m']['mean'] == .5                      # mean of per-record ratios
    pooled = group.get('pooled', {})
    assert pooled.get('recall_2m') == .75 and pooled.get('recall_2m_n') == 4
    assert pooled.get('class_acc') == .5 and pooled.get('class_pairs') == 4
    assert pooled.get('class_majority') == 0 and pooled.get('class_majority_share') == .75
