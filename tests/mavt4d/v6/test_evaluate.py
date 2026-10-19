import json
import math
from types import SimpleNamespace

import torch

from mavt4d.v6.contracts import ACTIVE, FREE
from mavt4d.v6.evaluate import auc, forecast_metrics, handoff_tracks, main, spawn_misses
from mavt4d.v6.train import main as train_main

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)


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
