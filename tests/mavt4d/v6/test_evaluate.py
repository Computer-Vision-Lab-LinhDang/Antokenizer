import json
import math

from mavt4d.v6.evaluate import auc, main
from mavt4d.v6.train import main as train_main

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)


def test_auc_counts_ties_half():
    import torch
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
