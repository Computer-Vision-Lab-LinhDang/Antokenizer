import argparse
import json

import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.losses import sequence_loss
from mavt4d.v6.model import V6Model
from mavt4d.v6.train import (InfiniteSampler, batch_to_sequence, box_velocity, lr_at, main, make_optimizer,
                             synthetic_batch, teacher_at)

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)


def test_box_velocity_follows_tracks():
    batch = synthetic_batch(V6Config(**TINY), 2, 3, 2, generator=torch.Generator().manual_seed(0))
    boxes, t = batch['boxes'], batch['t']
    for key in ('track', 'center', 'mask'):  # clip 1, step 1: same tracks in another box order
        boxes[key][1, 1] = boxes[key][1, 1].flip(0)
    for s in range(3):
        v, ok = box_velocity(boxes, t, s)
        o = 1 if s == 0 else s - 1
        for b in range(2):
            for m in range(8):
                if not boxes['mask'][b, s, m]:
                    assert not ok[b, m]
                    continue
                j = int((boxes['track'][b, o] == boxes['track'][b, s, m]).float().argmax())
                assert ok[b, m] == boxes['mask'][b, o, j]
                if ok[b, m]:
                    want = (boxes['center'][b, s, m] - boxes['center'][b, o, j]) / float(t[b, s] - t[b, o])
                    assert torch.allclose(v[b, m], want, atol=1e-5)
    _, targets = batch_to_sequence(batch, torch.device('cpu'))
    assert 'box_velocity' in targets[0] and targets[0]['box_velocity_valid'].dtype == torch.bool


def test_teacher_share_zero_never_forces_spawns():
    torch.manual_seed(0)
    cfg = V6Config(**TINY)
    model = V6Model(cfg).train()
    batch = synthetic_batch(cfg, 2, 2, 2, generator=torch.Generator().manual_seed(0))
    steps, targets = batch_to_sequence(batch, torch.device('cpu'))
    _, log = sequence_loss(model, steps, targets, seeds=['a', 'b'], rays_per_camera=8, teacher_p=0.)
    assert log['taught_clips'] == 0. and log.get('forced_spawns', 0.) == 0.
    _, log = sequence_loss(model, steps, targets, seeds=['a', 'b'], rays_per_camera=8, teacher_p=1.)
    assert log['taught_clips'] == 1. and 'slot_vel' in log


def test_schedules_and_backbone_groups():
    args = argparse.Namespace(lr=1., warmup=10, max_steps=110, min_lr_ratio=.1, schedule_from=0,
                              teacher_p=.3, teacher_anneal_steps=0, backbone_lr_mult=.1, weight_decay=.05)
    assert lr_at(0, args) == .1 and lr_at(10, args) == 1.
    args.schedule_from = 50
    assert lr_at(50, args) == .1 and abs(lr_at(110, args) - .1) < 1e-9
    assert teacher_at(0, args) == .3
    args.teacher_anneal_steps = 100
    assert teacher_at(50, args) == 1. and abs(teacher_at(100, args) - .65) < 1e-9 and teacher_at(500, args) == .3
    model = V6Model(V6Config(**TINY))
    opt = make_optimizer(model, args)
    by_mult = {}
    for g in opt.param_groups:
        by_mult.setdefault(g['lr_mult'], 0)
        by_mult[g['lr_mult']] += sum(p.numel() for p in g['params'])
    backbone = sum(p.numel() for n, p in model.named_parameters()
                   if n.startswith(('encoder.blocks.', 'encoder.patch_embed.')))
    assert by_mult[.1] >= backbone and by_mult[1.] > 0
    cls = model.encoder.cls_token
    assert all(g['weight_decay'] == 0. for g in opt.param_groups if any(p is cls for p in g['params']))


def test_infinite_sampler_keeps_going_and_splits_ranks():
    a = iter(InfiniteSampler(5, rank=0, world=2, seed=3))
    b = iter(InfiniteSampler(5, rank=1, world=2, seed=3))
    first = [next(a) for _ in range(2)], [next(b) for _ in range(2)]
    assert not set(first[0]) & set(first[1])
    assert len([next(a) for _ in range(10)]) == 10


def test_finetune_resume_resets_optimizer_on_new_groups(tmp_path):
    base = ['--dry-run', '--batch-size', '1', '--steps', '2', '--cameras', '1', '--rays-per-camera', '8',
            '--device', 'cpu', '--num-workers', '0', '--warmup', '1', '--ckpt-dir', str(tmp_path),
            '--config', json.dumps(TINY)]
    assert main(base + ['--max-steps', '1']) == 1
    assert main(base + ['--max-steps', '2', '--resume', 'auto', '--backbone-lr-mult', '.1', '--schedule-from', '1',
                        '--teacher-p', '.5', '--teacher-anneal-steps', '1']) == 2
    events = [json.loads(line).get('event') for line in (tmp_path / 'log.jsonl').read_text().splitlines()]
    assert 'optimizer_reset' in events
