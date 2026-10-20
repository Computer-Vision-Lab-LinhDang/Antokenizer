import copy
import itertools
import json

import pytest
import torch

from mavt4d.v6 import data_av2, train
from mavt4d.v6.config import V6Config
from mavt4d.v6.evaluate import pick_clips
from mavt4d.v6.losses import DEFAULT_WEIGHTS
from mavt4d.v6.train import allreduce_grads, batches, clip_seeds, main, parse_args, synthetic_batch

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)
# Terms that are a plain mean over clips here (one rig error per clip, the same patch and free-slot counts per clip).
# The others cannot match across batch sizes: render terms draw rays and samples from the global RNG in batch order,
# and the matched-slot terms pool over all matches of the batch, so clips weigh by their match count.
LINEAR = ('enc_depth', 'spawn', 'rig')


def _clips(batch, sl):
    return {k: _clips(v, sl) if isinstance(v, dict) else v[sl].clone() for k, v in batch.items()}


def _norm(xs):
    return float(sum(x.square().sum() for x in xs)) ** .5


def _train(tmp_path, monkeypatch, feed, *extra):
    """Two optimizer steps on the clips in feed; returns the first step's gradients, the update, step logs and the
    number of batches drawn."""
    used, first = [], {}

    def fake_batches(*_, **__):
        for b in itertools.cycle(feed):
            used.append(b)
            yield copy.deepcopy(b)

    def record(params, world):
        if not first:  # before any optimizer step: both runs start from the same weights
            first.update(params=params, init=[p.detach().clone() for p in params],
                         grads=[torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in params])
        allreduce_grads(params, world)

    monkeypatch.setattr(train, 'batches', fake_batches)
    monkeypatch.setattr(train, 'allreduce_grads', record)
    weights = {k: v if k in LINEAR else 0. for k, v in DEFAULT_WEIGHTS.items()}
    assert main(['--dry-run', '--steps', '2', '--cameras', '1', '--rays-per-camera', '8', '--device', 'cpu',
                 '--warmup', '1', '--max-steps', '2', '--ckpt-dir', str(tmp_path), '--backbone', 'dinov3_vits16',
                 '--config', json.dumps(TINY), '--weights', json.dumps(weights), *extra]) == 2
    logs = [json.loads(line) for line in (tmp_path / 'log.jsonl').read_text().splitlines()]
    update = [p.detach() - p0 for p, p0 in zip(first['params'], first['init'])]
    return first['grads'], update, [r for r in logs if 'grad_norm' in r], len(used)


def test_accum_two_halves_match_one_full_batch(tmp_path, monkeypatch):
    # 2 steps: every clip observes both (no recon / pred split); 1 camera: nothing is held out
    ab = synthetic_batch(V6Config(**TINY), 2, 2, 1, generator=torch.Generator().manual_seed(0))
    halves = [_clips(ab, slice(0, 1)), _clips(ab, slice(1, 2))]
    full_grads, full_update, full_logs, full_used = _train(tmp_path / 'full', monkeypatch, [ab], '--batch-size', '2')
    acc_grads, acc_update, acc_logs, acc_used = _train(tmp_path / 'accum', monkeypatch, halves, '--batch-size', '1',
                                                       '--accum', '2')
    assert (full_used, acc_used) == (2, 4)
    assert [r['step'] for r in acc_logs] == [r['step'] for r in full_logs] == [1, 2]
    for a, f in zip(acc_logs, full_logs):
        assert a['grad_norm'] > 0 and a['lr'] == f['lr']
        for key in ('grad_norm', 'loss', *LINEAR):
            assert a[key] == pytest.approx(f[key], rel=1e-4, abs=1e-7), key
    assert all(torch.allclose(a, f, rtol=1e-3, atol=1e-6) for a, f in zip(acc_grads, full_grads))
    # AdamW divides each entry by its own size, so float noise in near-zero gradient entries reaches the update
    assert _norm([a - f for a, f in zip(acc_update, full_update)]) < 1e-3 * _norm(full_update)


def test_clip_seeds_continue_over_micro_batches():
    ab = synthetic_batch(V6Config(**TINY), 2, 2, 1, generator=torch.Generator().manual_seed(0))
    a, b = _clips(ab, slice(0, 1)), _clips(ab, slice(1, 2))
    assert clip_seeds(ab, 0, 5, 1) == clip_seeds(a, 0, 5, 1) + clip_seeds(b, 0, 5, 1, offset=1)
    for x, i in ((ab, slice(0, 2)), (a, slice(0, 1)), (b, slice(1, 2))):  # real clips: seeded by log and time
        x['log_id'], x['timestamp_ns'] = ['l0', 'l1'][i], torch.tensor([[7], [8]])[i]
    assert clip_seeds(ab, 0, 5, 1) == clip_seeds(a, 0, 5, 1) + clip_seeds(b, 0, 5, 1, offset=1)


class _Clips(torch.utils.data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, i):
        return {'clip': torch.tensor(i)}


def test_overfit_clips_train_on_the_clips_evaluate_picks(monkeypatch):
    monkeypatch.setattr(data_av2, 'open_av2', lambda *_, **__: _Clips())
    cpu = torch.device('cpu')

    def seen(n, rank=0, world=1, draws=12):
        args = parse_args(['--data-root', 'x', '--overfit-clips', str(n), '--batch-size', '2', '--num-workers', '0'])
        data = batches(args, None, rank, world, cpu)
        return {i for _ in range(draws) for i in next(data)['clip'].tolist()}

    assert pick_clips(10, 3) == [0, 4, 9]
    assert seen(3) == {0, 4, 9}
    assert seen(3, 0, 2) | seen(3, 1, 2) <= {0, 4, 9}
    assert seen(0) == seen(20) == set(range(10))
    with pytest.raises(RuntimeError):
        seen(1, 0, 2)
