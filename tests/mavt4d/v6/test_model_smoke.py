import json

import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import RayBatch, V6State
from mavt4d.v6.losses import advance, sequence_loss, step_terms
from mavt4d.v6.model import V6Model
from mavt4d.v6.train import batch_to_sequence, main, synthetic_batch

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)  # 8 anchors: enough token mass per anchor to spawn at 64 px


def tiny():
    return V6Config(**TINY)


def _grad_ok(module, nonzero=True):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    finite = bool(grads) and all(torch.isfinite(g).all() for g in grads)
    return finite and (not nonzero or any(g.abs().sum() > 0 for g in grads))


def eager_spawn(model):
    # Fresh trunks rarely spawn; bias the existence head so step 2 sees allocated slots.
    with torch.no_grad():
        model.trunk.spawn_head.bias.fill_(6.)
    return model


def test_two_observed_one_predicted_backward():
    torch.manual_seed(0)
    cfg = tiny()
    model = eager_spawn(V6Model(cfg).train())
    batch = synthetic_batch(cfg, 2, 3, 2, generator=torch.Generator().manual_seed(0))
    steps, targets = batch_to_sequence(batch, torch.device('cpu'))
    state = model.initial_state(2, time=steps[0].t)
    total = 0.
    for t, observed in enumerate((True, True, False)):
        mask = torch.full((2,), observed)
        state, aux = advance(model, state, steps[t], mask)
        state.validate(cfg)
        terms, _ = step_terms(model, state, steps[t], targets[t], mask, aux=aux, rays_per_camera=32)
        assert all(torch.isfinite(v) for v in terms.values()), terms
        total = total + sum(terms.values())
    assert torch.equal(state.t, steps[-1].t)
    assert bool(state.allocated.any())
    total.backward()
    for name in ('encoder', 'trunk', 'decoder'):
        assert _grad_ok(getattr(model, name)), name
    # Writer params (event phi) reach the loss only through the dynamics law, whose output layer starts at 0.
    assert _grad_ok(model.writer, nonzero=False)


def test_predict_keeps_c_and_rays_are_unit():
    cfg = tiny()
    model = V6Model(cfg).eval()
    batch = synthetic_batch(cfg, 1, 2, 2, generator=torch.Generator().manual_seed(1))
    steps, _ = batch_to_sequence(batch, torch.device('cpu'))
    with torch.no_grad():
        state, _ = model.step(model.initial_state(1, time=steps[0].t), steps[0])
        pred = model.predict(state, .5)
        assert torch.equal(pred.c_obj, state.c_obj) and torch.equal(pred.anchor_canon, state.anchor_canon)
        assert torch.allclose(pred.t, state.t + .5)
        pix = torch.rand(1, 2, 5, 2) * 64
        rays = model.camera_rays(state, steps[0].intrinsics, steps[0].cam_to_rig, pix)
        assert isinstance(rays, RayBatch) and rays.origin.shape == (1, 10, 3)
        assert torch.allclose(rays.direction.norm(dim=-1), torch.ones(1, 10), atol=1e-5)
        out = model.render(state, rays)
        assert out.rgb.shape == (1, 10, 3)


def test_mixed_batch_unobserved_clip_is_prediction():
    torch.manual_seed(0)
    cfg = tiny()
    model = V6Model(cfg).eval()
    batch = synthetic_batch(cfg, 2, 2, 2, generator=torch.Generator().manual_seed(3))
    steps, _ = batch_to_sequence(batch, torch.device('cpu'))
    with torch.no_grad():
        state, _ = model.step(model.initial_state(2, time=steps[0].t), steps[0])
        mixed, aux = advance(model, state, steps[1], torch.tensor([True, False]))
        pred = model.predict(state, steps[1].t - state.t)
    assert aux is not None
    for key, value in mixed.select(torch.tensor([1])).tensor_dict().items():
        assert torch.equal(value, getattr(pred.select(torch.tensor([1])), key)), key


def test_sequence_loss_chunked_backward():
    torch.manual_seed(0)
    cfg = tiny()
    model = V6Model(cfg).train()
    batch = synthetic_batch(cfg, 2, 4, 2, generator=torch.Generator().manual_seed(2))
    steps, targets = batch_to_sequence(batch, torch.device('cpu'))
    calls = []
    total, log = sequence_loss(model, steps, targets, seeds=['a', 'b'], rays_per_camera=16, bptt=2,
                               backward=lambda l: (calls.append(1), l.backward()))
    assert len(calls) == 2 and torch.isfinite(total) and 'pred_rgb' in log and 'recon_rgb' in log
    assert _grad_ok(model.decoder)


def test_train_dry_run(tmp_path):
    args = ['--dry-run', '--max-steps', '2', '--batch-size', '1', '--steps', '3', '--cameras', '2',
            '--rays-per-camera', '16', '--device', 'cpu', '--num-workers', '0', '--warmup', '1',
            '--ckpt-dir', str(tmp_path), '--config', json.dumps(TINY), '--checkpoint-layers']
    assert main(args) == 2
    records = [json.loads(line) for line in (tmp_path / 'log.jsonl').read_text().splitlines()]
    assert any(r.get('step') == 2 and r['loss'] == r['loss'] for r in records)
    assert (tmp_path / 'last.pt').exists()
    assert main(args[:2] + ['3'] + args[3:] + ['--resume', 'auto']) == 3


def test_train_bad_backbone_weights_fail_loudly(tmp_path):
    bogus = tmp_path / 'missing-weights'
    bogus.mkdir()
    args = ['--dry-run', '--max-steps', '1', '--batch-size', '1', '--steps', '2', '--cameras', '1',
            '--device', 'cpu', '--ckpt-dir', str(tmp_path / 'run'), '--config', json.dumps(TINY),
            '--backbone', 'siglip2_b16', '--backbone-weights', str(bogus)]
    try:
        main(args)
    except SystemExit as exc:
        assert exc.code not in (0, None)
    else:
        raise AssertionError('expected SystemExit')
    events = [json.loads(line) for line in (tmp_path / 'run' / 'log.jsonl').read_text().splitlines()]
    assert any(e.get('event') == 'backbone_weights' and e['loaded'] == 0 for e in events)


def test_heldout_camera_is_a_target_not_an_input():
    cfg = tiny()
    batch = synthetic_batch(cfg, 2, 2, 3, generator=torch.Generator().manual_seed(3))
    steps, targets = batch_to_sequence(batch, torch.device('cpu'), heldout=True,
                                       generator=torch.Generator().manual_seed(0))
    for obs, tg in zip(steps, targets):
        held = tg['heldout']['camera_valid']
        assert held.sum(1).eq(1).all() and not (held & obs.camera_valid).any()
        assert (obs.camera_valid | held).sum(1).tolist() == batch['camera_valid'][:, 0].sum(1).tolist()
    model = V6Model(cfg).train()
    state = model.initial_state(2, time=steps[0].t)
    state, aux = advance(model, state, steps[0], torch.ones(2, dtype=torch.bool))
    terms, _ = step_terms(model, state, steps[0], targets[0], torch.ones(2, dtype=torch.bool), aux=aux,
                          rays_per_camera=16)
    assert all(torch.isfinite(v) for v in terms.values())
