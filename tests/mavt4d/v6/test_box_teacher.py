import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import FREE
from mavt4d.v6.losses import advance, step_terms, teacher_boxes
from mavt4d.v6.model import V6Model
from mavt4d.v6.train import batch_to_sequence, synthetic_batch

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)


def _setup(center, size=(200., 200., 200.)):
    torch.manual_seed(0)
    cfg = V6Config(**TINY)
    model = V6Model(cfg).train()
    with torch.no_grad():
        model.trunk.spawn_head.bias.fill_(-10.)   # the learned spawn path never fires
    batch = synthetic_batch(cfg, 2, 2, 2, generator=torch.Generator().manual_seed(0))
    batch['depth_valid'][:] = True  # every token carries a LiDAR tag
    batch['depth'] = 2 + 20 * torch.rand(batch['depth'].shape, generator=torch.Generator().manual_seed(1))
    b, t, m = batch['boxes']['mask'].shape
    batch['boxes']['mask'][:] = False
    batch['boxes']['mask'][:, :, 0] = True
    batch['boxes']['center'][:, :, 0] = torch.tensor(center)
    batch['boxes']['size'][:, :, 0] = torch.tensor(size)
    steps, targets = batch_to_sequence(batch, torch.device('cpu'))
    return cfg, model, steps, targets


def test_box_teacher_spawns_from_evidence_only():
    cfg, model, steps, targets = _setup((0., 0., 0.))
    state = model.initial_state(2, time=steps[0].t)
    observed = torch.ones(2, dtype=torch.bool)
    post, aux = advance(model, state, steps[0], observed, boxes=teacher_boxes(targets[0]))
    meas = aux['meas']
    assert meas.spawn_forced.sum(-1).tolist() == [1, 1]
    assert bool((post.status > FREE).any(-1).all())          # spawned although the spawn head says no
    # measured anchors are means of the box's own evidence points, spread over >= 3 box-grid anchors
    pts = model.trunk.token_points(aux['enc'], steps[0]).points.flatten(1, 2)
    for i in range(2):
        k = int(meas.spawn_forced[i].float().argmax())
        mine = pts[i][meas.forced_slot[i] == k]
        ok = meas.anchor_meas_valid[i, k]
        assert int(ok.sum()) >= 3
        got = meas.anchor_meas[i, k][ok]
        assert bool((got >= mine.amin(0) - 1e-4).all() & (got <= mine.amax(0) + 1e-4).all())
    terms, logs = step_terms(model, post, steps[0], targets[0], observed, aux=aux, rays_per_camera=16)
    assert torch.isfinite(terms['own_box']) and terms['own_box'] > 0
    assert torch.isfinite(terms['assign_box']) and terms['assign_box'] > 0
    assert logs['forced_spawns'] == 1.
    terms['own_box'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.trunk.own_heads.parameters())


def test_box_teacher_off_without_boxes_or_when_claimed():
    cfg, model, steps, targets = _setup((0., 0., 0.))
    state = model.initial_state(2, time=steps[0].t)
    observed = torch.ones(2, dtype=torch.bool)
    post, aux = advance(model, state, steps[0], observed)
    assert aux['meas'].spawn_forced is None
    assert not bool((post.status > FREE).any())
    post, _ = advance(model, state, steps[0], observed, boxes=teacher_boxes(targets[0]))
    # the slot now sits inside the box: the box is claimed and no second slot is forced
    post2, aux = advance(model, post, steps[1], observed, boxes=teacher_boxes(targets[1]))
    assert not bool(aux['meas'].spawn_forced.any())
    # ... but the slot holding it keeps measuring its tokens, so it survives the tentative check
    held = (post.status > FREE).float().argmax(-1)
    tok = aux['meas'].forced_slot
    assert bool((tok >= 0).any()) and bool(((tok == held[:, None]) | (tok < 0)).all())
    assert aux['meas'].spawn_box.gather(1, held[:, None]).squeeze(1).tolist() == [0, 0]
    assert bool((post2.status.gather(1, held[:, None]) > FREE).all())
