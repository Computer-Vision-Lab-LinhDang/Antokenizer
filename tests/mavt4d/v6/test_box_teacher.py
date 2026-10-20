import math

import pytest
import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import FREE
from mavt4d.v6.losses import DEFAULT_WEIGHTS, advance, sequence_loss, step_terms, teacher_boxes
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
    # every LiDAR target point lies in the (huge) box: the opacity term asks for opaque slots
    assert logs['obj_rays'] > 0 and torch.isfinite(terms['recon_occ']) and terms['recon_occ'] > 0
    (terms['own_box'] + terms['recon_occ']).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.trunk.own_heads.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.field.parameters())


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


def test_points_in_boxes_uses_box_yaw():
    from mavt4d.v6.losses import points_in_boxes
    yaw = torch.tensor(math.pi / 4)
    boxes = {'center': torch.tensor([[[10., 0., 0.]]]), 'size': torch.tensor([[[4., 1., 2.]]]),
             'quat': torch.stack((torch.cos(yaw / 2), torch.zeros(()), torch.zeros(()), torch.sin(yaw / 2)))[None, None],
             'valid': torch.tensor([[True]])}
    pts = torch.tensor([[[10 + 1.4, 1.4, 0.], [10 + 1.4, -1.4, 0.], [10., 0., 0.]]])
    got = points_in_boxes(pts, boxes, torch.eye(3)[None], torch.zeros(1, 3))
    assert got.tolist() == [[True, False, True]]


def test_unlabelled_clips_give_no_spawn_or_opacity_negatives():
    # a clip with no box at any step (OmniWorld) would only ever say "nothing to spawn, nothing opaque"
    cfg = V6Config(**TINY)
    weights = {'spawn': 1., 'recon_occ': 1., 'pred_occ': 1.}

    def run(batch, labelled=None):
        torch.manual_seed(0)
        model = V6Model(cfg).train()
        steps, targets = batch_to_sequence(batch, torch.device('cpu'))
        if labelled is not None:
            for tg in targets:
                tg['box_labelled'] = labelled
        torch.manual_seed(1)
        loss, log = sequence_loss(model, steps, targets, seeds=[3, 3], rays_per_camera=16,
                                  weights={k: weights.get(k, 0.) for k in DEFAULT_WEIGHTS})
        return float(loss.detach()), targets[0]['box_labelled']

    batch = synthetic_batch(cfg, 2, 3, 2, generator=torch.Generator().manual_seed(0))
    batch['boxes']['mask'][1] = False
    other = {**batch, 'images': batch['images'].clone()}
    other['images'][1] = torch.rand_like(other['images'][1].float())
    base, labelled = run(batch)
    assert labelled.tolist() == [True, False]
    assert run(other)[0] == pytest.approx(base, abs=1e-6)        # clip 1 does not enter these terms
    forced = torch.tensor([True, True])
    assert run(other, forced)[0] != pytest.approx(run(batch, forced)[0], abs=1e-6)  # before: it did
