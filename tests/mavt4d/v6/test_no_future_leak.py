"""No future leak: frames, depth and boxes after a clip's last observed step never reach its state or render.

Each test runs the same clips twice through losses.advance (the path train.py and evaluate.py share), the second
time with images, depth, depth_valid and boxes replaced by noise at every step a clip does not observe, and asks
for bit-identical states and renders. GT boxes go to the box teacher at every step (the worst case:
evaluate.py --box-teacher), and the scene map is on. rig_pose, intrinsics, cam_to_rig and t stay clean on purpose:
ego motion is the known action and t the clock, both given at every step.

Not covered: a loader that puts later frames into an earlier step (data_av2 / pack_av2), since the noise here
replaces whole steps of an already collated batch.
"""
from dataclasses import fields

import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.losses import advance, render_view, teacher_boxes
from mavt4d.v6.model import V6Model
from mavt4d.v6.train import batch_to_sequence, synthetic_batch

CFG = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8,
           scene_samples=8, scene_width=8, scene_cell=4., scene_cells_x=12, scene_cells_y=10,
           scene_x_min=-16., scene_y_min=-20., depth_conf_min=.01)
STEPS = 5


def _model():
    torch.manual_seed(0)
    model = V6Model(V6Config(**CFG)).eval()
    with torch.no_grad():
        model.trunk.spawn_head.bias.fill_(6.)   # spawn at once, so rolled-out steps move allocated slots
    return model


def _batch():
    return synthetic_batch(V6Config(**CFG), 2, STEPS, 2, generator=torch.Generator().manual_seed(4))


def _noise_future(batch, counts):
    """Clip b's images, depth, depth_valid and boxes replaced by noise at every step t >= counts[b]."""
    g = torch.Generator().manual_seed(99)
    future = torch.tensor([[t >= s for t in range(STEPS)] for s in counts])        # [B,T]

    def put(x, noise):
        m = future.view(*future.shape, *(1,) * (x.dim() - 2))
        return torch.where(m, noise.to(x.dtype), x)

    def rand(x):
        return torch.rand(x.shape, generator=g)

    valid = rand(batch['depth_valid']) < .3
    boxes = batch['boxes']
    quat = torch.nn.functional.normalize(torch.randn(boxes['quaternion'].shape, generator=g), dim=-1)
    return {**batch, 'images': put(batch['images'], rand(batch['images'])),
            'depth': put(batch['depth'], torch.where(valid, 1 + 79 * rand(batch['depth']), 0.)),
            'depth_valid': put(batch['depth_valid'], valid),
            'boxes': {**boxes,
                      'center': put(boxes['center'], (rand(boxes['center']) - .5) * 60),
                      'size': put(boxes['size'], .5 + 5 * rand(boxes['size'])),
                      'quaternion': put(boxes['quaternion'], quat),
                      'category': put(boxes['category'], (30 * rand(boxes['category'])).long()),
                      'mask': put(boxes['mask'], rand(boxes['mask']) < .5)}}


@torch.no_grad()
def _run(model, batch, counts, observe_all=False):
    """States and renders (fixed pixels of the input cameras, metadata pose) after every step."""
    steps, targets = batch_to_sequence(batch, torch.device('cpu'))
    b, v = steps[0].camera_valid.shape
    pixels = torch.rand(b, v, 16, 2, generator=torch.Generator().manual_seed(5)) * 63
    state = model.initial_state(b, time=steps[0].t)
    states, renders = [], []
    for t, (obs, tg) in enumerate(zip(steps, targets)):
        observed = torch.tensor([observe_all or t < s for s in counts])
        state, _ = advance(model, state, obs, observed, boxes=teacher_boxes(tg))
        view = render_view(state, obs)
        states.append(state)
        renders.append(model.render(view, model.camera_rays(view, obs.intrinsics, obs.cam_to_rig, pixels)))
    return states, renders


def _diff(a, b, clip):
    """Names of the tensor fields of two state / render dataclasses that differ for one clip."""
    return [f.name for f in fields(a) if torch.is_tensor(getattr(a, f.name))
            and not torch.equal(getattr(a, f.name)[clip], getattr(b, f.name)[clip])]


def _assert_same(run_a, run_b, clip):
    for t in range(STEPS):
        for kind, a, b in (('state', run_a[0][t], run_b[0][t]), ('render', run_a[1][t], run_b[1][t])):
            changed = _diff(a, b, clip)
            assert not changed, f'clip {clip} step {t}: future noise reached the {kind}: {changed}'


def test_rolled_out_steps_ignore_the_future():
    model, batch, counts = _model(), _batch(), [3, 3]
    clean = _run(model, batch, counts)
    noisy = _run(model, _noise_future(batch, counts), counts)
    assert bool((clean[0][counts[0] - 1].status[0] > 0).any()), 'no slot to roll out'
    for clip in range(2):
        _assert_same(clean, noisy, clip)


def test_mixed_batch_observed_and_rolled_out_clip():
    # clip 0 observes the first two steps, clip 1 every step: at t >= 2 advance runs step() on both clips and
    # keeps the prediction for clip 0 with where_state
    model, batch, counts = _model(), _batch(), [2, STEPS]
    clean = _run(model, batch, counts)
    assert bool((clean[0][1].status[0] > 0).any()), 'no slot to roll out'
    noisy = _run(model, _noise_future(batch, counts), counts)
    for clip in range(2):
        _assert_same(clean, noisy, clip)


def test_the_noise_changes_an_observed_step():
    # Without this the two tests above could pass on noise the model never feels.
    model, batch, counts = _model(), _batch(), [3, 3]
    clean = _run(model, batch, counts, observe_all=True)
    noisy = _run(model, _noise_future(batch, counts), counts, observe_all=True)
    for clip in range(2):
        assert _diff(clean[0][3], noisy[0][3], clip), clip
        assert 'rgb' in _diff(clean[1][3], noisy[1][3], clip), clip
