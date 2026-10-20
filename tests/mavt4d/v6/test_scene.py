"""Static scene map: bilinear splat/read, the C write rule, predict, the evidence gate, line of sight, --init."""
import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import ACTIVE, FREE, RayBatch, V6State
from mavt4d.v6.decoder import V6Decoder
from mavt4d.v6.losses import sequence_loss
from mavt4d.v6.model import V6Model
from mavt4d.v6.scene import GATE, SPLAT_RADIUS, cell_coords, scene_write, splat_footprint
from mavt4d.v6.train import batch_to_sequence, main, synthetic_batch

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)
SCENE = dict(scene_samples=8, scene_width=8, scene_cell=4., scene_cells_x=12, scene_cells_y=10,
             scene_x_min=-16., scene_y_min=-20.)
WRITES = dict(depth_conf_min=.01)  # fresh encoders report log-z std ~1: below the default confidence floor


def cfg(**extra):
    return V6Config(**{**TINY, **SCENE, **extra})


def test_splat_and_read_share_the_cell_convention():
    c = cfg()
    nx = c.scene_cells_x
    centre = (c.scene_x_min + 2.5 * c.scene_cell, c.scene_y_min + 3.5 * c.scene_cell)   # cell (ix 2, iy 3)
    _, acc_w = splat_footprint(c, torch.tensor([[centre]]), torch.ones(1, 1, 1), torch.ones(1, 1), torch.zeros(1, 1))
    assert acc_w[0].argmax() == 3 * nx + 2
    # the decoder reads a cell's value at its centre (same cell convention as the splat)
    model = V6Model(c).eval()
    state = V6State.initial(c, 1)
    scene = state.scene.clone()
    scene[0, 3, 2] = torch.arange(c.scene_width).half()
    ev = state.scene_evidence.clone()
    ev[0, 3, 2] = .5
    grid = model.decoder.scene.grid(replace(state, scene=scene, scene_evidence=ev))
    pos = cell_coords(c, torch.tensor([[centre]]))
    uv = (2 * pos + 1) / torch.tensor([c.scene_cells_x, c.scene_cells_y]) - 1
    read = F.grid_sample(grid, uv[:, :, None], align_corners=False)[0, :, 0, 0]
    assert torch.allclose(read[:c.scene_width], torch.arange(c.scene_width).float())
    assert torch.isclose(read[-1], torch.tensor(1 - torch.exp(torch.tensor(-GATE * .5)).item()))


def test_footprint_splat_peaks_at_the_point_and_widens_with_range():
    c = cfg(scene_cell=1., scene_cells_x=20, scene_cells_y=20, scene_x_min=-10., scene_y_min=-10.)
    centre = 9 * c.scene_cells_x + 10                                     # world (0.5, -0.5): cell ix 10, iy 9
    one = lambda xy, fp: splat_footprint(c, torch.tensor([[xy]]), torch.ones(1, 1, 1), torch.ones(1, 1),  # noqa: E731
                                         torch.tensor([[fp]]))
    (near_v, near), (_, far) = one((.5, -.5), .1), one((.5, -.5), 4.)
    assert near[0, centre] == 1 and far[0, centre] == 1 and torch.allclose(near_v[..., 0], near)
    assert (far > 1e-3).sum() > (near > 1e-3).sum() and far.sum() > near.sum()   # a far patch spreads wider
    assert (far > 0).sum() <= (2 * SPLAT_RADIUS + 1) ** 2
    assert one((30., 0.), .1)[1].sum() == 0                                # off the grid: dropped


def test_write_rule_keeps_unseen_cells_and_caps_evidence():
    c = cfg()
    g = torch.Generator().manual_seed(0)
    state = V6State.initial(c, 2)
    ny, nx, w = c.scene_shape
    prior = replace(state, scene=torch.randn(2, ny, nx, w, generator=g).half(),
                    scene_evidence=torch.rand(2, ny, nx, generator=g) * c.lambda_max)
    prior = replace(prior, scene_evidence=torch.where(torch.rand(2, ny, nx, generator=g) < .3, 0., prior.scene_evidence))
    lam = torch.where(torch.rand(2, ny, nx, generator=g) < .5, torch.rand(2, ny, nx, generator=g) + .1, 0.)
    prop = torch.randn(2, ny, nx, w, generator=g)
    out = scene_write(c, prior, prop, lam)
    unseen = lam == 0
    assert torch.equal(out['scene'][unseen], prior.scene[unseen])
    assert torch.equal(out['scene_evidence'][unseen], prior.scene_evidence[unseen])
    first = (lam > 0) & (prior.scene_evidence == 0)
    assert first.any() and torch.equal(out['scene'][first], prop.half()[first])   # gain 1 on a fresh cell
    assert (out['scene_evidence'] <= c.lambda_max).all() and out['scene'].dtype == torch.float16


def test_step_writes_scene_and_predict_keeps_it():
    torch.manual_seed(0)
    c = cfg(**WRITES)
    model = V6Model(c).eval()
    steps, _ = batch_to_sequence(synthetic_batch(c, 1, 2, 2, generator=torch.Generator().manual_seed(1)),
                                 torch.device('cpu'))
    with torch.no_grad():
        state, _ = model.step(model.initial_state(1, time=steps[0].t), steps[0])
        state.validate(c)
        assert (state.scene_evidence > 0).any() and (state.scene_evidence == 0).any()
        assert not state.scene[state.scene_evidence == 0].any()
        pred = model.predict(state, .5)
    assert torch.equal(pred.scene, state.scene) and torch.equal(pred.scene_evidence, state.scene_evidence)


def _rays(n=16):
    g = torch.Generator().manual_seed(3)
    d = F.normalize(torch.cat(((torch.rand(n, 2, generator=g) - .5) * .1, torch.ones(n, 1)), -1), dim=-1)
    d = torch.cat((d[: n // 2], d[n // 2:] + torch.tensor([-.6, .5, 0.])))  # half towards slot 0, half to nothing
    return RayBatch(origin=torch.zeros(1, n, 3), direction=F.normalize(d, dim=-1)[None],
                    near_far=torch.tensor([.1, 30.]).expand(1, n, 2).clone())


def _slot_state(c):
    g = torch.Generator().manual_seed(0)
    k, a, w = c.slots, c.anchors, c.latent_width
    s = V6State.initial(c, 1)
    cube = torch.tensor([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    return replace(s, status=torch.tensor([[ACTIVE, FREE, FREE]], dtype=torch.int8),
                   position=torch.tensor([[[0., 0., 10.], [0., 0., 0.], [0., 0., 0.]]]),
                   anchor_canon=cube.expand(1, k, a, 3).clone(), c_obj=torch.randn(1, k, 6, w, generator=g),
                   h=torch.randn(1, k, w, generator=g), cov_pv=.01 * torch.eye(6).expand(1, k, 6, 6).clone(),
                   background=torch.randn(1, c.background_tokens, w, generator=g).half())


def test_unseen_scene_is_transparent():
    on, off = cfg(), V6Config(**TINY)
    torch.manual_seed(0)
    dec_on = V6Decoder(on).eval()
    dec_off = V6Decoder(off).eval()
    missing = dec_off.load_state_dict({k: v for k, v in dec_on.state_dict().items() if not k.startswith('scene.')})
    assert not missing.missing_keys
    state_on, rays = _slot_state(on), _rays()
    state_off = replace(state_on, scene=V6State.initial(off, 1).scene, scene_evidence=V6State.initial(off, 1).scene_evidence)
    a, b = dec_on(state_on, rays), dec_off(state_off, rays)
    for name in ('rgb', 'depth', 'opacity', 'weights'):
        assert torch.allclose(getattr(a, name), getattr(b, name), atol=1e-5), name
    assert torch.equal(a.surface_slot, b.surface_slot)


def test_seen_scene_gives_depth_and_occludes():
    c = cfg()
    torch.manual_seed(0)
    dec = V6Decoder(c).eval()
    with torch.no_grad():  # a uniform 2 / m wall of density wherever the map has evidence
        dec.scene.density.weight.zero_()
        dec.scene.density.bias.fill_(2.)
    state = _slot_state(c)
    state = replace(state, scene_evidence=torch.full_like(state.scene_evidence, 10.))
    out = dec(state, _rays())
    assert (out.depth > 0).all() and (out.depth < 3).all()        # the first metres past scene_near
    assert (out.surface_slot == -1).all() and (out.opacity < 1e-3).all()   # the slot at 10 m is hidden
    assert torch.allclose(out.weights[..., -1], torch.ones(1, 16), atol=1e-3)
    assert torch.allclose(out.sample_w.sum(-1), torch.ones(1, 16), atol=1e-3)            # all mass absorbed


def test_line_of_sight_trains_scene_write_and_read():
    torch.manual_seed(0)
    c = cfg(**WRITES)
    model = V6Model(c).train()
    steps, targets = batch_to_sequence(synthetic_batch(c, 2, 3, 2, generator=torch.Generator().manual_seed(0)),
                                       torch.device('cpu'))
    _, log = sequence_loss(model, steps, targets, seeds=[0, 1], rays_per_camera=32, backward=lambda x: x.backward())
    assert all(torch.isfinite(torch.tensor(v)) for v in log.values()) and log['recon_los'] > 0
    for module in (model.writer.scene, model.decoder.scene):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)


def test_init_loads_old_weights_and_keeps_new_modules(tmp_path):
    base = ['--dry-run', '--batch-size', '1', '--steps', '2', '--cameras', '1', '--rays-per-camera', '8',
            '--device', 'cpu', '--num-workers', '0', '--warmup', '1', '--max-steps', '1']
    assert main(base + ['--ckpt-dir', str(tmp_path / 'old'), '--config', json.dumps(TINY)]) == 1
    assert main(base + ['--ckpt-dir', str(tmp_path / 'new'), '--config', json.dumps({**TINY, **SCENE}),
                        '--init', str(tmp_path / 'old' / 'last.pt')]) == 1
    events = [json.loads(line) for line in (tmp_path / 'new' / 'log.jsonl').read_text().splitlines()]
    init = next(e for e in events if e.get('event') == 'init')
    assert init['missing'] > 0 and all(m.startswith(('writer.scene', 'decoder.scene')) for m in init['new_modules'])
    with pytest.raises(SystemExit):  # a checkpoint with modules the model lacks is the wrong checkpoint
        main(base + ['--ckpt-dir', str(tmp_path / 'back'), '--config', json.dumps(TINY),
                     '--init', str(tmp_path / 'new' / 'last.pt')])


def test_slot_boxes_keep_their_points_out_and_views_use_the_metadata_pose():
    from mavt4d.v6.losses import render_view
    from mavt4d.v6.scene import inside_slots
    c = cfg()
    state = _slot_state(c)                                       # one active 1 m cube at (0, 0, 10)
    pts = torch.tensor([[[0., 0., 10.], [0., .7, 10.], [0., 0., 12.]]])
    assert inside_slots(state, pts).tolist() == [[True, True, False]]   # 0.5 m half extent + 0.25 m pad
    steps, _ = batch_to_sequence(synthetic_batch(c, 1, 2, 1, generator=torch.Generator().manual_seed(2)),
                                 torch.device('cpu'))
    view = render_view(V6State.initial(c, 1), steps[1])
    assert torch.allclose(view.rig_position, steps[1].rig_pose[:, :3, 3])
    assert torch.allclose(view.rig_orientation, torch.tensor([[1., 0., 0., 0.]]), atol=1e-6)


def test_line_of_sight_penalises_mass_in_front_even_when_the_target_is_seen():
    from mavt4d.v6.losses import render_loss
    torch.manual_seed(0)
    c = cfg(scene_cells_x=32, scene_cells_y=32, scene_x_min=-64., scene_y_min=-64.)   # every 2-60 m target on the map
    model = V6Model(c).eval()
    steps, targets = batch_to_sequence(synthetic_batch(c, 1, 1, 1, generator=torch.Generator().manual_seed(4)),
                                       torch.device('cpu'))
    cams = {'images': steps[0].images[:, :, -1], 'intrinsics': steps[0].intrinsics, 'cam_to_rig': steps[0].cam_to_rig,
            'camera_valid': steps[0].camera_valid, 'depth': targets[0]['target_depth'],
            'depth_valid': targets[0]['target_valid']}
    empty = V6State.initial(c, 1)
    terms, _ = render_loss(model, empty, cams, 32)
    assert float(terms['los'][0].detach()) == 0                 # nothing seen, nothing in front: no loss
    with torch.no_grad():                                       # an opaque wall right past scene_near everywhere
        model.decoder.scene.density.weight.zero_()
        model.decoder.scene.density.bias.fill_(5.)
    wall = replace(empty, scene_evidence=torch.full_like(empty.scene_evidence, 10.))
    terms, logs = render_loss(model, wall, cams, 32)
    # targets lie 2-60 m away: all mass is in front (1) and none in the band (1 more)
    assert float(terms['los'][0].detach()) > 1.5 and logs['los_front'] > .9
