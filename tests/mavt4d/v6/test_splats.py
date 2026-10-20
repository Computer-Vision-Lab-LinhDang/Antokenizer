"""Pixel-aligned Gaussians (v7a): config, rasterizer, write prior, ring, rendering from a moved camera, training."""
import json
import math
from dataclasses import replace

import pytest
import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import V6State
from mavt4d.v6.dense import DenseTeacher
from mavt4d.v6.evaluate import main as evaluate_main
from mavt4d.v6.losses import advance, step_terms, where_state
from mavt4d.v6.model import V6Model
from mavt4d.v6.splats import (ALPHA_MAX, ALPHA_MIN, NEAR, camera_to_world, composite_gsplat, composite_torch, project,
                              subcell_colours, subcell_pixels)
from mavt4d.v6.train import batch_to_sequence, init_weights, main as train_main, synthetic_batch

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)
SPLAT = dict(TINY, gaussians=4, gaussian_steps=2)
DENSE = dict(dense_dim=8, dense_teacher='dinov3_vits16')
SCENE = dict(scene_samples=4, scene_width=8, scene_cell=4., scene_cells_x=12, scene_cells_y=10,
             scene_x_min=-16., scene_y_min=-20.)
K = torch.tensor([[50., 0., 32.], [0., 50., 32.], [0., 0., 1.]])
GS = ('gs_mean', 'gs_log_scale', 'gs_quat', 'gs_opacity', 'gs_color', 'gs_valid', 'gs_count')


def _gaussians(centres, colours, log_scale=-1.6, opacity=8.):
    n = len(centres)
    return (torch.tensor(centres), torch.full((n, 3), log_scale), torch.tensor([[1., 0., 0., 0.]] * n),
            torch.full((n,), opacity), torch.tensor(colours), torch.ones(n, dtype=torch.bool))


def test_config_and_state_fields():
    assert V6Config(**TINY).gaussian_shape == (1, 1)
    cfg = V6Config(**SPLAT, **DENSE)
    assert cfg.gaussian_shape == (2 * 7 * 16 * 16, 3 + 8)
    state = V6State.initial(cfg, 2)
    assert state.gs_mean.shape == (2, 3584, 3) and state.gs_color.shape == (2, 3584, 11) and not state.gs_valid.any()
    state.validate(cfg)
    for bad in (dict(gaussians=3), dict(gaussians=4, scene_samples=8), dict(gaussians=4, gaussian_steps=0),
                dict(gaussian_backend='cuda')):
        with pytest.raises(ValueError):
            V6Config(**{**TINY, **bad})


def test_reference_draws_a_gaussian_where_it_projects():
    g = _gaussians([[0., 0., 5.]], [[.2, .4, .6]])
    pixels = torch.tensor([[32., 32.], [0., 0.]])
    rgb, alpha, z, _ = composite_torch(*g, torch.eye(4), K, pixels, 0)
    assert alpha[0] == pytest.approx(ALPHA_MAX, abs=1e-3) and alpha[1] == 0
    torch.testing.assert_close(rgb[0], ALPHA_MAX * torch.tensor([.2, .4, .6]), atol=1e-3, rtol=0)
    assert z[0] == pytest.approx(5.) and z[1] == 0
    centre, depth, _, radius = project(*g[:3], torch.eye(4), K)
    torch.testing.assert_close(centre, torch.tensor([[32., 32.]]))
    assert depth[0] == 5. and 0 < radius[0] < 32


def test_front_gaussian_wins_whatever_the_order():
    g = _gaussians([[0., 0., 4.], [0., 0., 8.]], [[1., 0., 0.], [0., 1., 0.]])
    pixels = torch.tensor([[32., 32.]])
    rgb, _, z, _ = composite_torch(*g, torch.eye(4), K, pixels, 0)
    flipped = tuple(x.flip(0) for x in g)
    rgb2, _, z2, _ = composite_torch(*flipped, torch.eye(4), K, pixels, 0)
    torch.testing.assert_close(rgb, rgb2)
    torch.testing.assert_close(z, z2)
    assert rgb[0, 0] > .98 and rgb[0, 1] < .01 and z[0] < 4.05


def test_reference_matches_compositing_pixel_by_pixel():
    torch.manual_seed(1)
    n = 300
    mean = torch.randn(n, 3) * torch.tensor([3., 3., 2.]) + torch.tensor([0., 0., 9.])
    g = (mean, torch.full((n, 3), -1.2) + .4 * torch.randn(n, 3), torch.nn.functional.normalize(torch.randn(n, 4), dim=-1),
         torch.randn(n), torch.rand(n, 5), torch.rand(n) < .9)
    pixels = torch.rand(40, 2) * 64
    rgb, alpha, z, feat = composite_torch(*g, torch.eye(4), K, pixels, 2)
    keep = g[5] & (mean[:, 2] > NEAR)
    centre, depth, conic, radius = project(mean[keep], g[1][keep], g[2][keep], torch.eye(4), K)
    order = depth.argsort()
    for i, px in enumerate(pixels):
        t, want = 1., torch.zeros(5)
        for j in order:
            d = px - centre[j]
            if (d.abs() > radius[j]).any():
                continue
            power = -.5 * d @ conic[j] @ d
            a = min(torch.sigmoid(g[3][keep][j]) * torch.exp(power), ALPHA_MAX)
            if a < ALPHA_MIN:
                continue
            want += t * a * g[4][keep][j]
            t *= 1 - a
        torch.testing.assert_close(torch.cat((rgb[i], feat[i])), want, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(alpha[i], torch.as_tensor(1 - t, dtype=torch.float32), atol=1e-5, rtol=1e-4)


def _observed(cfg, steps=1, seed=0, batch=None):
    torch.manual_seed(seed)
    model = V6Model(cfg).eval()
    batch = batch or synthetic_batch(cfg, 2, max(steps, 2), 2, generator=torch.Generator().manual_seed(seed))
    seq, targets = batch_to_sequence(batch, torch.device('cpu'))
    state = model.initial_state(2, time=seq[0].t)
    states = []
    with torch.no_grad():
        for t in range(steps):
            state, aux = model.step(state, seq[t])
            states.append(state)
    return model, seq, targets, states, aux


def test_a_fresh_head_writes_the_geometric_prior():
    cfg = V6Config(**SPLAT)
    model, seq, _, (state,), aux = _observed(cfg)
    obs, enc = seq[0], aux['enc']
    v, n, g2 = obs.images.shape[1], cfg.patches_per_camera, cfg.gaussians ** 2
    per_cam = n * g2
    hyp = enc.depth_hyp
    z_tok = hyp[..., 2].gather(-1, hyp[..., 4].argmax(-1, keepdim=True))[..., 0].exp()           # [B,V,N]
    c2w = camera_to_world(state, obs)
    for b in range(2):
        for cam in range(v):
            sl = slice(cam * per_cam, (cam + 1) * per_cam)
            w2c = torch.linalg.inv(c2w[b, cam])
            centre, z, _, _ = project(state.gs_mean[b, sl], state.gs_log_scale[b, sl], state.gs_quat[b, sl], w2c,
                                      obs.intrinsics[b, cam])
            torch.testing.assert_close(centre, subcell_pixels(cfg, centre.device).reshape(-1, 2), atol=1e-3, rtol=0)
            torch.testing.assert_close(z, z_tok[b, cam, :, None].expand(n, g2).reshape(-1), atol=1e-4, rtol=1e-4)
    colour = subcell_colours(obs.images[:, :, -1], cfg).clamp(.01, .99).reshape(2, -1, 3)
    torch.testing.assert_close(state.gs_color[:, :v * per_cam], colour, atol=1e-5, rtol=0)
    seen = (enc.valid & obs.camera_valid[..., None])[..., None].expand(2, v, n, g2).reshape(2, -1)
    assert seen.any() and torch.equal(state.gs_valid[:, :v * per_cam], seen) and not state.gs_valid[:, v * per_cam:].any()
    assert int(state.gs_count[0]) == 1


def test_the_ring_keeps_the_last_steps_and_predict_keeps_it_bit_identical():
    cfg = V6Config(**SPLAT)
    model, seq, _, states, _ = _observed(cfg, steps=3)
    per_step = cfg.gaussian_shape[0] // cfg.gaussian_steps
    first, second, third = states
    assert [int(s.gs_count[0]) for s in states] == [1, 2, 3]
    assert torch.equal(third.gs_mean[:, per_step:], second.gs_mean[:, per_step:])        # step 2 kept in slot 1
    assert not torch.equal(third.gs_mean[:, :per_step], first.gs_mean[:, :per_step])     # step 3 replaced step 1
    with torch.no_grad():
        rolled = model.predict(third, 0.5)
        mixed = where_state(torch.tensor([True, False]), third, first)
    for key in GS:
        assert torch.equal(getattr(rolled, key), getattr(third, key)), key
    assert int(mixed.gs_count[0]) == 3 and int(mixed.gs_count[1]) == 1


def test_a_sideways_camera_step_moves_the_drawing_by_its_parallax():
    cfg = V6Config(**SPLAT)
    model = V6Model(cfg).eval()
    state = V6State.initial(cfg, 1)
    cam_to_rig = torch.tensor([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])  # z fwd
    g = _gaussians([[10., 0., 0.]], [[1., 1., 1.]])
    state = replace(state, gs_mean=state.gs_mean.index_copy(1, torch.tensor([0]), g[0][None]),
                    gs_log_scale=state.gs_log_scale.index_copy(1, torch.tensor([0]), g[1][None]),
                    gs_opacity=state.gs_opacity.index_copy(1, torch.tensor([0]), g[3][None]),
                    gs_color=state.gs_color.index_copy(1, torch.tensor([0]), g[4][None]),
                    gs_valid=state.gs_valid.index_copy(1, torch.tensor([0]), g[5][None]))
    pixels = torch.stack((torch.arange(64.), torch.full((64,), 32.)), -1)[None, None]
    peaks = []
    for y in (0., 1.):   # the rig steps 1 m to its left: the point moves right in the image
        moved = replace(state, rig_position=torch.tensor([[0., y, 0.]]))
        with torch.no_grad():
            out, _, z_scale = model.render_pixels(moved, K[None, None], cam_to_rig[None, None], pixels, (64, 64))
        peak = peaks.append(int(out.opacity[0].argmax())) or peaks[-1]
        assert out.opacity[0].max() > .9 and (out.depth * z_scale)[0, peak] == pytest.approx(10., rel=1e-3)
    assert peaks == [32, 37]   # fx * 1 m / 10 m = 5 px


def test_rendering_the_input_camera_at_init_beats_its_mean_colour():
    cfg = V6Config(**SPLAT)
    torch.manual_seed(0)
    model = V6Model(cfg).eval()
    with torch.no_grad():
        model.splats.head.out.bias.view(cfg.gaussians ** 2, -1)[:, 10] = 6.   # opaque, so the sky stays out
    batch = synthetic_batch(cfg, 2, 2, 2, generator=torch.Generator().manual_seed(0))
    blocks = torch.rand(*batch['images'].shape[:-2], 8, 8, generator=torch.Generator().manual_seed(2))
    batch['images'] = blocks.repeat_interleave(8, -2).repeat_interleave(8, -1)   # 8 px blocks: structure, not noise
    seq, _ = batch_to_sequence(batch, torch.device('cpu'))
    with torch.no_grad():
        state, _ = model.step(model.initial_state(2, time=seq[0].t), seq[0])
        h, w = cfg.image_height, cfg.image_width
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pixels = torch.stack((xs, ys), -1).reshape(1, 1, -1, 2).float().expand(2, 2, -1, 2)
        out, _, _ = model.render_pixels(state, seq[0].intrinsics, seq[0].cam_to_rig, pixels, (h, w))
    target = seq[0].images[:, :, -1].float().flatten(-2).transpose(-1, -2)                     # [B,V,HW,3]
    ok = seq[0].camera_valid
    err = (out.rgb.view_as(target) - target).abs().mean((-1, -2))[ok].mean()
    const = (target - target.mean(2, keepdim=True)).abs().mean((-1, -2))[ok].mean()
    assert out.opacity.view(2, 2, -1)[ok].mean() > .8 and err < .5 * const, (float(err), float(const))


def test_render_losses_reach_the_head_and_the_encoder_depth():
    cfg = V6Config(**SPLAT, **DENSE)
    torch.manual_seed(0)
    model = V6Model(cfg).train()
    batch = synthetic_batch(cfg, 2, 3, 2, generator=torch.Generator().manual_seed(0))
    seq, targets = batch_to_sequence(batch, torch.device('cpu'), heldout=True, generator=torch.Generator().manual_seed(1))
    state = model.initial_state(2, time=seq[0].t)
    teacher = DenseTeacher(cfg)
    total = 0.
    for t, observed in enumerate((True, True, False)):
        mask = torch.full((2,), observed)
        state, aux = advance(model, state, seq[t], mask)
        terms, logs = step_terms(model, state, seq[t], targets[t], mask, aux=aux, rays_per_camera=32,
                                 dense_teacher=teacher)
        assert all(torch.isfinite(v) for v in terms.values()), terms
        total = total + terms['recon_rgb'] + terms['pred_rgb'] + terms['recon_depth'] + terms['recon_dense']
    assert float(terms['pred_rgb'].detach()) > 0 and float(terms['pred_dense'].detach()) > 0
    assert math.isfinite(logs['dense_cos'])
    total.backward()
    for module in (model.splats.head, model.encoder.depth_head):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)


def test_train_and_evaluate_dry_run_with_gaussians(tmp_path):
    config = json.dumps({**SPLAT, **DENSE})
    train_main(['--dry-run', '--max-steps', '1', '--batch-size', '1', '--steps', '2', '--cameras', '2',
                '--rays-per-camera', '16', '--device', 'cpu', '--num-workers', '0', '--warmup', '1',
                '--ckpt-dir', str(tmp_path / 'run'), '--config', config])
    out = tmp_path / 'eval'
    summary = evaluate_main(['--dry-run', '--checkpoint', str(tmp_path / 'run' / 'last.pt'), '--clips', '2',
                             '--batch-size', '2', '--steps', '3', '--observe', '2', '--stride', '4',
                             '--rays-per-camera', '32', '--vis', '1', '--device', 'cpu', '--out', str(out)])
    for key in ('held_psnr', 'in_psnr', 'held_absrel50', 'in_dense_cos'):
        assert math.isfinite(summary['groups']['observed'][key]['mean']), key
    assert math.isfinite(summary['groups']['rolled_out']['in_psnr']['mean'])
    assert len(list((out / 'vis').glob('*.png'))) == 1


def test_a_scene_map_checkpoint_initialises_a_gaussian_model(tmp_path):
    torch.manual_seed(0)
    old = V6Model(V6Config(**TINY, **SCENE)).state_dict()
    assert any(k.startswith('decoder.scene.') for k in old) and any(k.startswith('writer.scene.') for k in old)
    torch.save({'model': old}, tmp_path / 'scene.pt')
    model = V6Model(V6Config(**SPLAT))
    missing, unexpected = init_weights(model, tmp_path / 'scene.pt')
    assert not unexpected and missing and all(k.startswith('splats.') for k in missing)
    shared = model.state_dict()
    assert all(torch.equal(shared[k], v) for k, v in old.items() if k in shared)
    with pytest.raises(SystemExit):  # without Gaussians the scene map has to load
        init_weights(V6Model(V6Config(**TINY)), tmp_path / 'scene.pt')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='gsplat needs a GPU')
def test_gsplat_matches_the_reference():
    pytest.importorskip('gsplat')
    torch.manual_seed(0)
    n = 200
    mean = torch.randn(n, 3) * torch.tensor([2., 2., 1.]) + torch.tensor([0., 0., 8.])
    g = (mean, torch.full((n, 3), -1.5) + .3 * torch.randn(n, 3), torch.nn.functional.normalize(torch.randn(n, 4), dim=-1),
         torch.randn(n), torch.rand(n, 3 + 4), torch.ones(n, dtype=torch.bool))
    dev = torch.device('cuda')
    rgb, alpha, z, feat = composite_gsplat(*(x.to(dev) for x in g), torch.eye(4, device=dev)[None], K.to(dev)[None],
                                           64, 64, 4)
    ys, xs = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing='ij')
    pixels = torch.stack((xs, ys), -1).reshape(-1, 2)
    ref = composite_torch(*g, torch.eye(4), K, pixels, 4)
    torch.testing.assert_close(rgb[0].reshape(-1, 3).cpu(), ref[0], atol=2e-2, rtol=0)
    torch.testing.assert_close(alpha[0].reshape(-1).cpu(), ref[1], atol=2e-2, rtol=0)
    torch.testing.assert_close(feat[0].reshape(-1, 4).cpu(), ref[3], atol=2e-2, rtol=0)
