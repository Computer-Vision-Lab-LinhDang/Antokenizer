"""Dense features: the frozen DINOv3 teacher, pixel sampling, rendered channels with detached weights, the loss."""
import json
from dataclasses import replace

import pytest
import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.decoder import V6Decoder
from mavt4d.v6.dense import DenseTeacher, load_teacher, sample_features, teacher_features
from mavt4d.v6.losses import DEFAULT_WEIGHTS, sequence_loss
from mavt4d.v6.model import RETIRED, V6Model
from mavt4d.v6.train import batch_to_sequence, init_weights, main, synthetic_batch

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)
DENSE = dict(dense_dim=8, dense_teacher='dinov3_vits16')
SCENE = dict(scene_samples=4, scene_width=8, scene_cell=4., scene_cells_x=12, scene_cells_y=10,
             scene_x_min=-16., scene_y_min=-20.)


def cfg(**extra):
    return V6Config(**{**TINY, **DENSE, **extra})


def test_config_gates_the_dense_modules():
    off = V6Model(V6Config(**TINY))
    assert off.cfg.dense_target_dim == 0 and off.understanding.dense is None and off.decoder.field.dense_out is None
    on = V6Model(cfg(**SCENE))
    assert on.cfg.dense_target_dim == 384
    assert on.decoder.scene.dense is not None and on.decoder.background.dense is not None
    with pytest.raises(ValueError):
        V6Config(**TINY, dense_dim=8, dense_teacher='siglip2_b16')


def test_teacher_matches_transformers_dinov3(tmp_path):
    tf = pytest.importorskip('transformers')
    c = cfg(image_height=48, image_width=64)
    hf_cfg = tf.DINOv3ViTConfig(hidden_size=384, intermediate_size=1536, num_hidden_layers=12, num_attention_heads=6,
                                num_register_tokens=4, layerscale_value=1e-5, image_size=64, patch_size=16)
    torch.manual_seed(0)
    ref = tf.DINOv3ViTModel(hf_cfg).eval()
    with torch.no_grad():
        for p in ref.parameters():  # random LayerScale / biases so every weight matters
            p.normal_(0, .05)
    torch.save(ref.state_dict(), tmp_path / 'dinov3.pth')
    teacher, loaded = load_teacher(c, tmp_path / 'dinov3.pth')
    assert loaded == len(teacher.state_dict())
    images = torch.rand(2, 3, 48, 64)
    with torch.no_grad():
        want = ref(pixel_values=(images - teacher.pixel_mean) / teacher.pixel_std).last_hidden_state[:, 5:]
    got = teacher(images)
    assert got.shape == (2, 384, 3, 4)
    assert (got.flatten(2).transpose(1, 2) - want).abs().max() < 1e-4
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not teacher.train().training
    # uint8 frames read as /255
    assert torch.allclose(teacher((images * 255).round().to(torch.uint8)), teacher((images * 255).round() / 255))


def test_load_teacher_refuses_missing_weights(tmp_path):
    teacher = DenseTeacher(cfg())
    sd = {k: v for k, v in teacher.state_dict().items() if not k.startswith('blocks.11.')}
    torch.save(sd, tmp_path / 'part.pth')
    with pytest.raises(ValueError, match='not found'):
        load_teacher(cfg(), tmp_path / 'part.pth')
    with pytest.raises(ValueError):
        load_teacher(cfg(), tmp_path / 'missing')


def test_sample_features_reads_patch_centres():
    feats = torch.arange(2 * 3 * 4, dtype=torch.float32).view(1, 1, 2, 3, 4)  # D 2, grid 3 x 4 over 48 x 64
    centre = lambda i, j: [(j + .5) * 16 - .5, (i + .5) * 16 - .5]  # noqa: E731
    pixels = torch.tensor([[[centre(0, 0), centre(2, 3), centre(1, 2),
                             [(centre(1, 1)[0] + centre(1, 2)[0]) / 2, centre(1, 1)[1]], [-.5, -.5]]]])
    got = sample_features(feats, pixels, 48, 64)[0, 0]
    grid = feats[0, 0]
    want = torch.stack((grid[:, 0, 0], grid[:, 2, 3], grid[:, 1, 2], (grid[:, 1, 1] + grid[:, 1, 2]) / 2,
                        grid[:, 0, 0]))  # the last: border patches past the centres
    assert (got - want).abs().max() < 1e-5


def test_teacher_features_skip_invalid_cameras():
    torch.manual_seed(0)
    teacher = DenseTeacher(cfg())
    images = torch.rand(2, 3, 3, 64, 64)
    valid = torch.tensor([[True, False, True], [False, False, False]])
    out = teacher_features(teacher, images, valid)
    assert out.shape == (2, 3, 384, 4, 4) and not out[0, 1].any() and not out[1].any()
    assert torch.allclose(out[0, 2], teacher(images[0, 2:3])[0], atol=1e-5)


def test_dense_channels_change_nothing_else_and_never_move_density():
    torch.manual_seed(0)
    c = cfg(**SCENE)
    plain = V6Decoder(replace(c, dense_dim=0)).eval()
    dense = V6Decoder(c).eval()
    missing, unexpected = dense.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing and all('dense' in k for k in missing)
    model = V6Model(c).eval()
    state = model.initial_state(1)
    ev = state.scene_evidence.clone()
    ev[0, 4:7, 3:6] = 2.  # some seen cells in front of the rays
    state = replace(state, scene=torch.randn_like(state.scene.float()).to(state.scene.dtype), scene_evidence=ev)
    rays = model.camera_rays(state, torch.tensor([[[30., 0., 32.], [0., 30., 32.], [0., 0., 1.]]]),
                             torch.eye(4)[None, None], torch.rand(1, 1, 32, 2) * 64)
    a, b = plain(state, rays), dense(state, rays)
    assert a.feature is None and b.feature.shape == (1, 32, 8)
    for key in ('rgb', 'depth', 'opacity', 'weights', 'sample_w'):
        assert torch.allclose(getattr(a, key), getattr(b, key), atol=1e-6), key
    b.feature.square().sum().backward()
    density = [dense.field.density, dense.scene.density]
    assert all(p.grad is None or not p.grad.any() for m in density for p in m.parameters())
    assert dense.scene.dense.weight.grad.abs().sum() > 0 and dense.background.dense[-1].weight.grad.abs().sum() > 0


def _dense_run(c, weights, seed=0):
    torch.manual_seed(seed)
    model = V6Model(c).train()
    teacher = DenseTeacher(c)
    batch = synthetic_batch(c, 2, 3, 2, generator=torch.Generator().manual_seed(seed))
    steps, targets = batch_to_sequence(batch, 'cpu', heldout=True, generator=torch.Generator().manual_seed(1))
    loss, log = sequence_loss(model, steps, targets, seeds=[3, 4], weights=weights, rays_per_camera=16,
                              dense_teacher=teacher)
    return model, loss, log


def test_dense_terms_train_the_decoder_but_not_density():
    weights = {k: 0. for k in DEFAULT_WEIGHTS} | {'recon_dense': 1., 'pred_dense': 1.}
    model, loss, log = _dense_run(cfg(**SCENE), weights)
    assert 0 < log['recon_dense'] < 2 and -1 <= log['dense_cos'] <= 1 and 'pred_dense' in log
    loss.backward()
    grad = lambda m: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)  # noqa: E731
    assert grad(model.understanding.dense) > 0 and grad(model.decoder.background.dense) > 0
    assert grad(model.decoder.field.density) == 0 and grad(model.decoder.scene.density) == 0
    # without a teacher the terms are 0
    off_model = V6Model(cfg())
    batch = synthetic_batch(off_model.cfg, 1, 2, 1, generator=torch.Generator().manual_seed(0))
    steps, targets = batch_to_sequence(batch, 'cpu')
    off_loss, off_log = sequence_loss(off_model, steps, targets, seeds=[0], weights=weights, rays_per_camera=8)
    assert off_log['recon_dense'] == 0 and 'dense_cos' not in off_log and float(off_loss) == 0


def test_rolled_out_frames_are_dense_targets_only():
    # a rolled-out step's frame changes its dense target, never the state the model rolled out
    c = cfg()
    torch.manual_seed(0)
    model = V6Model(c).eval()
    teacher = DenseTeacher(c)
    batch = synthetic_batch(c, 1, 3, 1, generator=torch.Generator().manual_seed(0))
    other = {**batch, 'images': batch['images'].clone()}
    other['images'][:, 2] = torch.rand_like(other['images'][:, 2])
    logs = []
    for b in (batch, other):
        steps, targets = batch_to_sequence(b, 'cpu')
        torch.manual_seed(5)
        with torch.no_grad():
            logs.append(sequence_loss(model, steps, targets, seeds=[0], rays_per_camera=16, dense_teacher=teacher)[1])
    # seed 0 observes 2 of 3 steps: only the rolled-out step's targets differ
    assert logs[0]['observed_steps'] == 2
    assert logs[0]['recon_dense'] == pytest.approx(logs[1]['recon_dense'], abs=1e-6)
    assert logs[0]['pred_dense'] != pytest.approx(logs[1]['pred_dense'], abs=1e-6)


def test_train_dry_run_and_init_from_a_checkpoint_with_the_retired_head(tmp_path):
    old = V6Model(V6Config(**TINY)).state_dict()
    old.update({'understanding.dense_head.1.weight': torch.zeros(4, 4), 'understanding.dense_background': torch.zeros(4)})
    assert any(k.startswith(RETIRED) for k in old)
    torch.save({'model': old}, tmp_path / 'old.pt')
    model = V6Model(cfg())
    missing, unexpected = init_weights(model, tmp_path / 'old.pt')
    assert not unexpected and any(k.startswith('understanding.dense.') for k in missing)
    run = tmp_path / 'run'
    assert main(['--dry-run', '--steps', '2', '--cameras', '1', '--rays-per-camera', '8', '--device', 'cpu',
                 '--warmup', '1', '--max-steps', '1', '--ckpt-dir', str(run), '--backbone', 'dinov3_vits16',
                 '--config', json.dumps({**TINY, **DENSE}), '--init', str(tmp_path / 'old.pt')]) == 1
    logs = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines()]
    assert any(r.get('event') == 'dense_teacher' for r in logs)
    step = [r for r in logs if 'grad_norm' in r][0]
    assert 'recon_dense' in step and 'dense_cos' in step
