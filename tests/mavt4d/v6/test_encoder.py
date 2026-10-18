import json
import math

import pytest
import torch

from mavt4d.geometry.camera import patch_ray_dirs
from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import Observation
from mavt4d.v6.encoder import V6Encoder, load_backbone, load_dinov3
from mavt4d.v6.rays import fourier_log_depth, patch_depth_stats, patch_rays, plucker

CFG = V6Config(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64)


def make_obs(cfg, b=1, v=2, seed=0, dt=0.1):
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.image_height, cfg.image_width
    K = torch.tensor([[50., 0., w / 2], [0., 55., h / 2], [0., 0., 1.]]).expand(b, v, 3, 3).clone()
    return Observation(
        images=torch.rand(b, v, cfg.tubelet, 3, h, w, generator=g), tube_dt=torch.full((b,), dt),
        intrinsics=K, cam_to_rig=torch.eye(4).expand(b, v, 4, 4).clone(),
        camera_key=torch.arange(v).expand(b, v).clone(), camera_valid=torch.ones(b, v, dtype=torch.bool),
        depth=torch.rand(b, v, h, w, generator=g) * 50 + 1, depth_valid=torch.rand(b, v, h, w, generator=g) > 0.7,
        rig_pose=torch.eye(4).expand(b, 4, 4).clone(), rig_pose_valid=torch.ones(b, dtype=torch.bool),
        has_metric_depth=torch.ones(b, dtype=torch.bool), t=torch.zeros(b, dtype=torch.float64))


def replace_obs(obs, **kw):
    return Observation(**{**obs.tensor_dict(), **kw})


@pytest.fixture(scope='module')
def encoder():
    torch.manual_seed(0)
    return V6Encoder(CFG).eval()


@pytest.mark.parametrize('v', [1, 7])
def test_shapes(encoder, v):
    b, n, e, d = 2, CFG.patches_per_camera, CFG.encoder_width, CFG.depth_hypotheses
    obs = make_obs(CFG, b=b, v=v)
    with torch.no_grad():
        out = encoder(obs)
    assert out.tokens.shape == (b, v, n, e) and out.valid.shape == (b, v, n)
    assert out.depth_hyp.shape == (b, v, n, d, 5) and out.ray_cam.shape == (b, v, n, 3)
    assert out.camera_motion.shape == (b, v, 12)
    assert out.measured_depth.shape == (b, v, n, 2) and out.measured_valid.shape == (b, v, n)
    assert out.valid.all() and out.depth_hyp.dtype == torch.float32
    assert out.depth_hyp[..., :2].abs().max() <= 8
    for x in (out.tokens, out.depth_hyp, out.camera_motion, out.measured_depth, out.ray_cam):
        assert torch.isfinite(x).all()
    # measured tag: lower median log z and log1p(count) of the first patch
    z, ok = obs.depth[0, 0, :16, :16], obs.depth_valid[0, 0, :16, :16]
    assert torch.allclose(out.measured_depth[0, 0, 0], torch.stack((z[ok].median().log(), (ok.sum() + 1.).float().log())))


def test_too_many_cameras(encoder):
    with pytest.raises(ValueError):
        encoder(make_obs(CFG, v=8))


def test_dead_camera_and_camera_independence(encoder):
    obs = make_obs(CFG, b=2, v=3)
    obs = replace_obs(obs, camera_valid=torch.tensor([[True, False, True], [True, True, True]]))
    images, K = obs.images.clone(), obs.intrinsics.clone()
    images[0, 1] = torch.rand_like(images[0, 1])
    K[0, 1] = 0.
    with torch.no_grad():
        a = encoder(obs)
        b = encoder(replace_obs(obs, images=images, intrinsics=K))
    assert not a.valid[0, 1].any() and not a.measured_valid[0, 1].any()
    for x, y in ((a.tokens, b.tokens), (a.depth_hyp, b.depth_hyp), (a.camera_motion, b.camera_motion)):
        assert torch.equal(x, y)
    # a live camera's pixels only change its own outputs
    images = obs.images.clone()
    images[1, 1] = torch.rand_like(images[1, 1])
    with torch.no_grad():
        c = encoder(replace_obs(obs, images=images))
    assert not torch.equal(a.tokens[1, 1], c.tokens[1, 1])
    assert torch.equal(a.tokens[:, [0, 2]], c.tokens[:, [0, 2]]) and torch.equal(a.tokens[0], c.tokens[0])


def test_no_depth_copy_path():
    torch.manual_seed(0)
    enc = V6Encoder(CFG).train()
    obs = make_obs(CFG)
    depth = obs.depth.clone().requires_grad_(True)
    out = enc(replace_obs(obs, depth=depth))
    assert out.depth_hyp.requires_grad and not out.measured_depth.requires_grad
    (grad,) = torch.autograd.grad(out.depth_hyp.sum() + out.tokens.sum(), depth, allow_unused=True)
    assert grad is None or not grad.any()


def test_tube_dt_changes_tokens(encoder):
    obs = make_obs(CFG)
    with torch.no_grad():
        a = encoder(replace_obs(obs, tube_dt=torch.tensor([0.25])))
        b = encoder(replace_obs(obs, tube_dt=torch.tensor([1.0])))
    assert not torch.allclose(a.tokens, b.tokens)


def test_depth_and_line_resolution():
    z = torch.tensor([1., 2., 5., 10., 20., 50., 80., 120.])
    f0, f1 = fourier_log_depth(z.log(), CFG.fourier_bands), fourier_log_depth((1.02 * z).log(), CFG.fourier_bands)
    assert f0.dtype == torch.float32 and f0.shape == (8, 2 * CFG.fourier_bands)
    assert ((f0 - f1).abs().amax(-1) > 0.1).all()
    # lines at lateral distance z vs 1.02 z from the origin
    d = torch.tensor([0., 0., 1.]).expand(8, 3)
    o = torch.stack((z, torch.zeros_like(z), torch.zeros_like(z)), -1)
    p0, p1 = plucker(o, d), plucker(1.02 * o, d)
    assert p0.shape == (8, 6) and torch.allclose(p0[:, :3], d)
    assert ((p0 - p1).abs().amax(-1) > 5e-3).all()
    assert torch.allclose(p0[:, 3:].norm(dim=-1), torch.log1p(z))
    assert torch.isfinite(plucker(torch.zeros(3), torch.tensor([0., 0., 2.]))).all()


def test_patch_rays_match_camera_convention():
    K = torch.tensor([[50., 0.5, 30.], [0., 55., 33.], [0., 0., 1.]])
    rays = patch_rays(K.expand(1, 1, 3, 3), CFG)
    ref = patch_ray_dirs(K, torch.eye(4), CFG.image_height // 16, CFG.image_width // 16, 16).reshape(-1, 3)
    assert torch.allclose(rays[0, 0], ref, atol=1e-6)
    assert torch.isfinite(patch_rays(torch.zeros(1, 1, 3, 3), CFG)).all()


def test_patch_depth_stats_empty_patch():
    depth = torch.full((1, 1, 64, 64), 4.)
    valid = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    valid[0, 0, :16, :16] = True
    measured, ok = patch_depth_stats(depth, valid, CFG)
    assert ok[0, 0, 0] and not ok[0, 0, 1:].any()
    assert torch.allclose(measured[0, 0, 0], torch.tensor([math.log(4.), math.log(257.)]))
    assert not measured[0, 0, 1:].any()


def test_load_dinov3_missing_path(encoder, tmp_path):
    loaded, missing = load_dinov3(encoder, tmp_path / 'missing')
    assert loaded == [] and 'blocks.0.attn.qkv.weight' in missing
    assert load_dinov3(encoder, str(tmp_path / 'nope' / 'model.safetensors'))[0] == []


def _write_safetensors(path, sd):
    names = {torch.float32: 'F32'}
    header, blobs, offset = {}, [], 0
    for key, value in sd.items():
        raw = value.contiguous().numpy().tobytes()
        header[key] = {'dtype': names[value.dtype], 'shape': list(value.shape), 'data_offsets': [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode()
    path.write_bytes(len(head).to_bytes(8, 'little') + head + b''.join(blobs))


def test_load_dinov3_hf_layout(tmp_path):
    torch.manual_seed(1)
    enc = V6Encoder(CFG)
    e = CFG.encoder_width
    sd = {'embeddings.cls_token': torch.randn(1, 1, e), 'embeddings.mask_token': torch.zeros(1, 1, e),
          'embeddings.register_tokens': torch.randn(1, 4, e),
          'embeddings.patch_embeddings.weight': torch.randn(e, 3, 16, 16),
          'embeddings.patch_embeddings.bias': torch.randn(e), 'norm.weight': torch.randn(e), 'norm.bias': torch.randn(e)}
    for i in range(CFG.encoder_layers):
        p = f'layer.{i}.'
        for name in ('q', 'k', 'v', 'o'):
            sd[f'{p}attention.{name}_proj.weight'] = torch.randn(e, e)
            if name != 'k':
                sd[f'{p}attention.{name}_proj.bias'] = torch.randn(e)
        sd.update({f'{p}norm1.weight': torch.randn(e), f'{p}norm1.bias': torch.randn(e),
                   f'{p}norm2.weight': torch.randn(e), f'{p}norm2.bias': torch.randn(e),
                   f'{p}layer_scale1.lambda1': torch.randn(e), f'{p}layer_scale2.lambda1': torch.randn(e),
                   f'{p}mlp.up_proj.weight': torch.randn(4 * e, e), f'{p}mlp.up_proj.bias': torch.randn(4 * e),
                   f'{p}mlp.down_proj.weight': torch.randn(e, 4 * e), f'{p}mlp.down_proj.bias': torch.randn(e)})
    _write_safetensors(tmp_path / 'model.safetensors', sd)
    loaded, missing = load_dinov3(enc, tmp_path)
    assert 'blocks.1.attn.qkv.weight' in loaded and 'patch_embed.weight' in loaded and 'cls_token' in loaded
    assert len(loaded) == 6 + 14 * CFG.encoder_layers and 'geo_embed.0.weight' in missing
    b = enc.blocks[0]
    assert torch.equal(b.attn.qkv.weight, torch.cat([sd[f'layer.0.attention.{n}_proj.weight'] for n in 'qkv']))
    assert not b.attn.qkv.bias[e:2 * e].any() and torch.equal(b.ls2.gamma, sd['layer.0.layer_scale2.lambda1'])
    assert torch.equal(enc.patch_embed.weight[:, :, -1], sd['embeddings.patch_embeddings.weight'])
    assert not enc.patch_embed.weight[:, :, :-1].any()
    # same weights through the Meta checkpoint layout
    meta = {k: v.clone() for k, v in enc.state_dict().items() if not k.startswith(('patch_embed', 'register_tokens'))}
    meta['patch_embed.proj.weight'] = sd['embeddings.patch_embeddings.weight']
    meta['patch_embed.proj.bias'] = sd['embeddings.patch_embeddings.bias']
    meta['storage_tokens'] = sd['embeddings.register_tokens']
    torch.save(meta, tmp_path / 'dinov3_vits16.pth')
    enc2 = V6Encoder(CFG)
    loaded2, _ = load_dinov3(enc2, tmp_path / 'dinov3_vits16.pth')
    assert set(loaded) <= set(loaded2)
    assert torch.equal(enc2.blocks[1].attn.qkv.weight, enc.blocks[1].attn.qkv.weight)
    assert torch.equal(enc2.register_tokens, enc.register_tokens)


SIGLIP_CFG = V6Config.with_backbone('siglip2_b16', slots=3, trunk_layers=1, encoder_layers=2, encoder_width=64,
                                    encoder_heads=4, image_height=48, image_width=64)


def test_siglip2_shapes_and_load(tmp_path):
    torch.manual_seed(2)
    cfg, e = SIGLIP_CFG, SIGLIP_CFG.encoder_width
    # full SiglipModel layout: vision tower under vision_model., plus text tower and pooling head to skip
    sd = {'vision_model.embeddings.patch_embedding.weight': torch.randn(e, 3, 16, 16),
          'vision_model.embeddings.patch_embedding.bias': torch.randn(e),
          'vision_model.embeddings.position_embedding.weight': torch.randn(16, e),  # 4x4 grid, resized to 3x4
          'vision_model.post_layernorm.weight': torch.randn(e), 'vision_model.post_layernorm.bias': torch.randn(e),
          'vision_model.head.probe': torch.randn(1, 1, e), 'text_model.final_layer_norm.weight': torch.randn(e)}
    for i in range(cfg.encoder_layers):
        p = f'vision_model.encoder.layers.{i}.'
        for name in ('q', 'k', 'v', 'out'):
            sd[f'{p}self_attn.{name}_proj.weight'] = torch.randn(e, e)
            sd[f'{p}self_attn.{name}_proj.bias'] = torch.randn(e)
        for name in ('layer_norm1', 'layer_norm2'):
            sd[f'{p}{name}.weight'], sd[f'{p}{name}.bias'] = torch.randn(e), torch.randn(e)
        sd.update({f'{p}mlp.fc1.weight': torch.randn(4 * e, e), f'{p}mlp.fc1.bias': torch.randn(4 * e),
                   f'{p}mlp.fc2.weight': torch.randn(e, 4 * e), f'{p}mlp.fc2.bias': torch.randn(e)})
    _write_safetensors(tmp_path / 'model.safetensors', sd)
    enc = V6Encoder(cfg).eval()
    loaded, missing = load_backbone(enc, tmp_path)
    assert len(loaded) == 5 + 12 * cfg.encoder_layers and 'pos_embed' in loaded and 'blocks.0.ls1.gamma' in missing
    assert torch.equal(enc.blocks[1].attn.qkv.bias, torch.cat(
        [sd[f'vision_model.encoder.layers.1.self_attn.{n}_proj.bias'] for n in 'qkv']))
    assert enc.pos_embed.shape == (1, cfg.patches_per_camera, e) and torch.equal(enc.blocks[0].ls1.gamma, torch.ones(e))
    with torch.no_grad():
        out = enc(make_obs(cfg, v=3))
    assert out.tokens.shape == (1, 3, cfg.patches_per_camera, e) and torch.isfinite(out.tokens).all()
    assert load_backbone(V6Encoder(cfg), tmp_path / 'missing')[0] == []
