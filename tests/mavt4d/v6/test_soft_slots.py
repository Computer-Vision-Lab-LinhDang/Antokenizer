"""Soft slots (v6.1): existence re-estimated each step and rendered, no miss rule, weak anchors, gateless matching."""
import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import ACTIVE, FREE, Observation, RayBatch, TrunkOutput, V6State, present_slots
from mavt4d.v6.decoder import V6Decoder
from mavt4d.v6.lifecycle import V6Lifecycle
from mavt4d.v6.losses import DEFAULT_WEIGHTS, UNMATCHED_EXIST_WEIGHT, sequence_loss, slot_losses
from mavt4d.v6.model import V6Model
from mavt4d.v6.train import batch_to_sequence, synthetic_batch

TINY = dict(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64, anchors=8)
SOFT = V6Config(**TINY, soft_slots=True)


def _obs(b=1, t=.5):
    h, w = SOFT.image_height, SOFT.image_width
    kk = torch.tensor([[40., 0, 32], [0, 40., 32], [0, 0, 1]])
    return Observation(
        images=torch.zeros(b, 1, SOFT.tubelet, 3, h, w), tube_dt=torch.zeros(b), intrinsics=kk.expand(b, 1, 3, 3).clone(),
        cam_to_rig=torch.eye(4).expand(b, 1, 4, 4).clone(), camera_key=torch.zeros(b, 1, dtype=torch.long),
        camera_valid=torch.ones(b, 1, dtype=torch.bool), depth=torch.zeros(b, 1, h, w),
        depth_valid=torch.zeros(b, 1, h, w, dtype=torch.bool), rig_pose=torch.eye(4).expand(b, 4, 4).clone(),
        rig_pose_valid=torch.zeros(b, dtype=torch.bool), has_metric_depth=torch.ones(b, dtype=torch.bool),
        t=torch.full((b,), t, dtype=torch.float64))


def _meas(spawn_logit, spawn_box=None, cfg=SOFT):
    b, k, a, w = 1, cfg.slots, cfg.anchors, cfg.latent_width
    own = torch.zeros(b, cfg.patches_per_camera, k + 1)
    own[..., -1] = 1
    return TrunkOutput(
        ownership=own, ownership_by_layer=(), anchor_meas=torch.zeros(b, k, a, 3),
        anchor_meas_valid=torch.zeros(b, k, a, dtype=torch.bool), anchor_log_r=torch.full((b, k, a, 3), -4.),
        visibility=torch.ones(b, k, a), depth_conf=torch.ones(b, k, a), c_obj_prop=torch.zeros(b, k, 6, w),
        desc_prop=torch.zeros(b, k, a, cfg.descriptor_dim), h_prop=torch.zeros(b, k, w),
        anchor_affinity=torch.rand(b, k, 6, a), rig_meas=torch.zeros(b, 6), rig_log_r=torch.full((b, 6), -6.),
        background_prop=torch.zeros(b, cfg.background_tokens, w), background_vis=torch.zeros(b, cfg.background_tokens),
        spawn_logit=torch.tensor([spawn_logit]), spawn_canon=torch.zeros(b, k, a, 3),
        camera_rows=torch.zeros(b, 1, dtype=torch.long),
        spawn_box=None if spawn_box is None else torch.tensor([spawn_box]))


def _visible_slot(cfg=SOFT):
    """Slot 0 ACTIVE 10 m in front of the camera (expected visible), the others free."""
    s = V6State.initial(cfg, 1)
    pos = torch.zeros(1, cfg.slots, 3)
    pos[0, 0, 2] = 10.
    return replace(s, status=torch.tensor([[ACTIVE] + [FREE] * (cfg.slots - 1)], dtype=torch.int8), position=pos,
                   cov_pv=.01 * torch.eye(6).expand(1, cfg.slots, 6, 6).clone(), exist_logit=torch.full((1, cfg.slots), 4.))


def test_existence_not_misses_decides_retirement():
    life = V6Lifecycle(SOFT)
    state = _visible_slot()
    kept = state
    for i in range(4):   # missed while expected visible, but the trunk still says it exists
        kept = life(kept, _meas([3., -10., -10.]), _obs(t=.5 * (i + 1)))
    assert int(kept.status[0, 0]) == ACTIVE and float(kept.exist_logit[0, 0]) == 3.
    faded = life(state, _meas([-5., -10., -10.]), _obs())
    assert int(faded.status[0, 0]) == FREE
    held = life(state, _meas([-5., -10., -10.], spawn_box=[0, -1, -1]), _obs())   # the box teacher holds it
    assert int(held.status[0, 0]) == ACTIVE
    hard = V6Lifecycle(V6Config(**TINY))   # without soft slots: two expected-visible misses retire it
    s = replace(state, config_digest=V6State.initial(V6Config(**TINY), 1).config_digest)
    for i in range(2):
        s = hard(s, _meas([3., -10., -10.], cfg=V6Config(**TINY)), _obs(t=.5 * (i + 1)))
    assert int(s.status[0, 0]) == FREE


def _cube_state(cfg, exist):
    corners = torch.tensor([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    s = V6State.initial(cfg, 1)
    g = torch.Generator().manual_seed(0)
    return replace(s, status=torch.tensor([[ACTIVE, FREE, FREE]], dtype=torch.int8),
                   position=torch.tensor([[[0., 0., 10.], [9., 9., 9.], [9., 9., 9.]]]),
                   anchor_canon=corners.expand(1, cfg.slots, cfg.anchors, 3).clone(),
                   c_obj=torch.randn(1, cfg.slots, 6, cfg.latent_width, generator=g),
                   cov_pv=.01 * torch.eye(6).expand(1, cfg.slots, 6, 6).clone(),
                   exist_logit=torch.full((1, cfg.slots), float(exist)))


def test_density_scales_with_existence_only_with_soft_slots():
    rays = RayBatch(origin=torch.zeros(1, 4, 3), direction=F.normalize(torch.tensor([[[0., 0., 1.]] * 4]), dim=-1),
                    near_far=torch.tensor([.1, 30.]).expand(1, 4, 2).clone())
    torch.manual_seed(0)
    soft = V6Decoder(SOFT).eval()
    with torch.no_grad():
        for p in soft.field.parameters():   # an opaque field, so existence is what decides
            p.add_(torch.randn_like(p) * .1)
        on, off = soft(_cube_state(SOFT, 8.), rays).opacity, soft(_cube_state(SOFT, -8.), rays).opacity
    assert (on > 50 * off).all() and (off < 1e-2).all()
    torch.manual_seed(0)
    hard_cfg = V6Config(**TINY)
    hard = V6Decoder(hard_cfg).eval()
    with torch.no_grad():
        a, b = hard(_cube_state(hard_cfg, 8.), rays).opacity, hard(_cube_state(hard_cfg, -8.), rays).opacity
    assert torch.equal(a, b)


def _targets(center, labelled=True):
    return {'box_center': torch.tensor([[center]]), 'box_size': torch.ones(1, 1, 3), 'box_class': torch.zeros(1, 1, dtype=torch.long),
            'box_valid': torch.ones(1, 1, dtype=torch.bool), 'box_labelled': torch.tensor([labelled])}


def test_every_box_gets_a_slot_and_existence_is_supervised():
    torch.manual_seed(0)
    model = V6Model(SOFT)
    state = replace(_cube_state(SOFT, 0.), status=torch.tensor([[ACTIVE, ACTIVE, FREE]], dtype=torch.int8),
                    exist_logit=torch.tensor([[1., -1., 0.]]))
    terms, logs = slot_losses(model, state, _targets([0., 0., 30.]))   # 20 m from slot 0, 21+ from slot 1
    assert float(logs['slots_matched']) == 1 and float(logs['slots_near']) == 0   # no gate: matched anyway
    expect = (F.softplus(torch.tensor(-1.)) + UNMATCHED_EXIST_WEIGHT * F.softplus(torch.tensor(-1.))) / 2
    assert float(terms['exist']) == pytest.approx(float(expect), rel=1e-5)
    assert float(terms['exist_sparse']) == 0
    unlabelled = {**_targets([0., 0., 30.], labelled=False), 'box_valid': torch.zeros(1, 1, dtype=torch.bool)}
    terms, _ = slot_losses(model, state, unlabelled)
    assert float(terms['exist']) == 0
    assert float(terms['exist_sparse']) == pytest.approx(float(torch.sigmoid(torch.tensor([1., -1.])).mean()), rel=1e-5)
    hard_cfg = V6Config(**TINY)
    hard = replace(state, config_digest=V6State.initial(hard_cfg, 1).config_digest)
    _, logs = slot_losses(V6Model(hard_cfg), hard, _targets([0., 0., 30.]))
    assert float(logs['slots_matched']) == 0   # the 4 m gate without soft slots


def test_present_slots_need_existence_only_with_soft_slots():
    state = replace(_cube_state(SOFT, 0.), status=torch.tensor([[ACTIVE, ACTIVE, FREE]], dtype=torch.int8),
                    exist_logit=torch.tensor([[1., -1., 5.]]))
    assert present_slots(state, SOFT).tolist() == [[True, False, False]]
    assert present_slots(state, V6Config(**TINY)).tolist() == [[True, True, False]]


def _rgb_only_step(cfg):
    torch.manual_seed(0)
    model = V6Model(cfg).train()
    batch = synthetic_batch(cfg, 2, 3, 2, generator=torch.Generator().manual_seed(0))
    steps, targets = batch_to_sequence(batch, torch.device('cpu'))
    loss, log = sequence_loss(model, steps, targets, seeds=[3, 3], rays_per_camera=64, box_teacher=False,
                              weights={k: (1. if k == 'recon_rgb' else 0.) for k in DEFAULT_WEIGHTS})
    loss.backward()
    return float(loss), log, model.trunk.spawn_head.weight.grad


def test_images_alone_teach_existence_with_soft_slots():
    loss, log, grad = _rgb_only_step(SOFT)
    assert math.isfinite(loss) and log['slots_allocated'] > 0
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    _, log, grad = _rgb_only_step(V6Config(**TINY))   # before: no slot spawns, and images never reach the head
    assert log['slots_allocated'] == 0 and (grad is None or not grad.any())


def test_soft_slots_add_no_weights():
    assert V6Model(SOFT).state_dict().keys() == V6Model(V6Config(**TINY)).state_dict().keys()
