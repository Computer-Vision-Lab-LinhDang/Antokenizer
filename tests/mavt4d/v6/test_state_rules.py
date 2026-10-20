"""State rules: dynamics (R3, R4, R6), write rules (R1-R3, R6), slot lifecycle."""
from dataclasses import replace

import torch

from mavt4d.v6 import V6Config, V6State
from mavt4d.v6.contracts import ACTIVE, FREE, TENTATIVE, Observation, TrunkOutput
from mavt4d.v6.dynamics import V6Dynamics
from mavt4d.v6.geometry import quat_normalize, quat_to_matrix, rotate
from mavt4d.v6.lifecycle import V6Lifecycle
from mavt4d.v6.write import V6Writer

CFG = V6Config(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64)
C_FIELDS = ('c_obj', 'c_evt', 'evt_meta', 'h', 'evidence', 'anchor_canon', 'anchor_desc', 'atlas')
D_FIELDS = ('position', 'velocity', 'orientation', 'angular_velocity', 'offset', 'offset_velocity',
            'cov_pv', 'cov_rot', 'cov_offset', 'cross_rig')


def _pd(*shape, scale=1.):
    a = torch.randn(*shape, shape[-1])
    return scale * (a @ a.transpose(-1, -2) / shape[-1] + .1 * torch.eye(shape[-1]))


def active_state(status=(ACTIVE, ACTIVE, FREE), b=2, seed=0, covs=False):
    torch.manual_seed(seed)
    k, a, w = CFG.slots, CFG.anchors, CFG.latent_width
    s = V6State.initial(CFG, b)
    s = replace(s, status=torch.tensor(status, dtype=torch.int8).expand(b, k).clone(),
                position=torch.randn(b, k, 3) + torch.tensor([0., 0., 12.]), velocity=torch.randn(b, k, 3),
                orientation=quat_normalize(torch.randn(b, k, 4)), angular_velocity=.3 * torch.randn(b, k, 3),
                anchor_canon=torch.randn(b, k, a, 3), c_obj=torch.randn(b, k, 6, w), c_evt=torch.randn(b, k, w),
                h=torch.randn(b, k, w), evidence=torch.rand(b, k, a) + .5, evt_meta=torch.rand(b, k, 2),
                anchor_desc=torch.randn(b, k, a, CFG.descriptor_dim).half(),
                atlas=torch.randint(0, 256, (b, k, 64), dtype=torch.uint8),
                offset=.1 * torch.randn(b, k, a, 3), offset_velocity=.1 * torch.randn(b, k, a, 3))
    if covs:
        s = replace(s, cov_pv=_pd(b, k, 6), cov_rot=_pd(b, k, 6, scale=.1), cov_offset=_pd(b, k, a, 3, 2, scale=.05),
                    rig_cov=_pd(b, 12, scale=1e-2), cross_rig=1e-3 * torch.randn(b, k, 6, 12))
    return s


def make_obs(b=2, t=.5, metric=True, rig_valid=False):
    h, w, v = CFG.image_height, CFG.image_width, 1
    kk = torch.tensor([[40., 0, 32], [0, 40., 32], [0, 0, 1]])
    return Observation(
        images=torch.zeros(b, v, CFG.tubelet, 3, h, w), tube_dt=torch.zeros(b), intrinsics=kk.expand(b, v, 3, 3).clone(),
        cam_to_rig=torch.eye(4).expand(b, v, 4, 4).clone(), camera_key=torch.zeros(b, v, dtype=torch.long),
        camera_valid=torch.ones(b, v, dtype=torch.bool), depth=torch.zeros(b, v, h, w),
        depth_valid=torch.zeros(b, v, h, w, dtype=torch.bool), rig_pose=torch.eye(4).expand(b, 4, 4).clone(),
        rig_pose_valid=torch.full((b,), rig_valid), has_metric_depth=torch.full((b,), metric),
        t=torch.full((b,), t, dtype=torch.float64))


def anchors_in_rig(state, canon=None, shift=0.):
    """Anchors of the prior pose, rig frame (+ a world shift)."""
    canon = state.anchor_canon + state.offset if canon is None else canon
    world = state.position[..., None, :] + rotate(state.orientation, canon) + shift
    return (world - state.rig_position[:, None, None]) @ quat_to_matrix(state.rig_orientation)[:, None]


def make_meas(state, valid, anchor_meas=None, log_r=-4.):
    b, k, a = state.evidence.shape
    w, g = CFG.latent_width, CFG.background_tokens
    n = CFG.patches_per_camera
    own = torch.zeros(b, n, k + 1)
    own[..., -1] = 1
    return TrunkOutput(
        ownership=own, ownership_by_layer=(), anchor_meas=anchors_in_rig(state) if anchor_meas is None else anchor_meas,
        anchor_meas_valid=valid, anchor_log_r=torch.full((b, k, a, 3), log_r), visibility=torch.ones(b, k, a),
        depth_conf=torch.ones(b, k, a), c_obj_prop=torch.randn(b, k, 6, w), desc_prop=torch.randn(b, k, a, CFG.descriptor_dim),
        h_prop=torch.randn(b, k, w), anchor_affinity=torch.rand(b, k, 6, a), rig_meas=torch.zeros(b, 6),
        rig_log_r=torch.full((b, 6), -6.), background_prop=torch.randn(b, g, w), background_vis=torch.zeros(b, g),
        spawn_logit=torch.full((b, k), -5.), spawn_canon=torch.randn(b, k, a, 3),
        camera_rows=torch.zeros(b, 1, dtype=torch.long))


def _check_psd(m, tol=1e-6):
    assert torch.allclose(m, m.transpose(-1, -2), atol=1e-6)
    assert torch.linalg.eigvalsh(m.double()).min() >= -tol


# ---------------------------------------------------------------- dynamics

def test_predict_only_keeps_c_and_grows_cov():
    dyn, s = V6Dynamics(CFG), active_state()
    before = s.clone()
    traces = [s.cov_pv[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)]
    with torch.no_grad():
        for _ in range(4):
            s = dyn(s, .5)
            traces.append(s.cov_pv[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1))
    for f in C_FIELDS:
        assert torch.equal(getattr(s, f), getattr(before, f)), f
    alloc = before.allocated
    for t0, t1 in zip(traces, traces[1:]):
        assert (t1 > t0)[alloc].all()
    for f in D_FIELDS:  # free slots untouched
        assert torch.equal(getattr(s, f)[~alloc], getattr(before, f)[~alloc]), f
    assert not torch.equal(s.position[alloc], before.position[alloc])
    assert torch.allclose(s.t, before.t + 2.)
    s.validate(CFG)


def test_dynamics_never_reads_shape():
    dyn, s = V6Dynamics(CFG), active_state()
    c_obj, h = s.c_obj.clone().requires_grad_(), s.h.clone().requires_grad_()
    canon, c_evt = s.anchor_canon.clone().requires_grad_(), s.c_evt.clone().requires_grad_()
    out = dyn(replace(s, c_obj=c_obj, h=h, anchor_canon=canon, c_evt=c_evt), .5)
    loss = out.position.sum() + out.cov_pv.sum() + out.orientation.sum() + out.offset.sum()
    g = torch.autograd.grad(loss, [c_obj, h, canon, c_evt], allow_unused=True)
    assert g[0] is None and g[1] is None and g[2] is None
    assert g[3] is not None  # the law does read C_evt


def test_zero_law_is_constant_velocity():
    dyn, s = V6Dynamics(CFG), active_state(status=(ACTIVE,) * 3)
    law = {k: torch.zeros_like(v) for k, v in dyn.law_params(s).items()}
    for dt in (.1, .37, 2.):
        out = dyn.propagate(s, dt, law)
        assert torch.allclose(out.position, s.position + s.velocity * dt, atol=1e-6, rtol=0)
        assert torch.equal(out.velocity, s.velocity)


def test_frozen_law_composes_exactly():
    dyn, s = V6Dynamics(CFG), active_state(status=(ACTIVE,) * 3, covs=True)
    s = replace(s, rig_velocity=torch.randn(2, 3), rig_angular_velocity=.2 * torch.randn(2, 3))
    with torch.no_grad():
        law = {k: v.clone() for k, v in dyn.law_params(s).items()}
        law.update(kappa=torch.full_like(law['kappa'], .3), omega=.5 * torch.randn_like(law['omega']),
                   accel=torch.randn_like(law['accel']), kappa_w=torch.full_like(law['kappa_w'], .4),
                   spring=torch.full_like(law['spring'], 2.), damping=torch.full_like(law['damping'], .7))
        two = dyn.propagate(dyn.propagate(s, .3, law), .2, law)
        one = dyn.propagate(s, .5, law)
    for f in D_FIELDS + ('rig_position', 'rig_orientation', 'rig_cov', 't'):
        assert torch.allclose(getattr(two, f), getattr(one, f), atol=1e-5), f


# ---------------------------------------------------------------- write rules

def test_zero_innovation_keeps_c_evt():
    writer = V6Writer(CFG)
    c = torch.randn(2, 3, CFG.latent_width)
    nu, log_dt = torch.zeros(2, 3, 6), torch.zeros(2, 3)
    assert torch.equal(c + writer.event_delta(c, nu, log_dt), c)
    nu[0, 0, 1] = 2.
    d = writer.event_delta(c, nu, log_dt)
    assert d[0, 0].abs().sum() > 0 and torch.equal(d[0, 1:], torch.zeros_like(d[0, 1:]))


def test_first_write_copies_and_unseen_is_bit_identical():
    s = active_state()
    s = replace(s, evidence=torch.zeros_like(s.evidence))
    valid = torch.zeros(2, 3, CFG.anchors, dtype=torch.bool)
    valid[:, 0] = True   # slot 0 fully measured
    valid[:, 2] = True   # free slot: must be ignored
    meas = make_meas(s, valid)
    out = V6Writer(CFG)(s, meas, make_obs())
    out.validate(CFG)
    assert torch.allclose(out.c_obj[:, 0], meas.c_obj_prop[:, 0], atol=1e-6)
    assert torch.allclose(out.anchor_desc[:, 0].float(), meas.desc_prop[:, 0].half().float(), atol=1e-3)
    assert torch.allclose(out.evidence[:, 0], torch.ones_like(out.evidence[:, 0]))
    assert (out.last_seen[:, 0] == .5).all() and (out.seen_streak[:, 0] == 1).all()
    for f in C_FIELDS:   # slot 1 unseen, slot 2 free
        assert torch.equal(getattr(out, f)[:, 1:], getattr(s, f)[:, 1:]), f
    for f in D_FIELDS:
        assert torch.equal(getattr(out, f)[:, 2], getattr(s, f)[:, 2]), f
    assert (out.cov_pv[:, 0].diagonal(dim1=-2, dim2=-1).sum(-1) < s.cov_pv[:, 0].diagonal(dim1=-2, dim2=-1).sum(-1)).all()


def test_rgb_only_lambda_is_capped_and_scaled():
    s = replace(active_state(), evidence=torch.zeros(2, 3, CFG.anchors), metric_scale=torch.full((2,), 2.))
    valid = torch.zeros(2, 3, CFG.anchors, dtype=torch.bool)
    valid[:, 0] = True
    meas = make_meas(s, valid, anchor_meas=anchors_in_rig(s) / 2)   # unscaled depth units
    out = V6Writer(CFG)(s, meas, make_obs(metric=False))
    assert torch.allclose(out.evidence[:, 0], torch.full_like(out.evidence[:, 0], CFG.rgb_only_lambda_cap))
    assert torch.allclose(out.position, s.position, atol=1e-4)   # scaled measurement agrees with the prior


def test_joseph_update_keeps_covariances_psd():
    s = active_state(status=(ACTIVE, TENTATIVE, ACTIVE), covs=True)
    valid = torch.rand(2, 3, CFG.anchors) > .3
    valid[:, 2, 2:] = False   # slot 2: two anchors, translation only
    meas = make_meas(s, valid, anchor_meas=anchors_in_rig(s) + .3 * torch.randn(2, 3, CFG.anchors, 3))
    meas = replace(meas, anchor_log_r=torch.empty(2, 3, CFG.anchors, 3).uniform_(-9, 0), rig_meas=.05 * torch.randn(2, 6))
    obs = make_obs(rig_valid=True)
    obs = replace(obs, rig_pose=obs.rig_pose.clone())
    obs.rig_pose[:, :3, 3] = .1 * torch.randn(2, 3)
    writer = V6Writer(CFG)
    out = s
    for _ in range(3):
        out = writer(out, meas, obs)
        for f in ('cov_pv', 'cov_rot', 'cov_offset', 'rig_cov'):
            _check_psd(getattr(out, f))
    out.validate(CFG)


def test_two_anchor_slot_updates_d_not_c_obj():
    s = active_state()
    valid = torch.zeros(2, 3, CFG.anchors, dtype=torch.bool)
    valid[:, 0, :2] = True
    meas = make_meas(s, valid, anchor_meas=anchors_in_rig(s, shift=torch.tensor([.5, 0., 0.])))
    out = V6Writer(CFG)(s, meas, make_obs())
    assert ((out.position[:, 0] - s.position[:, 0])[:, 0] > .1).all()
    assert (out.cov_pv[:, 0, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)
            < s.cov_pv[:, 0, :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)).all()
    for f in ('c_obj', 'evidence', 'anchor_desc', 'atlas', 'anchor_canon'):
        assert torch.equal(getattr(out, f), getattr(s, f)), f
    assert torch.equal(out.offset, s.offset)   # no Procrustes residual with < 3 anchors


def test_write_gradients_are_finite():
    s = active_state(covs=True)
    valid = torch.rand(2, 3, CFG.anchors) > .5
    meas = make_meas(s, valid, anchor_meas=anchors_in_rig(s) + .2 * torch.randn(2, 3, CFG.anchors, 3))
    leaves = {f: getattr(meas, f).clone().requires_grad_() for f in ('anchor_meas', 'anchor_log_r', 'visibility', 'c_obj_prop')}
    writer = V6Writer(CFG)
    out = writer(s, replace(meas, **leaves), make_obs())
    loss = out.position.sum() + out.cov_pv.sum() + out.c_obj.sum() + out.c_evt.sum() + out.orientation.sum()
    loss.backward()
    for name, x in leaves.items():
        assert x.grad is not None and torch.isfinite(x.grad).all(), name
    for p in writer.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


# ---------------------------------------------------------------- lifecycle

def test_lifecycle_spawn_promote_retire():
    life, writer = V6Lifecycle(CFG), V6Writer(CFG)
    s = V6State.initial(CFG, 2)
    torch.manual_seed(1)
    a = CFG.anchors
    canon = .5 * torch.randn(2, 3, a, 3)
    q = quat_normalize(torch.randn(2, 3, 4))
    z = rotate(q, canon) + torch.tensor([0., 0., 10.])     # rig = world = camera frame
    valid = torch.zeros(2, 3, a, dtype=torch.bool)
    valid[:, 0, :4] = True
    meas = replace(make_meas(s, valid, anchor_meas=z), spawn_canon=canon,
                   spawn_logit=torch.tensor([10., -10., 10.]).expand(2, 3).clone())
    out = life(s, meas, make_obs(t=0.))
    assert (out.status[:, 0] == TENTATIVE).all() and (out.status[:, 1:] == FREE).all()  # slot 2: no anchors
    assert (out.uid[:, 0] > 0).all() and torch.equal(out.c_obj[:, 0], meas.c_obj_prop[:, 0])
    assert torch.allclose(out.position[:, 0], torch.tensor([0., 0., 10.]).expand(2, 3), atol=1e-4)
    assert torch.allclose((out.orientation[:, 0] * q[:, 0]).sum(-1).abs(), torch.ones(2), atol=1e-5)
    out.validate(CFG)

    # Seen again: first write is a copy, then TENTATIVE -> ACTIVE.
    valid2 = torch.zeros_like(valid)
    valid2[:, 0] = True
    meas2 = make_meas(out, valid2)
    post = life(writer(out, meas2, make_obs(t=.5)), meas2, make_obs(t=.5))
    assert (post.status[:, 0] == ACTIVE).all()
    assert torch.allclose(post.c_obj[:, 0], meas2.c_obj_prop[:, 0], atol=1e-6)

    # Expected visible but missed twice -> retired; out of view never counts.
    none = make_meas(post, torch.zeros_like(valid))
    behind = replace(post, position=post.position * torch.tensor([1., 1., -1.]))
    s1 = life(post, none, make_obs(t=1.))
    assert (s1.status[:, 0] == ACTIVE).all() and (s1.miss_streak[:, 0] == 1).all()
    s2 = life(s1, none, make_obs(t=1.5))
    assert (s2.status[:, 0] == FREE).all()
    hidden = behind
    for i in range(3):
        hidden = life(hidden, none, make_obs(t=1. + i))
    assert (hidden.status[:, 0] == ACTIVE).all() and (hidden.miss_streak[:, 0] == 0).all()

    # TENTATIVE not re-seen -> FREE.
    gone = life(out, none, make_obs(t=.5))
    assert (gone.status == FREE).all()
