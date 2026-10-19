"""Trunk: shapes, attention permissions (leaf h, C_obj isolation, C_evt never a query),
evidence-only measurements and camera-permutation equivariance."""
from dataclasses import fields, replace

import pytest
import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import ACTIVE, FREE, TENTATIVE, EncoderOutput, Observation, TrunkOutput, V6State
from mavt4d.v6.geometry import quat_normalize, quat_to_matrix, rotvec_to_quat
from mavt4d.v6.rays import patch_rays
from mavt4d.v6.trunk import V6Trunk

CFG = V6Config(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64,
               latent_width=32, trunk_width=64, trunk_heads=4, encoder_width=48, background_tokens=4)
B, V, K, A, W = 2, 2, CFG.slots, CFG.anchors, CFG.latent_width
N = CFG.patches_per_camera


def make_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    h, w = CFG.image_height, CFG.image_width
    intr = torch.tensor([[50., 0, w / 2], [0, 50., h / 2], [0, 0, 1]]).repeat(B, V, 1, 1)
    intr[:, 1, 0, 0] = 40.
    # cameras look along rig +x, yawed apart
    base = torch.tensor([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])
    yaw = quat_to_matrix(rotvec_to_quat(torch.tensor([[0, 0, .6], [0, 0, -.6]])))
    c2r = torch.eye(4).repeat(B, V, 1, 1)
    c2r[..., :3, :3] = yaw @ base
    c2r[..., :3, 3] = torch.tensor([[1.5, .3, 1.6], [1.5, -.3, 1.6]])
    obs = Observation(
        images=torch.rand(B, V, CFG.tubelet, 3, h, w, generator=g), tube_dt=torch.full((B,), .1),
        intrinsics=intr, cam_to_rig=c2r, camera_key=torch.tensor([[7, 3], [7, 3]]),
        camera_valid=torch.ones(B, V, dtype=torch.bool), depth=torch.zeros(B, V, h, w),
        depth_valid=torch.zeros(B, V, h, w, dtype=torch.bool), rig_pose=torch.eye(4).repeat(B, 1, 1),
        rig_pose_valid=torch.ones(B, dtype=torch.bool), has_metric_depth=torch.tensor([True, False]),
        t=torch.ones(B, dtype=torch.float64))
    hyp = torch.cat((8 * torch.tanh(r(B, V, N, CFG.depth_hypotheses, 2)), 1 + 2 * torch.rand(B, V, N, CFG.depth_hypotheses, 1, generator=g),
                     .3 * r(B, V, N, CFG.depth_hypotheses, 1), r(B, V, N, CFG.depth_hypotheses, 1)), -1)
    valid = torch.ones(B, V, N, dtype=torch.bool)
    valid[0, 1, :3] = False
    mvalid = torch.rand(B, V, N, generator=g) > .5
    enc = EncoderOutput(
        tokens=r(B, V, N, CFG.encoder_width), valid=valid, depth_hyp=hyp, ray_cam=patch_rays(intr, CFG),
        camera_motion=.1 * r(B, V, 12), measured_depth=torch.where(mvalid[..., None], r(B, V, N, 2).abs() + 1, 0.),
        measured_valid=mvalid)
    s = V6State.initial(CFG, B, time=.5)
    state = replace(
        s, status=torch.tensor([[ACTIVE, TENTATIVE] + [FREE] * (K - 2)] * B, dtype=torch.int8),
        c_obj=r(B, K, 6, W), h=r(B, K, W), c_evt=r(B, K, W), evt_meta=r(B, K, 2).abs(),
        anchor_canon=.5 * r(B, K, A, 3), anchor_desc=r(B, K, A, 16).half(), evidence=torch.rand(B, K, A, generator=g),
        position=torch.tensor([6., 0, 1]) + 2 * r(B, K, 3), velocity=r(B, K, 3), angular_velocity=.1 * r(B, K, 3),
        orientation=quat_normalize(r(B, K, 4)), offset=.05 * r(B, K, A, 3), cov_pv=.5 * torch.eye(6).repeat(B, K, 1, 1),
        exist_logit=r(B, K), background=r(B, CFG.background_tokens, W).half(), global_token=r(B, W),
        rig_position=r(B, 3), rig_orientation=quat_normalize(torch.tensor([1., 0, 0, 0]) + .1 * r(B, 4)))
    return state, enc, obs


def run(trunk, state, enc, obs):
    with torch.no_grad():
        return trunk(state, enc, obs)


def make_trunk(seed=0, **overrides):
    torch.manual_seed(seed)
    return V6Trunk(replace(CFG, **overrides)).eval()


def out_dict(out):
    d = {f.name: getattr(out, f.name) for f in fields(TrunkOutput) if getattr(out, f.name) is not None}
    for i, o in enumerate(d.pop('ownership_by_layer')):
        d[f'ownership_by_layer{i}'] = o
    return d


def test_shapes_and_backward():
    trunk = make_trunk()
    state, enc, obs = make_inputs()
    obs = replace(obs, camera_valid=torch.tensor([[True, True], [True, False]]),
                  intrinsics=torch.where(torch.tensor([[True, True], [True, False]])[..., None, None], obs.intrinsics, 0.))
    out = trunk(state, enc, obs)
    ne, g = V * N, CFG.background_tokens
    expected = dict(ownership=(B, ne, K + 1), anchor_meas=(B, K, A, 3), anchor_meas_valid=(B, K, A),
                    anchor_log_r=(B, K, A, 3), visibility=(B, K, A), depth_conf=(B, K, A), c_obj_prop=(B, K, 6, W),
                    desc_prop=(B, K, A, 16), h_prop=(B, K, W), anchor_affinity=(B, K, 6, A), rig_meas=(B, 6),
                    rig_log_r=(B, 6), background_prop=(B, g, W), background_vis=(B, g), spawn_logit=(B, K),
                    spawn_canon=(B, K, A, 3), camera_rows=(B, V))
    for name, shape in expected.items():
        value = getattr(out, name)
        assert tuple(value.shape) == shape, name
        if value.is_floating_point():
            assert torch.isfinite(value).all(), name
    assert len(out.ownership_by_layer) == 1 and out.ownership_by_layer[0].shape == (B, ne, K + 1)
    assert torch.allclose(out.ownership.sum(-1), torch.ones(B, ne), atol=1e-5)
    assert torch.allclose(out.anchor_affinity.sum(-1), torch.ones(B, K, 6), atol=1e-5)
    assert (out.anchor_log_r >= CFG.log_r_floor).all()
    # routing by key: key 3 < 7 takes row 0; the invalid camera gets no new row
    assert out.camera_rows.tolist() == [[1, 0], [0, -1]]
    # invalid tokens go to trash
    assert (out.ownership[0, N:N + 3, -1] == 1).all() and (out.ownership[1, N:, -1] == 1).all()
    loss = sum(v.float().square().mean() for v in out_dict(out).values() if v.is_floating_point())
    loss.backward()
    for name, p in trunk.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize('layers', [1, 3])
def test_h_is_a_leaf(layers):
    trunk = make_trunk(trunk_layers=layers, ownership_layers=(1, 2))
    state, enc, obs = make_inputs()
    h = state.h.clone()
    h[:, 0] += 5 * torch.randn(B, W)
    a, b = out_dict(run(trunk, state, enc, obs)), out_dict(run(trunk, replace(state, h=h), enc, obs))
    for name in a:
        if name == 'h_prop':
            assert not torch.equal(a[name][:, 0], b[name][:, 0])
            assert torch.equal(a[name][:, 1:], b[name][:, 1:])
        else:
            assert torch.equal(a[name], b[name]), name


def test_mask_rules():
    trunk = make_trunk()
    state, enc, obs = make_inputs()
    ne, tps, c = V * N, CFG.tokens_per_slot, CFG.c_obj_tokens
    ev_valid = enc.valid.flatten(1)
    rows = trunk.route_cameras(state, obs)
    ev_row = rows[:, :, None].expand(B, V, N).flatten(1)
    alloc = state.status != FREE
    m = trunk._mask(ev_valid, ev_row, alloc, torch.zeros(B, CFG.max_cameras, dtype=torch.bool).scatter(1, rows, True))[0]
    tok = lambda k, i: ne + k * tps + i
    evt, dpose, neo, h = c, c + 1, c + 2, tps - 1
    cam0 = ne + K * tps + int(rows[0, 0])
    assert not m[:, :ne][:, ~ev_valid[0]].any()                       # invalid patches are not keys
    for k in range(K):
        assert m[:, tok(k, h)].sum() == int(alloc[0, k])                # h: only its own row reads it
        assert m[tok(k, evt)].sum() == 1                                # C_evt: dummy self-only query
    assert not m[:ne, ne + 2 * tps:ne + 3 * tps].any()                  # evidence ignores the free slot
    assert m[0, tok(0, 0)] and m[0, tok(1, evt)] and not m[0, tok(0, h)]
    assert not m[tok(0, 0), tok(1, 0)] and m[tok(0, 0), tok(1, dpose)] and not m[tok(0, 0), tok(1, neo)]
    assert m[tok(0, 0), tok(0, evt)] and not m[tok(0, 0), tok(0, h)] and m[tok(0, 0), cam0]
    assert m[tok(0, dpose), tok(1, neo)] and not m[tok(0, dpose), tok(1, 0)] and not m[tok(0, neo), tok(1, evt)]
    assert m[tok(0, h), tok(0, 0)] and not m[tok(0, h), tok(1, dpose)]
    free = m[tok(2, dpose)]
    assert free[:ne].equal(ev_valid[0]) and not free[ne:ne + 2 * tps].any() and free[-1]
    assert m[cam0, :N].equal(ev_valid[0, :N]) and not m[cam0, N:ne].any() and m[cam0, tok(0, dpose)]
    assert not m[cam0, tok(0, 0)] and m[-1, tok(1, dpose)] and not m[-1, tok(1, neo)]


def test_c_obj_does_not_read_other_c_obj():
    trunk = make_trunk()
    state, enc, obs = make_inputs()
    c = state.c_obj.clone()
    c[:, 1] += 5 * torch.randn(B, 6, W)
    a, b = run(trunk, state, enc, obs), run(trunk, replace(state, c_obj=c), enc, obs)
    assert torch.equal(a.c_obj_prop[:, 0], b.c_obj_prop[:, 0])
    assert not torch.equal(a.c_obj_prop[:, 1], b.c_obj_prop[:, 1])


def test_c_evt_is_never_a_query():
    trunk = make_trunk()
    state, enc, obs = make_inputs()
    ne, tps = V * N, CFG.tokens_per_slot
    evt = [ne + k * tps + CFG.c_obj_tokens for k in range(K)]
    ref = out_dict(run(trunk, state, enc, obs))

    def scramble(module, inputs, output):  # garbage in every C_evt query output
        output = output.clone()
        output[:, evt] = 1e3 * torch.randn_like(output[:, evt])
        return output

    hooks = [blk.register_forward_hook(scramble) for blk in trunk.blocks]
    try:
        got = out_dict(run(trunk, state, enc, obs))
    finally:
        for hk in hooks:
            hk.remove()
    for name in ref:
        assert torch.equal(ref[name], got[name]), name
    # C_evt is still read as key/value by its own slot
    c_evt = state.c_evt.clone()
    c_evt[:, 0] += 5 * torch.randn(B, W)
    moved = run(trunk, replace(state, c_evt=c_evt), enc, obs)
    assert not torch.equal(moved.c_obj_prop[:, 0], ref['c_obj_prop'][:, 0])


def test_measurements_do_not_depend_on_prior_values():
    trunk = make_trunk()
    state, enc, obs = make_inputs()
    captured = []
    token_points, measure = trunk.token_points, trunk.measure_anchors

    def tp_hook(*args):
        captured.append(('points', token_points(*args)))
        return captured[-1][1]

    def meas_hook(points, weight):
        captured.append(('weight', weight))
        return measure(points, weight)

    trunk.token_points, trunk.measure_anchors = tp_hook, meas_hook
    # autograd: token points have no path from any state tensor
    grads = {f.name: getattr(state, f.name).clone().requires_grad_(True) for f in fields(V6State)
             if getattr(state, f.name).is_floating_point()}
    out = trunk(replace(state, **grads), enc, obs)
    pts = captured[0][1].points
    assert pts.requires_grad
    assert all(gr is None or not gr.any() for gr in torch.autograd.grad(pts.sum(), list(grads.values()), allow_unused=True))
    # shifting slot 0's prior by 1 m: token points identical, anchor_meas = same points under new routing weights
    captured.clear()
    with torch.no_grad():
        base = trunk(state, enc, obs)
        pos = state.position.clone()
        pos[:, 0] += torch.tensor([1., 0, 0])
        shifted = trunk(replace(state, position=pos), enc, obs)
    (_, tp_a), (_, w_a), (_, tp_b), (_, w_b) = captured
    assert torch.equal(tp_a.points, tp_b.points) and torch.equal(tp_a.prob, tp_b.prob)
    remeasured, valid, _ = measure(tp_a.points.flatten(1, 2), w_b)
    assert torch.equal(remeasured, shifted.anchor_meas) and torch.equal(valid, shifted.anchor_meas_valid)
    # with routing frozen to the unshifted weights, the shifted prior measures exactly the same anchors
    assert torch.equal(measure(tp_b.points.flatten(1, 2), w_a)[0], base.anchor_meas)
    # points live in the rig frame: independent of the prior rig pose too
    captured.clear()
    with torch.no_grad():
        trunk(replace(state, rig_position=state.rig_position + 3.), enc, obs)
    assert torch.equal(captured[0][1].points, tp_a.points)


@pytest.mark.parametrize('known_rows', [False, True])
def test_camera_permutation(known_rows):
    trunk = make_trunk()
    state, enc, obs = make_inputs()
    if known_rows:  # keys already own rows 5 and 2
        key = torch.full((B, CFG.max_cameras), -1)
        key[:, 5], key[:, 2] = 3, 7
        state = replace(state, camera_key=key, camera_valid=key >= 0)
    perm = torch.tensor([1, 0])
    pobs = replace(obs, **{f.name: getattr(obs, f.name)[:, perm] for f in fields(Observation)
                           if getattr(obs, f.name).dim() > 1 and getattr(obs, f.name).shape[1] == V and f.name != 'rig_pose'})
    penc = replace(enc, **{f.name: getattr(enc, f.name)[:, perm] for f in fields(EncoderOutput)})
    a, b = out_dict(run(trunk, state, enc, obs)), out_dict(run(trunk, state, penc, pobs))
    assert torch.equal(b['camera_rows'], a['camera_rows'][:, perm])
    for name in a:
        if name == 'camera_rows':
            continue
        x, y = a[name], b[name]
        if name.startswith('ownership') or name == 'points':
            x = x.view(B, V, N, -1)[:, perm].flatten(1, 2)
        elif name == 'point_log_sigma':
            x = x.view(B, V, N)[:, perm].flatten(1)
        if x.dtype == torch.bool:
            assert torch.equal(x, y), name
        else:
            assert torch.allclose(x, y, atol=1e-5, rtol=1e-5), name
    if known_rows:
        assert a['camera_rows'].tolist() == [[2, 5], [2, 5]]
