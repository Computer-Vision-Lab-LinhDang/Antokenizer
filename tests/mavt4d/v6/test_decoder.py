"""Sparse box field and understanding heads: shapes and the C/D read rules (R5)."""
import math
from dataclasses import fields, replace

import torch

from mavt4d.v6.config import V6Config
from mavt4d.v6.contracts import ACTIVE, FREE, RayBatch, V6State
from mavt4d.v6.decoder import V6Decoder
from mavt4d.v6.geometry import quat_mul, quat_normalize, quat_to_matrix, rotvec_to_quat
from mavt4d.v6.understanding import V6Understanding

CFG = V6Config(slots=3, trunk_layers=1, encoder_layers=2, image_height=64, image_width=64,
               latent_width=32, decoder_width=128)
K, A, W = CFG.slots, CFG.anchors, CFG.latent_width


def make_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    corners = torch.tensor([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    pts = torch.rand(A - 8, 3, generator=g) - .5
    axis = torch.randint(0, 3, (A - 8,), generator=g)
    pts[torch.arange(A - 8), axis] = torch.where(torch.rand(A - 8, generator=g) < .5, -.5, .5)
    cube = torch.cat((corners, pts))  # anchors on a 1 m cube
    s = V6State.initial(CFG, 1)
    return replace(
        s, status=torch.tensor([[ACTIVE, ACTIVE, FREE]], dtype=torch.int8),
        position=torch.tensor([[[0., 0., 10.], [4., 0., 10.], [0., 0., 5.]]]),
        orientation=quat_normalize(torch.randn(1, K, 4, generator=g)),
        anchor_canon=cube.expand(1, K, A, 3).clone(), offset=.05 * torch.randn(1, K, A, 3, generator=g),
        c_obj=torch.randn(1, K, 6, W, generator=g), h=torch.randn(1, K, W, generator=g),
        cov_pv=.01 * torch.eye(6).expand(1, K, 6, 6).clone(),
        background=torch.randn(1, CFG.background_tokens, W, generator=g).half(),
        rig_position=torch.randn(1, 3, generator=g), rig_orientation=quat_normalize(torch.randn(1, 4, generator=g)))


def make_rays(seed=0):
    g = torch.Generator().manual_seed(seed)
    jitter = lambda n: (torch.rand(n, 2, generator=g) - .5) * .1
    d0 = torch.cat((jitter(24), torch.ones(24, 1)), -1)                       # slot 0
    d1 = torch.cat((jitter(24) + torch.tensor([.4, 0.]), torch.ones(24, 1)), -1)  # slot 1
    d2 = torch.cat((jitter(16) + torch.tensor([-.6, .5]), torch.ones(16, 1)), -1)  # nothing
    d = torch.nn.functional.normalize(torch.cat((d0, d1, d2)), dim=-1)[None]
    return RayBatch(origin=torch.zeros(1, 64, 3), direction=d, near_far=torch.tensor([.1, 30.]).expand(1, 64, 2).clone())


def cat_states(*states):
    return replace(states[0], **{f.name: torch.cat([getattr(s, f.name) for s in states]) for f in fields(V6State)})


def decoder():
    torch.manual_seed(0)
    return V6Decoder(CFG).eval()


def test_shapes_and_miss_rays():
    dec, state, rays = decoder(), make_state(), make_rays()
    out = dec(state, rays)
    assert out.rgb.shape == (1, 64, 3) and out.depth.shape == (1, 64) and out.opacity.shape == (1, 64)
    assert out.surface_slot.dtype == torch.int64 and out.surface_canon.shape == (1, 64, 3)
    assert out.status.shape == (1, 64, 3) and out.weights.shape == (1, 64, K + 1)
    assert torch.allclose(out.status.sum(-1), torch.ones(1, 64)) and (out.status >= 0).all()
    assert torch.allclose(out.weights.sum(-1), torch.ones(1, 64), atol=1e-5)
    hits = dec.slot_hits(state, rays)
    assert hits[0, :24, 0].all() and hits[0, 24:48, 1].all() and not hits[0, 48:].any() and not hits[..., 2].any()
    assert (out.opacity[0, :48] > 0).all() and set(out.surface_slot[0, :48].tolist()) <= {0, 1}
    miss = slice(48, 64)
    assert (out.opacity[0, miss] == 0).all() and (out.surface_slot[0, miss] == -1).all()
    assert (out.weights[0, miss, -1] == 1).all()
    assert torch.allclose(out.rgb[:, miss], dec.background_rgb(state, rays.direction)[:, miss], atol=1e-6)
    # chunking and batching do not change the result
    small = dec(state, rays, chunk=7)
    assert torch.allclose(small.rgb, out.rgb, atol=1e-6) and torch.equal(small.surface_slot, out.surface_slot)
    other = make_state(1)
    both = dec(cat_states(state, other), RayBatch(*(torch.cat((x, x)) for x in (rays.origin, rays.direction, rays.near_far))))
    assert torch.allclose(both.rgb[:1], out.rgb, atol=1e-6)
    assert torch.allclose(both.rgb[1:], dec(other, rays).rgb, atol=1e-6)


def test_h_changes_only_colour():
    dec, state, rays = decoder(), make_state(), make_rays()
    a = dec(state, rays)
    b = dec(replace(state, h=state.h + torch.randn_like(state.h)), rays)
    assert torch.equal(a.depth, b.depth) and torch.equal(a.opacity, b.opacity)
    assert (a.rgb[0, :48] - b.rgb[0, :48]).abs().amax() > 1e-4


def test_motion_and_event_not_read():
    dec, state, rays = decoder(), make_state(), make_rays()
    a = dec(state, rays)
    b = dec(replace(state, velocity=torch.randn_like(state.velocity), c_evt=torch.randn_like(state.c_evt),
                    evt_meta=torch.rand_like(state.evt_meta), angular_velocity=torch.randn_like(state.angular_velocity),
                    offset_velocity=torch.randn_like(state.offset_velocity)), rays)
    assert torch.equal(a.rgb, b.rgb) and torch.equal(a.depth, b.depth)


def test_field_does_not_read_covariance():
    dec, state = decoder(), make_state()
    slots = torch.tensor([[0, 1, 0, 1, 2]])
    points = torch.randn(1, 5, 3)
    a = dec.field_at(state, slots, points)
    b = dec.field_at(replace(state, cov_pv=state.cov_pv * 7, cov_rot=state.cov_rot * 3,
                             cov_offset=state.cov_offset * 5), slots, points)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_rigid_equivariance():
    dec, state, rays = decoder(), make_state(), make_rays()
    q = rotvec_to_quat(torch.tensor([.3, -1.1, .7]))
    rot, shift = quat_to_matrix(q), torch.tensor([5., -2., 1.5])
    block = torch.block_diag(rot, rot)
    moved = replace(state, position=state.position @ rot.T + shift, orientation=quat_mul(q.expand(1, K, 4), state.orientation),
                    cov_pv=block @ state.cov_pv @ block.T, rig_position=state.rig_position @ rot.T + shift,
                    rig_orientation=quat_mul(q[None], state.rig_orientation))
    moved_rays = RayBatch(rays.origin @ rot.T + shift, rays.direction @ rot.T, rays.near_far)
    a, b = dec(state, rays), dec(moved, moved_rays)
    assert (a.rgb - b.rgb).abs().amax() < 1e-4 and (a.depth - b.depth).abs().amax() < 1e-4
    assert (a.opacity - b.opacity).abs().amax() < 1e-4 and torch.equal(a.surface_slot, b.surface_slot)


def test_c_obj_is_local_to_its_box():
    dec, state, rays = decoder(), make_state(), make_rays()
    hits = dec.slot_hits(state, rays)
    a = dec(state, rays)
    for i in (0, 1):
        c = state.c_obj.clone()
        c[:, i] += torch.randn_like(c[:, i])
        b = dec(replace(state, c_obj=c), rays)
        changed = (a.rgb - b.rgb).abs().amax(-1) > 0
        assert changed.any() and not (changed & ~hits[..., i]).any()
        assert torch.equal(a.depth[~hits[..., i]], b.depth[~hits[..., i]])


def test_gradients_reach_pose_and_shape():
    dec, state, rays = decoder().train(), make_state(), make_rays()
    state = replace(state, position=state.position.clone().requires_grad_(), c_obj=state.c_obj.clone().requires_grad_(),
                    offset=state.offset.clone().requires_grad_())
    out = dec(state, rays)
    (out.rgb.sum() + out.depth.sum()).backward()
    for x in (state.position, state.c_obj, state.offset):
        assert x.grad is not None and x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()


def test_understanding_reads_only_c():
    torch.manual_seed(0)
    und = V6Understanding(CFG).eval()
    state = make_state()
    emb = und.slot_embedding(state)
    assert emb.shape == (1, K, CFG.decoder_width)
    assert und.class_logits(emb).shape == (1, K, 30)
    text = und.text_proj(emb)
    assert text.shape == (1, K, 1152) and torch.allclose(text.norm(dim=-1), torch.ones(1, K))
    d_moved = replace(state, position=torch.randn_like(state.position), velocity=torch.randn_like(state.velocity),
                      orientation=quat_normalize(torch.randn_like(state.orientation)), offset=torch.randn_like(state.offset),
                      cov_pv=state.cov_pv * 4, cov_rot=state.cov_rot * 2, h=torch.randn_like(state.h),
                      c_evt=torch.randn_like(state.c_evt))
    assert torch.equal(und.slot_embedding(d_moved), emb)
    perm = torch.tensor([2, 0, 1])
    permuted = replace(state, **{k: getattr(state, k)[:, perm] for k in ('c_obj', 'anchor_canon', 'position', 'status')})
    assert (und.slot_embedding(permuted) - emb[:, perm]).abs().amax() < 1e-5


def test_dense_features_do_not_pull_geometry():
    torch.manual_seed(0)
    und, dec, rays = V6Understanding(CFG), decoder(), make_rays()
    state = make_state()
    state = replace(state, position=state.position.clone().requires_grad_(), offset=state.offset.clone().requires_grad_())
    render = dec(state, rays)
    dense = und.dense_features(state, render)
    assert dense.shape == (1, 64, 128)
    assert torch.equal(dense[0, 48:], und.dense_background.expand(16, -1))
    dense.square().sum().backward()
    assert state.position.grad is None and state.offset.grad is None


def test_field_attention_row_chunks_match():
    from mavt4d.v6 import decoder as dec
    torch.manual_seed(0)
    q, k, v = torch.randn(70, 4, 8, 16), torch.randn(70, 4, 6, 16), torch.randn(70, 4, 6, 16)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    old = dec.SDPA_MAX_ROWS
    try:
        dec.SDPA_MAX_ROWS = 16
        assert torch.allclose(dec._sdpa_rows(q, k, v), ref, atol=1e-6)
    finally:
        dec.SDPA_MAX_ROWS = old
