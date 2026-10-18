from dataclasses import replace
import torch
from mavt4d.object_state.v5.config import V5Config
from mavt4d.object_state.v5.model import V5Model
from mavt4d.object_state.v5.state import V5State
from mavt4d.object_state.observation import ObservationStep
from mavt4d.object_state.decoder import RayQuery


def tiny():
    return V5Config(slots=3, content_tokens=2, dynamic_tokens=2, width=12, evidence_width=12, spatial_heads=3, spatial_blocks=2, resampler_heads=3, relay_tokens=2, samples_per_ray=4, query_chunk_size=4, window_size=2)


def test_unified_state_rgb_only_write_and_roundtrip():
    torch.manual_seed(4)
    cfg = tiny()
    model = V5Model(cfg)
    initial = model.initial_state(1)
    assert not hasattr(initial, 'c_bg')
    assert initial.support_bits.shape == (1, 3, 64)
    obs = ObservationStep(torch.rand(1, 1, 2, 3, 32, 32), torch.ones(1, dtype=torch.float64))
    state, diag = model.observe(initial, obs, return_diagnostics=True)
    assert state.content_initialized.any()
    assert state.c.abs().sum() > 0
    assert diag['proposal_depth'].min() > 0
    restored = V5State.from_payload(state.to_payload(), cfg)
    for key, value in state.tensor_dict().items():
        assert torch.equal(value, restored.tensor_dict()[key]), key
    state.c.sum().backward()
    assert model.resampler.value.weight.grad is not None


def test_missing_observation_exact_predict_and_readout_status():
    torch.manual_seed(4)
    model = V5Model(tiny())
    obs = ObservationStep(torch.rand(1, 1, 2, 3, 32, 32), torch.ones(1, dtype=torch.float64))
    state = model.observe(model.initial_state(1), obs)
    missing = replace(obs, time=obs.time + .5, valid=torch.zeros(1, 1, 2, 1, 32, 32, dtype=torch.bool))
    observed = model.observe(state, missing)
    predicted = model.predict(state, .5)
    for key, value in observed.tensor_dict().items():
        torch.testing.assert_close(value, predicted.tensor_dict()[key])
    query = RayQuery(torch.zeros(1, 2, 3), torch.tensor([[[0., 0., 1.], [0., 0., 1.]]]), .1, 3., predicted.t)
    out = model.read(predicted, query)
    assert out['status_logits'].shape == (1, 2, 3)
    assert torch.isfinite(out['rgb']).all()
    assert torch.isfinite(out['status_logits']).all()


def test_warp_inverse_and_zero_control_identity():
    model = V5Model(tiny())
    x = torch.randn(1, 3, 7, 3)
    z = torch.randn(1, 3, 16) * .2
    y = model.decoder.warp(x, z)
    torch.testing.assert_close(model.decoder.warp(y, z, inverse=True), x, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(model.decoder.warp(x, z * 0), x)


def test_render_gradients_finite_status_detached_and_empty_support():
    torch.manual_seed(4)
    model = V5Model(tiny())
    obs = ObservationStep(torch.rand(1, 1, 2, 3, 32, 32), torch.ones(1, dtype=torch.float64))
    state = model.observe(model.initial_state(1), obs)
    query = RayQuery(torch.zeros(1, 2, 3), torch.tensor([[[0., 0., 1.], [0., 0., 1.]]]), .1, 3., state.t)
    output = model.read(state, query)
    output['status_logits'].sum().backward(retain_graph=True)
    assert model.resampler.value.weight.grad is None
    assert model.decoder.control_head[-1].weight.grad is None
    output['rgb'].sum().backward()
    assert model.decoder.control_head[-1].weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    empty = model.initial_state(1, time=state.t)
    output = model.read(empty, query)
    assert torch.equal(output['opacity'], torch.zeros_like(output['opacity']))
    assert torch.isfinite(output['status_logits']).all()


def test_canonical_write_has_no_direct_geometry_probability_gradient():
    torch.manual_seed(4)
    model = V5Model(tiny())
    obs = ObservationStep(torch.rand(1, 1, 2, 3, 32, 32), torch.ones(1, dtype=torch.float64))
    state = model.predict(model.initial_state(1), 1.)
    encoded = model.observer(obs)
    state, ev, _ = model._proposals(state, obs, encoded)
    ev['points'] = ev['points'].detach().requires_grad_()
    ev['probabilities'] = ev['probabilities'].detach().requires_grad_()
    # Freeze corrected controls, isolating the direct canonical-write path.
    state, diag = model.resampler(state, ev, model.decoder, torch.zeros(1, 3, 12), torch.ones(1))
    point_grad, probability_grad = torch.autograd.grad(state.c.sum(), (ev['points'], ev['probabilities']), allow_unused=True)
    # D correction receives geometry, but C query contains no raw D and C routing is detached.
    assert point_grad is None or torch.equal(point_grad, torch.zeros_like(point_grad))
    assert probability_grad is None or torch.equal(probability_grad, torch.zeros_like(probability_grad))


def test_rollout_camera_gradients_do_not_mutate_saved_views():
    torch.manual_seed(4)
    model = V5Model(tiny())
    obs = ObservationStep(torch.rand(2, 1, 2, 3, 32, 32), torch.ones(2, dtype=torch.float64))
    state = model.observe(model.initial_state(2), obs)
    state = model.observe(state, replace(obs, time=obs.time + .5))
    state = model.predict(model.predict(state, .5), .5)
    k, pose = model.camera_parameters(state)
    (state.d.square().mean() + k.square().mean() + pose.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_status_support_uses_visible_volume_samples_not_occluded_layers():
    cfg = tiny()
    cfg = replace(cfg, slots=2)
    model = V5Model(cfg)
    state = model.initial_state(1)
    c = state.c.clone()
    c[:,1] = 1.
    bits = state.support_bits.clone()
    bits[:,1] = 255
    state = replace(state,c=c,support_bits=bits,slot_status=torch.ones_like(state.slot_status),
                    content_initialized=torch.ones_like(state.content_initialized),p_exist=torch.ones_like(state.p_exist))

    class LayerField(torch.nn.Module):
        def forward(self,content,points,directions):
            back = content.mean((1,2)) > .5
            occupied = torch.where(back[:,None],points[...,2]>3,points[...,2]<1)
            return occupied.float()*100, torch.ones_like(points)*.5

    model.decoder.field = LayerField()
    query = RayQuery(torch.zeros(1,1,3),torch.tensor([[[0.,0.,1.]]]),0.,4.,state.t)
    output = model.read(state,query)
    assert output['opacity'].item() > .999
    assert output['evidence_support'].item() < 1e-6


def test_absolute_time_shift_invariance_and_config_fingerprint():
    import pytest
    torch.manual_seed(4)
    model=V5Model(tiny())
    obs=ObservationStep(torch.rand(1,1,2,3,32,32),torch.ones(1,dtype=torch.float64))
    a=model.observe(model.initial_state(1),obs)
    b=model.observe(model.initial_state(1,time=10000.),replace(obs,time=obs.time+10000))
    torch.testing.assert_close(a.c,b.c)
    torch.testing.assert_close(a.d,b.d)
    with pytest.raises(ValueError,match='configuration'):
        V5State.from_payload(a.to_payload(),replace(tiny(),atlas_extent=8.))


def test_pose_without_depth_does_not_claim_metric_and_masked_camera_nan_safe():
    torch.manual_seed(4)
    model=V5Model(tiny())
    image=torch.rand(1,1,2,3,32,32)
    obs=ObservationStep(image,torch.ones(1,dtype=torch.float64),intrinsics=torch.eye(3)[None,None],camera_c2w=torch.eye(4)[None,None],camera_valid=torch.ones(1,1,dtype=torch.bool),metric_valid=torch.zeros(1,1,dtype=torch.bool))
    state=model.observe(model.initial_state(1),obs)
    assert state.camera_valid[0,0] and not state.metric_valid.any()
    obs=replace(obs,intrinsics=obs.intrinsics*float('nan'),camera_c2w=obs.camera_c2w*float('nan'),camera_valid=torch.zeros(1,1,dtype=torch.bool))
    state=model.observe(model.initial_state(1),obs)
    assert torch.isfinite(state.c).all()


def test_reliability_targets_are_measured_depth_errors_not_gate_outputs():
    torch.manual_seed(4)
    model=V5Model(tiny())
    obs=ObservationStep(torch.rand(1,1,2,3,32,32),torch.ones(1,dtype=torch.float64),intrinsics=torch.eye(3)[None,None],camera_c2w=torch.eye(4)[None,None],depth=torch.ones(1,1,1,32,32)*10)
    _,diag=model.observe(model.initial_state(1),obs,return_diagnostics=True)
    assert diag['reliability_target'].shape==(1,3,2,1)
    assert diag['reliability_valid'].any()
    assert diag['reliability_target'][diag['reliability_valid']].max()<.1


def test_missing_endpoint_with_context_is_exact_prediction():
    torch.manual_seed(4)
    model=V5Model(tiny())
    initial=model.initial_state(1)
    valid=torch.ones(1,1,2,1,32,32,dtype=torch.bool)
    valid[:,:,-1]=False
    obs=ObservationStep(torch.rand(1,1,2,3,32,32),torch.ones(1,dtype=torch.float64),valid=valid)
    result=model.observe(initial,obs)
    expected=model.predict(initial,1.)
    for key,value in result.tensor_dict().items():
        assert torch.equal(value,expected.tensor_dict()[key]),key
