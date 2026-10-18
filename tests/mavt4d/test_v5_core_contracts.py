"""Regression contracts for the audited second v5 implementation."""
from dataclasses import replace
from unittest.mock import patch
import pytest
import torch
from mavt4d.object_state.v5.config import V5Config
from mavt4d.object_state.v5.model import V5Model
from mavt4d.object_state.observation import ObservationStep
from mavt4d.object_state.decoder import RayQuery


def fixture():
    torch.manual_seed(17)
    model=V5Model(V5Config(slots=2,width=12,evidence_width=12,spatial_heads=3,
                         resampler_heads=3,spatial_blocks=1,content_tokens=2,
                         dynamic_tokens=2,samples_per_ray=4))
    with torch.no_grad():
        model.resampler.ownership_key.weight.zero_()
        model.resampler.ownership_key.bias.fill_(1.)
        model.resampler.slot_seed.fill_(1.)
        model.resampler.dustbin.fill_(-10.)
        model.resampler.descriptor.weight.zero_()
        model.resampler.descriptor.bias.fill_(1.)
    obs=ObservationStep(torch.rand(1,1,2,3,32,32),torch.ones(1,dtype=torch.float64),
                        intrinsics=torch.eye(3)[None,None],camera_c2w=torch.eye(4)[None,None],
                        depth=torch.ones(1,1,1,32,32)*2.)
    prior=model.predict(model.initial_state(1),1.)
    prior,ev,_=model._proposals(prior,obs,model.observer(obs))
    return model,obs,prior,ev


@pytest.mark.parametrize('initialized',[False,True])
def test_reliability_labels_never_change_inference(initialized):
    model,obs,state,ev=fixture()
    if initialized:
        state,_=model.resampler(state,ev,model.decoder,torch.zeros(1,2,12),torch.ones(1))
    one=dict(ev,reliability_target=torch.ones_like(ev['reliability_target']),reliability_valid=torch.ones_like(ev['valid']))
    zero=dict(ev,reliability_target=torch.zeros_like(ev['reliability_target']),reliability_valid=torch.zeros_like(ev['valid']))
    a,da=model.resampler(state,one,model.decoder,torch.zeros(1,2,12),torch.ones(1))
    b,db=model.resampler(state,zero,model.decoder,torch.zeros(1,2,12),torch.ones(1))
    assert a.content_initialized.any()
    torch.testing.assert_close(da['reliability'],db['reliability'],rtol=0,atol=0)
    for key,value in a.tensor_dict().items():
        torch.testing.assert_close(value,b.tensor_dict()[key],rtol=0,atol=0)
    assert da['reliability_valid'].any() and not db['reliability_valid'].any()


def test_existence_metadata_cannot_modulate_render_density():
    model,obs,state,ev=fixture()
    state,_=model.resampler(state,ev,model.decoder,torch.zeros(1,2,12),torch.ones(1))
    q=RayQuery(torch.zeros(1,2,3),torch.tensor([[[0.,0.,1.],[0.,0.,1.]]]),.1,3.,state.t)
    controls=model.decoder.extended_controls(state.d)
    alternate=replace(state,d=state.d+10.,p_exist=torch.zeros_like(state.p_exist))
    with patch.object(model.decoder,'extended_controls',return_value=controls):
        a=model.read(state,q); b=model.read(alternate,q)
    assert a['opacity'].sum()>0
    for key in ('rgb','opacity','range'):
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    empty=replace(state,content_initialized=torch.zeros_like(state.content_initialized))
    assert not model.read(empty,q)['opacity'].any()


def test_metric_initial_camera_is_origin_with_external_mapping():
    from mavt4d.object_state.v5.gauge import prepare_observation_gauge
    model,obs,_,_=fixture()
    pose=torch.eye(4)[None,None].clone()
    pose[0,0,:3,:3]=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    pose[0,0,:3,3]=torch.tensor([10.,20.,3.])
    obs=replace(obs,camera_c2w=pose,metric_valid=torch.ones(1,1,dtype=torch.bool))
    state,normalized=prepare_observation_gauge(model.initial_state(1),obs)
    torch.testing.assert_close(normalized.camera_c2w,torch.eye(4)[None,None])
    torch.testing.assert_close(state.scene_to_metric,pose[:,0])
    torch.testing.assert_close(normalized.depth,obs.depth)


def test_sequence_anchor_is_first_admitted_camera_and_world_invariant():
    from mavt4d.object_state.v5.gauge import normalize_camera_sequence
    poses=torch.eye(4).repeat(4,2,1,1)
    poses[...,0,3]=torch.arange(8.).reshape(4,2)
    admitted=torch.zeros(4,2,dtype=torch.bool); admitted[1:3,0]=True
    valid=torch.ones(4,2,dtype=torch.bool)
    normalized,external=normalize_camera_sequence(poses,valid,admitted)
    world=torch.eye(4); world[:3,:3]=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    world[:3,3]=torch.tensor([7.,2.,-4.])
    shifted,shiftmap=normalize_camera_sequence(world@poses,valid,admitted)
    torch.testing.assert_close(normalized,shifted)
    torch.testing.assert_close(normalized[1,0],torch.eye(4))
    torch.testing.assert_close(external@normalized,poses)
    torch.testing.assert_close(shiftmap,world@external)
    with pytest.raises(ValueError,match='admitted'):
        normalize_camera_sequence(poses,valid,admitted&False)


def test_disabled_photometric_state_is_zero_and_versioned():
    model,obs,state,ev=fixture()
    with torch.no_grad(): model.camera_head[-1].bias[19:23].fill_(10.)
    state,_,_=model._proposals(state,obs,model.observer(obs))
    assert not state.camera[...,22:26].any()
    assert model.config.to_dict()['implementation_version']==2
    assert model.config.to_dict()['camera_photometric']=='disabled'


def test_ownership_diagnostics_describe_refined_distribution():
    model,obs,state,ev=fixture()
    _,diag=model.resampler(state,ev,model.decoder,torch.zeros(1,2,12),torch.ones(1))
    torch.testing.assert_close(diag['ownership_logits'].softmax(-1)*ev['valid'][...,None],diag['ownership'])
    assert torch.equal(diag['ownership_valid'],ev['valid'])
    assert 'capacity_overflow' not in diag
    assert diag['slot_capacity_full'].dtype==torch.bool
    assert (diag['unassigned_evidence_count']<=ev['valid'].sum(-1)).all()


def test_observe_returns_live_ownership_logits_for_gradient_telemetry():
    model,obs,_,_=fixture()
    state,diag=model.observe(model.initial_state(1),obs,return_diagnostics=True)
    diag['ownership_logits'].retain_grad()
    state.d.square().sum().backward()
    assert diag['ownership_logits'].grad is not None
    assert torch.isfinite(diag['ownership_logits'].grad).all()
    assert diag['ownership_logits'].grad.abs().sum()>0
