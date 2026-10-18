from dataclasses import replace
import pytest
import torch
from mavt4d.object_state.v5.gauge import fit_prefix_similarity, calibrate_state, metric_query_to_scene, scene_output_to_metric
from mavt4d.object_state.decoder import RayQuery
from mavt4d.object_state.v5.model import V5Model
from mavt4d.object_state.v5.config import V5Config


def test_prefix_fit_observability_and_future_rejection():
    x=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
    y=2*x+torch.tensor([1.,2.,3.])
    valid=torch.ones(1,4,dtype=torch.bool)
    transform,ok=fit_prefix_similarity(x,y,valid,torch.zeros(1,4),torch.ones(1))
    assert ok.all()
    torch.testing.assert_close(transform[0,:3,:3],torch.eye(3)*2,atol=1e-5,rtol=1e-5)
    with pytest.raises(ValueError,match='future'):
        fit_prefix_similarity(x,y,valid,torch.ones(1,4)*2,torch.ones(1))
    _,ok=fit_prefix_similarity(x*0,y,valid,torch.zeros(1,4),torch.ones(1))
    assert not ok.any()


def test_metric_adapter_preserves_render_and_scales_range():
    cfg=V5Config(slots=2,width=12,evidence_width=12,spatial_heads=3,resampler_heads=3,spatial_blocks=1,content_tokens=2,dynamic_tokens=2,samples_per_ray=4)
    model=V5Model(cfg)
    state=model.initial_state(1)
    state=replace(state,slot_status=torch.ones_like(state.slot_status),content_initialized=torch.ones_like(state.content_initialized),p_exist=torch.ones_like(state.p_exist))
    x=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
    state=calibrate_state(state,x,3*x+2,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,4))
    q=RayQuery(torch.zeros(1,1,3),torch.tensor([[[0.,0.,1.]]]),.1,3.,state.t)
    qm=replace(q,origins=q.origins*3+2,near=.3,far=9.)
    base=model.read(state,q)
    out=scene_output_to_metric(model.read(state,metric_query_to_scene(state,qm)),state)
    torch.testing.assert_close(base['rgb'],out['rgb'])
    torch.testing.assert_close(base['opacity'],out['opacity'])
    torch.testing.assert_close(base['range']*3,out['range'])


def test_first_metric_observation_and_sensor_return_require_explicit_bridge():
    from mavt4d.object_state.observation import ObservationStep
    cfg=V5Config(slots=2,width=12,evidence_width=12,spatial_heads=3,resampler_heads=3,spatial_blocks=1,content_tokens=2,dynamic_tokens=2)
    model=V5Model(cfg)
    obs=ObservationStep(torch.rand(1,1,2,3,16,16),torch.ones(1,dtype=torch.float64),intrinsics=torch.eye(3)[None,None],camera_c2w=torch.eye(4)[None,None],metric_valid=torch.ones(1,1,dtype=torch.bool))
    first=model.observe(model.initial_state(1),obs)
    assert first.metric_valid.all()
    rgb=replace(obs,intrinsics=None,camera_c2w=None,metric_valid=None)
    relative=model.observe(model.initial_state(1),rgb)
    with pytest.raises(ValueError,match='calibrat'):
        model.observe(relative,replace(obs,time=obs.time+1))


def test_symmetric_observable_calibration_is_explicitly_stop_gradient():
    x=torch.cat((torch.eye(3),-torch.eye(3)))[None].requires_grad_()
    y=2*x.detach()+1
    transform,valid=fit_prefix_similarity(x,y,torch.ones(1,6,dtype=torch.bool),torch.zeros(1,6),torch.ones(1))
    assert valid.all()
    assert not transform.requires_grad
    torch.testing.assert_close(transform[0,:3,:3],torch.eye(3)*2)
    # The external fixed gauge does not suppress downstream scene gradients.
    scene=x.square().sum()+transform.square().sum()
    scene.backward()
    assert torch.isfinite(x.grad).all()
