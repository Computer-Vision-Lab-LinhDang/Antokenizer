import torch
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.model import MAVTGrainPrior
from mavt.grain_prior.pilot import configure_pilot
from mavt.grain_prior.query import Camera
from mavt.grain_prior.scene_pilot import scene_step


def test_joint_objective_reaches_feature_density_and_rgb_but_not_frozen_encoder():
    cfg=tiny_config(readout_version='shared_field_v1',innovation_semantics='disabled',render_samples=4)
    m=MAVTGrainPrior(cfg)
    configure_pilot(m,True)
    camera=Camera(torch.tensor([[1.,1.,.5,.5]]),torch.eye(3)[None],torch.zeros(1,3))
    x=torch.rand(1,3,64,64)*2-1
    target=torch.rand_like(x)
    depth=torch.ones(1,64,64)*2
    result=scene_step(m,x,camera.kappa,target,camera,depth,rays=8,seed=0)
    assert set(result['terms'])=={'rgb','depth','feature'}
    assert result['rgb_kind']=='rgb_core'
    result['loss'].backward()
    for module in (m.query_decoder.heads.feature,m.query_decoder.field.density,m.query_decoder.field.radiance):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
    assert all(p.grad is None for p in m.stream_c.parameters())


def test_scene_sampling_does_not_use_target_depth_for_query_locations():
    cfg=tiny_config(readout_version='shared_ray_v1',innovation_semantics='disabled',render_samples=4)
    m=MAVTGrainPrior(cfg).eval()
    c=Camera(torch.tensor([[1.,1.,.5,.5]]),torch.eye(3)[None],torch.zeros(1,3))
    x=torch.rand(1,3,64,64)*2-1
    a=scene_step(m,x,c.kappa,x,c,torch.ones(1,64,64),rays=8,seed=1)
    b=scene_step(m,x,c.kappa,x,c,torch.ones(1,64,64)*10,rays=8,seed=1)
    assert torch.equal(a['uv'],b['uv'])
    assert torch.equal(a['predictions']['depth'],b['predictions']['depth'])
    assert a['terms']['depth'] != b['terms']['depth']
