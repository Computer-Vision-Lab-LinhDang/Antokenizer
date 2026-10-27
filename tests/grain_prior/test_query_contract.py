import dataclasses
import pytest
import torch
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.model import MAVTGrainPrior
from mavt.grain_prior.query import Camera, RayQuery, PointQuery, prepare_state


def source(model):
    x=torch.rand(1,3,64,64)*2-1
    st=model.stems(x)
    # Stems accepts [B,N,C,H,W]; keep this helper on the public encoder input convention.
    return st


def state(model):
    cfg=model.cfg
    from mavt.grain_prior import schema as sc
    B=1
    return dict(h=sc.new_header(cfg,B,torch.zeros(B),torch.zeros(B),torch.zeros(B),'cpu'),
                u=model.uq[:cfg.n_grains].unsqueeze(0).clone(), rho=torch.zeros(B,cfg.n_grains),
                sigma=torch.ones(B,cfg.n_grains)*.5, f=torch.randn(B,cfg.n_grains,cfg.d_f),
                c0=torch.randn(B,cfg.n_coarse,cfg.d_coarse), kappa0=torch.tensor([[1.,1.,.5,.5]]),
                zb=torch.zeros(B,cfg.n_coarse,cfg.d_coarse),recs=[])


def test_state_snapshot_rejects_unknown_target_fields_and_unavailable_time():
    m=MAVTGrainPrior(tiny_config())
    S=state(m)
    v=prepare_state(m,S)
    p=v.points.clone()
    S['rho'].add_(1)
    assert torch.equal(v.points,p)
    with pytest.raises(dataclasses.FrozenInstanceError): v.time_id=None
    with pytest.raises(ValueError): prepare_state(m,S,mode='future')
    with pytest.raises(ValueError): prepare_state(m,S,record_id=0)
    with pytest.raises(ValueError): prepare_state(m,{**S,'target_rgb':torch.zeros(1)})
    with pytest.raises(ValueError): v.check_time(torch.ones(1))
    v.check_time(torch.zeros(1))


def test_queries_validate_camera_bounds_and_extra_data():
    cam=Camera(torch.tensor([[1.,1.,.5,.5]]),torch.eye(3)[None],torch.zeros(1,3))
    q=RayQuery(cam,torch.tensor([[[.5,.5]]]),.1,4.,32)
    assert q.samples == 32
    with pytest.raises(ValueError): RayQuery(cam,q.uv,4.,.1,32)
    with pytest.raises(ValueError): RayQuery(cam,q.uv*float('nan'),.1,4.,32)
    with pytest.raises(ValueError): Camera(cam.kappa,cam.R*2,cam.t)
    with pytest.raises(TypeError): PointQuery(torch.zeros(1,1,3),torch.zeros(1),target_rgb=torch.zeros(1))
