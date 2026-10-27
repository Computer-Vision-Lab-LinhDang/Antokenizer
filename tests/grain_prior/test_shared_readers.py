import dataclasses
import pytest
import torch
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.model import MAVTGrainPrior
from mavt.grain_prior.query import Camera, RayQuery, PointQuery, prepare_state
from mavt.grain_prior.shared import SharedQueryDecoder
from .test_query_contract import state


def setup(version='shared_ray_v1'):
    cfg=tiny_config(readout_version=version,innovation_semantics='disabled',measurement_version=2)
    m=MAVTGrainPrior(cfg).eval()
    v=prepare_state(m,state(m))
    d=SharedQueryDecoder(cfg,m.stems.teacher_dim).eval()
    return m,v,d


def test_direct_heads_task_selection_and_chunk_invariance():
    m,v,d=setup()
    q=RayQuery(v.record_camera,m.uq[None],.1,4.,8)
    p=d.read_rays(v,q)
    a=d.decode(p,('rgb','depth','feature'))
    assert torch.equal(a['depth'],d.decode(p,('depth',))['depth'])
    assert torch.equal(a['rgb'],d.decode(p,('feature','rgb'))['rgb'])
    p2=d.read_rays(v,q,chunk=3)
    torch.testing.assert_close(p.h,p2.h,atol=1e-5,rtol=1e-5)
    assert p.readout_version == 'shared_ray_v1'
    with pytest.raises(ValueError): d.decode(p,('occupancy',))


def test_direct_innovation_is_one_hidden_increment_and_prefix_masked():
    m,v,d=setup()
    d.cfg.innovation_semantics='shared_hidden_v1'
    r=torch.randn(1,m.cfg.n_slots,m.cfg.d_r);r[...,:2].sigmoid_();r[...,3]=.5
    q=RayQuery(v.record_camera,m.uq[None],.1,4.,8)
    v0=dataclasses.replace(v,innovation=r,length=torch.zeros(1,dtype=torch.long))
    assert torch.equal(d.read_rays(v,q).h,d.read_rays(v0,q).h)
    va=dataclasses.replace(v0,length=torch.tensor([2]))
    r2=r.clone();r2[:,2:]=100
    vb=dataclasses.replace(va,innovation=r2)
    torch.testing.assert_close(d.read_rays(va,q).h,d.read_rays(vb,q).h)
    cam=Camera(q.camera.kappa,q.camera.R,q.camera.t+1)
    with pytest.raises(ValueError): d.read_rays(va,dataclasses.replace(q,camera=cam))

