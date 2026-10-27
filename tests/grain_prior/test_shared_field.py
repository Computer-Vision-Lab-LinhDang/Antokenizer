import dataclasses
import torch
from mavt.grain_prior.query import PointQuery
from .test_shared_readers import setup

def test_field_point_scaling_support_and_gradient():
    m,v,d=setup('shared_field_v1')
    q=PointQuery(v.points[:,:2],v.time_id)
    p=d.read_points(v,q)
    assert p.support.all() and p.h.shape[-1] == m.cfg.dim
    v2=dataclasses.replace(v,points=v.points*3,radius=v.radius*3,scale=v.scale*3)
    p2=d.read_points(v2,PointQuery(q.xyz*3,v.time_id))
    torch.testing.assert_close(p.h,p2.h,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(p.density/3,p2.density,atol=1e-5,rtol=1e-5)
    v3=dataclasses.replace(v,radius=torch.zeros_like(v.radius))
    empty=d.read_points(v3,q)
    assert not empty.support.any() and torch.isfinite(empty.h).all()
    (p.h.square().mean()+p.density.mean()).backward()
    grads=[p.grad for p in d.field.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    assert any(x.abs().sum()>0 for x in grads)
