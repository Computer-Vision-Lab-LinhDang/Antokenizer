import math
import torch
from mavt.grain_prior.rendering import compose_samples, make_samples
from mavt.grain_prior.query import RayQuery, PointQuery
from .test_shared_readers import setup


def test_constant_slab_transmittance_empty_and_two_surfaces():
    sig=torch.full((1,1,32),2.)
    delta=torch.full_like(sig,1/32)
    z=(torch.arange(32)[None,None]+.5)/32+1
    h=torch.ones(1,1,32,4);color=torch.ones(1,1,32,3)*.25
    out=compose_samples(sig,delta,z,h,color,torch.tensor([[2.]]),torch.zeros(4),torch.zeros(3))
    torch.testing.assert_close(out['opacity'],torch.tensor([[1-math.exp(-2)]]))
    torch.testing.assert_close(out['rgb_core'],out['opacity'][...,None]*.25*torch.ones(1,1,3))
    empty=compose_samples(sig*0,delta,z,h,color,torch.tensor([[2.]]),torch.ones(4),torch.ones(3))
    assert empty['depth'].item() == 2 and torch.equal(empty['h'],torch.ones(1,1,4))
    assert empty['opacity'].item()==0 and torch.equal(empty['rgb_core'],torch.ones(1,1,3))
    # Half terminates at surface 1, nearly all remaining mass at surface 2.
    two=compose_samples(torch.tensor([[[math.log(2),30.]]]),torch.ones(1,1,2),torch.tensor([[[1.,3.]]]),
                        h[:,:,:2],color[:,:,:2],torch.tensor([[4.]]),torch.zeros(4),torch.zeros(3))
    torch.testing.assert_close(two['depth'],torch.tensor([[2.]]))


def test_off_axis_z_depth_and_ray_point_identity():
    m,v,d=setup('shared_field_v1')
    q=RayQuery(v.record_camera,torch.tensor([[[.5,.5],[1.,.5]]]),.1,4.,8)
    xyz,z,delta,far,direction=make_samples(v,q)
    assert far[0,0] == 4
    assert far[0,1] < 4 # range is 4; camera z is smaller
    p=d.read_rays(v,q)
    direct=d.read_points(v,PointQuery(xyz.flatten(1,2),v.time_id))
    torch.testing.assert_close(p.sample_h.flatten(1,2),direct.h)
    torch.testing.assert_close(p.sample_density.flatten(1,2),direct.density)
    torch.testing.assert_close(p.depth,(p.weights*z).sum(-1)+(1-p.opacity)*far)
    p2=d.read_rays(v,q,chunk=1)
    torch.testing.assert_close(p.rgb_core,p2.rgb_core,atol=1e-5,rtol=1e-5)
    (d.decode(p)['rgb'].mean()+d.decode(p)['depth'].mean()).backward()
    assert all(torch.isfinite(x.grad).all() for x in d.parameters() if x.grad is not None)
