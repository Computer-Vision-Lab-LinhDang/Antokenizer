import dataclasses
import pytest
import torch
from mavt.grain_prior.evaluate_shared import cross_depth_error, depth_accuracy, paired_scene_ci, evaluate_nvs
from mavt.grain_prior.query import Camera
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.model import MAVTGrainPrior


def camera(): return Camera(torch.tensor([[1.,1.,.5,.5]]),torch.eye(3)[None],torch.zeros(1,3))


def test_invalid_projections_stay_in_fixed_denominator():
    a=torch.ones(1,4,4); mask=torch.ones_like(a,dtype=torch.bool)
    b=a.clone(); c=camera()
    assert cross_depth_error(a,b,c,c,mask)['cc_d']==0
    a[0,0,0]=float('nan');a[0,0,1]=-1
    o=cross_depth_error(a,b,c,c,mask)
    assert o['count']==16 and o['invalid_fraction']==2/16 and o['cc_d']==2/16
    off=Camera(c.kappa,c.R,torch.tensor([[100.,0.,0.]]))
    assert cross_depth_error(b,b,c,off,mask)['cc_d']==1
    with pytest.raises(ValueError): cross_depth_error(b,b,c,c,mask & False)


def test_constant_plane_consistency_cannot_replace_gt_guard():
    pred=torch.ones(1,4,4);mask=torch.ones_like(pred,dtype=torch.bool);c=camera()
    gt=pred*3
    assert cross_depth_error(pred,pred,c,c,mask)['cc_d']==0
    assert depth_accuracy(pred,gt,mask)['absrel'] > .6
    # Scene units, not pixels or duplicate seed rows.
    result=paired_scene_ci(torch.ones(4,3)*.5,torch.ones(4,3),draws=100)
    assert result['upper'] < 0 and result['scenes']==4 and result['seeds']==3


@pytest.mark.parametrize('version',['shared_ray_v1','shared_field_v1'])
def test_nvs_target_order_does_not_change_state_or_outputs(version):
    m=MAVTGrainPrior(tiny_config(readout_version=version,innovation_semantics='disabled',render_samples=4)).eval()
    source=torch.rand(1,3,64,64)*2-1
    c=camera(); c2=Camera(c.kappa,c.R,torch.tensor([[.1,0.,0.]]))
    a=evaluate_nvs(m,source,c.kappa,[c,c2],resolution=4)
    b=evaluate_nvs(m,source,c.kappa,[c2,c],resolution=4)
    assert torch.equal(a['state'][0],b['state'][0])
    torch.testing.assert_close(a['views'][0]['rgb'],b['views'][1]['rgb'])
    torch.testing.assert_close(a['views'][1]['depth'],b['views'][0]['depth'])
