import torch
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.readout import local_support
from mavt.grain_prior.layers import CrossAttn, attend
from mavt.grain_prior.model import MAVTGrainPrior


def test_support_is_local_and_requires_on_screen_grains():
    q = torch.tensor([[[.25,.25], [.75,.75]]])
    sigma = torch.tensor([[.2]])
    valid = torch.ones(1,1,dtype=torch.bool)
    assert local_support(q, q[:,:1], sigma, valid, 4).tolist() == [[True, False]]
    assert not local_support(q, torch.full((1,1,2),2.), sigma, valid,4).any()
    assert not local_support(q, q[:,:1], sigma, ~valid,4).any()


def test_strict_attention_preserves_far_bias_and_zero_after_projection():
    q=torch.tensor([[[1.]]]); k=torch.tensor([[[0.],[80.]]]); v=torch.tensor([[[0.],[1.]]])
    out=attend(q,k,v,1,torch.tensor([[[[0.,-100.]]]]),strict=True)
    assert out.item() < 1e-8
    ca=CrossAttn(4,1)
    ca.out.bias.data.fill_(.25)
    out=ca(torch.randn(1,2,4),torch.randn(1,3,4),valid=torch.zeros(1,3,dtype=torch.bool),strict=True)
    assert torch.count_nonzero(out)==0


def test_no_support_does_not_remove_bad_depth_from_v2_denominator():
    m=MAVTGrainPrior(tiny_config(measurement_version=2))
    B,N=1,m.cfg.n_cells
    S={'u':m.uq[:m.cfg.n_grains].unsqueeze(0),'rho':torch.zeros(B,m.cfg.n_grains)}
    der={'d':torch.ones(B,m.cfg.n_grains)}
    o=dict(depth_grain=torch.ones(B,N),depth_dense=torch.ones(B,N),support=torch.zeros(B,N,dtype=torch.bool))
    parts,diag={},{}
    m._geom(S,der,[o],torch.full((B,1,N),-2.3),torch.ones(B,1,N,dtype=torch.bool),torch.ones(B),parts,diag)
    assert diag['depth_grain_err_v2'] > .1
    assert diag['depth_grain_err_v1'] == 0
