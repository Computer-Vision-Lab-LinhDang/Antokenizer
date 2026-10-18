from dataclasses import replace
import torch
from mavt4d.object_state.v5.config import V5Config
from mavt4d.object_state.v5.model import V5Model
from mavt4d.object_state.v5.resampler import FunctionalResampler
from mavt4d.object_state.observation import ObservationStep


def test_candidate_cap_attention_and_all_invalid_exact_zero():
    cfg=V5Config(width=12,evidence_width=12,spatial_heads=3,resampler_heads=3,candidates=2)
    net=FunctionalResampler(cfg)
    q=torch.randn(1,2,3,12); k=torch.randn(1,2,9,12); v=torch.randn_like(k)
    weights=torch.rand(1,2,9); valid=torch.ones(1,2,9,dtype=torch.bool)
    result,attn=net.read(q,k,v,weights,valid,return_attention=True)
    assert (attn>0).sum(-1).max()<=2
    torch.testing.assert_close(attn.sum(-1),torch.ones(1,2,3))
    result,attn=net.read(q,k,v,weights,valid&False,return_attention=True)
    assert torch.equal(result,torch.zeros_like(result)) and not attn.any()


def test_nonfinite_geometry_keeps_appearance_d_correction_without_c_write():
    torch.manual_seed(4)
    cfg=V5Config(slots=3,width=12,evidence_width=12,spatial_heads=3,resampler_heads=3,spatial_blocks=1,content_tokens=2,dynamic_tokens=2)
    model=V5Model(cfg)
    obs=ObservationStep(torch.rand(1,1,2,3,32,32),torch.ones(1,dtype=torch.float64))
    prior=model.predict(model.initial_state(1),1.)
    encoded=model.observer(obs)
    prior,ev,_=model._proposals(prior,obs,encoded)
    ev['points']=ev['points']*float('nan')
    state,diag=model.resampler(prior,ev,model.decoder,torch.zeros(1,3,12),torch.ones(1))
    assert torch.isfinite(state.d).all() and state.d.abs().sum()>0
    assert not state.content_initialized.any()
    state.d.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_nonfinite_rays_keep_finite_appearance_backward_and_disable_c_write():
    torch.manual_seed(4)
    cfg=V5Config(slots=3,width=12,evidence_width=12,spatial_heads=3,resampler_heads=3,spatial_blocks=1,content_tokens=2,dynamic_tokens=2)
    model=V5Model(cfg)
    obs=ObservationStep(torch.rand(1,1,2,3,32,32),torch.ones(1,dtype=torch.float64))
    prior=model.predict(model.initial_state(1),1.)
    prior,ev,_=model._proposals(prior,obs,model.observer(obs))
    ev['origins']=ev['origins']*float('nan')
    ev['directions']=ev['directions']*float('nan')
    state,_=model.resampler(prior,ev,model.decoder,torch.zeros(1,3,12),torch.ones(1))
    assert torch.isfinite(state.d).all() and not state.content_initialized.any()
    state.d.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
