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


def test_default_architecture_commits_newborn_content_before_rematching():
    """A full-width smooth plane exposed a seed->descriptor dustbin dead end."""
    torch.manual_seed(17)
    model=V5Model(V5Config())
    h,w=192,256
    x=torch.linspace(0,1,w)[None].expand(h,w)
    y=torch.linspace(0,1,h)[:,None].expand(h,w)
    image=torch.stack((x,y,torch.full_like(x,.5)))
    rgb=torch.stack([image,(image+.02*torch.sin(torch.tensor(1.))).clamp(0,1)])[None,None]
    intrinsic=torch.tensor([[w,0,w/2],[0,w,h/2],[0,0,1.]])[None,None]
    obs=ObservationStep(rgb,torch.tensor([.25],dtype=torch.float64),intrinsics=intrinsic,camera_c2w=torch.eye(4)[None,None],depth=torch.full((1,1,1,h,w),2.),metric_valid=torch.ones(1,1,dtype=torch.bool))
    with torch.no_grad():
        state,diag=model.observe(model.initial_state(1),obs,return_diagnostics=True)
    assert diag['births'].any(), 'fixture must produce genuine proposal winners'
    assert state.content_initialized.any(), 'newborn loses proposal evidence before its first C write'
    assert diag['reliability_valid'].any(), 'measured assigned support must train reliability'


def _bootstrap_model(*,null_proposal):
    torch.manual_seed(17)
    cfg=V5Config(slots=3,width=12,evidence_width=12,spatial_heads=3,resampler_heads=3,spatial_blocks=1,content_tokens=2,dynamic_tokens=2)
    model=V5Model(cfg)
    with torch.no_grad():
        model.resampler.ownership_key.weight.zero_()
        model.resampler.ownership_key.bias.fill_(1.)
        model.resampler.slot_seed.fill_(-1.)
        model.resampler.slot_seed[0].fill_(1.)
        model.resampler.dustbin.fill_(2. if null_proposal else 0.)
        model.resampler.descriptor.weight.zero_()
        model.resampler.descriptor.bias.fill_(-100.)
    obs=ObservationStep(torch.rand(1,1,2,3,32,32),torch.ones(1,dtype=torch.float64),intrinsics=torch.eye(3)[None,None],camera_c2w=torch.eye(4)[None,None],depth=torch.ones(1,1,1,32,32)*2)
    return model,obs


def test_birth_evidence_survives_descriptor_jump_and_trains_content_and_reliability():
    model,obs=_bootstrap_model(null_proposal=False)
    state,diag=model.observe(model.initial_state(1),obs,return_diagnostics=True)
    assert diag['births'].sum()==1
    assert state.content_initialized[0,0]
    assert diag['reliability_valid'].any()
    loss=state.c.square().mean()+((diag['reliability']-diag['reliability_target']).square()*diag['reliability_valid']).sum()
    loss.backward()
    assert model.resampler.value.weight.grad.abs().sum()>0
    assert model.resampler.reliability[-1].weight.grad.abs().sum()>0
    with torch.no_grad(): model.resampler.descriptor.bias.fill_(1.)
    again,details=model.observe(state.detach(),replace(obs,time=obs.time+1),return_diagnostics=True)
    assert details['observed'][0,0] and details['reliability_valid'].any()
    assert not torch.equal(again.c,state.c)


def test_empty_all_dustbin_scene_bootstraps_once_without_forcing_later_matches():
    model,obs=_bootstrap_model(null_proposal=True)
    state,diag=model.observe(model.initial_state(1),obs,return_diagnostics=True)
    assert state.content_initialized.sum()==1
    assert diag['reliability_valid'].any()
    again,details=model.observe(state,replace(obs,time=obs.time+1),return_diagnostics=True)
    assert not details['observed'].any(), 'dustbin remains valid after initial bootstrap'
    assert torch.equal(state.c,again.c)
