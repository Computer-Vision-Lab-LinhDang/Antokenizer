import dataclasses
import pytest
import torch
from mavt.grain_prior.model import MAVTGrainPrior
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.query import Camera
from mavt.grain_prior.pilot import configure_pilot, pilot_step, warm_start_query_encoder


@pytest.mark.parametrize('version',['legacy','shared_ray_v1','shared_field_v1'])
@pytest.mark.parametrize('frozen',[True,False])
def test_pilot_gradients_and_output_provenance(version,frozen):
    cfg=tiny_config(readout_version=version,innovation_semantics='disabled',render_samples=4)
    m=MAVTGrainPrior(cfg)
    configure_pilot(m,frozen_encoder=frozen)
    camera=Camera(torch.tensor([[1.,1.,.5,.5]]),torch.eye(3)[None],torch.zeros(1,3))
    x=torch.rand(1,3,64,64)*2-1
    y=torch.rand_like(x)
    result=pilot_step(m,x,camera.kappa,y,camera,torch.ones(1,64,64),rays=8,seed=4)
    assert result['readout_version']==version
    result['loss'].backward()
    encoder=[p.grad for n,p in m.named_parameters() if n.startswith(('stream_c.','anchor.','stems.pixel.'))]
    if frozen: assert all(g is None for g in encoder)
    else: assert any(g is not None and g.abs().sum()>0 for g in encoder)
    prefix='query_decoder.' if version!='legacy' else 'readout.'
    core=[p.grad for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None]
    assert core and all(torch.isfinite(g).all() for g in core)
    assert any(g.abs().sum()>0 for g in core)
    assert all(p.grad is None for n,p in m.named_parameters() if n.startswith('stems.dino.'))


def test_warm_start_copies_encoder_but_initializes_new_reader_explicitly(tmp_path):
    m=MAVTGrainPrior(tiny_config())
    p=tmp_path/'old.pt'
    torch.save(dict(cfg=dataclasses.asdict(m.cfg),model=m.state_dict()),p)
    new,receipt=warm_start_query_encoder(p,'shared_field_v1')
    assert receipt['new_reader_initialized'] and receipt['innovation_semantics']=='disabled'
    for k,v in m.stream_c.state_dict().items(): assert torch.equal(v,new.stream_c.state_dict()[k])
    with pytest.raises(ValueError): warm_start_query_encoder(p,'shared_field_v1',d_f=17)
