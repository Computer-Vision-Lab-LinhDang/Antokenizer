import pytest
import torch
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.model import MAVTGrainPrior
from mavt.grain_prior.pilot import configure_pilot
from mavt.grain_prior.training_schedule import optimizer_groups, lr_multiplier, set_learning_rates


def test_group_lrs_and_decay_are_explicit_and_every_parameter_is_unique():
    m=MAVTGrainPrior(tiny_config(readout_version='shared_field_v1',innovation_semantics='disabled'))
    configure_pilot(m,frozen_encoder=False)
    groups=optimizer_groups(m,reader_lr=1e-4,encoder_lr=1e-5,weight_decay=.05)
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==sum(p.requires_grad for p in m.parameters())
    for g in groups:
        assert g['lr']==(1e-5 if g['role']=='encoder' else 1e-4)
        assert all((p.ndim>=2)==(g['weight_decay']>0) for p in g['params'])
    opt=torch.optim.AdamW(groups)
    set_learning_rates(opt,step=1,total_steps=100,warmup=10,schedule='cosine',min_ratio=.1)
    for g in opt.param_groups: assert g['lr']==pytest.approx(g['base_lr']*.1)
    set_learning_rates(opt,step=100,total_steps=100,warmup=10,schedule='cosine',min_ratio=.1)
    for g in opt.param_groups: assert g['lr']==pytest.approx(g['base_lr']*.1)


def test_warmup_cosine_boundary_and_resume_are_not_history_dependent():
    assert lr_multiplier(10,100,10,'cosine',.1)==1
    assert lr_multiplier(11,100,10,'cosine',.1)<1
    assert lr_multiplier(100,100,10,'constant',.1)==1
    values=[lr_multiplier(i,100,10,'cosine',.1) for i in range(10,101)]
    assert values==sorted(values,reverse=True)
    with pytest.raises(ValueError): lr_multiplier(1,100,100,'cosine',.1)
