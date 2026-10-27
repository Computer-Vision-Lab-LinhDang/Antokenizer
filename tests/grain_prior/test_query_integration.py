import dataclasses
import subprocess
import sys
import pytest
import torch
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.model import MAVTGrainPrior, load_grain_prior
from mavt.grain_prior.query import RayQuery, prepare_state
from mavt.grain_prior import schema as sc


@pytest.mark.parametrize('version',['shared_ray_v1','shared_field_v1'])
def test_source_only_encode_checkpoint_and_fresh_process_closure(version,tmp_path):
    cfg=tiny_config(readout_version=version,innovation_semantics='disabled',render_samples=4)
    torch.manual_seed(3)
    m=MAVTGrainPrior(cfg).eval()
    x=torch.rand(1,3,64,64)*2-1
    S=m.encode_source(x)
    v=m.prepare_state(S)
    q=RayQuery(v.record_camera,m.uq[None],.1,4.,4)
    out=m.decode_packet(m.read_rays(v,q))
    assert all(torch.isfinite(t).all() for t in out.values())
    path=tmp_path/'checkpoint.pt'
    torch.save(dict(cfg=dataclasses.asdict(cfg),model=m.state_dict(),step=1),path)
    flat=sc.serialize(cfg,S)
    torch.save(flat,tmp_path/'state.pt')
    # A new process has no encoder activations, packet, or KV cache.
    code='''
import torch, sys
from mavt.grain_prior.model import load_grain_prior
from mavt.grain_prior import schema as sc
from mavt.grain_prior.query import RayQuery
m,cfg,_=load_grain_prior(sys.argv[1])
S=sc.deserialize(cfg,torch.load(sys.argv[2],weights_only=True))
v=m.prepare_state(S)
q=RayQuery(v.record_camera,m.uq[None],.1,4.,4)
torch.save(m.decode_packet(m.read_rays(v,q)),sys.argv[3])
'''
    subprocess.run([sys.executable,'-c',code,str(path),str(tmp_path/'state.pt'),str(tmp_path/'out.pt')],check=True)
    actual=torch.load(tmp_path/'out.pt',weights_only=True)
    for key in out: torch.testing.assert_close(out[key],actual[key],atol=1e-5,rtol=1e-5)
    with pytest.raises(ValueError): m(x,'image') # never silently use the old trainer/decoder


def test_old_checkpoint_defaults_to_legacy_and_migration_is_explicit(tmp_path):
    torch.manual_seed(7)
    m=MAVTGrainPrior(tiny_config()).eval()
    cfg=dataclasses.asdict(m.cfg)
    for k in ('readout_version','query_version','render_profile','innovation_semantics'): cfg.pop(k)
    p=tmp_path/'old.pt'
    torch.save(dict(cfg=cfg,model=m.state_dict()),p)
    loaded,c,_=load_grain_prior(p)
    assert c.readout_version=='legacy'
    x=torch.rand(1,3,64,64)*2-1
    S=m.encode_source(x)
    for k,v in m.decode(S).items(): assert torch.equal(v,loaded.decode(S)[k])
    with pytest.raises(ValueError): load_grain_prior(p,readout_version='shared_ray_v1',strict=False)
