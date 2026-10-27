import json
from pathlib import Path
import subprocess
import sys
import torch
import hashlib
import dataclasses
from mavt.grain_prior.model import MAVTGrainPrior
from mavt.grain_prior.config import tiny_config


def test_scene_runner_dry_run_saves_schedule_and_both_learning_rates(tmp_path):
    root=Path(__file__).resolve().parents[2]
    out=tmp_path/'scene_run'
    subprocess.run([sys.executable,str(root/'scripts/train_query_scene.py'),'--dry-run','--stage','finetune',
                    '--steps','2','--warmup','0','--rays','4','--out',str(out)],check=True)
    log=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    assert len(log)==2 and log[-1]['finite_gradients']
    assert log[-1]['lr_reader']==1e-5 and abs(log[-1]['lr_encoder']-1e-6)<1e-12
    ck=torch.load(out/'step-000002.pt',weights_only=False)
    assert ck['step']==2 and ck['schedule']['kind']=='cosine'
    assert ck['protocol']=='shared-query-pilot-v1' and ck['optimizer']
    assert ck['cfg']['innovation_semantics']=='disabled'


def test_scene_runner_accepts_legacy_control_with_audited_bundle(tmp_path):
    m=MAVTGrainPrior(tiny_config())
    ckpt=tmp_path/'legacy.pt'
    torch.save(dict(cfg=dataclasses.asdict(m.cfg),model=m.state_dict()),ckpt)
    x=torch.rand(1,3,64,64)*2-1
    k=torch.tensor([[1.,1.,.5,.5]])
    bundle=tmp_path/'scene.pt'
    torch.save(dict(source=x,source_kappa=k,target=x.flip(-1),target_depth=torch.ones(1,64,64),
                    target_kappa=k,target_R=torch.eye(3)[None],target_t=torch.zeros(1,3)),bundle)
    row=dict(scene_id='scene',group_id='asset',split='train',source_ids=['a'],target_ids=['b'],
             tensor_file='scene.pt',sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps(row)+'\n')
    script=Path(__file__).resolve().parents[2]/'scripts/train_query_scene.py'
    subprocess.run([sys.executable,str(script),'--manifest',str(manifest),'--checkpoint',str(ckpt),
                    '--readout','legacy','--steps','1','--warmup','0','--rays','4','--device','cpu',
                    '--out',str(tmp_path/'control')],check=True)
