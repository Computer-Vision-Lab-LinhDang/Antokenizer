import json
from pathlib import Path
import subprocess
import sys
import torch


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
