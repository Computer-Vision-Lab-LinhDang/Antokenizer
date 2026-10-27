"""Single-device posed-scene pilot with separate encoder/reader LR groups.

Manifest JSONL rows use protocol.validate_manifest fields and tensor_file + sha256.
Each torch bundle (weights_only load) contains source[B,3,H,W], source_kappa[B,4],
target[B,3,H,W] in [-1,1], target_depth[B,H,W], and target_kappa/R/t tensors.
Camera transforms are canonical-to-target. Only train rows are admitted.
No datasets are downloaded. Run GPU work through the site's compute scheduler.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mavt.grain_prior.config import tiny_config
from mavt.grain_prior.model import MAVTGrainPrior, load_grain_prior
from mavt.grain_prior.pilot import configure_pilot, warm_start_query_encoder
from mavt.grain_prior.protocol import PROTOCOL_VERSION, validate_manifest
from mavt.grain_prior.query import Camera
from mavt.grain_prior.scene_pilot import scene_step
from mavt.grain_prior.training_schedule import optimizer_groups, set_learning_rates, lr_multiplier


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--init',help='legacy checkpoint: copy encoder and initialize a fresh reader')
    p.add_argument('--checkpoint',help='trained query checkpoint: start a new stage, reset optimizer')
    p.add_argument('--readout',choices=['legacy','shared_ray_v1','shared_field_v1'],default='shared_field_v1')
    p.add_argument('--stage',choices=['frozen','finetune'],default='frozen')
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--warmup',type=int,default=200)
    p.add_argument('--reader-lr',type=float,default=1e-4)
    p.add_argument('--encoder-lr',type=float,default=1e-5)
    p.add_argument('--wd',type=float,default=.05)
    p.add_argument('--schedule',choices=['constant','cosine'],default='cosine')
    p.add_argument('--min-lr-ratio',type=float,default=.1)
    p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--rays',type=int,default=512)
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda')
    p.add_argument('--bf16',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.init and a.checkpoint: p.error('choose --init or --checkpoint')
    if min(a.accumulation,a.save_every,a.grad_clip)<=0: p.error('invalid accumulation/save/clip')
    lr_multiplier(1,a.steps,a.warmup,a.schedule,a.min_lr_ratio)
    torch.manual_seed(a.seed)
    device=torch.device('cpu' if a.dry_run else a.device)
    receipt=None
    if a.dry_run:
        cfg=tiny_config(readout_version=a.readout,innovation_semantics='disabled',render_samples=4)
        model=MAVTGrainPrior(cfg)
        rows=[dict(scene_id='synthetic-smoke',split='train')]
        digest='synthetic-smoke-only'
    else:
        if not a.manifest or not (a.init or a.checkpoint):
            p.error('real pilot requires an audited --manifest and --init or --checkpoint')
        data=a.manifest.read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        rows=[json.loads(line) for line in data.decode().splitlines() if line.strip()]
        validate_manifest(rows)
        rows=[r for r in rows if r['split']=='train']
        if not rows: p.error('manifest contains no training scenes')
        if a.init:
            model,receipt=warm_start_query_encoder(a.init,a.readout)
        else:
            model,_,_=load_grain_prior(a.checkpoint)
            if model.cfg.readout_version!=a.readout: p.error('checkpoint readout mismatch')
        if model.cfg.innovation_semantics!='disabled': p.error('pilot requires disabled innovations')
    cfg=model.cfg
    if cfg.w_lpips!=0: p.error('scattered-ray protocol requires LPIPS disabled')
    model.to(device)
    configure_pilot(model,frozen_encoder=a.stage=='frozen')
    optimizer=torch.optim.AdamW(optimizer_groups(model,a.reader_lr,a.encoder_lr,a.wd),betas=(.9,.95),eps=1e-8)
    # Exclusive directory: a new stage cannot overwrite a previous experiment.
    a.out.mkdir(parents=True,exist_ok=False)
    schedule=dict(kind=a.schedule,total_steps=a.steps,warmup=a.warmup,min_ratio=a.min_lr_ratio)
    meta=dict(protocol=PROTOCOL_VERSION,cfg=dataclasses.asdict(cfg),args=vars(a),schedule=schedule,
              manifest_sha256=digest,warm_start=receipt,scene_count=len(rows),metric_claim='unmeasured')
    (a.out/'run.json').write_text(json.dumps(meta,default=str,indent=2))
    cache={}

    def batch_for(index):
        row=rows[index%len(rows)]
        if a.dry_run:
            gen=torch.Generator().manual_seed(a.seed+index)
            x=torch.rand(1,3,cfg.image_size,cfg.image_size,generator=gen)*2-1
            k=torch.tensor([[1.,1.,.5,.5]])
            return x,k,x.flip(-1),Camera(k,torch.eye(3)[None],torch.zeros(1,3)),torch.ones(1,cfg.image_size,cfg.image_size)*2
        path=(a.manifest.parent/row['tensor_file']).resolve()
        # Check each bundle once; no image tensors are kept as a hidden decoder cache.
        if str(path) not in cache:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
            if h.hexdigest()!=row.get('sha256'): raise ValueError('tensor bundle checksum mismatch')
            cache[str(path)]=True
        data=torch.load(path,map_location=device,weights_only=True)
        required={'source','source_kappa','target','target_depth','target_kappa','target_R','target_t'}
        if set(data)!=required: raise ValueError('unexpected/missing scene bundle fields')
        cam=Camera(data['target_kappa'],data['target_R'],data['target_t'])
        return data['source'],data['source_kappa'],data['target'],cam,data['target_depth']

    durations=[]
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    with (a.out/'metrics.jsonl').open('x') as log:
        for step in range(1,a.steps+1):
            set_learning_rates(optimizer,step,a.steps,a.warmup,a.schedule,a.min_lr_ratio)
            optimizer.zero_grad(set_to_none=True)
            if device.type=='cuda': torch.cuda.synchronize(device)
            start=time.perf_counter()
            terms={}
            for micro in range(a.accumulation):
                # Stable source draw order across arms; input bundles define real scene microbatches.
                draw=(step-1)*a.accumulation+micro
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=a.bf16 and device.type=='cuda'):
                    result=scene_step(model,*batch_for(draw),rays=a.rays,seed=a.seed*1000003+draw)
                if not torch.isfinite(result['loss']): raise FloatingPointError('nonfinite pilot loss')
                (result['loss']/a.accumulation).backward()
                for k,v in result['terms'].items(): terms[k]=terms.get(k,0.)+float(v.detach())/a.accumulation
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],a.grad_clip,error_if_nonfinite=True)
            optimizer.step()
            if device.type=='cuda': torch.cuda.synchronize(device)
            duration=time.perf_counter()-start
            if step>a.warmup: durations.append(duration)
            rates={f"lr_{g['role']}":g['lr'] for g in optimizer.param_groups}
            rec=dict(step=step,terms=terms,grad_norm=float(norm),finite_gradients=True,seconds=duration,**rates)
            log.write(json.dumps(rec)+'\n');log.flush()
            if step==1 or step%25==0 or step==a.steps: print(json.dumps(rec),flush=True)
            if step%a.save_every==0 or step==a.steps:
                checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),cfg=dataclasses.asdict(cfg),
                                step=step,schedule=schedule,protocol=PROTOCOL_VERSION,manifest_sha256=digest,
                                torch_rng_state=torch.get_rng_state(),stage=a.stage,seed=a.seed)
                with (a.out/f'step-{step:06d}.pt').open('xb') as f: torch.save(checkpoint,f)
    summary=dict(post_warmup_updates=len(durations),seconds_per_update=sum(durations)/max(1,len(durations)),
                 peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                 dry_run=a.dry_run,consistency_gate='not_evaluated')
    (a.out/'profile.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__': main()
