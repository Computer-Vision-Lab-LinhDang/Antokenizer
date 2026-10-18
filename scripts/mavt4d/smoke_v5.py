"""Correctness smoke, separate from the immutable full training budget.

Synthetic fixtures exercise mechanisms only; results are not research quality
scores. Default architecture/image size matches the full runner.
"""
import argparse
import json
from pathlib import Path
import time
import torch

from mavt4d.object_state.v5.config import V5Config
from mavt4d.object_state.v5.model import V5Model
from mavt4d.object_state.v5.state import V5State
from mavt4d.object_state.v5.training import compute_loss, observe_prefix
from mavt4d.object_state.v5.runner import _optimizer, _save, _restore_rng, _atomic_json
from mavt4d.object_state.runner import RunLogger, RuntimeConfig


def fixture(height, width, *, geometry):
    frames = 22
    # Smooth changes over a stationary plane; no claim of a natural-data score.
    x = torch.linspace(0,1,width)[None,:].expand(height,width)
    y = torch.linspace(0,1,height)[:,None].expand(height,width)
    image = torch.stack((x,y,torch.full_like(x,.5)))
    rgb = torch.stack([(image + .02*torch.sin(torch.tensor(float(t)))).clamp(0,1) for t in range(frames)])[:,None]
    valid = torch.ones(frames,1,1,height,width,dtype=torch.bool)
    clip = {'source':'arkitscenes' if geometry else 'openvid','group_id':'smoke-fixture',
            'rgb':rgb, 'rgb_valid':valid, 'times':torch.arange(frames).double()*.25,
            'time_valid':True,'timestamp_provenance':'fixture',
            'plan':{'prefix':[1,3,5,7,9],'observed':[True,True,False,False,True],
                    'future':[11,13,15,17,19,21], 'profile':'G-fixture'}}
    if geometry:
        k = torch.tensor([[width,0,width/2],[0,width,height/2],[0,0,1.]])
        depth = torch.full((frames,1,1,height,width),2.)
        # Keep patch centers. Alternate the held-out columns over time so a
        # current target can have trusted depth evidence in an earlier frame.
        # Input and target remain disjoint at every individual time step.
        input_valid=valid.clone()
        for t in range(frames): input_valid[t,...,1::2,(t//2)%2::2]=False
        clip.update(intrinsics=k.expand(frames,1,3,3).clone(),
                    camera_c2w=torch.eye(4).expand(frames,1,4,4).clone(),
                    camera_valid=torch.ones(frames,1,dtype=torch.bool),
                    metric_valid=torch.ones(frames,1,dtype=torch.bool),
                    depth=depth, depth_valid=valid & ~input_valid,
                    input_depth=torch.where(input_valid,depth,0), input_depth_valid=input_valid,
                    depth_provenance='measured',scene_static=True)
    return clip


def resolve_model_config(path=None, *, tiny=False):
    if tiny and path is not None:
        raise ValueError("--tiny cannot override an explicit --model-config")
    if path is not None:
        payload = json.loads(Path(path).read_text())
        return V5Config(**payload.get("model", payload))
    if tiny:
        return V5Config(slots=2,content_tokens=2,dynamic_tokens=2,width=16,evidence_width=24,
            spatial_blocks=1,spatial_heads=3,window_size=2,relay_tokens=2,samples_per_ray=4,query_chunk_size=4)
    return V5Config()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--tiny',action='store_true',help='CPU contract fixture, never a capacity estimate')
    parser.add_argument('--model-config',type=Path,help='Exact model dictionary or full run config; incompatible with --tiny')
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--project',default='mavt-v5')
    parser.add_argument('--restore',type=Path,help='Fresh-process state/rollout verification only')
    args=parser.parse_args()
    requested_config=resolve_model_config(args.model_config,tiny=args.tiny)
    torch.set_num_threads(2); torch.manual_seed(17)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.restore:
        saved=torch.load(args.restore,map_location=args.device,weights_only=False)
        if args.model_config is not None and requested_config.to_dict()!=saved['config']:
            raise ValueError('restore checkpoint differs from explicit model config')
        model=V5Model(V5Config(**saved['config'])).to(args.device).eval()
        model.load_state_dict(saved['model'])
        state=V5State.from_payload(saved['state'],model.config)
        with torch.no_grad(): result=model.predict(state,.5)
        for key,value in result.tensor_dict().items():
            torch.testing.assert_close(value,saved['predicted'][key],rtol=1e-5,atol=1e-6)
        _atomic_json(args.output/'fresh-process.json',{'pass':True,'schema':5})
        print('fresh-process restore PASS',flush=True)
        return
    if (args.output/'summary.json').exists(): raise FileExistsError('use a fresh smoke output directory')
    cfg=requested_config
    model=V5Model(cfg).to(args.device); optimizer=_optimizer(model)
    logger=RunLogger(args.output,config={'purpose':'v5-correctness-smoke','tiny':args.tiny,'model':cfg.to_dict()},
                     runtime=RuntimeConfig(project=args.project,run_name='v5-integration-smoke'),enabled=args.wandb)
    exit_code=1; records=[]
    try:
        for step,geometry in enumerate((False,True),1):
            started=time.monotonic(); optimizer.zero_grad(set_to_none=True)
            h,w=(32,32) if args.tiny else (192,256)
            clip=fixture(h,w,geometry=geometry)
            loss,metrics=compute_loss(model,[clip],step=4000,rays=4 if args.tiny else 32,seed=20,bridge=True)
            loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('smoke nonfinite loss/gradient')
            def module_grad_norm(module):
                return sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5
            content_grad=module_grad_norm(model.resampler.c_update)
            reliability_grad=module_grad_norm(model.resampler.reliability)
            if content_grad == 0 or (geometry and reliability_grad == 0):
                raise AssertionError('smoke must exercise content and supervised reliability gradients')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            model.eval()
            with torch.no_grad():
                state,_,diagnostics=observe_prefix(model,[clip],seed=20)
                measured_tokens=sum(int(d.get('geometry_measured',torch.tensor(0)).sum()) for d in diagnostics)
                reliability_labels=sum(int(d.get('reliability_valid',torch.tensor(0)).sum()) for d in diagnostics)
                initialized_slots=int(state.content_initialized.sum())
                if geometry and measured_tokens == 0:
                    raise AssertionError('geometry smoke must actually admit measured patch centers')
                if initialized_slots == 0 or (geometry and reliability_labels == 0):
                    raise AssertionError('smoke must initialize content and label reliability')
                predicted=model.predict(state,.5)
                restored=V5State.from_payload(state.to_payload(),cfg)
                again=model.predict(restored,.5)
                for key,value in predicted.tensor_dict().items():
                    torch.testing.assert_close(value,again.tensor_dict()[key])
                torch.testing.assert_close(state.c,predicted.c,rtol=0,atol=0)
                state.validate(cfg)
            model.train()
            if torch.device(args.device).type=='cuda': torch.cuda.synchronize()
            record={**{key:value for key,value in metrics.items() if not key.startswith('_')},'loss':float(loss.detach()),'gradient_norm':float(norm),
                    'geometry':int(geometry),'seconds':time.monotonic()-started,
                    'measured_geometry_tokens':measured_tokens,
                    'reliability_labeled_tokens':reliability_labels,'initialized_slots':initialized_slots,
                    'content_gradient_norm':content_grad,'reliability_gradient_norm':reliability_grad,
                    'state_bytes':state.total_nbytes,'peak_reserved':torch.cuda.max_memory_reserved() if torch.cuda.is_available() and 'cuda' in args.device else 0}
            records.append(record); logger.log(record,step=step)
            _save(args.output/'last.pt',model,optimizer,lock={'model':cfg.to_dict(),'purpose':'smoke'},step=step,run_id=logger.run_id,capacities={'fixture':1})
        torch.save({'config':cfg.to_dict(),'model':model.state_dict(),'state':state.to_payload(),
                    'predicted':predicted.tensor_dict()},args.output/'state-restore.pt')
        # Check that an Adam/RNG boundary restores before declaring the smoke done.
        saved=torch.load(args.output/'last.pt',map_location=args.device,weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); _restore_rng(saved['rng'])
        _atomic_json(args.output/'summary.json',{'pass':True,'purpose':'correctness-only',
                    'tiny':args.tiny,'records':records,'wandb_url':logger.url})
        exit_code=0
    finally: logger.finish(exit_code=exit_code)
    print(json.dumps({'pass':True,'wandb_url':logger.url}),flush=True)


if __name__=='__main__': main()
