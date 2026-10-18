"""Frozen synthetic reappearance diagnostic, separate from natural training.

Ground truth is used only by the evaluator. Slot identity is bound once before
an observation gap; pose comparison fixes the arbitrary slot frame at that same
prefix. A timed-out/subset run cannot supply an ownership-collapse gate score.
"""
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from ..data import FactorialSceneConfig, _camera_sequence, _render_geometry, _colour, _source_seed
from ..observation import ObservationStep
from ..decoder import RayQuery
from .history import history_labels

SCHEMA='v5-synthetic-gap-v1'
GAPS=(2,4,8,16)
REQUIRED_SOURCES=200


def make_gap_scene(seed, index):
    """Render one 24-step, two-view source on CPU with exact rigid trajectories."""
    cfg=FactorialSceneConfig(frames=24,prefix_frames=4,views=2,observed_views=1,
                             height=64,width=64,max_objects=2,fov_degrees=50.)
    source_seed=_source_seed('dev',seed,index)
    rng=torch.Generator().manual_seed(source_seed)
    times=torch.arange(24,dtype=torch.float64)*cfg.dt
    phase=float(torch.rand((),generator=rng)*2*math.pi)
    intrinsics,cameras=_camera_sequence(cfg,times,phase)
    crossing=index%4==0
    centres=torch.tensor([[5.1,-.97,.5],[6.6,.97,.5]])
    centres[:,0]+=(torch.rand(2,generator=rng)-.5)*.15
    sizes=torch.ones(2,3)
    yaw=torch.zeros(2)
    velocity=torch.tensor([[0.,.9,0.],[0.,-.9,0.]]) if crossing else torch.tensor([[0.,.12,0.],[0.,-.08,0.]])
    if not crossing:
        centres[:,1]=torch.tensor([-1.1,1.1])
        sizes=.85+.3*torch.rand(2,3,generator=rng)
        centres[:,2]=sizes[:,2]/2
    local,depth,instance,poses=_render_geometry(cfg,centres,sizes,yaw,velocity,
                                               torch.zeros(2),times,intrinsics,cameras)
    # Crossing objects share the exact material texture in their own coordinates.
    texture_ids=torch.where(instance>0,torch.ones_like(instance),instance) if crossing else instance
    rgb=_colour(local,texture_ids,0).reshape(24,2,64,64,3).permute(0,1,4,2,3).contiguous()
    anchor_inv=torch.linalg.inv(cameras[0,0])
    return {'schema':SCHEMA,'source_seed':source_seed,'source_index':index,
            'crossing':crossing,'same_appearance':crossing,
            'crossing_axis':anchor_inv[:3,:3]@torch.tensor([0.,1.,0.]),
            'rgb':rgb,'depth':depth.reshape(24,2,1,64,64),
            'depth_valid':instance.reshape(24,2,1,64,64)>=0,
            'instance':instance.reshape(24,2,64,64),
            'object_poses':anchor_inv@poses,'camera_c2w':anchor_inv@cameras,
            'intrinsics':intrinsics,'times':times}


def _hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def prepare_gap_suite(directory, *, seed=17, source_count=REQUIRED_SOURCES):
    """CPU-only preparation. Existing artifacts are verified, never overwritten."""
    if source_count<1:
        raise ValueError('source_count must be positive')
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('schema')!=SCHEMA or manifest.get('seed')!=seed or len(manifest['sources'])!=source_count:
            raise ValueError('immutable diagnostic suite contract mismatch')
        for row in manifest['sources']:
            if _hash(directory/row['file'])!=row['sha256']:
                raise ValueError('diagnostic source hash mismatch')
        return manifest
    rows=[]
    for index in range(source_count):
        path=directory/f'source-{index:04d}.pt'
        if path.exists():
            sample=torch.load(path,map_location='cpu',weights_only=True)
            if sample.get('schema')!=SCHEMA or sample.get('source_seed')!=_source_seed('dev',seed,index):
                raise ValueError('existing diagnostic source contract mismatch')
        else:
            sample=make_gap_scene(seed,index)
            with path.open('xb') as handle:
                torch.save(sample,handle)
        rows.append({'file':path.name,'source_index':index,'source_seed':sample['source_seed'],
                     'crossing':sample['crossing'],'sha256':_hash(path)})
    manifest={'schema':SCHEMA,'seed':seed,'sources':rows,'frames':24,'cameras':2,
              'heldout_camera':1,'height':64,'width':64,'gaps':list(GAPS),
              'required_sources':REQUIRED_SOURCES,'provenance':'exact_synthetic_rigid_renderer',
              'assignment':'pixel_weighted_patch_identity_bound_at_step_3',
              'pose':'slot_frame_aligned_at_step_3_no_future_alignment'}
    with manifest_path.open('x') as handle:
        json.dump(manifest,handle,sort_keys=True,indent=2)
    return manifest


def _votes(instance, ownership, patch_size, slots):
    """GT foreground pixel votes per observed patch; no centre-pixel aliasing."""
    predictions=ownership.argmax(-1)
    weights=torch.stack([F.avg_pool2d((instance==obj).float()[:,None],patch_size,
                                      ceil_mode=True).flatten(1)*patch_size**2 for obj in (1,2)],1)
    counts=weights.new_zeros(len(instance),2,slots+1)
    counts.scatter_add_(2,predictions[:,None].expand(-1,2,-1),weights)
    return counts[...,:slots],weights


def _bind_slots(counts):
    if counts.shape[-1]<2:
        raise ValueError('gap identity diagnostic requires at least two slots')
    choices=torch.tensor(list(itertools.permutations(range(counts.shape[-1]),2)),device=counts.device)
    score=counts[:,0,choices[:,0]]+counts[:,1,choices[:,1]]
    mapping=choices[score.argmax(-1)]
    mass=counts.gather(2,mapping[...,None]).squeeze(-1)
    return torch.where(mass>0,mapping,-torch.ones_like(mapping))


class _BudgetExpired(Exception):
    pass


@torch.no_grad()
def _run_gap_batch(model,samples,gap,seed, *, deadline=None):
    if gap not in GAPS:
        raise ValueError('unsupported diagnostic gap')
    device=next(model.parameters()).device
    data={k:torch.stack([s[k] for s in samples]).to(device) for k in
          ('rgb','depth','depth_valid','instance','object_poses','camera_c2w','intrinsics','times')}
    b=len(samples); cfg=model.config; reappearance=4+gap
    state=model.initial_state(b,time=data['times'][:,0])
    history=[]; stats=defaultdict(float)
    mapping=alignment=None
    for frame in range(24):
        if deadline is not None and time.monotonic()>=deadline:
            raise _BudgetExpired
        observed=not 4<=frame<reappearance
        if not observed:
            state=model.predict(state,data['times'][:,frame]-state.t)
            continue
        # Match the natural-data encoder range; render targets remain in [0,1].
        obs=ObservationStep(images=data['rgb'][:,frame,:1,None]*2-1,time=data['times'][:,frame],
            intrinsics=data['intrinsics'][:,frame,:1],camera_c2w=data['camera_c2w'][:,frame,:1],
            camera_valid=torch.ones(b,1,device=device,dtype=torch.bool),
            metric_valid=torch.ones(b,1,device=device,dtype=torch.bool),
            depth=data['depth'][:,frame,:1],depth_valid=data['depth_valid'][:,frame,:1])
        state,diag=model.observe(state,obs,return_diagnostics=True)
        history.append({'frame':frame,'input_mask':torch.ones(b,1,1,64,64,device=device,dtype=torch.bool)})
        if frame not in (3,reappearance):
            continue
        counts,weights=_votes(data['instance'][:,frame,0],diag['content_ownership'],cfg.patch_size,cfg.slots)
        if frame==3:
            mapping=_bind_slots(counts)
            controls=model.decoder.controls(state.d)
            selected=controls[torch.arange(b,device=device)[:,None],mapping.clamp_min(0)]
            alignment=torch.linalg.inv(data['object_poses'][:,frame])@selected
            continue
        valid_binding=mapping>=0
        correct=counts.gather(2,mapping.clamp_min(0)[...,None]).squeeze(-1)*valid_binding
        stats['assignment_correct']+=float(correct.sum())
        stats['assignment_count']+=float(weights.sum())
        stats['assignment_valid_cases']+=int((weights.sum((1,2))>0).sum())
        predicted=model.decoder.controls(state.d)[torch.arange(b,device=device)[:,None],mapping.clamp_min(0)]
        target=data['object_poses'][:,frame]@alignment
        error=(predicted[...,:3,3]-target[...,:3,3]).norm(dim=-1)
        relative=predicted[...,:3,:3].transpose(-1,-2)@target[...,:3,:3]
        angle=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1).acos()
        stats['pose_translation_sum']+=float(error[valid_binding].sum())
        stats['pose_rotation_sum']+=float(angle[valid_binding].sum())
        stats['pose_count']+=int(valid_binding.sum())
        # Fixed spatial rays in both views, chosen without consulting GT labels.
        yy,xx=torch.meshgrid(torch.tensor([8.,24.,40.,56.],device=device),
                             torch.tensor([8.,24.,40.,56.],device=device),indexing='ij')
        pixels=torch.stack((xx.flatten()+.5,yy.flatten()+.5),-1)[None].expand(b,-1,-1)
        homogeneous=torch.cat((pixels,torch.ones_like(pixels[...,:1])),-1)
        clips=[]
        for i,sample in enumerate(samples):
            obj=torch.eye(4,device=device).repeat(24,3,1,1)
            obj[:,1:]=data['object_poses'][i]
            clips.append({'rgb':data['rgb'][i],'depth':data['depth'][i],
                'depth_valid':data['depth_valid'][i],'input_depth':data['depth'][i],
                'input_depth_valid':data['depth_valid'][i],'depth_provenance':'measured',
                'instance_map':data['instance'][i],'object_to_world':obj,
                'object_pose_valid':torch.ones(24,3,device=device,dtype=torch.bool),
                'rigid_correspondence_trusted':True,'intrinsics':data['intrinsics'][i],
                'camera_c2w':data['camera_c2w'][i],'camera_valid':torch.ones(24,2,device=device,dtype=torch.bool)})
        for view in (0,1):
            camera=data['camera_c2w'][:,frame,view]
            ray=torch.linalg.solve(data['intrinsics'][:,frame,view],homogeneous.transpose(1,2)).transpose(1,2)
            direction=F.normalize(torch.einsum('bij,brj->bri',camera[:,:3,:3],ray),dim=-1)
            query=RayQuery(camera[:,:3,3][:,None].expand_as(direction),direction,.1,30.,state.t)
            output=model.read(state,query)
            x=pixels[0,:,0].long(); y=pixels[0,:,1].long()
            target_rgb=data['rgb'][:,frame,view,:,y,x].transpose(1,2)
            stats['rgb_abs_sum']+=float((output['rgb']-target_rgb).abs().mean(-1).sum())
            stats['rgb_rays']+=b*len(x)
            labels=history_labels(clips,frame,pixels,history,
                                  current_index=len(history)-1 if view==0 else None,view=view)
            known=labels>=0
            stats['status_correct']+=int((output['status_logits'].argmax(-1)[known]==labels[known]).sum())
            stats['status_labeled_rays']+=int(known.sum())
            for label in range(3): stats[f'status_{label}_count']+=int((labels==label).sum())
    return state,dict(stats)


@torch.no_grad()
def evaluate_gap_diagnostic(model,seed=17,seconds=60, *, suite_dir=None,batch_size=32):
    """Evaluate complete batches until deadline; missing coverage is not a score."""
    if seconds<0 or batch_size<1:
        raise ValueError('nonnegative budget and positive batch size required')
    started=time.monotonic(); deadline=started+seconds
    report={'schema':SCHEMA,'complete':False,'assignment_accuracy':None,
            'required_cases':REQUIRED_SOURCES*len(GAPS),'evaluated_cases':0,'gaps':{}}
    if suite_dir is None:
        return dict(report,status='requires_cpu_prepared_suite',elapsed_seconds=time.monotonic()-started)
    directory=Path(suite_dir); manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('seed')!=seed or manifest.get('gaps')!=list(GAPS):
        raise ValueError('frozen diagnostic suite contract mismatch')
    rows=manifest['sources']
    if len({r['source_seed'] for r in rows})!=len(rows):
        raise ValueError('diagnostic sources must be independent')
    report['suite_sha256']=_hash(directory/'manifest.json')
    totals=defaultdict(float); was_training=model.training; model.eval()
    evaluated_crossing=0
    try:
        for start in range(0,len(rows),batch_size):
            if time.monotonic()>=deadline: break
            selected=rows[start:start+batch_size]; samples=[]
            for row in selected:
                path=directory/row['file']
                if _hash(path)!=row['sha256']: raise ValueError('diagnostic source hash mismatch')
                samples.append(torch.load(path,map_location='cpu',weights_only=True))
            for gap in GAPS:
                try:
                    _,metrics=_run_gap_batch(model,samples,gap,seed,deadline=deadline)
                except _BudgetExpired:
                    break
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError('nonfinite diagnostic metric; coverage cannot certify a score')
                for key,value in metrics.items(): totals[key]+=value
                report['evaluated_cases']+=len(samples)
                evaluated_crossing+=sum(s['crossing'] for s in samples)
                report['gaps'][str(gap)]=report['gaps'].get(str(gap),0)+len(samples)
            if time.monotonic()>=deadline: break
    finally:
        model.train(was_training)
    report.update({k:int(v) for k,v in totals.items() if k.endswith(('_count','_cases','_rays'))})
    report['partial_assignment_accuracy']=(totals['assignment_correct']/totals['assignment_count'] if totals['assignment_count'] else None)
    if totals['rgb_rays']: report['reappearance_rgb_mae']=totals['rgb_abs_sum']/totals['rgb_rays']
    if totals['pose_count']:
        report['reappearance_pose_translation']=totals['pose_translation_sum']/totals['pose_count']
        report['reappearance_pose_rotation_radians']=totals['pose_rotation_sum']/totals['pose_count']
    if totals['status_labeled_rays']: report['status_accuracy']=totals['status_correct']/totals['status_labeled_rays']
    complete=(len(rows)>=REQUIRED_SOURCES and report['evaluated_cases']==len(rows)*len(GAPS)
              and evaluated_crossing>=.25*report['evaluated_cases']
              and totals['assignment_valid_cases']==report['evaluated_cases']
              and totals['rgb_rays']>0 and totals['status_labeled_rays']>0)
    report.update(complete=bool(complete),crossing_cases=evaluated_crossing,
                  status='complete' if complete else 'incomplete_coverage',
                  elapsed_seconds=time.monotonic()-started)
    if complete: report['assignment_accuracy']=report['partial_assignment_accuracy']
    return report
