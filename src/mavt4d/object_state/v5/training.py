"""Causal natural-data objective: measured masks, gap prediction and full batches."""
from __future__ import annotations
from dataclasses import replace
import math
import random

import torch
from torch.nn import functional as F

from ..observation import ObservationStep
from ..decoder import make_ray_query


def masked_mean(value, valid):
    valid = valid.to(device=value.device, dtype=torch.bool)
    finite = torch.isfinite(value)
    valid = valid & finite
    return torch.where(valid, value, 0).sum() / valid.sum().clamp_min(1)


def readout_objective(prediction, targets):
    """Unknown labels may mask RGB; absent labels never imply unknown/background."""
    valid = targets["rgb_valid"]
    status = targets.get("status", torch.full_like(valid, -1, dtype=torch.long))
    rgb_valid = valid & (status != 2)
    # Normalize each example before the logical-source average.
    err = (prediction["rgb"] - targets["rgb"]).abs().mean(-1)
    rgb = torch.stack([masked_mean(e, m) for e, m in zip(err, rgb_valid)]).mean()
    zero = prediction["rgb"].sum() * 0
    depth, status_loss = zero, zero
    if "range" in targets and "range" in prediction:
        mask = targets["depth_valid"] & (targets["range"] > 0)
        depth_error = (prediction["range"].clamp_min(1e-3).log() - targets["range"].clamp_min(1e-3).log()).abs()
        depth = torch.stack([masked_mean(e, m) for e, m in zip(depth_error, mask)]).mean()
    if "status_logits" in prediction:
        labels = status.clamp_min(0)
        ce = F.cross_entropy(prediction["status_logits"].transpose(1, 2), labels, reduction="none")
        # Each example is balanced independently, so splitting a logical batch
        # into physical microbatches cannot change its objective.
        examples = []
        for error, sample_status in zip(ce, status):
            terms = [masked_mean(error, sample_status == k) for k in range(3) if (sample_status == k).any()]
            examples.append(torch.stack(terms).mean() if terms else error.sum() * 0)
        status_loss = torch.stack(examples).mean()
    total = rgb + .5 * depth + .1 * status_loss
    return total, {"rgb": rgb.detach(), "depth": depth.detach(), "status": status_loss.detach(),
                   "status_label_coverage": (status >= 0).float().mean().detach(),
                   **{f"status_class_{k}_coverage": (status == k).float().mean().detach() for k in range(3)}}


def learning_rate(step, total=20000):
    if step < 1000:
        return 3e-4 * (step + 1) / 1000
    fraction = min(1., (step - 1000) / max(1, total - 1001))
    return 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * fraction))


def curriculum(step, seed):
    rng = random.Random(seed)
    ramp = min(1., max(0., (step - 1000) / 3000))
    u = rng.random()
    profile = "G" if u < .3 * ramp else "L" if u < .6 * ramp else "S"
    horizon = 1 if step < 1000 else 3 if step < 2500 else 6
    future_weight = .1 + .2 * step / 1000 if step < 1000 else .3 + .7 * min(1., (step - 1000) / 3000)
    return profile, horizon, future_weight


def clip_signature(clip):
    p = clip['plan']
    return (clip['source'], tuple(clip['rgb'].shape[1:]), tuple(p['prefix']),
            tuple(p['observed']), tuple(p['future']), bool(clip.get('time_valid',True)),
            tuple(key in clip for key in ('intrinsics','camera_c2w','camera_valid','metric_valid','depth','depth_valid',
                'input_depth','input_depth_valid','slot_pose','slot_pose_valid')),
            clip.get('depth_provenance','missing'),bool(clip.get('slot_assignment_trusted',False)))


def _frame(clips, key, frame, device):
    return torch.stack([c[key][frame] for c in clips]).to(device)


def _camera_mask(clips,frame,device):
    return torch.stack([c.get('camera_valid',torch.zeros(c['rgb'].shape[:2],dtype=torch.bool))[frame]
                        for c in clips]).to(device)


def _observation(clips,frame,device,*,metadata_drop=False,mask=None):
    from dataclasses import fields
    start=max(0,frame-1)
    images=torch.stack([c['rgb'][start:frame+1,:1] for c in clips]).permute(0,2,1,3,4,5).to(device)
    valid=torch.stack([c['rgb_valid'][start:frame+1,:1] for c in clips]).permute(0,2,1,3,4,5).to(device)
    if mask is not None:
        valid &= mask.to(device)
    time=_frame(clips,'times',frame,device)
    kwargs={}
    if not metadata_drop and all('intrinsics' in c and 'camera_c2w' in c for c in clips):
        camera_valid=_camera_mask(clips,frame,device)[:,:1]
        kwargs.update(intrinsics=_frame(clips,'intrinsics',frame,device)[:,:1],
                      camera_c2w=_frame(clips,'camera_c2w',frame,device)[:,:1],geometry_valid=camera_valid)
        # Optional metric flag is separate from merely knowing camera geometry.
        if 'metric_valid' in {f.name for f in fields(ObservationStep)}:
            kwargs['metric_valid']=torch.stack([c.get('metric_valid',torch.zeros(c['rgb'].shape[:2],dtype=torch.bool))[frame,:1]
                                               for c in clips]).to(device)
        if all('input_depth' in c for c in clips):
            depth=_frame(clips,'input_depth',frame,device)[:,:1]
            depth_valid=torch.stack([c.get('input_depth_valid',torch.zeros_like(c['input_depth'],dtype=torch.bool))[frame,:1]
                                     for c in clips]).to(device)
            trusted=torch.tensor([c.get('depth_provenance') in ('measured','reconstructed','sensor','lidar','metric_sensor') for c in clips],device=device)
            kwargs.update(depth=depth,depth_valid=depth_valid & valid[:,:,-1] & camera_valid[...,None,None,None] & trusted[:,None,None,None,None])
    return ObservationStep(images=images*2-1,time=time,valid=valid,**kwargs)


def _prefix_steps(model,clips,*,seed,metadata_drop=False,tube_mask=None):
    device=next(model.parameters()).device
    state=model.initial_state(len(clips),device=device,time=_frame(clips,'times',0,device))
    history=[]
    for ordinal,(frame,observed) in enumerate(zip(clips[0]['plan']['prefix'],clips[0]['plan']['observed'])):
        obs=_observation(clips,frame,device,metadata_drop=metadata_drop,
                         mask=None if tube_mask is None else tube_mask.get(frame))
        if not all(c.get('time_valid',True) for c in clips):
            # Frame order is not elapsed seconds. Preserve the sequence clock
            # and run only observation correction; every dynamics delta is zero.
            obs=replace(obs,time=state.t)
        previous=state
        if observed:
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                state,diagnostics=model.observe(state,obs,return_diagnostics=True)
            start=max(0,frame-1)
            # The encoder keeps a patch only when its endpoint coverage is
            # nonzero. Earlier-frame pixels in discarded patches are not inputs.
            endpoint=obs.valid[:,:,-1]
            batch,views,_,height,width=endpoint.shape
            patch=model.config.patch_size
            pad=(0,(-width)%patch,0,(-height)%patch)
            coverage=F.avg_pool2d(F.pad(endpoint.float().reshape(batch*views,1,height,width),pad),patch,patch)>0
            patch_admitted=coverage.repeat_interleave(patch,-2).repeat_interleave(patch,-1)[...,:height,:width].reshape_as(endpoint)
            for offset,admitted_frame in enumerate(range(start,frame+1)):
                # This also masks every sample whose entire endpoint is absent.
                admitted_mask=(obs.valid[:,:,offset] & patch_admitted).detach()
                record={'frame':admitted_frame,'input_mask':admitted_mask,
                        'admissions':[(frame,admitted_mask)],
                        'time':obs.time if admitted_frame==frame else (_frame(clips,'times',admitted_frame,device) if all(c.get('time_valid',True) for c in clips) else state.t)}
                if admitted_frame==frame:
                    record.update(camera=model.camera_parameters(state,view=0),diagnostics=diagnostics,diagnostic_mask=obs.valid[:,:,-1].detach())
                matches=[index for index,old in enumerate(history) if old['frame']==admitted_frame]
                if matches:
                    old=history[matches[0]]
                    record['input_mask']=record['input_mask']|old['input_mask']
                    record['admissions']=old.get('admissions',[(old['frame'],old['input_mask'])])+record['admissions']
                    if 'diagnostics' not in record and 'diagnostics' in old:
                        record.update(camera=old['camera'],diagnostics=old['diagnostics'],diagnostic_mask=old.get('diagnostic_mask',old['input_mask']))
                    history[matches[0]]=record
                else:
                    history.append(record)
        else:
            state=model.predict(state,obs.time-state.t)
            diagnostics={}
            obs=replace(obs,valid=torch.zeros_like(obs.valid))
        yield ordinal,frame,observed,previous,state,history,diagnostics,obs


def observe_prefix(model,clips,*,seed=17,metadata_drop=False,tube_mask=None):
    """Causal filtering API. Only RGB, input_* metadata, and prefix times enter it."""
    diagnostics=[]
    for _,_,_,_,state,history,diag,_ in _prefix_steps(model,clips,seed=seed,
            metadata_drop=metadata_drop,tube_mask=tube_mask):
        diagnostics.append(diag)
    return state,history,diagnostics


def _read_targets(model,state,clips,frame,*,rays,seed,observed_now=False,
                  input_mask=None,hidden_only=False,view=None,history=None,
                  metadata_drop=False,aligned=False):
    from .history import history_labels
    device,b=state.c.device,len(clips)
    image=_frame(clips,'rgb',frame,device)
    _,views,_,height,width=image.shape
    view=(1 if views>1 else 0) if view is None else view
    # Independent per-draw RNG keeps ray selection invariant to microbatch cuts.
    samples=[]
    for c in clips:
        import hashlib
        identity=str(c.get('group_id',c.get('group_hash',c.get('group',c.get('source','')))))
        offset=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'little')
        generator=torch.Generator(device=device).manual_seed(seed+offset)
        samples.append(torch.randint(height*width,(rays,),generator=generator,device=device))
    indices=torch.stack(samples)
    if hidden_only and input_mask is not None:
        for j in range(b):
            hidden=(~input_mask[j,0,-1,0]).flatten().nonzero().flatten()
            if len(hidden):
                identity=str(clips[j].get('group_id',clips[j].get('group_hash',clips[j].get('group',clips[j].get('source','')))))
                offset=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'little')
                generator=torch.Generator(device=device).manual_seed(seed+offset)
                indices[j]=hidden[torch.randint(len(hidden),(rays,),generator=generator,device=device)]
    y,x=indices//width,indices%width
    pixels=torch.stack((x.float()+.5,y.float()+.5),-1)
    estimated_k,estimated_camera=model.camera_parameters(state,view=min(view,model.config.camera_views-1))
    fallback_k,fallback_camera=model.camera_parameters(state,view=0)
    default_k=state.c.new_tensor([[max(height,width),0.,width/2],[0.,max(height,width),height/2],[0.,0.,1.]])
    fallback_k=torch.where(state.camera_valid[:,0,None,None],fallback_k,default_k)
    has_estimate=state.camera_valid[:,min(view,model.config.camera_views-1)]
    estimated_k=torch.where(has_estimate[:,None,None],estimated_k,fallback_k)
    estimated_camera=torch.where(has_estimate[:,None,None],estimated_camera,fallback_camera)
    known_camera=torch.zeros(b,dtype=torch.bool,device=device)
    k,camera=estimated_k,estimated_camera
    if not metadata_drop and all('intrinsics' in c for c in clips):
        known_camera=_camera_mask(clips,frame,device)[:,view]
        k=torch.where(known_camera[:,None,None],_frame(clips,'intrinsics',frame,device)[:,view],k)
        camera=torch.where(known_camera[:,None,None],_frame(clips,'camera_c2w',frame,device)[:,view],camera)
    near=.1
    far=torch.where(known_camera,80.,20.)[:,None]
    query=make_ray_query(k,camera,pixels,time=state.t,near=near,far=far)
    batch=torch.arange(b,device=device)[:,None]
    target={'rgb':image[:,view].permute(0,2,3,1)[batch,y,x],
            'rgb_valid':_frame(clips,'rgb_valid',frame,device)[:,view,0][batch,y,x],
            'status':torch.full((b,rays),-1,dtype=torch.long,device=device),
            'pixels':pixels,'view':view,'depth_provenance':[c.get('depth_provenance','missing') for c in clips]}
    if all('depth' in c and 'depth_valid' in c for c in clips) and not metadata_drop:
        z=_frame(clips,'depth',frame,device)[:,view,0][batch,y,x]
        homogeneous=torch.cat((pixels,torch.ones_like(pixels[...,:1])),-1)
        camera_ray=torch.einsum('bij,brj->bri',torch.linalg.inv(k),homogeneous)
        target['range']=z*camera_ray.norm(dim=-1)
        depth_valid=_frame(clips,'depth_valid',frame,device)[:,view,0][batch,y,x]
        provenance=torch.tensor([c.get('depth_provenance') not in (None,'missing','pseudo') for c in clips],device=device)
        target['depth_valid']=depth_valid & known_camera[:,None] & provenance[:,None] & torch.isfinite(z) & (z>0) & (target['range']>=near) & (target['range']<=far)
        if history:
            current=len(history)-1 if observed_now and history[-1]['frame']==frame else None
            target['status']=history_labels(clips,frame,pixels,history,current_index=current,view=view)
    if aligned:
        from .gauge import metric_query_to_scene,scene_output_to_metric
        prediction=scene_output_to_metric(model.read(state,metric_query_to_scene(state,query)),state)
    else:
        prediction=model.read(state,query)
    return prediction,target,query


def _bridge_state(model,branch,clips,history,diagnostics):
    """Fit once from independently reconstructed prefix points, never targets/future.

    Prefix camera/depth are teacher targets for this auxiliary only. Degenerate
    alignment is explicitly unavailable, leaving RGB reconstruction in scene gauge.
    """
    from .gauge import calibrate_state
    device=branch.c.device; scene=[]; metric=[]; valid=[]; times=[]
    for record in history:
        diag=record.get('diagnostics',{})
        if 'proposal_depth' not in diag:
            continue
        frame=record['frame']
        if not all('input_depth' in c and 'input_depth_valid' in c and 'intrinsics' in c for c in clips):
            continue
        p=model.config.patch_size; h,w=clips[0]['rgb'].shape[-2:]
        y=(torch.arange(math.ceil(h/p),device=device)*p+p//2).clamp(max=h-1)
        x=(torch.arange(math.ceil(w/p),device=device)*p+p//2).clamp(max=w-1)
        yy,xx=torch.meshgrid(y,x,indexing='ij'); yy,xx=yy.flatten(),xx.flatten()
        pixels=torch.stack((xx.float()+.5,yy.float()+.5,torch.ones_like(xx).float()),-1)
        sk,sp=record['camera']; mk=_frame(clips,'intrinsics',frame,device)[:,0]; mp=_frame(clips,'camera_c2w',frame,device)[:,0]
        estimate=(diag['proposal_depth']*diag['depth_prob']).sum(-1)
        depth=_frame(clips,'input_depth',frame,device)[:,0,0][:,yy,xx]
        sr=torch.einsum('bij,qj->bqi',torch.linalg.inv(sk),pixels)*estimate[...,None]
        mr=torch.einsum('bij,qj->bqi',torch.linalg.inv(mk),pixels)*depth[...,None]
        scene.append(torch.einsum('bij,bqj->bqi',sp[:,:3,:3],sr)+sp[:,None,:3,3])
        metric.append((torch.einsum('bij,bqj->bqi',mp[:,:3,:3],mr)+mp[:,None,:3,3]).detach())
        mask=_frame(clips,'input_depth_valid',frame,device)[:,0,0][:,yy,xx]
        mask &= record.get('diagnostic_mask',record['input_mask'])[:,0,0][:,yy,xx] & _camera_mask(clips,frame,device)[:,0,None]
        metric_flag=torch.stack([c.get('metric_valid',torch.zeros(c['rgb'].shape[:2],dtype=torch.bool))[frame,0] for c in clips]).to(device)
        valid.append(mask & metric_flag[:,None])
        times.append(record['time'].to(device)[:,None].expand_as(depth))
    if not scene:
        return branch
    return calibrate_state(branch,torch.cat(scene,1),torch.cat(metric,1),torch.cat(valid,1),torch.cat(times,1))


def _supported_content_losses(model,previous,state,clips,frame,history,*,rays,seed):
    from .history import history_labels
    from .objectives import per_example_mean
    zero=state.c.sum()*0
    if len(history)<2 or not any(c.get('scene_static',False) or c.get('rigid_correspondence_trusted',False) for c in clips):
        return zero,zero
    pred,target,query=_read_targets(model,state,clips,frame,rays=rays,seed=seed,history=history,observed_now=True,view=0)
    past_history=[]
    for record in history:
        earlier=[mask for admitted_at,mask in record.get('admissions',[(record['frame'],record['input_mask'])]) if admitted_at<frame]
        if earlier:
            past_history.append({**record,'input_mask':torch.stack(earlier).any(0)})
    past=history_labels(clips,frame,target['pixels'],past_history,current_index=None,view=0)
    support=(past==1)&(target['status']==0)
    if not bool(support.any()) or 'range' not in target:
        return zero,zero
    identity=(previous.slot_uid==state.slot_uid)&previous.content_initialized&state.content_initialized
    swapped=replace(state,c=torch.where(identity[...,None,None],previous.c,state.c))
    reused=model.read(swapped,query)
    reuse_target={**target,'rgb_valid':target['rgb_valid']&support}
    if 'depth_valid' in reuse_target:
        reuse_target['depth_valid']=reuse_target['depth_valid']&support
    reuse_target['status']=torch.full_like(target['status'],-1)
    reuse,_=readout_objective(reused,reuse_target)
    # Field consistency only at supported physical material points, with fixed
    # current canonical coordinates on both sides; teacher is stop-gradient.
    points=query.origins+target['range'][...,None]*query.directions
    transform,z=model.decoder.extended_controls(state.d)
    canonical=model.decoder.canonical(points[:,None].expand(-1,state.c.shape[1],-1,-1),transform,z).detach()
    directions=torch.einsum('bkji,bqj->bkqi',transform[...,:3,:3],query.directions).detach()
    args=(canonical.flatten(0,1),directions.flatten(0,1))
    current=model.decoder.field(state.c.flatten(0,1),*args)
    with torch.no_grad():
        teacher=model.decoder.field(previous.c.flatten(0,1),*args)
    def composite(values):
        density=values[0].reshape(*state.c.shape[:2],-1)*identity[...,None]
        color=values[1].reshape(*state.c.shape[:2],-1,3)
        mass=density.sum(1)
        return mass,(density[...,None]*color).sum(1)/mass[...,None].clamp_min(1e-8)
    density,color=composite(current); target_density,target_color=composite(teacher)
    error=(density-target_density).abs()+(color-target_color).abs().mean(-1)
    can=per_example_mean(error,support & identity.any(-1)[:,None])
    return reuse,can


def _bridge_read_loss(model,branch,clips,frame,*,rays,seed):
    """Independent alignment eligibility per draw preserves microbatch invariance."""
    total=branch.c.sum()*0
    for aligned in (False,True):
        indices=(branch.metric_valid==aligned).nonzero().flatten()
        if not len(indices): continue
        subset=[clips[int(i)] for i in indices]
        pred,target,_=_read_targets(model,branch.select(indices),subset,frame,rays=rays,seed=seed,
            view=0,metadata_drop=not aligned,aligned=aligned)
        loss,_=readout_objective(pred,target)
        total=total+loss*len(indices)/len(clips)
    return total


def compute_loss(model,clips,*,step,rays=256,seed=17,bridge=True,readout_callback=None,evaluation_view=None):
    """Homogeneous microbatch. All real horizons; no future enters filtering.

    Each microbatch mean is multiplied by its draw count / logical source count
    by the runner. This remains invariant to physical microbatch partitioning.
    """
    from .objectives import proposal_objective,reliability_objective,warp_nuisance_objective,supervised_auxiliary,camera_control_objective,pseudo_proposal_objective
    if not clips or any(clip_signature(c)!=clip_signature(clips[0]) for c in clips):
        raise ValueError('microbatch must share source and realized observation schedule')
    device=next(model.parameters()).device; b=len(clips); plan=clips[0]['plan']
    _,_,future_weight=curriculum(step,seed)
    zero=next(model.parameters()).sum()*0
    terms={key:zero for key in ('current','future','hidden','bridge','proposal','pseudo','reuse','canonical','reliability','controls','tracks','masks','warp')}
    metrics={key:0. for key in ('rgb','depth','status','status_label_coverage','status_class_0_coverage','status_class_1_coverage','status_class_2_coverage')}
    rng=random.Random(seed); tube_mask={}
    if step>=1000 and rng.random()<.25 and len(plan['prefix'])>1:
        frame=rng.choice(plan['prefix']); h,w=clips[0]['rgb'].shape[-2:]
        side=math.sqrt(rng.uniform(.05,.25)); mh,mw=max(1,int(h*side)),max(1,int(w*side))
        y,x=rng.randrange(h-mh+1),rng.randrange(w-mw+1)
        mask=torch.ones(b,1,min(2,frame+1),1,h,w,dtype=torch.bool,device=device)
        mask[...,y:y+mh,x:x+mw]=False; tube_mask[frame]=mask
    hidden_count=sum(not flag or frame in tube_mask for frame,flag in zip(plan['prefix'],plan['observed']))
    observed_count=max(1,sum(plan['observed'])); reads=0
    def emit(kind,frame,pred,target,query,state,history):
        if readout_callback is not None:
            readout_callback(kind=kind,frame=frame,prediction=pred,targets=target,query=query,state=state,history=history)
    for ordinal,frame,observed,previous,state,history,diag,obs in _prefix_steps(model,clips,seed=seed,tube_mask=tube_mask):
        if observed:
            proposal=proposal_objective(diag)
            if proposal is not None: terms['proposal']=terms['proposal']+proposal/observed_count
            pseudo=pseudo_proposal_objective(diag,clips,frame,model.config.patch_size)
            if pseudo is not None: terms['pseudo']=terms['pseudo']+pseudo/observed_count
            if all(key in diag for key in ('reliability','reliability_target','reliability_valid')):
                terms['reliability']=terms['reliability']+reliability_objective(diag['reliability'],diag['reliability_target'],diag['reliability_valid'])/observed_count
            for key,value in supervised_auxiliary(model,state,clips,frame,diag).items():
                terms[key]=terms[key]+value/observed_count
            terms['controls']=terms['controls']+camera_control_objective(previous,clips,frame,diag)/observed_count
            terms['warp']=terms['warp']+warp_nuisance_objective(model,state)/observed_count
            reuse,can=_supported_content_losses(model,previous,state,clips,frame,history,rays=rays,seed=seed+ordinal+771)
            terms['reuse']=terms['reuse']+reuse/observed_count; terms['canonical']=terms['canonical']+can/observed_count
        prediction,target,query=_read_targets(model,state,clips,frame,rays=rays,seed=seed+ordinal,
            observed_now=observed,input_mask=obs.valid,view=(0 if ordinal%2==0 else None) if evaluation_view is None else evaluation_view,history=history)
        loss,m=readout_objective(prediction,target); terms['current']=terms['current']+loss/len(plan['prefix'])
        for key in metrics: metrics[key]+=float(m[key])
        reads+=1; emit('current',frame,prediction,target,query,state,history)
        if not observed or frame in tube_mask:
            prediction,target,query=_read_targets(model,state,clips,frame,rays=rays,seed=seed+ordinal+97,
                input_mask=obs.valid,hidden_only=True,view=0,history=history,observed_now=observed)
            hidden,_=readout_objective(prediction,target); terms['hidden']=terms['hidden']+hidden/max(1,hidden_count)
            emit('hidden',frame,prediction,target,query,state,history)
    cutoff=state
    bridge_active=bridge and all('intrinsics' in c for c in clips) and seed%4==0
    branch=None; bridge_aligned=False
    if bridge_active:
        branch,branch_history,branch_diag=observe_prefix(model,clips,seed=seed,metadata_drop=True,tube_mask=tube_mask)
        branch=_bridge_state(model,branch,clips,branch_history,branch_diag)
        # Homogeneous batches may still have per-example alignment validity.
        # Read aligned and unaligned subbatches separately to avoid fake metric GT.
        bridge_aligned=float(branch.metric_valid.float().mean())
        frame=plan['prefix'][-1]
        count=1+(len(plan['future']) if all(c.get('time_valid',True) for c in clips) else 0)
        terms['bridge']=_bridge_read_loss(model,branch,clips,frame,rays=rays,seed=seed+31)/count
    futures=plan['future'] if all(c.get('time_valid',True) for c in clips) else []
    for frame in sorted(futures):
        time=_frame(clips,'times',frame,device); state=model.predict(state,time-state.t)
        pred,target,query=_read_targets(model,state,clips,frame,rays=rays,seed=seed+frame,history=history,view=evaluation_view)
        loss,_=readout_objective(pred,target); terms['future']=terms['future']+loss/max(1,len(futures))
        emit('future',frame,pred,target,query,state,history)
        if branch is not None:
            branch=model.predict(branch,time-branch.t)
            loss_b=_bridge_read_loss(model,branch,clips,frame,rays=rays,seed=seed+frame)
            terms['bridge']=terms['bridge']+loss_b/(1+len(futures))
    bridge_weight=.2*min(1.,step/1000)
    total=terms['current']+future_weight*terms['future']+terms['hidden']+bridge_weight*terms['bridge']
    total=total+.1*(terms['proposal']+terms['reliability']+terms['reuse']+terms['controls']+terms['tracks']+terms['masks'])+.05*terms['canonical']+.01*terms['warp']+.2*terms['pseudo']
    metrics={key:value/max(1,reads) for key,value in metrics.items()}
    metrics.update({key:float(value.detach()) for key,value in terms.items()})
    metrics.update(observed_steps=sum(plan['observed']),gap_steps=plan['observed'].count(False),future_valid=len(futures),
        hidden_draw=float(hidden_count>0),microbatch=b,bridge_active=float(bridge_active),bridge_aligned=float(bridge_aligned),
        timestamp_fps_derived=sum(c.get('timestamp_provenance')=='frame_index_fps' for c in clips)/b,
        prefix_span_seconds=sum(float(c['times'][plan['prefix'][-1]]-c['times'][0]) if c.get('time_valid',True) else 0. for c in clips)/b,
        prefix_span_valid=sum(bool(c.get('time_valid',True)) for c in clips)/b)
    return total,metrics
