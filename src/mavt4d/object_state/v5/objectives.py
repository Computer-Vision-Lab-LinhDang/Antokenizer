"""Differentiable, availability-masked objectives for one logical-source draw.

Every reduction first normalizes valid elements per example, then averages all
examples (including absent terms). Logical-batch weighting belongs to the runner.
"""
import torch
from torch.nn import functional as F


def per_example_mean(value, valid):
    valid = valid.to(device=value.device, dtype=torch.bool).expand_as(value)
    valid = valid & torch.isfinite(value)
    clean = torch.where(valid, value, 0.)
    if value.ndim == 1:
        return clean.sum() / valid.sum().clamp_min(1)
    return (clean.flatten(1).sum(-1) / valid.flatten(1).sum(-1).clamp_min(1)).mean()


def reliability_objective(prediction, target, valid):
    return per_example_mean(F.binary_cross_entropy(prediction.clamp(1e-6,1-1e-6),
        target.detach().clamp(0,1),reduction='none'),valid)


def proposal_objective(diagnostics):
    proposal=diagnostics.get('proposal_depth')
    if proposal is None:
        return None
    estimate=(proposal*diagnostics['depth_prob']).sum(-1)
    target=diagnostics['measured_depth'].detach()
    return per_example_mean((estimate.clamp_min(1e-3).log()-target.clamp_min(1e-3).log()).abs(),
                            diagnostics['geometry_measured'])


def warp_nuisance_objective(model,state):
    """Reference identity and bounded exposure regularizers; never confidence masks."""
    _,z=model.decoder.extended_controls(state.d)
    active=state.content_initialized
    reference=state.c.new_tensor([[-1.,0.,0.],[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    points=reference[None,None].expand(*z.shape[:2],-1,-1)
    displacement=model.decoder.warp(points,z)-points
    identity=per_example_mean(displacement.square(),active[...,None,None])
    # Displacement change under a small local perturbation limits nonsmooth warp.
    shifted=model.decoder.warp(points+.01,z)-(points+.01)
    smooth=per_example_mean(((shifted-displacement)/.01).square(),active[...,None,None])
    rigid=per_example_mean(displacement.mean(-2).square(),active[...,None])
    nuisance=per_example_mean(state.camera[...,22:26].square(),state.camera_valid[...,None])
    return identity+.1*smooth+rigid+nuisance


def supervised_auxiliary(model,state,clips,frame,diagnostics):
    """Optional *explicitly matched* labels; object labels alone are not slot IDs.

    slot_pose/slot_pose_valid require a documented persistent slot assignment.
    track_prediction/target/valid and ownership_logits/target/valid are supplied
    by a correspondence adapter; this function never invents correspondences.
    """
    zero=state.c.sum()*0
    result={key:zero for key in ('controls','tracks','masks')}
    if all('slot_pose' in c and c.get('slot_assignment_trusted',False) for c in clips):
        pose=model.decoder.controls(state.d)
        target=torch.stack([c['slot_pose'][frame] for c in clips]).to(pose).detach()
        valid=torch.stack([c['slot_pose_valid'][frame] for c in clips]).to(pose.device)
        # Frobenius SO(3) distance avoids unstable acos gradients at identity.
        translation=(pose[...,:3,3]-target[...,:3,3]).abs().mean(-1)
        rotation=(pose[...,:3,:3]-target[...,:3,:3]).square().sum((-2,-1))/8
        result['controls']=per_example_mean(translation+rotation,valid)
    for key in ('tracks','masks'):
        prediction=diagnostics.get(key+'_prediction')
        target=diagnostics.get(key+'_target')
        valid=diagnostics.get(key+'_valid')
        if prediction is None or target is None or valid is None:
            continue
        if key=='tracks':
            error=(prediction-target.detach()).norm(dim=-1)
            diagonal=diagnostics.get('image_diagonal',1.)
            result[key]=per_example_mean(error/diagonal,valid)
        else:
            error=F.binary_cross_entropy_with_logits(prediction,target.detach().float(),reduction='none')
            result[key]=per_example_mean(error,valid)
    return result


def camera_control_objective(previous,clips,frame,diagnostics):
    """Supervise the estimate *before* metadata replacement, in its known gauge."""
    estimate=diagnostics.get('camera_estimate_before_metadata')
    if estimate is None or not all('camera_c2w' in c and 'camera_valid' in c for c in clips):
        return previous.c.sum()*0
    target=torch.stack([c['camera_c2w'][frame,:estimate.shape[1],:3] for c in clips]).to(estimate).detach()
    valid=torch.stack([c['camera_valid'][frame,:estimate.shape[1]] for c in clips]).to(estimate.device)
    # First observed RGB-only camera fixes its own gauge; there is no pose target
    # in that gauge until an explicit Sim(3) bridge has been accepted.
    valid &= previous.camera_valid[:,:estimate.shape[1]]
    prior_metric=torch.stack([c.get('metric_valid',torch.zeros(c['rgb'].shape[:2],dtype=torch.bool))[frame,:estimate.shape[1]] for c in clips]).to(estimate.device)
    valid &= ~prior_metric | previous.metric_valid[:,None]
    if hasattr(previous,'scene_to_metric'):
        from .gauge import similarity_components
        scale,rotation_map,translation_map=similarity_components(previous)
        mapped_rotation=rotation_map[:,None]@estimate[...,:3]
        mapped_translation=scale[:,None,None]*torch.einsum('bij,bvj->bvi',rotation_map,estimate[...,3])+translation_map[:,None]
        mapped=torch.cat((mapped_rotation,mapped_translation[...,None]),-1)
        estimate=torch.where(prior_metric[...,None,None],mapped,estimate)
    rotation=(estimate[...,:3]-target[...,:3]).square().sum((-2,-1))/8
    translation=(estimate[...,3]-target[...,3]).abs().mean(-1)
    error=rotation+translation
    intrinsic=diagnostics.get('camera_intrinsics_before_metadata')
    if intrinsic is not None and all('intrinsics' in c for c in clips):
        k=torch.stack([c['intrinsics'][frame,:estimate.shape[1]] for c in clips]).to(intrinsic)
        ref=torch.stack((k[...,0,0],k[...,1,1],k[...,0,2],k[...,1,2]),-1)
        image_scale=float(max(clips[0]['rgb'].shape[-2:]))
        error=error+(intrinsic-ref.detach()).abs().mean(-1)/image_scale
    return per_example_mean(error,valid)


def pseudo_proposal_objective(diagnostics,clips,frame,patch_size):
    """Opt-in offline pseudo depth only trains proposals, never observation GT."""
    proposal=diagnostics.get('proposal_depth')
    if proposal is None:
        return None
    estimate=(proposal*diagnostics['depth_prob']).sum(-1)
    losses=[]
    for index,c in enumerate(clips):
        if not c.get('pseudo_depth_approved',False) or c.get('pseudo_depth_provenance')!='pseudo' or not all(k in c for k in ('pseudo_depth','pseudo_depth_valid')):
            losses.append(estimate[index].sum()*0); continue
        target=c['pseudo_depth'][frame,0,0].to(estimate)
        valid=c['pseudo_depth_valid'][frame,0,0].to(estimate.device)
        h,w=target.shape
        yy=(torch.arange((h+patch_size-1)//patch_size,device=estimate.device)*patch_size+patch_size//2).clamp(max=h-1)
        xx=(torch.arange((w+patch_size-1)//patch_size,device=estimate.device)*patch_size+patch_size//2).clamp(max=w-1)
        y,x=torch.meshgrid(yy,xx,indexing='ij'); target=target[y,x].flatten().detach(); valid=valid[y,x].flatten()
        valid &= torch.isfinite(target)&(target>0)&torch.isfinite(estimate[index])
        if valid.sum()<3:
            losses.append(estimate[index].sum()*0); continue
        pred=estimate[index,valid]; truth=target[valid]
        # Affine-normalized depth is invariant to the unknown teacher scale/shift.
        pred=pred-pred.median(); truth=truth-truth.median()
        pred=pred/pred.abs().mean().clamp_min(1e-4); truth=truth/truth.abs().mean().clamp_min(1e-4)
        losses.append((pred-truth).abs().mean())
    return torch.stack(losses).mean()
