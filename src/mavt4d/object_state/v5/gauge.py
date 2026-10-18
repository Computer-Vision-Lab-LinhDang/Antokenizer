"""Explicit prefix-only Sim(3) calibration; latent state stays in scene units.

A camera pose is not proof of metric scale. Callers supply trusted, paired
observed-prefix points. Missing/degenerate alignment never changes validity.
"""
from dataclasses import replace
import torch


@torch.no_grad()
def fit_prefix_similarity(scene_points, metric_points, valid, times, cutoff,
                          *, residual_tolerance=.05):
    """Fit x_metric=s R x_scene+b with at least 3 noncollinear correspondences.

    Calibration is explicitly stop-gradient: SVD rotation derivatives are
    undefined at repeated singular values, even for observable fits. Downstream
    scene/readout gradients remain live through the fixed transform.
    `times` proves temporal eligibility, not
    correspondence correctness; callers must not pass future or predicted GT.
    Return identity and False for insufficient/degenerate/inconsistent fits.
    """
    if scene_points.ndim != 3 or scene_points.shape[-1] != 3 or metric_points.shape != scene_points.shape:
        raise ValueError('paired points must be [B,N,3]')
    b,n,_=scene_points.shape
    if valid.shape != (b,n) or valid.dtype != torch.bool or times.shape != (b,n) or cutoff.shape != (b,):
        raise ValueError('invalid correspondence validity or timestamps')
    if not torch.isfinite(times[valid]).all() or not torch.isfinite(cutoff).all():
        raise ValueError('prefix timestamps must be finite')
    if (valid & (times > cutoff[:,None])).any():
        raise ValueError('future correspondences cannot calibrate a prefix state')
    if not 0 < residual_tolerance < 1:
        raise ValueError('residual tolerance must be in (0,1)')
    output=[]; observable=[]
    for index in range(b):
        mask=valid[index]&torch.isfinite(scene_points[index]).all(-1)&torch.isfinite(metric_points[index]).all(-1)
        identity=torch.eye(4,device=scene_points.device,dtype=scene_points.dtype)
        x=scene_points[index,mask].double(); y=metric_points[index,mask].double()
        ok=len(x)>=3
        if ok:
            xm=x.mean(0); ym=y.mean(0); xc=x-xm; yc=y-ym
            # A collapsed/collinear fit cannot identify a 3D similarity.
            singular=torch.linalg.svdvals(xc.detach())
            target_singular=torch.linalg.svdvals(yc.detach())
            ok=bool(singular[1]>singular[0].clamp_min(1e-12)*1e-5) and bool(target_singular[1]>target_singular[0].clamp_min(1e-12)*1e-5)
        if ok:
            u,s,vh=torch.linalg.svd(yc.T@xc/len(x))
            correction=torch.ones(3,device=x.device,dtype=x.dtype)
            correction[-1]=torch.sign(torch.linalg.det(u@vh)).detach()
            rotation=(u*correction)@vh
            scale=(s*correction).sum()/(xc.square().sum()/len(x)).clamp_min(1e-12)
            translation=ym-scale*(rotation@xm)
            predicted=scale*(x@rotation.T)+translation
            normalized=(predicted-y).square().mean().sqrt()/yc.square().mean().sqrt().clamp_min(1e-12)
            ok=bool(torch.isfinite(scale)&(scale>1e-8)&(normalized.detach()<=residual_tolerance))
            if ok:
                top=torch.cat((scale*rotation,translation[:,None]),-1)
                identity=torch.cat((top,identity.double()[3:]),0).to(scene_points.dtype)
        output.append(identity); observable.append(ok)
    return torch.stack(output),torch.tensor(observable,device=scene_points.device,dtype=torch.bool)


def calibrate_state(state,scene_points,metric_points,valid,times):
    transform,observable=fit_prefix_similarity(scene_points,metric_points,valid,times,state.t)
    # One accepted bridge per sequence: never realign each future frame/object.
    update=observable&~state.metric_valid
    return replace(state,scene_to_metric=torch.where(update[:,None,None],transform.to(state.scene_to_metric),state.scene_to_metric),metric_valid=state.metric_valid|update)


def similarity_components(state):
    matrix=state.scene_to_metric
    scale=torch.linalg.vector_norm(matrix[:,:3,0],dim=-1)
    return scale,matrix[:,:3,:3]/scale[:,None,None],matrix[:,:3,3]


def metric_query_to_scene(state,query):
    if not state.metric_valid.all():
        raise ValueError('metric query requires observable metric calibration')
    scale,rotation,translation=similarity_components(state)
    origin=torch.einsum('bij,brj->bri',rotation.transpose(-1,-2),query.origins-translation[:,None])/scale[:,None,None]
    direction=torch.einsum('bij,brj->bri',rotation.transpose(-1,-2),query.directions)
    near=torch.as_tensor(query.near,device=origin.device,dtype=origin.dtype).expand(origin.shape[:2])/scale[:,None]
    far=torch.as_tensor(query.far,device=origin.device,dtype=origin.dtype).expand(origin.shape[:2])/scale[:,None]
    return replace(query,origins=origin,directions=direction,near=near,far=far)


def scene_output_to_metric(output,state):
    """RGB/opacity unchanged: marcher remains in scene units (sigma*ds invariant)."""
    if not state.metric_valid.all():
        raise ValueError('metric output requires observable metric calibration')
    scale,rotation,translation=similarity_components(state)
    result=dict(output)
    if 'range' in result: result['range']=result['range']*scale[:,None]
    if 'depth' in result: result['depth']=result['depth']*scale[:,None]
    if 'points' in result: result['points']=scale[:,None,None]*torch.einsum('bij,brj->bri',rotation,result['points'])+translation[:,None]
    if 'density' in result: result['density']=result['density']/scale.reshape(-1,*([1]*(result['density'].ndim-1)))
    if 'controls' in result:
        controls=result['controls']
        r=rotation[:,None]@controls[...,:3,:3]
        t=scale[:,None,None]*torch.einsum('bij,bkj->bki',rotation,controls[...,:3,3])+translation[:,None]
        result['controls']=torch.cat((torch.cat((r,t[...,None]),-1),controls[...,3:,:]),-2)
    return result


def prepare_observation_gauge(state,observation):
    """Establish a known initial sensor gauge or convert a calibrated return.

    An already-relative sequence cannot silently ingest metric coordinates.
    Its caller must fit trusted prefix correspondences with calibrate_state.
    """
    if observation.metric_valid is None or not observation.metric_valid.any():
        return state,observation
    if observation.intrinsics is None or observation.camera_c2w is None:
        raise ValueError('metric observations require valid camera metadata')
    measured=observation.metric_valid
    camera_valid=observation.camera_valid if observation.camera_valid is not None else observation.geometry_valid
    if camera_valid is not None: measured=measured&camera_valid
    if observation.valid is not None:
        measured=measured&observation.valid[:,:,-1].flatten(2).any(-1)
    known=measured.any(-1)
    initial=~state.camera_valid.any(-1)&~state.allocated.any(-1)
    if (known&~state.metric_valid&~initial).any():
        raise ValueError('sensor return requires observable prefix calibration before correction')
    state=replace(state,metric_valid=state.metric_valid|(known&initial))
    scale,rotation,translation=similarity_components(state)
    raw=observation.camera_c2w
    r=rotation[:,None].transpose(-1,-2)@raw[...,:3,:3]
    t=torch.einsum('bij,bvj->bvi',rotation.transpose(-1,-2),raw[...,:3,3]-translation[:,None])/scale[:,None,None]
    converted=torch.cat((torch.cat((r,t[...,None]),-1),raw[...,3:,:]),-2)
    pose=torch.where(measured[...,None,None],converted,raw)
    depth=observation.depth
    if depth is not None:
        depth=torch.where(measured[...,None,None,None],depth/scale[:,None,None,None,None],depth)
    return state,replace(observation,camera_c2w=pose,depth=depth)
