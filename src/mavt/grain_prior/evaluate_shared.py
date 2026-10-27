"""Source-only NVS and fixed-denominator consistency metrics (protocol v1).

GT visibility masks are supplied by the dataset/evaluator, never by decoder support or opacity.
CC-D is structural agreement; GT depth and RGB quality must be reported alongside it.
"""
import torch
import torch.nn.functional as F
from . import geometry as g, schema as sc
from .query import RayQuery


def _grid(height,width,device):
    y,x = torch.meshgrid((torch.arange(height,device=device)+.5)/height,
                         (torch.arange(width,device=device)+.5)/width,indexing='ij')
    return torch.stack([x,y],-1).reshape(-1,2)


def _sample(image,uv):
    return F.grid_sample(image.float(),uv[:,None]*2-1,align_corners=False,padding_mode='border').squeeze(2).transpose(1,2)


def cross_depth_error(depth_a,depth_b,camera_a,camera_b,gt_visible):
    """One direction of CC-D; call both directions for a pair. Invalid predictions cost 1.

    Return uncapped finite quantiles and failure rate separately. Empty GT mask is an error,
    not a perfect score. Bilinear lookup uses edge-normalized pixel coordinates.
    """
    if depth_a.ndim != 3 or gt_visible.shape != depth_a.shape or gt_visible.dtype != torch.bool:
        raise ValueError('depth and fixed GT mask require [B,H,W]')
    if not gt_visible.any(): raise ValueError('no GT-visible evaluation queries')
    B,H,W = depth_a.shape
    da = depth_a.float().reshape(B,-1)
    valid_a = torch.isfinite(da) & (da > 0)
    uv = _grid(H,W,da.device).expand(B,-1,-1)
    pa = g.unproject(torch.where(valid_a,da,torch.ones_like(da)),camera_a.kappa,uv)
    canonical = (pa-camera_a.t[:,None]) @ camera_a.R
    pb = canonical @ camera_b.R.transpose(-1,-2)+camera_b.t[:,None]
    ub,zb,valid_b = g.project(pb,camera_b.kappa,1e-6)
    in_image = ((ub >= 0) & (ub <= 1)).all(-1)
    # Preserve invalid target samples: they must not disappear from the denominator.
    db = _sample(depth_b[:,None],ub).squeeze(-1)
    valid = valid_a & valid_b & in_image & torch.isfinite(db) & (db > 0)
    logerr = (zb.clamp_min(1e-6).log()-torch.nan_to_num(db,nan=1.,posinf=1.,neginf=1.).clamp_min(1e-6).log()).abs()
    error = torch.where(valid,logerr.clamp_max(1),torch.ones_like(logerr))
    mask = gt_visible.reshape(B,-1)
    finite = logerr[mask & valid]
    quantiles = torch.quantile(finite,finite.new_tensor([.5,.9,.99])) if finite.numel() else None
    return dict(cc_d=error[mask].mean(),count=int(mask.sum()),invalid_fraction=(~valid)[mask].float().mean(),
                finite_log_quantiles=quantiles)


def depth_accuracy(pred,gt,mask):
    fixed = mask & torch.isfinite(gt) & (gt > 0)
    if not fixed.any(): raise ValueError('no valid GT depth')
    valid = torch.isfinite(pred) & (pred > 0)
    safe = torch.nan_to_num(pred,nan=0.,posinf=0.,neginf=0.)
    absrel = torch.where(valid,(safe-gt).abs()/gt.clamp_min(1e-6),torch.ones_like(gt))
    return dict(absrel=absrel[fixed].mean(),invalid_fraction=(~valid)[fixed].float().mean(),count=int(fixed.sum()))


def paired_scene_ci(candidate,control,draws=2000,seed=0):
    """Inputs [scene,seed], aligned by scene IDs by the caller. Bootstrap scene means.

    CI is conditional on the supplied seeds; never treats pixels or seed replicates as scenes.
    """
    if candidate.shape != control.shape or candidate.ndim != 2 or candidate.shape[0]<2 or candidate.shape[1]!=3:
        raise ValueError('primary comparison requires >=2 paired scenes and exactly 3 seeds')
    if not torch.isfinite(candidate).all() or not torch.isfinite(control).all():
        raise ValueError('nonfinite scene scores')
    delta = (candidate-control).float().cpu()
    means = delta.mean(1)
    idx = torch.randint(len(means),(draws,len(means)),generator=torch.Generator().manual_seed(seed))
    ci = torch.quantile(means[idx].mean(1),torch.tensor([.025,.975]))
    seed_ci = torch.quantile(delta[idx].mean(1),torch.tensor([.025,.975]),dim=0)
    return dict(delta=means.mean().item(),lower=ci[0].item(),upper=ci[1].item(),
                scenes=len(means),seeds=delta.shape[1],per_seed=delta.mean(0).tolist(),
                per_seed_scene_ci=seed_ci.T.tolist(),seed_sd=delta.mean(0).std().item())


def legacy_queries(model,state,camera,uv):
    """E0 adapter: retain native convolutional RGB and dense depth, sample a common ray grid.

    Its footprint differs from the pointwise heads; report this as the legacy quality control.
    It does not claim S1/S2 or support arbitrary point queries.
    """
    B = state['h'].shape[0]
    o = model.decode_obs(state,model.derived(state),state['c0'],g.se3_log(camera.R,camera.t),camera.kappa,
                         torch.zeros(B,model.cfg.n_motion,3,device=uv.device))
    rgb = model.rgb_of(o['F_hat'],o['H_hat'])
    feature = g.read_coarse(o['F_hat'],uv,model.cfg.grid)
    depth = g.read_coarse(o['depth_dense'][...,None],uv,model.cfg.grid).squeeze(-1)
    return dict(rgb=_sample(rgb,uv),feature=feature,depth=depth)


@torch.no_grad()
def evaluate_nvs(model,source,source_kappa,target_cameras,resolution=None):
    """No target RGB/depth arguments; encode once, close state, decode cameras in any order."""
    if model.training: raise ValueError('NVS evaluation requires eval mode')
    if not target_cameras: raise ValueError('no target cameras')
    size = resolution or model.cfg.image_size
    S = model.encode_source(source,source_kappa)
    flats = sc.serialize(model.cfg,S)
    S = sc.deserialize(model.cfg,flats)
    view = model.prepare_state(S,mode='novel_view')
    uv = g.cell_centres(size,source.device).expand(source.shape[0],-1,-1)
    outs = []
    for camera in target_cameras:
        if model.cfg.readout_version == 'legacy':
            out = legacy_queries(model,S,camera,uv)
        else:
            query = RayQuery(camera,uv,model.cfg.render_near,model.cfg.render_far,model.cfg.render_samples)
            packet = model.read_rays(view,query)
            out = model.decode_packet(packet)
        outs.append(out)
    return dict(state=flats,views=outs,readout_version=model.cfg.readout_version,
                rgb_kind='rgb_core' if model.cfg.readout_version=='shared_field_v1' else 'rgb_direct',
                resolution=size,metric_version='cc-d-v1')
