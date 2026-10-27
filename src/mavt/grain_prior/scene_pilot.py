"""Posed C-only pilot objective, separate from the concurrent ImageNet/CRSR trainer.

RGB, teacher feature, and labelled log-depth have unit weight, as in pilot protocol v1.
No LPIPS on scattered rays. Consistency and spatial innovations remain gated.
"""
import torch
from . import geometry as g
from .evaluate_shared import legacy_queries
from .losses import feat_loss, huber_log_depth
from .query import RayQuery


def scene_step(model,source,source_kappa,target,camera,target_depth,rays=512,seed=0):
    cfg = model.cfg
    if target.shape != source.shape or target_depth.shape != (target.shape[0],*target.shape[2:]):
        raise ValueError('source/target must be [B,3,H,W]; depth [B,H,W]')
    if source.shape[-2:] != (cfg.image_size,cfg.image_size):
        raise ValueError('image resolution does not match config')
    B,_,H,W = target.shape
    if not 0 < rays <= H*W:
        raise ValueError('invalid ray count')
    if cfg.w_lpips != 0:
        raise ValueError('scattered-ray pilot requires w_lpips=0 in every arm')
    gen = torch.Generator(device=source.device).manual_seed(seed)
    idx = torch.stack([torch.randperm(H*W,generator=gen,device=source.device)[:rays] for _ in range(B)])
    uv = g.cell_centres(H,source.device)[idx]
    # Explicit boundary: only source image + source intrinsics enter the state encoder.
    state = model.encode_source(source,source_kappa)
    if cfg.readout_version == 'legacy':
        pred = legacy_queries(model,state,camera,uv)
    else:
        view = model.prepare_state(state,mode='novel_view')
        q = RayQuery(camera,uv,cfg.render_near,cfg.render_far,cfg.render_samples)
        pred = model.decode_packet(model.read_rays(view,q))
    # Teacher target is only a supervision tensor and is never passed to a reader/query.
    with torch.no_grad():
        feature,_,_ = model.stems.dino(target)
        feature = g.read_coarse(feature.float(),uv,cfg.grid)
    rgb = ((target.float()+1)/2).flatten(2).transpose(1,2).gather(1,idx[...,None].expand(-1,-1,3))
    depth = target_depth.float().flatten(1).gather(1,idx)
    mask = torch.isfinite(depth) & (depth > 0)
    terms = dict(rgb=(pred['rgb']-rgb).abs().mean(),
                 feature=feat_loss(pred['feature'],feature,cfg.w_logn),
                 depth=huber_log_depth(pred['depth'],depth,mask,cfg.depth_delta))
    return dict(loss=sum(terms.values()),terms=terms,predictions=pred,uv=uv,
                depth_label_count=int(mask.sum()),readout_version=cfg.readout_version,
                rgb_kind='rgb_core' if cfg.readout_version=='shared_field_v1' else 'rgb_direct')
