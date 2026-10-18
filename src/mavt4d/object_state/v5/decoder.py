"""Unified canonical fields, invertible deformation and evidence classification."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..decoder import _CanonicalField, _rotation_6d, volume_render


class V5Readout(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.control_head=nn.Sequential(nn.Linear(cfg.width*cfg.dynamic_tokens,cfg.width),nn.SiLU(),nn.Linear(cfg.width,25))
        nn.init.normal_(self.control_head[-1].weight,std=1e-3)
        nn.init.zeros_(self.control_head[-1].bias)
        with torch.no_grad():
            self.control_head[-1].bias[:6].copy_(torch.tensor([1.,0.,0.,0.,1.,0.]))
        self.couplings=nn.ModuleList([nn.Sequential(nn.Linear(2,32),nn.SiLU(),nn.Linear(32,32)) for _ in range(3)])
        self.field=_CanonicalField(cfg.width,cfg.resampler_heads)
        self.status_head=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,3))

    def extended_controls(self,d):
        values=self.control_head(d.flatten(-2).float())
        rotation=_rotation_6d(values[...,:6])
        top=torch.cat((rotation,values[...,6:9,None]),-1)
        bottom=torch.zeros_like(top[...,:1,:]); bottom[...,0,3]=1
        return torch.cat((top,bottom),-2),torch.tanh(values[...,9:])

    def controls(self,d):
        return self.extended_controls(d)[0]

    def relation_controls(self,d):
        transform,z=self.extended_controls(d)
        return torch.cat((transform[...,:3,:].flatten(-2),z),-1)

    def warp(self,points,z,*,inverse=False):
        x=points
        for index in (range(2,-1,-1) if inverse else range(3)):
            other=[j for j in range(3) if j != index]
            bases=self.couplings[index](x[...,other]).reshape(*x.shape[:-1],16,2)
            parameters=(bases*z[...,None,:,None]).sum(-2)/4
            scale=.1*torch.tanh(parameters[...,0])
            shift=.25*torch.tanh(parameters[...,1])
            changed=(x[...,index]-shift)*torch.exp(-scale) if inverse else x[...,index]*torch.exp(scale)+shift
            x=torch.stack([changed if j == index else x[...,j] for j in range(3)],-1)
        return x

    def canonical(self,points,transform,z):
        local=torch.einsum('bkji,bkqj->bkqi',transform[...,:3,:3],points-transform[...,:3,3][:,:,None])
        return self.warp(local,z,inverse=True)

    def forward(self,state,query,*,ray_chunk_size=None):
        from .evidence import atlas_lookup
        if query.time.shape != state.t.shape or not torch.allclose(query.time.double(),state.t,atol=1e-8,rtol=0):
            raise ValueError('query timestamp must equal state timestamp')
        b,r,_=query.origins.shape
        if r<1 or query.directions.shape != query.origins.shape or b != len(state.c):
            raise ValueError('query requires matching nonempty [B,R,3] rays')
        valid=torch.ones(b,r,device=state.c.device,dtype=torch.bool) if query.valid is None else query.valid
        if valid.shape != (b,r):
            raise ValueError('ray validity shape mismatch')
        near=torch.as_tensor(query.near,device=state.c.device,dtype=torch.float32).expand(b,r)
        far=torch.as_tensor(query.far,device=state.c.device,dtype=torch.float32).expand(b,r)
        if not torch.isfinite(query.origins[valid]).all() or not torch.isfinite(query.directions[valid]).all() or not torch.isfinite(near[valid]).all() or not torch.isfinite(far[valid]).all() or ((near[valid]<0)|(far[valid]<=near[valid])).any():
            raise ValueError('invalid finite rays/bounds')
        if not torch.allclose(query.directions[valid].norm(dim=-1),torch.ones_like(near[valid]),atol=1e-4,rtol=1e-4):
            raise ValueError('ray directions must be normalized')
        transforms,z=self.extended_controls(state.d)
        k=state.c.shape[1]
        # Only explicit lifecycle eligibility gates a field. Learned confidence
        # must not become a second density/appearance channel outside C.
        existence=state.allocated*state.content_initialized
        fractions=(torch.arange(self.cfg.samples_per_ray,device=state.c.device)+.5)/self.cfg.samples_per_ray
        chunks=[]
        for start in range(0,r,ray_chunk_size or self.cfg.query_chunk_size):
            end=min(start+(ray_chunk_size or self.cfg.query_chunk_size),r)
            good=valid[:,start:end]
            origin=torch.where(good[...,None],query.origins[:,start:end].float(),0.)
            direction=torch.where(good[...,None],query.directions[:,start:end].float(),0.)
            lo=torch.where(good,near[:,start:end],0.)
            hi=torch.where(good,far[:,start:end],1.)
            distances=lo[...,None]+(hi-lo)[...,None]*fractions
            points=origin[:,:,None]+distances[...,None]*direction[:,:,None]
            flat=points.flatten(1,2)
            canonical=self.canonical(flat[:,None].expand(-1,k,-1,-1),transforms,z)
            dirs=torch.einsum('bkji,bqj->bkqi',transforms[...,:3,:3],direction[:,:,None].expand_as(points).flatten(1,2))
            inputs=(state.c.flatten(0,1),canonical.flatten(0,1),dirs.flatten(0,1))
            if self.training and self.cfg.decoder_checkpoint and torch.is_grad_enabled():
                density,color=checkpoint(self.field,*inputs,use_reentrant=False)
            else:
                density,color=self.field(*inputs)
            density=density.reshape(b,k,-1)*existence[...,None]
            color=color.reshape(b,k,-1,3)
            total=density.sum(1)
            rgb=(density[...,None]*color).sum(1)/total[...,None].clamp_min(1e-12)
            out=volume_render(total.reshape_as(distances),rgb.reshape(*distances.shape,3),distances,
                              ((hi-lo)/self.cfg.samples_per_ray)[...,None].expand_as(distances),torch.zeros_like(origin))
            out['points']=origin+out['range'][...,None]*direction
            # Approximate atlas support is an input feature, never a GT label.
            with torch.no_grad():
                support=atlas_lookup(state.support_bits,canonical,extent=self.cfg.atlas_extent).reshape(b,k,end-start,-1)
                sample_density=total.reshape_as(distances)
                optical=sample_density*((hi-lo)/self.cfg.samples_per_ray)[...,None]
                before=optical.cumsum(-1)-optical
                sample_weights=torch.exp(-before)*(-torch.expm1(-optical))
                slot_fraction=density.reshape(b,k,end-start,-1)/sample_density[:,None].clamp_min(1e-12)
                contributions=sample_weights[:,None]*slot_fraction
                normalized=contributions/contributions.sum((1,3),keepdim=True).clamp_min(1e-12)
                mix=normalized.sum(-1)
                supported=(normalized*support).sum((1,3))
                age=(state.t[:,None]-state.last_seen).float().clamp_min(0)
                features=torch.stack((out['opacity'],supported,
                                      (mix*torch.log1p(age)[:,:,None]).sum(1),
                                      (mix*state.provenance[:,:,None]).sum(1),mix.sum(1)),dim=-1)
            out['evidence_support']=supported
            out['status_logits']=self.status_head(features.detach())
            # This is classification confidence, not calibrated RGB uncertainty.
            out['confidence']=out['status_logits'].softmax(-1).amax(-1)
            for name,value in out.items():
                out[name]=torch.where(good[...,None] if value.ndim==3 else good,value,torch.zeros_like(value))
            chunks.append(out)
        result={name:torch.cat([part[name] for part in chunks],1) for name in chunks[0]}
        result.update(controls=transforms,valid=valid,deformation=z)
        return result
