"""Two functional reads with shared attention and detached canonical routing."""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F


class FunctionalResampler(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        d,e=cfg.width,cfg.evidence_width
        self.norm=nn.LayerNorm(d)
        self.evidence_norm=nn.LayerNorm(e)
        self.query=nn.Linear(d,d)
        self.key=nn.Linear(e,d)
        self.value=nn.Linear(e,d,bias=False)
        self.output=nn.Linear(d,d,bias=False)
        self.role=nn.Parameter(torch.randn(2,d)*.02)
        self.c_index=nn.Parameter(torch.randn(cfg.content_tokens,d)*.02)
        self.d_index=nn.Parameter(torch.randn(cfg.dynamic_tokens,d)*.02)
        self.content_summary=nn.Linear(d,d,bias=False)
        self.slot_seed=nn.Parameter(torch.randn(cfg.slots,d)/math.sqrt(d))
        self.birth_d=nn.Parameter(torch.randn(cfg.slots,cfg.dynamic_tokens,d)*.02)
        self.birth_evidence=nn.Linear(e,cfg.dynamic_tokens*d,bias=False)
        self.descriptor=nn.Linear(2*d,d)
        self.ownership_key=nn.Linear(e,d)
        self.dustbin=nn.Parameter(torch.randn(d)/math.sqrt(d))
        self.register_buffer('temperature',torch.tensor(cfg.ownership_temperature))
        self.register_buffer('frequencies',torch.tensor([1.,2.,4.,8.]),persistent=False)
        self.geometry=nn.Sequential(nn.Linear(90,d),nn.SiLU(),nn.Linear(d,d))
        self.d_update=nn.Sequential(nn.Linear(3*d+4,2*d),nn.SiLU(),nn.Linear(2*d,d))
        self.d_gate=nn.Linear(3*d+4,d)
        self.c_update=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d))
        self.reliability=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,1))

    def ownership(self,state,ev,decoder,*,proposals=False):
        desc=self.descriptor(torch.cat((state.c.mean(-2),state.d.mean(-2)),-1))
        # Keep the birth identity descriptor until its first content write.
        # Switching immediately to a random projection of empty C can route
        # every newborn back to dustbin before either read gets evidence.
        desc=torch.where(state.content_initialized[...,None],desc,self.slot_seed[None])
        keys=self.ownership_key(ev['tokens'])
        scores=torch.einsum('bnd,bkd->bnk',keys,torch.cat((desc,self.dustbin[None,None].expand(len(desc),1,-1)),1))/math.sqrt(self.cfg.width)
        centers=decoder.controls(state.d)[...,:3,3]
        origins,directions=ev['origins'],ev['directions']
        ray_valid=torch.isfinite(origins).all(-1)&torch.isfinite(directions).all(-1)
        if not ray_valid.all():
            # Sanitize before subtraction/dot products. Masking their results
            # cannot repair NaN derivatives already created by that arithmetic.
            origins=torch.nan_to_num(origins.detach(),nan=0.,posinf=0.,neginf=0.)
            directions=torch.nan_to_num(directions.detach(),nan=0.,posinf=0.,neginf=0.)
        offset=centers[:,None]-origins[:,:,None]
        ray=directions[:,:,None]
        distance=(offset-(offset*ray).sum(-1,keepdim=True)*ray).square().sum(-1)
        bias=-.1*torch.log1p(torch.nan_to_num(distance,nan=0.,posinf=1e4).clamp(max=1e4))*state.allocated[:,None]*ray_valid[...,None]
        scores=scores/self.temperature.clamp_min(.1)+torch.cat((bias,torch.zeros_like(bias[...,:1])),-1)
        if not proposals:
            allowed=torch.cat((state.allocated,torch.ones_like(state.allocated[:,:1])),1)
            scores=scores.masked_fill(~allowed[:,None],-1e9)
        return scores.softmax(-1)*ev['valid'][...,None],scores

    def positional(self,canonical,origins,directions,measured):
        g=torch.cat((canonical,origins,directions,measured[...,None].float()),-1)
        g=torch.asinh(g/self.cfg.atlas_extent)
        angles=g[...,None]*self.frequencies
        return self.geometry(torch.cat((g,angles.sin().flatten(-2),angles.cos().flatten(-2)),-1))

    def read(self,q,k,v,weights,valid,*,return_attention=False):
        b,slots,nq,d=q.shape
        original_n=k.shape[-2]
        count=min(self.cfg.candidates,original_n)
        indices=weights.masked_fill(~valid,-1).topk(count,dim=-1).indices
        k=k.gather(-2,indices[...,None].expand(-1,-1,-1,d))
        v=v.gather(-2,indices[...,None].expand(-1,-1,-1,d))
        weights=weights.gather(-1,indices)
        valid=valid.gather(-1,indices)
        heads=self.cfg.resampler_heads
        head=d//heads
        qs=q.reshape(b,slots,nq,heads,head).transpose(-3,-2)
        ks=k.reshape(b,slots,-1,heads,head).transpose(-3,-2)
        vs=v.reshape(b,slots,-1,heads,head).transpose(-3,-2)
        logits=qs@ks.transpose(-1,-2)/math.sqrt(head)
        logits=logits+weights.clamp_min(1e-12).log()[:,:,None,None]
        attn=logits.masked_fill(~valid[:,:,None,None],-1e9).softmax(-1)*valid[:,:,None,None]
        attn=attn/attn.sum(-1,keepdim=True).clamp_min(1e-12)
        result=self.output((attn@vs).transpose(-3,-2).reshape(b,slots,nq,d))
        if return_attention:
            support=attn.mean(-3)
            full=support.new_zeros(b,slots,nq,original_n)
            full=full.scatter(-1,indices[:,:,None].expand(-1,-1,nq,-1),support)
            return result,full
        return result

    def forward(self,state,ev,decoder,relations,dt):
        from .evidence import atlas_update
        b,n,_=ev['tokens'].shape
        k=self.cfg.slots
        hypotheses=ev['points'].shape[2]
        proposal,proposal_logits=self.ownership(state,ev,decoder,proposals=True)
        winners=F.one_hot(proposal.argmax(-1),k+1)[...,:k].bool()&ev['valid'][...,None]
        # Empty scenes need one bounded bootstrap proposal even if a random
        # dustbin logit initially beats every slot. Reserve one observed patch,
        # never a fabricated geometry/status label; later dustbin remains free.
        bootstrap=(~state.content_initialized.any(-1))&(~winners.any((1,2)))&ev['valid'].any(-1)
        if bootstrap.any():
            scores=proposal_logits[...,:k].detach().masked_fill(~ev['valid'][...,None],-torch.inf)
            chosen=scores.flatten(1).argmax(-1)
            reservation=F.one_hot(chosen,n*k).reshape(b,n,k).bool()
            winners=winners|(reservation&bootstrap[:,None,None])
        reserved=winners&(~state.content_initialized)[:,None]
        reserved_rows=reserved.any(-1,keepdim=True)
        births=(~state.allocated)&winners.any(1)
        mass=proposal[...,:k].sum(1)
        pooled=torch.einsum('bnk,bne->bke',proposal[...,:k],ev['tokens'])/mass.clamp_min(1e-12)[...,None]
        born=self.birth_d[None]+self.birth_evidence(pooled).reshape(b,k,self.cfg.dynamic_tokens,self.cfg.width)
        initialized=replace(state,d=torch.where(births[...,None,None],born,state.d),
                            slot_status=torch.where(births,torch.ones_like(state.slot_status),state.slot_status),slot_uid=state.slot_uid+births.long())
        ownership,logits=self.ownership(initialized,ev,decoder)
        winner=F.one_hot(ownership.argmax(-1),k+1)[...,:k].bool()&ev['valid'][...,None]
        winner=torch.where(reserved_rows,reserved,winner)
        observed=winner.any(1)&initialized.allocated
        base_key=self.key(self.evidence_norm(ev['tokens']))
        base_val=self.value(self.evidence_norm(ev['tokens']))
        keys=base_key[:,None,:,None].expand(-1,k,-1,hypotheses,-1).reshape(b,k,n*hypotheses,-1)
        values=base_val[:,None,:,None].expand(-1,k,-1,hypotheses,-1).reshape_as(keys)
        points=ev['points'].flatten(1,2)[:,None].expand(-1,k,-1,-1)
        origins=ev['origins'][:,:,None].expand(-1,-1,hypotheses,-1).flatten(1,2)[:,None].expand(-1,k,-1,-1)
        directions=ev['directions'][:,:,None].expand(-1,-1,hypotheses,-1).flatten(1,2)[:,None].expand(-1,k,-1,-1)
        measured=ev['measured'][:,:,None].expand(-1,-1,hypotheses).flatten(1,2)[:,None].expand(-1,k,-1)
        geometry_available=torch.isfinite(points).all(-1)&torch.isfinite(origins).all(-1)&torch.isfinite(directions).all(-1)
        # Invalid upstream geometry can itself have undefined derivatives. In
        # this exceptional path retain appearance gradients and detach geometry.
        if not geometry_available.all():
            points=torch.nan_to_num(points.detach(),nan=0.,posinf=0.,neginf=0.)
            origins=torch.nan_to_num(origins.detach(),nan=0.,posinf=0.,neginf=0.)
            directions=torch.nan_to_num(directions.detach(),nan=0.,posinf=0.,neginf=0.)
        def weights_and_valid(own):
            weights=(own[...,:k].transpose(1,2)[...,None]*ev['probabilities'][:,None]).flatten(-2)
            valid=(ev['valid'][:,:,None]&(ev['probabilities']>0)).flatten(1,2)[:,None].expand(-1,k,-1)
            return weights,valid
        weights,valid=weights_and_valid(ownership)
        transform,z=decoder.extended_controls(initialized.d)
        canonical=decoder.canonical(points,transform,z)
        rot=transform[...,:3,:3]
        local_orig=torch.einsum('bkji,bkqj->bkqi',rot,origins-transform[...,:3,3][:,:,None])
        local_dir=torch.einsum('bkji,bkqj->bkqi',rot,directions)
        finite=geometry_available&torch.isfinite(canonical).all(-1)&torch.isfinite(local_orig).all(-1)&torch.isfinite(local_dir).all(-1)
        geometry=self.positional(torch.nan_to_num(canonical),torch.nan_to_num(local_orig),torch.nan_to_num(local_dir),measured)
        # Appearance-only D read remains available when geometry is unavailable.
        dkeys=keys+torch.where(finite[...,None],geometry,torch.zeros_like(geometry))
        q=self.query(self.norm(initialized.d)+self.role[1]+self.d_index[None,None]+self.content_summary(self.norm(initialized.c).mean(-2))[:,:,None])
        read=self.read(q,dkeys,values,weights,valid)
        delta=torch.as_tensor(dt,device=state.d.device,dtype=torch.float32).expand(b)
        time=torch.stack((delta,torch.log1p(delta),delta.sin(),delta.cos()),-1)[:,None,None].expand(-1,k,self.cfg.dynamic_tokens,-1)
        inp=torch.cat((self.norm(initialized.d),read,relations[:,:,None].expand(-1,-1,self.cfg.dynamic_tokens,-1),time),-1)
        candidate=initialized.d+torch.sigmoid(self.d_gate(inp))*self.d_update(inp)
        d=torch.where(observed[...,None,None],candidate,initialized.d)
        corrected=replace(initialized,d=d)
        refined,refined_logits=self.ownership(corrected,ev,decoder)
        winning=F.one_hot(refined.argmax(-1),k+1)[...,:k].bool()&ev['valid'][...,None]
        winning=torch.where(reserved_rows,reserved,winning)
        observed_c=winning.any(1)&initialized.allocated
        transform,z=decoder.extended_controls(d)
        canonical=decoder.canonical(points,transform,z)
        local_orig=torch.einsum('bkji,bkqj->bkqi',transform[...,:3,:3],origins-transform[...,:3,3][:,:,None])
        local_dir=torch.einsum('bkji,bkqj->bkqi',transform[...,:3,:3],directions)
        # Canonical routing cannot let RGB train depth, ownership, or pose by
        # moving its write coordinates. Renderer still trains pose and warp.
        ckeys=keys+self.positional(torch.nan_to_num(canonical.detach()),torch.nan_to_num(local_orig.detach()),torch.nan_to_num(local_dir.detach()),measured)
        cweights,cvalid=weights_and_valid(refined.detach())
        cweights=cweights.detach()
        cvalid=cvalid&geometry_available&torch.isfinite(canonical).all(-1)&observed_c[...,None]
        # A newborn's first write reads its reserved support, so top-k cannot
        # discard the only bootstrap patch in favor of unassigned equal scores.
        first_support=winning.transpose(1,2)[...,None].expand(-1,-1,-1,hypotheses).flatten(-2)
        cvalid=cvalid&(initialized.content_initialized[...,None]|first_support)
        cq=self.query(self.norm(initialized.c)+self.role[0]+self.c_index[None,None])
        content_read,attention=self.read(cq,ckeys,values,cweights,cvalid,return_attention=True)
        content=self.c_update(content_read)
        support=(refined[...,:k]*winning).sum(1)
        entropy=-(refined.clamp_min(1e-12).log()*refined).sum(-1)
        ambiguity=(refined[...,:k]*entropy[...,None]).sum(1)/refined[...,:k].sum(1).clamp_min(1e-12)
        measured_fraction=(refined[...,:k]*winning*ev['measured'][...,None]).sum(1)/support.clamp_min(1e-12)
        assigned=(winning.transpose(1,2)[...,None]&(ev['probabilities'][:,None]>0)).flatten(-2)&cvalid
        token_support=(attention*assigned[:,:,None]).sum(-1,keepdim=True).detach()
        measured_support=measured&assigned
        token_measured=(attention*measured_support[:,:,None]).sum(-1,keepdim=True).detach()
        # Independent measurement residuals supervise reliability. Unmeasured
        # RGB-only content is writable but never invents a reliability label.
        target=ev.get('reliability_target',torch.zeros_like(ev['valid'],dtype=values.dtype)).detach()
        target_valid=ev.get('reliability_valid',torch.zeros_like(ev['valid'])).detach()
        target=target[:,:,None].expand(-1,-1,hypotheses).flatten(1,2)
        target_valid=target_valid[:,:,None].expand(-1,-1,hypotheses).flatten(1,2)[:,None]&assigned
        target_mass=(attention.detach()*target_valid[:,:,None]).sum(-1,keepdim=True)
        reliability_target=(attention.detach()*target[:,None,None]*target_valid[:,:,None]).sum(-1,keepdim=True)/target_mass.clamp_min(1e-12)
        token_ambiguity=ambiguity[:,:,None,None].expand_as(token_support).detach()
        uncertainty=-(ev['probabilities'].clamp_min(1e-12).log()*ev['probabilities']).sum(-1)
        uncertainty=uncertainty[:,:,None].expand(-1,-1,hypotheses).flatten(1,2)
        token_uncertainty=(attention.detach()*uncertainty[:,None,None]).sum(-1,keepdim=True).detach()
        # Inference uses observable support/provenance/ambiguity only. The
        # measured residual label above is supervision, never a gate feature.
        trust=torch.cat((token_support,token_ambiguity,token_measured,token_uncertainty),-1).detach()
        reliability=torch.sigmoid(self.reliability(trust))
        alpha=(token_support*reliability).detach()
        can_write=observed_c&cvalid.any(-1)
        merged=(1-alpha)*initialized.c+alpha*content
        merged=torch.where(initialized.content_initialized[...,None,None],merged,content)
        c=torch.where(can_write[...,None,None],merged,initialized.c)
        atlas_valid=(winning.transpose(1,2)[...,None]&(ev['probabilities'][:,None]>0)).flatten(-2)&cvalid
        bits=atlas_update(state.support_bits,canonical.detach(),atlas_valid,extent=self.cfg.atlas_extent)
        # Preserve the state schema's legacy p_exist field as a lifecycle
        # indicator, not an unsupervised learned probability/density channel.
        exist=(state.content_initialized|can_write).to(state.p_exist.dtype)
        output=replace(corrected,c=c,support_bits=bits,content_initialized=state.content_initialized|can_write,
                       slot_status=torch.where(observed_c,torch.ones_like(state.slot_status),corrected.slot_status),
                       last_seen=torch.where(observed_c,state.t[:,None],state.last_seen),
                       p_exist=torch.where(observed_c,exist,state.p_exist),
                       provenance=torch.where(can_write,measured_fraction,state.provenance))
        return output,{'ownership':ownership,'ownership_logits':logits,
                       'ownership_entropy':-(ownership.clamp_min(1e-12).log()*ownership).sum(-1),
                       'ownership_valid':ev['valid'],'content_ownership':refined,
                       'content_ownership_logits':refined_logits,'content_ownership_entropy':entropy,'births':births,
                       'observed':observed_c,'bootstrap':bootstrap,'support':support,'reliability':reliability,
                       'reliability_valid':(target_mass>1e-8)&can_write[:,:,None,None],
                       'reliability_target':reliability_target, 'content_support':token_support,
                       'content_attention':attention,'canonical_points':canonical,'canonical_valid':atlas_valid,
                       'candidate_dropped_mass':(cweights.sum(-1)-cweights.masked_fill(~cvalid,0).topk(min(self.cfg.candidates,cweights.shape[-1]),dim=-1).values.sum(-1)).clamp_min(0),
                       # Token assignments do not identify unseen object counts.
                       'unassigned_evidence_count':(refined.argmax(-1).eq(k)&ev['valid']).sum(-1),
                       'slot_capacity_full':output.allocated.all(-1),
                       'canonical_outside_fraction':((canonical.abs()>=self.cfg.atlas_extent).any(-1)&atlas_valid).float().sum()/atlas_valid.sum().clamp_min(1)}
