"""Causal unified filtering: one encoder, camera correction, then two reads."""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..observation import SparseObservationEncoder, ObservationStep
from ..decoder import _rotation_6d
from .config import V5Config
from .state import V5State
from .dynamics import CConditionedDynamics
from .decoder import V5Readout
from .resampler import FunctionalResampler


class V5Model(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        self.config=config or V5Config()
        cfg=self.config
        self.observer=SparseObservationEncoder(cfg.encoder_config())
        self.decoder=V5Readout(cfg)
        self.dynamics=CConditionedDynamics(cfg)
        self.resampler=FunctionalResampler(cfg)
        self.depth_head=nn.Sequential(nn.LayerNorm(cfg.evidence_width),nn.Linear(cfg.evidence_width,cfg.evidence_width),nn.SiLU(),nn.Linear(cfg.evidence_width,cfg.depth_hypotheses*2))
        self.camera_head=nn.Sequential(nn.LayerNorm(cfg.evidence_width),nn.Linear(cfg.evidence_width,cfg.width),nn.SiLU(),nn.Linear(cfg.width,23))
        nn.init.zeros_(self.camera_head[-1].weight)
        nn.init.zeros_(self.camera_head[-1].bias)

    def initial_state(self,batch_size,*,device=None,time=0.,frame_id=0):
        return V5State.initial(self.config,batch_size,device=device or next(self.parameters()).device,time=time,frame_id=frame_id)

    def camera_parameters(self,state,view=0):
        if not 0<=view<self.config.camera_views:
            raise ValueError('camera view outside state capacity')
        values=state.camera[:,view]
        top=values[:,:12].reshape(-1,3,4)
        bottom=torch.zeros_like(top[:,:1]); bottom[:,0,3]=1
        pose=torch.cat((top,bottom),1)
        fx,fy,cx,cy=values[:,12:16].unbind(-1)
        zero=torch.zeros_like(fx); one=torch.ones_like(fx)
        intrinsic=torch.stack((fx,zero,cx,zero,fy,cy,zero,zero,one),-1).reshape(-1,3,3)
        return intrinsic,pose

    def predict(self,state,dt,action=None):
        with torch.autocast(device_type=state.c.device.type,enabled=False):
            return self.dynamics(state,dt,self.decoder.relation_controls(state.d),action)

    def read(self,state,query,**kwargs):
        with torch.autocast(device_type=state.c.device.type,enabled=False):
            return self.decoder(state,query,**kwargs)

    def rollout(self,state,dts,actions=None):
        result=[]
        for index,dt in enumerate(dts):
            state=self.predict(state,dt,None if actions is None else actions[index])
            result.append(state)
        return tuple(result)

    def _proposals(self,state,obs,encoded):
        cfg=self.config
        b,views,_,_,height,width=obs.images.shape
        if views>cfg.camera_views:
            raise ValueError('observation camera count exceeds bounded state')
        gh,gw=math.ceil(height/cfg.patch_size),math.ceil(width/cfg.patch_size)
        n=gh*gw
        tokens=encoded.tokens.float()
        valid=encoded.valid.reshape(b,views,n)
        pooled=(tokens.reshape(b,views,n,-1)*valid[...,None]).sum(2)/valid.sum(2).clamp_min(1)[...,None]
        proposals=self.camera_head(pooled)
        prior=state.camera[:,:views]
        prior_valid=state.camera_valid[:,:views]
        identity6=proposals.new_tensor([1.,0.,0.,0.,1.,0.])
        delta_rotation=_rotation_6d(identity6+.05*proposals[...,:6])
        prior_pose=prior[...,:12].reshape(b,views,3,4)
        rotation=prior_pose[...,:3]@delta_rotation
        translation=prior_pose[...,3]+.1*proposals[...,6:9]
        pose=torch.cat((rotation,translation[...,None]),-1)
        identity=torch.eye(4,device=tokens.device,dtype=tokens.dtype)[:3]
        # First RGB-only camera fixes sequence gauge; later corrections are learned.
        pose=torch.where(prior_valid[...,None,None],pose,identity)
        default_k=proposals.new_tensor([max(height,width),max(height,width),width/2,height/2])
        old_k=torch.where(prior_valid[...,None],prior[...,12:16],default_k)
        focal=old_k[...,:2]*torch.exp(.1*torch.tanh(proposals[...,9:11]))
        principal=old_k[...,2:4]+.05*proposals.new_tensor([width,height])*torch.tanh(proposals[...,11:13])
        camera_k=torch.cat((focal,principal),-1)
        camera_estimate=pose
        intrinsic_estimate=camera_k
        measured_cam=torch.zeros(b,views,device=tokens.device,dtype=torch.bool)
        if obs.intrinsics is not None and obs.camera_c2w is not None:
            camera_mask=obs.camera_valid if obs.camera_valid is not None else obs.geometry_valid
            measured_cam=torch.ones_like(measured_cam) if camera_mask is None else camera_mask
            pose=torch.where(measured_cam[...,None,None],obs.camera_c2w[...,:3,:].float(),pose)
            mk=torch.stack((obs.intrinsics[...,0,0],obs.intrinsics[...,1,1],obs.intrinsics[...,0,2],obs.intrinsics[...,1,2]),-1).float()
            camera_k=torch.where(measured_cam[...,None],mk,camera_k)
        # The current ray query has no target-view photometric identity. Keep
        # reserved exposure/white-balance channels fixed at zero, with no
        # learned output weights for an otherwise unused nuisance prediction.
        packed=torch.cat((pose.flatten(-2),camera_k,.1*torch.tanh(proposals[...,13:19]),
                          torch.zeros_like(proposals[...,19:23]),torch.sigmoid(proposals[...,19:23]),torch.zeros_like(proposals[...,:2])),-1)
        view_observed=valid.any(-1)
        camera=state.camera.clone()
        camera=torch.cat((torch.where(view_observed[...,None],packed,prior),camera[:,views:]),1)
        camera_valid=torch.cat((prior_valid|view_observed,state.camera_valid[:,views:]),1)
        yy=(torch.arange(gh,device=tokens.device)*cfg.patch_size+cfg.patch_size//2).clamp(max=height-1)
        xx=(torch.arange(gw,device=tokens.device)*cfg.patch_size+cfg.patch_size//2).clamp(max=width-1)
        y,x=torch.meshgrid(yy,xx,indexing='ij')
        pixels=torch.stack((x.flatten()+.5,y.flatten()+.5),-1)
        ray_xy=(pixels[None,None]-camera_k[:,:,None,2:])/camera_k[:,:,None,:2]
        ray_camera=torch.cat((ray_xy,torch.ones_like(ray_xy[...,:1])),-1)
        ray_scene=torch.einsum('bvij,bvnj->bvni',pose[...,:3],ray_camera).reshape(b,views*n,3)
        origins=pose[...,3][:,:,None].expand(-1,-1,n,-1).reshape(b,views*n,3)
        head=self.depth_head(tokens)
        raw_depth=F.softplus(head[...,:cfg.depth_hypotheses])+.05
        probabilities=head[...,cfg.depth_hypotheses:].softmax(-1)
        anchor=(raw_depth.mean(-1)*encoded.valid).sum(-1)/encoded.valid.sum(-1).clamp_min(1)
        anchor=torch.where(state.depth_anchor_valid,state.depth_anchor,anchor.detach()).clamp_min(.05)
        estimated=raw_depth/anchor[:,None,None]
        measured=torch.zeros_like(encoded.valid)
        target_depth=torch.zeros_like(encoded.valid,dtype=torch.float32)
        if obs.depth is not None:
            dv=(torch.isfinite(obs.depth)&(obs.depth>0)) if obs.depth_valid is None else obs.depth_valid
            target_depth=obs.depth[:,:,0,y,x].reshape(b,-1).float()
            measured=dv[:,:,0,y,x].reshape(b,-1)&encoded.valid&measured_cam[:,:,None].expand(-1,-1,n).reshape(b,-1)
            target_depth=torch.where(measured,target_depth,0.)
        # Physical depth on metric cameras; otherwise one persistent relative gauge.
        # Pose calibration does not establish meters. Camera/depth values with
        # a known pose share its scene gauge even when reconstructed/nonmetric.
        calibrated=measured_cam[:,:,None].expand(-1,-1,n).reshape(b,-1)
        predicted=torch.where(calibrated[...,None],raw_depth,estimated)
        actual=torch.where(measured[...,None],target_depth[...,None],predicted)
        one_hot=torch.zeros_like(probabilities); one_hot[...,0]=1
        probability=torch.where(measured[...,None],one_hot,probabilities)
        points=origins[:,:,None]+ray_scene[:,:,None]*actual[...,None]
        next_state=replace(state,camera=camera,camera_valid=camera_valid,depth_anchor=anchor,
                           depth_anchor_valid=state.depth_anchor_valid|encoded.valid.any(-1))
        ev={'tokens':tokens,'valid':encoded.valid,'origins':origins,'directions':F.normalize(ray_scene,dim=-1),
            'points':points,'probabilities':probability,'measured':measured,
            'reliability_target':torch.exp(-((predicted*probabilities).sum(-1).clamp_min(1e-6).log()-target_depth.clamp_min(1e-6).log()).abs()/.2).detach(),
            'reliability_valid':measured}
        diag={'proposal_depth':predicted,'depth_prob':probabilities,'geometry_measured':measured,
              'measured_depth':target_depth,'camera_proposal':packed,'depth_anchor':anchor,
              'proposal_points':origins[:,:,None]+ray_scene[:,:,None]*predicted[...,None],
              'source_camera_valid':measured_cam,
              'camera_estimate_before_metadata':camera_estimate,
              'camera_intrinsics_before_metadata':intrinsic_estimate,
              'source_metric_valid':torch.zeros_like(measured_cam) if obs.metric_valid is None else obs.metric_valid}
        return next_state,ev,diag

    def observe(self,state,observation,action=None,*,return_diagnostics=False):
        delta=observation.time.double()-state.t
        prior=self.predict(state,delta,action)
        if observation.valid is None:
            available=torch.ones(len(state.c),device=state.c.device,dtype=torch.bool)
        else:
            available=observation.valid[:,:,-1].flatten(1).any(-1)
        indices=available.nonzero().flatten()
        if not len(indices):
            return (prior,{}) if return_diagnostics else prior
        selected=prior.select(indices)
        obs=ObservationStep(**{key:value.index_select(0,indices) if isinstance(value,torch.Tensor) else value for key,value in vars(observation).items()})
        from .gauge import prepare_observation_gauge
        selected,obs=prepare_observation_gauge(selected,obs)
        with torch.autocast(device_type=state.c.device.type,dtype=torch.bfloat16,enabled=state.c.device.type=='cuda'):
            # One endpoint per encoded tubelet: absolute wall-clock phase is
            # meaningless. Recurrence receives only elapsed time below.
            encoded=self.observer(replace(obs,time=torch.zeros_like(obs.time)))
        with torch.autocast(device_type=state.c.device.type,enabled=False):
            selected,ev,diag=self._proposals(selected,obs,encoded)
            _,relations=self.dynamics.relations(selected,self.decoder.relation_controls(selected.d))
            result,details=self.resampler(selected,ev,self.decoder,relations,delta.index_select(0,indices))
        result=replace(prior,**{key:value.index_copy(0,indices,result.tensor_dict()[key]) for key,value in prior.tensor_dict().items()})
        diagnostics={}
        for key,value in {**diag,**details}.items():
            # Keep the actual live logits: retaining grad on an index_copy
            # descendant unused by the task loss would report a false zero.
            if key not in ('ownership_logits','content_ownership_logits') and value.ndim and value.shape[0]==len(indices):
                value=value.new_zeros(len(state.c),*value.shape[1:]).index_copy(0,indices,value)
            diagnostics[key]=value
        diagnostics['ownership_logits_batch_indices']=indices
        return (result,diagnostics) if return_diagnostics else result

    forward=observe
