"""C-conditioned block generators; raw D never enters coefficient prediction."""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F


class CConditionedDynamics(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        z = cfg.width
        self.content_norm = nn.LayerNorm(z)
        self.relation_input = nn.Linear(z+28+1,z)
        self.qkv = nn.Linear(z,3*z)
        self.transition = nn.Sequential(nn.Linear(2*z+cfg.action_dim,2*z),nn.SiLU(),nn.Linear(2*z,z),nn.SiLU())
        self.omega = nn.Linear(z,cfg.dynamic_tokens*z//2)
        self.kappa = nn.Linear(z,cfg.dynamic_tokens*z//2)
        self.force_u = nn.Linear(z,cfg.write_rank*z)
        self.force_v = nn.Linear(z,cfg.write_rank*cfg.dynamic_tokens)
        nn.init.zeros_(self.omega.weight)
        nn.init.zeros_(self.omega.bias)
        nn.init.zeros_(self.kappa.weight)
        nn.init.constant_(self.kappa.bias,math.log(math.expm1(.05)))
        # Only one factor zero: forcing starts at zero without dead gradients.
        nn.init.zeros_(self.force_u.weight)
        nn.init.zeros_(self.force_u.bias)

    def relations(self,state,controls):
        c = self.content_norm(state.c).mean(-2)
        x = self.relation_input(torch.cat((c,controls,state.allocated[...,None].float()),-1))
        q,k,v = self.qkv(x).chunk(3,-1)
        scores = (q @ k.transpose(-1,-2)/math.sqrt(self.cfg.width)).masked_fill(~state.allocated[:,None],-1e9).softmax(-1)
        scores = scores*state.allocated[:,None]
        scores = scores/scores.sum(-1,keepdim=True).clamp_min(1e-12)
        return c, scores@v

    def coefficients(self,state,controls,action=None):
        c,rel = self.relations(state,controls)
        args=[c,rel]
        if self.cfg.action_dim:
            if action is None or action.shape != (len(c),self.cfg.action_dim):
                raise ValueError('action required with configured dimension')
            args.append(action[:,None].expand(-1,self.cfg.slots,-1).float())
        elif action is not None:
            raise ValueError('action-free model does not accept actions')
        h=self.transition(torch.cat(args,-1))
        shape=(*h.shape[:2],self.cfg.dynamic_tokens,self.cfg.width//2)
        omega=self.omega(h).reshape(shape)
        kappa=F.softplus(self.kappa(h)).reshape(shape)
        u=self.force_u(h).reshape(*h.shape[:2],self.cfg.write_rank,self.cfg.width)
        v=self.force_v(h).reshape(*h.shape[:2],self.cfg.write_rank,self.cfg.dynamic_tokens)
        force=torch.einsum('bkrz,bkrn->bknz',u,v)/math.sqrt(self.cfg.write_rank)
        return omega,kappa,force

    @staticmethod
    def frozen_step(h,omega,kappa,forcing,dt):
        elapsed=torch.as_tensor(dt,device=h.device,dtype=h.dtype)
        if elapsed.ndim == 1:
            elapsed=elapsed[:,None,None,None]
        angle=omega*elapsed
        decay=torch.exp(-kappa*elapsed)
        a=decay*angle.cos()
        b=decay*angle.sin()
        den=kappa.square()+omega.square()
        # Integral exp(sG)ds = A I + B J, J=[[0,1],[-1,0]].
        safe=den.clamp_min(1e-12)
        gain_a=(kappa*(1-a)+omega*b)/safe
        gain_b=(omega*(1-a)-kappa*b)/safe
        # Select by dimensionless ||G dt||, not rates alone.
        small=den*elapsed.square()<1e-6
        ga_series=elapsed-.5*kappa*elapsed.square()+(kappa.square()-omega.square())*elapsed**3/6
        gb_series=.5*omega*elapsed.square()-kappa*omega*elapsed**3/3
        gain_a=torch.where(small,ga_series,gain_a)
        gain_b=torch.where(small,gb_series,gain_b)
        hp=h.reshape(*h.shape[:-1],-1,2)
        fp=forcing.reshape_as(hp)
        rotate=lambda x: torch.stack((x[...,1],-x[...,0]),-1)
        out=a[...,None]*hp+b[...,None]*rotate(hp)+gain_a[...,None]*fp+gain_b[...,None]*rotate(fp)
        return out.flatten(-2)

    def forward(self,state,dt,controls,action=None):
        delta=torch.as_tensor(dt,device=state.d.device,dtype=torch.float64).expand(len(state.d))
        if not torch.isfinite(delta).all() or (delta<0).any():
            raise ValueError('dt must be finite and nonnegative')
        omega,kappa,force=self.coefficients(state,controls,action)
        next_d=self.frozen_step(state.d,omega,kappa,force,delta.float())
        changed=state.allocated & (delta>0)[:,None]
        # Camera velocity is explicit state, with no hidden recurrent cache.
        camera=state.camera.clone()
        pose=camera[...,:12].reshape(*camera.shape[:2],3,4).clone()
        angular=camera[...,19:22]*delta.float()[:,None,None]
        ax,ay,az=angular.unbind(-1)
        zero=torch.zeros_like(ax)
        skew=torch.stack((zero,-az,ay,az,zero,-ax,-ay,ax,zero),-1).reshape(*angular.shape[:-1],3,3)
        rotation=pose[...,:3,:3]@torch.linalg.matrix_exp(skew)
        translation=pose[...,:3,3]+camera[...,16:19]*delta.float()[:,None,None]
        pose=torch.cat((rotation,translation[...,None]),-1)
        camera=torch.cat((pose.flatten(-2),camera[...,12:]),-1)
        return replace(state,d=torch.where(changed[...,None,None],next_d,state.d),camera=camera,
                       slot_status=torch.where(changed,torch.full_like(state.slot_status,2),state.slot_status),t=state.t+delta)
