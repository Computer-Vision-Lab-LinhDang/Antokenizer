"""Canonical latent field; density has no access to the query camera or task identity."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import geometry as g
from .layers import attend, mlp2


@dataclass(frozen=True)
class PointPacket:
    h: torch.Tensor
    density: torch.Tensor
    support: torch.Tensor


class FieldLayer(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.heads = cfg.heads
        self.q = nn.Linear(cfg.dim,cfg.dim)
        self.kv = nn.Linear(cfg.dim,2*cfg.dim)
        self.out = nn.Linear(cfg.dim,cfg.dim)
        self.ref = mlp2(2*cfg.dim,cfg.dim,cfg.mlp_hidden)
        self.norm = nn.LayerNorm(cfg.dim,elementwise_affine=False)
        self.temperature = nn.Parameter(torch.zeros(()))

    def forward(self,h,tokens,bias,valid):
        k,v = self.kv(tokens).chunk(2,-1)
        norm = lambda x: F.normalize(x.unflatten(-1,(self.heads,-1)),dim=-1).flatten(-2)
        q = norm(self.q(self.norm(h))) * (8*self.temperature.sigmoid()) * (h.shape[-1]//self.heads)**.5
        e = self.out(attend(q,norm(k),v,self.heads,bias[:,None],valid,strict=True))
        e = e * valid.any(-1)[:,None,None]
        return h + self.ref(torch.cat([h,e],-1))


class CanonicalField(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.query = mlp2(g.pe_dim(3,cfg.pe_freq),cfg.dim,cfg.mlp_hidden)
        self.context = nn.Linear(2*cfg.d_coarse,cfg.dim)
        self.grains = nn.Linear(cfg.d_f+g.pe_dim(3,cfg.pe_freq),cfg.dim)
        self.layers = nn.ModuleList([FieldLayer(cfg) for _ in range(cfg.read_layers)])
        self.density = nn.Linear(cfg.dim,1)
        self.radiance = mlp2(cfg.dim+3,3,cfg.mlp_hidden)
        self.background_h = nn.Parameter(torch.zeros(cfg.dim))

    def forward(self,view,xyz):
        if view.innovation is not None and (view.length > 0).any():
            raise ValueError('spatial innovations are gated; field pilot requires L=0')
        if xyz.ndim != 3 or xyz.shape[0] != view.points.shape[0] or xyz.shape[-1] != 3 or not torch.isfinite(xyz).all():
            raise ValueError('field expects finite xyz[B,Q,3]')
        # Keep geometric kernels, density and accumulation in FP32 even under autocast.
        with g.fp32(xyz.device):
            scale = view.scale.float()[:,None,None]
            points = view.points.float()/scale
            widths = view.radius.float()/view.scale[:,None]
            valid = torch.isfinite(points).all(-1) & (widths > 0)
            tokens = self.grains(torch.cat([view.features.float(),g.fourier(points,self.cfg.pe_freq)],-1))
            context = self.context(torch.cat([view.c0.float().mean(1),view.z.float().mean(1)],-1))[:,None]
            packets = []
            for x in xyz.float().split(self.cfg.query_chunk,1):
                x = x/scale
                d2 = (x[:,:,None]-points[:,None]).square().sum(-1)
                bias = -d2/(2*widths[:,None].clamp_min(1e-6).square())
                support = ((d2 <= 9*widths[:,None].square()) & valid[:,None]).any(-1)
                h = self.query(g.fourier(x,self.cfg.pe_freq)) + context
                for layer in self.layers: h = layer(h,tokens,bias,valid)
                density = F.softplus(self.density(h).squeeze(-1))/view.scale[:,None]
                packets.append(PointPacket(h,density,support))
            return PointPacket(*(torch.cat([getattr(p,k) for p in packets],1) for k in ('h','density','support')))
