"""Shared query readers. Task selection occurs only after the common read packet is built."""
from dataclasses import fields, replace
import torch
import torch.nn as nn
from . import geometry as g
from .layers import mlp2
from .query import ReadPacket
from .readout import Readout, pool_geom, projected_radius


def join_packets(packets):
    """Concatenate ray chunks without rerunning task-specific attention."""
    first = packets[0]
    updates = {}
    for f in fields(first):
        value = getattr(first,f.name)
        if torch.is_tensor(value) and f.name != 'scale':
            updates[f.name] = torch.cat([getattr(p,f.name) for p in packets],1)
    return replace(first, **updates)


class PacketHeads(nn.Module):
    """These heads have no reference to S, a model encoder, or raw images."""
    def __init__(self,cfg,teacher_dim):
        super().__init__()
        self.feature = nn.Linear(cfg.dim,teacher_dim)
        self.rgb = mlp2(cfg.dim,3,cfg.mlp_hidden)
        self.depth = mlp2(cfg.dim,1,cfg.mlp_hidden)

    def forward(self,packet,tasks=('rgb','depth','feature')):
        if not tasks or len(set(tasks)) != len(tasks) or set(tasks)-{'rgb','depth','feature'}:
            raise ValueError('tasks must be a nonempty unique subset of rgb/depth/feature')
        out = {}
        for task in tasks:
            if task == 'feature': out[task] = self.feature(packet.h).float()
            elif task == 'rgb': out[task] = packet.rgb_core if packet.rgb_core is not None else self.rgb(packet.h).float().sigmoid()
            else: out[task] = packet.depth if packet.depth is not None else self.depth(packet.h).float().squeeze(-1).clamp(-20,20).exp()*packet.scale[:,None]
        return out


class SharedQueryDecoder(nn.Module):
    def __init__(self,cfg,teacher_dim):
        super().__init__()
        self.cfg = cfg
        self.heads = PacketHeads(cfg,teacher_dim)
        if cfg.readout_version == 'shared_ray_v1':
            self.ray = Readout(cfg,teacher_dim)
        elif cfg.readout_version == 'shared_field_v1':
            from .field import CanonicalField
            self.field = CanonicalField(cfg)
        else:
            raise ValueError('use the legacy model decoder for legacy checkpoints')

    def decode(self,packet,tasks=('rgb','depth','feature')):
        if packet.readout_version != self.cfg.readout_version:
            raise ValueError('packet/readout version mismatch')
        return self.heads(packet,tasks)

    def read_points(self,view,query):
        if self.cfg.readout_version != 'shared_field_v1':
            raise ValueError('direct ray control has no point-field contract')
        view.check_time(query.time_id)
        return self.field(view,query.xyz)

    def read_rays(self,view,query,chunk=None):
        view.check_time(query.time_id)
        chunk = self.cfg.query_chunk if chunk is None else chunk
        if chunk < 1: raise ValueError('chunk must be positive')
        if query.uv.shape[1] > chunk:
            return join_packets([self.read_rays(view,replace(query,uv=u),chunk) for u in query.uv.split(chunk,1)])
        if self.cfg.readout_version == 'shared_field_v1':
            from .rendering import render_rays
            return render_rays(self.field,view,query,self.cfg)
        cam = query.camera
        Pcam = view.points @ cam.R.transpose(-1,-2) + cam.t[:,None]
        uh, depth, valid = g.project(Pcam,cam.kappa,self.cfg.z_eps)
        pg = pool_geom(g.se3_log(cam.R,cam.t),cam.kappa,view.kappa0,view.rho,view.motion,valid)
        sigma = view.sigma
        if self.cfg.read_radius_scale:
            sigma = projected_radius(view.sigma,view.points[...,2],depth,view.kappa0,cam.kappa)
        o = self.ray.base(view.features,view.points,view.uv,uh,depth,sigma,valid,view.scale,
                          view.c0,view.z,pg,uq=query.uv)
        h = o['a']
        if view.innovation is not None:
            if self.cfg.innovation_semantics != 'shared_hidden_v1' and (view.length > 0).any():
                raise ValueError('innovation requires shared_hidden_v1 semantics')
            if (view.length > 0).any() and any(not torch.equal(getattr(cam,k),getattr(view.record_camera,k)) for k in ('kappa','R','t')):
                raise ValueError('image innovations may only be queried in their own camera')
            _, dh = self.ray.innovation(h,o['pe'],view.innovation,view.length,True,uq=query.uv)
            h = h + dh
        return ReadPacket(h,o['support_v2'],view.provenance,self.cfg.readout_version,view.scale)
