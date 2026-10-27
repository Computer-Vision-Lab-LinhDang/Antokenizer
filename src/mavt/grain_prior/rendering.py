"""Uniform midpoint quadrature with one termination distribution for all outputs."""
import torch
from . import geometry as g
from .query import ReadPacket


def make_samples(view,query):
    """Unit canonical rays. near/far are range in source gauge, output depth is camera z."""
    with g.fp32(query.uv.device):
        cam = query.camera
        ones = torch.ones(query.uv.shape[:2],device=query.uv.device)
        vcam = g.unproject(ones,cam.kappa,query.uv)
        vcam = vcam/vcam.norm(dim=-1,keepdim=True)
        direction = vcam @ cam.R
        origin = -(cam.t[:,None] @ cam.R)
        edges = torch.linspace(query.near,query.far,query.samples+1,device=query.uv.device)[None,None] * view.scale[:,None,None]
        mid = (edges[...,1:]+edges[...,:-1])*.5
        xyz = origin[:,:,None] + mid[...,None]*direction[:,:,None]
        z = mid*vcam[...,2,None]
        delta = (edges[...,1:]-edges[...,:-1]).expand_as(z)
        far = edges[...,-1]*vcam[...,2]
        return xyz,z,delta,far,direction


def compose_samples(density,delta,z,h,color,far,background_h,background_rgb):
    """All arrays [B,Q,S,...]. Background mass remains in primary depth and RGB.

    hit_depth is diagnostic only; it is zero on empty rays. Density is not occupancy.
    """
    with g.fp32(density.device):
        optical = density.float()*delta.float()
        alpha = -torch.expm1(-optical)
        prefix = torch.cat([torch.zeros_like(optical[...,:1]),optical[...,:-1]],-1).cumsum(-1)
        weights = torch.exp(-prefix)*alpha
        # exp(-total optical thickness) is stable even when opacity rounds to 1.
        bg = torch.exp(-optical.sum(-1))
        opacity = weights.sum(-1)
        depth_sum = (weights*z.float()).sum(-1)
        return dict(h=(weights[...,None]*h.float()).sum(-2)+bg[...,None]*background_h.float(),
                    depth=depth_sum+bg*far.float(),rgb_core=(weights[...,None]*color.float()).sum(-2)+bg[...,None]*background_rgb.float(),
                    weights=weights,opacity=opacity,hit_depth=depth_sum/opacity.clamp_min(1e-8))


def render_rays(field,view,query,cfg):
    xyz,z,delta,far,direction = make_samples(view,query)
    B,Q,S,_ = xyz.shape
    point = field(view,xyz.reshape(B,Q*S,3))
    h = point.h.reshape(B,Q,S,-1)
    density = point.density.reshape(B,Q,S)
    with g.fp32(h.device):
        color = field.radiance(torch.cat([h,direction[:,:,None].expand(B,Q,S,3)],-1)).sigmoid()
        out = compose_samples(density,delta,z,h,color,far,field.background_h,h.new_tensor(cfg.background_rgb))
    return ReadPacket(support=point.support.reshape(B,Q,S).any(-1),provenance=view.provenance,
                      readout_version='shared_field_v1',scale=view.scale,sample_xyz=xyz,sample_h=h,
                      sample_density=density,sample_z=z,**out)
