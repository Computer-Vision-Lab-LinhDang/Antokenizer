"""Typed, snapshot-owned query API. Coordinates are canonical; camera is canonical-to-camera.

Frozen dataclasses prevent field rebinding. Tensors belong to the snapshot (no aliases to the
caller's state); callers must treat tensor contents as read-only. No persistent decoder cache.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from . import geometry as g, schema as sc


def finite(x, name):
    if not torch.is_tensor(x) or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be a finite tensor")


@dataclass(frozen=True)
class Camera:
    kappa: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor

    def __post_init__(self):
        for name in ('kappa', 'R', 't'):
            x = getattr(self, name)
            finite(x, name)
            object.__setattr__(self, name, x.float().clone())
        b = self.kappa.shape[0]
        if self.kappa.shape != (b,4) or self.R.shape != (b,3,3) or self.t.shape != (b,3):
            raise ValueError('camera requires kappa[B,4], R[B,3,3], t[B,3]')
        eye = torch.eye(3, device=self.R.device).expand(b,-1,-1)
        if (self.kappa[:,:2] <= 0).any() or not torch.allclose(self.R.transpose(-1,-2) @ self.R, eye, atol=1e-4, rtol=1e-4) or not torch.allclose(torch.linalg.det(self.R), torch.ones(b,device=self.R.device), atol=1e-4):
            raise ValueError('camera needs positive focal and a proper rotation')


@dataclass(frozen=True)
class PointQuery:
    xyz: torch.Tensor
    time_id: torch.Tensor

    def __post_init__(self):
        finite(self.xyz, 'xyz'); finite(self.time_id, 'time_id')
        if self.xyz.ndim != 3 or self.xyz.shape[-1] != 3 or self.time_id.shape != self.xyz.shape[:1]:
            raise ValueError('point query requires xyz[B,Q,3] and time_id[B]')


@dataclass(frozen=True)
class RayQuery:
    camera: Camera
    uv: torch.Tensor                 # normalized pixel centres, explicit footprint/resolution outside core
    near: float                      # ray distances in source gauge units
    far: float
    samples: int
    time_id: torch.Tensor | None = None

    def __post_init__(self):
        finite(self.uv, 'uv')
        if self.uv.ndim != 3 or self.uv.shape[0] != self.camera.kappa.shape[0] or self.uv.shape[-1] != 2 or not self.uv.shape[1]:
            raise ValueError('uv must be nonempty [B,Q,2]')
        if ((self.uv < 0) | (self.uv > 1)).any():
            raise ValueError('query outside image')
        if not math.isfinite(self.near) or not math.isfinite(self.far) or not 0 < self.near < self.far or not isinstance(self.samples,int) or self.samples < 1:
            raise ValueError('invalid ray bounds/samples')


@dataclass(frozen=True)
class StateView:
    points: torch.Tensor
    features: torch.Tensor
    radius: torch.Tensor
    uv: torch.Tensor
    sigma: torch.Tensor
    scale: torch.Tensor
    c0: torch.Tensor
    z: torch.Tensor
    rho: torch.Tensor
    kappa0: torch.Tensor
    time_id: torch.Tensor
    record_id: int | None
    provenance: str
    innovation: torch.Tensor | None
    length: torch.Tensor | None
    record_camera: Camera
    motion: torch.Tensor

    def check_time(self, time_id):
        if time_id is not None and (time_id.shape != self.time_id.shape or not torch.equal(time_id, self.time_id)):
            raise ValueError('time is unavailable in this state view; interpolation is not defined')


@dataclass(frozen=True)
class ReadPacket:
    h: torch.Tensor                    # [B,Q,D] common hidden or integrated hidden
    support: torch.Tensor
    provenance: str
    readout_version: str
    scale: torch.Tensor
    depth: torch.Tensor | None = None
    rgb_core: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    opacity: torch.Tensor | None = None
    sample_xyz: torch.Tensor | None = None
    sample_h: torch.Tensor | None = None
    sample_density: torch.Tensor | None = None
    sample_z: torch.Tensor | None = None
    hit_depth: torch.Tensor | None = None


def prepare_state(model, state, record_id=None, mode='anchor'):
    """Derive one view without changing S; clone preserves gradients for fine-tuning.

    record_id is an index into the *retained* queue, not an episode frame index.
    Novel-view pilot is anchor-only. Observed records read only their own innovations.
    """
    if mode not in ('anchor','observed','novel_view'):
        raise ValueError('unsupported mode (future/spatial innovations are gated)')
    allowed = {'h','u','rho','sigma','f','c0','kappa0','zb','recs'}
    if set(state) != allowed:
        raise ValueError('state must contain only the serialized C/D fields')
    if mode != 'observed' and record_id is not None or mode == 'observed' and record_id is None:
        raise ValueError('record selection and mode disagree')
    if record_id is not None and (not isinstance(record_id,int) or not 0 <= record_id < len(state['recs'])):
        raise ValueError('record is no longer available')
    for flat in sc.serialize(model.cfg, state):
        sc.validate(model.cfg, flat)
    clone = lambda x: x.clone() if torch.is_tensor(x) else [{k:v.clone() for k,v in r.items()} for r in x]
    S = {k:clone(v) for k,v in state.items()}
    cfg, der = model.cfg, model.derived(S)
    B, dev = S['h'].shape[0], S['h'].device
    if record_id is None:
        z, M, xi, kappa = S['c0'], torch.zeros(B,cfg.n_motion,3,device=dev), torch.zeros(B,6,device=dev), S['kappa0']
        time_id, prov, r, L = S['h'][:,sc.H_T0], 'anchor', None, None
    else:
        rec = S['recs'][record_id]
        if (S['h'][:,sc.H_PROV+record_id] != 0).any():
            raise ValueError('predicted-state query awaits the codec consistency gate')
        z = model.replay({**S, 'recs':S['recs'][:record_id+1]})[-1]
        M, xi, kappa, r, L = (rec[k] for k in ('M','xi','kappa','r','L'))
        time_id, prov = S['h'][:,sc.H_T0]+S['h'][:,sc.H_TT+record_id], 'observed'
        if cfg.innovation_semantics == 'disabled' and (L != 0).any():
            raise ValueError('pilot requires L=0; legacy innovation rho is not spatial evidence')
    points = der['P'] + der['beta'] @ M.float()
    features = model.appear(S['f'].float(),S['u'],z,S['c0']) if cfg.memory else S['f'].float()
    radius = S['sigma'].float()/cfg.grid * der['d'] / S['kappa0'][:,:2].prod(-1).sqrt()[:,None]
    R,t = g.se3_exp(xi)
    return StateView(points, features, radius, S['u'], S['sigma'], der['scale'], S['c0'], z,
                     S['rho'],S['kappa0'],time_id,record_id,prov,r,L,Camera(kappa,R,t),M)
