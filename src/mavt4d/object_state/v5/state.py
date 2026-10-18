"""All persistent memory, including a packed, approximate spatial support atlas."""
from dataclasses import dataclass, fields, replace
import hashlib
import json
import torch
from torch import Tensor


@dataclass(frozen=True)
class V5State:
    c: Tensor
    d: Tensor
    camera: Tensor
    camera_valid: Tensor
    support_bits: Tensor
    content_initialized: Tensor
    slot_status: Tensor
    slot_uid: Tensor
    p_exist: Tensor
    last_seen: Tensor
    provenance: Tensor  # last canonical-write fraction of measured geometry
    t: Tensor
    metric_valid: Tensor
    frame_id: Tensor
    depth_anchor: Tensor
    depth_anchor_valid: Tensor
    scene_to_metric: Tensor  # external Sim(3); C/D and camera remain scene units
    config_digest: Tensor  # SHA256 architecture/config fingerprint [B,32]

    @classmethod
    def initial(cls, cfg, batch_size, *, device='cpu', time=0., frame_id=0):
        b, k, d = batch_size, cfg.slots, cfg.width
        if b < 1:
            raise ValueError('batch_size must be positive')
        def zero(*shape, dtype=torch.float32):
            return torch.zeros(*shape, device=device, dtype=dtype)
        t = torch.as_tensor(time, device=device, dtype=torch.float64).expand(b).clone()
        camera = zero(b, cfg.camera_views, 32)
        camera[..., :12] = torch.eye(4, device=device)[:3].reshape(12)
        return cls(zero(b,k,cfg.content_tokens,d), zero(b,k,cfg.dynamic_tokens,d), camera,
                   zero(b,cfg.camera_views,dtype=torch.bool), zero(b,k,64,dtype=torch.uint8),
                   zero(b,k,dtype=torch.bool), zero(b,k,dtype=torch.int64), zero(b,k,dtype=torch.int64),
                   zero(b,k), t[:,None].expand(b,k).clone(), zero(b,k), t,
                   zero(b,dtype=torch.bool), torch.as_tensor(frame_id,device=device,dtype=torch.int64).expand(b).clone(),
                   torch.ones(b,device=device),zero(b,dtype=torch.bool),
                   torch.eye(4,device=device).expand(b,4,4).clone(),
                   torch.tensor(list(hashlib.sha256(json.dumps(cfg.to_dict(),sort_keys=True).encode()).digest()),device=device,dtype=torch.uint8).expand(b,32).clone())

    @property
    def allocated(self):
        return self.slot_status > 0

    @property
    def native_d(self):
        return self.d.transpose(-1,-2)

    def tensor_dict(self):
        return {f.name: getattr(self,f.name) for f in fields(self)}

    def flat_tokens(self):
        return torch.cat((self.c,self.d),-2).flatten(1,2)

    @property
    def total_nbytes(self):
        return sum(x.numel()*x.element_size() for x in self.tensor_dict().values())

    @property
    def learned_nbytes(self):
        return sum(x.numel()*x.element_size() for x in (self.c,self.d,self.camera))

    def clone(self):
        return replace(self, **{k:v.clone() for k,v in self.tensor_dict().items()})

    def detach(self):
        return replace(self, **{k:v.detach() for k,v in self.tensor_dict().items()})

    def to(self,device):
        return replace(self, **{k:v.to(device) for k,v in self.tensor_dict().items()})

    def select(self,indices):
        return replace(self, **{k:v.index_select(0,indices) for k,v in self.tensor_dict().items()})

    def to_payload(self):
        return {'schema_version':5, **{k:v.detach().clone().contiguous() for k,v in self.tensor_dict().items()}}

    @classmethod
    def from_payload(cls,payload,cfg):
        if payload.get('schema_version') != 5 or set(payload) != {f.name for f in fields(cls)} | {'schema_version'}:
            raise ValueError('v5 state schema mismatch; legacy states cannot be resumed')
        state = cls(**{f.name:payload[f.name] for f in fields(cls)})
        state.validate(cfg)
        return state.clone()

    def validate(self,cfg):
        b = self.c.shape[0]
        reference = self.initial(cfg,b,device=self.c.device,time=self.t)
        for key,value in self.tensor_dict().items():
            expected = getattr(reference,key)
            if value.shape != expected.shape or value.dtype != expected.dtype or value.device != self.c.device:
                raise ValueError(f'invalid v5 state field {key}')
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f'nonfinite v5 state field {key}')

        if not torch.equal(self.config_digest,reference.config_digest):
            raise ValueError('state configuration fingerprint mismatch')
        if ((self.slot_status<0)|(self.slot_status>2)).any() or (self.slot_uid<0).any():
            raise ValueError('invalid slot lifecycle metadata')
        if ((self.p_exist<0)|(self.p_exist>1)).any() or (self.last_seen>self.t[:,None]).any():
            raise ValueError('invalid probability or last-seen timestamp')
        if (self.content_initialized & ~self.allocated).any():
            raise ValueError('content cannot belong to an unallocated slot')
        matrix=self.scene_to_metric[:,:3,:3]
        scale=matrix.norm(dim=-2)
        rotation=matrix/scale[:,None].clamp_min(1e-12)
        if (scale<=0).any() or not torch.allclose(scale,scale[:,:1].expand_as(scale),atol=1e-5,rtol=1e-5) or not torch.allclose(rotation.transpose(-1,-2)@rotation,torch.eye(3,device=matrix.device).expand_as(matrix),atol=1e-4,rtol=1e-4) or (torch.linalg.det(rotation)<=0).any() or not torch.allclose(self.scene_to_metric[:,3],reference.scene_to_metric[:,3]):
            raise ValueError('invalid scene-to-metric similarity')
