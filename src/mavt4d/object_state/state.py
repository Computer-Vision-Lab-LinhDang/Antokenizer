"""Complete stream state and transient observation evidence; no hidden cache."""

from dataclasses import dataclass, fields, replace
from typing import ClassVar

import torch
from torch import Tensor

from .config import ObjectStateConfig


@dataclass(frozen=True)
class Evidence:
    tokens: Tensor  # [B,N,E]
    valid: Tensor  # [B,N] availability
    ray_origins: Tensor  # [B,N,3], endpoint reference frame
    ray_directions: Tensor  # [B,N,3], endpoint reference frame
    time: Tensor  # [B], physical end timestamp
    points: Tensor | None = None  # [B,N,3], input measurements only
    point_valid: Tensor | None = None
    geometry_valid: Tensor | None = None

    def validate(self, config: ObjectStateConfig):
        if self.tokens.ndim != 3 or self.tokens.shape[-1] != config.evidence_width:
            raise ValueError("evidence tokens must be [B,N,evidence_width]")
        b, n, _ = self.tokens.shape
        if b < 1 or n < 1 or not self.tokens.is_floating_point():
            raise ValueError("evidence must be nonempty floating point")
        if self.valid.shape != (b, n) or self.valid.dtype != torch.bool:
            raise ValueError("evidence validity must be boolean [B,N]")
        for name in ("ray_origins", "ray_directions"):
            value = getattr(self, name)
            if value.shape != (b, n, 3) or not value.is_floating_point():
                raise ValueError(f"{name} must be floating [B,N,3]")
        if (
            self.time.shape != (b,)
            or not self.time.is_floating_point()
            or not torch.isfinite(self.time).all()
        ):
            raise ValueError("evidence time must be finite [B]")
        for name in ("point_valid", "geometry_valid"):
            value = getattr(self, name)
            if value is not None and (
                value.shape != (b, n) or value.dtype != torch.bool
            ):
                raise ValueError(f"{name} must be boolean [B,N]")
        if (self.points is None) != (self.point_valid is None):
            raise ValueError("points and point_valid must be provided together")
        if self.points is not None and (
            self.points.shape != (b, n, 3) or not self.points.is_floating_point()
        ):
            raise ValueError("points must be floating [B,N,3]")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor) and value.device != self.tokens.device:
                raise ValueError("all evidence tensors must share a device")
        if not torch.isfinite(self.tokens[self.valid]).all():
            raise ValueError("available evidence tokens must be finite")
        geo = (
            self.valid
            if self.geometry_valid is None
            else self.valid & self.geometry_valid
        )
        if (
            not torch.isfinite(self.ray_origins[geo]).all()
            or not torch.isfinite(self.ray_directions[geo]).all()
        ):
            raise ValueError("valid ray geometry must be finite")
        if (self.ray_directions[geo].norm(dim=-1) <= 0).any():
            raise ValueError("valid ray directions must be nonzero")
        if (
            self.points is not None
            and not torch.isfinite(self.points[self.valid & self.point_valid]).all()
        ):
            raise ValueError("valid input points must be finite")

    def select(self, indices: Tensor):
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).index_select(0, indices)
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
        )


@dataclass(frozen=True)
class ObjectState:
    c: Tensor
    d: Tensor
    c_bg: Tensor
    slot_status: Tensor  # int64: 0 empty, 1 observed, 2 unobserved
    slot_uid: Tensor  # monotonically incremented generation per slot
    p_exist: Tensor
    last_seen: Tensor  # float64; finite initial value, status says unseen
    t: Tensor  # float64
    geometry_mode: Tensor  # int64: 0 metric_rigid; 1 image_proxy reserved
    frame_id: Tensor  # int64 reference/gauge ID, no learned payload
    SCHEMA_VERSION: ClassVar[int] = 1

    @classmethod
    def initial(
        cls,
        config: ObjectStateConfig,
        batch_size: int,
        *,
        device="cpu",
        time=0.0,
        frame_id=0,
    ):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        b, k, z = batch_size, config.slots, config.width

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, device=device, dtype=dtype)

        t = torch.as_tensor(time, device=device, dtype=torch.float64).expand(b).clone()
        result = cls(
            zeros(b, k, config.content_tokens, z),
            zeros(b, k, config.dynamic_tokens, z),
            zeros(b, config.background_tokens, z),
            zeros(b, k, dtype=torch.int64),
            zeros(b, k, dtype=torch.int64),
            zeros(b, k),
            t[:, None].expand(b, k).clone(),
            t,
            zeros(b, dtype=torch.int64),
            torch.as_tensor(frame_id, device=device, dtype=torch.int64)
            .expand(b)
            .clone(),
        )
        result.validate(config)
        return result

    @property
    def native_d(self):
        return self.d.transpose(-1, -2)

    @property
    def allocated(self):
        return self.slot_status > 0

    @property
    def learned_nbytes(self):
        return sum(x.numel() * x.element_size() for x in (self.c, self.d, self.c_bg))

    @property
    def total_nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.tensor_dict().values())

    @property
    def retained_storage_nbytes(self):
        storages = {
            x.untyped_storage().data_ptr(): x.untyped_storage().nbytes()
            for x in self.tensor_dict().values()
        }
        return sum(storages.values())

    def flat_tokens(self):
        objects = torch.cat((self.c, self.d), dim=-2).flatten(1, 2)
        return torch.cat((objects, self.c_bg), dim=1)

    def tensor_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def clone(self):
        return replace(
            self, **{name: value.clone() for name, value in self.tensor_dict().items()}
        )

    def detach(self):
        return replace(
            self, **{name: value.detach() for name, value in self.tensor_dict().items()}
        )

    def to(self, device):
        return replace(
            self,
            **{name: value.to(device) for name, value in self.tensor_dict().items()},
        )

    def select(self, indices: Tensor):
        return replace(
            self,
            **{
                name: value.index_select(0, indices)
                for name, value in self.tensor_dict().items()
            },
        )

    def to_payload(self):
        # Compact copies prevent a state snapshot from retaining observation/history storage.
        return {
            "schema_version": self.SCHEMA_VERSION,
            **{
                name: value.detach().clone().contiguous()
                for name, value in self.tensor_dict().items()
            },
        }

    @classmethod
    def from_payload(cls, payload, config: ObjectStateConfig):
        expected = {f.name for f in fields(cls)} | {"schema_version"}
        if set(payload) != expected:
            raise ValueError("serialized state fields do not match whitelist")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("serialized state schema mismatch")
        result = cls(**{f.name: payload[f.name] for f in fields(cls)})
        result.validate(config)
        return result.clone()

    def validate(self, config: ObjectStateConfig):
        if not isinstance(self.c, Tensor) or self.c.ndim != 4:
            raise ValueError("c must be [B,K,NC,Z]")
        b = self.c.shape[0]
        k = config.slots
        if b < 1:
            raise ValueError("state batch must be nonempty")
        shapes = {
            "c": (b, k, config.content_tokens, config.width),
            "d": (b, k, config.dynamic_tokens, config.width),
            "c_bg": (b, config.background_tokens, config.width),
            "slot_status": (b, k),
            "slot_uid": (b, k),
            "p_exist": (b, k),
            "last_seen": (b, k),
            "t": (b,),
            "geometry_mode": (b,),
            "frame_id": (b,),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            dtype = (
                torch.float64
                if name in ("last_seen", "t")
                else (
                    torch.int64
                    if name in ("slot_status", "slot_uid", "geometry_mode", "frame_id")
                    else torch.float32
                )
            )
            if (
                not isinstance(value, Tensor)
                or value.shape != shape
                or value.dtype != dtype
                or value.device != self.c.device
            ):
                raise ValueError(f"{name} has invalid shape, dtype or device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if ((self.slot_status < 0) | (self.slot_status > 2)).any() or (
            self.slot_uid < 0
        ).any():
            raise ValueError("invalid slot lifecycle metadata")
        if ((self.p_exist < 0) | (self.p_exist > 1)).any() or (
            self.geometry_mode != 0
        ).any():
            raise ValueError("invalid confidence or unsupported geometry mode")
        if (self.last_seen > self.t[:, None]).any():
            raise ValueError("last_seen cannot be in the future")
