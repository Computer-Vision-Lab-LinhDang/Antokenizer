"""Complete C/D + bounded-KV restart state, separate from the decoder code.

State tensors are the entire sequence-dependent payload. Model configuration
and model weights are external immutable inputs identified by SHA-256. No
deserialization method synthesizes a missing cache or accepts a decoder code
as an exact restart checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from typing import ClassVar

import torch
from torch import Tensor

from .config import HybridConfig, HybridV3Config, is_v3


_STATE_FIELDS = (
    "c", "d", "k", "v", "last_time", "step_count", "write_index",
    "steps_filled", "slot_times", "slot_step_ids", "token_valid",
)
_HEADER_FIELDS = ("schema_version", "config_sha256", "weights_sha256")


def _tensor(name: str, value: Tensor, shape: tuple[int, ...], dtype: torch.dtype,
            device: torch.device | None = None) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name}: expected dense strided storage")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {value.dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name}: expected device {device}, got {value.device}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name}: expected finite values")


def _code_tensors(c: Tensor, d: Tensor, last_time: Tensor, config: HybridConfig) -> int:
    if not isinstance(config, HybridConfig):
        raise TypeError("config: expected HybridConfig")
    if not isinstance(c, Tensor) or c.ndim != 4 or c.shape[0] <= 0:
        raise ValueError("c: expected nonempty [B, stages, tokens, dim]")
    batch = c.shape[0]
    shape = (batch, config.stage_count, config.state_tokens, config.state_dim)
    _tensor("c", c, shape, torch.float32)
    _tensor("d", d, shape, torch.float32, c.device)
    _tensor("last_time", last_time, (batch,), torch.float64, c.device)
    return batch


def _logical_nbytes(values) -> int:
    return sum(value.numel() * value.element_size() for value in values)


def _storage_nbytes(values) -> int:
    storages = {}
    for value in values:
        storage = value.untyped_storage()
        storages[(value.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def _hash_bytes(value: str, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name}: expected SHA-256 as 64 hexadecimal characters")
    try:
        result = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name}: expected SHA-256 hexadecimal characters") from error
    if len(result) != 32:
        raise ValueError(f"{name}: expected 32 digest bytes")
    return result


def weights_fingerprint(state_dict: Mapping[str, Tensor]) -> str:
    """Hash sorted tensor names, dtypes, shapes and exact bytes, including BF16.

    Pass the complete model ``state_dict()`` so parameters and static buffers
    both participate. This never caches the digest: callers must recompute it
    after modifying weights. Non-tensor extra-state objects are rejected.
    """
    if not isinstance(state_dict, Mapping) or not all(isinstance(name, str) for name in state_dict):
        raise TypeError("state_dict: expected a mapping with string keys")
    digest = hashlib.sha256()
    digest.update(b"mavt-hybrid-weights-v2\0")
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, Tensor) or value.layout != torch.strided:
            raise TypeError(f"state_dict[{name!r}]: expected a dense tensor")
        metadata = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":"),
        ).encode("utf-8")
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CDCode:
    """C/D-only readout code; this object cannot restart the temporal encoder."""

    c: Tensor
    d: Tensor
    last_time: Tensor
    config: HybridConfig = field(kw_only=True)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _code_tensors(self.c, self.d, self.last_time, self.config)

    @property
    def total_nbytes(self) -> int:
        return _logical_nbytes((self.c, self.d, self.last_time))

    @property
    def retained_storage_nbytes(self) -> int:
        return _storage_nbytes((self.c, self.d, self.last_time))


@dataclass(frozen=True, slots=True)
class HybridCDKVState:
    """Version 2 restart payload with independently clocked batch streams.

    C/D are FP32. K/V are BF16 and fully allocated even for an empty ring.
    Accepted step IDs start at zero. Empty slots have ID -1, timestamp zero,
    false validity and zero K/V. Invalid tokens in occupied missing slots also
    have zero K/V. ``write_index`` points to the next slot to overwrite.

    Byte properties count logical tensor payload or directly retained storage;
    neither includes allocator overhead, weights, serialized-container headers,
    or memory reachable through a training autograd graph. ``snapshot`` and
    ``to_payload`` detach and compact every tensor. ``clone(detach=False)``
    preserves autograd, including paths through cached BF16 tensors.
    """

    c: Tensor
    d: Tensor
    k: Tensor
    v: Tensor
    last_time: Tensor
    step_count: Tensor
    write_index: Tensor
    steps_filled: Tensor
    slot_times: Tensor
    slot_step_ids: Tensor
    token_valid: Tensor
    config: HybridConfig = field(kw_only=True)

    SCHEMA_VERSION: ClassVar[int] = 2
    SCHEMA: ClassVar[str] = "mavt-hybrid-cd2-kv-v2"
    HEADER_NBYTES: ClassVar[int] = 68
    STATE_FIELDS: ClassVar[tuple[str, ...]] = _STATE_FIELDS

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def initial(cls, config: HybridConfig, *, batch_size: int,
                device: torch.device | str, last_time: float | Tensor) -> HybridCDKVState:
        """Allocate a canonical empty ring at an explicit initial stream time."""
        if not isinstance(config, HybridConfig):
            raise TypeError("config: expected HybridConfig")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size: expected a positive integer")
        target = torch.device(device)
        if isinstance(last_time, Tensor):
            _tensor("last_time", last_time, (batch_size,), torch.float64)
            times = last_time.to(device=target).clone(memory_format=torch.contiguous_format)
        elif type(last_time) in (float, int):
            times = torch.full((batch_size,), float(last_time), dtype=torch.float64, device=target)
        else:
            raise TypeError("last_time: expected seconds or a float64 [B] tensor")
        s, n, d = config.stage_count, config.state_tokens, config.state_dim
        w, p, h, hd = config.cache_steps, config.summary_tokens, config.kv_heads, config.head_dim
        cd_shape = (batch_size, s, n, d)
        kv_shape = (batch_size, s, w, p, h, hd)
        return cls(
            c=torch.zeros(cd_shape, dtype=torch.float32, device=target),
            d=torch.zeros(cd_shape, dtype=torch.float32, device=target),
            k=torch.zeros(kv_shape, dtype=torch.bfloat16, device=target),
            v=torch.zeros(kv_shape, dtype=torch.bfloat16, device=target),
            last_time=times,
            step_count=torch.zeros(batch_size, dtype=torch.int64, device=target),
            write_index=torch.zeros(batch_size, dtype=torch.int64, device=target),
            steps_filled=torch.zeros(batch_size, dtype=torch.int64, device=target),
            slot_times=torch.zeros((batch_size, w), dtype=torch.float64, device=target),
            slot_step_ids=torch.full((batch_size, w), -1, dtype=torch.int64, device=target),
            token_valid=torch.zeros((batch_size, s, w, p), dtype=torch.bool, device=target),
            config=config,
        )

    def validate(self) -> None:
        batch = _code_tensors(self.c, self.d, self.last_time, self.config)
        cfg, device = self.config, self.c.device
        s, w, p = cfg.stage_count, cfg.cache_steps, cfg.summary_tokens
        kv_shape = (batch, s, w, p, cfg.kv_heads, cfg.head_dim)
        _tensor("k", self.k, kv_shape, torch.bfloat16, device)
        _tensor("v", self.v, kv_shape, torch.bfloat16, device)
        for name in ("step_count", "write_index", "steps_filled"):
            _tensor(name, getattr(self, name), (batch,), torch.int64, device)
        _tensor("slot_times", self.slot_times, (batch, w), torch.float64, device)
        _tensor("slot_step_ids", self.slot_step_ids, (batch, w), torch.int64, device)
        _tensor("token_valid", self.token_valid, (batch, s, w, p), torch.bool, device)
        if bool((self.step_count < 0).any()):
            raise ValueError("step_count: expected a nonnegative accepted-step count")
        if not torch.equal(self.write_index, self.step_count.remainder(w)):
            raise ValueError("write_index: inconsistent with step_count")
        if not torch.equal(self.steps_filled, self.step_count.clamp_max(w)):
            raise ValueError("steps_filled: inconsistent with step_count")

        slots = torch.arange(w, dtype=torch.int64, device=device).unsqueeze(0)
        latest = slots + torch.div(self.step_count[:, None] - 1 - slots, w, rounding_mode="floor") * w
        expected_ids = torch.where(latest >= 0, latest, torch.full_like(latest, -1))
        if not torch.equal(self.slot_step_ids, expected_ids):
            raise ValueError("slot_step_ids: inconsistent ring position, count or eviction")
        empty = self.slot_step_ids < 0
        if bool((self.slot_times[empty] != 0).any()):
            raise ValueError("slot_times: empty slots must have canonical zero timestamps")
        if bool((self.token_valid & empty[:, None, :, None]).any()):
            raise ValueError("token_valid: empty slots cannot contain valid tokens")
        invalid = ~self.token_valid[..., None, None].expand_as(self.k)
        if bool((self.k[invalid] != 0).any()) or bool((self.v[invalid] != 0).any()):
            raise ValueError("k/v: invalid or empty tokens must have zero cache values")

        order = self.slot_step_ids.argsort(dim=-1)
        ordered_ids = self.slot_step_ids.gather(1, order)
        ordered_times = self.slot_times.gather(1, order)
        consecutive = (ordered_ids[:, :-1] >= 0) & (ordered_ids[:, 1:] >= 0)
        if bool(((ordered_times[:, 1:] < ordered_times[:, :-1]) & consecutive).any()):
            raise ValueError("slot_times: accepted-step timestamps must be nondecreasing")
        occupied = self.step_count > 0
        newest_slot = (self.step_count - 1).remainder(w).unsqueeze(1)
        newest_time = self.slot_times.gather(1, newest_slot).squeeze(1)
        if bool((newest_time[occupied] != self.last_time[occupied]).any()):
            raise ValueError("last_time: must equal the newest accepted slot timestamp")

    @property
    def cd_nbytes(self) -> int:
        return _logical_nbytes((self.c, self.d))

    @property
    def kv_nbytes(self) -> int:
        return _logical_nbytes((self.k, self.v))

    @property
    def metadata_nbytes(self) -> int:
        return _logical_nbytes(getattr(self, name) for name in type(self).STATE_FIELDS[4:])

    @property
    def total_nbytes(self) -> int:
        return self.cd_nbytes + self.kv_nbytes + self.metadata_nbytes

    @property
    def payload_nbytes(self) -> int:
        return self.total_nbytes + self.HEADER_NBYTES

    @property
    def retained_storage_nbytes(self) -> int:
        return _storage_nbytes(getattr(self, name) for name in type(self).STATE_FIELDS)

    def code(self) -> CDCode:
        """Expose the exact C/D tensors as a readout-only view, without a cache."""
        self.validate()
        return CDCode(self.c, self.d, self.last_time, config=self.config)

    def clone(self, *, detach: bool = False) -> HybridCDKVState:
        """Copy every tensor into compact storage; preserve gradients by default."""
        self.validate()
        values = {}
        for name in type(self).STATE_FIELDS:
            value = getattr(self, name)
            values[name] = (value.detach() if detach else value).clone(memory_format=torch.contiguous_format)
        return type(self)(**values, config=self.config)

    def detach(self) -> HybridCDKVState:
        """Truncate all C/D and KV gradient paths without changing their values."""
        self.validate()
        return type(self)(**{name: getattr(self, name).detach() for name in type(self).STATE_FIELDS}, config=self.config)

    def snapshot(self) -> HybridCDKVState:
        return self.clone(detach=True)

    def to_payload(self, *, weights_hash: str) -> dict[str, Tensor]:
        """Detached, compact tensor-only payload suitable for weights-only I/O.

        Identity headers are canonical CPU tensors. State tensors retain their
        current device; callers may use ``torch.load(..., map_location=...)``
        to move a serialized payload before resuming.
        """
        digest = _hash_bytes(weights_hash, "weights_hash")
        snapshot = self.snapshot()
        payload = {name: getattr(snapshot, name) for name in type(self).STATE_FIELDS}
        payload.update(
            schema_version=torch.tensor(self.SCHEMA_VERSION, dtype=torch.int32),
            config_sha256=torch.tensor(list(_hash_bytes(self.config.fingerprint(), "config_hash")), dtype=torch.uint8),
            weights_sha256=torch.tensor(list(digest), dtype=torch.uint8),
        )
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Tensor], *, config: HybridConfig,
                     weights_hash: str) -> HybridCDKVState:
        """Restore only the declared schema with the exact supplied identities."""
        expected = set(cls.STATE_FIELDS) | set(_HEADER_FIELDS)
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError(f"payload: expected exactly these keys: {sorted(expected)}")
        if not isinstance(config, HybridConfig):
            raise TypeError("config: expected HybridConfig")
        expected_weights = _hash_bytes(weights_hash, "weights_hash")
        _tensor("schema_version", payload["schema_version"], (), torch.int32)
        if payload["schema_version"].item() != cls.SCHEMA_VERSION:
            raise ValueError(f"schema_version: expected {cls.SCHEMA_VERSION}")
        for name, expected_digest in (
            ("config_sha256", _hash_bytes(config.fingerprint(), "config_hash")),
            ("weights_sha256", expected_weights),
        ):
            _tensor(name, payload[name], (32,), torch.uint8)
            if bytes(payload[name].detach().cpu().tolist()) != expected_digest:
                raise ValueError(f"{name}: mismatch with the requested model")
        state = cls(**{name: payload[name] for name in cls.STATE_FIELDS}, config=config)
        return state.snapshot()


def _appearance_shapes(config: HybridConfig, batch: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    if not is_v3(config):
        raise TypeError("config: expected HybridV3Config")
    hp, wp = config.innovation_grid if config.innovation else (0, 0)
    return {
        "appearance": ((batch, config.appearance_tokens, config.state_dim), torch.float32),
        "appearance_valid": ((batch,), torch.bool),
        "prev_tokens": ((batch, config.reference_views, hp, wp, config.spatial_width), torch.float32),
        "prev_seen": ((batch, config.reference_views, hp, wp), torch.bool),
    }


@dataclass(frozen=True, slots=True)
class CDACode(CDCode):
    """C/D readout code plus the write-once appearance bank."""

    appearance: Tensor

    def validate(self) -> None:
        CDCode.validate(self)
        shape, dtype = _appearance_shapes(self.config, self.c.shape[0])["appearance"]
        _tensor("appearance", self.appearance, shape, dtype, self.c.device)

    @property
    def total_nbytes(self) -> int:
        return _logical_nbytes((self.c, self.d, self.last_time, self.appearance))

    @property
    def retained_storage_nbytes(self) -> int:
        return _storage_nbytes((self.c, self.d, self.last_time, self.appearance))


@dataclass(frozen=True, slots=True)
class HybridCDAKVState(HybridCDKVState):
    """Version 3 restart payload: v2 state plus appearance bank and previous tokens."""

    appearance: Tensor
    appearance_valid: Tensor
    prev_tokens: Tensor
    prev_seen: Tensor

    SCHEMA_VERSION: ClassVar[int] = 3
    SCHEMA: ClassVar[str] = "mavt-hybrid-cd2-kv-v3"
    STATE_FIELDS: ClassVar[tuple[str, ...]] = _STATE_FIELDS + (
        "appearance", "appearance_valid", "prev_tokens", "prev_seen",
    )

    @classmethod
    def initial(cls, config: HybridConfig, *, batch_size: int,
                device: torch.device | str, last_time: float | Tensor) -> HybridCDAKVState:
        base = HybridCDKVState.initial(config, batch_size=batch_size, device=device, last_time=last_time)
        extra = {
            name: torch.zeros(shape, dtype=dtype, device=base.c.device)
            for name, (shape, dtype) in _appearance_shapes(config, batch_size).items()
        }
        return cls(**{name: getattr(base, name) for name in _STATE_FIELDS}, **extra, config=config)

    def validate(self) -> None:
        HybridCDKVState.validate(self)
        for name, (shape, dtype) in _appearance_shapes(self.config, self.c.shape[0]).items():
            _tensor(name, getattr(self, name), shape, dtype, self.c.device)

    def code(self) -> CDACode:
        self.validate()
        return CDACode(self.c, self.d, self.last_time, self.appearance, config=self.config)
