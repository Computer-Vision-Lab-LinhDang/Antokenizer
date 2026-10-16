"""Explicit C/D memory and observation-only interfaces for recurrent experiments.

The state payload contains every persistent, sequence-dependent value admitted
by this interface. It deliberately has no convolution cache, feature history,
or oracle labels. A model using this interface must separately demonstrate that
serialization is sufficient to resume its computation; no cell exists here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import ClassVar, Literal

import torch
from torch import Tensor


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name}: expected a dense strided tensor")


def _float_tensor(name: str, value: Tensor) -> None:
    _tensor(name, value)
    if value.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{name}: expected float16, bfloat16, float32, or float64")


def _shape(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {tuple(value.shape)}")


def _same_device(device: torch.device, **values: Tensor) -> None:
    for name, value in values.items():
        _tensor(name, value)
        if value.device != device:
            raise ValueError(f"{name}: expected device {device}, got {value.device}")


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name}: expected finite values")


def _timestamps(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    _tensor(name, value)
    _shape(name, value, expected)
    if value.dtype != torch.float64:
        raise ValueError(f"{name}: expected float64 seconds")
    _finite(name, value)


def _calibration(
    intrinsics: Tensor | None,
    cam_to_world: Tensor | None,
    *,
    device: torch.device,
    views: int,
    frames: int | None = None,
) -> None:
    """Check explicit pinhole calibration, without constructing missing geometry."""
    if intrinsics is None and cam_to_world is None:
        return
    if intrinsics is None or cam_to_world is None:
        raise ValueError("intrinsics and cam_to_world: provide both or neither")
    _float_tensor("intrinsics", intrinsics)
    _float_tensor("cam_to_world", cam_to_world)
    for name, value in (("intrinsics", intrinsics), ("cam_to_world", cam_to_world)):
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name}: calibration requires float32 or float64")
    _same_device(device, intrinsics=intrinsics, cam_to_world=cam_to_world)
    _shape("intrinsics", intrinsics, (views, 3, 3))
    pose_shape = (views, 4, 4) if frames is None else (views, frames, 4, 4)
    _shape("cam_to_world", cam_to_world, pose_shape)
    _finite("intrinsics", intrinsics)
    _finite("cam_to_world", cam_to_world)

    # Calibration precision is independent of image/state precision. Strict
    # rigid-pose validation is not reliable after half-precision rounding.
    k = intrinsics.double()
    if not bool((torch.linalg.det(k).abs() > 1e-12).all()):
        raise ValueError("intrinsics: expected nonsingular calibration")
    if not bool(((k[:, 0, 0] > 0) & (k[:, 1, 1] > 0)).all()):
        raise ValueError("intrinsics: expected positive focal lengths")
    if not torch.allclose(k[:, 2], k.new_tensor([0, 0, 1]).expand(views, 3), atol=1e-5, rtol=0):
        raise ValueError("intrinsics: expected pinhole homogeneous row [0, 0, 1]")
    poses = cam_to_world.double().reshape(-1, 4, 4)
    last_row = poses.new_tensor([0, 0, 0, 1]).expand(poses.shape[0], 4)
    if not torch.allclose(poses[:, 3], last_row, atol=1e-5, rtol=0):
        raise ValueError("cam_to_world: expected homogeneous row [0, 0, 0, 1]")
    rotation = poses[:, :3, :3]
    identity = torch.eye(3, device=device, dtype=torch.float64).expand_as(rotation)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4, rtol=1e-4):
        raise ValueError("cam_to_world: rotation must be orthonormal")
    if not torch.allclose(torch.linalg.det(rotation), poses.new_ones(poses.shape[0]), atol=1e-4, rtol=1e-4):
        raise ValueError("cam_to_world: rotation must have determinant +1")


@dataclass(frozen=True, slots=True)
class CDStateSpec:
    """Fixed state budget, independent of how many views are currently observed.

    Token counts are explicit; ``reference_views`` never silently multiplies
    them. The default has 4096 floating-point values per sample. Physical input
    steps contain at most ``frames_per_step`` real frames in each view.
    """

    c_tokens: int = 64
    d_tokens: int = 64
    token_dim: int = 32
    frames_per_step: int = 2
    reference_views: int = 1

    def __post_init__(self) -> None:
        for name in ("c_tokens", "d_tokens", "token_dim", "frames_per_step", "reference_views"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}: expected a positive integer")

    @property
    def num_state_values(self) -> int:
        """Number of C and D scalar values per sample, excluding metadata."""
        return (self.c_tokens + self.d_tokens) * self.token_dim


@dataclass(frozen=True, slots=True)
class CDState:
    """Batched recurrent memory with the same C/D tensors exposed to readout.

    ``last_time`` is the only dynamic metadata: float64 physical seconds, one
    value per stream. Tensor contents remain mutable in PyTorch despite the
    frozen container, so methods revalidate before exporting or copying.

    ``state_nbytes``, ``metadata_nbytes`` and ``total_nbytes`` count the logical
    tensor payload across the batch. ``retained_storage_nbytes`` additionally
    exposes unused backing storage retained through views, counting shared
    storage only once. Neither includes allocator overhead, autograd graphs,
    or pickle headers. ``snapshot`` and ``to_payload`` produce detached compact
    copies, so exported state does not retain oversized tensor backing stores.
    """

    c: Tensor
    d: Tensor
    last_time: Tensor
    spec: CDStateSpec = field(default_factory=CDStateSpec, kw_only=True)

    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.spec, CDStateSpec):
            raise TypeError("spec: expected CDStateSpec")
        _float_tensor("c", self.c)
        _float_tensor("d", self.d)
        _same_device(self.c.device, d=self.d, last_time=self.last_time)
        if self.c.ndim != 3 or self.c.shape[0] <= 0:
            raise ValueError("c: expected nonempty [B, C_tokens, token_dim]")
        batch = self.c.shape[0]
        _shape("c", self.c, (batch, self.spec.c_tokens, self.spec.token_dim))
        _shape("d", self.d, (batch, self.spec.d_tokens, self.spec.token_dim))
        if self.d.dtype != self.c.dtype:
            raise ValueError("d: must share c dtype")
        _timestamps("last_time", self.last_time, (batch,))
        _finite("c", self.c)
        _finite("d", self.d)

    @property
    def num_state_values(self) -> int:
        """Total number of C/D scalar values, including the batch dimension."""
        return self.c.numel() + self.d.numel()

    @property
    def state_nbytes(self) -> int:
        return self.c.numel() * self.c.element_size() + self.d.numel() * self.d.element_size()

    @property
    def metadata_nbytes(self) -> int:
        return self.last_time.numel() * self.last_time.element_size()

    @property
    def total_nbytes(self) -> int:
        return self.state_nbytes + self.metadata_nbytes

    @property
    def retained_storage_nbytes(self) -> int:
        """Actual directly retained tensor storage, including unused view tails.

        Deduplicate underlying allocations across C, D and timestamp tensors.
        This is distinct from the logical payload budget and excludes memory
        reachable only through a training autograd graph or allocator caches.
        """
        storages = {}
        for value in (self.c, self.d, self.last_time):
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr())
            storages[key] = storage.nbytes()
        return sum(storages.values())

    def detach(self) -> CDState:
        """Truncate gradients without resetting values; tensor storage is shared."""
        self.validate()
        return CDState(self.c.detach(), self.d.detach(), self.last_time.detach(), spec=self.spec)

    def snapshot(self) -> CDState:
        """Capture an independent, detached copy of every persistent value."""
        self.validate()
        return CDState(
            self.c.detach().clone(memory_format=torch.contiguous_format),
            self.d.detach().clone(memory_format=torch.contiguous_format),
            self.last_time.detach().clone(memory_format=torch.contiguous_format),
            spec=self.spec,
        )

    def to_payload(self) -> dict:
        """Return only built-in values and tensors, safe for weights-only loading."""
        snapshot = self.snapshot()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "spec": asdict(self.spec),
            "c": snapshot.c,
            "d": snapshot.d,
            "last_time": snapshot.last_time,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> CDState:
        """Restore the exact declared schema; unknown memory is an error."""
        expected = {"schema_version", "spec", "c", "d", "last_time"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError(f"payload: expected exactly these keys: {sorted(expected)}")
        version = payload["schema_version"]
        if type(version) is not int or version != cls.SCHEMA_VERSION:
            raise ValueError(f"schema_version: expected {cls.SCHEMA_VERSION}")
        spec_payload = payload["spec"]
        spec_keys = {entry.name for entry in fields(CDStateSpec)}
        if not isinstance(spec_payload, dict) or set(spec_payload) != spec_keys:
            raise ValueError(f"spec: expected exactly these keys: {sorted(spec_keys)}")
        state = cls(
            payload["c"], payload["d"], payload["last_time"], spec=CDStateSpec(**spec_payload)
        )
        return state.snapshot()


@dataclass(frozen=True, slots=True)
class ObservationStep:
    """An unbatched, current-step observation with explicit pixel availability.

    Shapes: images [V,F,3,H,W], observed [V,F,H,W], frame_times [V,F].
    Optional calibration is intrinsics [V,3,3] in image pixels and camera-to-
    world poses [V,F,4,4], independently float32 or float64 even for half-
    precision images. Timestamps increase strictly within each view; views may
    have different timestamps. The step becomes available at ``end_time``.

    ``observed=False`` means missing data, not a naturally occluded object.
    Raw hidden values can be arbitrary, including NaN. Call ``masked_images``
    before Patchify and retain ``observed`` as a separate coverage signal.
    A single image uses F=1. A tubelet consumer is responsible for padding and
    marking padding internally, without introducing phantom physical times.
    """

    images: Tensor
    observed: Tensor
    frame_times: Tensor
    intrinsics: Tensor | None = None
    cam_to_world: Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _float_tensor("images", self.images)
        _same_device(self.images.device, observed=self.observed, frame_times=self.frame_times)
        if self.images.ndim != 5 or self.images.shape[2] != 3 or any(size <= 0 for size in self.images.shape):
            raise ValueError("images: expected nonempty [V,F,3,H,W]")
        views, frames, _, height, width = self.images.shape
        _shape("observed", self.observed, (views, frames, height, width))
        if self.observed.dtype != torch.bool:
            raise ValueError("observed: expected boolean availability mask")
        _timestamps("frame_times", self.frame_times, (views, frames))
        if frames > 1 and not bool((self.frame_times[:, 1:] > self.frame_times[:, :-1]).all()):
            raise ValueError("frame_times: must increase strictly within each view")
        available = self.observed.unsqueeze(2).expand_as(self.images)
        _finite("images at observed pixels", self.images[available])
        _calibration(
            self.intrinsics, self.cam_to_world,
            device=self.images.device, views=views, frames=frames,
        )

    def validate_against(self, spec: CDStateSpec) -> None:
        """Check the fixed view table and maximum real frames for this pilot."""
        self.validate()
        if not isinstance(spec, CDStateSpec):
            raise TypeError("spec: expected CDStateSpec")
        if self.images.shape[0] != spec.reference_views:
            raise ValueError("images: view count must equal fixed reference_views, even when missing")
        if self.images.shape[1] > spec.frames_per_step:
            raise ValueError("images: frame count exceeds frames_per_step")

    @property
    def end_time(self) -> Tensor:
        """The causal availability time, including scheduled missing frames."""
        self.validate()
        return self.frame_times.max()

    def masked_images(self) -> Tensor:
        """Return a zero-filled copy; multiplication is unsafe for hidden NaNs."""
        self.validate()
        return torch.where(self.observed.unsqueeze(2), self.images, torch.zeros_like(self.images))


@dataclass(frozen=True, slots=True)
class DecoderQuery:
    """Readout locations and declared geometry, without observations or features.

    Coordinates are [Q,3] with an explicit ``coordinate_space``:

    * ``image_pixel`` (default): (x,y,0) in pixels, x right and y down. Pixel
      centers are (column+0.5,row+0.5); continuous subpixel queries are allowed.
      Image extents are not available here, so no spatial extent is inferred.
    * ``world_xyz``: (x,y,z) in meters in the declared scene world frame, the
      same frame as any supplied camera poses.
    * ``triplane_uv``: (u,v,0), u and v in [-1,1]. View IDs 0,1,2 designate
      XY,XZ,YZ respectively, with u along the first named axis and v along the
      second. Camera calibration is not meaningful and must be omitted.

    Times [Q] are float64 seconds, view_ids [Q] are nonnegative int64 indices.
    The query does not perform a future state transition.

    If supplied, intrinsics [V,3,3] and cam_to_world [V,4,4] form an explicit
    view-indexed float32/float64 table valid at the requested times. Moving-
    camera queries with different poses need separate tables/calls; poses are
    never inferred from IDs or timestamps. Without calibration IDs have no
    implied geometry. This metadata is external per-query input; retaining it
    between steps would create additional persistent memory to declare/count.
    """

    coordinates: Tensor
    view_ids: Tensor
    times: Tensor
    intrinsics: Tensor | None = None
    cam_to_world: Tensor | None = None
    coordinate_space: Literal["image_pixel", "world_xyz", "triplane_uv"] = "image_pixel"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _float_tensor("coordinates", self.coordinates)
        _same_device(self.coordinates.device, view_ids=self.view_ids, times=self.times)
        if self.coordinates.ndim != 2 or self.coordinates.shape[0] <= 0 or self.coordinates.shape[1] != 3:
            raise ValueError("coordinates: expected nonempty [Q,3]")
        count = self.coordinates.shape[0]
        _shape("view_ids", self.view_ids, (count,))
        if self.view_ids.dtype != torch.int64:
            raise ValueError("view_ids: expected int64")
        if not bool((self.view_ids >= 0).all()):
            raise ValueError("view_ids: expected nonnegative indices")
        _timestamps("times", self.times, (count,))
        _finite("coordinates", self.coordinates)
        spaces = ("image_pixel", "world_xyz", "triplane_uv")
        if not isinstance(self.coordinate_space, str) or self.coordinate_space not in spaces:
            raise ValueError(f"coordinate_space: expected one of {spaces}")
        if self.coordinate_space in ("image_pixel", "triplane_uv"):
            if not bool((self.coordinates[:, 2] == 0).all()):
                raise ValueError("coordinates: z must be zero for a two-dimensional query space")
        if self.coordinate_space == "triplane_uv":
            if not bool((self.coordinates[:, :2].abs() <= 1).all()):
                raise ValueError("coordinates: triplane u and v must be in [-1,1]")
            if not bool((self.view_ids < 3).all()):
                raise ValueError("view_ids: triplane planes are 0=XY, 1=XZ, 2=YZ")
            if self.intrinsics is not None or self.cam_to_world is not None:
                raise ValueError("triplane_uv: camera calibration must be omitted")
            return
        if self.intrinsics is None and self.cam_to_world is None:
            return
        if self.intrinsics is None or self.cam_to_world is None:
            raise ValueError("intrinsics and cam_to_world: provide both or neither")
        _tensor("intrinsics", self.intrinsics)
        if self.intrinsics.ndim != 3 or self.intrinsics.shape[0] <= 0:
            raise ValueError("intrinsics: expected nonempty [V,3,3]")
        views = self.intrinsics.shape[0]
        _calibration(self.intrinsics, self.cam_to_world, device=self.coordinates.device, views=views)
        if not bool((self.view_ids < views).all()):
            raise ValueError("view_ids: index outside the supplied calibration table")
