"""Fixed, serializable architecture and lifecycle policy for object-state v1."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ObjectStateConfig:
    slots: int = 16
    content_tokens: int = 8
    dynamic_tokens: int = 8
    width: int = 128
    background_tokens: int = 16
    evidence_width: int = 384
    spatial_blocks: int = 8
    spatial_heads: int = 6
    window_size: int = 8
    relay_tokens: int = 8
    geometric_neighbors: int = 32
    resampler_heads: int = 4
    candidates: int = 128
    action_dim: int = 0
    write_rank: int = 2
    image_channels: int = 3
    patch_size: int = 16
    tubelet_size: int = 2
    mlp_ratio: int = 4
    samples_per_ray: int = 32
    query_chunk_size: int = 128
    decoder_checkpoint: bool = True
    birth_min_mass: float = 0.5
    birth_min_peak: float = 0.05
    observation_min_mass: float = 0.01
    geometry_bias_scale: float = 0.1
    schema_version: int = 1

    def __post_init__(self):
        for name in (
            "slots",
            "content_tokens",
            "dynamic_tokens",
            "width",
            "background_tokens",
            "evidence_width",
            "spatial_blocks",
            "spatial_heads",
            "window_size",
            "relay_tokens",
            "geometric_neighbors",
            "resampler_heads",
            "candidates",
            "write_rank",
            "image_channels",
            "patch_size",
            "tubelet_size",
            "mlp_ratio",
            "samples_per_ray",
            "query_chunk_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.width % self.resampler_heads
            or self.evidence_width % self.spatial_heads
        ):
            raise ValueError("attention widths must divide their head counts")
        if type(self.action_dim) is not int or self.action_dim < 0:
            raise ValueError("action_dim must be nonnegative")
        if type(self.decoder_checkpoint) is not bool:
            raise ValueError("decoder_checkpoint must be boolean")
        for name in (
            "birth_min_mass",
            "birth_min_peak",
            "observation_min_mass",
            "geometry_bias_scale",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.birth_min_peak > 1 or self.schema_version != 1:
            raise ValueError("invalid birth probability or schema version")

    def to_dict(self):
        return asdict(self)
