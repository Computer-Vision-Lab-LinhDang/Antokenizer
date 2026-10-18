"""Versioned architecture for unified functional state (legacy v1 is unchanged)."""
from dataclasses import dataclass, asdict
from ..config import ObjectStateConfig


@dataclass(frozen=True)
class V5Config:
    slots: int = 16
    content_tokens: int = 8
    dynamic_tokens: int = 8
    width: int = 128
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
    camera_views: int = 6
    depth_hypotheses: int = 4
    atlas_extent: float = 4.0
    ownership_temperature: float = 1.0
    schema_version: int = 5

    def __post_init__(self):
        self.encoder_config()
        if self.width % 2 or self.schema_version != 5:
            raise ValueError('v5 requires even latent width and schema version 5')
        if self.camera_views < 1 or self.depth_hypotheses < 1 or self.atlas_extent <= 0 or self.ownership_temperature <= 0:
            raise ValueError('invalid camera, depth, atlas or ownership configuration')

    def encoder_config(self):
        values = asdict(self)
        fields = ObjectStateConfig.__dataclass_fields__
        values = {key: value for key, value in values.items() if key in fields and key != 'schema_version'}
        return ObjectStateConfig(**values)

    def to_dict(self):
        return asdict(self)
