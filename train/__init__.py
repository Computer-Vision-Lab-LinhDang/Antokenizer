"""Training utilities and loop helpers."""

from .loop import Trainer
from .data import create_dataloader
from .ema import EMA
from .degradation import (
    DegradationConfig,
    DegradationPipeline,
    VideoTemporalDegradation,
)
from .dataloader import (
    DataConfig,
    ImageDataset,
    PairedImageDataset,
    AnnotatedDataset,
    VideoDataset,
    create_diffusion_dataloader,
    collate_diffusion_batch,
    get_dataset_stats,
    PairedTransform,
    VideoTransform,
)
from .mavt_task import MAVTTask
from .optim import OptimConfig, build_optimizer, build_scheduler
from .packing import PackedSequence, compute_token_count, compute_video_token_count, greedy_pack, build_block_attn_mask
from .navit_dataset import NaViTCollator, MultiResImageDataset, VideoClipDataset, SyntheticDataset, build_image_dataloader, build_mixed_dataloader
from .curriculum import StageConfig, STAGE1, STAGE2, STAGE3, STAGES, apply_stage

__all__ = [
    # Training loop
    "Trainer",
    "create_dataloader",
    "EMA",
    # Degradation
    "DegradationConfig",
    "DegradationPipeline",
    "VideoTemporalDegradation",
    # DataLoader
    "DataConfig",
    "ImageDataset",
    "PairedImageDataset",
    "AnnotatedDataset",
    "VideoDataset",
    "create_diffusion_dataloader",
    "collate_diffusion_batch",
    "get_dataset_stats",
    "PairedTransform",
    "VideoTransform",
    # MAVT task
    "MAVTTask",
    # Optimizer utilities
    "OptimConfig",
    "build_optimizer",
    "build_scheduler",
    # NaViT packing
    "PackedSequence",
    "compute_token_count",
    "compute_video_token_count",
    "greedy_pack",
    "build_block_attn_mask",
    # NaViT datasets
    "NaViTCollator",
    "MultiResImageDataset",
    "VideoClipDataset",
    "SyntheticDataset",
    "build_image_dataloader",
    "build_mixed_dataloader",
    # Curriculum
    "StageConfig",
    "STAGE1",
    "STAGE2",
    "STAGE3",
    "STAGES",
    "apply_stage",
]
