"""Training curriculum: Stage 1 → Stage 2 → Stage 3.

Stage 1 — Image Foundation (200k steps)
  - Image only, multi-resolution [64→512], NaViT packing
  - SigLIP2 fully frozen
  - STF: DWT spatial only

Stage 2 — Video Dynamics (200k steps)
  - Image + Video mixed, task ratios Ir=22.2% Vu=11.1% Vr=66.6%
  - SigLIP2 last 4 layers unfrozen
  - STF: DWT + STFT temporal

Stage 3 — 3D Geometry (50k steps)
  - Image + Video + 3D mixed
  - SigLIP2 fully unfrozen
  - STF: DWT + STFT temporal + STFT depth
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    name: str
    n_steps: int

    # Sequence packing
    max_seq_len: int = 4096

    # Modalities
    modalities: list[str] = field(default_factory=lambda: ["image"])
    image_resolutions: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    video_resolutions: list[int] = field(default_factory=list)
    triplane_sizes: list[int] = field(default_factory=list)

    # Task ratios (must sum to 1.0)
    task_ratios: dict[str, float] = field(default_factory=lambda: {"image_recon": 1.0})

    # SigLIP2 freezing: 0=all frozen, -1=all unfrozen, N=unfreeze last N transformer layers
    freeze_siglip2: bool = True
    siglip2_unfreeze_last_n: int = 0

    # STF activation
    stf_temporal: bool = False
    stf_depth: bool = False

    # Optimiser
    lr: float = 1e-4
    warmup_steps: int = 2000
    kl_weight: float = 1e-4
    grad_clip: float = 1.0


# ─── Stage definitions ────────────────────────────────────────────────────────

STAGE1 = StageConfig(
    name="stage1_image",
    n_steps=200_000,
    max_seq_len=4096,
    modalities=["image"],
    image_resolutions=[64, 128, 256, 512],
    video_resolutions=[],
    task_ratios={"image_recon": 1.0},
    freeze_siglip2=True,
    siglip2_unfreeze_last_n=0,
    stf_temporal=False,
    stf_depth=False,
    lr=1e-4,
    warmup_steps=2000,
)

STAGE2 = StageConfig(
    name="stage2_video",
    n_steps=200_000,
    max_seq_len=4096,
    modalities=["image", "video"],
    image_resolutions=[64, 128, 256, 512, 1024],
    video_resolutions=[64, 128, 256, 512],
    task_ratios={
        "image_recon":      0.222,
        "video_understand": 0.111,
        "video_recon":      0.667,
    },
    freeze_siglip2=False,
    siglip2_unfreeze_last_n=4,
    stf_temporal=True,
    stf_depth=False,
    lr=5e-5,
    warmup_steps=1000,
)

STAGE3 = StageConfig(
    name="stage3_3d",
    n_steps=50_000,
    max_seq_len=4096,
    modalities=["image", "video", "3d"],
    image_resolutions=[64, 128, 256, 512, 1024, 2048],
    video_resolutions=[64, 128, 256, 512, 1024],
    triplane_sizes=[32, 64],
    task_ratios={
        "image_recon":      0.222,
        "video_understand": 0.111,
        "video_recon":      0.444,
        "3d_understand":    0.111,
        "3d_recon":         0.111,
    },
    freeze_siglip2=False,
    siglip2_unfreeze_last_n=-1,
    stf_temporal=True,
    stf_depth=True,
    lr=2e-5,
    warmup_steps=500,
)

STAGES = [STAGE1, STAGE2, STAGE3]


# ─── Stage application ────────────────────────────────────────────────────────

def apply_stage(model, stage: StageConfig) -> None:
    """Configure the model (freeze/unfreeze SigLIP2) for this training stage.

    Args:
        model:  A MAVTTask or MAVTokenizer instance.
        stage:  The StageConfig to apply.
    """
    tok = getattr(model, "tokenizer", model)
    siglip = tok.patchify.siglip2

    # Always start fully frozen, then selectively unfreeze.
    siglip.requires_grad_(False)

    if stage.siglip2_unfreeze_last_n == -1:
        siglip.requires_grad_(True)
        logger.info("[%s] SigLIP2 fully unfrozen.", stage.name)
    elif stage.siglip2_unfreeze_last_n > 0:
        # For Conv2d backbone: just unfreeze patch_proj.
        # For full SigLIP2 ViT: unfreeze last N transformer blocks (extend here).
        if hasattr(siglip, "patch_proj"):
            siglip.patch_proj.requires_grad_(True)
        logger.info("[%s] SigLIP2 patch_proj unfrozen (%d layers requested).",
                    stage.name, stage.siglip2_unfreeze_last_n)
    else:
        logger.info("[%s] SigLIP2 fully frozen.", stage.name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[%s] Trainable params after freeze/unfreeze: %.1f M",
                stage.name, n_trainable / 1e6)


__all__ = ["StageConfig", "STAGE1", "STAGE2", "STAGE3", "STAGES", "apply_stage"]
