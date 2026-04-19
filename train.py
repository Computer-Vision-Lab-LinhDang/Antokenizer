#!/usr/bin/env python3
"""MAVT training entry point using Lightning CLI.

Usage examples:

  # Stage 1: image-only with synthetic data (smoke test)
  python train.py fit \
    --config configs/train/stage1_image.yaml \
    --model.training_stage 1 \
    --trainer.max_steps 500

  # Stage 1: real data, single GPU
  python train.py fit --config configs/train/stage1_image.yaml \
    --data.image_root /path/to/open-images

  # Stage 1: DDP on 8 GPUs
  python train.py fit --config configs/train/stage1_image.yaml \
    --trainer.devices 8 --trainer.strategy ddp \
    --data.image_root /path/to/open-images

  # Stage 2: resume from stage 1 checkpoint
  python train.py fit --config configs/train/stage2_video.yaml \
    --ckpt_path checkpoints/stage1/mavt-stage1-best.ckpt

  # WandB logging
  python train.py fit --config configs/train/stage1_image.yaml \
    --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project mavt \
    --trainer.logger.init_args.name stage1-image
"""

from lightning.pytorch.cli import LightningCLI

from mavt.training.lightning_module import MAVTLightningModule
from mavt.data.datamodule import MAVTDataModule


def main():
    cli = LightningCLI(
        model_class=MAVTLightningModule,
        datamodule_class=MAVTDataModule,
        save_config_callback=None,   # avoid overwrite on multi-run
    )


if __name__ == '__main__':
    main()
