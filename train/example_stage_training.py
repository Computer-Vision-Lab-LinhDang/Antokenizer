"""Example: Complete stage-to-stage training for MAVT.

This script demonstrates how to use the new dataloader system for
progressive training across stages 1, 2, and 3.

Usage:
    python -m train.example_stage_training --data-root /path/to/data

Structure:
    /path/to/data/
        images/          # For Stage 1+2+3
        videos/          # For Stage 2+3
        3d_objects/      # For Stage 3
"""
import argparse
import logging

from mavt.config import MAVTConfig
from mavt.tokenizer import MAVTokenizer
from train.curriculum import STAGE1, STAGE2, STAGE3
from train.stage_trainer import StageTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MAVT Stage-to-Stage Training")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing images/, videos/, 3d_objects/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=0,
        help="Stage to start from (0=Stage1, 1=Stage2, 2=Stage3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("MAVT Stage-to-Stage Training")
    logger.info("="*70)
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Start stage: {args.start_stage + 1}")
    logger.info(f"Device: {args.device}")
    logger.info("="*70)

    # ── Initialize model ──
    logger.info("\nInitializing MAVTokenizer...")
    model_config = MAVTConfig(
        patch_size=16,
        temporal_patch_size=2,
        latent_dim=1152,
        # Add other config as needed
    )
    model = MAVTokenizer(model_config)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable")

    # ── Setup data paths ──
    from pathlib import Path
    data_root = Path(args.data_root)

    # Check symlinks and subdirectories for data
    def find_data_path(root, name, subdirs=None):
        """Find data path, checking direct path and subdirectories."""
        direct = root / name
        if direct.exists() or direct.is_symlink():
            return str(direct)
        if subdirs:
            for sub in subdirs:
                p = root / name / sub
                if p.exists():
                    return str(root / name)
        return None

    data_roots = {
        "image": find_data_path(data_root, "images"),
        "video": find_data_path(data_root, "videos", ["webvid", "panda"]),
        "3d": find_data_path(data_root, "3d_objects"),
    }

    logger.info("\nData paths:")
    for modality, path in data_roots.items():
        status = "✓" if path else "✗"
        logger.info(f"  {status} {modality}: {path or 'NOT FOUND'}")

    # ── Create trainer ──
    logger.info("\nCreating trainer...")
    trainer = StageTrainer(
        model=model,
        data_roots=data_roots,
        output_dir=args.output_dir,
        log_every=100,
        val_every=5000,
        save_every=10000,
    )

    # ── Run training ──
    logger.info("\nStarting training across all stages...")

    stages = [STAGE1, STAGE2, STAGE3]

    try:
        trainer.train_all_stages(
            stages=stages,
            start_stage=args.start_stage,
        )
        logger.info("\n" + "="*70)
        logger.info("Training complete!")
        logger.info("="*70)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(
            Path(args.output_dir) / "interrupted.pt",
            stage_num=-1,
            step=-1
        )

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
