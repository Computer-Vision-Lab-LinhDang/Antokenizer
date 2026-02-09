#!/usr/bin/env python3
"""Training and inference script for the D3PM diffusion generator.

This module provides:
1. Model creation utilities
2. Training pipeline with DataLoader
3. Inference utilities for restoration

Usage (command line):
    # Single folder (degradation applied on-the-fly)
    python -m atoken.scripts.train_diffusion train --data_dir /path/to/images

    # Paired folders (input/groundtruth)
    python -m atoken.scripts.train_diffusion train --data_dir /path/to/degraded --gt_dir /path/to/clean

    # Inference
    python -m atoken.scripts.train_diffusion infer --checkpoint model.pt --input img.jpg --output out.jpg

Usage (Python):
    from atoken.scripts.train_diffusion import create_model, train

    # Create model
    generator = create_model(d_model=768, detail_timesteps=50)

    # Train
    train(data_dir="/path/to/images", output_dir="./checkpoints")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torchvision import transforms
    from PIL import Image
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

# Import AToken components
from atoken.diffusion import DiffusionGenerator
from atoken.tasks import DiffusionGenerationTask
from atoken.train import (
    DegradationPipeline,
    DegradationConfig,
    Trainer,
    EMA,
    DataConfig,
    create_diffusion_dataloader,
)
from atoken.infer import DiffusionInference


# ============================================================================
# Model Creation
# ============================================================================

def create_model(
    # Model dimensions
    d_model: int = 512,
    num_heads: int = 8,
    # Encoder/Decoder
    encoder_depth: int = 8,
    decoder_depth: int = 6,
    # Semantic branch
    semantic_depth: int = 4,
    semantic_vocab_size: int = 4096,
    semantic_mask_ratio: float = 0.15,
    # Detail branch
    detail_depth: int = 6,
    detail_vocab_size: int = 8192,
    detail_timesteps: int = 50,
    detail_transition_type: str = "absorbing",
    detail_schedule_type: str = "cosine",
    # Input/Output
    in_channels: int = 3,
    patch_size: Tuple[int, int, int] = (1, 16, 16),
    # Other
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    use_ema_quantizer: bool = True,
) -> DiffusionGenerator:
    """Create a DiffusionGenerator model.

    Args:
        d_model: Model embedding dimension.
        num_heads: Number of attention heads.
        encoder_depth: Number of encoder transformer layers.
        decoder_depth: Number of decoder transformer layers.
        semantic_depth: Depth of semantic branch.
        semantic_vocab_size: Semantic codebook size.
        semantic_mask_ratio: Mask ratio for semantic training.
        detail_depth: Depth of detail branch.
        detail_vocab_size: Detail codebook size.
        detail_timesteps: Number of D3PM diffusion steps.
        detail_transition_type: "absorbing" or "uniform".
        detail_schedule_type: Noise schedule type.
        in_channels: Input channels (3 for RGB).
        patch_size: Patch size (T, H, W).
        mlp_ratio: MLP hidden dimension ratio.
        dropout: Dropout rate.
        use_ema_quantizer: Use EMA for codebook updates.

    Returns:
        Configured DiffusionGenerator model.

    Example:
        >>> model = create_model(d_model=768, detail_timesteps=100)
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    """
    return DiffusionGenerator(
        in_channels=in_channels,
        patch_size=patch_size,
        d_model=d_model,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        semantic_vocab_size=semantic_vocab_size,
        detail_vocab_size=detail_vocab_size,
        use_ema_quantizer=use_ema_quantizer,
        semantic_depth=semantic_depth,
        semantic_mask_ratio=semantic_mask_ratio,
        detail_depth=detail_depth,
        detail_timesteps=detail_timesteps,
        detail_transition_type=detail_transition_type,
        detail_schedule_type=detail_schedule_type,
        shallow_layers=4,
        shallow_base_channels=64,
        degradation_dim=32,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )


# ============================================================================
# Training
# ============================================================================

def train(
    data_dir: str,
    output_dir: str = "./checkpoints",
    groundtruth_dir: Optional[str] = None,
    # Training settings
    batch_size: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    # Save/Eval
    save_every: int = 5,
    eval_every: int = 1,
    # Data settings
    image_size: int = 256,
    num_workers: int = 4,
    # Video settings
    is_video: bool = False,
    num_frames: int = 16,
    # Model settings
    d_model: int = 512,
    detail_timesteps: int = 50,
    # Other
    device: str = "cuda",
    resume_from: Optional[str] = None,
    use_amp: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Train the diffusion generator.

    Args:
        data_dir: Path to training data.
        output_dir: Directory to save checkpoints.
        groundtruth_dir: Path to groundtruth (for paired training).
        batch_size: Batch size per GPU.
        num_epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: LR warmup steps.
        grad_accum_steps: Gradient accumulation steps.
        max_grad_norm: Max gradient norm for clipping.
        save_every: Save checkpoint every N epochs.
        eval_every: Evaluate every N epochs.
        image_size: Training image size.
        num_workers: DataLoader workers.
        is_video: Train on video data.
        num_frames: Frames per video clip.
        d_model: Model dimension.
        detail_timesteps: D3PM diffusion steps.
        device: Training device.
        resume_from: Checkpoint to resume from.
        use_amp: Use automatic mixed precision.
        seed: Random seed.

    Returns:
        Dictionary with final training metrics.

    Example:
        >>> metrics = train(
        ...     data_dir="/path/to/images",
        ...     output_dir="./checkpoints",
        ...     batch_size=8,
        ...     num_epochs=100,
        ... )
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data configuration
    data_config = DataConfig(
        data_dir=data_dir,
        groundtruth_dir=groundtruth_dir,
        image_size=(image_size, image_size),
        use_augmentation=True,
        horizontal_flip=True,
        random_crop=True,
        apply_degradation=(groundtruth_dir is None),
        is_video=is_video,
        num_frames=num_frames,
        degradation_config=DegradationConfig(
            blur_enabled=True,
            noise_enabled=True,
            jpeg_enabled=True,
            resize_enabled=True,
        ),
    )

    # Create dataloader
    train_loader = create_diffusion_dataloader(
        data_dir=data_dir,
        groundtruth_dir=groundtruth_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        config=data_config,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    print(f"Created dataloader with {len(train_loader.dataset)} samples")

    # Create model
    print("Creating model...")
    generator = create_model(
        d_model=d_model,
        detail_timesteps=detail_timesteps,
    )
    generator = generator.to(device)

    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Create training task
    task = DiffusionGenerationTask(
        generator=generator,
        lambda_semantic=1.0,
        lambda_detail=1.0,
        lambda_vq=1.0,
        lambda_recon=0.0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        task.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.99),
    )

    # Learning rate scheduler
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema = EMA(task.parameters(), decay=0.9999)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    # Create trainer
    trainer = Trainer(
        task=task,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        ema=ema,
        grad_accum_steps=grad_accum_steps,
        max_norm=max_grad_norm,
    )

    # Resume from checkpoint
    start_epoch = 0
    if resume_from:
        print(f"Resuming from {resume_from}")
        trainer.load_checkpoint(resume_from)
        try:
            start_epoch = int(Path(resume_from).stem.split("_")[-1])
        except ValueError:
            pass

    # Training loop
    print("Starting training...")
    final_metrics = {}

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        scheduler.step()

        # Log metrics
        print(f"  Loss: {train_metrics['loss']:.4f}")
        if "semantic_loss" in train_metrics:
            print(f"  Semantic Loss: {train_metrics['semantic_loss']:.4f}")
        if "detail_loss" in train_metrics:
            print(f"  Detail Loss: {train_metrics['detail_loss']:.4f}")
        if "vq_loss" in train_metrics:
            print(f"  VQ Loss: {train_metrics['vq_loss']:.4f}")

        final_metrics = train_metrics

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = output_dir / f"checkpoint_{epoch + 1:04d}.pt"
            trainer.save_checkpoint(str(ckpt_path))
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        "generator": generator.state_dict(),
        "config": {
            "d_model": d_model,
            "detail_timesteps": detail_timesteps,
            "image_size": image_size,
        },
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    return final_metrics


# ============================================================================
# Inference
# ============================================================================

def demo_inference(
    checkpoint_path: str,
    input_image: str,
    output_path: str,
    degradation_type: str = "auto",
    semantic_steps: int = 10,
    detail_steps: int = 50,
    temperature: float = 1.0,
    device: str = "cuda",
) -> None:
    """Run inference on a single image.

    Args:
        checkpoint_path: Path to model checkpoint.
        input_image: Path to input image.
        output_path: Path to save restored image.
        degradation_type: Degradation type ("auto", "blur", "noise", "jpeg", "sr").
        semantic_steps: MaskGIT decoding steps.
        detail_steps: D3PM sampling steps.
        temperature: Sampling temperature.
        device: Inference device.

    Example:
        >>> demo_inference(
        ...     checkpoint_path="model.pt",
        ...     input_image="degraded.jpg",
        ...     output_path="restored.jpg",
        ... )
    """
    if not HAS_VISION:
        raise ImportError("torchvision and PIL required for inference")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    generator = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    elif "task" in checkpoint:
        # Extract generator state from task checkpoint
        state_dict = {}
        for k, v in checkpoint["task"].items():
            if k.startswith("generator."):
                state_dict[k[10:]] = v
        generator.load_state_dict(state_dict, strict=False)
    else:
        generator.load_state_dict(checkpoint)

    # Create inference wrapper
    inference = DiffusionInference(generator, device=device)

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image = Image.open(input_image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Restore
    print(f"Running inference (degradation_type={degradation_type})...")
    restored = inference.restore(
        image_tensor,
        degradation_type=degradation_type,
        semantic_steps=semantic_steps,
        detail_steps=detail_steps,
        temperature=temperature,
    )

    # Save result
    output = transforms.ToPILImage()(restored.squeeze(0).cpu().clamp(0, 1))
    output.save(output_path)
    print(f"Saved restored image to {output_path}")


def batch_inference(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    degradation_type: str = "auto",
    semantic_steps: int = 10,
    detail_steps: int = 50,
    batch_size: int = 4,
    device: str = "cuda",
) -> None:
    """Run inference on a directory of images.

    Args:
        checkpoint_path: Path to model checkpoint.
        input_dir: Directory containing input images.
        output_dir: Directory to save restored images.
        degradation_type: Degradation type.
        semantic_steps: MaskGIT steps.
        detail_steps: D3PM steps.
        batch_size: Inference batch size.
        device: Inference device.
    """
    if not HAS_VISION:
        raise ImportError("torchvision and PIL required for inference")

    from pathlib import Path

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    generator = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "generator" in checkpoint:
        generator.load_state_dict(checkpoint["generator"])
    else:
        generator.load_state_dict(checkpoint)

    inference = DiffusionInference(generator, device=device)

    # Find images
    extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))

    print(f"Found {len(images)} images")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Process images
    for i, img_path in enumerate(images):
        print(f"Processing {i+1}/{len(images)}: {img_path.name}")

        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        restored = inference.restore(
            image_tensor,
            degradation_type=degradation_type,
            semantic_steps=semantic_steps,
            detail_steps=detail_steps,
        )

        output = transforms.ToPILImage()(restored.squeeze(0).cpu().clamp(0, 1))
        output.save(output_dir / img_path.name)

    print(f"Done! Restored images saved to {output_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="D3PM Diffusion Generator Training and Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on single folder (degradation auto-applied)
  python -m atoken.scripts.train_diffusion train --data_dir /path/to/images

  # Train on paired folders
  python -m atoken.scripts.train_diffusion train --data_dir /path/to/degraded --gt_dir /path/to/clean

  # Run inference
  python -m atoken.scripts.train_diffusion infer --checkpoint model.pt --input img.jpg --output out.jpg

  # Batch inference
  python -m atoken.scripts.train_diffusion batch --checkpoint model.pt --input_dir ./inputs --output_dir ./outputs
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Path to training images")
    train_parser.add_argument("--gt_dir", type=str, default=None, help="Path to groundtruth images")
    train_parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--image_size", type=int, default=256, help="Training image size")
    train_parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    train_parser.add_argument("--detail_timesteps", type=int, default=50, help="D3PM timesteps")
    train_parser.add_argument("--device", type=str, default="cuda", help="Device")
    train_parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    train_parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    train_parser.add_argument("--is_video", action="store_true", help="Train on video data")
    train_parser.add_argument("--num_frames", type=int, default=16, help="Frames per video clip")
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference on single image")
    infer_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    infer_parser.add_argument("--input", type=str, required=True, help="Input image path")
    infer_parser.add_argument("--output", type=str, required=True, help="Output image path")
    infer_parser.add_argument("--degradation", type=str, default="auto", help="Degradation type")
    infer_parser.add_argument("--semantic_steps", type=int, default=10, help="MaskGIT steps")
    infer_parser.add_argument("--detail_steps", type=int, default=50, help="D3PM steps")
    infer_parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Batch inference command
    batch_parser = subparsers.add_parser("batch", help="Run batch inference")
    batch_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    batch_parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    batch_parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    batch_parser.add_argument("--degradation", type=str, default="auto", help="Degradation type")
    batch_parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    batch_parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    if args.command == "train":
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            groundtruth_dir=args.gt_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            d_model=args.d_model,
            detail_timesteps=args.detail_timesteps,
            device=args.device,
            resume_from=args.resume_from,
            use_amp=not args.no_amp,
            num_workers=args.num_workers,
            is_video=args.is_video,
            num_frames=args.num_frames,
            seed=args.seed,
        )
    elif args.command == "infer":
        demo_inference(
            checkpoint_path=args.checkpoint,
            input_image=args.input,
            output_path=args.output,
            degradation_type=args.degradation,
            semantic_steps=args.semantic_steps,
            detail_steps=args.detail_steps,
            device=args.device,
        )
    elif args.command == "batch":
        batch_inference(
            checkpoint_path=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            degradation_type=args.degradation,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
