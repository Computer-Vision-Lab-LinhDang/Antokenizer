#!/usr/bin/env python3
"""Multi-modal training: Image + Video + 3D — PyTorch Lightning.

Each optimizer step consists of 3 forward passes (one per modality).
Gradients are accumulated across the 3 modality batches before stepping.

Data shapes fed to the model:
    image: (B, C, H, W)
    video: (B, C, T, H, W)
    3d:    (B, 3, C_tri, S, S)   triplane

Usage:
    # 8-GPU training
    python train_multimodal_ddp.py --gpus 8

    # Single-GPU with sample data
    python train_multimodal_ddp.py --gpus 1 --data-root datasets/datasets/ready_sample_100

    # Smoke test with synthetic data
    python train_multimodal_ddp.py --gpus 1 --synthetic --steps 50

    # Resume from checkpoint
    python train_multimodal_ddp.py --checkpoint checkpoints/multimodal/last.ckpt
"""
import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
torch.set_float32_matmul_precision("high")
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from mavt.config import MAVTConfig
from mavt.tokenizer import MAVTokenizer
from losses.mavt_loss import LossWeights, MAVTLoss
from train.curriculum import STAGE3, apply_stage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path("datasets/datasets/ready_sample_100")
CKPT_DIR = Path("checkpoints/multimodal")

_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD = [0.5, 0.5, 0.5]


# ============================================================================
# Datasets — one per modality
# ============================================================================

class SimpleImageDataset(Dataset):
    """Load images from a directory, resize to fixed resolution."""

    _EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root: str, resolution: int = 256):
        self.root = Path(root)
        self.resolution = resolution
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self._EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images in {root}")
        log.info("SimpleImageDataset: %d images at %dpx", len(self.paths), resolution)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
            tensor = TF.normalize(TF.to_tensor(img), _NORM_MEAN, _NORM_STD)
        except Exception:
            tensor = torch.zeros(3, self.resolution, self.resolution)
        return tensor  # (3, H, W)


class SimpleVideoDataset(Dataset):
    """Load video clips from a directory of MP4 files."""

    _EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, root: str, resolution: int = 128, n_frames: int = 8):
        self.root = Path(root)
        self.resolution = resolution
        self.n_frames = n_frames
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self._EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No videos in {root}")
        log.info("SimpleVideoDataset: %d videos at %dpx, %d frames",
                 len(self.paths), resolution, n_frames)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            frames = self._load_video(self.paths[idx])
        except Exception:
            frames = torch.zeros(3, self.n_frames, self.resolution, self.resolution)
        return frames  # (3, T, H, W)

    def _load_video(self, path: Path) -> torch.Tensor:
        import cv2
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Cannot read {path}")

        # Uniform sampling
        T = self.n_frames
        if total < T:
            indices = list(range(total)) + [total - 1] * (T - total)
        else:
            indices = [int(i) for i in torch.linspace(0, total - 1, T)]

        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(3, self.resolution, self.resolution))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resolution, self.resolution))
            t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            t = TF.normalize(t, _NORM_MEAN, _NORM_STD)
            frames.append(t)

        cap.release()
        return torch.stack(frames, dim=1)  # (3, T, H, W)


class SimpleObject3DDataset(Dataset):
    """Load 3D objects as triplanes from pre-rendered views.

    Each object dir has view_0.png..view_7.png + cameras.json.
    Triplane is built from 3 canonical views resized to (S, S).
    Output shape: (3, 3, S, S) — 3 planes, each RGB, at S resolution.

    Note: triplane_res must be >= 64 so that total tokens (3 * (S/16)^2)
    exceeds the number of attention heads (16), otherwise 4D RoPE
    layout detection fails.
    """

    def __init__(self, render_dir: str, triplane_res: int = 64):
        self.render_dir = Path(render_dir)
        self.triplane_res = triplane_res
        self.object_dirs = sorted(
            d for d in self.render_dir.iterdir()
            if d.is_dir() and (d / "view_0.png").exists()
        )
        if not self.object_dirs:
            raise FileNotFoundError(f"No 3D objects in {render_dir}")
        log.info("SimpleObject3DDataset: %d objects at %dpx triplane",
                 len(self.object_dirs), triplane_res)

    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        obj_dir = self.object_dirs[idx]
        try:
            triplane = self._build_triplane(obj_dir)
        except Exception:
            triplane = torch.zeros(3, 3, self.triplane_res, self.triplane_res)
        return triplane  # (3, 3, S, S)

    def _build_triplane(self, obj_dir: Path) -> torch.Tensor:
        """Build triplane from 3 views: front (0), side (2), top (4)."""
        planes = []
        for view_idx in [0, 2, 4]:  # roughly XY, XZ, YZ canonical views
            view_path = obj_dir / f"view_{view_idx}.png"
            img = Image.open(view_path).convert("RGB")
            t = TF.to_tensor(img)  # (3, H, W), [0, 1]
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.triplane_res, self.triplane_res),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            t = t * 2 - 1  # normalize to [-1, 1]
            planes.append(t)
        return torch.stack(planes, dim=0)  # (3, 3, S, S)


class SyntheticMultiModalDataset(Dataset):
    """Synthetic data for smoke testing a single modality."""

    def __init__(self, n: int, modality: str, resolution: int = 128,
                 n_frames: int = 8, triplane_res: int = 64):
        self.n = n
        self.modality = modality
        self.resolution = resolution
        self.n_frames = n_frames
        self.triplane_res = triplane_res

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.modality == "image":
            return torch.randn(3, self.resolution, self.resolution)
        elif self.modality == "video":
            return torch.randn(3, self.n_frames, self.resolution, self.resolution)
        else:  # 3d
            return torch.randn(3, 3, self.triplane_res, self.triplane_res)


# ============================================================================
# Lightning Module — 3-modality gradient accumulation
# ============================================================================

class MultiModalLitModule(pl.LightningModule):
    """MAVT multi-modal training with per-modality gradient accumulation.

    Each training step:
        1. Forward image batch  → loss_img.backward()   (grads accumulate)
        2. Forward video batch  → loss_vid.backward()   (grads accumulate)
        3. Forward 3D batch     → loss_3d.backward()    (grads accumulate)
        4. clip_gradients → optimizer.step → scheduler.step
    """

    def __init__(
        self,
        lr: float = 2e-5,
        warmup_steps: int = 500,
        total_steps: int = 50_000,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg = MAVTConfig()
        self.model = MAVTokenizer(cfg)
        apply_stage(self.model, STAGE3)

        self.loss_fn = MAVTLoss(LossWeights(recon=1.0, kl=kl_weight))
        self.automatic_optimization = False

    def forward(self, x, modality=None):
        return self.model(x, modality)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        total_loss = 0.0
        modality_losses = {}

        # batch is a dict: {"image": tensor, "video": tensor, "3d": tensor}
        modalities = [
            ("image", batch.get("image"), "image"),
            ("video", batch.get("video"), "video"),
            ("3d",    batch.get("3d"),    "3d"),
        ]

        n_valid = sum(1 for _, data, _ in modalities if data is not None)
        if n_valid == 0:
            opt.step()
            sch.step()
            return

        for name, data, mod_key in modalities:
            if data is None:
                continue

            try:
                dec_out, lat_out = self.model(data, mod_key)
                result = self.loss_fn(
                    recon=dec_out.reconstruction,
                    target=data,
                    mu=lat_out.mu,
                    log_var=lat_out.log_var,
                )
                # Divide by n_valid so total gradient magnitude is normalized
                loss = result["loss"] / n_valid
                self.manual_backward(loss)

                modality_losses[name] = result["loss"].detach()
                total_loss += result["loss"].detach()

            except Exception as e:
                if self.global_rank == 0:
                    log.warning("Skip %s: %s", name, e)

        self.clip_gradients(opt, gradient_clip_val=1.0)
        opt.step()
        sch.step()

        # Logging
        if modality_losses:
            self.log("train/loss", total_loss / max(len(modality_losses), 1),
                     prog_bar=True, sync_dist=True)
            for name, val in modality_losses.items():
                self.log(f"train/{name}_loss", val, prog_bar=True, sync_dist=True)
        self.log("lr", sch.get_last_lr()[0], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        total_loss = 0.0
        n = 0
        for name, mod_key in [("image", "image"), ("video", "video"), ("3d", "3d")]:
            data = batch.get(name)
            if data is None:
                continue
            try:
                dec_out, lat_out = self.model(data, mod_key)
                result = self.loss_fn(
                    recon=dec_out.reconstruction,
                    target=data,
                    mu=lat_out.mu,
                    log_var=lat_out.log_var,
                )
                self.log(f"val/{name}_loss", result["loss"], sync_dist=True)
                total_loss += result["loss"]
                n += 1
            except Exception:
                pass
        if n > 0:
            self.log("val/loss", total_loss / n, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=self.hparams.lr, betas=(0.9, 0.95), weight_decay=0.05)

        warmup = self.hparams.warmup_steps
        total = self.hparams.total_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# ============================================================================
# Data Module — 3 independent DataLoaders combined
# ============================================================================

def _collate_tensors(batch):
    """Simple collate: stack tensors into a batch."""
    return torch.stack(batch, dim=0)


class MultiModalDataModule(pl.LightningDataModule):
    """Provides 3 DataLoaders (one per modality) via CombinedLoader.

    Each training step receives one batch from each modality.
    """

    def __init__(
        self,
        data_root: str = str(DATA_ROOT),
        synthetic: bool = False,
        image_res: int = 256,
        video_res: int = 128,
        video_frames: int = 8,
        triplane_res: int = 32,
        batch_size_image: int = 4,
        batch_size_video: int = 2,
        batch_size_3d: int = 4,
        num_workers: int = 2,
        synthetic_n: int = 100,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.synthetic = synthetic
        self.image_res = image_res
        self.video_res = video_res
        self.video_frames = video_frames
        self.triplane_res = triplane_res
        self.batch_size_image = batch_size_image
        self.batch_size_video = batch_size_video
        self.batch_size_3d = batch_size_3d
        self.num_workers = num_workers
        self.synthetic_n = synthetic_n

    def setup(self, stage=None):
        if self.synthetic:
            self.train_image = SyntheticMultiModalDataset(
                self.synthetic_n, "image", self.image_res)
            self.train_video = SyntheticMultiModalDataset(
                self.synthetic_n, "video", self.video_res, self.video_frames)
            self.train_3d = SyntheticMultiModalDataset(
                self.synthetic_n, "3d", triplane_res=self.triplane_res)

            self.val_image = SyntheticMultiModalDataset(
                max(8, self.synthetic_n // 10), "image", self.image_res)
            self.val_video = SyntheticMultiModalDataset(
                max(8, self.synthetic_n // 10), "video", self.video_res, self.video_frames)
            self.val_3d = SyntheticMultiModalDataset(
                max(8, self.synthetic_n // 10), "3d", triplane_res=self.triplane_res)
        else:
            img_dir = self.data_root / "images"
            vid_dir = self.data_root / "videos"
            obj_dir = self.data_root / "3d_objects" / "renders"

            try:
                self.train_image = SimpleImageDataset(str(img_dir), self.image_res)
            except FileNotFoundError:
                log.warning("No images at %s, using synthetic", img_dir)
                self.train_image = SyntheticMultiModalDataset(
                    self.synthetic_n, "image", self.image_res)

            try:
                self.train_video = SimpleVideoDataset(
                    str(vid_dir), self.video_res, self.video_frames)
            except FileNotFoundError:
                log.warning("No videos at %s, using synthetic", vid_dir)
                self.train_video = SyntheticMultiModalDataset(
                    self.synthetic_n, "video", self.video_res, self.video_frames)

            try:
                self.train_3d = SimpleObject3DDataset(str(obj_dir), self.triplane_res)
            except FileNotFoundError:
                log.warning("No 3D objects at %s, using synthetic", obj_dir)
                self.train_3d = SyntheticMultiModalDataset(
                    self.synthetic_n, "3d", triplane_res=self.triplane_res)

            # Validation: use last 10% of each dataset or synthetic
            self._split_val()

    def _split_val(self):
        """Split off 10% for validation from each training set."""
        from torch.utils.data import random_split

        for attr, val_attr in [
            ("train_image", "val_image"),
            ("train_video", "val_video"),
            ("train_3d", "val_3d"),
        ]:
            ds = getattr(self, attr)
            n_val = max(1, int(len(ds) * 0.1))
            n_train = len(ds) - n_val
            train_ds, val_ds = random_split(
                ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            setattr(self, attr, train_ds)
            setattr(self, val_attr, val_ds)

    def train_dataloader(self):
        from lightning.pytorch.utilities.combined_loader import CombinedLoader

        loaders = {
            "image": DataLoader(
                self.train_image,
                batch_size=self.batch_size_image,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=_collate_tensors,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
            ),
            "video": DataLoader(
                self.train_video,
                batch_size=self.batch_size_video,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=_collate_tensors,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
            ),
            "3d": DataLoader(
                self.train_3d,
                batch_size=self.batch_size_3d,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=_collate_tensors,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        from lightning.pytorch.utilities.combined_loader import CombinedLoader

        loaders = {
            "image": DataLoader(
                self.val_image,
                batch_size=max(1, self.batch_size_image // 2),
                shuffle=False,
                num_workers=max(0, self.num_workers // 2),
                collate_fn=_collate_tensors,
                pin_memory=True,
            ),
            "video": DataLoader(
                self.val_video,
                batch_size=max(1, self.batch_size_video // 2),
                shuffle=False,
                num_workers=max(0, self.num_workers // 2),
                collate_fn=_collate_tensors,
                pin_memory=True,
            ),
            "3d": DataLoader(
                self.val_3d,
                batch_size=max(1, self.batch_size_3d // 2),
                shuffle=False,
                num_workers=max(0, self.num_workers // 2),
                collate_fn=_collate_tensors,
                pin_memory=True,
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MAVT Multi-modal Training — Lightning")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--kl-weight", type=float, default=1e-4)

    # Resolutions
    parser.add_argument("--image-res", type=int, default=256)
    parser.add_argument("--video-res", type=int, default=128)
    parser.add_argument("--video-frames", type=int, default=8,
                        help="Number of frames per video clip (must be divisible by tau=2)")
    parser.add_argument("--triplane-res", type=int, default=64,
                        help="Triplane resolution (must be >= 64 for RoPE compatibility)")

    # Batch sizes (per modality per GPU)
    parser.add_argument("--batch-image", type=int, default=4)
    parser.add_argument("--batch-video", type=int, default=2)
    parser.add_argument("--batch-3d", type=int, default=4)

    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=25000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    n_gpus = args.gpus or torch.cuda.device_count() or 1

    # Model
    model = MultiModalLitModule(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        kl_weight=args.kl_weight,
    )

    if args.compile:
        model.model = torch.compile(model.model)

    # Data
    data = MultiModalDataModule(
        data_root=args.data_root,
        synthetic=args.synthetic,
        image_res=args.image_res,
        video_res=args.video_res,
        video_frames=args.video_frames,
        triplane_res=args.triplane_res,
        batch_size_image=args.batch_image,
        batch_size_video=args.batch_video,
        batch_size_3d=args.batch_3d,
        num_workers=args.workers,
    )

    # Callbacks
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="multimodal-1-{step}",
            every_n_train_steps=args.save_every,
            save_top_k=-1,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="multimodal-1-best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    precision = "32" if args.no_amp else "16-mixed"
    strategy = "auto" if n_gpus <= 1 else DDPStrategy(find_unused_parameters=True)
    tb_logger = TensorBoardLogger(save_dir=str(CKPT_DIR), name="multimodal_logs_1")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=n_gpus,
        strategy=strategy,
        precision=precision,
        max_steps=args.steps,
        log_every_n_steps=args.log_every,
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        logger=tb_logger,
        enable_progress_bar=True,
        default_root_dir=str(CKPT_DIR),
    )

    log.info("=" * 70)
    log.info("Multi-modal Training: %d steps, lr=%.1e, %d GPUs, precision=%s",
             args.steps, args.lr, n_gpus, precision)
    log.info("  Image: %dpx, batch=%d", args.image_res, args.batch_image)
    log.info("  Video: %dpx x %d frames, batch=%d",
             args.video_res, args.video_frames, args.batch_video)
    log.info("  3D:    %dpx triplane, batch=%d", args.triplane_res, args.batch_3d)
    log.info("  Data root: %s", args.data_root)
    log.info("  Gradient = sum(image_grad + video_grad + 3d_grad) per step")
    log.info("=" * 70)

    trainer.fit(model, datamodule=data, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
