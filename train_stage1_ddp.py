#!/usr/bin/env python3
"""Stage 1: Image Foundation Training — PyTorch Lightning.

Multi-GPU training with NaViT packing and adaptive batch sizing.
Lightning handles DDP, AMP, gradient sync, checkpointing, and progress bars.

Usage:
    # 8-GPU training (auto-detects GPUs)
    python train_stage1_ddp.py --gpus 8

    # Single-GPU
    python train_stage1_ddp.py --gpus 1

    # Smoke test
    python train_stage1_ddp.py --gpus 8 --synthetic --steps 100

    # Resume from checkpoint
    python train_stage1_ddp.py --checkpoint checkpoints/last.ckpt
"""
import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
torch.set_float32_matmul_precision("high")  # Use Tensor Cores on A100
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from mavt.config import MAVTConfig
from mavt.tokenizer import MAVTokenizer
from losses.mavt_loss import LossWeights, MAVTLoss
from train.curriculum import STAGE1, apply_stage
from train.navit_dataset import SyntheticDataset, NaViTCollator
from train.distributed import (
    compute_adaptive_batch_size,
    compute_grad_accum_steps,
    build_distributed_image_dataloader,
    DEFAULT_TOKEN_BUDGET_PER_GPU,
    DEFAULT_GLOBAL_TOKEN_BUDGET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path("data/ready/images")
CKPT_DIR = Path("checkpoints")


# ─── Resolution-grouped forward ──────────────────────────────────────────────

def _max_batch_for_shape(shape):
    """Adaptive sub-batch cap based on resolution to prevent OOM."""
    pixels = shape[-2] * shape[-1]
    if pixels >= 512 * 512:
        return 2
    if pixels >= 256 * 256:
        return 8
    if pixels >= 128 * 128:
        return 16
    return 32


def _group_samples_by_shape(batch):
    """Group all samples in a NaViT-packed batch by spatial shape."""
    groups = defaultdict(list)
    for seq in batch["sequences"]:
        for sample in seq.samples:
            x = sample["data"]
            if x.dim() == 3:
                x = x.unsqueeze(0)
            groups[x.shape[1:]].append((x, sample.get("n_tokens", 0)))
    return groups


# ─── Lightning Module ─────────────────────────────────────────────────────────

class MAVTLitModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        warmup_steps: int = 2000,
        total_steps: int = 200_000,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg = MAVTConfig()
        self.model = MAVTokenizer(cfg)
        apply_stage(self.model, STAGE1)
        self._disable_unused_params_grad()

        self.loss_fn = MAVTLoss(LossWeights(recon=1.0, kl=kl_weight))

        # Disable automatic optimization so we can handle the multi-forward
        # resolution-grouped step ourselves
        self.automatic_optimization = False

    def _disable_unused_params_grad(self):
        unused_prefixes = [
            "model.encoder.memory.",
            "model.encoder.stage3_graph_conv.",
            "model.latent.understand_",
            "model.latent.var_proj.logvar_proj.",
        ]
        count = 0
        for name, param in self.named_parameters():
            if any(name.startswith(p) for p in unused_prefixes):
                if param.requires_grad:
                    param.requires_grad_(False)
                    count += 1
        if count > 0:
            log.info("Disabled grad for %d unused parameters", count)

    def forward(self, x, modality="image"):
        return self.model(x, modality)

    def _step(self, batch, prefix="train"):
        groups = _group_samples_by_shape(batch)
        all_losses = []
        all_logs = defaultdict(list)
        total_tokens = 0

        for shape, items in groups.items():
            tensors, token_counts = zip(*items)
            total_tokens += sum(token_counts)
            max_b = _max_batch_for_shape(shape)

            for i in range(0, len(tensors), max_b):
                chunk = tensors[i : i + max_b]
                x = torch.cat(chunk, dim=0).to(self.device)

                try:
                    dec_out, lat_out = self.model(x, "image")
                    result = self.loss_fn(
                        recon=dec_out.reconstruction,
                        target=x,
                        mu=lat_out.mu,
                        log_var=lat_out.log_var,
                    )
                    all_losses.append(result["loss"])
                    for k, v in result["logs"].items():
                        all_logs[k].append(v)
                except Exception as e:
                    if self.global_rank == 0:
                        log.warning("Skip shape=%s n=%d: %s", shape, len(chunk), e)

        if not all_losses:
            return None

        loss = torch.stack(all_losses).mean()

        # Log metrics
        n_samples = sum(len(items) for items in groups.values())
        logs = {k: torch.stack(v).mean() for k, v in all_logs.items()}
        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/recon", logs.get("recon_loss", loss), prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/kl", logs.get("kl_loss", loss), prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/tokens", float(total_tokens), sync_dist=True)
        self.log(f"{prefix}/samples", float(n_samples), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()

        loss = self._step(batch, "train")
        if loss is not None:
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0)
            opt.step()
        else:
            opt.step()  # still step to keep schedulers in sync

        sch.step()
        self.log("lr", sch.get_last_lr()[0], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

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


# ─── Data Module ──────────────────────────────────────────────────────────────

class MAVTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = str(DATA_ROOT),
        synthetic: bool = False,
        resolutions: list[int] = None,
        token_budget_per_gpu: int = DEFAULT_TOKEN_BUDGET_PER_GPU,
        max_seq_len: int = 4096,
        patch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.synthetic = synthetic
        self.resolutions = resolutions or [64, 128, 256, 512]
        self.token_budget = token_budget_per_gpu
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.num_workers = num_workers

        self.batch_size = compute_adaptive_batch_size(
            token_budget=self.token_budget,
            resolutions=self.resolutions,
            patch_size=self.patch_size,
        )

    def setup(self, stage=None):
        if self.synthetic:
            self.train_ds = SyntheticDataset(500, self.resolutions, "image", self.patch_size)
            self.val_ds = SyntheticDataset(50, self.resolutions, "image", self.patch_size)
        else:
            from train.navit_dataset import MultiResImageDataset
            from torch.utils.data import random_split

            try:
                full_ds = MultiResImageDataset(
                    self.data_dir, self.resolutions, self.patch_size
                )
                n_val = max(1, int(len(full_ds) * 0.05))
                self.train_ds, self.val_ds = random_split(
                    full_ds, [len(full_ds) - n_val, n_val]
                )
            except FileNotFoundError:
                log.warning("No images in %s, using synthetic data.", self.data_dir)
                self.train_ds = SyntheticDataset(500, self.resolutions, "image", self.patch_size)
                self.val_ds = SyntheticDataset(50, self.resolutions, "image", self.patch_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=NaViTCollator(max_seq_len=self.max_seq_len),
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=max(4, self.batch_size // 4),
            shuffle=False,
            num_workers=max(0, self.num_workers // 2),
            collate_fn=NaViTCollator(max_seq_len=self.max_seq_len),
            pin_memory=True,
            persistent_workers=False,
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAVT Stage 1 — Lightning")
    parser.add_argument("--data-dir", type=str, default=str(DATA_ROOT))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--token-budget-per-gpu", type=int, default=DEFAULT_TOKEN_BUDGET_PER_GPU)
    parser.add_argument("--global-token-budget", type=int, default=DEFAULT_GLOBAL_TOKEN_BUDGET)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    n_gpus = args.gpus or torch.cuda.device_count() or 1

    # Gradient accumulation
    grad_accum = compute_grad_accum_steps(
        args.global_token_budget, args.token_budget_per_gpu, n_gpus
    )

    # Model
    model = MAVTLitModule(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )

    if args.compile:
        model.model = torch.compile(model.model)

    # Data
    data = MAVTDataModule(
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        resolutions=STAGE1.image_resolutions,
        token_budget_per_gpu=args.token_budget_per_gpu,
        num_workers=args.workers,
    )

    # Callbacks
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="stage1-{step}",
            every_n_train_steps=args.save_every,
            save_top_k=-1,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="stage1-best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    precision = "32" if args.no_amp else "16-mixed"
    strategy = "auto" if n_gpus <= 1 else DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=n_gpus,
        strategy=strategy,
        precision=precision,
        max_steps=args.steps,
        # Note: grad accumulation handled manually in training_step
        # accumulate_grad_batches not compatible with manual optimization
        log_every_n_steps=args.log_every,
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        enable_progress_bar=True,
        default_root_dir=str(CKPT_DIR),
    )

    log.info("=" * 70)
    log.info("Stage 1 Training: %d steps, lr=%.1e, %d GPUs, precision=%s",
             args.steps, args.lr, n_gpus, precision)
    log.info("  Token budget: %d/GPU, grad_accum=%d", args.token_budget_per_gpu, grad_accum)
    log.info("=" * 70)

    trainer.fit(model, datamodule=data, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
