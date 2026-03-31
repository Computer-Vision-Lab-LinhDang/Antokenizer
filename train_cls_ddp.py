#!/usr/bin/env python3
"""Classification Training with MAVTClassifier — PyTorch Lightning.

Multi-GPU training for image classification using the MAVTClassifier model,
which jointly optimizes reconstruction + KL + classification losses.

Usage:
    # 8-GPU training on CIFAR-10
    python train_cls_ddp.py --gpus 8 --dataset cifar10

    # Single-GPU on ImageFolder dataset
    python train_cls_ddp.py --gpus 1 --dataset imagefolder --data-dir data/my_dataset

    # Fine-tune from stage-1 pretrained checkpoint
    python train_cls_ddp.py --gpus 8 --dataset cifar10 --pretrained checkpoints/stage1-best.ckpt

    # Resume training
    python train_cls_ddp.py --checkpoint checkpoints/cls-last.ckpt

    # Smoke test
    python train_cls_ddp.py --gpus 1 --dataset cifar10 --steps 100
"""
import argparse
import logging
import math
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
torch.set_float32_matmul_precision("high")
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

from mavt.config import MAVTConfig
from mavt.mavt_cls import MAVTClassifier
from train.curriculum import STAGE1, apply_stage
from train.data import DataConfig, create_dataloader, ClassificationWrapper, OpenImagesDataset, _image_transforms

from train_multimodal_ddp import MultiModalLitModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CKPT_DIR = Path("checkpoints/cls")

# ─── Dataset info ────────────────────────────────────────────────────────────

DATASET_NUM_CLASSES = {
    "cifar10": 10,
    "mnist": 10,
}


def _get_num_classes(dataset_name: str, data_dir: str, data_module=None) -> int:
    """Resolve number of classes for the dataset."""
    name = dataset_name.lower()
    if name == "openimages" and data_module is not None and data_module.num_classes is not None:
        return data_module.num_classes
    if name in DATASET_NUM_CLASSES:
        return DATASET_NUM_CLASSES[name]
    # ImageFolder: count subdirectories
    data_path = Path(data_dir)
    if data_path.is_dir():
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if subdirs:
            return len(subdirs)
    raise ValueError(
        f"Cannot determine num_classes for '{dataset_name}' at '{data_dir}'. "
        "Use --num-classes to specify explicitly."
    )


# ─── Lightning Module ─────────────────────────────────────────────────────────

def get_classifier_model_with_pretrained(num_classes, classifier_name, cfg, checkpoint_path="/home/sagemaker-user/ws/Antoken/checkpoints/multimodal/multimodal-1-step=50000.ckpt"):
    base_model = MultiModalLitModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    classifier = MAVTClassifier(num_classes=num_classes, classifier_name=classifier_name, cfg=cfg)
    classifier.patchify = base_model.model.patchify
    classifier.encoder = base_model.model.encoder
    classifier.latent = base_model.model.latent
    classifier.decoder = base_model.model.decoder 
    return classifier

class MAVTClsLitModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        classifier_name: str = "fastformer",
        lr: float = 1e-4,
        warmup_steps: int = 2000,
        total_steps: int = 50_000,
        kl_weight: float = 1e-4,
        cls_weight: float = 1.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg = MAVTConfig()
        # self.model = MAVTClassifier(
        #     num_classes=num_classes,
        #     classifier_name=classifier_name,
        #     cfg=cfg,
        # )
        self.model = get_classifier_model_with_pretrained(
            num_classes=num_classes, 
            classifier_name=classifier_name,
            cfg=cfg
        )
        # apply_stage(self.model, STAGE1)

        if freeze_backbone:
            self._freeze_backbone()

        self.automatic_optimization = False

    def _freeze_backbone(self):
        """Freeze patchify + encoder + latent, only train classifier."""
        count = 0
        for name, param in self.model.named_parameters():
            if not name.startswith("classifier."):
                if param.requires_grad:
                    param.requires_grad_(False)
                    count += 1
        log.info("Froze %d backbone parameters (training classifier only)", count)

    def forward(self, x, modality="image"):
        return self.model(x, modality)

    def _step(self, batch, prefix="train"):
        x = batch["image"].to(self.device)
        target = batch["target"].to(self.device)

        decoder_out, latent_out, cls_out = self.model(x, "image")

        losses = self.model.compute_loss(
            x=x,
            decoder_out=decoder_out,
            latent_out=latent_out,
            cls_out=cls_out,
            cls_target=target,
            cls_weight=self.hparams.cls_weight,
            kl_weight=self.hparams.kl_weight,
        )

        # Metrics: multi-label (float target) vs single-label (long target)
        if target.dtype == torch.float32:
            preds = (cls_out.sigmoid() > 0.5).float()
            acc = ((preds == target).float().mean(dim=-1)).mean()
            tp = (preds * target).sum()
            fp = (preds * (1 - target)).sum()
            fn = ((1 - preds) * target).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        else:
            preds = cls_out.argmax(dim=-1)
            acc = (preds == target).float().mean()
            precision = acc
            recall = acc
            f1 = acc  # single-label: micro-F1 == accuracy

        self.log(f"{prefix}/loss", losses["total_loss"], prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/recon", losses["recon_loss"], prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/kl", losses["kl_loss"], sync_dist=True)
        self.log(f"{prefix}/cls", losses["cls_loss"], prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/acc", acc, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}/precision", precision, sync_dist=True)
        self.log(f"{prefix}/recall", recall, sync_dist=True)
        self.log(f"{prefix}/f1", f1, prog_bar=True, sync_dist=True)

        return losses["total_loss"]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()

        loss = self._step(batch, "train")
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0)
        opt.step()

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

ANNOTATIONS_DIR = Path("datasets/datasets/open_images_v7/annotations")
DEFAULT_TRAIN_IMG_DIR = Path("datasets/ready/images")
DEFAULT_VAL_IMG_DIR = Path("datasets/ready/images_val")
DEFAULT_TRAIN_ANN = ANNOTATIONS_DIR / "oidv7-train-annotations-human-imagelabels.csv"
DEFAULT_VAL_ANN = ANNOTATIONS_DIR / "oidv7-val-annotations-human-imagelabels.csv"


class ClsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "openimages",
        data_dir: str = "./data",
        train_img_dir: str = str(DEFAULT_TRAIN_IMG_DIR),
        val_img_dir: str = str(DEFAULT_VAL_IMG_DIR),
        train_ann: str = str(DEFAULT_TRAIN_ANN),
        val_ann: str = str(DEFAULT_VAL_ANN),
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.train_img_dir = train_img_dir
        self.val_img_dir = val_img_dir
        self.train_ann = train_ann
        self.val_ann = val_ann
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.num_classes = None

    def _make_dataloader(self, train: bool) -> DataLoader:
        cfg = DataConfig(
            name=self.dataset_name,
            root=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
            pin_memory=True,
            persistent_workers=self.num_workers > 0 and train,
        )
        return create_dataloader(cfg, task="classification", train=train)

    def setup(self, stage=None):
        name = self.dataset_name.lower()
        self._is_openimages = name == "openimages"
        self._has_builtin_split = name in {"cifar10", "mnist"}

        if self._is_openimages:
            train_transform = _image_transforms(
                self.image_size, train=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
            )
            val_transform = _image_transforms(
                self.image_size, train=False, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
            )
            log.info("Loading Open Images train annotations from %s ...", self.train_ann)
            self._train_ds = OpenImagesDataset(
                image_dir=self.train_img_dir,
                annotation_csv=self.train_ann,
                transform=train_transform,
            )
            # Share label vocabulary with val set
            label_to_idx = self._train_ds.label_to_idx
            self.num_classes = self._train_ds.num_classes

            log.info("Loading Open Images val annotations from %s ...", self.val_ann)
            self._val_ds = OpenImagesDataset(
                image_dir=self.val_img_dir,
                annotation_csv=self.val_ann,
                transform=val_transform,
                label_to_idx=label_to_idx,
            )
            log.info("Open Images: %d train, %d val, %d classes",
                     len(self._train_ds), len(self._val_ds), self.num_classes)

        elif not self._has_builtin_split:
            from torchvision import datasets as tv_datasets
            cfg = DataConfig(name=self.dataset_name, root=self.data_dir, image_size=self.image_size)
            transform = _image_transforms(
                self.image_size, train=True, mean=list(cfg.mean), std=list(cfg.std),
            )
            base = tv_datasets.ImageFolder(root=self.data_dir, transform=transform)
            n_val = max(1, int(len(base) * self.val_split))
            train_ds, val_ds = random_split(base, [len(base) - n_val, n_val])
            self._train_ds = ClassificationWrapper(train_ds)
            self._val_ds = ClassificationWrapper(val_ds)

    def train_dataloader(self):
        if not self._is_openimages and self._has_builtin_split:
            return self._make_dataloader(train=True)
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        if not self._is_openimages and self._has_builtin_split:
            return self._make_dataloader(train=False)
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )


# ─── Pretrained weight loading ───────────────────────────────────────────────

def load_pretrained_backbone(model: MAVTClassifier, ckpt_path: str) -> None:
    """Load Stage-1 MAVTokenizer weights into the classifier's backbone."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Lightning checkpoint: state_dict lives under "state_dict" key
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip "model." prefix from Lightning module keys (e.g. "model.encoder.xxx" -> "encoder.xxx")
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v

    # Load with strict=False to skip the classifier head (not in stage-1 checkpoint)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    log.info(
        "Loaded pretrained backbone from %s (missing=%d, unexpected=%d)",
        ckpt_path, len(missing), len(unexpected),
    )
    if missing:
        log.info("  Missing keys (expected for classifier head): %s", missing[:10])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAVT Classification — Lightning")
    parser.add_argument("--dataset", type=str, default="openimages",
                        help="Dataset name: openimages, cifar10, mnist, or imagefolder")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--train-img-dir", type=str, default=str(DEFAULT_TRAIN_IMG_DIR))
    parser.add_argument("--val-img-dir", type=str, default=str(DEFAULT_VAL_IMG_DIR))
    parser.add_argument("--train-ann", type=str, default=str(DEFAULT_TRAIN_ANN))
    parser.add_argument("--val-ann", type=str, default=str(DEFAULT_VAL_ANN))
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Override num_classes (auto-detected if omitted)")
    parser.add_argument("--classifier", type=str, default="fastformer",
                        choices=["linear", "fastformer"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze encoder/decoder, train classifier only")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to stage-1 checkpoint for backbone init")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume full training from this checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    n_gpus = args.gpus or torch.cuda.device_count() or 1

    # Data — setup first so OpenImages can auto-detect num_classes
    data = ClsDataModule(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        train_img_dir=args.train_img_dir,
        val_img_dir=args.val_img_dir,
        train_ann=args.train_ann,
        val_ann=args.val_ann,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    data.setup()

    # Resolve num_classes
    num_classes = args.num_classes
    if num_classes is None:
        num_classes = _get_num_classes(args.dataset, args.data_dir, data_module=data)

    # Model
    model = MAVTClsLitModule(
        num_classes=num_classes,
        classifier_name=args.classifier,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        kl_weight=args.kl_weight,
        cls_weight=args.cls_weight,
        freeze_backbone=True,
    )

    # # Load pretrained backbone
    # if args.pretrained:
    #     load_pretrained_backbone(model.model, args.pretrained)

    if args.compile:
        model.model = torch.compile(model.model)

    # Callbacks
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="cls-{step}",
            every_n_train_steps=args.save_every,
            save_top_k=-1,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="cls-best",
            monitor="val/f1",
            mode="max",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    precision = "32" if args.no_amp else "16-mixed"
    strategy = "auto" if n_gpus <= 1 else DDPStrategy(find_unused_parameters=True)
    tb_logger = TensorBoardLogger(save_dir=str(CKPT_DIR), name="cls_logs")

    trainer = pl.Trainer(
        # accelerator = "cpu",
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
    log.info("Classification Training: %d steps, lr=%.1e, %d GPUs, precision=%s",
             args.steps, args.lr, n_gpus, precision)
    log.info("  Dataset: %s, num_classes=%d, classifier=%s",
             args.dataset, num_classes, args.classifier)
    log.info("  Batch size: %d, freeze_backbone=%s", args.batch_size, args.freeze_backbone)
    log.info("=" * 70)

    trainer.fit(model, datamodule=data, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
