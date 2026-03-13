#!/usr/bin/env python3
"""Stage 1: Image Foundation Training + Evaluation.

Train MAVTokenizer on images from Open Images V7 with NaViT packing.
- Multi-resolution images [64, 128, 256, 512]
- SigLIP2 fully frozen
- Reconstruction loss (L1) + KL divergence

Usage:
    # Train (uses GPU if available)
    python train_stage1.py

    # Train with custom settings
    python train_stage1.py --steps 5000 --batch-size 32 --lr 1e-4

    # Eval only (from checkpoint)
    python train_stage1.py --eval-only --checkpoint checkpoints/stage1_best.pt

    # Quick smoke test with synthetic data
    python train_stage1.py --synthetic --steps 50 --batch-size 8
"""
import argparse
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from mavt.config import MAVTConfig
from mavt.tokenizer import MAVTokenizer
from losses.mavt_loss import LossWeights, MAVTLoss
from train.navit_dataset import (
    MultiResImageDataset,
    SyntheticDataset,
    NaViTCollator,
    build_image_dataloader,
)
from train.curriculum import STAGE1, apply_stage

torch.set_float32_matmul_precision("high")  # Tensor Cores on A100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_ROOT = Path("data/ready/images")
CKPT_DIR = Path("checkpoints")


# ─── Resolution-grouped batching ──────────────────────────────────────────────

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
            groups[x.shape[1:]].append(x)
    return groups


# ─── Training ─────────────────────────────────────────────────────────────────

def train_step(model, loss_fn, batch, device, use_amp=True):
    """Process one NaViT-packed batch with resolution-grouped batching."""
    groups = _group_samples_by_shape(batch)
    all_losses = []
    all_logs = defaultdict(list)

    for shape, tensors in groups.items():
        max_b = _max_batch_for_shape(shape)
        for i in range(0, len(tensors), max_b):
            chunk = tensors[i : i + max_b]
            x = torch.cat(chunk, dim=0).to(device)

            try:
                with autocast(device_type="cuda", enabled=use_amp):
                    dec_out, lat_out = model(x, "image")
                    result = loss_fn(
                        recon=dec_out.reconstruction,
                        target=x,
                        mu=lat_out.mu,
                        log_var=lat_out.log_var,
                    )
                all_losses.append(result["loss"])
                for k, v in result["logs"].items():
                    all_logs[k].append(v)
            except Exception as e:
                log.warning("Skip shape=%s n=%d: %s", shape, len(chunk), e)

    if not all_losses:
        return torch.tensor(0.0, device=device, requires_grad=True), {}

    loss = torch.stack(all_losses).mean()
    logs = {k: torch.stack(v).mean().item() for k, v in all_logs.items()}
    logs["n_samples"] = sum(len(t) for t in groups.values())
    return loss, logs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Model ──
    cfg = MAVTConfig()
    model = MAVTokenizer(cfg).to(device)
    apply_stage(model, STAGE1)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %.1fM params (%.1fM trainable)", n_params / 1e6, n_train / 1e6)

    # ── Loss ──
    loss_fn = MAVTLoss(LossWeights(recon=1.0, kl=1e-4))

    # ── Data ──
    resolutions = STAGE1.image_resolutions  # [64, 128, 256, 512]
    if args.synthetic:
        log.info("Using SYNTHETIC data for smoke test")
        from torch.utils.data import DataLoader
        train_ds = SyntheticDataset(500, resolutions, "image", 16)
        val_ds = SyntheticDataset(50, resolutions, "image", 16)
        collator = NaViTCollator(max_seq_len=4096)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=collator, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=8, shuffle=False,
                            num_workers=1, collate_fn=collator)
    else:
        data_path = str(args.data_dir)
        log.info("Loading images from %s", data_path)
        train_dl, val_dl = build_image_dataloader(
            root=data_path,
            resolutions=resolutions,
            batch_size=args.batch_size,
            max_seq_len=4096,
            patch_size=16,
            num_workers=args.workers,
        )
    log.info("Train batches per epoch: %d", len(train_dl))

    # ── Optimizer ──
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    warmup = args.warmup_steps
    total = args.steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Resume ──
    start_step = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        log.info("Resumed from %s (step %d)", args.checkpoint, start_step)

    # ── AMP scaler ──
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    log.info("AMP: %s", "enabled (fp16)" if use_amp else "disabled")

    # ── Training loop ──
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    best_val_loss = float("inf")
    step = start_step
    t0 = time.time()

    log.info("=" * 60)
    log.info("Stage 1 Training: %d steps, lr=%.1e, batch=%d", total, args.lr, args.batch_size)
    log.info("=" * 60)

    pbar = tqdm(total=total, initial=start_step, desc="Stage 1", unit="step",
                dynamic_ncols=True)

    while step < total:
        for batch in train_dl:
            if step >= total:
                break

            # Forward + backward
            loss, logs = train_step(model, loss_fn, batch, device, use_amp=use_amp)

            optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad and loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

            # ── Progress bar ──
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{logs.get('recon_loss', 0):.4f}",
                kl=f"{logs.get('kl_loss', 0):.4f}",
                lr=f"{lr:.2e}",
                n=logs.get("n_samples", 0),
            )
            pbar.update(1)

            # ── Log ──
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                log.info(
                    "step %6d | loss %.4f | recon %.4f | kl %.4f | lr %.2e | %.1f s/step | samples %d",
                    step,
                    loss.item(),
                    logs.get("recon_loss", 0),
                    logs.get("kl_loss", 0),
                    lr,
                    elapsed / max(step - start_step, 1),
                    logs.get("n_samples", 0),
                )

            # ── Validate ──
            if step > 0 and step % args.val_every == 0:
                val_loss = evaluate(model, loss_fn, val_dl, device, max_batches=20, use_amp=use_amp)
                log.info(">>> VALIDATION step %d: loss=%.4f", step, val_loss)
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_ckpt(model, optimizer, step, CKPT_DIR / "stage1_best.pt")
                    log.info(">>> New best model saved (loss=%.4f)", val_loss)

            # ── Checkpoint ──
            if step > 0 and step % args.save_every == 0:
                save_ckpt(model, optimizer, step, CKPT_DIR / f"stage1_step{step}.pt")

            step += 1

    pbar.close()
    # Save final
    save_ckpt(model, optimizer, step, CKPT_DIR / "stage1_final.pt")
    elapsed = time.time() - t0
    log.info("Training done: %d steps in %.1f min", total, elapsed / 60)

    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loss_fn, val_dl, device, max_batches=50, use_amp=True):
    """Evaluate model on validation set. Returns average loss."""
    model.eval()
    total_loss = 0
    count = 0

    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        loss, logs = train_step(model, loss_fn, batch, device, use_amp=use_amp)
        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def eval_reconstruction(model, device, data_dir, output_dir="eval_outputs", n_samples=16):
    """Evaluate reconstruction quality and save visual results.

    Saves:
    - Side-by-side original vs reconstruction images
    - Per-sample metrics (L1, PSNR, SSIM)
    - Summary statistics
    """
    import json
    from PIL import Image
    from torchvision.transforms import functional as TF

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images at various resolutions
    resolutions = [64, 128, 256, 512]
    img_exts = {".jpg", ".jpeg", ".png"}

    data_path = Path(data_dir)
    all_imgs = [p for p in data_path.iterdir() if p.suffix.lower() in img_exts]
    if not all_imgs:
        log.error("No images found in %s", data_dir)
        return

    import random
    random.seed(42)
    selected = random.sample(all_imgs, min(n_samples, len(all_imgs)))

    results = []
    for idx, img_path in enumerate(selected):
        for res in resolutions:
            try:
                # Load and preprocess
                img = Image.open(img_path).convert("RGB")
                img = img.resize((res, res), Image.BICUBIC)
                x = TF.to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W) in [0,1]
                x = x * 2 - 1  # normalize to [-1, 1]

                # Forward
                dec_out, lat_out = model(x, "image")
                recon = dec_out.reconstruction

                # Metrics
                l1 = F.l1_loss(recon, x).item()
                mse = F.mse_loss(recon, x).item()
                psnr = 10 * math.log10(4.0 / max(mse, 1e-10))  # range [-1,1] so max diff=2, max diff^2=4

                results.append({
                    "image": img_path.name,
                    "resolution": res,
                    "l1": round(l1, 4),
                    "mse": round(mse, 6),
                    "psnr": round(psnr, 2),
                    "latent_shape": list(lat_out.z.shape),
                })

                # Save comparison image (original | reconstruction)
                x_vis = (x.squeeze(0).clamp(-1, 1) + 1) / 2  # back to [0,1]
                r_vis = (recon.squeeze(0).clamp(-1, 1) + 1) / 2

                comparison = torch.cat([x_vis, r_vis], dim=2)  # side by side
                comp_img = TF.to_pil_image(comparison.cpu())
                comp_img.save(output_dir / f"sample{idx:02d}_{res}px.png")

            except Exception as e:
                log.warning("Failed eval for %s at %dpx: %s", img_path.name, res, e)

    # Summary
    if results:
        avg_l1 = sum(r["l1"] for r in results) / len(results)
        avg_psnr = sum(r["psnr"] for r in results) / len(results)

        summary = {
            "n_samples": len(results),
            "avg_l1": round(avg_l1, 4),
            "avg_psnr": round(avg_psnr, 2),
            "per_resolution": {},
        }
        for res in resolutions:
            res_results = [r for r in results if r["resolution"] == res]
            if res_results:
                summary["per_resolution"][str(res)] = {
                    "count": len(res_results),
                    "avg_l1": round(sum(r["l1"] for r in res_results) / len(res_results), 4),
                    "avg_psnr": round(sum(r["psnr"] for r in res_results) / len(res_results), 2),
                }

        # Save results
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump({"summary": summary, "per_sample": results}, f, indent=2)

        log.info("=" * 60)
        log.info("EVALUATION RESULTS")
        log.info("=" * 60)
        log.info("  Samples evaluated: %d", len(results))
        log.info("  Average L1 loss:   %.4f", avg_l1)
        log.info("  Average PSNR:      %.2f dB", avg_psnr)
        for res, stats in summary["per_resolution"].items():
            log.info("  %spx: L1=%.4f  PSNR=%.2f dB (%d samples)",
                     res, stats["avg_l1"], stats["avg_psnr"], stats["count"])
        log.info("  Results saved to: %s", output_dir)
        log.info("=" * 60)

    return results


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_ckpt(model, optimizer, step, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "stage": 1,
    }, path)
    log.info("Checkpoint saved: %s", path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAVT Stage 1: Image Foundation")
    # Data
    parser.add_argument("--data-dir", type=str, default=str(DATA_ROOT))
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    # Training
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=32, help="Samples per NaViT pack step")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    # Logging
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=10000)
    # Eval
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--eval-output", type=str, default="eval_outputs/stage1")
    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        if not args.checkpoint:
            # Try to find best checkpoint
            best = CKPT_DIR / "stage1_best.pt"
            final = CKPT_DIR / "stage1_final.pt"
            args.checkpoint = str(best if best.exists() else final)

        log.info("Eval-only mode, loading %s", args.checkpoint)
        cfg = MAVTConfig()
        model = MAVTokenizer(cfg).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        log.info("Loaded checkpoint (step %d)", ckpt.get("step", -1))

        eval_reconstruction(model, device, args.data_dir, args.eval_output, args.eval_samples)
    else:
        model = train(args)
        log.info("\nRunning evaluation on trained model...")
        eval_reconstruction(model, device, args.data_dir, args.eval_output, args.eval_samples)


if __name__ == "__main__":
    main()
