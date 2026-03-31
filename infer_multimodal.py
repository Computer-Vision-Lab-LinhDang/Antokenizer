#!/usr/bin/env python3
"""Inference & visualization for MAVT multi-modal model.

Loads a checkpoint from train_multimodal_ddp.py, runs inference on
image / video / 3D inputs, and produces side-by-side visualizations
(input vs. reconstruction).

Usage:
    # Infer on a single image
    python infer_multimodal.py --checkpoint checkpoints/multimodal/last.ckpt \
        --image path/to/image.jpg

    # Infer on a video
    python infer_multimodal.py --checkpoint checkpoints/multimodal/last.ckpt \
        --video path/to/video.mp4

    # Infer on a 3D object (render directory)
    python infer_multimodal.py --checkpoint checkpoints/multimodal/last.ckpt \
        --object3d path/to/object_render_dir

    # Infer on a whole directory of images
    python infer_multimodal.py --checkpoint checkpoints/multimodal/last.ckpt \
        --image-dir datasets/datasets/ready_sample_100/images --max-samples 8

    # Smoke test with synthetic data (no checkpoint needed)
    python infer_multimodal.py --synthetic --no-checkpoint

    # Save outputs instead of showing
    python infer_multimodal.py --checkpoint checkpoints/multimodal/last.ckpt \
        --image path/to/image.jpg --save-dir outputs/
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD = [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Optional[str], device: torch.device):
    """Load the MultiModalLitModule from a Lightning checkpoint."""
    from train_multimodal_ddp import MultiModalLitModule

    if checkpoint_path:
        log.info("Loading checkpoint: %s", checkpoint_path)
        model = MultiModalLitModule.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
    else:
        log.info("No checkpoint — using randomly initialized model")
        model = MultiModalLitModule()

    model = model.to(device)
    model.eval()
    log.info("Model loaded on %s", device)
    return model


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo [-1, 1] normalization → [0, 1]."""
    return tensor * 0.5 + 0.5


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor in [0,1] → (H, W, C) uint8 numpy."""
    t = tensor.detach().cpu().clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_image(path: str, resolution: int = 256) -> torch.Tensor:
    """Load and preprocess a single image → (1, 3, H, W)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.BICUBIC)
    tensor = TF.normalize(TF.to_tensor(img), _NORM_MEAN, _NORM_STD)
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def prepare_video(path: str, resolution: int = 128, n_frames: int = 8) -> torch.Tensor:
    """Load and preprocess a video → (1, 3, T, H, W)."""
    import cv2

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {path}")

    if total < n_frames:
        indices = list(range(total)) + [total - 1] * (n_frames - total)
    else:
        indices = [int(i) for i in torch.linspace(0, total - 1, n_frames)]

    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            if frames:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros(3, resolution, resolution))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resolution, resolution))
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        t = TF.normalize(t, _NORM_MEAN, _NORM_STD)
        frames.append(t)

    cap.release()
    video = torch.stack(frames, dim=1)  # (3, T, H, W)
    return video.unsqueeze(0)  # (1, 3, T, H, W)


def prepare_3d(render_dir: str, triplane_res: int = 64) -> torch.Tensor:
    """Build triplane from rendered views → (1, 3, 3, S, S)."""
    render_dir = Path(render_dir)
    planes = []
    for view_idx in [0, 2, 4]:
        view_path = render_dir / f"view_{view_idx}.png"
        img = Image.open(view_path).convert("RGB")
        t = TF.to_tensor(img)
        t = F.interpolate(
            t.unsqueeze(0), size=(triplane_res, triplane_res),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        t = t * 2 - 1
        planes.append(t)
    triplane = torch.stack(planes, dim=0)  # (3, 3, S, S)
    return triplane.unsqueeze(0)  # (1, 3, 3, S, S)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(model, x: torch.Tensor, modality: str, device: torch.device):
    """Run model forward pass, return (input, reconstruction, latent_info)."""
    x = x.to(device)
    dec_out, lat_out = model(x, modality)
    recon = dec_out.reconstruction
    latent_info = {
        "z_shape": lat_out.z.shape,
        "mu_mean": lat_out.mu.mean().item(),
        "mu_std": lat_out.mu.std().item(),
        "logvar_mean": lat_out.log_var.mean().item(),
    }
    return x.cpu(), recon.cpu(), latent_info


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_image(input_t: torch.Tensor, recon_t: torch.Tensor,
                    latent_info: dict, save_path: Optional[str] = None):
    """Side-by-side: input vs reconstruction for a single image."""
    inp = tensor_to_numpy(denormalize(input_t[0]))
    rec = tensor_to_numpy(denormalize(recon_t[0]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(inp)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(rec)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    l1 = F.l1_loss(denormalize(recon_t), denormalize(input_t)).item()
    fig.suptitle(
        f"Image | L1={l1:.4f} | z={list(latent_info['z_shape'])} | "
        f"μ={latent_info['mu_mean']:.3f}±{latent_info['mu_std']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, save_path)


def visualize_video(input_t: torch.Tensor, recon_t: torch.Tensor,
                    latent_info: dict, save_path: Optional[str] = None):
    """Grid: top row = input frames, bottom row = reconstructed frames."""
    # input_t: (1, 3, T, H, W), recon_t: (1, 3, T', H, W)
    inp = denormalize(input_t[0])   # (3, T, H, W)
    rec = denormalize(recon_t[0])   # (3, T', H, W)

    T_in = inp.shape[1]
    T_rec = rec.shape[1]
    T_show = min(T_in, T_rec, 8)

    fig, axes = plt.subplots(2, T_show, figsize=(T_show * 2.5, 5))
    if T_show == 1:
        axes = axes.reshape(2, 1)

    for t in range(T_show):
        frame_in = tensor_to_numpy(inp[:, t])
        axes[0, t].imshow(frame_in)
        axes[0, t].set_title(f"In t={t}")
        axes[0, t].axis("off")

        t_rec = min(t, T_rec - 1)
        frame_rec = tensor_to_numpy(rec[:, t_rec])
        axes[1, t].imshow(frame_rec)
        axes[1, t].set_title(f"Rec t={t_rec}")
        axes[1, t].axis("off")

    l1 = F.l1_loss(rec[:, :T_show], inp[:, :T_show]).item()
    fig.suptitle(
        f"Video | L1={l1:.4f} | z={list(latent_info['z_shape'])}",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, save_path)


def visualize_3d(input_t: torch.Tensor, recon_t: torch.Tensor,
                 latent_info: dict, save_path: Optional[str] = None):
    """Show triplane input planes (top) vs reconstruction (bottom).

    Input: (1, 3, 3, S, S) — 3 planes each with 3 channels.
    Recon: (1, 3, H, W) — decoded as a single plane (v1 decoder).
    """
    inp = denormalize(input_t[0])  # (3, 3, S, S) — 3 planes

    # Reconstruction is a single 2D image from v1 decoder
    rec = denormalize(recon_t[0])  # (3, H, W)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    plane_names = ["XY Plane", "XZ Plane", "YZ Plane"]
    for i in range(3):
        plane_img = tensor_to_numpy(inp[i])  # (3, S, S) → (S, S, 3)
        axes[0, i].imshow(plane_img)
        axes[0, i].set_title(f"Input {plane_names[i]}")
        axes[0, i].axis("off")

    # Show reconstruction in center, blank the other two
    rec_img = tensor_to_numpy(rec)
    axes[1, 0].axis("off")
    axes[1, 1].imshow(rec_img)
    axes[1, 1].set_title("Reconstruction (1st plane)")
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    fig.suptitle(
        f"3D Triplane | z={list(latent_info['z_shape'])} | "
        f"μ={latent_info['mu_mean']:.3f}±{latent_info['mu_std']:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    _save_or_show(fig, save_path)


def visualize_batch_images(inputs: list, recons: list,
                           save_path: Optional[str] = None):
    """Grid of N images: top row inputs, bottom row reconstructions."""
    n = len(inputs)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        inp = tensor_to_numpy(denormalize(inputs[i]))
        rec = tensor_to_numpy(denormalize(recons[i]))
        axes[0, i].imshow(inp)
        axes[0, i].set_title(f"Input {i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(rec)
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis("off")

    fig.suptitle("Batch Image Reconstruction", fontsize=13)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path: Optional[str]):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved: %s", save_path)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAVT Inference & Visualization")

    # Inputs
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--video", type=str, help="Path to a single video")
    parser.add_argument("--object3d", type=str, help="Path to 3D render directory")
    parser.add_argument("--image-dir", type=str, help="Directory of images for batch inference")
    parser.add_argument("--max-samples", type=int, default=8,
                        help="Max images from --image-dir")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Use random weights (for testing the pipeline)")

    # Resolutions
    parser.add_argument("--image-res", type=int, default=256)
    parser.add_argument("--video-res", type=int, default=128)
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--triplane-res", type=int, default=64)

    # Synthetic
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on synthetic random data")

    # Output
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save visualizations to this directory (otherwise show)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda / cpu (auto-detected if omitted)")

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate input
    has_input = any([args.image, args.video, args.object3d, args.image_dir, args.synthetic])
    if not has_input:
        parser.error("Provide at least one of: --image, --video, --object3d, --image-dir, --synthetic")

    if not args.checkpoint and not args.no_checkpoint:
        parser.error("Provide --checkpoint or use --no-checkpoint for random weights")

    # Load model
    ckpt = args.checkpoint if not args.no_checkpoint else None
    model = load_model(ckpt, device)

    save_dir = args.save_dir

    # --- Image ---
    if args.image:
        log.info("Inferring image: %s", args.image)
        x = prepare_image(args.image, args.image_res)
        inp, rec, info = infer(model, x, "image", device)
        save_path = os.path.join(save_dir, "image_recon.png") if save_dir else None
        visualize_image(inp, rec, info, save_path)

    # --- Image directory ---
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        img_dir = Path(args.image_dir)
        paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
        paths = paths[:args.max_samples]
        log.info("Inferring %d images from %s", len(paths), args.image_dir)

        all_inputs, all_recons = [], []
        for p in paths:
            x = prepare_image(str(p), args.image_res)
            inp, rec, info = infer(model, x, "image", device)
            all_inputs.append(inp[0])
            all_recons.append(rec[0])
            log.info("  %s — L1=%.4f", p.name,
                     F.l1_loss(denormalize(rec), denormalize(inp)).item())

        save_path = os.path.join(save_dir, "batch_image_recon.png") if save_dir else None
        visualize_batch_images(all_inputs, all_recons, save_path)

    # --- Video ---
    if args.video:
        log.info("Inferring video: %s", args.video)
        x = prepare_video(args.video, args.video_res, args.video_frames)
        inp, rec, info = infer(model, x, "video", device)
        save_path = os.path.join(save_dir, "video_recon.png") if save_dir else None
        visualize_video(inp, rec, info, save_path)

    # --- 3D ---
    if args.object3d:
        log.info("Inferring 3D object: %s", args.object3d)
        x = prepare_3d(args.object3d, args.triplane_res)
        inp, rec, info = infer(model, x, "3d", device)
        save_path = os.path.join(save_dir, "3d_recon.png") if save_dir else None
        visualize_3d(inp, rec, info, save_path)

    # --- Synthetic ---
    if args.synthetic:
        log.info("Running synthetic inference for all modalities")

        # Image
        x = torch.randn(1, 3, args.image_res, args.image_res)
        inp, rec, info = infer(model, x, "image", device)
        save_path = os.path.join(save_dir, "synthetic_image.png") if save_dir else None
        visualize_image(inp, rec, info, save_path)

        # Video
        x = torch.randn(1, 3, args.video_frames, args.video_res, args.video_res)
        inp, rec, info = infer(model, x, "video", device)
        save_path = os.path.join(save_dir, "synthetic_video.png") if save_dir else None
        visualize_video(inp, rec, info, save_path)

        # 3D
        x = torch.randn(1, 3, 3, args.triplane_res, args.triplane_res)
        inp, rec, info = infer(model, x, "3d", device)
        save_path = os.path.join(save_dir, "synthetic_3d.png") if save_dir else None
        visualize_3d(inp, rec, info, save_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
