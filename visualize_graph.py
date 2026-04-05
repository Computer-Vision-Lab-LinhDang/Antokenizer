#!/usr/bin/env python3
"""Visualize Antokenizer computation graph.

Usage:
    python visualize_graph.py                            # image, forward+backward
    python visualize_graph.py --forward                  # image, forward only
    python visualize_graph.py --modality video           # video, forward+backward
    python visualize_graph.py --modality 3d --forward    # 3d, forward only
"""

import sys
import argparse
import os

# Force offline mode — tránh download SigLIP2 (sẽ dùng fallback Conv2d)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--forward", action="store_true",
                    help="Forward-only graph via torchview (không có XxxBackward0)")
parser.add_argument("--modality", choices=["image", "video", "3d"], default="image",
                    help="Modality để visualize (default: image)")
args = parser.parse_args()

from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from mavt.config import MAVTConfig, PatchifyConfig, EncoderConfig, LatentConfig, DecoderConfig
from mavt.tokenizer import MAVTokenizer

# ── Config nhỏ để chạy nhanh (không cần GPU lớn) ─────────────────────────────
cfg = MAVTConfig(
    patchify=PatchifyConfig(
        patch_size_spatial=16,
        patch_size_temporal=2,
        embed_dim=128,          # nhỏ hơn default 1152
        siglip2_model="google/siglip2-so400m-patch16-384",
        freeze_siglip2=True,
    ),
    encoder=EncoderConfig(
        d_model=128,
        n_blocks_stage3=2,
        n_blocks_stage4=2,
    ),
    latent=LatentConfig(
        d_encoder=128,
        latent_dim=16,
        d_understand=64,
    ),
    decoder=DecoderConfig(
        latent_dim=16,
        d_model=64,
        n_attn_blocks=2,
        n_attn_heads=4,
        cnn_channels=(64, 32, 16, 8),
        out_channels=3,
        patch_size=16,
    ),
    content_dynamics=None,  # tắt để giảm độ phức tạp graph
)

# ── Build input tensor theo modality ──────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" / "ready_sample_100" / "images"

def _load_image_tensor(path: Path) -> torch.Tensor:
    """Load one image → (1, 3, 128, 128) normalized to [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((128, 128), Image.BILINEAR)
    x = TF.to_tensor(img).unsqueeze(0)   # (1, 3, 128, 128)
    return (x - 0.5) / 0.5

def build_input(modality: str) -> torch.Tensor:
    img_paths = sorted(DATA_DIR.glob("*.jpg")) or sorted(DATA_DIR.glob("*.png"))
    assert img_paths, f"Không tìm thấy ảnh trong {DATA_DIR}"

    if modality == "image":
        # (1, 3, 128, 128)
        print(f"Đang dùng ảnh: {img_paths[0].name}")
        return _load_image_tensor(img_paths[0])

    if modality == "video":
        # (1, 3, T, 128, 128): stack T frames along new dim 1, add batch dim
        T = 4
        frame_tensors = [_load_image_tensor(img_paths[i % len(img_paths)]).squeeze(0)
                         for i in range(T)]   # T × (3, 128, 128)
        x = torch.stack(frame_tensors, dim=1).unsqueeze(0)  # (1, 3, T, 128, 128)
        print(f"Video: {T} frames từ {len(img_paths)} ảnh, shape={tuple(x.shape)}")
        return x

    if modality == "3d":
        # (1, 3, 3, 128, 128): 3 planes (XY/XZ/YZ), each a 3-ch RGB image
        plane_tensors = [_load_image_tensor(img_paths[i % len(img_paths)]).squeeze(0)
                         for i in range(3)]   # 3 × (3, 128, 128)
        x = torch.stack(plane_tensors, dim=0).unsqueeze(0)  # (1, 3, 3, 128, 128)
        print(f"3D triplane: 3 planes RGB 128×128, shape={tuple(x.shape)}")
        return x

    raise ValueError(f"Unknown modality: {modality}")

x = build_input(args.modality)

# ── Khởi tạo model ─────────────────────────────────────────────────────────────
print("Khởi tạo MAVTokenizer...")
model = MAVTokenizer(cfg)
model.eval()

out_dir = Path(__file__).parent / "outputs" / "visualize"
out_dir.mkdir(parents=True, exist_ok=True)

mode_label = "forward" if args.forward else "forward_backward"
base_name = f"antokenizer_graph_{args.modality}_{mode_label}"
print(f"Tạo computation graph (modality={args.modality}, mode={mode_label})...")

if args.forward:
    # ── torchview: clean forward-only graph (không có XxxBackward0) ────────────
    from torchview import draw_graph

    graph = draw_graph(
        model,
        input_data=(x,),
        modality=args.modality,
        depth=5,
        device="cpu",
        expand_nested=True,
    )
    vg = graph.visual_graph
    vg.format = "png"
    print(f"Đã lưu: {vg.render(str(out_dir / base_name), cleanup=True)}")
    vg.format = "svg"
    print(f"Đã lưu: {vg.render(str(out_dir / base_name), cleanup=True)}")

else:
    # ── torchviz: forward + backward autograd graph ────────────────────────────
    from torchviz import make_dot

    x = x.requires_grad_(True)
    decoder_out, latent_out = model(x, modality=args.modality)
    recon = decoder_out.reconstruction
    loss_dict = model.compute_loss(x, decoder_out, latent_out)
    total_loss = loss_dict["total_loss"]

    print(f"  recon shape : {recon.shape}")
    print(f"  latent z    : {latent_out.z.shape}")
    print(f"  total_loss  : {total_loss.item():.6f}")

    params = {
        **{f"model.{n}": p for n, p in model.named_parameters() if p.requires_grad},
        "input": x,
    }
    dot = make_dot(total_loss, params=params, show_attrs=True, show_saved=True)
    dot.format = "png"
    print(f"Đã lưu: {dot.render(str(out_dir / base_name), cleanup=True)}")
    dot.format = "svg"
    print(f"Đã lưu: {dot.render(str(out_dir / base_name), cleanup=True)}")

print("\nDone! Files saved:")
print(f"  PNG: {out_dir}/{base_name}.png")
print(f"  SVG: {out_dir}/{base_name}.svg")
