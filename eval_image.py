#!/usr/bin/env python3
"""Evaluate MAVT image reconstruction + understanding quality.

Reconstruction metrics:
  PSNR  (per-image, higher is better)
  SSIM  (per-image, higher is better)
  LPIPS (per-image, AlexNet, lower is better)
  FID   (Fréchet Inception Distance, Inception-V3 features, lower is better)

Understanding metrics (vs frozen SigLIP2 teacher):
  cos_sim_teacher : mean cosine similarity between MAVT.semantic and teacher
                    pooler_output. This is the same signal as training's
                    `loss_sem = 1 - cos_sim`, so a value of 1.0 = perfect
                    distillation, 0.0 = random.
  linear_probe_acc: optional — needs labels (skipped in default eval set)

Pipeline mirrors eval_video.py: load Lightning ckpt, pre-create cd_split
poolers found in the ckpt, then run forward over WDSImageDataset and
accumulate metrics.

Usage:
  PYTHONPATH=src .venv/bin/python eval_image.py \\
      --ckpt checkpoints/stage1/balanced/mavt-stage1-balanced-step=0015000-val/loss=0.2320.ckpt \\
      --image_shards_dir dataset/image10k/train \\
      --max_images 1024 \\
      --output eval_image.json
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio

import lpips

from mavt.training.lightning_module import MAVTLightningModule
from mavt.data.datasets import WDSImageDataset
from mavt.data.datamodule import _collate


def _to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]."""
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Lightning .ckpt path')
    ap.add_argument('--image_shards_dir', required=True,
                    help='Dir containing WDS .tar shards')
    ap.add_argument('--output', default='eval_image.json')
    ap.add_argument('--max_images', type=int, default=1024,
                    help='Cap total images evaluated (None = all)')
    ap.add_argument('--image_resolution', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--lpips_chunk', type=int, default=32,
                    help='Sub-batch size for LPIPS to control memory')
    ap.add_argument('--fid_feature', type=int, default=2048,
                    choices=[64, 192, 768, 2048],
                    help='Inception feature dim for FID (2048 = pool3 default)')
    ap.add_argument('--semantic', action=argparse.BooleanOptionalAction, default=True,
                    help='Compute cosine similarity to frozen SigLIP2 teacher')
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # --- Model: pre-create cd_split poolers from ckpt before loading -------
    print(f'[eval] loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    raw_hp = dict(ckpt.get('hyper_parameters', {}))
    state = ckpt.get('state_dict', {})

    valid = set(inspect.signature(MAVTLightningModule.__init__).parameters)
    hparams = {k: v for k, v in raw_hp.items() if k in valid}
    module = MAVTLightningModule(**hparams)

    pooler_combos = set()
    for k in state.keys():
        if k.startswith('model.cd_split._content_poolers.'):
            shape = k.split('.')[3]
            if '_' in shape and all(s.isdigit() for s in shape.split('_')):
                a, b = shape.split('_')
                pooler_combos.add((int(a), int(b)))
    for n_c, n_d in sorted(pooler_combos):
        module.model.cd_split.prepare_poolers(n_c, n_d)
    print(f'[eval] pre-created poolers for combos: {sorted(pooler_combos)}')

    missing, unexpected = module.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith('semantic_teacher.')]
    print(f'[eval] load: {len(real_missing)} missing (excl. teacher), '
          f'{len(unexpected)} unexpected')
    if real_missing:
        print(f'[eval]   missing sample: {real_missing[:5]}')
    if unexpected:
        print(f'[eval]   unexpected sample: {unexpected[:5]}')

    module.eval().to(device)

    # --- Data ---------------------------------------------------------------
    ds = WDSImageDataset(args.image_shards_dir, args.image_resolution)
    if args.max_images and args.max_images < len(ds):
        ds = Subset(ds, list(range(args.max_images)))
    print(f'[eval] {len(ds)} images in eval set')

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        collate_fn=_collate, drop_last=False,
    )

    # --- Reconstruction metrics --------------------------------------------
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    fid_metric = FrechetInceptionDistance(
        feature=args.fid_feature, normalize=True,
    ).to(device)

    # --- Understanding metric: load frozen SigLIP2 teacher ------------------
    teacher = None
    teacher_input_size = 224
    if args.semantic:
        teacher_name = hparams.get('siglip2_model_name', 'google/siglip2-base-patch16-224')
        print(f'[eval] loading semantic teacher: {teacher_name}')
        from transformers import AutoModel
        siglip = AutoModel.from_pretrained(teacher_name)
        teacher = siglip.vision_model.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        try:
            teacher_input_size = int(siglip.config.vision_config.image_size)
        except AttributeError:
            teacher_input_size = 224
        print(f'[eval] teacher input size: {teacher_input_size}')
    cos_sim_sum, cos_sim_n = 0.0, 0

    lpips_sum, lpips_n = 0.0, 0
    autocast_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    for bi, batch in enumerate(loader):
        x = batch['data'].to(device, non_blocking=True)  # (B, 3, H, W) in [-1, 1]

        with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=device.type == 'cuda'):
            out = module.model(x, 'image', decode=True)
        recon = out.reconstruction.float().clamp(-1.0, 1.0)  # (B, 3, H, W)

        rec01 = _to_unit(recon)
        tgt01 = _to_unit(x)

        psnr_metric.update(rec01, tgt01)
        ssim_metric.update(rec01, tgt01)

        # LPIPS expects [-1, 1]
        for s in range(0, recon.shape[0], args.lpips_chunk):
            with torch.no_grad():
                d = lpips_fn(recon[s:s + args.lpips_chunk],
                             x[s:s + args.lpips_chunk])
            lpips_sum += d.sum().item()
            lpips_n += d.numel()

        # FID expects uint8 OR normalized float in [0, 1] when normalize=True
        fid_metric.update(tgt01, real=True)
        fid_metric.update(rec01, real=False)

        # Understanding: cosine sim between MAVT.semantic and teacher's
        # pooler_output on the SAME input image.
        if teacher is not None:
            with torch.no_grad(), torch.amp.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=device.type == 'cuda'):
                # Resize input to teacher's expected size
                if x.shape[-1] != teacher_input_size:
                    teacher_in = torch.nn.functional.interpolate(
                        x, size=teacher_input_size, mode='bilinear', align_corners=False)
                else:
                    teacher_in = x
                t_emb = teacher(pixel_values=teacher_in).pooler_output  # (B, D)
                m_emb = out.semantic.float()                             # (B, D)
            cos = torch.nn.functional.cosine_similarity(
                m_emb.float(), t_emb.float(), dim=-1)
            cos_sim_sum += cos.sum().item()
            cos_sim_n += cos.numel()

        if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
            cos_str = f'  cos_sim={cos_sim_sum / max(1, cos_sim_n):.4f}' if cos_sim_n else ''
            print(f'[eval] {bi + 1}/{len(loader)} batches  '
                  f'PSNR={psnr_metric.compute().item():.3f}  '
                  f'SSIM={ssim_metric.compute().item():.4f}  '
                  f'LPIPS={lpips_sum / max(1, lpips_n):.4f}{cos_str}')

    fid = float(fid_metric.compute().item())

    results = {
        'ckpt': args.ckpt,
        'image_shards_dir': args.image_shards_dir,
        'n_images': len(ds),
        'image_resolution': args.image_resolution,
        'psnr': float(psnr_metric.compute().item()),
        'ssim': float(ssim_metric.compute().item()),
        'lpips_alex': lpips_sum / max(1, lpips_n),
        'fid_inception': fid,
        'cos_sim_teacher': cos_sim_sum / cos_sim_n if cos_sim_n else None,
        'fid_feature_dim': args.fid_feature,
    }
    print(json.dumps(results, indent=2))
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'[eval] wrote {args.output}')


if __name__ == '__main__':
    main()
