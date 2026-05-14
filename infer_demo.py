#!/usr/bin/env python3
"""Quick demo inference: 1 image + 1 video → save GT vs recon visualizations,
plus cosine similarity to SigLIP2 teacher (understanding).

Usage:
  PYTHONPATH=src .venv/bin/python infer_demo.py \\
      --ckpt checkpoints/stage1_3/balanced/.../loss=0.1962.ckpt \\
      --image_shards_dir dataset/image10k/train \\
      --video_shards_dir dataset/dataset_10m \\
      --outdir results/infer_demo
"""
from __future__ import annotations
import argparse, inspect, json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.utils import make_grid

from mavt.training.lightning_module import MAVTLightningModule
from mavt.data.datasets import WDSImageDataset, ShardVideoDataset


def to_pil(t: torch.Tensor) -> Image.Image:
    """t: (3,H,W) in [-1,1] → PIL (H,W,3)."""
    t = ((t.clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(t)


def video_to_strip(clip: torch.Tensor, n_frames: int = 8) -> Image.Image:
    """clip: (3,T,H,W) in [-1,1] → strip of n_frames horizontally."""
    T = clip.shape[1]
    idx = torch.linspace(0, T - 1, n_frames).long()
    frames = clip[:, idx].permute(1, 0, 2, 3)  # (n, 3, H, W)
    grid = make_grid(((frames.clamp(-1, 1) + 1) * 0.5),
                     nrow=n_frames, padding=2, pad_value=1.0)
    arr = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def load_module(ckpt_path: str, device):
    print(f'[infer] loading {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    raw_hp = dict(ckpt.get('hyper_parameters', {}))
    state = ckpt.get('state_dict', {})

    valid = set(inspect.signature(MAVTLightningModule.__init__).parameters)
    hparams = {k: v for k, v in raw_hp.items() if k in valid}
    module = MAVTLightningModule(**hparams)

    # pre-create poolers found in ckpt
    combos = set()
    for k in state.keys():
        if k.startswith('model.cd_split._content_poolers.'):
            shape = k.split('.')[3]
            if '_' in shape and all(s.isdigit() for s in shape.split('_')):
                a, b = shape.split('_')
                combos.add((int(a), int(b)))
    for n_c, n_d in sorted(combos):
        module.model.cd_split.prepare_poolers(n_c, n_d)
    print(f'[infer] pre-created poolers: {sorted(combos)}')

    missing, unexpected = module.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith('semantic_teacher.')]
    print(f'[infer] load: {len(real_missing)} missing (excl teacher), {len(unexpected)} unexpected')

    module.eval().to(device)
    return module, hparams


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image_shards_dir', required=True)
    ap.add_argument('--video_shards_dir', required=True)
    ap.add_argument('--outdir', default='results/infer_demo')
    ap.add_argument('--image_idx', type=int, default=0)
    ap.add_argument('--video_idx', type=int, default=0)
    ap.add_argument('--video_max_shards', type=int, default=2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    module, hp = load_module(args.ckpt, device)
    autocast = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                   enabled=device.type == 'cuda')

    # --- Load teacher for understanding metric ---
    teacher_name = hp.get('siglip2_model_name', 'google/siglip2-base-patch16-224')
    print(f'[infer] loading teacher: {teacher_name}')
    from transformers import AutoModel
    siglip = AutoModel.from_pretrained(teacher_name)
    teacher = siglip.vision_model.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_size = int(siglip.config.vision_config.image_size)

    results = {'ckpt': args.ckpt}

    # ============= IMAGE =============
    print('[infer] ===== image =====')
    ds_img = WDSImageDataset(args.image_shards_dir, 256)
    sample = ds_img[args.image_idx]
    x = sample['data'].unsqueeze(0).to(device)  # (1, 3, 256, 256)
    print(f'  caption: {sample.get("caption", "")[:80]}')

    with autocast:
        out = module.model(x, 'image', decode=True)
    recon = out.reconstruction.float().clamp(-1, 1)

    to_pil(x[0]).save(outdir / 'image_input.png')
    to_pil(recon[0]).save(outdir / 'image_recon.png')

    # Side-by-side
    pair = torch.cat([x[0], recon[0]], dim=2)  # (3, H, 2W)
    to_pil(pair).save(outdir / 'image_side_by_side.png')

    # Understanding: cos sim teacher vs MAVT.semantic
    teacher_in = F.interpolate(x, size=teacher_size, mode='bilinear', align_corners=False)
    with autocast:
        t_emb = teacher(pixel_values=teacher_in).pooler_output.float()
    cos_img = F.cosine_similarity(out.semantic.float(), t_emb, dim=-1).item()

    # Pixel metrics on this single image
    rec01 = (recon.clamp(-1, 1) + 1) * 0.5
    tgt01 = (x.clamp(-1, 1) + 1) * 0.5
    mse = F.mse_loss(rec01, tgt01).item()
    psnr = -10 * np.log10(mse + 1e-12)

    results['image'] = {
        'shape': list(x.shape),
        'caption': sample.get('caption', ''),
        'cos_sim_teacher': cos_img,
        'recon_psnr_single': psnr,
        'recon_l1_single': F.l1_loss(rec01, tgt01).item(),
        'files': {
            'input': str(outdir / 'image_input.png'),
            'recon': str(outdir / 'image_recon.png'),
            'side_by_side': str(outdir / 'image_side_by_side.png'),
        },
    }
    print(f'  cos_sim={cos_img:.4f}, single PSNR={psnr:.2f}, L1={results["image"]["recon_l1_single"]:.4f}')

    # ============= VIDEO (caveat: stage1 ckpt has random video poolers) =============
    print('[infer] ===== video (caveat: random video poolers if stage1 ckpt) =====')
    ds_vid = ShardVideoDataset(args.video_shards_dir, n_frames=16, resolution=256,
                                max_shards=args.video_max_shards)
    sample = ds_vid[args.video_idx]
    x = sample['data'].unsqueeze(0).to(device)  # (1, 3, T, H, W)
    print(f'  caption: {sample.get("caption", "")[:80]}, shape: {tuple(x.shape)}')

    with autocast:
        out_v = module.model(x, 'video', decode=True)
    recon_v = out_v.reconstruction.float().clamp(-1, 1)  # (1, 3, Tp, H, W)
    print(f'  recon shape: {tuple(recon_v.shape)}')

    # Subsample target to match Tp
    t_patch = int(hp.get('t_patch', 2))
    tgt_v = x[:, :, ::t_patch]  # (1, 3, Tp, H, W)

    video_to_strip(x[0], n_frames=8).save(outdir / 'video_input_strip.png')
    video_to_strip(recon_v[0], n_frames=min(8, recon_v.shape[2])).save(outdir / 'video_recon_strip.png')
    video_to_strip(tgt_v[0], n_frames=min(8, tgt_v.shape[2])).save(outdir / 'video_gt_subsampled_strip.png')

    # Understanding (use middle frame as image, since teacher is image-based)
    mid_frame = x[:, :, x.shape[2] // 2]  # (1, 3, H, W)
    teacher_in = F.interpolate(mid_frame, size=teacher_size, mode='bilinear', align_corners=False)
    with autocast:
        t_emb = teacher(pixel_values=teacher_in).pooler_output.float()
    cos_vid = F.cosine_similarity(out_v.semantic.float(), t_emb, dim=-1).item()

    rec01 = (recon_v.clamp(-1, 1) + 1) * 0.5
    tgt01 = (tgt_v.clamp(-1, 1) + 1) * 0.5
    mse = F.mse_loss(rec01, tgt01).item()
    psnr_v = -10 * np.log10(mse + 1e-12)

    results['video'] = {
        'input_shape': list(x.shape),
        'recon_shape': list(recon_v.shape),
        'caption': sample.get('caption', ''),
        'cos_sim_teacher_midframe': cos_vid,
        'recon_psnr_single': psnr_v,
        'recon_l1_single': F.l1_loss(rec01, tgt01).item(),
        'caveat': 'stage1 ckpt has no trained video pooler — recon is roughly random',
        'files': {
            'input_strip': str(outdir / 'video_input_strip.png'),
            'recon_strip': str(outdir / 'video_recon_strip.png'),
            'gt_subsampled_strip': str(outdir / 'video_gt_subsampled_strip.png'),
        },
    }
    print(f'  cos_sim={cos_vid:.4f}, single PSNR={psnr_v:.2f} (caveat: random video pooler)')

    # Save JSON summary
    json_path = outdir / 'summary.json'
    json_path.write_text(json.dumps(results, indent=2))
    print(f'[infer] wrote {json_path}')


if __name__ == '__main__':
    main()
