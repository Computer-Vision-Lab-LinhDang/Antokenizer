#!/usr/bin/env python3
"""Evaluate MAVT video reconstruction quality.

Metrics:
  PSNR  (frame-level, higher is better)
  SSIM  (frame-level, higher is better)
  LPIPS (frame-level, AlexNet, lower is better)
  rFVD  (clip-level Fréchet distance over r3d_18 features, lower is better)

For frame-level metrics each video clip is split into its Tp reconstructed
frames (Tp = T // t_patch) and compared against the corresponding GT frames
(`x[:, :, ::t_patch]`).

For rFVD we extract per-clip features with a Kinetics-400-pretrained
torchvision r3d_18 (penultimate 512-d embedding), then compute the Fréchet
distance between the two Gaussian fits (GT vs reconstruction). This is the
standard Fréchet distance, applied to a video feature extractor — equivalent
to FVD in spirit. Note: canonical FVD uses I3D-Kinetics (TF port); r3d_18 is a
reproducible, dependency-light substitute. Numbers are not directly
comparable to I3D-FVD published results.

Usage:
  PYTHONPATH=src .venv/bin/python eval_video.py \\
      --ckpt checkpoints/stage1/balanced/.../loss=0.3686.ckpt \\
      --video_shards_dir dataset/dataset_10m \\
      --max_videos 512 --max_shards 4 \\
      --output eval_video.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.models.video import r3d_18, R3D_18_Weights

import lpips

from mavt.training.lightning_module import MAVTLightningModule
from mavt.data.datasets import ShardVideoDataset
from mavt.data.datamodule import _collate


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]."""
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def _flatten_clip_to_frames(clip: torch.Tensor) -> torch.Tensor:
    """(B, 3, T, H, W) → (B*T, 3, H, W)."""
    B, C, T, H, W = clip.shape
    return clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray,
                      eps: float = 1e-6) -> float:
    """Symmetric Fréchet distance (a.k.a. Wasserstein-2 on Gaussians)."""
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset),
                                        disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2)
                 - 2.0 * np.trace(covmean))


# --------------------------------------------------------------------------- #
#  Video feature extractor (r3d_18)                                            #
# --------------------------------------------------------------------------- #

# r3d_18 was pretrained on 16-frame 112×112 clips. We resize spatial dims to
# 112 and keep the model fully-convolutional in time so any T ≥ 4 works.
_R3D_INPUT_HW = 112


class _R3DFeatureExtractor(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1
        net = r3d_18(weights=weights)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(device)
        tfm = weights.transforms()
        # KINETICS400 stats; broadcast to (1, 3, 1, 1, 1) for video tensors.
        mean = torch.tensor(tfm.mean).view(1, 3, 1, 1, 1).to(device)
        std = torch.tensor(tfm.std).view(1, 3, 1, 1, 1).to(device)
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std', std, persistent=False)

    @torch.no_grad()
    def forward(self, clip01: torch.Tensor) -> torch.Tensor:
        """clip01: (B, 3, T, H, W) in [0, 1]. Returns (B, 512)."""
        B, C, T, H, W = clip01.shape
        if (H, W) != (_R3D_INPUT_HW, _R3D_INPUT_HW):
            x = F.interpolate(
                clip01.reshape(B * C, T, H, W).unsqueeze(0),
                size=(T, _R3D_INPUT_HW, _R3D_INPUT_HW),
                mode='trilinear', align_corners=False,
            ).squeeze(0).reshape(B, C, T, _R3D_INPUT_HW, _R3D_INPUT_HW)
        else:
            x = clip01
        x = (x - self.mean) / self.std
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Lightning .ckpt path')
    ap.add_argument('--video_shards_dir', required=True,
                    help='Root with NNNNN/<id>.mp4 sub-shards (video2dataset layout)')
    ap.add_argument('--output', default='eval_video.json')
    ap.add_argument('--max_videos', type=int, default=512,
                    help='Cap total clips evaluated (None = all)')
    ap.add_argument('--max_shards', type=int, default=4,
                    help='Cap number of sub-shards scanned (faster index build)')
    ap.add_argument('--video_frames', type=int, default=16)
    ap.add_argument('--video_resolution', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--lpips_chunk', type=int, default=16,
                    help='Sub-batch frame count for LPIPS to control memory')
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # --- Model --------------------------------------------------------------
    # Lightning's load_from_checkpoint runs __init__ then load_state_dict, but
    # cd_split poolers are created lazily (in setup('fit') during training);
    # they don't exist on the freshly-built module so the ckpt's pooler weights
    # would be silently dropped → recon would be random. Pre-create the exact
    # set of poolers found in the ckpt before loading.
    print(f'[eval] loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    raw_hp = dict(ckpt.get('hyper_parameters', {}))
    state = ckpt.get('state_dict', {})

    # Filter to keys MAVTLightningModule.__init__ actually accepts (Lightning
    # injects internals like _instantiator into hyper_parameters).
    import inspect
    valid = set(inspect.signature(MAVTLightningModule.__init__).parameters)
    hparams = {k: v for k, v in raw_hp.items() if k in valid}
    module = MAVTLightningModule(**hparams)

    # Find unique (N_c, N_d) combos from ckpt pooler keys, then pre-create.
    pooler_combos = set()
    for k in state.keys():
        if k.startswith('model.cd_split._content_poolers.'):
            shape = k.split('.')[3]                       # e.g. "512_204"
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
    t_patch = int(module.hparams.t_patch)
    print(f'[eval] model t_patch = {t_patch} → recon Tp = {args.video_frames // t_patch}')

    # --- Data ---------------------------------------------------------------
    ds = ShardVideoDataset(
        args.video_shards_dir,
        n_frames=args.video_frames,
        resolution=args.video_resolution,
        max_shards=args.max_shards,
    )
    if args.max_videos and args.max_videos < len(ds):
        ds = Subset(ds, list(range(args.max_videos)))
    print(f'[eval] {len(ds)} videos in eval set')

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        collate_fn=_collate, drop_last=False,
    )

    # --- Frame metrics (use update/compute for correct global mean) --------
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    feat_net = _R3DFeatureExtractor(device)

    feats_gt: List[torch.Tensor] = []
    feats_re: List[torch.Tensor] = []
    lpips_sum, lpips_n = 0.0, 0

    autocast_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    for bi, batch in enumerate(loader):
        x = batch['data'].to(device, non_blocking=True)  # (B, 3, T, H, W) in [-1, 1]

        with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=device.type == 'cuda'):
            out = module.model(x, 'video', decode=True)
        recon = out.reconstruction.float().clamp(-1.0, 1.0)  # (B, 3, Tp, H, W)
        target = x[:, :, ::t_patch]                           # (B, 3, Tp, H, W)

        # Frame-level: PSNR / SSIM on [0, 1], LPIPS on [-1, 1]
        rec01 = _flatten_clip_to_frames(_to_unit(recon))
        tgt01 = _flatten_clip_to_frames(_to_unit(target))
        psnr_metric.update(rec01, tgt01)
        ssim_metric.update(rec01, tgt01)

        rec_pm = _flatten_clip_to_frames(recon)
        tgt_pm = _flatten_clip_to_frames(target)
        for s in range(0, rec_pm.shape[0], args.lpips_chunk):
            with torch.no_grad():
                d = lpips_fn(rec_pm[s:s + args.lpips_chunk],
                             tgt_pm[s:s + args.lpips_chunk])
            lpips_sum += d.sum().item()
            lpips_n += d.numel()

        # rFVD features (clip-level)
        feats_gt.append(feat_net(_to_unit(target)).cpu())
        feats_re.append(feat_net(_to_unit(recon)).cpu())

        if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
            print(f'[eval] {bi + 1}/{len(loader)} batches  '
                  f'PSNR={psnr_metric.compute().item():.3f}  '
                  f'SSIM={ssim_metric.compute().item():.4f}  '
                  f'LPIPS={lpips_sum / max(1, lpips_n):.4f}')

    # --- rFVD ---------------------------------------------------------------
    fg = torch.cat(feats_gt).numpy().astype(np.float64)
    fr = torch.cat(feats_re).numpy().astype(np.float64)
    if fg.shape[0] < 2:
        rfvd: float = float('nan')
        print(f'[eval] WARNING: only {fg.shape[0]} clips — rFVD undefined')
    else:
        mu_g, sig_g = fg.mean(axis=0), np.cov(fg, rowvar=False)
        mu_r, sig_r = fr.mean(axis=0), np.cov(fr, rowvar=False)
        rfvd = _frechet_distance(mu_g, sig_g, mu_r, sig_r)

    results = {
        'ckpt': args.ckpt,
        'video_shards_dir': args.video_shards_dir,
        'n_videos': int(fg.shape[0]),
        'video_frames': args.video_frames,
        'video_resolution': args.video_resolution,
        't_patch': t_patch,
        'psnr': float(psnr_metric.compute().item()),
        'ssim': float(ssim_metric.compute().item()),
        'lpips_alex': lpips_sum / max(1, lpips_n),
        'rfvd_r3d18': rfvd,
        'feature_extractor': 'torchvision r3d_18 KINETICS400_V1 (pre-fc 512-d)',
    }
    print(json.dumps(results, indent=2))
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'[eval] wrote {args.output}')


if __name__ == '__main__':
    main()
