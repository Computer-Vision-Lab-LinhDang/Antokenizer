#!/usr/bin/env python3
"""MAVT evaluation script.

Usage:
  python evaluate.py --ckpt checkpoints/stage1/best.ckpt \
                     --modality image \
                     --data_root /path/to/images \
                     --resolution 256
"""

import argparse
import torch
from torch.utils.data import DataLoader

from mavt.training.lightning_module import MAVTLightningModule
from mavt.data.datasets import (
    ImageFolderDataset, VideoDataset, ThreeDDataset, SyntheticMultiModalDataset
)
from mavt.evaluation.metrics import (
    compute_image_metrics, compute_video_metrics,
    compute_threed_metrics, FIDTracker
)


def _collate(batch):
    modality = batch[0]['modality']
    return {'data': torch.stack([b['data'] for b in batch]), 'modality': modality}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       required=True)
    parser.add_argument('--modality',   default='image',
                        choices=['image', 'video', 'threed'])
    parser.add_argument('--data_root',  default=None)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--n_frames',   type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load model
    print(f'Loading checkpoint: {args.ckpt}')
    module = MAVTLightningModule.load_from_checkpoint(args.ckpt, map_location='cpu')
    model = module.model.to(args.device).eval()

    # Build dataset
    modality = args.modality
    if args.data_root:
        if modality == 'image':
            ds = ImageFolderDataset(args.data_root, args.resolution)
        elif modality == 'video':
            ds = VideoDataset(args.data_root, args.n_frames, args.resolution)
        else:
            ds = ThreeDDataset(args.data_root)
    else:
        print('No data_root provided — using synthetic data.')
        ds = SyntheticMultiModalDataset(64, modality, args.resolution)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, collate_fn=_collate)

    fid_tracker = FIDTracker() if modality == 'image' else None
    all_metrics: dict = {}

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.max_batches:
                break
            x = batch['data'].to(args.device)
            out = model(x, modality, decode=True)
            pred   = out.reconstruction.float().clamp(-1, 1) * 0.5 + 0.5  # [0,1]
            target = x.float().clamp(-1, 1) * 0.5 + 0.5

            if modality == 'image':
                m = compute_image_metrics(pred, target)
                if fid_tracker:
                    fid_tracker.update_real(target)
                    fid_tracker.update_fake(pred)
            elif modality == 'video':
                m = compute_video_metrics(pred, target)
            else:
                m = compute_threed_metrics(pred, target)

            for k, v in m.items():
                all_metrics.setdefault(k, []).append(v)

            print(f'Batch {i+1}: {m}')

    print('\n=== Summary ===')
    for k, vals in all_metrics.items():
        print(f'  {k}: {sum(vals)/len(vals):.4f}')

    if fid_tracker:
        fid = fid_tracker.compute()
        if fid is not None:
            print(f'  FID: {fid:.2f}')


if __name__ == '__main__':
    main()
