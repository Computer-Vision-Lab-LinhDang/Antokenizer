"""V6 training CLI (torchrun ready).

    PYTHONPATH=src python -m mavt4d.v6.train --data-root ~/data/argoverse2/sensor --split train --bf16 \
        --checkpoint-layers --ckpt-dir runs/v6
    PYTHONPATH=src torchrun --nproc_per_node 8 -m mavt4d.v6.train ...
    PYTHONPATH=src python -m mavt4d.v6.train --dry-run --max-steps 2      # synthetic batch, no dataset

Gradients are all-reduced by hand once per optimizer step, after the truncated-BPTT chunks of all --accum
micro-batches (each chunk is backpropagated right away), so unused modules and multiple backward passes per step
need no DDP special cases.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import signal
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .config import V6Config
from .contracts import Observation
from .losses import DEFAULT_WEIGHTS, sequence_loss
from .model import V6Model, drop_retired

MIN_BOX_POINTS = 5        # box "seen": >= 5 LiDAR points inside
BUCKET_ELEMENTS = 1 << 24


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Train MAVT v6 on Argoverse 2 Sensor clips.')
    p.add_argument('--data-root', default=None, help='e.g. ~/data/argoverse2/sensor, a pack_av2.py or '
                   'pack_omniworld.py output, or several of them comma-separated (clips concatenated)')
    p.add_argument('--log-fraction', type=float, default=1.0,
                   help='train on this share of the logs (nested: a smaller share is a subset of a larger one)')
    p.add_argument('--split', default='train')
    p.add_argument('--overfit-clips', type=int, default=0,
                   help='overfit check: train only on the N clips evaluate.py --clips N scores (same --split, '
                        '--log-fraction, --steps, --step-dt); 0: all clips')
    p.add_argument('--steps', type=int, default=6, help='time steps per clip')
    p.add_argument('--step-dt', type=float, default=.5, help='seconds between clip steps')
    p.add_argument('--batch-size', type=int, default=2, help='clips per process and micro-batch')
    p.add_argument('--accum', type=int, default=1, help='micro-batches per optimizer step (gradient accumulation)')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--schedule-from', type=int, default=0,
                   help='step where warmup and cosine start (finetune: the resumed step)')
    p.add_argument('--backbone-lr-mult', type=float, default=1.,
                   help='lr factor for the pretrained encoder (patch embed, blocks, norm, prefix tokens)')
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--max-steps', type=int, default=100_000)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--keep', type=int, default=3, help='numbered checkpoints kept besides last.pt')
    p.add_argument('--time-limit', type=float, default=0., help='minutes of wall time; stop and save last.pt '
                   'while the slowest step and save so far still fit (0: no limit). SIGTERM also stops and saves')
    p.add_argument('--ckpt-dir', default='runs/v6')
    p.add_argument('--resume', default=None, help="checkpoint path, or 'auto' for <ckpt-dir>/last.pt if present")
    p.add_argument('--init', default=None, help='checkpoint whose weights start a new run (step 0, fresh optimizer); '
                   'modules it lacks keep their init. Ignored when --resume finds a checkpoint')
    p.add_argument('--backbone', default='dinov3_vitl16',
                   choices=('dinov3_vitl16', 'dinov3_vits16', 'siglip2_so400m16', 'siglip2_b16'),
                   help='encoder layout (sets encoder width/heads/layers/registers)')
    p.add_argument('--backbone-weights', '--dinov3', dest='backbone_weights', default=None,
                   help='pretrained encoder weights: HF repo id, local dir or file '
                        '(server: ~/mavt/weights/dinov3-vitl16, dinov3-vits16 or siglip2-*-patch16-256)')
    p.add_argument('--dense-teacher-weights', default=None,
                   help='weights of the frozen dense teacher (with --config dense_dim > 0): HF repo id, local dir or '
                        "file; default --backbone-weights when the teacher is the encoder's backbone, else its HF repo")
    p.add_argument('--bf16', action='store_true', help='bf16 autocast (Kalman/geometry stay fp32 inside modules)')
    p.add_argument('--checkpoint-layers', action='store_true', help='activation checkpointing of encoder/trunk blocks')
    p.add_argument('--cameras', default='all', help="'all', 'random' or a camera count")
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rays-per-camera', type=int, default=256)
    p.add_argument('--bptt', type=int, default=3, help='detach the state every N steps')
    p.add_argument('--match-gate', type=float, default=4., help='slot-box match radius (m)')
    p.add_argument('--weights', default='{}', help='JSON loss-weight overrides, e.g. \'{"pred_rgb": 2}\'')
    p.add_argument('--config', default='{}', help='JSON V6Config overrides, e.g. \'{"slots": 24}\'')
    p.add_argument('--device', default=None, help='default: cuda:LOCAL_RANK if available, else cpu')
    p.add_argument('--dry-run', action='store_true', help='synthetic random batches, no dataset')
    p.add_argument('--no-box-teacher', dest='box_teacher', action='store_false',
                   help='no GT-box routing of evidence to free slots (default: on while training)')
    p.add_argument('--teacher-p', type=float, default=1.,
                   help='share of clips the box teacher routes, reached after --teacher-anneal-steps')
    p.add_argument('--teacher-anneal-steps', type=int, default=0,
                   help='steps from --schedule-from over which the teacher share falls from 1 to --teacher-p')
    p.add_argument('--no-heldout', dest='heldout', action='store_false',
                   help='feed every camera (default: hold one camera per clip out as a render target)')
    args = p.parse_args(argv)
    if not args.dry_run and not args.data_root:
        p.error('--data-root is required unless --dry-run')
    if args.accum < 1:
        p.error('--accum must be >= 1')
    c = str(args.cameras)
    args.cameras = int(c) if c.isdigit() else c
    return args


# ---------------------------------------------------------------- distributed

def setup_distributed(device_arg=None):
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():  # ROCm builds report as cuda
        device = torch.device('cuda', local)
    else:
        device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        # nccl maps to RCCL on ROCm
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=30))
    return rank, world, device


def broadcast_model(model, world):
    if world > 1:
        for t in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(t.data, 0)


def allreduce_grads(params, world):
    """Mean gradients over ranks; unused parameters contribute zeros."""
    if world == 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    bucket, size = [], 0
    for p in params + [None]:
        if p is not None:
            bucket.append(p.grad)
            size += p.numel()
        if bucket and (p is None or size >= BUCKET_ELEMENTS):
            flat = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat)
            flat.div_(world)
            for g, f in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                g.copy_(f)
            bucket, size = [], 0


def average_logs(logs):
    """Per-key mean over log dicts; a key counts where present."""
    if len(logs) == 1:
        return logs[0]
    keys = set().union(*logs)
    return {k: sum(g[k] for g in logs if k in g) / sum(k in g for g in logs) for k in sorted(keys)}


def mean_logs(log, world):
    if world == 1:
        return log
    gathered = [None] * world
    dist.all_gather_object(gathered, log)
    return average_logs(gathered)


# ---------------------------------------------------------------- data

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def observation(batch, s):
    images = batch['images'][:, s]
    if not images.is_floating_point():
        images = images.float().div_(255)
    return Observation(images=images, tube_dt=batch['tube_dt'][:, s], intrinsics=batch['intrinsics'][:, s],
                       cam_to_rig=batch['cam_to_rig'][:, s], camera_key=batch['camera_key'][:, s],
                       camera_valid=batch['camera_valid'][:, s], depth=batch['depth'][:, s],
                       depth_valid=batch['depth_valid'][:, s], rig_pose=batch['rig_pose'][:, s],
                       rig_pose_valid=batch['rig_pose_valid'][:, s],
                       has_metric_depth=batch['has_metric_depth'][:, s], t=batch['t'][:, s])


def pick_heldout(camera_valid, generator=None):
    """One camera per clip kept out of the input as a render target: [B,T,V] valid -> [B,V] mask.
    Only clips with >= 2 valid cameras at the first step give one up."""
    ok = camera_valid[:, 0].bool()
    score = torch.rand(ok.shape, generator=generator).to(ok.device).masked_fill(~ok, -1.)
    held = torch.zeros_like(ok)
    held.scatter_(1, score.argmax(1, keepdim=True), True)
    return held & ok & (ok.sum(1, keepdim=True) >= 2)


def batch_to_sequence(batch, device, min_box_points=MIN_BOX_POINTS, heldout=False, generator=None):
    """Collated clip batch (data_av2.collate_clips layout, step dim 1) -> list[Observation], list[targets].
    heldout: one camera per clip is removed from the input at every step and rendered as a target."""
    batch = to_device(batch, device)
    steps, targets = [], []
    boxes = batch.get('boxes')
    # a clip with no box at any step carries no box labels (OmniWorld): no spawn or opacity negatives from it
    labelled = None if boxes is None else boxes['mask'].bool().flatten(1).any(1)
    held = pick_heldout(batch['camera_valid'], generator) if heldout else None
    for s in range(batch['t'].shape[1]):
        obs = observation(batch, s)
        tg = {'target_depth': batch['target_depth'][:, s], 'target_valid': batch['target_valid'][:, s]}
        if held is not None and bool(held.any()):
            cam_ok = obs.camera_valid.bool()
            hv = (held & cam_ok)[..., None, None]
            tg['heldout'] = {'images': obs.images[:, :, -1], 'intrinsics': obs.intrinsics, 'cam_to_rig': obs.cam_to_rig,
                             'camera_valid': held & cam_ok,
                             'depth': torch.where(obs.depth_valid.bool(), obs.depth, tg['target_depth']),
                             'depth_valid': (obs.depth_valid.bool() | tg['target_valid'].bool()) & hv}
            obs = dataclasses.replace(obs, camera_valid=cam_ok & ~held)
        steps.append(obs)
        if boxes is not None:
            if 'track' in boxes:
                tg['box_velocity'], tg['box_velocity_valid'] = box_velocity(boxes, batch['t'], s)
            tg.update(box_center=boxes['center'][:, s], box_size=boxes['size'][:, s], box_quat=boxes['quaternion'][:, s],
                      box_class=boxes['category'][:, s], box_labelled=labelled,
                      box_valid=boxes['mask'][:, s].bool() & (boxes['num_interior_pts'][:, s] >= min_box_points))
        targets.append(tg)
    return steps, targets


def box_velocity(boxes, t, s):
    """World velocity [B,M,3] of the boxes at step s from the same track one step back (forward at s=0)."""
    o = s - 1 if s > 0 else s + 1
    if o >= t.shape[1]:
        m = boxes['center'].shape[2]
        return boxes['center'].new_zeros(t.shape[0], m, 3), torch.zeros(t.shape[0], m, dtype=torch.bool, device=t.device)
    track, ok = boxes['track'].long(), boxes['mask'].bool() & (boxes['track'] >= 0)
    same = (track[:, s, :, None] == track[:, o, None, :]) & ok[:, s, :, None] & ok[:, o, None, :]   # [B,M,M]
    other = same.float() @ boxes['center'][:, o].float()
    dt = (t[:, s] - t[:, o]).float()[:, None, None]
    return (boxes['center'][:, s].float() - other) / dt, same.any(-1)


def clip_seeds(batch, base_seed, step, rank, offset=0):
    """offset: clips of this rank's earlier micro-batches in the step (synthetic clips are seeded by position)."""
    logs = batch.get('log_id')
    ts = batch.get('timestamp_ns')
    n = batch['t'].shape[0]
    if logs is None or ts is None:
        return [f'{base_seed}:{step}:{rank}:{offset + i}' for i in range(n)]
    return [f'{base_seed}:{step}:{logs[i]}:{int(ts[i, 0])}' for i in range(n)]


def _ego_from_cam():
    # camera +z forward, +x right, +y down -> ego +x forward, +y left, +z up
    return torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def synthetic_batch(cfg, batch_size, steps, cameras, step_dt=.5, generator=None, max_boxes=8):
    """Random batch in the collate_clips layout (for --dry-run and tests)."""
    g = generator
    b, t, f, h, w = batch_size, steps, cfg.tubelet, cfg.image_height, cfg.image_width
    v = min(cfg.max_cameras, cameras if isinstance(cameras, int) else cfg.max_cameras)
    images = torch.rand(b, t, v, f, 3, h, w, generator=g)
    k = torch.tensor([[.8 * w, 0., w / 2 - .5], [0., .8 * w, h / 2 - .5], [0., 0., 1.]])
    c2r = torch.eye(4).repeat(v, 1, 1)
    for i in range(v):
        yaw = 2 * math.pi * i / v
        rz = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
        c2r[i, :3, :3] = rz @ _ego_from_cam()
        c2r[i, :3, 3] = torch.tensor([1.5, 0., 1.6])
    camera_valid = torch.ones(b, t, v, dtype=torch.bool)
    if b > 1 and v > 1:
        camera_valid[-1, :, -1] = False  # one dead camera row
    z = 2 + 58 * torch.rand(b, t, v, h, w, generator=g)
    u = torch.rand(b, t, v, h, w, generator=g)
    depth_valid, target_valid = u < .05, (u >= .05) & (u < .07)
    time_s = torch.arange(t, dtype=torch.float64)[None].repeat(b, 1) * step_dt
    rig = torch.eye(4).repeat(b, t, 1, 1)
    rig[..., 0, 3] = 5. * time_s.float()  # ego drives forward at 5 m/s
    center = torch.rand(b, max_boxes, 3, generator=g) * torch.tensor([30., 20., 0.]) + torch.tensor([5., -10., .8])
    velocity = torch.randn(b, max_boxes, 3, generator=g) * torch.tensor([3., 1., 0.])
    boxes = dict(center=center[:, None] + velocity[:, None] * time_s.float()[..., None, None],
                 size=torch.tensor([4.5, 1.9, 1.6]).expand(b, t, max_boxes, 3).clone(),
                 quaternion=torch.tensor([1., 0., 0., 0.]).expand(b, t, max_boxes, 4).clone(),
                 category=torch.randint(0, 30, (b, 1, max_boxes), generator=g).expand(b, t, max_boxes).clone(),
                 track=torch.arange(max_boxes).expand(b, t, max_boxes).clone(),
                 num_interior_pts=torch.full((b, t, max_boxes), 20, dtype=torch.int64),
                 mask=torch.rand(b, 1, max_boxes, generator=g).expand(b, t, max_boxes) < .75)
    return dict(images=images, tube_dt=torch.full((b, t), .1), intrinsics=k.expand(b, t, v, 3, 3).clone(),
                cam_to_rig=c2r.expand(b, t, v, 4, 4).clone(), camera_key=torch.arange(v).expand(b, t, v).clone(),
                camera_valid=camera_valid, depth=torch.where(depth_valid, z, 0.), depth_valid=depth_valid,
                target_depth=torch.where(target_valid, z, 0.), target_valid=target_valid,
                rig_pose=rig, rig_pose_valid=torch.ones(b, t, dtype=torch.bool),
                has_metric_depth=torch.ones(b, t, dtype=torch.bool), t=time_s, boxes=boxes)


class InfiniteSampler(torch.utils.data.Sampler):
    """This rank's share of a fresh permutation per pass, without end: one DataLoader iterator for the whole
    run, so the prefetch queue never drains at an epoch boundary."""

    def __init__(self, n, rank=0, world=1, seed=0):
        self.n, self.rank, self.world, self.seed = n, rank, world, seed

    def __iter__(self):
        per, epoch = self.n // self.world, 0
        while True:
            order = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + epoch))
            yield from order[self.rank * per:(self.rank + 1) * per].tolist()
            epoch += 1


def batches(args, cfg, rank, world, device, start=0):
    """Infinite batch iterator; start (the resumed step) reseeds the order so a resume sees new clips."""
    if args.dry_run:
        g = torch.Generator().manual_seed(args.seed * 1000 + rank)
        while True:
            yield synthetic_batch(cfg, args.batch_size, args.steps, args.cameras, args.step_dt, generator=g)
    from torch.utils.data import DataLoader, Subset
    from .data_av2 import collate_clips
    from .data_omniworld import open_clips
    dataset = open_clips(args.data_root, args.split, cfg, steps=args.steps, step_dt=args.step_dt,
                         cameras=args.cameras, seed=args.seed, uint8_images=True, log_fraction=args.log_fraction)
    if rank == 0 and hasattr(dataset, 'datasets'):  # several roots: clips per source set the mixture
        print(json.dumps({'event': 'data', 'roots': args.data_root, 'clips': [len(d) for d in dataset.datasets]}),
              flush=True)
    if args.overfit_clips > 0:
        from .evaluate import pick_clips  # evaluate imports this module
        dataset = Subset(dataset, pick_clips(len(dataset), args.overfit_clips))
    if len(dataset) < world:  # a rank with no share would wait forever
        raise RuntimeError(f'{len(dataset)} clips under {args.data_root}/{args.split} for {world} ranks')
    sampler = InfiniteSampler(len(dataset), rank, world, seed=args.seed * 100003 + start)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                        collate_fn=collate_clips, drop_last=True, pin_memory=device.type == 'cuda',
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=2 if args.num_workers > 0 else None)
    yield from loader


# ---------------------------------------------------------------- optimisation

PRETRAINED = ('encoder.patch_embed.', 'encoder.blocks.', 'encoder.norm.', 'encoder.cls_token',
              'encoder.register_tokens', 'encoder.pos_embed')
NO_DECAY = ('cls_token', 'register_tokens', 'pos_embed')


def init_weights(model, path):
    """Load a checkpoint's model weights; keys the model lacks are an error, keys the checkpoint lacks keep their init."""
    src = torch.load(Path(path).expanduser(), map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(drop_retired(src.get('model', src)), strict=False)
    if unexpected:
        raise SystemExit(f'--init {path}: {len(unexpected)} keys the model does not have, e.g. {unexpected[:3]}')
    return missing, unexpected


def make_dense_teacher(cfg, args, device, emit):
    """The frozen dense teacher (None with cfg.dense_dim 0). A dry run without weights uses a random one."""
    if not cfg.dense_dim:
        return None
    from .dense import DenseTeacher, load_teacher
    source = args.dense_teacher_weights or (args.backbone_weights if cfg.dense_teacher == cfg.backbone else None)
    if source and Path(source).expanduser().exists():
        source = str(Path(source).expanduser())
    if args.dry_run and not source:
        teacher, loaded = DenseTeacher(cfg), 0
    else:
        teacher, loaded = load_teacher(cfg, source)
    emit({'event': 'dense_teacher', 'teacher': cfg.dense_teacher, 'source': source, 'loaded': loaded})
    return teacher.to(device)


def make_optimizer(model, args):
    """AdamW; the pretrained encoder trains at lr x --backbone-lr-mult; no decay on vectors and prefix tokens."""
    groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            mult = args.backbone_lr_mult if name.startswith(PRETRAINED) else 1.
            decay = p.ndim >= 2 and not name.endswith(NO_DECAY)
            groups.setdefault((mult, decay), []).append(p)
    return torch.optim.AdamW([{'params': ps, 'lr_mult': mult, 'weight_decay': args.weight_decay if decay else 0.}
                              for (mult, decay), ps in groups.items()], lr=args.lr, betas=(.9, .95))


def lr_at(step, args):
    step -= args.schedule_from
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    frac = min(1., (step - args.warmup) / max(1, args.max_steps - args.schedule_from - args.warmup))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * .5 * (1 + math.cos(math.pi * frac)))


def teacher_at(step, args):
    """Share of clips the box teacher routes: 1 until --schedule-from, then linear down to --teacher-p."""
    if not args.teacher_anneal_steps:
        return args.teacher_p
    frac = max(0., step - args.schedule_from) / args.teacher_anneal_steps
    return args.teacher_p if frac >= 1 else 1. + (args.teacher_p - 1.) * frac


def save_checkpoint(path, model, opt, step, cfg, args):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'step': step,
                'config': cfg.to_dict(), 'args': vars(args)}, tmp)
    os.replace(tmp, path)


def prune(ckpt_dir, keep):
    numbered = sorted(ckpt_dir.glob('step_*.pt'))
    for old in numbered[:max(0, len(numbered) - keep)]:
        old.unlink(missing_ok=True)


def any_rank(flag, world, device):
    if world == 1:
        return flag
    t = torch.tensor(float(flag), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def main(argv=None):
    t_start = time.time()
    args = parse_args(argv)
    rank, world, device = setup_distributed(args.device)
    main_rank = rank == 0
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed)  # same init on every rank (broadcast below anyway)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg = V6Config.with_backbone(args.backbone, **json.loads(args.config))
    weights = {**DEFAULT_WEIGHTS, **json.loads(args.weights)}
    ckpt_dir = Path(args.ckpt_dir).expanduser()
    if main_rank:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = ckpt_dir / 'log.jsonl'

    def emit(record):
        if main_rank:
            line = json.dumps(record, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
            print(line, flush=True)
            with open(log_file, 'a') as fh:
                fh.write(line + '\n')

    model = V6Model(cfg).to(device)
    if args.backbone_weights:
        from .encoder import load_backbone
        source = args.backbone_weights
        if Path(source).expanduser().exists():  # local path; anything else is an HF repo id
            source = str(Path(source).expanduser())
        loaded, missing = load_backbone(model.encoder, source)
        share = len(loaded) / max(1, len(loaded) + len(missing))
        emit({'event': 'backbone_weights', 'backbone': cfg.backbone, 'source': source, 'loaded': len(loaded),
              'missing': len(missing), 'loaded_share': share})
        if share < .5:  # a wrong path or layout must not silently train from scratch
            raise SystemExit(f'backbone weights {source!r}: only {len(loaded)} of {len(loaded) + len(missing)} '
                             f'encoder keys loaded into {cfg.backbone}')
    dense_teacher = make_dense_teacher(cfg, args, device, emit)
    if args.checkpoint_layers:
        emit({'event': 'checkpointing', **model.set_activation_checkpointing(True)})
    opt = make_optimizer(model, args)
    params = [p for p in model.parameters() if p.requires_grad]
    step = 0
    resume = args.resume
    if resume == 'auto':
        resume = ckpt_dir / 'last.pt' if (ckpt_dir / 'last.pt').exists() else None
    if resume:
        state = torch.load(Path(resume).expanduser(), map_location='cpu', weights_only=False)
        model.load_state_dict(drop_retired(state['model']))
        try:
            opt.load_state_dict(state['optimizer'])
        except ValueError as err:  # other parameter groups (e.g. new --backbone-lr-mult): fresh moments
            emit({'event': 'optimizer_reset', 'reason': str(err)})
        step = int(state['step'])
        emit({'event': 'resume', 'path': str(resume), 'step': step})
    elif args.init:
        missing, _ = init_weights(model, args.init)
        emit({'event': 'init', 'path': args.init, 'missing': len(missing),
              'new_modules': sorted({k.rsplit('.', 2)[0] for k in missing})})
    broadcast_model(model, world)
    torch.manual_seed(args.seed * 7919 + rank + step)  # per-rank ray / sample noise
    emit({'event': 'start', 'world': world, 'device': str(device), 'params': sum(p.numel() for p in params),
          'config': cfg.to_dict(), 'args': vars(args), 'weights': weights})

    autocast = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if args.bf16 else None

    def backward(loss):
        if loss.requires_grad:
            (loss / args.accum).backward()

    terminated = []
    on_term = signal.signal(signal.SIGTERM, lambda *_: terminated.append(True))
    data = batches(args, cfg, rank, world, device, start=step)
    model.train()
    skipped = 0
    t_last = time.time()
    worst_step, worst_save = 0., 60.  # a 4 GB save is assumed to take a minute until one is timed
    while step < args.max_steps:
        t0 = time.time()
        lr = lr_at(step, args)
        for group in opt.param_groups:
            group['lr'] = lr * group.get('lr_mult', 1.)
        opt.zero_grad(set_to_none=True)
        t_data, logs = 0., []
        for micro in range(args.accum):  # gradients add up; backward() scales each loss by 1 / accum
            t1 = time.time()
            batch = next(data)
            t_data += time.time() - t1
            steps, targets = batch_to_sequence(batch, device, heldout=args.heldout)
            seeds = clip_seeds(batch, args.seed, step, rank, offset=micro * args.batch_size)
            _, log = sequence_loss(model, steps, targets, seeds=seeds, weights=weights,
                                   rays_per_camera=args.rays_per_camera, bptt=args.bptt, backward=backward,
                                   autocast=autocast, gate=args.match_gate, box_teacher=args.box_teacher,
                                   teacher_p=teacher_at(step, args), dense_teacher=dense_teacher)
            logs.append(log)
        log = average_logs(logs)
        allreduce_grads(params, world)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        if math.isfinite(gnorm):
            opt.step()
        else:
            skipped += 1
            opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            log = mean_logs(log, world)
            now = time.time()
            record = {'step': step, 'lr': lr, 'grad_norm': gnorm, 'skipped': skipped, 'data_s': t_data,
                      'step_s': now - t0, 'wall_s': now - t_last, **log}
            if device.type == 'cuda':
                record['mem_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            emit(record)
            t_last = now
        worst_step = max(worst_step, time.time() - t0)
        left = args.time_limit * 60 - (time.time() - t_start)
        stop = any_rank(bool(terminated) or (args.time_limit > 0 and left < worst_step + 2 * worst_save), world,
                        device)
        if main_rank and (step % args.save_every == 0 or step == args.max_steps or stop):
            t_save = time.time()
            save_checkpoint(ckpt_dir / 'last.pt', model, opt, step, cfg, args)
            if step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f'step_{step:07d}.pt', model, opt, step, cfg, args)
                prune(ckpt_dir, args.keep)
            worst_save = max(worst_save, time.time() - t_save)
        if stop:
            emit({'event': 'stop', 'step': step, 'reason': 'sigterm' if terminated else 'time_limit'})
            break
    signal.signal(signal.SIGTERM, on_term)
    emit({'event': 'done', 'step': step, 'skipped': skipped})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return step


if __name__ == '__main__':
    main()
