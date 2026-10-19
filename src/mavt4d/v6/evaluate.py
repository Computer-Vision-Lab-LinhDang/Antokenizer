"""Evaluate a v6 checkpoint on AV2 clips with the box teacher off: slots must spawn on their own.

    PYTHONPATH=src torchrun --nproc_per_node 4 -m mavt4d.v6.evaluate --checkpoint runs/v6/step_0001750.pt \
        --data-root ~/data/argoverse2/sensor --split val --clips 512 --out runs/v6/eval-1750
    PYTHONPATH=src python -m mavt4d.v6.evaluate --dry-run --out /tmp/ev     # random weights, synthetic clips

Protocol: each clip observes its first --observe steps and is rolled out with predict() after; one camera per
clip (with >= 2 cameras) never enters the input and is rendered on a pixel grid (every --stride-th pixel) at
every step, from the camera's metadata pose (ego motion is the known action; rig_* metrics score the state's own
pose). Records per clip and step go to records_rank*.jsonl, then summary.json / summary.md by step:
rendering (PSNR, SSIM, L1) on the held-out camera and on sampled rays of the input cameras next to a
constant-colour baseline; LiDAR depth where the decoder has a surface (coverage, AbsRel, RMSE, delta<1.25)
next to a constant-median baseline, and on rays that end on a solid surface (mass >= 0.5) by range; object opacity on LiDAR points in / out of GT boxes and its AUC; slots vs
GT boxes (Hungarian on centre distance: precision, recall, centre / extent / velocity error, class accuracy,
track identity kept across steps, hold-last-position baseline on rolled-out steps); rig pose error; and the
training loss terms.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import V6Config
from .contracts import FREE
from .geometry import quat_to_matrix
from .losses import (DEFAULT_WEIGHTS, _assign, advance, points_in_boxes, render_view, rig_to_world, step_terms,
                     teacher_boxes)
from .model import V6Model
from .train import batch_to_sequence, setup_distributed, synthetic_batch

GATES = (2., 4.)            # m, slot-box match radius for precision / recall; errors use the larger one
HEADLINE = ('held_psnr', 'held_psnr_const', 'held_ssim', 'in_psnr', 'held_depth_cov', 'held_absrel',
            'held_absrel_const', 'held_delta1', 'held_depth_cov50', 'held_absrel50', 'held_absrel50_near',
            'held_absrel50_mid', 'held_absrel50_far', 'obj_opacity', 'bg_opacity', 'opacity_auc', 'slots', 'boxes',
            'precision_2m', 'recall_2m', 'precision_4m', 'recall_4m', 'center_err', 'center_err_hold',
            'vel_err', 'vel_err_zero', 'id_kept', 'class_acc', 'rig_t_err', 'rig_r_deg', 'loss_terms')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Evaluate a MAVT v6 checkpoint on AV2 clips.')
    p.add_argument('--checkpoint', default=None, help='train.py checkpoint; none: random init (baseline)')
    p.add_argument('--backbone-weights', default=None, help='with no checkpoint: pretrained encoder weights')
    p.add_argument('--data-root', default=None)
    p.add_argument('--split', default='val')
    p.add_argument('--clips', type=int, default=256, help='evenly spaced over the split (0: all)')
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--step-dt', type=float, default=.5)
    p.add_argument('--observe', type=int, default=4, help='steps fed to the model; the rest are rolled out')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--rays-per-camera', type=int, default=1024, help='sampled rays per input camera')
    p.add_argument('--stride', type=int, default=1, help='held-out camera pixel grid step')
    p.add_argument('--box-teacher', action='store_true', help='diagnostic: GT boxes route evidence as in training')
    p.add_argument('--vis', type=int, default=6, help='clips drawn to vis/ (rank 0)')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--threads', type=int, default=0, help='CPU threads (0: torch default)')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--device', default=None)
    p.add_argument('--config', default='{}', help='JSON V6Config overrides (on top of the checkpoint config)')
    p.add_argument('--dry-run', action='store_true', help='synthetic clips, no dataset')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if not args.dry_run and not args.data_root:
        p.error('--data-root is required unless --dry-run')
    return args


def load_model(args, device):
    if args.checkpoint:
        saved = torch.load(Path(args.checkpoint).expanduser(), map_location='cpu', weights_only=False)
        model = V6Model(V6Config(**{**saved['config'], **json.loads(args.config)}))  # e.g. spawn_threshold
        model.load_state_dict(saved['model'])
        return model.to(device).eval(), int(saved.get('step', -1))
    model = V6Model(V6Config(**json.loads(args.config)))
    if args.backbone_weights:
        from .encoder import load_backbone
        load_backbone(model.encoder, str(Path(args.backbone_weights).expanduser()))
    return model.to(device).eval(), 0


def pick_clips(total, n):
    if n <= 0 or n >= total:
        return list(range(total))
    return sorted({round(i * (total - 1) / max(1, n - 1)) for i in range(n)})


def clip_batches(args, cfg, rank, world):
    """(clip ids, collated batch) for this rank's share of the evenly spaced clips."""
    if args.dry_run:
        ids = list(range(args.clips))[rank::world]
        g = torch.Generator().manual_seed(args.seed + rank)
        for i in range(0, len(ids), args.batch_size):
            chunk = ids[i:i + args.batch_size]
            yield chunk, synthetic_batch(cfg, len(chunk), args.steps, 'all', args.step_dt, generator=g)
        return
    from torch.utils.data import DataLoader, Subset
    from .data_av2 import Av2ClipDataset, collate_clips
    ds = Av2ClipDataset(Path(args.data_root).expanduser(), args.split, cfg, steps=args.steps, step_dt=args.step_dt,
                        cameras='all', seed=args.seed, uint8_images=True)
    ids = pick_clips(len(ds), args.clips)[rank::world]
    loader = DataLoader(Subset(ds, ids), batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate_clips)
    for i, batch in enumerate(loader):
        yield ids[i * args.batch_size:(i + 1) * args.batch_size], batch


# ---------------------------------------------------------------- metrics

def cmean(x, m):
    """Per-row mean of x [B,R] over m (nan where m is empty)."""
    m = m & torch.isfinite(x)
    return torch.where(m, x, 0.).sum(-1) / m.sum(-1)


def psnr(mse):
    return -10 * torch.log10(mse.clamp_min(1e-10))


def ssim(x, y):
    """Mean SSIM of images [N,3,H,W] in [0, 1] (11x11 Gaussian window, sigma 1.5) -> [N]."""
    g = torch.exp(-(torch.arange(11., device=x.device) - 5).square() / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None]).expand(3, 1, 11, 11).contiguous()
    blur = lambda t: F.conv2d(t, w, groups=3)  # noqa: E731
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    c1, c2 = .01 ** 2, .03 ** 2
    s = (2 * mx * my + c1) * (2 * sxy + c2) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3))


def auc(score, positive, valid):
    """Per-row ROC AUC (ties count half) of score for positive vs negative among valid [B,R] -> [B]."""
    out = []
    for s, p, v in zip(score.double(), positive, valid):
        s, p = s[v], p[v]
        n_pos, n_neg = int(p.sum()), int((~p).sum())
        if n_pos == 0 or n_neg == 0:
            out.append(math.nan)
            continue
        uniq, inv = torch.unique(s, return_inverse=True)
        counts = torch.bincount(inv, minlength=len(uniq)).double()
        rank = (counts.cumsum(0) - (counts - 1) / 2)[inv]       # average rank of tied scores
        out.append(float((rank[p].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)))
    return torch.tensor(out, dtype=score.dtype, device=score.device)


def render_metrics(model, state, cams, idx, boxes=None, rig=None):
    """Rendering, depth and opacity metrics [B] on pixels idx [B,V,R] of cams (render_loss layout)."""
    images = cams['images'].float()
    b, v, _, h, w = images.shape
    r = idx.shape[-1]
    pixels = torch.stack(((idx % w).float(), torch.div(idx, w, rounding_mode='floor').float()), -1)
    ok_cam = cams['camera_valid'].bool()
    k = torch.where(ok_cam[..., None, None], cams['intrinsics'].float(), torch.eye(3, device=idx.device))
    c2r = torch.where(ok_cam[..., None, None], cams['cam_to_rig'].float(), torch.eye(4, device=idx.device))
    rays, z_scale = model.camera_rays(state, k, c2r, pixels, return_z_scale=True)
    out = model.render(state, rays)
    rgb_t = images.flatten(-2).gather(-1, idx[:, :, None].expand(b, v, 3, r)).permute(0, 1, 3, 2)   # [B,V,R,3]
    rgb = out.rgb.float().clamp(0, 1).view(b, v, r, 3)
    mse = (rgb - rgb_t).square().mean((-1, -2))
    mse_const = (rgb_t - rgb_t.mean(2, keepdim=True)).square().mean((-1, -2))
    m = {'psnr': cmean(psnr(mse), ok_cam), 'psnr_const': cmean(psnr(mse_const), ok_cam),
         'l1': cmean((rgb - rgb_t).abs().mean((-1, -2)), ok_cam)}
    z_t = cams['depth'].float().flatten(-2).gather(-1, idx).reshape(b, v * r)
    lidar = cams['depth_valid'].bool().flatten(-2).gather(-1, idx).reshape(b, v * r) & (z_t > 0)
    lidar &= ok_cam[:, :, None].expand(b, v, r).reshape(b, v * r)
    z = out.depth.float() * z_scale
    cov = lidar & (out.depth > 0)
    zt = z_t.clamp_min(1e-2)
    med = torch.stack([zt[i][lidar[i]].median() if bool(lidar[i].any()) else zt.new_tensor(1.) for i in range(b)])
    m.update(depth_cov=cov.sum(-1) / lidar.sum(-1), absrel=cmean((z - zt).abs() / zt, cov),
             absrel_const=cmean((med[:, None] - zt).abs() / zt, cov), rmse=cmean((z - zt).square(), cov).sqrt(),
             delta1=cmean((torch.maximum(z / zt, zt / z.clamp_min(1e-2)) < 1.25).float(), cov),
             opacity=cmean(out.opacity.float(), ok_cam[:, :, None].expand(b, v, r).reshape(b, v * r)))
    # rays that end on a real surface (half the mass or more): coverage and AbsRel, also by LiDAR range
    solid = out.sample_w.float().sum(-1) if out.sample_w is not None else (out.depth > 0).float()
    cov50 = lidar & (solid >= .5)
    rel = (z - zt).abs() / zt
    m.update(depth_cov50=cov50.sum(-1) / lidar.sum(-1), absrel50=cmean(rel, cov50),
             absrel50_near=cmean(rel, cov50 & (zt < 20)), absrel50_mid=cmean(rel, cov50 & (zt >= 20) & (zt < 40)),
             absrel50_far=cmean(rel, cov50 & (zt >= 40)))
    if boxes is not None:
        uv1 = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
        p_cam = uv1 @ torch.linalg.inv(k).transpose(-1, -2) * z_t.view(b, v, r, 1)
        p_rig = (p_cam @ c2r[..., :3, :3].transpose(-1, -2) + c2r[..., None, :3, 3]).reshape(b, v * r, 3)
        on = points_in_boxes(p_rig, boxes, *rig)
        p = out.opacity.float()
        m.update(obj_opacity=cmean(p, lidar & on), bg_opacity=cmean(p, lidar & ~on), opacity_auc=auc(p, on, lidar),
                 obj_points=(lidar & on).sum(-1).float())
    return m, out, z


def heldout_view(cams):
    """The one held-out camera per clip as a V=1 camera dict, and which clips have one [B]."""
    held = cams['camera_valid'].bool()
    v = held.float().argmax(1)
    pick = lambda x: x[torch.arange(len(v), device=v.device), v][:, None]  # noqa: E731
    return {key: pick(val) for key, val in cams.items()}, held.any(1)


def slot_metrics(model, state, tg, prev, last_pos=None):
    """Slots vs GT boxes [per clip lists]. prev: per clip {track: (centre, uid, t)} of the previous step."""
    alloc = (state.status > FREE).cpu()
    center, size = tg['box_center'].float().cpu(), tg['box_size'].float().cpu()
    ok, cls, track = tg['box_valid'].bool().cpu(), tg['box_class'].long().cpu(), tg['box_track'].long().cpu()
    pos, vel, uid = state.position.float().cpu(), state.velocity.float().cpu(), state.uid.cpu()
    canon = state.anchor_canon.float().cpu()
    extent = canon.amax(2) - canon.amin(2)
    pred_cls = model.understanding.class_logits(model.understanding.slot_embedding(state)).argmax(-1).cpu()
    t_now = state.t.double().cpu()
    d = torch.cdist(pos, center)
    keys = ('slots', 'boxes', 'center_err', 'center_err_hold', 'extent_err', 'class_acc', 'vel_err', 'vel_err_zero',
            'id_kept') + tuple(f'{n}_{g:g}m' for g in GATES for n in ('precision', 'recall'))
    rec, nxt = {key: [] for key in keys}, []
    nan = math.nan
    mean = lambda xs: sum(xs) / len(xs) if xs else nan  # noqa: E731
    for i in range(len(d)):
        di = d[i].masked_fill(~alloc[i, :, None] | ~ok[i, None, :], math.inf)
        n_slot, n_box = int(alloc[i].sum()), int(ok[i].sum())
        rec['slots'].append(n_slot)
        rec['boxes'].append(n_box)
        for g in GATES:
            n = len(_assign(di, g))
            rec[f'precision_{g:g}m'].append(n / n_slot if n_slot else nan)
            rec[f'recall_{g:g}m'].append(n / n_box if n_box else nan)
        pairs = _assign(di, max(GATES))
        cerr, herr, xerr, cacc, verr, vzero, kept = [], [], [], [], [], [], []
        here = {}
        for s, c in pairs:
            tr = int(track[i, c])
            cerr.append(float(di[s, c]))
            xerr.append(float((extent[i, s] - size[i, c]).abs().mean()))
            cacc.append(float(pred_cls[i, s] == cls[i, c]))
            if last_pos is not None:
                herr.append(float((last_pos[i, s] - center[i, c]).norm()))
            if tr >= 0 and tr in prev[i]:
                c0, u0, t0 = prev[i][tr]
                dt = float(t_now[i] - t0)
                if dt > 0:
                    v_gt = (center[i, c] - c0) / dt
                    verr.append(float((vel[i, s] - v_gt).norm()))
                    vzero.append(float(v_gt.norm()))
                if u0 is not None:
                    kept.append(float(int(uid[i, s]) == u0))
            here[tr] = (center[i, c], int(uid[i, s]))
        for c in torch.nonzero(ok[i]).flatten().tolist():  # unmatched boxes still give the next velocity target
            tr = int(track[i, c])
            if tr >= 0 and tr not in here:
                here[tr] = (center[i, c], None)
        nxt.append({tr: (c, u, float(t_now[i])) for tr, (c, u) in here.items()})
        for key, xs in (('center_err', cerr), ('center_err_hold', herr), ('extent_err', xerr), ('class_acc', cacc),
                        ('vel_err', verr), ('vel_err_zero', vzero), ('id_kept', kept)):
            rec[key].append(mean(xs))
    return rec, nxt


def rig_metrics(state, obs):
    ok = obs.rig_pose_valid.bool()
    pose = obs.rig_pose.float()
    t_err = (state.rig_position.float() - pose[:, :3, 3]).norm(dim=-1)
    rel = quat_to_matrix(state.rig_orientation.float()).transpose(-1, -2) @ pose[:, :3, :3]
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    nan = torch.full_like(t_err, math.nan)
    return {'rig_t_err': torch.where(ok, t_err, nan), 'rig_r_deg': torch.where(ok, torch.rad2deg(torch.acos(cos)), nan)}


# ---------------------------------------------------------------- pictures

def _gray(x, lo, hi):
    return ((x - lo) / (hi - lo)).clamp(0, 1)[None].expand(3, -1, -1)


def _bev(state, tg, obs, size=(192, 256), metres_per_px=.25):
    """Top view in the rig frame (forward up): GT boxes green, allocated slots red, ego blue."""
    from PIL import Image, ImageDraw
    h, w = size
    img = Image.new('RGB', (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    r_wr, t_wr = rig_to_world(state, obs)
    r_wr, t_wr = r_wr[0].cpu(), t_wr[0].cpu()
    to_px = lambda p: (w / 2 - p[1] / metres_per_px, h * .8 - p[0] / metres_per_px)  # noqa: E731

    def rig(p):
        return (p - t_wr) @ r_wr
    for c in torch.nonzero(tg['box_valid'][0].cpu()).flatten().tolist():
        rot = r_wr.T @ quat_to_matrix(tg['box_quat'][0, c].float().cpu())
        half = tg['box_size'][0, c].float().cpu() / 2
        mid = rig(tg['box_center'][0, c].float().cpu())
        corners = [mid + rot @ torch.tensor([sx * half[0], sy * half[1], 0.]) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        draw.polygon([to_px(p) for p in corners], outline=(60, 220, 60))
    alloc = (state.status[0] > FREE).cpu()
    for s in torch.nonzero(alloc).flatten().tolist():
        x, y = to_px(rig(state.position[0, s].float().cpu()))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(230, 60, 60))
    x, y = to_px(torch.zeros(3))
    draw.rectangle((x - 3, y - 6, x + 3, y + 6), fill=(80, 140, 255))
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255


def save_picture(path, rows):
    from PIL import Image
    grid = torch.cat([torch.cat(r, -1) for r in rows], -2).clamp(0, 1)
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)


def picture_row(model, state, held, tg, obs):
    """GT | render | depth (near bright) | opacity | top view, for clip 0 of the batch, at full resolution."""
    cams = {k: v[:1] for k, v in held.items()}
    h, w = cams['images'].shape[-2:]
    idx = torch.arange(h * w, device=cams['images'].device)[None, None]
    _, out, z = render_metrics(model, render_view(_slice(state, 1), _slice(obs, 1)), cams, idx)
    gt = cams['images'][0, 0].float().cpu()
    rgb = out.rgb[0].float().clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    inv = torch.where(z[0] > 0, 1 / z[0].clamp_min(1.), 0.).view(h, w).cpu()
    op = out.opacity[0].float().view(h, w).cpu()
    bev = _bev(_slice(state, 1), {k: v[:1] for k, v in tg.items() if torch.is_tensor(v)}, _slice(obs, 1), (h, w))
    return [gt, rgb, _gray(inv, 0., .5), _gray(op, 0., 1.), bev]


def _slice(x, n):
    return replace(x, **{f.name: getattr(x, f.name)[:n] for f in fields(x) if torch.is_tensor(getattr(x, f.name))})


# ---------------------------------------------------------------- run

@torch.no_grad()
def evaluate_batch(model, batch, ids, args, device, autocast, vis_dir=None):
    gen = torch.Generator().manual_seed(args.seed * 7919 + ids[0])
    steps, targets = batch_to_sequence(batch, device, heldout=True, generator=gen)
    for s, tg in enumerate(targets):
        tg['box_track'] = batch['boxes']['track'][:, s].to(device)
    b, n = len(ids), len(steps)
    observe = min(args.observe, n)
    state = model.initial_state(b, device=device, time=steps[0].t)
    prev = [{} for _ in range(b)]
    last_pos, records, rows = None, [[] for _ in range(b)], []
    for t in range(n):
        observed = torch.full((b,), t < observe, device=device)
        obs, tg = steps[t], targets[t]
        torch.manual_seed(args.seed * 104729 + ids[0] * 31 + t)   # sampled rays repeat across checkpoints
        with autocast():
            boxes = teacher_boxes(tg) if args.box_teacher else None
            state, aux = advance(model, state, obs, observed, boxes=boxes)
            terms, _ = step_terms(model, state, obs, tg, observed, aux=aux, rays_per_camera=256)
            boxes, rig = teacher_boxes(tg), rig_to_world(state, obs)
            m = {}
            cams = {'images': obs.images[:, :, -1], 'intrinsics': obs.intrinsics, 'cam_to_rig': obs.cam_to_rig,
                    'camera_valid': obs.camera_valid, 'depth': tg['target_depth'], 'depth_valid': tg['target_valid']}
            hh, ww = cams['images'].shape[-2:]
            idx = torch.randint(0, hh * ww, (b, cams['images'].shape[1], args.rays_per_camera), device=device)
            view = render_view(state, obs)   # targets are seen from the camera's actual pose
            got, _, _ = render_metrics(model, view, cams, idx, boxes, rig)
            m.update({f'in_{k}': v for k, v in got.items()
                      if k in ('psnr', 'psnr_const', 'l1', 'depth_cov', 'absrel', 'depth_cov50', 'absrel50')})
            if 'heldout' in tg:
                held, has = heldout_view(tg['heldout'])
                ys = torch.arange(0, hh, args.stride, device=device)
                xs = torch.arange(0, ww, args.stride, device=device)
                grid = (ys[:, None] * ww + xs[None]).flatten()
                got, out, _ = render_metrics(model, view, held, grid[None, None].expand(b, 1, -1), boxes, rig)
                nan = torch.full((b,), math.nan, device=device)
                m.update({f'held_{k}': torch.where(has, v, nan) for k, v in got.items()
                          if not k.startswith(('obj_', 'bg_', 'opacity_'))})
                m.update({k: v for k, v in got.items() if k.startswith(('obj_', 'bg_', 'opacity_'))})
                img = out.rgb.float().clamp(0, 1).view(b, len(ys), len(xs), 3).permute(0, 3, 1, 2)
                gt = held['images'][:, 0].float()[:, :, ys][:, :, :, xs]
                if min(len(ys), len(xs)) >= 11:
                    m['held_ssim'] = torch.where(has, ssim(img, gt), nan)
                if vis_dir is not None and bool(has[0]):
                    rows.append(picture_row(model, state, held, tg, obs))
            m.update(rig_metrics(state, obs))
        slot, prev = slot_metrics(model, state, tg, prev, last_pos=None if t < observe else last_pos)
        if t == observe - 1:
            last_pos = state.position.float().cpu().clone()
        loss_terms = sum(DEFAULT_WEIGHTS.get(k, 0.) * float(v) for k, v in terms.items())
        for i in range(b):
            rec = {'clip': ids[i], 'step': t, 'observed': t < observe, 'horizon': max(0, t - observe + 1),
                   'log_id': batch['log_id'][i] if 'log_id' in batch else None, 'loss_terms': loss_terms,
                   **{f'term_{k}': float(v) for k, v in terms.items()}}
            rec.update({k: float(v[i]) for k, v in m.items()})
            rec.update({k: float(v[i]) for k, v in slot.items()})
            records[i].append(rec)
    if rows:
        save_picture(vis_dir / f'clip_{ids[0]:06d}.png', rows)
    return [r for clip in records for r in clip]


def summarize(records, meta):
    """Means over clips per step and per regime (observed / rolled out), with a 95% normal interval."""
    def stats(rs):
        out = {'records': len(rs)}
        for key in sorted({k for r in rs for k, v in r.items() if isinstance(v, float)}):
            xs = [r[key] for r in rs if isinstance(r.get(key), float) and math.isfinite(r[key])]
            if xs:
                mu = sum(xs) / len(xs)
                sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1))
                out[key] = {'mean': mu, 'ci95': 1.96 * sd / math.sqrt(len(xs)), 'n': len(xs)}
        return out
    groups = {f'step_{s}': [r for r in records if r['step'] == s] for s in sorted({r['step'] for r in records})}
    groups['observed'] = [r for r in records if r['observed']]
    groups['rolled_out'] = [r for r in records if not r['observed']]
    return {**meta, 'clips': len({r['clip'] for r in records}), 'groups': {k: stats(v) for k, v in groups.items() if v}}


def summary_markdown(summary):
    groups = summary['groups']
    names = list(groups)
    lines = [f"# v6 eval: {summary.get('checkpoint')} (step {summary.get('step')}), {summary.get('split')}, "
             f"{summary['clips']} clips, box teacher {'on' if summary.get('box_teacher') else 'off'}", '',
             '| metric | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for key in HEADLINE:
        cells = []
        for g in names:
            v = groups[g].get(key)
            cells.append('' if v is None else f"{v['mean']:.3f}")
        if any(cells):
            lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    rank, world, device = setup_distributed(args.device)
    model, step = load_model(args, device)
    out_dir = Path(args.out).expanduser()
    vis_dir = out_dir / 'vis'
    if rank == 0:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    autocast = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if args.bf16 else contextlib.nullcontext
    shard = out_dir / f'records_rank{rank}.jsonl'
    shard.unlink(missing_ok=True)
    done, t0 = 0, time.time()
    for ids, batch in clip_batches(args, model.cfg, rank, world):
        records = evaluate_batch(model, batch, ids, args, device, autocast,
                                 vis_dir if rank == 0 and done < args.vis else None)
        with open(shard, 'a') as fh:
            fh.writelines(json.dumps(r) + '\n' for r in records)
        done += len(ids)
        if rank == 0:
            print(json.dumps({'clips_done': done, 'rank': rank, 's_per_clip': (time.time() - t0) / done}), flush=True)
    if world > 1:
        dist.barrier()
    summary = None
    if rank == 0:
        records = [json.loads(line) for f in sorted(out_dir.glob('records_rank*.jsonl')) for line in open(f)]
        summary = summarize(records, {'checkpoint': args.checkpoint, 'step': step, 'split': args.split,
                                      'observe': args.observe, 'steps': args.steps, 'stride': args.stride,
                                      'box_teacher': args.box_teacher, 'seed': args.seed})
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
        (out_dir / 'summary.md').write_text(summary_markdown(summary))
        print(summary_markdown(summary), flush=True)
    if world > 1:
        dist.destroy_process_group()
    return summary


if __name__ == '__main__':
    main()
