"""Fast v1 objective: reconstruction, prediction rollout, encoder depth, slot boxes, rig pose.

Each term returns (scalar, log dict). ``sequence_loss`` runs a clip: observe the first S steps
(S per clip from its own RNG), then roll out with ``predict``; truncated BPTT every ``bptt`` steps.
Sampled pixel (x, y) is the ray through (u, v) = (x, y): pixel centres at integers, as data_av2's
intrinsics and LiDAR z-buffer (rays.py patch centres use i + 0.5; half a pixel, ignored here).
"""
from __future__ import annotations

import contextlib
import math
import random
from dataclasses import fields, replace

import torch
import torch.nn.functional as F

from .contracts import FREE, Observation, V6State
from .geometry import quat_to_matrix

try:  # Hungarian when available, greedy otherwise
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - depends on the environment
    linear_sum_assignment = None

DEFAULT_WEIGHTS = {
    'recon_rgb': 1., 'recon_depth': .5, 'recon_occ': .5,
    'pred_rgb': 1., 'pred_depth': .5, 'pred_occ': .5,
    'enc_depth': .1,
    'slot_pos': .1, 'slot_vel': .1, 'slot_extent': .1, 'slot_class': .1, 'spawn': .1, 'proposal': .05,
    'own_box': .1, 'assign_box': .1,
    'rig': .1,
}
BOX_KEYS = ('box_center', 'box_size', 'box_class', 'box_valid')
OCC_MARGIN = .05   # m, box pad when testing whether a LiDAR target point is on an object


def masked_mean(value, mask):
    mask = mask.to(value.device).expand_as(value) & torch.isfinite(value)
    return torch.where(mask, value, 0.).sum() / mask.sum().clamp_min(1)


def _zero(ref):
    return ref.sum() * 0. if ref.is_floating_point() and ref.requires_grad else torch.zeros((), device=ref.device)


def where_state(mask, a: V6State, b: V6State) -> V6State:
    """Per-clip select: a where mask[B] else b."""
    out = {}
    for f in fields(V6State):
        x, y = getattr(a, f.name), getattr(b, f.name)
        m = mask.to(x.device).view(-1, *([1] * (x.dim() - 1)))
        out[f.name] = torch.where(m, x, y)
    return replace(a, **out)


def observed_counts(seeds, steps: int, low: int = 2):
    """S per clip, each from its own stream: S in [low, steps - 1] (all steps when too short)."""
    if steps <= low:
        return [steps] * len(seeds)
    return [random.Random(f'v6-observe:{s}').randint(low, steps - 1) for s in seeds]


# ---------------------------------------------------------------- rendering

def sample_pixels(depth_valid, rays_per_camera: int):
    """Flat pixel indices [B,V,R]: half uniform, half biased to target-depth pixels."""
    b, v, h, w = depth_valid.shape
    r_uniform = rays_per_camera - rays_per_camera // 2
    uniform = torch.randint(0, h * w, (b, v, r_uniform), device=depth_valid.device)
    r_depth = rays_per_camera - r_uniform
    if r_depth == 0:
        return uniform
    weights = depth_valid.reshape(b * v, h * w).float() + 1e-6
    biased = torch.multinomial(weights, r_depth, replacement=True).reshape(b, v, r_depth)
    return torch.cat((uniform, biased), -1)


def _clip_mean(value, mask):
    """Per-clip masked mean over rays: ([B], counts [B])."""
    mask = mask & torch.isfinite(value)
    count = mask.sum(-1)
    return torch.where(mask, value, 0.).sum(-1) / count.clamp_min(1), count


def clips_mean(per_clip, count, mask):
    """Mean over the clips in mask [B] that have data."""
    ok = mask.to(count.device) & (count > 0)
    return torch.where(ok, per_clip, 0.).sum() / ok.sum().clamp_min(1)


def points_in_boxes(points, boxes: dict, r_wr, t_wr, margin=OCC_MARGIN):
    """points [B,N,3] rig frame, world boxes {center, size, quat, valid}, rig -> world r_wr [B,3,3], t_wr [B,3]
    -> [B,N] bool: inside any valid box."""
    c_rig = torch.einsum('bji,bmj->bmi', r_wr, boxes['center'].float() - t_wr[:, None])
    r_box = r_wr.transpose(-1, -2)[:, None] @ quat_to_matrix(boxes['quat'].float())
    local = torch.einsum('bmji,bnmj->bnmi', r_box, points[:, :, None] - c_rig[:, None])
    inside = (local.abs() <= boxes['size'].float()[:, None] * .5 + margin).all(-1)
    return (inside & boxes['valid'].bool()[:, None]).any(-1)


def rig_to_world(state: V6State, obs: Observation):
    """Rig -> world (R [B,3,3], t [B,3]): metadata pose where known, else the state's."""
    known = obs.rig_pose_valid.bool()
    pose = obs.rig_pose.float()
    r = torch.where(known[:, None, None], pose[:, :3, :3], quat_to_matrix(state.rig_orientation.float()))
    return r, torch.where(known[:, None], pose[:, :3, 3], state.rig_position.float())


def render_loss(model, state: V6State, cams: dict, rays_per_camera: int, boxes=None, rig=None):
    """Per-clip L1 RGB on sampled pixels and log-depth L1 on target-depth pixels.

    cams: images [B,V,3,H,W], intrinsics [B,V,3,3], cam_to_rig [B,V,4,4], camera_valid [B,V],
    depth [B,V,H,W] (target depth, camera z), depth_valid [B,V,H,W].
    With world GT boxes and rig (rig_to_world): class-balanced opacity BCE on target-depth rays that cross a
    slot box (others have no density to move), 1 where the LiDAR point lies in a GT box, else 0 (objects are
    opaque slots; the rest belongs to the background).
    Returns {rgb, rgb_n, depth, depth_n, occ, occ_n} ([B] each) and a log dict.
    """
    images = cams['images']
    b, v, _, h, w = images.shape
    idx = sample_pixels(cams['depth_valid'].bool(), rays_per_camera)          # [B,V,R]
    r = idx.shape[-1]
    pixels = torch.stack(((idx % w).float(), torch.div(idx, w, rounding_mode='floor').float()), -1)
    ok_cam = cams['camera_valid'].bool()
    k = torch.where(ok_cam[..., None, None], cams['intrinsics'].float(), torch.eye(3, device=idx.device))
    c2r = torch.where(ok_cam[..., None, None], cams['cam_to_rig'].float(), torch.eye(4, device=idx.device))
    rays, z_scale = model.camera_rays(state, k, c2r, pixels, return_z_scale=True)
    out = model.render(state, rays)
    rgb_t = images.float().flatten(-2).gather(-1, idx[:, :, None].expand(b, v, 3, r))  # [B,V,3,R]
    rgb_t = rgb_t.permute(0, 1, 3, 2).reshape(b, v * r, 3)
    z_t = cams['depth'].float().flatten(-2).gather(-1, idx).reshape(b, v * r)
    z_ok = cams['depth_valid'].bool().flatten(-2).gather(-1, idx).reshape(b, v * r) & (z_t > 0)
    ray_ok = ok_cam[:, :, None].expand(b, v, r).reshape(b, v * r)
    rgb, rgb_n = _clip_mean((out.rgb.float() - rgb_t).abs().mean(-1), ray_ok)
    # The decoder has no background depth (0 where no slot surface): depth only where a surface exists.
    surf = out.depth.detach() > 0
    z_pred = (out.depth.float() * z_scale).clamp_min(1e-2)
    depth, depth_n = _clip_mean((z_pred.log() - z_t.clamp_min(1e-2).log()).abs(), ray_ok & z_ok & surf)
    logs = {'depth_rays': (ray_ok & z_ok).sum().detach().float(),
            'depth_rays_on_surface': (ray_ok & z_ok & surf).sum().detach().float(),
            'opacity': masked_mean(out.opacity.float().detach(), ray_ok)}
    occ, occ_n = torch.zeros_like(rgb), torch.zeros_like(rgb_n)
    if boxes is not None:
        with torch.autocast(device_type=idx.device.type, enabled=False):
            uv1 = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
            p_cam = uv1 @ torch.linalg.inv(k).transpose(-1, -2) * z_t.view(b, v, r, 1)
            p_rig = (p_cam @ c2r[..., :3, :3].transpose(-1, -2) + c2r[..., None, :3, 3]).reshape(b, v * r, 3)
            on = points_in_boxes(p_rig, boxes, *rig)
            ok = ray_ok & z_ok & model.decoder.slot_hits(state, rays).any(-1)
            p = out.opacity.float().clamp(1e-4, 1 - 1e-4)
            bce = -torch.where(on, p.log(), (1 - p).log())
            pos, pos_n = _clip_mean(bce, ok & on)
            neg, neg_n = _clip_mean(bce, ok & ~on)
            occ = torch.where((pos_n > 0) & (neg_n > 0), .5 * (pos + neg), torch.where(pos_n > 0, pos, neg))
            occ_n = pos_n + neg_n
            logs.update(obj_rays=(ok & on).sum().float(), obj_opacity=masked_mean(p.detach(), ok & on),
                        bg_opacity=masked_mean(p.detach(), ok & ~on))
    return {'rgb': rgb, 'rgb_n': rgb_n, 'depth': depth, 'depth_n': depth_n, 'occ': occ, 'occ_n': occ_n}, logs


# ---------------------------------------------------------------- encoder depth

def encoder_depth_loss(enc, obs: Observation, cfg=None):
    """Mixture Gaussian NLL of the depth hypotheses (log z) vs the per-patch median of the input LiDAR."""
    target, valid = enc.measured_depth, enc.measured_valid
    if cfg is not None and not bool(valid.any()):
        from .rays import patch_depth_stats  # encoder left the tag empty: compute it here
        target, valid = patch_depth_stats(obs.depth, obs.depth_valid, cfg)
    hyp = enc.depth_hyp.float()
    mu, log_sigma, logit = hyp[..., 2], hyp[..., 3].clamp(-6., 4.), hyp[..., 4]
    x = target[..., 0].float().detach()[..., None]
    logp = (F.log_softmax(logit, -1) - .5 * ((x - mu) * torch.exp(-log_sigma)).square()
            - log_sigma - .5 * math.log(2 * math.pi))
    nll = -torch.logsumexp(logp, -1)
    mask = valid.bool() & enc.valid.bool() & obs.camera_valid.bool()[..., None]
    loss = masked_mean(nll, mask)
    best = mu.gather(-1, logit.argmax(-1, keepdim=True))[..., 0]
    return loss, {'enc_depth_abs_log': masked_mean((best - x[..., 0]).abs(), mask).detach(),
                  'enc_depth_patches': mask.sum().detach()}


# ---------------------------------------------------------------- slots vs boxes

def _assign(cost, gate):
    """One-to-one (row, col) pairs with cost < gate. cost: CPU [K,M]."""
    if cost.numel() == 0 or not bool(torch.isfinite(cost).any()):
        return []
    if linear_sum_assignment is not None:
        # pairs beyond the gate cost the same, so they never displace a pair inside it
        rows, cols = linear_sum_assignment(torch.where(cost < gate, cost, 1e9).numpy())
        return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < gate]
    pairs, cost = [], cost.clone()
    while True:
        r, c = divmod(int(cost.argmin()), cost.shape[1])
        if not cost[r, c] < gate:
            return pairs
        pairs.append((r, c))
        cost[r, :] = math.inf
        cost[:, c] = math.inf


def match_slots(slot_pos, slot_ok, box_pos, box_ok, gate, taken=None):
    """Per clip, slots [B,K,3] to boxes [B,M,3] by world distance. Returns box index [B,K] (-1: none), taken [B,M]."""
    b, k = slot_pos.shape[:2]
    out = torch.full((b, k), -1, dtype=torch.long)
    taken = torch.zeros(box_ok.shape, dtype=torch.bool) if taken is None else taken.clone()
    d = torch.cdist(slot_pos.detach().float().cpu(), box_pos.detach().float().cpu())
    free_box = box_ok.cpu() & ~taken
    d = d.masked_fill(~slot_ok.cpu()[:, :, None] | ~free_box[:, None, :], math.inf)
    for i in range(b):
        for r, c in _assign(d[i], gate):
            out[i, r] = c
            taken[i, c] = True
    return out.to(slot_pos.device), taken


def _proposal_position(state: V6State, meas):
    """World mean of each slot's valid measured anchors (rig frame -> world): [B,K,3], any-valid [B,K]."""
    ok = meas.anchor_meas_valid.bool()
    rig = (meas.anchor_meas.float() * ok[..., None]).sum(2) / ok.sum(2, keepdim=True).clamp_min(1)
    rot = quat_to_matrix(state.rig_orientation.float())
    return rig @ rot.transpose(-1, -2) + state.rig_position.float()[:, None], ok.any(-1)


def slot_losses(model, state: V6State, targets: dict, *, spawn_mask=None, prior=None, meas=None, gate: float = 4.):
    """Slots vs world boxes (box_center/size/class/valid).

    1. Allocated slots (incl. just spawned) matched within ``gate``: L1 position, L1 anchor extent vs box
       size, class CE (on a detached C_obj / anchors: understanding does not pull the state). A slot the box
       teacher paired with a box is matched to it at any distance.
    2. Slots free in the prior and not spawned: their proposal (mean measured anchor, world) is matched
       without a gate to the boxes left over; L1 proposal -> box pulls proposals onto objects.
    Spawn BCE on slots free in the prior (clips in spawn_mask): 1 if spawned onto a box (step 1), the
    proposal is within ``gate`` of a left-over box (step 2) or the box teacher routed a box to it, else 0.
    3. Box teacher: ownership CE pulls the tokens of each box onto the slot holding or given it (every
       ownership layer); anchor CE pulls each onto its box-grid anchor.
    """
    zero = _zero(state.position)
    terms = {'slot_pos': zero, 'slot_vel': zero, 'slot_extent': zero, 'slot_class': zero, 'spawn': zero, 'proposal': zero,
             'own_box': zero, 'assign_box': zero}
    if any(key not in targets for key in BOX_KEYS):
        return terms, {}
    with torch.autocast(device_type=state.position.device.type, enabled=False):
        center, size = targets['box_center'].float(), targets['box_size'].float()
        cls, box_ok = targets['box_class'].long(), targets['box_valid'].bool()
        allocated = state.status > FREE
        paired = None if meas is None or meas.spawn_box is None else torch.where(allocated, meas.spawn_box, -1)
        if paired is None:
            match, taken = match_slots(state.position, allocated, center, box_ok, gate)
        else:
            m = center.shape[1]
            taken = torch.zeros(paired.shape[0], m + 1, dtype=torch.bool, device=paired.device)
            taken = taken.scatter(1, torch.where(paired >= 0, paired, m), True)[:, :m].cpu()
            match, taken = match_slots(state.position, allocated & (paired < 0), center, box_ok, gate, taken=taken)
            match = torch.where(paired >= 0, paired, match)
        hit = match >= 0
        idx = match.clamp_min(0)
        box_c = center.gather(1, idx[..., None].expand(-1, -1, 3))
        box_s = size.gather(1, idx[..., None].expand(-1, -1, 3))
        box_k = cls.gather(1, idx)
        terms['slot_pos'] = masked_mean((state.position.float() - box_c).abs().sum(-1), hit)
        if 'box_velocity' in targets:  # world velocity of the box's track from neighbouring steps
            box_v = targets['box_velocity'].float().gather(1, idx[..., None].expand(-1, -1, 3))
            v_ok = hit & targets['box_velocity_valid'].bool().gather(1, idx)
            terms['slot_vel'] = masked_mean((state.velocity.float() - box_v).abs().sum(-1), v_ok)
        canon = state.anchor_canon.float()
        terms['slot_extent'] = masked_mean(((canon.amax(2) - canon.amin(2)) - box_s).abs().sum(-1), hit)
        frozen = replace(state, c_obj=state.c_obj.detach(), anchor_canon=state.anchor_canon.detach())
        logits = model.understanding.class_logits(model.understanding.slot_embedding(frozen)).float()
        n_cls = logits.shape[-1]
        cls_ok = hit & (box_k >= 0) & (box_k < n_cls)
        ce = F.cross_entropy(logits.flatten(0, 1), box_k.clamp(0, n_cls - 1).flatten(), reduction='none')
        terms['slot_class'] = masked_mean(ce.view_as(box_k), cls_ok)
        logs = {'slots_allocated': allocated.float().sum(-1).mean(), 'boxes': box_ok.float().sum(-1).mean(),
                'slots_matched': hit.float().sum(-1).mean(),
                'class_acc': masked_mean((logits.argmax(-1) == box_k).float(), cls_ok)}
        if meas is not None and prior is not None:
            free = prior.status == FREE
            clips = torch.ones_like(free[:, 0]) if spawn_mask is None else spawn_mask.to(free.device).bool()
            prop, prop_ok = _proposal_position(state, meas)
            cand = free & ~allocated & prop_ok & clips[:, None]
            late, _ = match_slots(prop, cand, center, box_ok, math.inf, taken=taken)
            got = late >= 0
            box_p = center.gather(1, late.clamp_min(0)[..., None].expand(-1, -1, 3))
            dist = (prop - box_p).abs().sum(-1)
            terms['proposal'] = masked_mean(dist, got)
            near = got & ((prop - box_p).detach().norm(dim=-1) < gate)
            target = (hit & free) | near
            if meas.spawn_forced is not None:
                target = target | (meas.spawn_forced & free)
                tok = meas.forced_slot
                nll = [-own.float().gather(-1, tok.clamp_min(0)[..., None]).squeeze(-1).clamp_min(1e-6).log()
                       for own in meas.ownership_by_layer]
                terms['own_box'] = masked_mean(torch.stack(nll).mean(0), tok >= 0)
                terms['assign_box'] = masked_mean(meas.assign_nll.float(), tok >= 0)
                logs['forced_spawns'] = meas.spawn_forced.float().sum(-1).mean()
            target = target.float()
            mask = free & clips[:, None]
            bce = F.binary_cross_entropy_with_logits(meas.spawn_logit.float(), target, reduction='none')
            terms['spawn'] = masked_mean(bce, mask)
            logs.update(spawn_pos_rate=masked_mean(target, mask), proposals_matched=got.float().sum(-1).mean(),
                        proposal_dist=masked_mean((prop - box_p).detach().norm(dim=-1), got))
    return terms, {k: v.detach() for k, v in logs.items()}


# ---------------------------------------------------------------- rig pose

def rig_pose_loss(state: V6State, obs: Observation):
    """L1 on rig position and rotation matrix vs metadata ego pose, where valid."""
    with torch.autocast(device_type=state.rig_position.device.type, enabled=False):
        target = obs.rig_pose.float()
        ok = obs.rig_pose_valid.bool()
        t_err = (state.rig_position.float() - target[:, :3, 3]).abs().sum(-1)
        r_err = (quat_to_matrix(state.rig_orientation.float()) - target[:, :3, :3]).abs().sum((-1, -2))
        loss = masked_mean(t_err + r_err, ok)
    return loss, {'rig_t_err': masked_mean(t_err, ok).detach(), 'rig_r_err': masked_mean(r_err, ok).detach()}


# ---------------------------------------------------------------- one step / one clip

def input_cameras(obs: Observation, targets: dict):
    """Input cameras at time t, with the held-out LiDAR as depth target."""
    return {'images': obs.images[:, :, -1], 'intrinsics': obs.intrinsics, 'cam_to_rig': obs.cam_to_rig,
            'camera_valid': obs.camera_valid, 'depth': targets['target_depth'],
            'depth_valid': targets['target_valid']}


def step_terms(model, state, obs, targets, observed, *, aux=None, rays_per_camera=256, gate=4.):
    """All terms for one step. observed [B]: clip took step() (posterior) rather than predict().
    One render per camera set; clips split into recon (observed) and pred (rolled out)."""
    cam_sets = [input_cameras(obs, targets)]
    if 'heldout' in targets:  # cameras never fed to the model
        cam_sets.append(targets['heldout'])
    boxes = teacher_boxes(targets)
    rig = None if boxes is None else rig_to_world(state, obs)
    rendered = [render_loss(model, state, cams, rays_per_camera, boxes=boxes, rig=rig) for cams in cam_sets]
    terms, logs = {}, dict(rendered[0][1])
    for name, mask in (('recon', observed), ('pred', ~observed)):
        for key in ('rgb', 'depth', 'occ'):
            terms[f'{name}_{key}'] = torch.stack([clips_mean(r[key], r[f'{key}_n'], mask) for r, _ in rendered]).mean()
    if aux is not None:
        terms['enc_depth'], l = encoder_depth_loss(aux['enc'], obs, model.cfg)
        logs.update(l)
    slot, l = slot_losses(model, state, targets, spawn_mask=observed, gate=gate,
                          prior=None if aux is None else aux['prior'], meas=None if aux is None else aux['meas'])
    terms.update(slot)
    logs.update(l)
    terms['rig'], l = rig_pose_loss(state, obs)
    logs.update(l)
    return terms, logs


def teacher_boxes(targets: dict):
    """World GT boxes for the trunk's box teacher, or None when the targets carry no box orientation."""
    if 'box_quat' not in targets or any(key not in targets for key in BOX_KEYS):
        return None
    return {'center': targets['box_center'], 'size': targets['box_size'], 'quat': targets['box_quat'],
            'valid': targets['box_valid']}


def advance(model, state, obs, observed, boxes=None):
    """step() for clips in observed [B], predict() for the rest. Returns (state, aux or None)."""
    if not bool(observed.any()):
        return model.predict(state, obs.t - state.t), None
    post, aux = model.step(state, obs, boxes=boxes)
    if not bool(observed.all()):
        post = where_state(observed, post, model.predict(state, obs.t - state.t))
    return post, aux


def sequence_loss(model, steps, targets, *, seeds, weights=None, rays_per_camera=256, bptt=3,
                  backward=None, autocast=None, gate=4., box_teacher=True, teacher_p=1.):
    """Run one batch of clips. steps: list[Observation] (T), targets: list[dict] (T).

    Clip b observes its first S_b steps (S_b from seeds[b]) and is rolled out with predict() after.
    The state is detached every ``bptt`` steps; with ``backward`` (callable on a chunk loss) each chunk
    is backpropagated right away to free its graph. Returns (total loss: tensor if backward is None,
    else detached), log dict of floats. box_teacher: in training, GT boxes route evidence to free slots,
    for each clip with probability teacher_p (drawn from its seed); the other clips spawn on their own.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    ctx = autocast or contextlib.nullcontext
    n = len(steps)
    device = steps[0].t.device
    counts = observed_counts(seeds, n)
    observed = torch.tensor([[t < s for t in range(n)] for s in counts], device=device)
    taught = torch.tensor([random.Random(f'v6-teacher:{s}').random() < teacher_p for s in seeds], device=device)
    state = model.initial_state(steps[0].t.shape[0], device=device, time=steps[0].t)
    total, chunk, sums, hits = 0., 0., {}, {}
    for t in range(n):
        with ctx():
            boxes = teacher_boxes(targets[t]) if box_teacher and model.training and bool(taught.any()) else None
            if boxes is not None:
                boxes = {**boxes, 'valid': boxes['valid'].bool() & taught[:, None]}
            state, aux = advance(model, state, steps[t], observed[:, t], boxes=boxes)
            terms, logs = step_terms(model, state, steps[t], targets[t], observed[:, t], aux=aux,
                                     rays_per_camera=rays_per_camera, gate=gate)
            loss = sum(weights.get(k, 0.) * v.float() for k, v in terms.items()) / n
        chunk = chunk + loss
        for k, v in list(terms.items()) + list(logs.items()):
            sums[k] = sums.get(k, 0.) + v.detach().float()
            hits[k] = hits.get(k, 0) + 1
        if (t + 1) % max(1, bptt) == 0 or t == n - 1:
            if backward is not None:
                backward(chunk)
                total = total + chunk.detach()
            else:
                total = total + chunk
            chunk = 0.
            state = state.detach()
    keys = sorted(sums)
    values = torch.stack([sums[k] / hits[k] for k in keys]).tolist() if keys else []
    log = dict(zip(keys, values))
    log['loss'] = float(total.detach())
    log['observed_steps'] = sum(counts) / len(counts)
    log['taught_clips'] = float(taught.float().mean())
    return total, log
