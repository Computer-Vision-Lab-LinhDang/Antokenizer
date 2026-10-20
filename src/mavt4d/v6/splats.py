"""Pixel-aligned 3D Gaussians (v7a): written from each observed step's tokens, kept in the state, rasterized.

Write: every patch token of an input camera emits g x g Gaussians (g = cfg.gaussians), one per sub-cell of its
patch. A Gaussian sits on the ray through its sub-cell centre (+ a learned shift inside the sub-cell) at the camera
depth of the token's most likely encoder depth hypothesis times a learned factor; it starts round, as wide as half
its sub-cell's footprint at that depth; its colour starts at the sub-cell's mean input colour (a logit skip) and
its dense channels come from the token. The last layer starts at zero, so a fresh head writes exactly that prior.
The state keeps the Gaussians of the last cfg.gaussian_steps observed steps in a ring, world frame; a step that is
not observed (predict) leaves them bit-identical, so a rolled-out step renders the last observations from the
camera's new pose.
Render: alpha compositing of the Gaussians projected into each camera (3DGS EWA splatting: Jacobian of the
pinhole projection, EPS2D low-pass, alpha = opacity * exp(-d^T conic d / 2) clamped to ALPHA_MAX and dropped below
ALPHA_MIN, front to back in the depth order of the means), sky colour from the background tokens in the left-over
transmittance. Dense channels are mixed with detached weights, as in the field decoder: features never move
geometry. Pixel (x, y) is the ray through (u, v) = (x, y), as in losses.py.
Backends: gsplat rasterizes whole images on the GPU (AMD ships a ROCm build for MI300X); 'torch' composites only
the requested pixels and is the reference (tests, CPU).
"""
from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import Observation, RenderOutput, V6State
from .geometry import quat_mul, quat_to_matrix
from .rays import fourier_log_depth, no_autocast
from .scene import rig_to_world
from .write import matrix_to_quat

EPS2D = .3              # px^2 added to every projected covariance (gsplat's eps2d)
ALPHA_MIN = 1 / 255     # a Gaussian adds nothing to a pixel below this alpha
ALPHA_MAX = .99
NEAR = .1               # m, camera z below which a Gaussian is not drawn
HIT_OPACITY = 1e-3      # accumulated alpha below this: the pixel has no surface (depth 0)
LOG_Z_CLAMP = (math.log(.5), math.log(250.))
LOG_DEPTH_STEP = 2.     # learned log-depth factor in [-2, 2]
LOG_SCALE_STEP = 3.     # learned log-size factor in [-3, 3]
PAIR_BUDGET = 1 << 24   # candidate (pixel, Gaussian) pairs per chunk in the torch backend
FOOTPRINT_BUDGET = 1 << 22  # camera x Gaussian projections per chunk when pairing (no grad)
TILE = 16               # px: the torch backend bins Gaussians into tiles before pairing them with pixels
PARAMS = 14             # log depth, shift (2), log scale (3), quaternion (4), opacity, colour (3)


def _gsplat():
    try:
        import gsplat
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return gsplat


def backend(cfg, device) -> str:
    """'gsplat' or 'torch' for cfg.gaussian_backend on this device."""
    if cfg.gaussian_backend == 'torch' or (cfg.gaussian_backend == 'auto'
                                            and (device.type == 'cpu' or _gsplat() is None)):
        return 'torch'
    if _gsplat() is None:
        raise RuntimeError('gaussian_backend gsplat: the gsplat package is not importable')
    return 'gsplat'


def subcell_pixels(cfg, device) -> Tensor:
    """[N,g*g,2] (u, v) of each patch's sub-cell centres, patches row-major (rays.py: pixel i covers [i, i + 1))."""
    p, g = cfg.patch_size, cfg.gaussians
    hp, wp = cfg.image_height // p, cfg.image_width // p
    r, c = torch.meshgrid(torch.arange(hp, device=device), torch.arange(wp, device=device), indexing='ij')
    i, j = torch.meshgrid(torch.arange(g, device=device), torch.arange(g, device=device), indexing='ij')
    u = c.reshape(-1, 1) * p + (j.reshape(1, -1) + .5) * (p / g)
    v = r.reshape(-1, 1) * p + (i.reshape(1, -1) + .5) * (p / g)
    return torch.stack((u, v), -1).float()


def subcell_colours(images: Tensor, cfg) -> Tensor:
    """Last tube frame [B,V,3,H,W] -> mean colour of every sub-cell [B,V,N,g*g,3], token order of subcell_pixels."""
    b, v, _, h, w = images.shape
    p, g = cfg.patch_size, cfg.gaussians
    x = images.float() / 255 if images.dtype == torch.uint8 else images.float()
    pooled = F.avg_pool2d(x.flatten(0, 1), p // g)                                      # [BV,3,H/s,W/s]
    hp, wp = h // p, w // p
    pooled = pooled.view(b, v, 3, hp, g, wp, g).permute(0, 1, 3, 5, 4, 6, 2)            # [B,V,Hp,Wp,g,g,3]
    return pooled.reshape(b, v, hp * wp, g * g, 3)


def camera_to_world(state: V6State, obs: Observation) -> Tensor:
    """[B,V,4,4] camera -> world at obs.t: cam_to_rig, then the rig pose (metadata where known, else the state's)."""
    r, t = rig_to_world(state, obs)
    rig = torch.eye(4, device=r.device).expand(r.shape[0], 4, 4).clone()
    rig[:, :3, :3], rig[:, :3, 3] = r, t
    return rig[:, None] @ obs.cam_to_rig.float()


class SplatHead(nn.Module):
    """Patch tokens of one observed step -> its pixel-aligned Gaussians, world frame."""

    def __init__(self, cfg, hidden: int = 512):
        super().__init__()
        self.cfg = cfg
        g2 = cfg.gaussians ** 2
        geo = 2 * cfg.fourier_bands + 3 + 3       # Fourier(log z), measured tag (log z, log1p count, valid), ray
        self.norm = nn.LayerNorm(cfg.encoder_width)
        self.inp = nn.Linear(cfg.encoder_width + geo, hidden)
        self.out = nn.Linear(hidden, g2 * (PARAMS + cfg.dense_dim))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, state: V6State, obs: Observation, enc) -> dict:
        """-> {mean [B,M,3], log_scale [B,M,3], quat [B,M,4], opacity [B,M] logit, color [B,M,3 + dense_dim],
        valid [B,M]}, M = V * N * g^2 in camera, patch, sub-cell order."""
        cfg = self.cfg
        b, v, n, _ = enc.tokens.shape
        g2 = cfg.gaussians ** 2
        hyp = enc.depth_hyp.float()
        log_z = hyp[..., 2].gather(-1, hyp[..., 4].argmax(-1, keepdim=True))[..., 0]    # [B,V,N] best hypothesis
        tag = torch.cat((enc.measured_depth.float(), enc.measured_valid.float()[..., None]), -1)
        geo = torch.cat((fourier_log_depth(log_z.detach(), cfg.fourier_bands), tag, enc.ray_cam.float()), -1)
        x = torch.cat((self.norm(enc.tokens.float()), geo), -1)
        raw = self.out(F.silu(self.inp(x))).float().view(b, v, n, g2, PARAMS + cfg.dense_dim)
        with no_autocast(raw):
            cell = cfg.patch_size / cfg.gaussians
            uv = subcell_pixels(cfg, raw.device) + .5 * cell * torch.tanh(raw[..., 1:3])   # [B,V,N,g2,2]
            k = obs.intrinsics.float()[:, :, None, None]                                   # [B,V,1,1,3,3]
            fx, fy, cx, cy, skew = k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2], k[..., 0, 1]
            fx = torch.where(fx.abs() > 1e-6, fx, torch.ones_like(fx))
            fy = torch.where(fy.abs() > 1e-6, fy, torch.ones_like(fy))
            yn = (uv[..., 1] - cy) / fy
            xn = (uv[..., 0] - cx - skew * yn) / fx
            z = (log_z[..., None] + LOG_DEPTH_STEP * torch.tanh(raw[..., 0] / LOG_DEPTH_STEP)).clamp(*LOG_Z_CLAMP).exp()
            p_cam = torch.stack((xn * z, yn * z, z), -1)                                    # [B,V,N,g2,3]
            c2w = camera_to_world(state, obs)                                               # [B,V,4,4]
            mean = torch.einsum('bvij,bvnmj->bvnmi', c2w[..., :3, :3], p_cam) + c2w[:, :, None, None, :3, 3]
            # round, as wide as half the sub-cell's footprint at its depth
            size = .5 * cell * z / fx
            log_scale = size.clamp_min(1e-4).log()[..., None] + LOG_SCALE_STEP * torch.tanh(raw[..., 3:6] / LOG_SCALE_STEP)
            q_local = F.normalize(raw[..., 6:10] + raw.new_tensor([1., 0., 0., 0.]), dim=-1)
            q_c2w = matrix_to_quat(c2w[..., :3, :3])[:, :, None, None]
            quat = F.normalize(quat_mul(q_c2w, q_local), dim=-1)
            colour = subcell_colours(obs.images[:, :, -1], cfg).clamp(.01, .99)
            rgb = torch.sigmoid(torch.logit(colour) + raw[..., 11:14])
            color = torch.cat((rgb, raw[..., PARAMS:]), -1)
            valid = (enc.valid & obs.camera_valid[:, :, None])[..., None].expand(b, v, n, g2)
        return {'mean': mean.reshape(b, -1, 3), 'log_scale': log_scale.reshape(b, -1, 3),
                'quat': quat.reshape(b, -1, 4), 'opacity': raw[..., 10].reshape(b, -1),
                'color': color.reshape(b, -1, color.shape[-1]), 'valid': valid.reshape(b, -1)}


def write_ring(state: V6State, splats: dict, cfg) -> V6State:
    """Put one step's Gaussians into ring slot gs_count % gaussian_steps (camera rows past V stay invalid)."""
    b, ng = state.gs_valid.shape
    per_step = ng // cfg.gaussian_steps
    m = splats['valid'].shape[1]
    slot = (state.gs_count % cfg.gaussian_steps)                                                # [B]
    pos = torch.arange(ng, device=slot.device)
    here = (pos[None] // per_step) == slot[:, None]                                               # [B,Ng]
    src = (pos[None] - slot[:, None] * per_step).clamp(0, per_step - 1)                          # [B,Ng]

    def put(old, new):  # rows past m (camera rows the batch lacks) get zeros, so they write invalid
        new = new.to(old.dtype)
        if m < per_step:
            new = torch.cat((new, new.new_zeros(b, per_step - m, *new.shape[2:])), 1)
        new = new.gather(1, src.view(b, ng, *([1] * (new.dim() - 2))).expand(b, ng, *new.shape[2:]))
        mask = here.view(b, ng, *([1] * (old.dim() - 2)))
        return torch.where(mask, new, old)

    return replace(state, gs_mean=put(state.gs_mean, splats['mean']),
                   gs_log_scale=put(state.gs_log_scale, splats['log_scale']), gs_quat=put(state.gs_quat, splats['quat']),
                   gs_opacity=put(state.gs_opacity, splats['opacity']), gs_color=put(state.gs_color, splats['color']),
                   gs_valid=put(state.gs_valid, splats['valid']), gs_count=state.gs_count + 1)


# ---------------------------------------------------------------- rasterization

def project(mean, log_scale, quat, w2c, k):
    """Gaussians [...,N,3] into cameras (w2c [...,4,4], k [...,3,3], broadcast against the Gaussians' leading
    dims) -> centre [...,2] px, camera z [...], conic [...,2,2] (inverse 2D covariance), radius [...] px (3 sigma).
    Pixel centres at integers."""
    rot_w = w2c[..., :3, :3]
    p = (rot_w @ mean[..., None])[..., 0] + w2c[..., :3, 3]
    z = p[..., 2].clamp_min(1e-6)
    fx, fy, cx, cy, s = k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2], k[..., 0, 1]
    centre = torch.stack((fx * p[..., 0] / z + s * p[..., 1] / z + cx, fy * p[..., 1] / z + cy), -1)
    rot = quat_to_matrix(F.normalize(quat, dim=-1))
    cov = rot @ torch.diag_embed(torch.exp(2 * log_scale)) @ rot.transpose(-1, -2)
    cov = rot_w @ cov @ rot_w.transpose(-1, -2)
    zero = torch.zeros_like(z)
    jac = torch.stack((torch.stack((fx / z, s / z, -(fx * p[..., 0] + s * p[..., 1]) / z ** 2), -1),
                       torch.stack((zero, fy / z, -fy * p[..., 1] / z ** 2), -1)), -2)
    cov2 = jac @ cov @ jac.transpose(-1, -2) + EPS2D * torch.eye(2, device=p.device)
    det = (cov2[..., 0, 0] * cov2[..., 1, 1] - cov2[..., 0, 1] * cov2[..., 1, 0]).clamp_min(1e-12)
    conic = torch.stack((torch.stack((cov2[..., 1, 1], -cov2[..., 0, 1]), -1),
                         torch.stack((-cov2[..., 1, 0], cov2[..., 0, 0]), -1)), -2) / det[..., None, None]
    with torch.no_grad():  # culling only (sqrt has no finite gradient at round Gaussians)
        half = .5 * (cov2[..., 0, 0] + cov2[..., 1, 1])
        radius = 3 * (half + (half.square() - det).clamp_min(0).sqrt()).sqrt()
    return centre, p[..., 2], conic, radius


def _segments(count: Tensor) -> tuple:
    """Run lengths [M] -> (owner [sum], offset within its run [sum])."""
    owner = torch.repeat_interleave(torch.arange(len(count), device=count.device), count)
    return owner, torch.arange(len(owner), device=count.device) - (count.cumsum(0) - count)[owner]


def _pairs(centre, radius, alive, pixels, size) -> tuple:
    """No grad. Cameras C with centres [C,N,2], radii [C,N], alive [C,N] and pixels [C,R,2] in images of size
    (H, W) -> (pixel index into C*R, Gaussian index into C*N) for every pixel inside a Gaussian's 3-sigma box.
    Gaussians list the TILE px tiles their box touches; a pixel only tests the Gaussians of its own tile."""
    c, n = radius.shape
    r = pixels.shape[1]
    h, w = size
    nx, ny = -(-w // TILE), -(-h // TILE)
    lo, hi = centre - radius[..., None], centre + radius[..., None]
    alive = alive & (hi[..., 0] >= 0) & (lo[..., 0] < w) & (hi[..., 1] >= 0) & (lo[..., 1] < h)
    g = alive.flatten().nonzero()[:, 0]
    top = torch.tensor([nx - 1, ny - 1], device=g.device)
    t0 = (lo.flatten(0, 1)[g] / TILE).floor().long().clamp_min(0).minimum(top)
    t1 = (hi.flatten(0, 1)[g] / TILE).floor().long().clamp_min(0).minimum(top)
    span = t1 - t0 + 1
    gi, o = _segments(span[:, 0] * span[:, 1])
    tile = (t0[gi, 1] + o // span[gi, 0]) * nx + t0[gi, 0] + o % span[gi, 0]
    key, order = ((g[gi] // n) * (nx * ny) + tile).sort()
    owner = g[gi[order]]                                                     # Gaussian of each sorted entry
    px = pixels.flatten(0, 1)
    pt = (px / TILE).floor().long().clamp_min(0).minimum(top)
    pkey = torch.arange(c, device=g.device).repeat_interleave(r) * (nx * ny) + pt[:, 1] * nx + pt[:, 0]
    first = torch.searchsorted(key, pkey)
    count = torch.searchsorted(key, pkey, right=True) - first
    total = count.cumsum(0)
    cuts = torch.searchsorted(total, torch.arange(1, int(total[-1]) // PAIR_BUDGET + 1, device=g.device)
                              * PAIR_BUDGET).tolist() if len(total) else []
    pix, gs = [], []
    for a, z in zip([0] + cuts, cuts + [len(count)]):  # pixel ranges of ~PAIR_BUDGET candidates
        pi, o = _segments(count[a:z])
        pi = pi + a
        cand = owner[first[pi] + o]
        hit = ((px[pi] - centre.flatten(0, 1)[cand]).abs() <= radius.flatten()[cand][:, None]).all(-1)
        pix.append(pi[hit]), gs.append(cand[hit])
    return torch.cat(pix), torch.cat(gs)


def composite_batch(mean, log_scale, quat, opacity, color, valid, w2c, k, pixels, size, dense: int,
                    camera_valid=None):
    """Clips B with Gaussians [B,N,...] (world) seen by cameras w2c [B,V,4,4], k [B,V,3,3] at pixels [B,V,R,2] of
    images of size (H, W) -> rgb [B,V,R,3], alpha [B,V,R], camera z [B,V,R] (expected depth of the drawn mass, 0
    without a surface), dense [B,V,R,dense] (detached weights). Each pixel takes the Gaussians whose 3-sigma box
    holds it, front to back; only those pairs, and the Gaussians in them, enter the graph."""
    b, v, r, _ = pixels.shape
    n = mean.shape[1]
    c = b * v
    w2c, k = w2c.reshape(c, 4, 4), k.reshape(c, 3, 3)
    with torch.no_grad():
        pix, gs = [], []
        step = max(1, FOOTPRINT_BUDGET // max(1, n))
        for c0 in range(0, c, step):
            cams = torch.arange(c0, min(c, c0 + step), device=mean.device)
            clip = cams // v
            centre, z, _, radius = project(mean[clip], log_scale[clip], quat[clip], w2c[cams, None], k[cams, None])
            alive = valid[clip] & (z > NEAR)
            if camera_valid is not None:
                alive = alive & camera_valid.reshape(c)[cams, None]
            p, g = _pairs(centre, radius, alive, pixels.reshape(c, r, 2)[cams], size)
            pix.append(p + c0 * r), gs.append(g + c0 * n)
        pix, gs = torch.cat(pix), torch.cat(gs)
        drawn, which = torch.unique(gs, return_inverse=True)                       # (camera, Gaussian) pairs
    cam, gi = drawn // n, drawn % n
    clip = cam // v
    centre, z, conic, _ = project(mean[clip, gi], log_scale[clip, gi], quat[clip, gi], w2c[cam], k[cam])
    with torch.no_grad():  # pixel by pixel, front to back
        order = z[which].argsort(stable=True)
        order = order[pix[order].argsort(stable=True)]
    pix, which = pix[order], which[order]
    d = pixels.reshape(c * r, 2)[pix] - centre[which]
    q = conic[which]
    power = -.5 * (q[:, 0, 0] * d[:, 0] ** 2 + q[:, 1, 1] * d[:, 1] ** 2) - q[:, 0, 1] * d[:, 0] * d[:, 1]
    alpha = (torch.sigmoid(opacity[clip, gi])[which] * torch.exp(power.clamp_max(0))).clamp_max(ALPHA_MAX)
    alpha = torch.where((alpha >= ALPHA_MIN) & (power <= 0), alpha, 0.)
    # transmittance before each pair: a running sum of log(1 - alpha) restarted per pixel (f64 over all pairs)
    log_keep = torch.log1p(-alpha).double()
    before = log_keep.cumsum(0) - log_keep
    count = torch.bincount(pix, minlength=c * r)
    weight = alpha * torch.exp(before - before[(count.cumsum(0) - count)[pix]]).float()
    col = color[clip, gi][which]

    def total(x):
        return x.new_zeros(c * r, *x.shape[1:]).index_add(0, pix, x).view(b, v, r, *x.shape[1:])

    acc = total(weight)
    rgb = total(weight[:, None] * col[:, :3])
    depth = total(weight * z[which])
    depth = torch.where(acc > HIT_OPACITY, depth / acc.clamp_min(HIT_OPACITY), 0.)
    feat = total(weight.detach()[:, None] * col[:, 3:]) if dense else None
    return rgb, acc, depth, feat


def composite_torch(mean, log_scale, quat, opacity, color, valid, w2c, k, pixels, dense: int):
    """One camera (w2c [4,4], k [3,3]) on pixels [R,2]: composite_batch with one clip and one camera -> rgb [R,3],
    alpha [R], camera z [R], dense [R,dense]."""
    size = (int(pixels[:, 1].max()) + 1, int(pixels[:, 0].max()) + 1) if len(pixels) else (1, 1)
    out = composite_batch(*(x[None] for x in (mean, log_scale, quat, opacity, color, valid)), w2c[None, None],
                          k[None, None], pixels[None, None], size, dense)
    return tuple(None if x is None else x[0, 0] for x in out)


def composite_gsplat(mean, log_scale, quat, opacity, color, valid, w2c, k, height, width, dense: int):
    """One clip, all its cameras (w2c [V,4,4], k [V,3,3]) as whole images with gsplat -> rgb [V,H,W,3],
    alpha [V,H,W], camera z [V,H,W], dense [V,H,W,dense] (second pass with geometry detached)."""
    gsplat = _gsplat()
    ks = k.clone()
    ks[:, 0, 2] += .5   # gsplat samples pixel x at x + 0.5
    ks[:, 1, 2] += .5
    keep = valid.nonzero().squeeze(-1)  # padded rows hold zero quaternions, which gsplat turns into NaN
    if not keep.numel():
        v = w2c.shape[0]
        return (w2c.new_zeros(v, height, width, 3), w2c.new_zeros(v, height, width),
                w2c.new_zeros(v, height, width), w2c.new_zeros(v, height, width, dense) if dense else None)
    mean, log_scale, quat, color = mean[keep], log_scale[keep], quat[keep], color[keep]
    opac = torch.sigmoid(opacity[keep])
    args = dict(viewmats=w2c, Ks=ks, width=width, height=height, near_plane=NEAR, eps2d=EPS2D, packed=False)
    img, alpha, _ = gsplat.rasterization(mean, F.normalize(quat, dim=-1), log_scale.exp(), opac,
                                         color[:, :3].float(), render_mode='RGB+ED', **args)
    acc = alpha[..., 0]
    z = torch.where(acc > HIT_OPACITY, img[..., 3], 0.)
    feat = None
    if dense:
        feat, _, _ = gsplat.rasterization(mean.detach(), F.normalize(quat.detach(), dim=-1), log_scale.detach().exp(),
                                          opac.detach(), color[:, 3:].float(), render_mode='RGB', **args)
    return img[..., :3], acc, z, feat


class SplatRenderer(nn.Module):
    """Writes Gaussians into the state on observed steps and renders them at camera pixels."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head = SplatHead(cfg)

    def write(self, state: V6State, obs: Observation, enc) -> V6State:
        return write_ring(state, self.head(state, obs, enc), self.cfg)

    def forward(self, state: V6State, intrinsics: Tensor, cam_to_rig: Tensor, pixels: Tensor, size, rays,
                z_scale: Tensor, background, camera_valid=None) -> RenderOutput:
        """pixels [B,V,R,2] (integer pixel centres) of cameras intrinsics [B,V,3,3], cam_to_rig [B,V,4,4] at the
        state's rig pose; size (H, W); rays, z_scale from model.camera_rays for the same pixels; background(dirs)
        -> sky colour (+ dense) for world directions [B,R,3]; camera_valid [B,V] (None: all), the rest draw no
        Gaussians. RenderOutput over the flattened V*R pixels."""
        cfg = self.cfg
        b, v, r, _ = pixels.shape
        dense = cfg.dense_dim
        with no_autocast(pixels):
            rig = torch.eye(4, device=pixels.device).expand(b, 4, 4).clone()
            rig[:, :3, :3] = quat_to_matrix(state.rig_orientation.float())
            rig[:, :3, 3] = state.rig_position.float()
            w2c = torch.linalg.inv(rig[:, None] @ cam_to_rig.float())                   # [B,V,4,4]
            k = intrinsics.float()
            g = (state.gs_mean.float(), state.gs_log_scale.float(), state.gs_quat.float(), state.gs_opacity.float(),
                 state.gs_color.float(), state.gs_valid)
            if backend(cfg, pixels.device) == 'gsplat':
                h, w = size
                rgb, acc, z, feat = [], [], [], []
                for i in range(b):
                    img, a, zz, f = composite_gsplat(*(x[i] for x in g), w2c[i], k[i], h, w, dense)
                    flat = (pixels[i, ..., 1].round().long().clamp(0, h - 1) * w
                            + pixels[i, ..., 0].round().long().clamp(0, w - 1))             # [V,R]
                    take = lambda x: x.flatten(1, 2).gather(1, flat.view(v, r, *([1] * (x.dim() - 3)))
                                                             .expand(v, r, *x.shape[3:]))
                    rgb.append(take(img)), acc.append(take(a)), z.append(take(zz))
                    feat.append(take(f) if dense else None)
                rgb, acc, z = torch.stack(rgb), torch.stack(acc), torch.stack(z)
                feat = torch.stack(feat) if dense else None
            else:
                rgb, acc, z, feat = composite_batch(*g, w2c, k, pixels, size, dense, camera_valid)
            rgb, acc, z = rgb.reshape(b, v * r, 3), acc.reshape(b, v * r), z.reshape(b, v * r)
            sky = background(rays.direction.float()).float()                             # [B,V*R,3 (+ dense)]
            empty = (1 - acc).clamp(0, 1)
            color = rgb + empty[..., None] * sky[..., :3]
            feature = None
            if dense:
                feature = feat.reshape(b, v * r, dense) + empty.detach()[..., None] * sky[..., 3:]
            depth = torch.where(acc > HIT_OPACITY, z / z_scale.clamp_min(1e-6), 0.)     # along the ray
        k_slots = state.status.shape[1]
        weights = torch.cat((acc.new_zeros(b, v * r, k_slots), acc.new_ones(b, v * r, 1)), -1)
        return RenderOutput(rgb=color, depth=depth, opacity=acc, surface_slot=torch.full_like(z, -1, dtype=torch.long),
                            surface_canon=acc.new_zeros(b, v * r, 3),
                            status=torch.stack((acc, torch.zeros_like(acc), empty), -1), weights=weights,
                            sample_t=depth[..., None], sample_w=acc[..., None], feature=feature)
