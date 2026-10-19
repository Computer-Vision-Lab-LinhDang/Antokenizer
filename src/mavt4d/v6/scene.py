"""Static scene map (C): world-frame BEV cells written from evidence-only token points, read by the decoder.

Cell (iy, ix) covers world x in [x_min + ix * cell, x_min + (ix + 1) * cell) and the same for y; values sit at
cell centres and are read bilinearly (grid_sample with align_corners=False). The world is the rig frame at the
clip's first step, so the grid follows the clip, not the city.

Write (same Lambda-gain rule as C_obj and the background tokens), after the lifecycle so it sees what slots
claim: a token point writes with the ownership mass no allocated slot holds, times its depth confidence, and
not at all inside an allocated slot's box, without measured depth in its patch (sky, beyond the LiDAR) or when
the box teacher routes it to a slot. Its weight spreads over a Gaussian as wide as its patch's footprint (one
token per 16 px patch is sparser than 1 m cells past ~15 m), peak 1 at the nearest cell; a cell's lambda is the
sum, its proposal the weighted mean of the point features. The last two channels of a cell are not learned:
they hold the mean height and mean squared height of its points (Z_NORM units), so the decoder reads density
against the cell's own surface height. Cells with lambda 0 and every cell under predict() stay bit-identical.
Read: density is gated by 1 - exp(-GATE * Lambda), so cells never seen stay transparent (unknown, not invented).
"""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import quat_to_matrix

POINT_SIGMA = 0.2      # std of log z at which a token point's depth confidence halves
LAMBDA_MIN = 1e-3      # per-cell lambda below this does not write
GATE = 4.0             # density gate 1 - exp(-GATE * Lambda): one confident point opens its cell
GATE_SEEN = .05        # gate above which a point counts as seen (losses supervise depth there)
Z_PERIOD, Z_BANDS = 64.0, 4      # m, absolute height encoding (coarse: which storey)
REL_PERIOD, REL_BANDS = 8.0, 6   # m, height relative to the cell's mean point height: periods 8 m .. 0.25 m
REL_STD_MIN = .1       # m, floor on the cell's height spread
Z_NORM = 8.0           # m, scale of the stored height moments
DENSITY_BIAS = -3.0    # softplus(-3) ~ 0.05 / m in an open cell
SPLAT_RADIUS = 2       # cells: footprint splats cover a (2 r + 1)^2 window
SPLAT_STD = (.5, 1.)   # cells: footprint std bounds (half the patch footprint, clamped)
BOX_MARGIN = .25       # m, pad on an allocated slot's anchor box: points inside never reach the map


def fourier1(x: Tensor, period: float, bands: int) -> Tensor:
    """[...] -> [..., 1 + 2 bands]: x / period and sin/cos at periods period / 2^i."""
    freq = (2 * math.pi / period) * 2. ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angle = x[..., None] * freq
    return torch.cat((x[..., None] / period, angle.sin(), angle.cos()), -1)


def cell_coords(cfg, xy: Tensor) -> Tensor:
    """World xy [...,2] -> continuous cell coordinates (x, y), cell centres at integers."""
    lo = xy.new_tensor([cfg.scene_x_min, cfg.scene_y_min])
    return (xy - lo) / cfg.scene_cell - .5


def splat_footprint(cfg, xy: Tensor, value: Tensor, weight: Tensor, footprint: Tensor):
    """Gaussian splat of value [B,P,C] with weight [B,P] at world xy [B,P,2] over the (2 r + 1)^2 cells around
    each point, std = half its footprint [B,P] (m) clamped to SPLAT_STD cells, peak 1 at the nearest cell; cells off
    the grid drop. -> (sum of weight * value [B,NY*NX,C], sum of weight [B,NY*NX])."""
    ny, nx = cfg.scene_cells_y, cfg.scene_cells_x
    b, p, c = value.shape
    pos = cell_coords(cfg, xy.float())
    base = pos.round()
    std = (.5 * footprint.float() / cfg.scene_cell).clamp(*SPLAT_STD)
    r = torch.arange(-SPLAT_RADIUS, SPLAT_RADIUS + 1, device=xy.device, dtype=pos.dtype)
    off = torch.stack(torch.meshgrid(r, r, indexing='ij'), -1).view(-1, 2)              # [W,2] (dx, dy)
    cell = base[:, :, None] + off                                                       # [B,P,W,2]
    gauss = torch.exp(-(cell - pos[:, :, None]).square().sum(-1) / (2 * std[..., None] ** 2))
    gauss = gauss / gauss.amax(-1, keepdim=True)
    inside = ((cell >= 0) & (cell < cell.new_tensor([nx, ny]))).all(-1)
    w = torch.where(inside, weight.float()[..., None] * gauss, 0.).flatten(1)          # [B,P*W]
    idx = (cell[..., 1].clamp(0, ny - 1) * nx + cell[..., 0].clamp(0, nx - 1)).long().flatten(1)
    acc_w = w.new_zeros(b, ny * nx).scatter_add(1, idx, w)
    vals = (w.view(b, p, -1)[..., None] * value.float()[:, :, None]).flatten(1, 2)      # [B,P*W,C]
    acc_v = w.new_zeros(b, ny * nx, c).scatter_add(1, idx[..., None].expand(-1, -1, c), vals)
    return acc_v, acc_w


def rig_to_world(state, obs):
    """Rig -> world (R [B,3,3], t [B,3]) for placing measurements: metadata pose where known, else the state's."""
    known = obs.rig_pose_valid.bool()
    pose = obs.rig_pose.float()
    r = torch.where(known[:, None, None], pose[:, :3, :3], quat_to_matrix(state.rig_orientation.float()))
    return r, torch.where(known[:, None], pose[:, :3, 3], state.rig_position.float())


def inside_slots(state, points: Tensor, margin: float = BOX_MARGIN) -> Tensor:
    """World points [B,P,3] -> [B,P] bool: inside an allocated slot's anchor box (padded by margin)."""
    rot = quat_to_matrix(state.orientation.float())                                    # object -> world
    anchors = state.anchor_canon.float() + state.offset.float()
    lo, hi = anchors.amin(-2), anchors.amax(-2)
    local = torch.einsum('bpkj,bkji->bpki', points[:, :, None] - state.position.float()[:, None], rot)
    inside = ((local - ((lo + hi) / 2)[:, None]).abs() <= ((hi - lo) / 2)[:, None] + margin).all(-1)
    return (inside & state.allocated[:, None]).any(-1)


def patch_colours(images: Tensor, cfg) -> Tensor:
    """Last tube frame [B,V,3,H,W] -> per-patch 2x2 mean colours [B,V,N,12], centred on 0."""
    b, v, _, h, w = images.shape
    p = cfg.patch_size
    x = images.float() / 255 if images.dtype == torch.uint8 else images.float()
    pooled = F.avg_pool2d(x.flatten(0, 1), p // 2)                               # [BV,3,2Hp,2Wp]
    hp, wp = h // p, w // p
    pooled = pooled.view(b, v, 3, hp, 2, wp, 2).permute(0, 1, 3, 5, 2, 4, 6)    # [B,V,Hp,Wp,3,2,2]
    return pooled.reshape(b, v, hp * wp, 12) - .5


class SceneMeasure(nn.Module):
    """Token points (evidence only) -> per-cell proposal [B,NY,NX,Cs] and lambda [B,NY,NX]."""

    def __init__(self, cfg, hidden: int = 64):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.encoder_width)
        self.tok = nn.Linear(cfg.encoder_width, hidden)
        self.geo = nn.Linear(12 + 1 + 2 * Z_BANDS + 2, hidden)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, cfg.scene_width - 2))

    def forward(self, state, meas, obs, enc):
        """state: the posterior after the lifecycle (its slots decide which points are not scene)."""
        cfg = self.cfg
        b, v, n, _ = enc.tokens.shape
        ny, nx = cfg.scene_cells_y, cfg.scene_cells_x
        k = state.status.shape[1]
        with torch.autocast(enc.tokens.device.type, enabled=False):
            pts = meas.points.float()                                                # [B,Ne,3] rig
            r, t = rig_to_world(state, obs)
            world = torch.einsum('bij,bnj->bni', r, pts) + t[:, None]
            log_sigma = meas.point_log_sigma.float()
            valid = (enc.valid & obs.camera_valid[:, :, None]).flatten(1) & torch.isfinite(world).all(-1)
            # LiDAR stays a tag: with metric depth, only patches that have some may write (no sky floaters)
            tagged = enc.measured_valid.flatten(1) | ~obs.has_metric_depth[:, None]
            # routing and confidence decide how much a point writes; neither learns from the scene
            own = meas.ownership[..., :k].float().detach()
            scene_mass = (1 - (own * state.allocated[:, None].float()).sum(-1)).clamp(0, 1)
            conf = 1 / (1 + (log_sigma.detach().exp() / POINT_SIGMA) ** 2)
            ok = valid & tagged & (conf >= cfg.depth_conf_min) & ~inside_slots(state, world.detach())
            if meas.forced_slot is not None:  # training: the box teacher routes this token to a slot
                ok = ok & (meas.forced_slot < 0)
            weight = torch.where(ok, scene_mass * conf, 0.)
            z = world[..., 2]
            geo = torch.cat((patch_colours(obs.images[:, :, -1], cfg).flatten(1, 2), fourier1(z, Z_PERIOD, Z_BANDS),
                             log_sigma[..., None], pts.norm(dim=-1).clamp_min(1e-3).log()[..., None]), -1)
            geo = torch.where(valid[..., None], geo, 0.)
            learned = self.out(self.tok(self.norm(enc.tokens.float().flatten(1, 2))) + self.geo(geo))
            feat = torch.cat((learned, (z / Z_NORM)[..., None], (z / Z_NORM).square()[..., None]), -1)
            # a patch spans patch_size / fx radians: its footprint grows with the range from its camera
            cam = obs.cam_to_rig.float()[:, :, None, :3, 3].expand(b, v, n, 3).flatten(1, 2)
            fx = obs.intrinsics.float()[..., 0, 0].abs().clamp_min(1e-6)[:, :, None].expand(b, v, n).flatten(1)
            footprint = cfg.patch_size * (pts - cam).norm(dim=-1).detach() / fx
            acc_v, acc_w = splat_footprint(cfg, world[..., :2], feat, weight, footprint)
            prop = acc_v / acc_w.clamp_min(1e-6)[..., None]
            lam = acc_w.detach()
            lam = torch.where(obs.has_metric_depth[:, None], lam, lam.clamp_max(cfg.rgb_only_lambda_cap))
            lam = torch.where(lam >= LAMBDA_MIN, lam, 0.)
        return prop.view(b, ny, nx, -1), lam.view(b, ny, nx)


def scene_write(cfg, prior, prop: Tensor, lam: Tensor) -> dict:
    """Lambda-gain write: only cells with lambda > 0 change; Lambda is capped at lambda_max."""
    wrote = lam > 0
    evidence = torch.where(wrote, (prior.scene_evidence + lam).clamp_max(cfg.lambda_max), prior.scene_evidence)
    gain = torch.where(wrote, lam / torch.where(wrote, evidence, 1.), 0.)
    old = prior.scene.float()
    new = old + gain[..., None] * (prop - old)
    return dict(scene=torch.where(wrote[..., None], new.to(prior.scene.dtype), prior.scene), scene_evidence=evidence)


class SceneField(nn.Module):
    """World points -> (density gated by the cell evidence, colour), read from the scene map."""

    def __init__(self, cfg, width: int = 64):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(cfg.scene_width + 1 + (1 + 2 * Z_BANDS) + 1 + (1 + 2 * REL_BANDS), width)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.density = nn.Linear(width, 1)
        nn.init.constant_(self.density.bias, DENSITY_BIAS)
        self.color = nn.Linear(width, 3)

    def grid(self, state) -> Tensor:
        """[B, Cs + 2, NY, NX]: cell features (height moments last), log1p(Lambda), evidence gate."""
        ev = state.scene_evidence.float()
        g = torch.cat((state.scene.float(), torch.log1p(ev)[..., None], -torch.expm1(-GATE * ev)[..., None]), -1)
        return g.permute(0, 3, 1, 2)

    def read(self, grid: Tensor, points: Tensor) -> Tensor:
        """grid from grid(), world points [B,P,3] -> [B,P,Cs+2] bilinear (zeros off the grid)."""
        cfg = self.cfg
        pos = cell_coords(cfg, points[..., :2].float())
        size = pos.new_tensor([cfg.scene_cells_x, cfg.scene_cells_y])
        uv = (2 * pos + 1) / size - 1                                                # align_corners=False
        return F.grid_sample(grid.float(), uv[:, :, None], mode='bilinear', padding_mode='zeros',
                             align_corners=False)[..., 0].transpose(1, 2)

    def forward(self, grid: Tensor, points: Tensor):
        """grid from grid(), points [B,P,3] world -> sigma [B,P] (per metre), rgb [B,P,3]."""
        with torch.autocast(points.device.type, enabled=False):
            read = self.read(grid, points)
            gate = read[..., -1].clamp(0, 1)
            z = points[..., 2].float()
            c = self.cfg.scene_width
            mean, mean2 = read[..., c - 2] * Z_NORM, read[..., c - 1] * Z_NORM ** 2
            std = (mean2 - mean.square()).clamp_min(0).sqrt().clamp_min(REL_STD_MIN)
            dz = z - mean
            x = torch.cat((read[..., :-1], fourier1(z, Z_PERIOD, Z_BANDS), (dz / std).clamp(-10, 10)[..., None] / 10,
                           fourier1(dz, REL_PERIOD, REL_BANDS)), -1)
        h = self.mlp(self.inp(x))
        sigma = F.softplus(self.density(h).float().squeeze(-1)) * gate
        return sigma, torch.sigmoid(self.color(h).float())
