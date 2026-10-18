"""Sparse box-limited field (R5): shape from C_obj, pose and deformation from D, h only in colour.

Each ray reads only the slots whose box (from D, inflated by the position std) it crosses, samples
inside those segments, maps samples to the slot's canonical frame and queries a cross-attention
field over that slot's C_obj tokens. Left-over transmittance takes a colour from the background
tokens. Reads only the state; geometry and compositing run in fp32.
"""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import V6Config
from .contracts import RayBatch, RenderOutput, V6State
from .geometry import quat_to_matrix

MIN_HALF_EXTENT = 0.1   # m, box floor per axis
FIELD_MAX_VALUES = 1 << 28
# ROCm attention kernels put the batch in a grid dimension capped at 65535: batch 32 x 1536 rays x 2 slots
# (98k) fails with hipErrorInvalidValue, so field attention runs in row chunks below that cap.
SDPA_MAX_ROWS = 1 << 15
KNN = 4                 # anchors used by the inverse deformation
ATLAS_EXTENT = 4.0      # m, atlas: 8^3 cells over [-extent, extent)^3, cell = x + 8y + 64z, LSB first (v5 layout)
HIT_OPACITY = 1e-4      # below this a ray has no surface
POSE_STD_SCALE = 0.5    # m, position std at which pose certainty halves
SEEN_AGE = 1e-3         # s, a slot counts as seen now when t - last_seen <= this


def fourier(x: Tensor, bands: int, period: float = 16.) -> Tensor:
    """[..., C] -> [..., C * (1 + 2 bands)], periods period / 2^i."""
    freq = (2 * math.pi / period) * 2. ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = (x[..., None] * freq).flatten(-2)
    return torch.cat((x, angles.sin(), angles.cos()), -1)


def gather_slots(x: Tensor, idx: Tensor) -> Tensor:
    """x [B,K,...], idx [B,...] slot indices -> [B,..., ...]."""
    b, k = x.shape[:2]
    offset = torch.arange(b, device=idx.device).view(b, *([1] * (idx.ndim - 1))) * k
    return x.flatten(0, 1)[idx + offset]


def _heads(x: Tensor, heads: int) -> Tensor:
    return x.unflatten(-1, (heads, -1)).transpose(-2, -3)


def _merge(x: Tensor) -> Tensor:
    return x.transpose(-2, -3).flatten(-2)


def _sdpa_rows(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    if q.shape[0] <= SDPA_MAX_ROWS:
        return F.scaled_dot_product_attention(q, k, v)
    return torch.cat([F.scaled_dot_product_attention(q[i:i + SDPA_MAX_ROWS], k[i:i + SDPA_MAX_ROWS], v[i:i + SDPA_MAX_ROWS])
                      for i in range(0, q.shape[0], SDPA_MAX_ROWS)])


def _no_autocast(t: Tensor):
    return torch.autocast(device_type=t.device.type, enabled=False)


class _Field(nn.Module):
    """Canonical point -> (density, colour), cross-attending to one slot's C_obj tokens."""

    def __init__(self, cfg: V6Config, layers: int = 2):
        super().__init__()
        w, c = cfg.decoder_width, cfg.latent_width
        self.heads = max(1, w // 64)
        self.bands = cfg.fourier_bands
        self.query = nn.Sequential(nn.Linear(3 * (1 + 2 * cfg.fourier_bands), w), nn.SiLU(), nn.Linear(w, w))
        self.token_norm = nn.LayerNorm(c)
        self.kv = nn.ModuleList(nn.Linear(c, 2 * w) for _ in range(layers))
        self.q_norm = nn.ModuleList(nn.LayerNorm(w) for _ in range(layers))
        self.q_proj = nn.ModuleList(nn.Linear(w, w) for _ in range(layers))
        self.o_proj = nn.ModuleList(nn.Linear(w, w) for _ in range(layers))
        self.ffn = nn.ModuleList(nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 2 * w), nn.SiLU(), nn.Linear(2 * w, w))
                                 for _ in range(layers))
        self.out_norm = nn.LayerNorm(w)
        self.density = nn.Linear(w, 1)
        nn.init.constant_(self.density.bias, -2.0)
        # colour = MLP(feature + h + object-frame direction); density never sees h
        self.color_feat = nn.Linear(w, w)
        self.color_h = nn.Linear(c, w)
        self.color_dir = nn.Linear(3, w)
        self.color_out = nn.Sequential(nn.SiLU(), nn.Linear(w, 3))

    def slot_cache(self, c_obj: Tensor, h: Tensor):
        """Per-slot K/V [B,K,6,2w] per layer and h feature [B,K,w]; computed once per forward."""
        tokens = self.token_norm(c_obj.float())
        return [kv(tokens) for kv in self.kv], self.color_h(h.float())

    def forward(self, canon: Tensor, dirs: Tensor, h_feat: Tensor, *kv: Tensor):
        # canon [N,S,3], dirs [N,3], h_feat [N,w], kv per layer [N,6,2w]
        q = self.query(fourier(canon, self.bands))
        for i, slot_kv in enumerate(kv):
            k, v = slot_kv.chunk(2, -1)
            qh = _heads(self.q_proj[i](self.q_norm[i](q)), self.heads)
            kh, vh = _heads(k, self.heads), _heads(v, self.heads)
            q = q + self.o_proj[i](_merge(_sdpa_rows(qh, kh.to(qh.dtype), vh.to(qh.dtype))))
            q = q + self.ffn[i](q)
        feat = self.out_norm(q)
        sigma = F.softplus(self.density(feat).squeeze(-1))
        color = self.color_feat(feat) + (h_feat + self.color_dir(dirs))[:, None]
        return sigma, torch.sigmoid(self.color_out(color))


class _Background(nn.Module):
    """Ray direction (rig frame) cross-attends to the background tokens -> colour. No density."""

    def __init__(self, cfg: V6Config, width: int = 128, heads: int = 2, bands: int = 4):
        super().__init__()
        self.heads, self.bands = heads, bands
        self.norm = nn.LayerNorm(cfg.latent_width)
        self.kv = nn.Linear(cfg.latent_width, 2 * width)
        self.query = nn.Sequential(nn.Linear(3 * (1 + 2 * bands), width), nn.SiLU(), nn.Linear(width, width))
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 3))

    def tokens(self, background: Tensor) -> Tensor:
        return self.kv(self.norm(background.float()))

    def forward(self, kv: Tensor, dirs: Tensor) -> Tensor:
        # kv [B,G,2w], dirs [B,R,3] unit, rig frame
        q = self.query(fourier(dirs, self.bands, period=2 * math.pi))
        k, v = kv.chunk(2, -1)
        qh = _heads(q, self.heads)
        read = _merge(_sdpa_rows(qh, _heads(k, self.heads).to(qh.dtype), _heads(v, self.heads).to(qh.dtype)))
        return torch.sigmoid(self.out(torch.cat((read, q), -1)))


class V6Decoder(nn.Module):
    def __init__(self, cfg: V6Config, checkpoint_field: bool = True):
        super().__init__()
        self.cfg = cfg
        self.field = _Field(cfg)
        self.background = _Background(cfg)
        self.checkpoint_field = checkpoint_field

    # ---- geometry from D (fp32) ----
    def slot_geometry(self, state: V6State) -> dict:
        """Per-slot pose and soft box in the object frame. Box = anchor bounds, floored, inflated by std."""
        cfg = self.cfg
        with _no_autocast(state.position):
            rot = quat_to_matrix(state.orientation.float())               # object -> world
            anchors = state.anchor_canon.float() + state.offset.float()  # deformed, object frame
            lo, hi = anchors.amin(-2), anchors.amax(-2)
            half = ((hi - lo) / 2).clamp_min(MIN_HALF_EXTENT)
            # position std along object axes: diag(R^T Sigma_p R); sampling choice only, no gradient
            cov = state.cov_pv[..., :3, :3].float().detach()
            var = torch.einsum('bkji,bkjl,bkli->bki', rot.detach(), cov, rot.detach())
            half = half + cfg.box_sigma_inflation * var.clamp_min(0).sqrt()
            return dict(rot=rot, pos=state.position.float(), anchors=anchors, offset=state.offset.float(),
                        centre=(lo + hi) / 2, half=half, alloc=state.allocated)

    def _to_object(self, geo: dict, origin: Tensor, direction: Tensor):
        o = torch.einsum('brkj,bkji->brki', origin[:, :, None] - geo['pos'][:, None], geo['rot'])
        d = torch.einsum('brj,bkji->brki', direction, geo['rot'])
        return o, d

    def _slab(self, geo: dict, o: Tensor, d: Tensor, near: Tensor, far: Tensor):
        """Ray vs expanded box (inflated + soft margin), per slot: tmin, tmax, hit [B,R,K]."""
        margin = max(self.cfg.box_soft_margin, 0.)
        half = (geo['half'] + margin)[:, None]
        centre = geo['centre'][:, None]
        inv = 1 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9).copysign(d), d)
        t0, t1 = (centre - half - o) * inv, (centre + half - o) * inv
        tmin = torch.minimum(t0, t1).amax(-1).maximum(near[..., None])
        tmax = torch.maximum(t0, t1).amin(-1).minimum(far[..., None])
        hit = (tmax > tmin) & geo['alloc'][:, None]
        return tmin, tmax, hit

    def slot_hits(self, state: V6State, rays: RayBatch) -> Tensor:
        """[B,R,K] bool: ray crosses the slot's expanded box (the only slots it can read)."""
        geo = self.slot_geometry(state)
        with _no_autocast(rays.origin):
            o, d = self._to_object(geo, rays.origin.float(), rays.direction.float())
            return self._slab(geo, o, d, rays.near_far[..., 0].float(), rays.near_far[..., 1].float())[2]

    # ---- field and background ----
    def _field(self, canon, dirs, h_feat, kv):
        if self.checkpoint_field and self.training and torch.is_grad_enabled():
            return checkpoint(self.field, canon, dirs, h_feat, *kv, use_reentrant=False)
        return self.field(canon, dirs, h_feat, *kv)

    def field_at(self, state: V6State, slots: Tensor, points: Tensor, directions: Tensor | None = None):
        """Field at canonical points [B,Q,3] of slots [B,Q] -> sigma [B,Q], rgb [B,Q,3]. Reads c_obj, h."""
        kv, h_feat = self.field.slot_cache(state.c_obj, state.h)
        if directions is None:
            directions = torch.zeros_like(points)
            directions[..., 2] = 1
        b, q = slots.shape
        sigma, rgb = self.field(points.float().reshape(b * q, 1, 3), directions.float().reshape(b * q, 3),
                                gather_slots(h_feat, slots).flatten(0, 1),
                                *[gather_slots(x, slots).flatten(0, 1) for x in kv])
        return sigma.float().reshape(b, q), rgb.float().reshape(b, q, 3)

    def background_rgb(self, state: V6State, direction: Tensor) -> Tensor:
        """Background colour for world directions [B,R,3]; direction is read in the rig frame."""
        with _no_autocast(direction):
            rig = quat_to_matrix(state.rig_orientation.float())
            dirs = torch.einsum('brj,bji->bri', direction.float(), rig)
        return self.background(self.background.tokens(state.background), dirs).float()

    # ---- render ----
    def forward(self, state: V6State, rays: RayBatch, chunk: int = 4096) -> RenderOutput:
        geo = self.slot_geometry(state)
        kv, h_feat = self.field.slot_cache(state.c_obj, state.h)
        bg_kv = self.background.tokens(state.background)
        with _no_autocast(rays.origin):
            rig = quat_to_matrix(state.rig_orientation.float())
        # Field calls see B * chunk * slots-per-ray * samples * width values; past ~2^31 bytes ROCm kernels fail
        # (hipErrorInvalidValue at batch 32), so the ray chunk shrinks with the batch.
        per_ray = rays.origin.shape[0] * min(self.cfg.max_slots_per_ray, self.cfg.slots) * self.cfg.samples_per_segment \
            * self.cfg.decoder_width
        chunk = max(64, min(chunk, FIELD_MAX_VALUES // per_ray))
        parts = []
        for start in range(0, rays.origin.shape[1], chunk):
            sl = slice(start, start + chunk)
            parts.append(self._render_chunk(geo, kv, h_feat, bg_kv, rig, rays.origin[:, sl], rays.direction[:, sl],
                                            rays.near_far[:, sl]))
        out = {key: torch.cat([p[key] for p in parts], 1) for key in parts[0]}
        out['status'] = self.status(state, out['opacity'], out['surface_slot'], out['surface_canon'])
        return RenderOutput(**out)

    def _render_chunk(self, geo, kv, h_feat, bg_kv, rig, origin, direction, near_far):
        cfg = self.cfg
        b, r = origin.shape[:2]
        k = geo['rot'].shape[1]
        m, s = min(cfg.max_slots_per_ray, k), cfg.samples_per_segment
        with _no_autocast(origin):
            origin, direction = origin.float(), direction.float()
            near, far = near_far[..., 0].float(), near_far[..., 1].float()
            o, d = self._to_object(geo, origin, direction)
            tmin, tmax, hit = self._slab(geo, o, d, near, far)
            # keep the m nearest crossed slots
            key, idx = torch.where(hit, tmin, torch.full_like(tmin, math.inf)).topk(m, -1, largest=False)
            ok = torch.isfinite(key)
            t0 = torch.where(ok, tmin.gather(-1, idx), far[..., None])
            t1 = torch.where(ok, tmax.gather(-1, idx), far[..., None])
            o_sel = o.gather(2, idx[..., None].expand(-1, -1, -1, 3))
            d_sel = d.gather(2, idx[..., None].expand(-1, -1, -1, 3))
            # stratified samples per segment
            u = torch.rand(b, r, m, s, device=origin.device) if self.training else torch.full((b, r, m, s), .5, device=origin.device)
            t = t0[..., None] + (t1 - t0)[..., None] * (torch.arange(s, device=origin.device) + u) / s
            delta = ((t1 - t0) / s)[..., None].expand_as(t)
            x = o_sel[..., None, :] + t[..., None] * d_sel[..., None, :]          # [B,R,m,S,3] object frame
            # soft box: 1 inside the inflated box, smoothstep to 0 across the margin
            margin = max(cfg.box_soft_margin, 1e-6)
            excess = (x - gather_slots(geo['centre'], idx)[..., None, :]).abs() - gather_slots(geo['half'], idx)[..., None, :]
            ramp = (1 - excess / margin).clamp(0, 1)
            occ = (ramp * ramp * (3 - 2 * ramp)).prod(-1)
            # remove deformation: inverse-distance mean of the k nearest anchors' offsets
            anchors = gather_slots(geo['anchors'], idx)                            # [B,R,m,A,3]
            offsets = gather_slots(geo['offset'], idx)
            dist2 = (x[..., :, None, :] - anchors[..., None, :, :]).square().sum(-1)  # [B,R,m,S,A]
            d2, nn_idx = dist2.topk(min(KNN, dist2.shape[-1]), -1, largest=False)
            wgt = 1 / (d2 + 1e-6)
            wgt = wgt / wgt.sum(-1, keepdim=True)
            near_off = offsets[..., None, :, :].expand(*x.shape[:-1], -1, 3).gather(-2, nn_idx[..., None].expand(*nn_idx.shape, 3))
            canon = x - (wgt[..., None] * near_off).sum(-2)
            dir_rig = torch.einsum('brj,bji->bri', direction, rig)
        # field over the selected slots' C_obj (may run under autocast)
        sigma, rgb = self._field(canon.flatten(0, 2), d_sel.flatten(0, 2), gather_slots(h_feat, idx).flatten(0, 2),
                                 [gather_slots(x_, idx).flatten(0, 2) for x_ in kv])
        bg = self.background(bg_kv, dir_rig)
        with _no_autocast(origin):
            sigma = sigma.float().reshape(b, r, m, s) * occ * ok[..., None]
            rgb, bg = rgb.float().reshape(b, r, m, s, 3), bg.float()
            # composite all samples sorted by depth
            t_all, order = t.flatten(-2).sort(dim=-1, stable=True)
            take = lambda v: v.gather(-1, order)
            sigma, delta = take(sigma.flatten(-2)), take(delta.flatten(-2))
            slot_id = take(idx[..., None].expand(-1, -1, -1, s).flatten(-2))
            rgb = rgb.flatten(2, 3).gather(2, order[..., None].expand(-1, -1, -1, 3))
            canon = canon.flatten(2, 3).gather(2, order[..., None].expand(-1, -1, -1, 3))
            optical = sigma * delta
            total = optical.cumsum(-1)
            weight = torch.exp(-(total - optical)) * -torch.expm1(-optical)
            residual = torch.exp(-total[..., -1])
            opacity = -torch.expm1(-total[..., -1])
            color = (weight[..., None] * rgb).sum(-2) + residual[..., None] * bg
            hit_ray = opacity > HIT_OPACITY
            depth = torch.where(hit_ray, (weight * t_all).sum(-1) / opacity.clamp_min(HIT_OPACITY), torch.zeros_like(opacity))
            # first surface: accumulated weight crosses 0.5, else the max-weight sample
            crossed = weight.cumsum(-1) >= .5
            first = torch.where(crossed.any(-1), crossed.float().argmax(-1), weight.argmax(-1))
            surface_slot = torch.where(hit_ray, slot_id.gather(-1, first[..., None]).squeeze(-1), torch.full_like(first, -1))
            surface_canon = canon.gather(2, first[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
            surface_canon = torch.where(hit_ray[..., None], surface_canon, torch.zeros_like(surface_canon))
            per_slot = torch.zeros(b, r, k, device=origin.device).scatter_add(-1, slot_id, weight)
            weights = torch.cat((per_slot, residual[..., None]), -1)
        return dict(rgb=color, depth=depth, opacity=opacity, surface_slot=surface_slot,
                    surface_canon=surface_canon, weights=weights)

    # ---- seen / inferred / unknown (rule-based, no gradient) ----
    @torch.no_grad()
    def status(self, state: V6State, opacity: Tensor, surface_slot: Tensor, surface_canon: Tensor) -> Tensor:
        """seen = opacity * atlas bit * seen-now; inferred = opacity * atlas bit * not-now * pose and
        event trust; unknown = rest (no surface, unseen surface, large std, low event confidence)."""
        with _no_autocast(opacity):
            opacity = opacity.float()
            has = surface_slot >= 0
            slot = surface_slot.clamp_min(0)
            bit = atlas_bit(state, slot, surface_canon.float()) * has
            age = gather_slots((state.t[:, None] - state.last_seen).float(), slot)
            now = (age <= SEEN_AGE).float()
            var = state.cov_pv[..., :3, :3].float().diagonal(dim1=-2, dim2=-1).mean(-1)
            pose_trust = 1 / (1 + gather_slots(var.clamp_min(0), slot) / POSE_STD_SCALE ** 2)
            evt_trust = .5 + .5 * gather_slots(state.evt_meta[..., 0].float(), slot).clamp(0, 1)
            seen = opacity * bit * now
            inferred = opacity * bit * (1 - now) * pose_trust * evt_trust
            return torch.stack((seen, inferred, 1 - seen - inferred), -1)


def atlas_bit(state: V6State, slot: Tensor, canon: Tensor) -> Tensor:
    """Atlas bit at the anchor nearest to canonical points canon [B,R,3] of slots [B,R] -> [B,R] float."""
    anchors = gather_slots(state.anchor_canon.float(), slot)                 # [B,R,A,3]
    nearest = (anchors - canon[..., None, :]).square().sum(-1).argmin(-1)
    p = anchors.gather(-2, nearest[..., None, None].expand(-1, -1, 1, 3)).squeeze(-2)
    inside = ((p >= -ATLAS_EXTENT) & (p < ATLAS_EXTENT)).all(-1)
    xyz = ((p + ATLAS_EXTENT) * (4 / ATLAS_EXTENT)).floor().long().clamp(0, 7)
    cell = xyz[..., 0] + 8 * xyz[..., 1] + 64 * xyz[..., 2]
    byte = gather_slots(state.atlas, slot).gather(-1, (cell // 8)[..., None]).squeeze(-1).long()
    return (((byte >> (cell % 8)) & 1).bool() & inside).float()
