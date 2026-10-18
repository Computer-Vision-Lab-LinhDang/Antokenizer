"""v6 trunk: one transformer over evidence and state tokens; C/D permissions live in the attention mask.

Token layout: [evidence V*N | K slots x (C_obj x c, C_evt, D_pose, D_neo x n, h) | camera rows Vmax,
background G, global]. Measured values (token points, anchor_meas) come from evidence only: the prior may
route tokens (ownership, anchor assignment) but never moves a measured value.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .contracts import FREE, EncoderOutput, Observation, TrunkOutput, V6State
from .geometry import quat_to_matrix
from .rays import fourier_log_depth, no_autocast, plucker

C_OBJ, C_EVT, D_POSE, D_NEO, H_TOK = range(5)
MEAS_MASS_MIN = 0.5    # token-weight mass for a valid anchor measurement
GEO_D2_MAX = 25.0      # bound on the normalised squared distance of the geometric ownership prior
NO_RIG_LOG_R = 20.0    # rig measurement log-variance when no camera is routed
ASSIGN_DIM = 64
OWN_DIM = 64
LOG_Z_CLAMP = (-5.0, 6.0)
BOX_MIN_TOKENS = 4     # box teacher: evidence tokens inside a box before it may seed a slot
BOX_MARGIN = (1.1, .25)  # box teacher: half-extent scale and pad (m) for the inside test
BOX_CLAIM_M = 1.5      # box teacher: an allocated slot within max(this, half box diagonal) already holds the box


def _anchor_grid(a: int):
    """Box-frame anchor grid (x, y, z) with x * y * z == a: the box teacher's token -> anchor routing."""
    gz = 2 if a % 2 == 0 else 1
    rest = a // gz
    gy = math.isqrt(rest)
    while rest % gy:
        gy -= 1
    return rest // gy, gy, gz


def _mlp(i, o, hidden=None):
    return nn.Sequential(nn.Linear(i, hidden or o), nn.GELU(), nn.Linear(hidden or o, o))


def _symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def _log_diag(cov):
    return torch.diagonal(cov.float(), dim1=-2, dim2=-1).clamp_min(1e-12).log()


def _soft_floor(x, floor):
    return floor + F.softplus(x - floor)


def _param(*shape):
    return nn.Parameter(torch.randn(*shape) * .02)


def _cam_features(rot9, trans, k4, hw):
    """Calibration row features: rotation, symlog translation, normalised intrinsics, log size."""
    hw = hw.clamp_min(1.)
    h, w = hw.unbind(-1)
    fx, fy, cx, cy = k4.unbind(-1)
    return torch.cat((rot9, _symlog(trans), torch.stack((fx / w, fy / h, cx / w, cy / h), -1), (hw / 100).log()), -1)


class TokenPoints(NamedTuple):
    points: Tensor     # [B,V,N,3] rig frame, fp32, evidence only
    log_z: Tensor      # [B,V,N] selected log depth (camera z)
    log_sigma: Tensor  # [B,V,N] selected log sigma
    prob: Tensor       # [B,V,N,D] hypothesis selection probabilities
    valid: Tensor      # [B,V,N] token usable (patch and camera valid)
    features: Tensor   # [B,V,N,F] per-token evidence inputs: hypotheses, measured tag, source flag


# Guard: keep one masked SDPA call under 2^28 scores (B*H*T*T); very large calls hit hipErrorInvalidValue on ROCm.
SDPA_MAX_SCORES = 1 << 28


def _sdpa_batched(q, k, v, mask):
    b, h, t, _ = q.shape
    step = max(1, SDPA_MAX_SCORES // (h * t * k.shape[2]))
    if b <= step:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.)
    return torch.cat([F.scaled_dot_product_attention(q[i:i + step], k[i:i + step], v[i:i + step],
                                                     attn_mask=mask[i:i + step] if mask.shape[0] > 1 else mask,
                                                     scale=1.) for i in range(0, b, step)])


class _Block(nn.Module):
    """Pre-norm attention + MLP. QK-norm with a learned per-head logit scale, capped."""

    def __init__(self, width, heads, scale_max):
        super().__init__()
        self.heads, self.scale_max = heads, scale_max
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.log_scale = nn.Parameter(torch.full((heads, 1, 1), math.log(10.)))
        self.mlp = _mlp(width, width, 4 * width)

    def forward(self, x, mask):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q = F.normalize(q, dim=-1) * self.log_scale.exp().clamp(max=self.scale_max).to(q.dtype)
        k = F.normalize(k, dim=-1)
        if mask.dtype != torch.bool:
            mask = mask.to(q.dtype)
        y = _sdpa_batched(q, k, v, mask)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        return x + self.mlp(self.norm2(x))


class _OwnHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.q, self.kd, self.kc = (nn.Linear(width, OWN_DIM) for _ in range(3))


class V6Trunk(nn.Module):
    """forward(state, enc, obs) -> TrunkOutput; state is the prior already drifted to obs.t."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False
        w, lw, e = cfg.trunk_width, cfg.latent_width, cfg.encoder_width
        d, nb, apt, a, tps = cfg.depth_hypotheses, cfg.fourier_bands, cfg.anchors_per_token, cfg.anchors, cfg.tokens_per_slot
        self.ev_dim = d * (2 * nb + 4) + 2 * nb + 2 + 1
        # evidence-only hypothesis selection (never sees state tokens); zero init = encoder's own logits
        self.select = _mlp(e + self.ev_dim, d, 128)
        nn.init.zeros_(self.select[-1].weight)
        nn.init.zeros_(self.select[-1].bias)
        self.ev_proj, self.ev_feat, self.ev_ray = nn.Linear(e, w), nn.Linear(self.ev_dim, w), _mlp(6, w)
        # slot tokens
        self.cobj_in, self.cobj_sum, self.cobj_pos = nn.Linear(lw, w), nn.Linear(4, w), _param(cfg.c_obj_tokens, w)
        self.evt_in = nn.Linear(lw + 2, w)
        self.pose_in, self.status_emb = _mlp(31, w), nn.Embedding(3, w)
        self.neo_in, self.neo_pos = _mlp(apt * 31, w), _param(cfg.neo_tokens, w)
        self.h_in = nn.Linear(lw, w)
        self.role_emb, self.empty_tok, self.empty_slot = _param(tps, w), _param(tps, w), _param(cfg.slots, w)
        # scene tokens
        self.cam_in = _mlp(33, w)
        self.bg_in, self.bg_pos = nn.Linear(lw + 1, w), _param(cfg.background_tokens, w)
        self.global_in = nn.Linear(lw + 4, w)
        self.kind_emb = _param(4, w)  # evidence, camera row, background, global
        # trunk
        self.blocks = nn.ModuleList(_Block(w, cfg.trunk_heads, cfg.logit_scale_max) for _ in range(cfg.trunk_layers))
        self.norm = nn.LayerNorm(w)
        layers = sorted({min(max(int(i), 1), cfg.trunk_layers) for i in cfg.ownership_layers}) or [cfg.trunk_layers]
        self.own_layers = tuple(layers)
        self.own_heads = nn.ModuleList(_OwnHead(w) for _ in layers)
        self.trash_key = _param(OWN_DIM)
        self.own_log_scale = nn.Parameter(torch.tensor(math.log(10.)))
        self.free_bias, self.trash_bias = nn.Parameter(torch.tensor(-2.)), nn.Parameter(torch.tensor(0.))
        self.geo_weight = nn.Parameter(torch.tensor(math.log(math.e - 1)))  # softplus -> 1
        # token -> anchor assignment
        self.assign_q, self.assign_key_tok = nn.Linear(w, ASSIGN_DIM), nn.Linear(w, apt * ASSIGN_DIM)
        self.assign_key_state, self.assign_emb = nn.Linear(cfg.descriptor_dim + 3, ASSIGN_DIM), _param(a, ASSIGN_DIM)
        self.assign_log_scale = nn.Parameter(torch.tensor(math.log(10.)))
        # heads
        self.anchor_tok, self.anchor_stats, self.anchor_out = nn.Linear(w, apt * 32), nn.Linear(3, 32), nn.Linear(32, 5)
        self.cobj_out, self.h_out = nn.Linear(w, lw), nn.Linear(w, lw)
        self.desc_tok, self.desc_ev = nn.Linear(w, apt * cfg.descriptor_dim), nn.Linear(w, cfg.descriptor_dim)
        self.aff_q, self.aff_k, self.aff_emb = nn.Linear(w, ASSIGN_DIM), nn.Linear(w, apt * ASSIGN_DIM), _param(a, ASSIGN_DIM)
        self.bg_out, self.cam_out = nn.Linear(w, lw + 1), nn.Linear(w, 12)
        self.spawn_head, self.canon_head = nn.Linear(w, 1), nn.Linear(w, apt * 3)
        role = [C_OBJ] * cfg.c_obj_tokens + [C_EVT, D_POSE] + [D_NEO] * cfg.neo_tokens + [H_TOK]
        self.register_buffer('slot_role', torch.tensor(role), persistent=False)

    # ---- routing -------------------------------------------------------------------------------------------
    def route_cameras(self, state: V6State, obs: Observation) -> Tensor:
        """State camera row per input camera [B,V] by rig key; new keys take unused rows in key order, -1 if none."""
        keys, valid, rows_key = obs.camera_key.cpu(), obs.camera_valid.cpu(), state.camera_key.cpu()
        out = torch.full(keys.shape, -1, dtype=torch.int64)
        for b in range(keys.shape[0]):
            live = [int(k) for k, ok in zip(keys[b], valid[b]) if ok]
            if len(live) != len(set(live)):
                raise ValueError('duplicate camera_key among valid cameras')
            lookup = {int(k): r for r, k in enumerate(rows_key[b]) if k >= 0}
            free = [r for r, k in enumerate(rows_key[b]) if k < 0]
            for v in sorted(range(keys.shape[1]), key=lambda i: int(keys[b, i])):
                key = int(keys[b, v])
                if key not in lookup and valid[b, v] and key >= 0 and free:
                    lookup[key] = free.pop(0)
                out[b, v] = lookup.get(key, -1)
        return out.to(obs.camera_key.device)

    # ---- evidence-only measurement path -------------------------------------------------------------------
    def token_points(self, enc: EncoderOutput, obs: Observation) -> TokenPoints:
        """Per-token 3D point in the rig frame. Reads only evidence and calibration, never the state."""
        cfg = self.cfg
        b, v, n, _ = enc.tokens.shape
        valid = enc.valid & obs.camera_valid[:, :, None]
        with no_autocast(enc.tokens):
            hyp = torch.where(valid[..., None, None], enc.depth_hyp.float(), 0.)
            du, dv, lz, ls, lg = hyp.unbind(-1)
            hf = torch.cat((fourier_log_depth(lz, cfg.fourier_bands), ls[..., None], lg.softmax(-1)[..., None],
                            du[..., None] / cfg.patch_size, dv[..., None] / cfg.patch_size), -1).flatten(-2)
            mv = (enc.measured_valid & valid).float()[..., None]
            md = torch.where(mv > 0, enc.measured_depth.float(), 0.)
            tag = torch.cat((fourier_log_depth(md[..., 0], cfg.fourier_bands) * mv, md[..., 1:2] * mv, mv), -1)
            src = obs.has_metric_depth.float()[:, None, None, None].expand(b, v, n, 1)
            feats = torch.where(valid[..., None], torch.cat((hf, tag, src), -1), 0.)
        delta = self.select(torch.cat((enc.tokens.to(feats.dtype), feats), -1)).float()
        with no_autocast(enc.tokens):
            prob = (lg + delta).softmax(-1)
            idx = prob.argmax(-1, keepdim=True)

            def pick(x):  # hard value, soft (straight-through) gradient
                soft = (prob * x).sum(-1)
                return x.gather(-1, idx).squeeze(-1) + soft - soft.detach()

            log_z, log_sigma, sel_du, sel_dv = pick(lz), pick(ls), pick(du), pick(dv)
            k = obs.intrinsics.float()
            fx, fy = k[..., 0, 0, None], k[..., 1, 1, None]
            fx = torch.where(fx.abs() > 1e-6, fx, torch.ones_like(fx))
            fy = torch.where(fy.abs() > 1e-6, fy, torch.ones_like(fy))
            ray = torch.where(valid[..., None], enc.ray_cam.float(), enc.ray_cam.new_tensor([0., 0., 1.]))
            rz = ray[..., 2].clamp_min(1e-6)
            # ray through patch centre + (du, dv) px, scaled so its camera z equals the selected depth
            z = log_z.clamp(*LOG_Z_CLAMP).exp()
            p_cam = torch.stack((ray[..., 0] / rz + sel_du / fx, ray[..., 1] / rz + sel_dv / fy, torch.ones_like(z)), -1) * z[..., None]
            t = obs.cam_to_rig.float()
            p_rig = torch.einsum('bvij,bvnj->bvni', t[..., :3, :3], p_cam) + t[:, :, None, :3, 3]
            p_rig = torch.where(valid[..., None], p_rig, 0.)
        return TokenPoints(p_rig, log_z, log_sigma, prob, valid, feats)

    @staticmethod
    def measure_anchors(points: Tensor, weight: Tensor):
        """points [B,Ne,3] (evidence only), weight [B,Ne,K,A] routing. -> anchor_meas [B,K,A,3], valid, mass."""
        with no_autocast(points):
            w = weight.float()
            mass = w.sum(1)
            meas = torch.einsum('bnka,bnc->bkac', w, points.float()) / mass.clamp_min(1e-6)[..., None]
            valid = mass >= MEAS_MASS_MIN
            return torch.where(valid[..., None], meas, 0.), valid, mass

    # ---- token assembly ------------------------------------------------------------------------------------
    def _slot_tokens(self, state, obs, alloc):
        cfg = self.cfg
        b, k = alloc.shape
        with no_autocast(state.c_obj):
            lam = state.evidence.float()
            bits = (state.atlas[..., None] >> torch.arange(8, device=alloc.device, dtype=torch.uint8)) & 1
            summary = torch.stack((torch.log1p(lam).mean(-1), torch.log1p(lam).amax(-1), (lam > 0).float().mean(-1),
                                   bits.flatten(-2).float().mean(-1)), -1)
            rot = quat_to_matrix(state.orientation.float())[..., :2].flatten(-2)
            since = torch.log1p((obs.t[:, None] - state.last_seen).float().clamp_min(0))
            pose = torch.cat((_symlog(state.position.float()), _symlog(state.velocity.float()), rot,
                              state.angular_velocity.float(), .1 * _log_diag(state.cov_pv), .1 * _log_diag(state.cov_rot),
                              .1 * _log_diag(state.cov_offset).mean((2, 3)), since[..., None], state.exist_logit.float()[..., None]), -1)
            neo = torch.cat((state.anchor_canon.float(), state.anchor_desc.float(), state.offset.float(),
                             state.offset_velocity.float(), .1 * _log_diag(state.cov_offset).flatten(-2)), -1)
            neo = neo.reshape(b, k, cfg.neo_tokens, -1)
        tok = torch.cat((
            self.cobj_in(state.c_obj.float()) + self.cobj_sum(summary)[:, :, None] + self.cobj_pos,
            self.evt_in(torch.cat((state.c_evt.float(), state.evt_meta.float()), -1))[:, :, None],
            (self.pose_in(pose) + self.status_emb(state.status.long()))[:, :, None],
            self.neo_in(neo) + self.neo_pos,
            self.h_in(state.h.float())[:, :, None]), 2) + self.role_emb
        # free slots never read their state: learned empty queries, distinct per slot
        empty = self.empty_tok + self.empty_slot[:, None]
        return torch.where(alloc[:, :, None, None], tok, empty.to(tok.dtype))

    def _scene_tokens(self, state, enc, obs, onehot, row_has):
        cfg = self.cfg
        with no_autocast(state.camera):
            t = obs.cam_to_rig.float()
            k = obs.intrinsics.float()
            hw = torch.tensor([obs.images.shape[-2], obs.images.shape[-1]], device=t.device, dtype=torch.float32)
            obs_cal = _cam_features(t[..., :3, :3].flatten(-2), t[..., :3, 3],
                                    torch.stack((k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2]), -1),
                                    hw.expand(*t.shape[:2], 2))
            cm = enc.camera_motion.float()
            motion = torch.cat((torch.einsum('bvij,bvj->bvi', t[..., :3, :3], cm[..., :3]),
                                torch.einsum('bvij,bvj->bvi', t[..., :3, :3], cm[..., 3:6]), cm[..., 6:]), -1)
            obs_row = torch.einsum('bvr,bvf->brf', onehot, torch.cat((obs_cal, motion), -1))
            sc = state.camera.float()
            top = sc[..., :12].reshape(*sc.shape[:2], 3, 4)
            st_cal = _cam_features(top[..., :3].flatten(-2), top[..., 3], sc[..., 12:16], sc[..., 16:18])
            cal = torch.where(row_has[..., None], obs_row[..., :18], st_cal)
            dt = torch.log1p(obs.tube_dt.float().clamp_min(0))
            cam_feat = torch.cat((cal, obs_row[..., 18:], row_has.float()[..., None], state.camera_valid.float()[..., None],
                                  dt[:, None, None].expand(*row_has.shape, 1)), -1)
            glob = torch.stack((state.metric_scale.float().clamp_min(1e-6).log(), state.metric_valid.float(),
                                obs.has_metric_depth.float(), dt), -1)
            bg = torch.cat((state.background.float(), torch.log1p(state.background_evidence.float())[..., None]), -1)
        return torch.cat((self.cam_in(cam_feat) + self.kind_emb[1],
                          self.bg_in(bg) + self.bg_pos + self.kind_emb[2],
                          (self.global_in(torch.cat((state.global_token.float(), glob), -1)) + self.kind_emb[3])[:, None]), 1)

    # ---- attention mask ------------------------------------------------------------------------------------
    def _mask(self, ev_valid, ev_row, alloc, row_used):
        """Boolean [B,T,T] (query, key), same in every layer."""
        cfg = self.cfg
        b, ne = ev_valid.shape
        k, tps, vm, g = cfg.slots, cfg.tokens_per_slot, cfg.max_cameras, cfg.background_tokens
        s = k * tps
        dev = ev_valid.device
        role = self.slot_role.repeat(k)                                        # [S]
        sid = torch.arange(k, device=dev).repeat_interleave(tps)               # [S]
        alloc_t = alloc[:, sid]                                                # [B,S]

        def full(ev=None, slot=None, cam=None, bg=None, gl=None):
            z = lambda n: torch.zeros(b, n, dtype=torch.bool, device=dev)
            return torch.cat((z(ne) if ev is None else ev, z(s) if slot is None else slot, z(vm) if cam is None else cam,
                              z(g) if bg is None else bg, z(1) if gl is None else gl), 1)

        ones = lambda n: torch.ones(b, n, dtype=torch.bool, device=dev)
        ev_k = full(ev=ev_valid)
        slot_noh = full(slot=alloc_t & (role != H_TOK))
        dpose = full(slot=alloc_t & (role == D_POSE))
        dtok = full(slot=alloc_t & ((role == D_POSE) | (role == D_NEO)))
        cam_k = full(cam=row_used)
        bggl = full(bg=ones(g), gl=ones(1))
        scene = cam_k | bggl

        ev_rows = (ev_k | slot_noh | scene)[:, None].expand(b, ne, -1)
        # slot rows
        t = ne + s + vm + g + 1
        tok_sid = torch.full((t,), -1, device=dev)
        tok_sid[ne:ne + s] = sid
        tok_role = torch.full((t,), -1, device=dev)
        tok_role[ne:ne + s] = role
        same = sid[:, None] == tok_sid[None]                                   # [S,T]
        own_noh = same & (tok_role[None] != H_TOK)
        base = (ev_k | cam_k)[:, None]
        r = role[None, :, None]
        row_cobj = base | own_noh | (dpose[:, None] & ~same)
        row_d = base | own_noh | (dtok[:, None] & ~same)
        row_h = base | same
        rows = torch.where(r == C_OBJ, row_cobj, torch.where(r == H_TOK, row_h, row_d))
        rows = torch.where(alloc_t[..., None], rows, (ev_k | scene)[:, None] | own_noh)
        evt_self = torch.zeros(s, t, dtype=torch.bool, device=dev)
        evt_self[torch.arange(s, device=dev), ne + torch.arange(s, device=dev)] = True
        slot_rows = torch.where(r == C_EVT, evt_self, rows)                    # C_evt: dummy query, output dropped
        # camera rows: own camera's evidence, D_pose of all slots, background, global
        ev_row_k = torch.cat((ev_row, torch.full((b, t - ne), -1, device=dev, dtype=ev_row.dtype)), 1)
        own_cam = ev_k[:, None] & (ev_row_k[:, None] == torch.arange(vm, device=dev)[None, :, None])
        cam_rows = own_cam | (dpose | bggl)[:, None]
        scene_rows = (ev_k | dpose | scene)[:, None].expand(b, g + 1, -1)
        return torch.cat((ev_rows, slot_rows, cam_rows, scene_rows), 1)

    def _biased_mask(self, allowed, log_own):
        """Float mask: log-ownership on evidence <-> slot logits, -inf where not allowed."""
        cfg = self.cfg
        ne, s = log_own.shape[1], cfg.slots * cfg.tokens_per_slot
        sid = torch.arange(cfg.slots, device=allowed.device).repeat_interleave(cfg.tokens_per_slot)
        lo = log_own[:, :, sid]                                                # [B,Ne,S]
        bias = torch.zeros(allowed.shape, device=allowed.device, dtype=lo.dtype)
        bias[:, :ne, ne:ne + s] = lo
        bias[:, ne:ne + s, :ne] = lo.transpose(1, 2)
        return bias.masked_fill(~allowed, float('-inf'))

    # ---- ownership -----------------------------------------------------------------------------------------
    def _geo_prior(self, state, points, alloc):
        """Bounded geometric log-prior [B,Ne,K]: token point vs nearest drifted anchor (routing only)."""
        b, ne, _ = points.shape
        with no_autocast(points):
            rwr = quat_to_matrix(state.rig_orientation.float())
            p_w = torch.einsum('bij,bnj->bni', rwr, points.float()) + state.rig_position.float()[:, None]
            local = state.anchor_canon.float() + state.offset.float()
            anchors = torch.einsum('bkij,bkaj->bkai', quat_to_matrix(state.orientation.float()), local) + state.position.float()[:, :, None]
            d2 = torch.cdist(p_w, anchors.flatten(1, 2)).square().view(b, ne, *anchors.shape[1:3]).amin(-1)
            sigma2 = torch.diagonal(state.cov_pv.float()[..., :3, :3], dim1=-2, dim2=-1).mean(-1).clamp_min(0) + 1.
            bias = -.5 * F.softplus(self.geo_weight) * (d2 / sigma2[:, None]).clamp(max=GEO_D2_MAX)
            return torch.where(alloc[:, None], bias, self.free_bias)

    def _ownership(self, head, x, prior, ev_valid, ne):
        cfg = self.cfg
        b = x.shape[0]
        xn = head.norm(x)
        xs = xn[:, ne:ne + cfg.slots * cfg.tokens_per_slot].view(b, cfg.slots, cfg.tokens_per_slot, -1)
        q = F.normalize(head.q(xn[:, :ne]).float(), dim=-1)
        keys = head.kd(xs[:, :, cfg.c_obj_tokens + 1]) + head.kc(xs[:, :, :cfg.c_obj_tokens].mean(2))
        keys = F.normalize(torch.cat((keys.float(), self.trash_key.float().expand(b, 1, -1)), 1), dim=-1)
        prior = torch.cat((prior, self.trash_bias.expand(b, ne, 1)), -1)
        logits = self.own_log_scale.exp().clamp(max=cfg.logit_scale_max) * torch.einsum('bnd,bkd->bnk', q, keys) + prior
        own = logits.softmax(-1)
        trash = F.one_hot(torch.tensor(cfg.slots, device=x.device), cfg.slots + 1).to(own.dtype)
        return torch.where(ev_valid[..., None], own, trash)

    # ---- box teacher (training only) -------------------------------------------------------------------------
    def _box_teacher(self, state, enc, obs, tp, own, alloc, boxes):
        """GT boxes choose which evidence tokens a slot measures, and on which anchor; values stay evidence-only.

        A token is in a box when its LiDAR-tagged or its own depth point falls inside; its anchor is its cell
        in a box-frame grid (so a seeded slot always has spread anchors). A box with an allocated slot near its
        centre is held by the nearest one; each other box with enough tokens goes (greedy) to the free slot that
        already owns most of them.
        Returns forced [B,K] bool, token slot [B,Ne] (slot holding or given the token's box, -1: none),
        token anchor [B,Ne], slot box [B,K] (box a slot holds or was given, -1: none).
        """
        k = self.cfg.slots
        b, v, n, _ = tp.points.shape
        dev = tp.points.device
        with no_autocast(tp.points):
            center, size, box_ok = boxes['center'].float(), boxes['size'].float(), boxes['valid'].bool()
            m = center.shape[1]
            forced = torch.zeros(b, k, dtype=torch.bool, device=dev)
            if m == 0:
                none = torch.full((b, v * n), -1, dtype=torch.long, device=dev)
                return forced, none, none.clamp_min(0), torch.full((b, k), -1, dtype=torch.long, device=dev)
            # rig -> world: metadata pose where known (a training teacher may use it), else the prior's
            known = obs.rig_pose_valid.bool().to(dev)
            pose = obs.rig_pose.float()
            r_wr = torch.where(known[:, None, None], pose[:, :3, :3], quat_to_matrix(state.rig_orientation.float()))
            t_wr = torch.where(known[:, None], pose[:, :3, 3], state.rig_position.float())
            c_rig = torch.einsum('bji,bmj->bmi', r_wr, center - t_wr[:, None])
            r_box = r_wr.transpose(-1, -2)[:, None] @ quat_to_matrix(boxes['quat'].float())   # box -> rig
            half = size * .5 * BOX_MARGIN[0] + BOX_MARGIN[1]

            def local(p):  # [B,Ne,3] rig -> box frame scaled to [-1, 1] inside the inflated box, [B,Ne,M,3]
                return torch.einsum('bmji,bnmj->bnmi', r_box, p[:, :, None] - c_rig[:, None]) / half[:, None]

            ray = torch.where(tp.valid[..., None], enc.ray_cam.float(), enc.ray_cam.new_tensor([0., 0., 1.]).float())
            z = enc.measured_depth.float()[..., 0].clamp(*LOG_Z_CLAMP).exp()
            t = obs.cam_to_rig.float()
            p_lid = torch.einsum('bvij,bvnj->bvni', t[..., :3, :3], ray / ray[..., 2:].clamp_min(1e-6) * z[..., None]) \
                + t[:, :, None, :3, 3]
            lid_ok = (enc.measured_valid & tp.valid).flatten(1)[..., None]
            u_lid, u_tok = local(p_lid.flatten(1, 2)), local(tp.points.float().flatten(1, 2).detach())
            in_lid = (u_lid.abs() <= 1).all(-1) & lid_ok
            hit = in_lid | (u_tok.abs() <= 1).all(-1)
            hit = hit & tp.valid.flatten(1)[..., None] & box_ok[:, None]
            hit = hit & (hit.long().cumsum(-1) == 1)                  # one box per token
            u = torch.where(in_lid[..., None], u_lid, u_tok)          # the point that put the token inside
            tok_box = torch.where(hit.any(-1), hit.float().argmax(-1), m)
            u = u.gather(2, tok_box.clamp_max(m - 1)[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
            grid = torch.tensor(_anchor_grid(self.cfg.anchors), device=dev)
            cell = ((u + 1) * .5 * grid).floor().long().clamp(min=0).minimum(grid - 1)
            token_anchor = (cell[..., 0] * grid[1] + cell[..., 1]) * grid[2] + cell[..., 2]
            count = hit.sum(1)                                       # [B,M]
            reach = torch.clamp(.5 * size.norm(dim=-1), min=BOX_CLAIM_M)
            dist = torch.cdist(state.position.float(), center)
            near = (dist < reach[:, None]) & alloc[..., None]
            claimed = near.any(1)
            # a held box with no token inside leaves its slot to the model's own routing
            holder = torch.where(claimed & box_ok & (count > 0), dist.masked_fill(~near, float('inf')).argmin(1), -1)
            eligible = box_ok & ~claimed & (count >= BOX_MIN_TOKENS)
            score = torch.einsum('bnm,bnk->bmk', hit.float(), own[..., :k].detach().float()) + 1e-3 * count[..., None]
            score = score.masked_fill(~eligible[..., None] | alloc[:, None], float('-inf'))
            slot_box = torch.full((b, k), m, dtype=torch.long, device=dev)
            rows = torch.arange(b, device=dev)
            for _ in range(min(m, k)):
                best, flat = score.flatten(1).max(-1)
                ok = torch.isfinite(best)
                if not bool(ok.any()):
                    break
                bi, ki = flat // k, flat % k
                forced[rows, ki] |= ok
                slot_box[rows, ki] = torch.where(ok, bi, slot_box[rows, ki])
                score[rows, bi] = torch.where(ok[:, None], float('-inf'), score[rows, bi])
                score[rows, :, ki] = torch.where(ok[:, None], float('-inf'), score[rows, :, ki])
            box_slot = F.pad(holder, (0, 1), value=-1)
            box_slot.scatter_(1, slot_box, torch.where(forced, torch.arange(k, device=dev), -1))
            box_slot[:, m] = -1
            # slot -> box: given boxes, else one box it holds
            held = torch.full((b, k + 1), -1, dtype=torch.long, device=dev)
            held.scatter_(1, torch.where(holder >= 0, holder, k), torch.arange(m, device=dev).expand(b, m).contiguous())
            spawn_box = torch.where(forced, slot_box, held[:, :k])
            return forced, box_slot.gather(1, tok_box), token_anchor, spawn_box

    # ---- forward -------------------------------------------------------------------------------------------
    def forward(self, state: V6State, enc: EncoderOutput, obs: Observation, boxes: dict | None = None) -> TrunkOutput:
        cfg = self.cfg
        b, v, n, _ = enc.tokens.shape
        k, a, tps, vm = cfg.slots, cfg.anchors, cfg.tokens_per_slot, cfg.max_cameras
        ne, s = v * n, k * tps
        dev = enc.tokens.device
        alloc = state.status != FREE

        rows = self.route_cameras(state, obs)
        cam_ok = obs.camera_valid & (rows >= 0)
        onehot = ((rows[:, :, None] == torch.arange(vm, device=dev)) & cam_ok[:, :, None]).float()  # [B,V,Vmax]
        row_has = onehot.sum(1) > 0
        row_used = row_has | (state.camera_key >= 0)

        tp = self.token_points(enc, obs)
        ev_valid = tp.valid.flatten(1)
        points = tp.points.flatten(1, 2)
        ev_row = torch.where(cam_ok, rows, -1)[:, :, None].expand(b, v, n).flatten(1)

        # world Plücker ray per patch from the prior rig pose: positional input only
        with no_autocast(enc.tokens):
            t = obs.cam_to_rig.float()
            rwr = quat_to_matrix(state.rig_orientation.float())[:, None]
            origin = torch.einsum('bvij,bvj->bvi', rwr, t[..., :3, 3]) + state.rig_position.float()[:, None]
            direction = torch.einsum('bvij,bvnj->bvni', rwr @ t[..., :3, :3], enc.ray_cam.float())
            ray = torch.where(tp.valid[..., None], plucker(origin[:, :, None], direction), 0.)
        x_ev = self.ev_proj(enc.tokens) + self.ev_feat(tp.features) + self.ev_ray(ray) + self.kind_emb[0]
        x_slot = self._slot_tokens(state, obs, alloc)
        x_scene = self._scene_tokens(state, enc, obs, onehot, row_has)
        x = torch.cat((x_ev.flatten(1, 2), x_slot.flatten(1, 2), x_scene.to(x_ev.dtype)), 1)

        allowed = self._mask(ev_valid, ev_row, alloc, row_used)
        mask = allowed[:, None]
        evt_rows = torch.zeros(x.shape[1], dtype=torch.bool, device=dev)
        evt_rows[ne:ne + s] = self.slot_role.repeat(k) == C_EVT
        prior = self._geo_prior(state, points, alloc)
        owns = []
        for i, block in enumerate(self.blocks, 1):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                y = checkpoint(block, x, mask, use_reentrant=False)
            else:
                y = block(x, mask)
            x = torch.where(evt_rows[None, :, None], x, y)  # C_evt is key/value only
            if i in self.own_layers:
                own = self._ownership(self.own_heads[self.own_layers.index(i)], x, prior, ev_valid, ne)
                owns.append(own)
                # detached: ownership learns through the measurement weights, not through attention bias
                mask = self._biased_mask(allowed, own[..., :k].detach().clamp_min(1e-4).log())[:, None]
        own = owns[-1]

        xf = self.norm(x)
        xe = xf[:, :ne]
        xs = xf[:, ne:ne + s].view(b, k, tps, -1)
        c, dp = cfg.c_obj_tokens, cfg.c_obj_tokens + 1
        neo = xs[:, :, dp + 1:dp + 1 + cfg.neo_tokens]                          # [B,K,neo,Wt]
        x_cam, x_bg = xf[:, ne + s:ne + s + vm], xf[:, ne + s + vm:ne + s + vm + cfg.background_tokens]

        # token -> anchor assignment inside each slot (routing), then evidence-only anchor values
        state_key = self.assign_key_state(torch.cat((state.anchor_desc.float(), state.anchor_canon.float()), -1))
        akey = self.assign_key_tok(neo).view(b, k, a, -1) + self.assign_emb + torch.where(alloc[..., None, None], state_key, 0.)
        cos = torch.einsum('bnd,bkad->bnka', F.normalize(self.assign_q(xe).float(), dim=-1), F.normalize(akey.float(), dim=-1))
        assign = (self.assign_log_scale.exp().clamp(max=cfg.logit_scale_max) * cos).softmax(-1)
        weight = own[..., :k, None] * assign * ev_valid[:, :, None, None]
        forced = token_slot = spawn_box = assign_nll = None
        if boxes is not None:
            forced, token_slot, token_anchor, spawn_box = self._box_teacher(state, enc, obs, tp, own, alloc, boxes)
            # a slot given or holding a box measures exactly the box's tokens, each on its box-grid anchor
            mine = (token_slot[..., None] == torch.arange(k, device=dev)) & ev_valid[..., None]       # [B,Ne,K]
            hot = F.one_hot(token_anchor, a).to(weight.dtype)                                         # [B,Ne,A]
            weight = torch.where((spawn_box >= 0)[:, None, :, None], mine[..., None] * hot[:, :, None], weight)
            picked = assign.gather(2, token_slot.clamp_min(0)[:, :, None, None].expand(-1, -1, 1, a)).squeeze(2)
            assign_nll = -picked.gather(-1, token_anchor[..., None]).squeeze(-1).clamp_min(1e-6).log()
        anchor_meas, meas_valid, mass = self.measure_anchors(points, weight)

        with no_autocast(points):
            wn = weight / mass.clamp_min(1e-6)[:, None]
            stats = torch.stack((torch.log(mass + 1e-3), torch.einsum('bnka,bn->bka', wn, tp.log_sigma.flatten(1)),
                                 torch.einsum('bnka,bn->bka', wn, enc.measured_valid.flatten(1).float())), -1)
        head = self.anchor_out(F.gelu(self.anchor_tok(neo).view(b, k, a, 32) + self.anchor_stats(stats))).float()
        pooled = torch.einsum('bnka,bnc->bkac', wn.to(xe.dtype), xe)
        desc_prop = self.desc_tok(neo).view(b, k, a, -1) + self.desc_ev(pooled)
        aq = self.aff_q(xs[:, :, :c])
        ak = self.aff_k(neo).view(b, k, a, -1) + self.aff_emb
        affinity = (torch.einsum('bkid,bkad->bkia', aq.float(), ak.float()) / math.sqrt(ASSIGN_DIM)).softmax(-1)
        bg = self.bg_out(x_bg).float()

        # rig measurement: precision-weighted fusion over routed camera rows
        with no_autocast(points):
            cam = self.cam_out(x_cam).float()
            cam_log_r = _soft_floor(cam[..., 6:], cfg.log_r_floor)
            prec = row_has.float()[..., None] * torch.exp(-cam_log_r)
            total = prec.sum(1)
            rig_meas = (prec * cam[..., :6]).sum(1) / total.clamp_min(1e-12)
            rig_log_r = torch.where(total > 0, -total.clamp_min(1e-12).log(), NO_RIG_LOG_R).clamp_min(cfg.log_r_floor)

        return TrunkOutput(
            ownership=own, ownership_by_layer=tuple(owns),
            anchor_meas=anchor_meas, anchor_meas_valid=meas_valid,
            anchor_log_r=_soft_floor(head[..., :3], cfg.log_r_floor),
            visibility=torch.sigmoid(head[..., 3]), depth_conf=torch.sigmoid(head[..., 4]),
            c_obj_prop=self.cobj_out(xs[:, :, :c]).float(), desc_prop=desc_prop.float(),
            h_prop=self.h_out(xs[:, :, -1]).float(), anchor_affinity=affinity,
            rig_meas=rig_meas, rig_log_r=rig_log_r,
            background_prop=bg[..., :-1], background_vis=torch.sigmoid(bg[..., -1]),
            spawn_logit=self.spawn_head(xs[:, :, dp]).float().squeeze(-1),
            spawn_canon=self.canon_head(neo).view(b, k, a, 3).float(),
            camera_rows=rows, spawn_forced=forced, forced_slot=token_slot, spawn_box=spawn_box,
            assign_nll=assign_nll)
