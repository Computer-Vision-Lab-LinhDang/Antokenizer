"""MAVT Grain-Prior C-D v1: conditional initialisation -> evidence refinement -> closed state.

forward(x, modality, cam) -> dict(loss, parts, diag), the interface of the v0 trainer:

  image   [B, 3, H, W]      mode image: C only (header + C = 6 180 scalars)
  video   [B, 3, T, H, W]   mode video: anchor = frame 0, targets coded in order from the previous state,
                            cameras estimated, anchor median depth 1 (gauge); T > 4 folds the queue into z_b
  threed  [B, 3, V, H, W]   mode static_views: cameras copied, M = 0, z_i = U(c_0, b_i, 1), metric depth
  world   [B, 3, T, H, W]   OmniWorld clips: per sample, calibrated_video (cameras copied, metric) with
                            probability world_given_p, else video with the given camera as a label (L_pose)

Encoder side reads the current image; below the hard bottleneck only S, shared weights and valid queries.
`decode(S)` is the public decoder: it reads a (deserialized) state alone.
"""
from __future__ import annotations

from typing import Optional

import math
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cd32.depth import cell_log_inv_depth
from ..cd32.losses import psnr
from . import geometry as g
from . import schema as sc
from .anchor import AnchorHeads, Stems, StreamC
from .config import GrainPriorConfig
from .innovation import InnovationEncoder
from .layers import mlp2
from .losses import feat_loss, huber_log_depth, pose_loss, sample_cells
from .memory import Appearance, Recurrence
from .prior import CausalPrior
from .readout import Readout, RGBHead, pool_geom, projected_radius
from .refinement import CorrRead, UpdateHead
from .trunk import Bank, PInit, StreamD

MODALITIES = ("image", "video", "threed", "world")
M_IMAGE, M_VIDEO, M_STATIC, M_CALIB = range(4)


def kappa_to_logits(kappa: torch.Tensor, cfg: GrainPriorConfig) -> torch.Tensor:
    """Inverse of geometry.kappa_from_logits (clamped inside the bounds)."""
    lo, hi = g.focal_bounds(cfg.fov_min, cfg.fov_max)
    p = ((torch.log(kappa[..., :2].float().clamp_min(1e-6)) - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
    q = ((kappa[..., 2:].float() - 0.5) / cfg.pp_range).clamp(-0.999, 0.999)
    return torch.cat([torch.log(p / (1 - p)), torch.atanh(q)], -1)


def _pearson(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    n = m.sum(-1).clamp_min(1)
    am = (a * m).sum(-1, keepdim=True) / n.unsqueeze(-1)
    bm = (b * m).sum(-1, keepdim=True) / n.unsqueeze(-1)
    cov = ((a - am) * (b - bm) * m).sum(-1)
    va, vb = (((a - am) ** 2) * m).sum(-1), (((b - bm) ** 2) * m).sum(-1)
    return cov / (va * vb).clamp_min(1e-12).sqrt()


class MAVTGrainPrior(nn.Module):
    COMPILE_REGIONS = ("Block", "ReadLayer")
    MODALITIES = MODALITIES

    def __init__(self, cfg: GrainPriorConfig, use_checkpoint: bool = False):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        self.stems = Stems(cfg)
        Dt = self.stems.teacher_dim
        self.stream_c = StreamC(cfg)
        self.anchor = AnchorHeads(cfg, Dt)
        self.beta = mlp2(cfg.d_f + g.pe_dim(2, cfg.pe_freq) + cfg.d_coarse, cfg.n_motion, cfg.mlp_hidden)
        self.bank = Bank(cfg)
        self.p_init = PInit(cfg)
        self.stream_d = StreamD(cfg)
        self.corr = CorrRead(cfg)
        self.upd = nn.ModuleList([UpdateHead(cfg, Dt) for _ in range(cfg.refine_passes)])
        self.recur = Recurrence(cfg)
        self.appear = Appearance(cfg)
        # shared readouts own their reader (query_decoder); a second, unused Readout would stall DDP
        self.readout = Readout(cfg, Dt) if cfg.readout_version == "legacy" else None
        self.rgb = RGBHead(cfg, cfg.dim if cfg.rgb_from == "hidden" else Dt)
        self.innov = InnovationEncoder(cfg, Dt)
        self.prior = CausalPrior(cfg)
        if cfg.readout_version != "legacy":
            from .shared import SharedQueryDecoder
            self.query_decoder = SharedQueryDecoder(cfg, Dt)
        self.lpips = None
        if cfg.w_lpips > 0:
            from ..dino_v0.losses import LPIPSLoss, ignore_lpips_keys
            self.lpips = LPIPSLoss()
            self.register_load_state_dict_post_hook(ignore_lpips_keys)
        for n, p in self.named_parameters():                          # T1 / T2: the prior waits; T3: only it
            if n.startswith("prior."):
                p.requires_grad_(cfg.phase == "prior")
            elif cfg.phase == "prior":
                p.requires_grad_(False)
        self.register_buffer("uq", g.cell_centres(cfg.grid), persistent=False)
        if cfg.query_grid:
            self.register_buffer("uq_rgb", g.cell_centres(cfg.query_grid), persistent=False)

    # ============================================================================== state helpers
    def derived(self, S: dict) -> dict:
        """Tensors re-derived from C alone: depth, canonical points, blend weights, gauge scale."""
        cfg = self.cfg
        d = torch.exp(-S["rho"].float())
        P = g.unproject(d, S["kappa0"], S["u"])
        x = torch.cat([S["f"].float(), g.fourier(S["u"], cfg.pe_freq), g.read_coarse(S["c0"].float(), S["u"], cfg.coarse_grid)], -1)
        beta = torch.softmax(self.beta(x).float(), -1)
        return dict(d=d, P=P, beta=beta, scale=d.median(-1).values.clamp_min(1e-6))

    def memory_step(self, z: torch.Tensor, b: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.recur(z, b, delta) if self.cfg.memory else z

    def replay(self, S: dict) -> list:
        """z at every record of the queue, from z_b and the stored b and times (static views: U(c_0, b, 1))."""
        h = S["h"]
        B = h.shape[0]
        static = h[:, sc.H_MODE] == M_STATIC
        z, tb, out = S["zb"].float(), h[:, sc.H_TB], []
        one = torch.ones(B, device=h.device)
        for j, rec in enumerate(S["recs"]):
            t = h[:, sc.H_TT + j]
            zv = self.memory_step(z, rec["b"], torch.where(static, one, t - tb))
            zs = self.memory_step(S["c0"].float(), rec["b"], one)
            zj = torch.where(static.view(B, 1, 1), zs, zv)
            out.append(zj)
            z, tb = torch.where(static.view(B, 1, 1), z, zj), torch.where(static, tb, t)
        return out

    def append_and_fold(self, S: dict, rec: dict, t: torch.Tensor) -> None:
        """Append a record at time offset t [B]; a fourth record folds the oldest into z_b (video) or evicts it
        (static views). In place; the header follows."""
        cfg = self.cfg
        h = S["h"]
        B = h.shape[0]
        if len(S["recs"]) == cfg.queue:
            old, t_old = S["recs"][0], h[:, sc.H_TT]
            static = h[:, sc.H_MODE] == M_STATIC
            zf = self.memory_step(S["zb"].float(), old["b"], t_old - h[:, sc.H_TB])
            S["zb"] = torch.where(static.view(B, 1, 1), S["zb"].float(), zf)
            h[:, sc.H_TB] = torch.where(static, h[:, sc.H_TB], t_old)
            S["recs"] = S["recs"][1:]
            h[:, sc.H_TT:sc.H_TT + cfg.queue - 1] = h[:, sc.H_TT + 1:sc.H_TT + cfg.queue].clone()
            h[:, sc.H_LEN:sc.H_LEN + cfg.queue - 1] = h[:, sc.H_LEN + 1:sc.H_LEN + cfg.queue].clone()
            h[:, sc.H_PROV:sc.H_PROV + cfg.queue - 1] = h[:, sc.H_PROV + 1:sc.H_PROV + cfg.queue].clone()
            h[:, sc.H_VIEW + 1:sc.H_VIEW + cfg.queue] = h[:, sc.H_VIEW + 2:sc.H_VIEW + 1 + cfg.queue].clone()
        j = len(S["recs"])
        S["recs"] = S["recs"] + [rec]
        h[:, sc.H_NT] = j + 1
        h[:, sc.H_TT + j] = t
        h[:, sc.H_LEN + j] = rec["L"].float()
        h[:, sc.H_PROV + j] = rec.get("prov", torch.zeros_like(t))
        h[:, sc.H_VIEW + 1 + j] = rec.get("view", torch.zeros_like(t))

    # ============================================================================== public decoder
    def encode_source(self, image, kappa=None):
        """C-only NVS encoder. Target images, depths and target cameras are not arguments.

        image is [B,3,H,W] in [-1,1]. Supplied source intrinsics establish calibrated
        metric mode; they do not supply target depth or an additional hidden scene code.
        """
        if image.ndim != 4 or image.shape[1:] != (3,self.cfg.image_size,self.cfg.image_size):
            raise ValueError('source image has incorrect shape')
        B,dev = image.shape[0],image.device
        known = torch.full((B,),kappa is not None,dtype=torch.bool,device=dev)
        mode = torch.full((B,),M_STATIC if kappa is not None else M_IMAGE,device=dev)
        return self.encode_anchor(self.stems(image),mode,known,known,kappa)[0]

    def prepare_state(self, S, record_id=None, mode='anchor'):
        from .query import prepare_state
        return prepare_state(self,S,record_id,mode)

    def read_points(self, view, query):
        if not hasattr(self,'query_decoder'): raise ValueError('legacy has no point field')
        return self.query_decoder.read_points(view,query)

    def read_rays(self, view, query, chunk=None):
        if not hasattr(self,'query_decoder'): raise ValueError('use legacy decode for legacy checkpoints')
        return self.query_decoder.read_rays(view,query,chunk)

    def decode_packet(self, packet, tasks=('rgb','depth','feature')):
        return self.query_decoder.decode(packet,tasks)

    def _decode_shared_state(self,S):
        from .query import RayQuery
        outputs = []
        for record_id in [None]+list(range(len(S['recs']))):
            view = self.prepare_state(S,record_id,'anchor' if record_id is None else 'observed')
            B = view.scale.shape[0]
            q = RayQuery(view.record_camera,self.uq.expand(B,-1,-1),self.cfg.render_near,
                         self.cfg.render_far,self.cfg.render_samples)
            low = self.decode_packet(self.read_rays(view,q),('depth','feature'))
            uv = g.cell_centres(self.cfg.image_size,self.uq.device).expand(B,-1,-1)
            high = self.decode_packet(self.read_rays(view,dataclasses.replace(q,uv=uv)),('rgb',))
            rgb = high['rgb'].transpose(1,2).reshape(B,3,self.cfg.image_size,self.cfg.image_size)
            outputs.append(dict(F=low['feature'],depth=low['depth'],rgb=rgb*2-1))
        return {k:torch.stack([o[k] for o in outputs],1) for k in outputs[0]}

    def decode_obs(self, S: dict, der: dict, z: torch.Tensor, xi: torch.Tensor, kappa: torch.Tensor,
                   M: torch.Tensor, r: Optional[torch.Tensor] = None, L: Optional[torch.Tensor] = None,
                   R_t: Optional[tuple] = None, rgb: bool = True) -> dict:
        """Observation with record (xi, kappa, M[, r, L]) and memory z -> dict(base, F_hat, ...). With
        cfg.query_grid (and rgb), a second read on that grid gives the RGB state H_hat (a_rgb, pe_rgb)."""
        cfg = self.cfg
        R, t = g.se3_exp(xi) if R_t is None else R_t
        f_t = self.appear(S["f"].float(), S["u"], z, S["c0"]) if cfg.memory else S["f"].float()
        tr = g.transport_grains(der["P"], der["beta"], M, R, t, kappa, cfg.z_eps)
        pg = pool_geom(xi, kappa, S["kappa0"], S["rho"], M, tr["valid"])
        sigma = S["sigma"]
        if cfg.read_radius_scale:                 # radius on this observation's image (1 at the anchor)
            sigma = projected_radius(S["sigma"], tr["P_can"][..., 2], tr["z"], S["kappa0"], kappa, cfg.read_radius_clip)
        grains = (f_t, tr["P_can"], S["u"], tr["u"], tr["z"], sigma, tr["valid"], der["scale"], S["c0"], z, pg)
        out = self.readout.base(*grains)
        out["tr"] = tr
        out["F_hat"], out["H_hat"] = out["F_base"], out["a"]
        if r is not None:
            out["dF"], dH = self.readout.innovation(out["a"], out["pe"], r, L, return_hidden=True)
            out["F_hat"], out["H_hat"] = out["F_base"] + out["dF"], out["a"] + dH
        if cfg.query_grid and rgb:
            fine = self.readout.base(*grains, uq=self.uq_rgb.expand(f_t.shape[0], -1, -1))
            out["a_rgb"], out["pe_rgb"] = fine["a"], fine["pe"]
            out["H_hat"] = self._rgb_state(out, r, L)
        return out

    def _rgb_state(self, out: dict, r: Optional[torch.Tensor], L: Optional[torch.Tensor]) -> torch.Tensor:
        """Decoder state on the query_grid read, plus its innovation increment when r is given."""
        a = out["a_rgb"]
        if r is None:
            return a
        uq = self.uq_rgb.expand(a.shape[0], -1, -1)
        return a + self.readout.innovation(a, out["pe_rgb"], r, L, return_hidden=True, uq=uq)[1]

    def rgb_of(self, F: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """RGB in [0, 1] of one observation: from the DINO-space feature F or the decoder state H (cfg.rgb_from)."""
        return self.rgb(H if self.cfg.rgb_from == "hidden" else F)

    def decode_anchor(self, S: dict, der: dict) -> dict:
        B = S["c0"].shape[0]
        dev = S["c0"].device
        return self.decode_obs(S, der, S["c0"].float(), torch.zeros(B, 6, device=dev), S["kappa0"],
                               torch.zeros(B, self.cfg.n_motion, 3, device=dev))

    def decode(self, S: dict) -> dict:
        """Public decoding of a closed state: anchor and every record in the queue (replayed memory, own r).
        -> dict(F [B, 1 + n_rec, N, D_t], rgb [B, 1 + n_rec, 3, H, W] in [-1, 1],
        depth [B, 1 + n_rec, N])."""
        if self.cfg.readout_version != "legacy":
            return self._decode_shared_state(S)
        der = self.derived(S)
        outs = [self.decode_anchor(S, der)]
        for rec, z in zip(S["recs"], self.replay(S)):
            outs.append(self.decode_obs(S, der, z, rec["xi"], rec["kappa"], rec["M"], rec["r"], rec["L"]))
        Fh = torch.stack([o["F_hat"] for o in outs], 1)
        Hh = torch.stack([o["H_hat"] for o in outs], 1)
        rgb = self.rgb_of(Fh.flatten(0, 1), Hh.flatten(0, 1)).view(Fh.shape[0], Fh.shape[1], 3, self.cfg.image_size,
                                                                   self.cfg.image_size)
        return dict(F=Fh, rgb=rgb * 2 - 1, depth=torch.stack([o["depth_dense"] for o in outs], 1))

    # ============================================================================== encoder
    def encode_anchor(self, st: dict, mode: torch.Tensor, known: torch.Tensor, metric: torch.Tensor,
                      kappa_cam: Optional[torch.Tensor]) -> tuple:
        cfg = self.cfg
        Hc = self.stream_c(st["Y"], st["g"])
        A = self.anchor(Hc, st["F"], st["P"])
        B = Hc.shape[0]
        kappa0 = g.kappa_from_logits(A["kappa_logits"], cfg)
        if cfg.image_fov_fixed:
            f0 = 0.5 / math.tan(math.radians(cfg.fov_init) / 2)
            k_fix = torch.tensor([f0, f0, 0.5, 0.5], device=kappa0.device, dtype=kappa0.dtype).expand_as(kappa0)
            kappa0 = torch.where((mode == M_IMAGE).view(B, 1), k_fix, kappa0)
        if kappa_cam is not None:
            kappa0 = torch.where(known.view(B, 1), kappa_cam.float(), kappa0)
        rho = torch.where(metric.view(B, 1), A["rho_raw"], A["rho_raw"] - A["rho_raw"].median(-1, keepdim=True).values)
        h = sc.new_header(cfg, B, mode, metric.long(), known.long(), Hc.device)
        S = dict(h=h, u=A["u"], rho=rho, sigma=A["sigma"], f=A["f"], c0=A["c0"], kappa0=kappa0,
                 zb=A["c0"], recs=[])
        return S, A

    def encode_target(self, S: dict, der: dict, st: dict, Xi: torch.Tensor, t_i: torch.Tensor, z_prev: torch.Tensor,
                      static: torch.Tensor, known: torch.Tensor, cam_i: Optional[tuple], L: torch.Tensor) -> dict:
        """Posterior coding of observation i from S_{i-1}. st: stems of observation i (F, P, Y, g); Xi in [0, 1];
        cam_i (xi, kappa) given cameras (used where known); static [B] bool; L [B] prefix length."""
        cfg = self.cfg
        B = Xi.shape[0]
        dev = Xi.device
        one = torch.ones(B, device=dev)
        times = [S["h"][:, sc.H_TT + j] for j in range(len(S["recs"]))]
        t_prev = times[-1] if times else torch.zeros(B, device=dev)
        delta = torch.where(static, one, t_i - t_prev)
        # ---- bank of the previous state (static views: anchor + the view's known camera)
        bt = self.bank.temporal(S, z_prev, times)
        if cam_i is not None and bool(static.any()):
            bv = self.bank.view(S, cam_i[0], cam_i[1])
            if bool(static.all()):
                bank = bv
            else:                                               # mixed batch: pad the shorter bank
                n = max(bt[0].shape[1], bv[0].shape[1])
                pad = lambda x, v: (F.pad(x, (0, 0, 0, n - x.shape[1])), F.pad(v, (0, n - v.shape[1])))
                (a0, a1), (b0, b1) = pad(*bt), pad(*bv)
                s = static.view(B, 1)
                bank = (torch.where(s.unsqueeze(-1), b0, a0), torch.where(s, b1, a1))
        else:
            bank = bt
        past = self.p_init(bank, delta)
        prev = S["recs"][-1] if S["recs"] else None
        k0 = kappa_to_logits(S["kappa0"], cfg)
        past_kz = (kappa_to_logits(prev["kappa"], cfg) if prev is not None else k0) + past["kz"]
        past_xi = (prev["xi"] if prev is not None else torch.zeros(B, 6, device=dev)) + past["xi"]
        past_M = (prev["M"] if prev is not None else 0) + past["M"]
        if cam_i is not None:
            kn = known.view(B, 1)
            past_xi = torch.where(kn, cam_i[0], past_xi)
            past_kz = torch.where(kn, kappa_to_logits(cam_i[1], cfg), past_kz)
        past_M = past_M * (~static).float().view(B, 1, 1)
        H_D, est = self.stream_d(bank, st["Y"], st["g"], dict(xi=past_xi, kz=past_kz, M=past_M, b=past["b"]))
        # ---- first estimates; known quantities overwrite
        xi0 = est["xi"]
        zeta = k0 + est["kz"]
        M = est["M"] * (~static).float().view(B, 1, 1)
        b = est["b"]
        R, t = g.se3_exp(xi0)
        if cam_i is not None:
            Rg, tg = g.se3_exp(cam_i[0])
            kn = known.view(B, 1, 1)
            R, t = torch.where(kn, Rg, R), torch.where(known.view(B, 1), tg, t)
        kz_given = kappa_to_logits(cam_i[1], cfg) if cam_i is not None else None

        def kappa_of(z_):
            k = g.kappa_from_logits(z_, cfg)
            return torch.where(known.view(B, 1), cam_i[1].float(), k) if cam_i is not None else k

        def mem(b_):
            zv = self.memory_step(z_prev, b_, delta)
            zs = self.memory_step(S["c0"].float(), b_, one)
            return torch.where(static.view(B, 1, 1), zs, zv)

        passes = []
        for ell in range(cfg.refine_passes):
            z_l = mem(b)
            xi_l = g.se3_log(R, t)
            base_l = self.decode_obs(S, der, z_l, xi_l, kappa_of(zeta), M, R_t=(R, t), rgb=False)
            tr = base_l["tr"]
            passes.append(base_l)
            e, v_bar = self.corr(S["f"], S["u"], tr["u"], tr["valid"], st["Y"], ell)
            dlt = self.upd[ell](H_D, e, v_bar, tr["u"], tr["valid"], st["F"] - base_l["F_base"])
            R2, t2 = g.compose_left(dlt["xi"], R, t)
            if cam_i is not None:
                R2, t2 = torch.where(known.view(B, 1, 1), R, R2), torch.where(known.view(B, 1), t, t2)
            R, t = R2, t2
            zeta = zeta + dlt["kz"]
            M = M + dlt["M"] * (~static).float().view(B, 1, 1)
            b = b + dlt["b"]
        xi = g.se3_log(R, t)
        if cam_i is not None:                                       # copied metadata, exactly
            xi = torch.where(known.view(B, 1), cam_i[0].float(), xi)
        kappa = kappa_of(zeta)
        z_i = mem(b)
        base = self.decode_obs(S, der, z_i, xi, kappa, M)
        X_base = self.rgb_of(base["F_base"], base["a_rgb"] if cfg.query_grid else base["a"])
        eF = st["F"] - base["F_base"].detach()
        eX = Xi - X_base.detach()
        r = self.innov(eF, eX, z_i, xi, kappa, M)
        keep = (torch.arange(cfg.n_slots, device=dev).view(1, -1) < L.view(-1, 1)).float().unsqueeze(-1)
        r = r * keep
        dF, dH = self.readout.innovation(base["a"], base["pe"], r, L, return_hidden=True)
        F_hat, H_hat = base["F_base"] + dF, (self._rgb_state(base, r, L) if cfg.query_grid else base["a"] + dH)
        rec = dict(xi=xi, M=M, kappa=kappa, b=b, r=r, L=L)
        return dict(rec=rec, z=z_i, base=base, X_base=X_base, F_hat=F_hat, H_hat=H_hat, passes=passes,
                    est0=(xi0, est["M"]))

    def sample_L(self, B: int, device) -> torch.Tensor:
        cfg = self.cfg
        if not self.training:
            return torch.full((B,), cfg.n_slots, dtype=torch.long, device=device)
        opts = torch.tensor(cfg.nested_L, device=device)
        return opts[torch.randint(len(opts), (B,), device=device)]

    # ============================================================================== one segment
    def _segment(self, frames: torch.Tensor, kind: str, cam: Optional[dict], parts: dict, diag: dict,
                 L_given: Optional[torch.Tensor] = None) -> tuple:
        """frames [B, n, 3, H, W] in [-1, 1]; kind image | video | static | world."""
        cfg = self.cfg
        B, n = frames.shape[:2]
        dev = frames.device
        S_img = cfg.image_size
        st_all = self.stems(frames.flatten(0, 1))
        st = [{k: v.view(B, n, *v.shape[1:])[:, i] for k, v in st_all.items()} for i in range(n)]
        X01 = (frames.float() + 1) / 2
        false = torch.zeros(B, dtype=torch.bool, device=dev)
        if kind == "world":                     # eval: all given if world_given_p > 0, all estimated at 0
            known = torch.rand(B, device=dev) < cfg.world_given_p if self.training else false | (cfg.world_given_p > 0)
        else:
            known = ~false if kind == "static" else false
        static = ~false if kind == "static" else false
        metric = known.clone()
        mode = torch.full((B,), {"image": M_IMAGE, "video": M_VIDEO, "static": M_STATIC, "world": M_VIDEO}[kind],
                          device=dev)
        if kind == "world":
            mode = torch.where(known, torch.full_like(mode, M_CALIB), mode)
        kappa_cam = xi_cam = None
        if cam is not None:
            kappa_cam = g.kappa_from_K(cam["K"].float(), S_img, S_img)                      # [B, n, 4]
            Tr = g.relative_pose(cam["c2w"])                                                # [B, n, 4, 4]
            xi_cam = g.se3_log(Tr[..., :3, :3], Tr[..., :3, 3])
        S, A = self.encode_anchor(st[0], mode, known, metric, None if kappa_cam is None else kappa_cam[:, 0])
        der = self.derived(S)
        # ---- gauge of the labels: metric where known, else the anchor's median GT depth
        lid = lmask = None
        s_scene = torch.ones(B, device=dev)
        if cam is not None and cam.get("depth") is not None:
            dep = cam["depth"].float()
            lid, lmask = cell_log_inv_depth(dep.flatten(0, 1), dep.flatten(0, 1) > 0, cfg.grid)
            lid, lmask = lid.view(B, n, -1), lmask.view(B, n, -1)
            med = []
            for b_ in range(B):
                v = lid[b_, 0][lmask[b_, 0]]
                med.append(torch.exp(-v.median()) if v.numel() else torch.ones((), device=dev))
            s_scene = torch.where(metric, torch.ones(B, device=dev), torch.stack(med))
        # ---- anchor decode (z = c_0, T = I, M = 0, no innovation)
        out0 = self.decode_anchor(S, der)
        Fh, Fb, Fi, Xb_list, outs = [out0["F_hat"]], [], [out0["F_init"]], [], [out0]
        Hh = [out0["H_hat"]]
        z_cur = S["c0"].float()
        tg = []
        for i in range(1, n):
            t_i = torch.full((B,), 0.0 if kind == "static" else i * cfg.frame_seconds, device=dev)
            L = L_given[:, i - 1].to(dev).long() if L_given is not None else self.sample_L(B, dev)
            cam_i = None if cam is None else (xi_cam[:, i], kappa_cam[:, i])
            o = self.encode_target(S, der, st[i], X01[:, i], t_i, z_cur, static, known, cam_i, L)
            o["rec"]["view"] = torch.full((B,), float(i), device=dev)
            self.append_and_fold(S, o["rec"], t_i)
            z_cur = torch.where(static.view(B, 1, 1), z_cur, o["z"])
            Fh.append(o["F_hat"])
            Hh.append(o["H_hat"])
            Fb.append(o["base"]["F_base"])
            Fi.append(o["base"]["F_init"])
            Xb_list.append(o["X_base"])
            outs.append(o["base"])
            tg.append(o)
        Fh, Hh = torch.stack(Fh, 1), torch.stack(Hh, 1)
        rgb = self.rgb_of(Fh.flatten(0, 1), Hh.flatten(0, 1)).view(B, n, 3, S_img, S_img)
        Ft = torch.stack([s["F"] for s in st], 1)
        # ---- L_rec on every observation (full), base and initializer on the targets, init also on the anchor
        loss = self._rec(rgb, X01, Fh, Ft, parts, diag, "rec")
        lf_init = feat_loss(torch.stack(Fi, 1), Ft, cfg.w_logn)
        parts["init_feat"] = lf_init.detach()
        loss = loss + cfg.w_init * lf_init
        if n > 1:
            Xb = torch.stack(Xb_list, 1)
            lb = self._rec(Xb, X01[:, 1:], torch.stack(Fb, 1), Ft[:, 1:], parts, diag, "base", lpips=False)
            loss = loss + cfg.w_base * lb
            Ms = torch.stack([o["rec"]["M"] for o in tg], 1)
            lm = (Ms ** 2).sum(-1).mean()
            parts["motion"] = lm.detach()
            loss = loss + cfg.w_motion * lm
        code = [S["f"].pow(2).mean(), S["c0"].pow(2).mean()]
        for o in tg:
            code += [o["rec"]["b"].pow(2).mean(), o["rec"]["r"][..., 4:].pow(2).mean()]
        lc = torch.stack(code).mean()
        parts["code"] = lc.detach()
        loss = loss + cfg.w_code * lc
        # ---- geometry labels
        if lid is not None:
            loss = loss + self._geom(S, der, outs, lid, lmask, s_scene, parts, diag)
        if cam is not None and n > 1 and kind == "world":
            loss = loss + self._pose(tg, xi_cam, known, s_scene, parts, diag)
        self._diag(S, der, A, tg, rgb, frames, X01, Ft, Xb_list, diag, kind)
        return loss, dict(S=S, rgb=rgb * 2 - 1, F_hat=Fh, F=Ft)

    def _rec(self, rgb, X01, Fh, Ft, parts, diag, name, lpips=True) -> torch.Tensor:
        cfg = self.cfg
        lf = feat_loss(Fh, Ft, cfg.w_logn)
        lx = (rgb.float() - X01).abs().mean()
        parts[f"{name}_feat"], parts[f"{name}_l1"] = lf.detach(), lx.detach()
        loss = lf + lx
        if lpips and self.lpips is not None:
            a, b = rgb.flatten(0, 1), X01.flatten(0, 1)
            k = cfg.lpips_max_frames
            if 0 < k < a.shape[0]:
                sel = torch.randperm(a.shape[0], device=a.device)[:k]
                a, b = a[sel], b[sel]
            lp = self.lpips(a * 2 - 1, b * 2 - 1)
            parts["lpips"] = lp.detach()
            loss = loss + cfg.w_lpips * lp
        return loss

    def _geom(self, S, der, outs, lid, lmask, s_scene, parts, diag) -> torch.Tensor:
        """Huber log-depth of the anchor grains and of the grain / dense depth readouts on every observation."""
        cfg = self.cfg
        B = lid.shape[0]
        gt = torch.exp(-lid) / s_scene.view(B, 1, 1)                                   # [B, n, N] in the gauge
        gk = sample_cells(lid[:, 0], S["u"], cfg.grid)
        mk = sample_cells(lmask[:, 0].float(), S["u"], cfg.grid) > 0.99
        terms = [huber_log_depth(der["d"], torch.exp(-gk) / s_scene.view(B, 1), mk, cfg.depth_delta)]
        grain_v1, grain_v2 = [], []
        for j, o in enumerate(outs):
            grain_v1.append(huber_log_depth(o["depth_grain"], gt[:, j], lmask[:, j] & o.get("support_v1", o["support"]), cfg.depth_delta))
            grain_v2.append(huber_log_depth(o["depth_grain"], gt[:, j], lmask[:, j], cfg.depth_delta))
            terms.append((grain_v1 if cfg.measurement_version == 1 else grain_v2)[-1])
            terms.append(huber_log_depth(o["depth_dense"], gt[:, j], lmask[:, j], cfg.depth_delta))
        lg = torch.stack(terms).mean()
        parts["geom"] = lg.detach()
        with torch.no_grad():
            diag["rho_gt_corr"] = _pearson(S["rho"].float(), gk, mk).mean()
            diag["depth_grain_err"] = torch.stack(terms[1::2]).mean()
            diag["depth_grain_err_v1"] = torch.stack(grain_v1).mean()
            diag["depth_grain_err_v2"] = torch.stack(grain_v2).mean()
            diag["depth_dense_err"] = torch.stack(terms[2::2]).mean()
        return cfg.w_geom * lg

    def _pose(self, tg, xi_cam, known, s_scene, parts, diag) -> torch.Tensor:
        """Estimated cameras of world clips against the given ones (the decoder never sees the latter)."""
        cfg = self.cfg
        est = ~known
        if not bool(est.any()):
            return xi_cam.sum() * 0
        ls, rot, tr = [], [], []
        for j, o in enumerate(tg):
            R, t = g.se3_exp(o["rec"]["xi"])
            Rg, tgv = g.se3_exp(xi_cam[:, j + 1])
            ls.append(pose_loss(R[est], t[est], Rg[est], tgv[est] / s_scene[est].view(-1, 1), torch.ones_like(s_scene[est])))
            with torch.no_grad():
                rot.append(g.rotation_angle_deg(R[est].transpose(-1, -2) @ Rg[est]).mean())
                tr.append((t[est] - tgv[est] / s_scene[est].view(-1, 1)).norm(dim=-1).mean())
        lp = torch.stack(ls).mean()
        parts["pose"] = lp.detach()
        diag["cam_rot_err_deg"], diag["cam_trans_err"] = torch.stack(rot).mean(), torch.stack(tr).mean()
        return cfg.w_pose * lp

    @torch.no_grad()
    def _diag(self, S, der, A, tg, rgb, frames, X01, Ft, Xb_list, diag, kind) -> None:
        cfg = self.cfg
        to11 = lambda x: x * 2 - 1
        diag["psnr"] = psnr(to11(rgb).flatten(0, 1), frames.flatten(0, 1))
        diag["psnr_anchor"] = psnr(to11(rgb[:, 0]), frames[:, 0])
        diag["numbers"] = sc.payload(cfg, S).float().mean()
        diag["fov_deg"] = torch.rad2deg(2 * torch.atan(0.5 / S["kappa0"][:, 0].clamp_min(1e-6))).mean()
        diag["sigma_mean"] = S["sigma"].mean()
        if kind != "image":
            diag["rho_std"] = S["rho"].std(-1).mean()
        if not tg:
            return
        n = rgb.shape[1]
        diag["psnr_target"] = psnr(to11(rgb[:, 1:]).flatten(0, 1), frames[:, 1:].flatten(0, 1))
        diag["psnr_base"] = psnr(to11(torch.stack(Xb_list, 1)).flatten(0, 1), frames[:, 1:].flatten(0, 1))
        diag["psnr_copy"] = psnr(frames[:, :1].expand(-1, n - 1, -1, -1, -1).flatten(0, 1), frames[:, 1:].flatten(0, 1))
        cos = lambda a, b: F.cosine_similarity(a.float(), b.float(), -1).mean()
        for ell in range(cfg.refine_passes):
            diag[f"base_cos_pass{ell}"] = torch.stack([cos(o["passes"][ell]["F_base"], Ft[:, j + 1])
                                                       for j, o in enumerate(tg)]).mean()
        diag["base_cos_final"] = torch.stack([cos(o["base"]["F_base"], Ft[:, j + 1]) for j, o in enumerate(tg)]).mean()
        diag["init_cos"] = torch.stack([cos(o["base"]["F_init"], Ft[:, j + 1]) for j, o in enumerate(tg)]).mean()
        diag["feat_cos"] = torch.stack([cos(o["F_hat"], Ft[:, j + 1]) for j, o in enumerate(tg)]).mean()
        diag["support_frac"] = torch.stack([o["base"]["support"].float().mean() for o in tg]).mean()
        diag["L_mean"] = torch.stack([o["rec"]["L"].float().mean() for o in tg]).mean()
        R, t = g.se3_exp(torch.stack([o["rec"]["xi"] for o in tg], 1))
        diag["xi_rot_deg"] = g.rotation_angle_deg(R).mean()
        diag["xi_trans"] = t.norm(dim=-1).mean()
        diag["mem_delta"] = torch.stack([(o["z"] - S["c0"]).abs().mean() for o in tg]).mean()

    # ============================================================================== prior (phase T3)
    def predict(self, S: dict, dt: torch.Tensor, cam: Optional[tuple] = None) -> dict:
        """Predicted record D^_{t+dt} from S_t alone (cam: the query camera of a static novel view)."""
        cfg = self.cfg
        B = S["h"].shape[0]
        zs = self.replay(S)
        z = zs[-1] if zs else S["zb"].float()
        times = [S["h"][:, sc.H_TT + j] for j in range(len(S["recs"]))]
        bank = self.bank.temporal(S, z, times)
        o = self.prior(bank, dt, cam)
        prev = S["recs"][-1] if S["recs"] else None
        k_prev = prev["kappa"] if prev is not None else S["kappa0"]
        xi = (prev["xi"] if prev is not None else 0) + o["xi"]
        kappa = g.kappa_from_logits(kappa_to_logits(k_prev, cfg) + o["kz"], cfg)
        M = (prev["M"] if prev is not None else 0) + o["M"]
        if cam is not None:
            xi, kappa, M = cam[0].float(), cam[1].float(), M * 0
        rr = o["r_raw"]
        r = torch.cat([torch.sigmoid(rr[..., :2]), cfg.rho_lo + (cfg.rho_hi - cfg.rho_lo) * torch.sigmoid(rr[..., 2:3]),
                       cfg.sigma_min + cfg.sigma_span * torch.sigmoid(rr[..., 3:4]), rr[..., 4:]], -1)
        L = torch.full((B,), cfg.n_slots, dtype=torch.long, device=r.device)
        return dict(xi=xi, M=M, kappa=kappa, b=o["b"], r=r, L=L, prov=torch.ones(B, device=r.device))

    def _prior_segment(self, frames, kind, cam, parts, diag) -> tuple:
        """Posterior states of the prefix (frozen tokenizer, no grad), then a free rollout of P_psi; the loss
        decodes the predicted states at the horizons and compares with the real future frames."""
        cfg = self.cfg
        B, n = frames.shape[:2]
        dev = frames.device
        H = max(cfg.horizons)
        t0 = n - 1 - H
        if t0 < 0:
            raise ValueError(f"prior phase needs clips of at least {H + 1} frames")
        with torch.no_grad():
            self.eval()
            _, out = self._segment(frames[:, :t0 + 1], "video", None, {}, {})
            S = out["S"]
            st = self.stems(frames[:, t0 + 1:].flatten(0, 1))
            Ft = st["F"].view(B, H, *st["F"].shape[1:])
        self.train()
        X01 = (frames.float() + 1) / 2
        loss = 0.0
        for h in range(1, H + 1):
            t = torch.full((B,), (t0 + h) * cfg.frame_seconds, device=dev)
            times = [S["h"][:, sc.H_TT + j] for j in range(len(S["recs"]))]
            t_last = times[-1] if times else torch.zeros(B, device=dev)
            rec = self.predict(S, t - t_last)
            S = dict(S, recs=list(S["recs"]), h=S["h"].clone())
            self.append_and_fold(S, rec, t)
            if h in cfg.horizons:
                dec = self.decode(S)
                lr = self._rec((dec["rgb"][:, -1:] + 1) / 2, X01[:, t0 + h:t0 + h + 1], dec["F"][:, -1:], Ft[:, h - 1:h],
                               parts, diag, f"prior_h{h}", lpips=False)
                loss = loss + lr
                with torch.no_grad():
                    diag[f"psnr_prior_h{h}"] = psnr(dec["rgb"][:, -1], frames[:, t0 + h])
                    diag[f"psnr_copy_h{h}"] = psnr(frames[:, t0], frames[:, t0 + h])
        return loss, dict(S=S)

    # ============================================================================== forward
    def forward(self, x: torch.Tensor, modality: str, cam: Optional[dict] = None,
                L: Optional[torch.Tensor] = None, return_outputs: bool = False) -> dict:
        """x in [-1, 1]; L (eval) innovation lengths [B, n - 1] or None."""
        if self.cfg.readout_version != "legacy":
            raise ValueError('shared query training uses pilot_step; legacy posterior/prior training is gated')
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")
        if modality in ("threed", "world") and cam is None:
            raise ValueError(f"{modality} needs cam = dict(c2w, K)")
        frames = x.unsqueeze(1) if modality == "image" else x.permute(0, 2, 1, 3, 4)
        kind = {"image": "image", "video": "video", "threed": "static", "world": "world"}[modality]
        parts, diag = {}, {}
        if self.cfg.phase == "prior":
            if modality not in ("video", "world"):
                loss, outputs = frames.sum() * 0, {}
            else:
                loss, outputs = self._prior_segment(frames, "video", None, parts, diag)
        else:
            loss, outputs = self._segment(frames, kind, cam, parts, diag, L)
        dev = x.device
        diag = {k: (v if torch.is_tensor(v) else torch.tensor(float(v))).to(dev).float().detach() for k, v in diag.items()}
        out = dict(loss=loss, parts=parts, diag=diag)
        if return_outputs:
            out["outputs"] = outputs
        return out


def load_grain_prior(ckpt_path: str, device="cpu", strict: bool = True, **overrides):
    """Checkpoint of scripts/train_grain_prior.py -> (model in eval mode, cfg, step)."""
    from .config import config_from_dict
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    version = ck.get('cfg',{}).get('readout_version','legacy')
    if overrides.get('readout_version',version) != version:
        raise ValueError('readout migration must be explicit; use warm_start_query_encoder')
    cfg = config_from_dict(ck.get("cfg", {}), **{"w_lpips": 0.0, **overrides})
    model = MAVTGrainPrior(cfg)
    sd = ck["model"] if "model" in ck else ck
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("stems.dino.model.")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return model.to(device).eval(), cfg, int(ck.get("step", -1))
