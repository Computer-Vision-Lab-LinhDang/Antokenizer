"""MAVT C-D tokenizer, bản 3.2, on a frozen DINOv3 with the two-stream backbone.

forward(x, modality, cam) -> dict(loss, parts, diag), the interface of dino_v0.MAVTDinoV0, so the v0
trainer (scripts/train_cd32.py) and MixedStep run it unchanged.

  image   [B, 3, H, W]      C only (47 + 1 channels per cell); xi = I, m = 0, so p(u) = u; L = 0
  video   [B, 3, T, H, W]   T = S * K; per segment: C at the keyframe (+ focal), D_geom (6 + 3 g^2) and up to
                            L_max innovation tokens at each other frame, a 7-number link for segments k >= 1
  threed  [B, 3, V, H, W]   posed views of a static object, cam = dict(c2w [B, V, 4, 4] OpenCV, K [B, V, 3, 3]);
                            view 0 is the keyframe, the others are reordered by angle; D_geom from the known
                            poses (0 numbers), M = 0; up to L_max innovation tokens per view

Frames after the keyframe are coded in order: predict (geometry + previous decoded frame) -> residual ->
Enc_R -> innovation -> decoded frame, which is the reference of the next one.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..dino_v0.camera import patch_rays
from . import geometry as geo
from .backbone import CHeads, DHeads, TwoStreamBackbone
from .config import CD32Config
from .decoder import Decoder
from .enc_r import ResidualEncoder
from .encoder import FrameEncoder
from ..dino_v0.losses import ignore_lpips_keys
from .losses import charbonnier, feat_loss, gaussian_sample, kl_masked, psnr, rgb_l1
from .regs import DepthRegs, _pearson  # noqa: F401  (_pearson re-exported)

MODALITIES = ("image", "video", "threed")



class MAVTCD32(DepthRegs, nn.Module):
    COMPILE_REGIONS = ("StreamBlock", "InnovBlock", "WindowRead")     # repeated blocks for regional compile

    def __init__(self, cfg: CD32Config, use_checkpoint: bool = False):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg)
        feat_dim = self.encoder.dino.out_dim
        self.backbone = TwoStreamBackbone(cfg)
        self.backbone.use_checkpoint = use_checkpoint
        self.c_heads = CHeads(cfg)
        self.d_heads = DHeads(cfg)
        self.flow_w = nn.Linear(cfg.d_c_video, 9)                 # convex upsampling weights from c_k
        nn.init.zeros_(self.flow_w.weight)
        nn.init.zeros_(self.flow_w.bias)
        self.enc_r = ResidualEncoder(cfg, feat_dim)
        self.decoder = Decoder(cfg, feat_dim)
        self.lpips = None
        if cfg.w_lpips > 0:
            from ..dino_v0.losses import LPIPSLoss
            self.lpips = LPIPSLoss()
        self.register_load_state_dict_post_hook(ignore_lpips_keys)   # frozen VGG is never saved: not "missing"
        self.init_probe(cfg, feat_dim)

    # ------------------------------------------------------------------------------ helpers
    @staticmethod
    def to_frames(x: torch.Tensor, modality: str) -> torch.Tensor:
        """image [B, 3, H, W] -> [B, 1, 3, H, W]; video / threed [B, 3, T, H, W] -> [B, T, 3, H, W]."""
        return x.unsqueeze(1) if modality == "image" else x.permute(0, 2, 1, 3, 4)

    def sample_L(self, n: int, device) -> torch.Tensor:
        """Nested dropout: L = 0 with probability p_l0, else uniform in 1..L_max (eval: L_max)."""
        Lm = self.cfg.L_max
        if not self.training:
            return torch.full((n,), Lm, dtype=torch.long, device=device)
        L = torch.randint(1, Lm + 1, (n,), device=device)
        return torch.where(torch.rand(n, device=device) < self.cfg.p_l0, torch.zeros_like(L), L)

    def _identity(self, n: int, device) -> tuple:
        return torch.eye(3, device=device).expand(n, 3, 3), torch.zeros(n, 3, device=device)

    def _code_frames(self, c, projs: list, dts: list, rays: list, h_first: torch.Tensor,
                     targets: torch.Tensor, L_given: Optional[torch.Tensor]) -> dict:
        """Frames after the keyframe, in encode order. targets [B', n, N, feat_dim]."""
        cfg = self.cfg
        Bp = c.shape[0]
        prev = h_first.detach()
        out = dict(h=[], h_geom=[], h_predc=[], L=[], kl=[], tokens=[])
        for j, proj in enumerate(projs):
            h_geom, h_prev, d_min = self.decoder.predict(c, "video", proj, dts[j], rays[j], prev)
            h_pred = h_geom if h_prev is None else h_geom + h_prev
            x_pred = self.decoder.feat(h_pred).detach().float()
            mu, lv = self.enc_r(targets[:, j].float() - x_pred, x_pred, c.detach())
            r = gaussian_sample(mu, lv, self.training)
            L = self.sample_L(Bp, c.device) if L_given is None else L_given[:, j]
            h_inn = self.decoder.innov(h_pred.detach(), d_min, r, L)
            h = torch.where((L > 0).view(-1, 1, 1), h_pred.detach() + h_inn, h_pred)   # no sg when L = 0
            prev = h.detach()
            keep = torch.arange(cfg.L_max, device=c.device).view(1, -1) < L.view(-1, 1)
            out["h"].append(h)
            out["h_geom"].append(h_geom)
            if h_prev is not None:
                out["h_predc"].append(h_geom.detach() + h_prev)
            out["L"].append(L)
            out["kl"].append(kl_masked(mu, lv, keep))
            out["tokens"].append(r * keep.unsqueeze(-1))
        return out

    def _rec(self, H: torch.Tensor, tgt: torch.Tensor, img: torch.Tensor, parts: dict, diag: dict):
        """L_rec on every decoded frame. H [B', T, N, D], tgt [B', T, N, C], img [B', T, 3, H, W]."""
        cfg = self.cfg
        f = feat_loss(self.decoder.feat(H), tgt)
        rgb = self.decoder.rgb(H.flatten(0, 1))
        im = img.flatten(0, 1)
        l1 = rgb_l1(rgb, im)
        parts["rec_feat"], parts["rec_rgb"] = f.detach(), l1.detach()
        diag["psnr"] = psnr(rgb.detach(), im)
        loss = cfg.w_feat * f + cfg.w_rgb * l1
        if self.lpips is not None:
            n = cfg.lpips_max_frames
            if 0 < n < rgb.shape[0]:                              # random subset of the decoded frames
                sel = torch.randperm(rgb.shape[0], device=rgb.device)[:n]
                lp = self.lpips(rgb[sel], im[sel])
            else:
                lp = self.lpips(rgb, im)
            parts["lpips"] = lp.detach()
            loss = loss + cfg.w_lpips * lp
        return loss, rgb

    def _geo(self, co: dict, tgt: torch.Tensor, img: torch.Tensor, parts: dict, diag: dict):
        """L_geo (geometry-only decode) and L_pred (sg[h_geom] + previous-frame read) on frames after the keyframe."""
        cfg = self.cfg
        Hg = torch.stack(co["h_geom"], 1)
        g = charbonnier(self.decoder.feat(Hg), tgt, cfg.charbonnier_eps)
        parts["geo_feat"] = g.detach()
        if cfg.geo_rgb:
            rg = self.decoder.rgb(Hg.flatten(0, 1))
            im = img.flatten(0, 1)
            gr = charbonnier(rg, im, cfg.charbonnier_eps)
            parts["geo_rgb"] = gr.detach()
            diag["psnr_geo"] = psnr(rg.detach(), im)
            g = g + gr
        loss = cfg.w_geo * g
        if co["h_predc"]:
            p = charbonnier(self.decoder.feat(torch.stack(co["h_predc"], 1)), tgt, cfg.charbonnier_eps)
            parts["pred_feat"] = p.detach()
            loss = loss + cfg.w_pred * p
        return loss

    def _c(self, hc, mem_key, modality, other_views=None) -> dict:
        ch = self.c_heads(hc, mem_key, modality, other_views)
        ch["c"] = gaussian_sample(ch["mu"], ch["logvar"], self.training)
        return ch

    # ------------------------------------------------------------------------------ modalities
    def _image(self, frames, parts, diag, L_given=None):
        cfg = self.cfg
        B = frames.shape[0]
        enc = self.encoder(frames[:, 0])
        bb = self.backbone(enc["x0"], enc["mem"], torch.zeros(B, device=frames.device))
        ch = self._c(bb["hc"], enc["mem"], "image")
        rho = geo.median_normalise(ch["rho"])
        rho_dec = rho.detach() if cfg.detach_rho_image else rho
        Kc = geo.intrinsics_from_focal(geo.fov_to_focal(torch.full((B,), cfg.fov_init, device=frames.device),
                                                        cfg.grid), cfg.grid)       # f has no effect when xi = I
        R, tau = self._identity(B, frames.device)
        h, _, _ = self.decoder.predict(ch["c"], "image", self.decoder.project(rho_dec, Kc, R, tau), 0.0)
        loss, rgb = self._rec(h.unsqueeze(1), enc["feat_target"].unsqueeze(1), frames, parts, diag)
        loss = loss + self._c_regs(ch, rho, enc["feat_target"], parts, diag, enc["layers"], "image")
        diag["numbers"] = torch.tensor(float(cfg.segment_numbers("image")), device=frames.device)
        return loss, dict(rgb=rgb.view_as(frames), c=ch["mu"], rho=rho)

    def _video(self, frames, parts, diag, L_given=None):
        cfg = self.cfg
        B, T = frames.shape[:2]
        K, G, dev = cfg.keyframe_every, cfg.grid, frames.device
        if T % K:
            raise ValueError(f"video length {T} must be a multiple of keyframe_every={K}")
        S = T // K
        Bp = B * S
        enc = self.encoder(frames.flatten(0, 1))
        seg = lambda t: t.reshape(Bp, K, *t.shape[1:])
        x0, mem, tgt = seg(enc["x0"]), seg(enc["mem"]), seg(enc["feat_target"])
        img = frames.reshape(Bp, K, *frames.shape[2:])
        tk = (torch.arange(S, device=dev) * K).float().repeat(B)                        # frame index of keyframes
        td = tk.view(-1, 1) + torch.arange(1, K, device=dev).float()
        bb = self.backbone(x0[:, 0], mem[:, 0], tk, x0[:, 1:], mem[:, 1:], td, n_seg=S)
        ch = self._c(bb["hc"], mem[:, 0], "video")
        c = ch["c"]
        rho = geo.median_normalise(ch["rho"])
        Kc = geo.intrinsics_from_focal(geo.fov_to_focal(ch["fov"], G), G)
        dh = self.d_heads(bb["hd"], bb["link"])
        w_flow = self.flow_w(c)
        R0, t0 = self._identity(Bp, dev)
        h_key, _, _ = self.decoder.predict(c, "video", self.decoder.project(rho, Kc, R0, t0), 0.0)
        projs, ms, Rs = [], [], []
        for j in range(K - 1):
            R, tau = geo.pose_matrix(dh["xi"][:, j])
            m = geo.convex_upsample(dh["M"][:, j], w_flow, G)
            projs.append(self.decoder.project(rho, Kc, R, tau, m))
            ms.append(m)
            Rs.append(R)
        Lg = None if L_given is None else L_given.reshape(Bp, K - 1)
        co = self._code_frames(c, projs, [(j + 1) / K for j in range(K - 1)], [None] * (K - 1), h_key,
                               tgt[:, 1:], Lg)
        H = torch.stack([h_key] + co["h"], 1)
        loss, rgb = self._rec(H, tgt, img, parts, diag)
        loss = loss + self._geo(co, tgt[:, 1:], img[:, 1:], parts, diag)
        m_all = torch.stack(ms, 1)
        fl = m_all.abs().mean()
        parts["flow_l1"] = fl.detach()
        loss = loss + cfg.w_flow * fl
        kd = torch.stack(co["kl"]).mean()
        parts["kl_d"] = kd.detach()
        loss = loss + cfg.w_kl_d * kd
        loss = loss + self._c_regs(ch, rho, tgt[:, 0], parts, diag, [seg(f)[:, 0] for f in enc["layers"]], "video")
        # link: keyframe k decoded from C_{k-1} through the 7 link numbers (pose + log scale)
        sel = torch.nonzero(torch.arange(Bp, device=dev) % S > 0).squeeze(1)
        if sel.numel() > 0 and cfg.w_link > 0:
            prv = sel - 1
            lk = dh["link"][sel]
            Rl, tl = geo.pose_matrix(lk[:, :6])
            proj = self.decoder.project(rho[prv], Kc[prv], Rl, tl, None, Kc[sel], scale=lk[:, 6])
            hl, _, _ = self.decoder.predict(c[prv], "video", proj, 0.0)
            ll = charbonnier(self.decoder.feat(hl), tgt[sel, 0], cfg.charbonnier_eps)
            parts["link"] = ll.detach()
            loss = loss + cfg.w_link * ll
            diag["link_scale"] = lk[:, 6].detach().abs().mean()
        else:                                                    # keep every parameter in the graph (DDP)
            loss = loss + 0.0 * dh["link"].sum()
        Ls = torch.stack(co["L"], 1).float()                     # [B', K-1]
        R_all = torch.stack(Rs, 1)
        diag["L_mean"] = Ls.mean()
        diag["xi_rot_deg"] = geo.rotation_angle_deg(R_all.detach()).mean()
        diag["xi_trans"] = dh["xi"][..., 3:].detach().norm(dim=-1).mean()
        diag["fov_deg"] = ch["fov"].detach().mean()
        diag["numbers"] = (cfg.c_numbers("video") + (K - 1) * cfg.geom_numbers + (Ls * cfg.d_r).sum(1).mean()
                           + cfg.link_numbers * (S - 1) / S)            # per segment, as coded this step
        outputs = dict(rgb=rgb.view(B, T, *rgb.shape[1:]), c=ch["mu"], rho=rho, fov=ch["fov"], xi=dh["xi"],
                       M=dh["M"], link=dh["link"], tokens=torch.stack(co["tokens"], 1), L=Ls.long())
        return loss, outputs

    def _threed(self, frames, cam, parts, diag, L_given=None):
        cfg = self.cfg
        B, V = frames.shape[:2]
        G, dev = cfg.grid, frames.device
        order = geo.view_order(cam["c2w"].float())
        take = lambda t: torch.gather(t, 1, order.view(B, V, *([1] * (t.dim() - 2))).expand_as(t))
        frames, c2w, Kpix = take(frames), take(cam["c2w"].float()), take(cam["K"].float())
        with geo.fp32(dev):
            rel = torch.linalg.inv(c2w[:, :1]) @ c2w                                      # views in camera 0
            rays = patch_rays(rel, Kpix, cfg.image_size, G)                               # [B, V, N, 6]
            T_v = torch.linalg.inv(rel)                                                   # camera 0 -> camera v
        enc = self.encoder(frames.flatten(0, 1), rays.flatten(0, 1))
        sv = lambda t: t.reshape(B, V, *t.shape[1:])
        x0, mem, tgt = sv(enc["x0"]), sv(enc["mem"]), sv(enc["feat_target"])
        bb = self.backbone(x0[:, 0], mem[:, 0], torch.zeros(B, device=dev))
        ch = self._c(bb["hc"], mem[:, 0], "threed", other_views=x0[:, 1:] if V > 1 else None)
        c, rho = ch["c"], ch["rho"]                                                       # metric: poses are known
        Kc = geo.intrinsics_to_cells(Kpix, cfg.patch)
        R0, t0 = self._identity(B, dev)
        h0, _, _ = self.decoder.predict(c, "video", self.decoder.project(rho, Kc[:, 0], R0, t0), 0.0,
                                        rays=rays[:, 0])
        projs = [self.decoder.project(rho, Kc[:, 0], T_v[:, v, :3, :3], T_v[:, v, :3, 3], None, Kc[:, v])
                 for v in range(1, V)]
        co = self._code_frames(c, projs, [0.0] * (V - 1), [rays[:, v] for v in range(1, V)], h0, tgt[:, 1:],
                               L_given)
        H = torch.stack([h0] + co["h"], 1)
        loss, rgb = self._rec(H, tgt, frames, parts, diag)
        if V > 1:
            loss = loss + self._geo(co, tgt[:, 1:], frames[:, 1:], parts, diag)
            kd = torch.stack(co["kl"]).mean()
            parts["kl_d"] = kd.detach()
            loss = loss + cfg.w_kl_d * kd
            Ls = torch.stack(co["L"], 1).float()
            diag["L_mean"] = Ls.mean()
            diag["numbers"] = cfg.c_numbers("threed") + (Ls * cfg.d_r).sum(1).mean()
        depth_gt = take(cam["depth"].float())[:, 0] if cam.get("depth") is not None else None
        loss = loss + self._c_regs(ch, rho, tgt[:, 0], parts, diag, [sv(f)[:, 0] for f in enc["layers"]],
                                   "threed", depth_gt)
        diag["rho_refine_gate"] = torch.tanh(self.c_heads.alpha).detach()
        return loss, dict(rgb=rgb.view(B, V, *rgb.shape[1:]), c=ch["mu"], rho=rho, order=order)

    # ------------------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, modality: str, cam: Optional[dict] = None,
                L: Optional[torch.Tensor] = None, return_outputs: bool = False) -> dict:
        """x in [-1, 1]; L (optional, eval) innovation lengths [B, frames after keyframes]."""
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")
        frames = self.to_frames(x, modality)
        parts, diag = {}, {}
        if modality == "image":
            loss, outputs = self._image(frames, parts, diag)
        elif modality == "video":
            loss, outputs = self._video(frames, parts, diag, L)
        else:
            if cam is None:
                raise ValueError("threed needs cam = dict(c2w, K)")
            loss, outputs = self._threed(frames, cam, parts, diag, L)
        d = self.decoder.read
        diag["lam_d"] = torch.nn.functional.softplus(d.lam_d).detach().mean()
        diag["lam_z"] = torch.nn.functional.softplus(d.lam_z).detach().mean()
        dev = x.device
        diag = {k: (v if torch.is_tensor(v) else torch.tensor(float(v))).to(dev).float().detach()
                for k, v in diag.items()}
        out = dict(loss=loss, parts=parts, diag=diag)
        if return_outputs:
            out["outputs"] = outputs
        return out


def load_model(ckpt_path: str, device="cpu", strict: bool = True, **overrides):
    """Checkpoint of scripts/train_cd32.py -> (model in eval mode, cfg, step). Training-only parts are off
    (LPIPS); `overrides` replace config fields, e.g. depth_probe=<probe file> or depth_probe=None."""
    import dataclasses
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fields = {f.name for f in dataclasses.fields(CD32Config)}
    cfg_d = {k: v for k, v in dict(ck.get("cfg", {})).items() if k in fields}
    for k in ("dino_layers", "rope_dims", "conv_channels", "depth_prior_on"):
        if isinstance(cfg_d.get(k), list):
            cfg_d[k] = tuple(cfg_d[k])
    cfg_d.update(w_lpips=0.0, depth_probe=None, w_depth_prior=0.0)
    cfg_d.update(overrides)
    cfg = CD32Config(**cfg_d)
    model = MAVTCD32(cfg)
    sd = ck["model"] if "model" in ck else ck
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.dino.model.")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return model.to(device).eval(), cfg, int(ck.get("step", -1))
