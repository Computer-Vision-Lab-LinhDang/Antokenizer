"""MAVT C-D, vanilla variant (src/mavt/cd32/vanilla.py): forward(x, modality, cam) -> dict(loss, parts, diag),
the interface of MAVTDinoV0 and MAVTCD32, so scripts/train_cdv.py is the v0 trainer.

  image   [B, 3, H, W]      C only; identity projection; L = 0
  video   [B, 3, T, H, W]   T = S * K; per segment C at the keyframe (+ focal), (xi, M, r[:L]) at the others
  threed  [B, 3, V, H, W]   posed views, cam = dict(c2w, K [, depth]); view 0 = keyframe, xi from the cameras

All frames of a segment are decoded in one batched pass of the decoder.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..dino_v0.camera import patch_rays
from ..dino_v0.losses import ignore_lpips_keys
from . import geometry as geo
from .encoder import FrameEncoder
from .losses import charbonnier, feat_loss, gaussian_sample, kl_masked, psnr, rgb_l1
from .regs import DepthRegs
from .vanilla import CDVConfig, VanillaBackbone, VanillaDecoder, VanillaHeads

MODALITIES = ("image", "video", "threed")


class MAVTCDV(DepthRegs, nn.Module):
    COMPILE_REGIONS = ("StreamBlock", "DecBlock")

    def __init__(self, cfg: CDVConfig, use_checkpoint: bool = False):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg)
        feat_dim = self.encoder.dino.out_dim
        self.backbone = VanillaBackbone(cfg)
        self.backbone.use_checkpoint = use_checkpoint
        self.heads = VanillaHeads(cfg)
        self.flow_w = nn.Linear(cfg.d_c_video, 9)                 # convex upsampling weights from c_k
        nn.init.zeros_(self.flow_w.weight)
        nn.init.zeros_(self.flow_w.bias)
        self.decoder = VanillaDecoder(cfg, feat_dim)
        self.lpips = None
        if cfg.w_lpips > 0:
            from ..dino_v0.losses import LPIPSLoss
            self.lpips = LPIPSLoss()
        self.register_load_state_dict_post_hook(ignore_lpips_keys)   # frozen VGG is never saved: not "missing"
        self.init_probe(cfg, feat_dim)

    # ------------------------------------------------------------------------------ helpers
    @staticmethod
    def to_frames(x: torch.Tensor, modality: str) -> torch.Tensor:
        return x.unsqueeze(1) if modality == "image" else x.permute(0, 2, 1, 3, 4)

    def sample_L(self, shape: tuple, device) -> torch.Tensor:
        """Nested dropout: L = 0 with probability p_l0, else uniform in 1..L_max (eval: L_max)."""
        Lm = self.cfg.L_max
        if not self.training:
            return torch.full(shape, Lm, dtype=torch.long, device=device)
        L = torch.randint(1, Lm + 1, shape, device=device)
        return torch.where(torch.rand(shape, device=device) < self.cfg.p_l0, torch.zeros_like(L), L)

    def _tokens(self, enc: dict) -> torch.Tensor:
        """Input cell tokens: the v0-style projection of the last layer + stem, plus the 4 DINO layers."""
        return enc["x0"] + enc["mem"].sum(1)

    def _identity(self, n: int, device) -> tuple:
        return torch.eye(3, device=device).expand(n, 3, 3), torch.zeros(n, 3, device=device)

    def _decode(self, c, kind: str, projs: list, dts: list, tokens, L, rays=None) -> torch.Tensor:
        """All frames of a segment in one pass. c [B, N, d_c]; projs: F tuples (p, z, valid, cull);
        dts: F floats; tokens [B, F, L_max, d_r]; L [B, F]; rays [B, F, N, 6] or None -> h [B, F, N, D]."""
        B, F = L.shape
        cf = c.unsqueeze(1).expand(-1, F, -1, -1).flatten(0, 1)
        proj = tuple(None if any(t is None for t in comp) else torch.stack(comp, 1).flatten(0, 1)
                     for comp in zip(*projs))
        dt = torch.tensor(dts, device=c.device, dtype=torch.float32).repeat(B)
        h = self.decoder(cf, kind, proj, dt, tokens.flatten(0, 1), L.flatten(),
                         None if rays is None else rays.flatten(0, 1))
        return h.view(B, F, *h.shape[1:])

    def _zero_tokens(self, B: int, F: int, device) -> tuple:
        cfg = self.cfg
        return (torch.zeros(B, F, cfg.L_max, cfg.d_r, device=device),
                torch.zeros(B, F, dtype=torch.long, device=device))

    def _rec(self, H, tgt, img, parts, diag):
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

    def _geo(self, c, kind, projs, dts, tgt, img, parts, diag, rays=None):
        """L_geo: the frames after the keyframe decoded with L = 0 (geometry only), Charbonnier."""
        cfg = self.cfg
        B, F = tgt.shape[:2]
        tokens, L0 = self._zero_tokens(B, F, c.device)
        Hg = self._decode(c, kind, projs, dts, tokens, L0, rays)
        g = charbonnier(self.decoder.feat(Hg), tgt, cfg.charbonnier_eps)
        parts["geo_feat"] = g.detach()
        if cfg.geo_rgb:
            rg = self.decoder.rgb(Hg.flatten(0, 1))
            im = img.flatten(0, 1)
            gr = charbonnier(rg, im, cfg.charbonnier_eps)
            parts["geo_rgb"] = gr.detach()
            diag["psnr_geo"] = psnr(rg.detach(), im)
            g = g + gr
        return cfg.w_geo * g

    def _innovation(self, hd: dict, B: int, F: int, L_given) -> tuple:
        """Sampled innovation tokens with the nested-dropout prefix -> (tokens [B, F+1, L_max, d_r] with the
        keyframe's zeros first, L [B, F+1], keep [B, F, L_max])."""
        cfg = self.cfg
        dev = hd["r_mu"].device
        r = gaussian_sample(hd["r_mu"], hd["r_logvar"], self.training)
        L = self.sample_L((B, F), dev) if L_given is None else L_given.reshape(B, F)
        keep = torch.arange(cfg.L_max, device=dev).view(1, 1, -1) < L.unsqueeze(-1)
        z_tok, z_L = self._zero_tokens(B, 1, dev)
        tokens = torch.cat([z_tok, r * keep.unsqueeze(-1)], 1)
        return tokens, torch.cat([z_L, L], 1), keep

    # ------------------------------------------------------------------------------ modalities
    def _image(self, frames, parts, diag):
        cfg = self.cfg
        B, dev = frames.shape[0], frames.device
        enc = self.encoder(frames[:, 0])
        hd = self.heads(self.backbone(self._tokens(enc)), "image")
        c = gaussian_sample(hd["mu"], hd["logvar"], self.training)
        rho = geo.median_normalise(hd["rho"])
        rho_dec = rho.detach() if cfg.detach_rho_image else rho
        Kc = geo.intrinsics_from_focal(geo.fov_to_focal(torch.full((B,), cfg.fov_init, device=dev), cfg.grid), cfg.grid)
        R, tau = self._identity(B, dev)
        tokens, L0 = self._zero_tokens(B, 1, dev)
        H = self._decode(c, "image", [self.decoder.project(rho_dec, Kc, R, tau)], [0.0], tokens, L0)
        loss, rgb = self._rec(H, enc["feat_target"].unsqueeze(1), frames, parts, diag)
        loss = loss + self._c_regs(hd, rho, enc["feat_target"], parts, diag, enc["layers"], "image")
        diag["numbers"] = torch.tensor(float(cfg.segment_numbers("image")), device=dev)
        return loss, dict(rgb=rgb.view_as(frames), c=hd["mu"], rho=rho)

    def _video(self, frames, parts, diag, L_given=None):
        cfg = self.cfg
        B, T = frames.shape[:2]
        K, G, N, dev = cfg.keyframe_every, cfg.grid, cfg.n_cells, frames.device
        if T % K:
            raise ValueError(f"video length {T} must be a multiple of keyframe_every={K}")
        S = T // K
        Bp = B * S
        enc = self.encoder(frames.flatten(0, 1))
        seg = lambda t: t.reshape(Bp, K, *t.shape[1:])
        x, tgt = seg(self._tokens(enc)), seg(enc["feat_target"])
        img = frames.reshape(Bp, K, *frames.shape[2:])
        hd = self.heads(self.backbone(x[:, 0], x[:, 1:]), "video")
        c = gaussian_sample(hd["mu"], hd["logvar"], self.training)
        rho = geo.median_normalise(hd["rho"])
        Kc = geo.intrinsics_from_focal(geo.fov_to_focal(hd["fov"], G), G)
        w_flow = self.flow_w(c)
        R0, t0 = self._identity(Bp, dev)
        projs, ms, Rs = [self.decoder.project(rho, Kc, R0, t0)], [], []
        for j in range(K - 1):
            if cfg.geometry:
                R, tau = geo.pose_matrix(hd["xi"][:, j])
                m = geo.convex_upsample(hd["M"][:, j], w_flow, G)
                projs.append(self.decoder.project(rho, Kc, R, tau, m))
            else:
                R, m = R0, torch.zeros(Bp, N, 3, device=dev)
                projs.append(projs[0])
            ms.append(m)
            Rs.append(R)
        tokens, L_all, keep = self._innovation(hd, Bp, K - 1, L_given)
        dts = [j / K for j in range(K)]
        H = self._decode(c, "video", projs, dts, tokens, L_all)
        loss, rgb = self._rec(H, tgt, img, parts, diag)
        if cfg.w_geo > 0:
            loss = loss + self._geo(c, "video", projs[1:], dts[1:], tgt[:, 1:], img[:, 1:], parts, diag)
        fl = torch.stack(ms, 1).abs().mean()
        parts["flow_l1"] = fl.detach()
        loss = loss + cfg.w_flow * fl
        kd = kl_masked(hd["r_mu"].flatten(0, 1), hd["r_logvar"].flatten(0, 1), keep.flatten(0, 1))
        parts["kl_d"] = kd.detach()
        loss = loss + cfg.w_kl_d * kd
        loss = loss + self._c_regs(hd, rho, tgt[:, 0], parts, diag, [seg(f)[:, 0] for f in enc["layers"]], "video")
        L = L_all[:, 1:].float()
        diag["L_mean"] = L.mean()
        diag["xi_rot_deg"] = geo.rotation_angle_deg(torch.stack(Rs, 1).detach()).mean()
        diag["xi_trans"] = hd["xi"][..., 3:].detach().norm(dim=-1).mean()
        diag["fov_deg"] = hd["fov"].detach().mean()
        diag["numbers"] = cfg.c_numbers("video") + (K - 1) * cfg.geom_numbers + (L * cfg.d_r).sum(1).mean()
        outputs = dict(rgb=rgb.view(B, T, *rgb.shape[1:]), c=hd["mu"], rho=rho, fov=hd["fov"], xi=hd["xi"],
                       M=hd["M"], tokens=tokens[:, 1:], L=L.long())
        return loss, outputs

    def _threed(self, frames, cam, parts, diag, L_given=None):
        cfg = self.cfg
        B, V = frames.shape[:2]
        G, N, dev = cfg.grid, cfg.n_cells, frames.device
        order = geo.view_order(cam["c2w"].float())
        take = lambda t: torch.gather(t, 1, order.view(B, V, *([1] * (t.dim() - 2))).expand_as(t))
        frames, c2w, Kpix = take(frames), take(cam["c2w"].float()), take(cam["K"].float())
        with geo.fp32(dev):
            rel = torch.linalg.inv(c2w[:, :1]) @ c2w                                      # views in camera 0
            rays = patch_rays(rel, Kpix, cfg.image_size, G)                               # [B, V, N, 6]
            T_v = torch.linalg.inv(rel)                                                   # camera 0 -> camera v
        enc = self.encoder(frames.flatten(0, 1), rays.flatten(0, 1))
        sv = lambda t: t.reshape(B, V, *t.shape[1:])
        x, tgt = sv(self._tokens(enc)), sv(enc["feat_target"])
        hd = self.heads(self.backbone(x[:, 0], x[:, 1:] if V > 1 else None), "threed")
        c, rho = gaussian_sample(hd["mu"], hd["logvar"], self.training), hd["rho"]   # metric: poses are known
        Kc = geo.intrinsics_to_cells(Kpix, cfg.patch)
        R0, t0 = self._identity(B, dev)
        projs = [self.decoder.project(rho, Kc[:, 0], R0, t0)]
        for v in range(1, V):
            projs.append(self.decoder.project(rho, Kc[:, 0], T_v[:, v, :3, :3], T_v[:, v, :3, 3], None, Kc[:, v])
                         if cfg.geometry else projs[0])
        if V > 1:
            tokens, L_all, keep = self._innovation(hd, B, V - 1, L_given)
        else:
            tokens, L_all = self._zero_tokens(B, 1, dev)
        dts = [0.0] * V
        H = self._decode(c, "video", projs, dts, tokens, L_all, rays)
        loss, rgb = self._rec(H, tgt, frames, parts, diag)
        if V > 1:
            if cfg.w_geo > 0:
                loss = loss + self._geo(c, "video", projs[1:], dts[1:], tgt[:, 1:], frames[:, 1:], parts, diag,
                                        rays[:, 1:])
            kd = kl_masked(hd["r_mu"].flatten(0, 1), hd["r_logvar"].flatten(0, 1), keep.flatten(0, 1))
            parts["kl_d"] = kd.detach()
            loss = loss + cfg.w_kl_d * kd
            L = L_all[:, 1:].float()
            diag["L_mean"] = L.mean()
            diag["numbers"] = cfg.c_numbers("threed") + (L * cfg.d_r).sum(1).mean()
        depth_gt = take(cam["depth"].float())[:, 0] if cam.get("depth") is not None else None
        loss = loss + self._c_regs(hd, rho, tgt[:, 0], parts, diag, [sv(f)[:, 0] for f in enc["layers"]],
                                   "threed", depth_gt)
        return loss, dict(rgb=rgb.view(B, V, *rgb.shape[1:]), c=hd["mu"], rho=rho, order=order)

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


def load_cdv(ckpt_path: str, device="cpu", strict: bool = True, **overrides):
    """Checkpoint of scripts/train_cdv.py -> (model in eval mode, cfg, step); training-only parts off."""
    import dataclasses
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fields = {f.name for f in dataclasses.fields(CDVConfig)}
    cfg_d = {k: v for k, v in dict(ck.get("cfg", {})).items() if k in fields}
    for k in ("dino_layers", "rope_dims", "conv_channels", "depth_prior_on"):
        if isinstance(cfg_d.get(k), list):
            cfg_d[k] = tuple(cfg_d[k])
    cfg_d.update(w_lpips=0.0, depth_probe=None, w_depth_prior=0.0)
    cfg_d.update(overrides)
    cfg = CDVConfig(**cfg_d)
    model = MAVTCDV(cfg)
    sd = ck["model"] if "model" in ck else ck
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.dino.model.")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return model.to(device).eval(), cfg, int(ck.get("step", -1))
