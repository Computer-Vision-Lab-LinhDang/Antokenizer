"""MAVT on DINOv3, v0: the whole model and its training losses.

  frames -> FrameEncoder (frozen DINOv3 + pixel stem) -> Trunk (in-frame + causal t-1 attention)
         -> CDState (C at keyframes, D every frame) -> LatentHeads (z_r with KL) / SemanticPool (z_s)
         -> QueryDecoder (reads z_r of C_k at (u, v) + delta(D_t), reads D_t by content)

Input layout follows mavt.data: image [B, 3, H, W], video [B, 3, T, H, W], both in [-1, 1].
Posed 3D views (modality "threed") use the video layout over views plus cam = dict(c2w [B,T,4,4]
OpenCV camera-to-world, K [B,T,3,3] pixels at the model resolution): tokens and decoder queries get
Plücker rays, and the prediction of view t+1 is conditioned on its pose (novel-view synthesis).
Training: every frame t is decoded from (C of its keyframe, D_t)  -> L_rec; the prediction
D_next_t made before X_{t+1} is decoded at frame t+1 against DINOv3 features -> L_fut.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from mavt.dino_v0 import losses as L
from mavt.dino_v0.camera import patch_rays
from mavt.dino_v0.cd_state import CDState
from mavt.dino_v0.config import DinoV0Config
from mavt.dino_v0.decoder import QueryDecoder, image_to_patches, patches_to_image, query_coords
from mavt.dino_v0.encoder import FrameEncoder
from mavt.dino_v0.latent import LatentHeads, SemanticPool
from mavt.dino_v0.rope import grid_coords, modality_mask
from mavt.dino_v0.trunk import Trunk


def to_frames(data: torch.Tensor, modality: str) -> torch.Tensor:
    """image [B,3,H,W] -> [B,1,3,H,W]; video [B,3,T,H,W] -> [B,T,3,H,W]."""
    if modality == "image":
        return data.unsqueeze(1)
    return data.permute(0, 2, 1, 3, 4).contiguous()


def _mm(modality: str) -> str:
    """Model path of a modality: images alone; video and posed 3D views share the sequence path."""
    return "image" if modality == "image" else "video"


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, N, C], idx [B, Q] -> [B, Q, C]."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _rays_t(e: dict, t: int) -> Optional[torch.Tensor]:
    return None if e.get("rays") is None else e["rays"][:, t]


class MAVTDinoV0(nn.Module):
    def __init__(self, cfg: DinoV0Config, use_checkpoint: bool = False):
        super().__init__()
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg)
        self.trunk = Trunk(cfg, use_checkpoint=use_checkpoint)
        self.cd = CDState(cfg)
        self.latent = LatentHeads(cfg)
        self.pool = SemanticPool(cfg)
        self.feat_dim = self.encoder.dino.out_dim
        self.decoder = QueryDecoder(cfg, feat_dim=self.feat_dim)
        self.lpips = L.LPIPSLoss() if cfg.w_lpips > 0 else None

    # ------------------------------------------------------------------ encode
    def _coords(self, T: int, modality: str, device) -> tuple:
        G, N = self.cfg.grid, self.cfg.n_patches
        coords = torch.stack([grid_coords(G, float(t), device=device) for t in range(T)])   # [T,N,4]
        mask = modality_mask(modality, N, device=device).unsqueeze(0).expand(T, -1, -1)
        return coords, mask

    def encode(self, data: torch.Tensor, modality: str, cam: Optional[dict] = None) -> dict:
        """Tiers A-D. Returns frame targets, C/D states and their latents."""
        mm = _mm(modality)
        frames = to_frames(data, mm)
        B, T = frames.shape[:2]
        rays = None
        if cam is not None:
            rays = patch_rays(cam["c2w"], cam["K"], self.cfg.image_size, self.cfg.grid)   # [B,T,N,6]
        enc = self.encoder(frames.flatten(0, 1), None if rays is None else rays.flatten(0, 1))
        x = enc["x"].unflatten(0, (B, T))
        coords, mask = self._coords(T, mm, data.device)
        X = self.trunk(x, coords, mask)
        st = self.cd(X, coords, mask, None if cam is None else cam["c2w"])
        S = st["C"].shape[1]
        zc, mu_c, lv_c = self.latent.encode_c(st["C"].flatten(0, 1), mm)
        out = dict(frames=frames, coords=coords, mask=mask, state=st, rays=rays, modality=mm,
                   feat_target=enc["feat_target"].unflatten(0, (B, T)), cls=enc["cls"].unflatten(0, (B, T)),
                   mean_energy=enc["mean_energy"], zc=zc.unflatten(0, (B, S)), kl_c=(mu_c, lv_c))
        if st["D"] is not None:
            zd, mu_d, lv_d = self.latent.encode_d(st["D"].flatten(0, 1))
            out["zd"] = zd.unflatten(0, (B, T))
            out["kl_d"] = (mu_d, lv_d)
            zn, _, _ = self.latent.encode_d(st["D_next"].flatten(0, 1))
            out["zd_next"] = zn.unflatten(0, (B, T))
        return out

    def semantic(self, e: dict) -> torch.Tensor:
        """z_s per frame [B, T, zs_dim] from Z_t = C_k U up(D_t)."""
        st = e["state"]
        B, T = e["frames"].shape[:2]
        C_t = st["C"][:, st["seg"]]                                     # [B,T,N,dim]
        D_up = None if st["D"] is None else self.cd.d_state.up(st["D"]).flatten(0, 1)
        return self.pool(C_t.flatten(0, 1), D_up).unflatten(0, (B, T))

    # ------------------------------------------------------------------ decode
    def decode_frame(self, zc: torch.Tensor, zd: Optional[torch.Tensor], t: float, modality: str,
                     idx: Optional[torch.Tensor] = None, rays: Optional[torch.Tensor] = None) -> dict:
        """Decode queries of one frame. zc [B,N,cc], zd [B,M,zr_d] | None, idx [B,Q] | None (all N),
        rays [B,N,6] of the target view (posed views) | None."""
        mm = _mm(modality)
        B = zc.shape[0]
        qc = query_coords(self.cfg.grid, t, B, idx, device=zc.device)
        qm = modality_mask(mm, qc.shape[1], device=zc.device).unsqueeze(0).expand(B, -1, -1)
        if rays is not None and idx is not None:
            rays = _gather(rays, idx)
        return self.decoder(zc, zd, qc, qm, mm, q_rays=rays)

    @torch.no_grad()
    def reconstruct(self, data: torch.Tensor, modality: str, cam: Optional[dict] = None) -> torch.Tensor:
        """Full reconstruction in the input layout, in [-1, 1]."""
        e = self.encode(data, modality, cam)
        B, T = e["frames"].shape[:2]
        seg = e["state"]["seg"]
        outs = []
        for t in range(T):
            zd = e["zd"][:, t] if "zd" in e else None
            o = self.decode_frame(e["zc"][:, seg[t]], zd, float(t), modality, rays=_rays_t(e, t))
            outs.append(patches_to_image(o["rgb"], self.cfg.grid, self.cfg.patch))
        rec = torch.stack(outs, 1).clamp(-1, 1)
        return rec[:, 0] if _mm(modality) == "image" else rec.permute(0, 2, 1, 3, 4)

    # ------------------------------------------------------------------ train
    def forward(self, data: torch.Tensor, modality: str, cam: Optional[dict] = None) -> dict:
        cfg = self.cfg
        e = self.encode(data, modality, cam)
        frames, st = e["frames"], e["state"]
        B, T = frames.shape[:2]
        N, p = cfg.n_patches, cfg.patch
        Q = N if not self.training else min(cfg.train_queries_per_frame, N)
        seg = st["seg"]
        parts, diag = {}, {}

        rgb_l, feat_l, gates, deltas, psnr = [], [], [], [], []
        for t in range(T):
            idx = torch.rand(B, N, device=data.device).argsort(-1)[:, :Q]
            zd = e["zd"][:, t] if "zd" in e else None
            o = self.decode_frame(e["zc"][:, seg[t]], zd, float(t), modality, idx, _rays_t(e, t))
            tgt_rgb = _gather(image_to_patches(frames[:, t], p), idx)
            tgt_feat = _gather(e["feat_target"][:, t], idx)
            rgb_l.append(L.rgb_l1(o["rgb"], tgt_rgb))
            feat_l.append(L.feat_loss(o["feat"], tgt_feat))
            gates.append(o["gate"])
            deltas.append(o["delta"])
            with torch.no_grad():
                mse = ((o["rgb"].float().clamp(-1, 1) - tgt_rgb.float()) ** 2).mean()
                psnr.append(10 * torch.log10(4.0 / mse.clamp_min(1e-10)))
        parts["rgb"] = torch.stack(rgb_l).mean()
        parts["feat"] = torch.stack(feat_l).mean()

        if "zd_next" in e and T > 1:
            fut = []
            for t in range(T - 1):
                idx = torch.rand(B, N, device=data.device).argsort(-1)[:, :Q]
                o = self.decode_frame(e["zc"][:, seg[t]], e["zd_next"][:, t], float(t + 1), modality, idx,
                                      _rays_t(e, t + 1))
                fut.append(L.feat_loss(o["feat"], _gather(e["feat_target"][:, t + 1], idx)))
            parts["fut"] = torch.stack(fut).mean()

        if self.lpips is not None:
            t = T - 1
            zd = e["zd"][:, t] if "zd" in e else None
            o = self.decode_frame(e["zc"][:, seg[t]], zd, float(t), modality, rays=_rays_t(e, t))
            parts["lpips"] = self.lpips(patches_to_image(o["rgb"], cfg.grid, p), frames[:, t])

        kl = L.kl(*e["kl_c"])
        if "kl_d" in e:
            kl = kl + L.kl(*e["kl_d"])
        parts["kl"] = kl
        parts["var"] = L.var_cov(st["C"].flatten(0, 1).float())
        zs = self.semantic(e)
        parts["sem"] = L.sem_loss(zs.flatten(0, 1).float(), e["cls"].flatten(0, 1).float())
        if _mm(modality) != "image":
            parts["gate"] = L.gate_reg(torch.cat([g.flatten() for g in gates]))

        w = dict(rgb=cfg.w_rgb, feat=cfg.w_feat, fut=cfg.w_fut, lpips=cfg.w_lpips, kl=cfg.w_kl,
                 var=cfg.w_var, sem=cfg.w_sem, gate=cfg.w_gate)
        loss = sum(w[k] * v for k, v in parts.items())
        with torch.no_grad():
            diag["psnr_queries"] = torch.stack(psnr).mean()
            diag["mean_energy"] = e["mean_energy"].float()
            diag["gate_mean"] = torch.cat([g.flatten() for g in gates]).float().mean()
            diag["delta_abs_cells"] = torch.cat([d.flatten() for d in deltas]).float().abs().mean() * cfg.grid
        return dict(loss=loss, parts={k: v.detach() for k, v in parts.items()}, diag=diag)
