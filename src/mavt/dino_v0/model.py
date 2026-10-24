"""MAVT on DINOv3: the whole model and its training losses (v0, plus the v1 phase-1 options of config.py).

  frames -> FrameEncoder (frozen DINOv3 + pixel stem) -> Trunk (in-frame + causal t-1 attention)
         -> CDState (C at keyframes, D every frame, forecasts of D) -> LatentHeads (z_r with KL) / SemanticPool (z_s)
         -> QueryDecoder (reads z_r of C_k at (u, v) + delta(D_t), reads D_t by content, gate o)

Input layout follows mavt.data: image [B, 3, H, W], video [B, 3, T, H, W], both in [-1, 1].
Posed 3D views (modality "threed") use the video layout over views plus cam = dict(c2w [B,T,4,4]
OpenCV camera-to-world, K [B,T,3,3] pixels at the model resolution): tokens and decoder queries get
Plücker rays, and the forecast of view t+h is conditioned on its pose (novel-view synthesis).

Training. Each kind of decode is ONE decoder call over rows = B * T' flattened frames, each row with
its own query time (the decoder is batch-agnostic):
  L_rec   every frame t from (C of its keyframe, D_t): rgb L1 + DINOv3 feature loss. rgb_head "linear":
          a random subset of train_queries_per_frame queries per frame (v0); "conv": all N queries,
          rgb L1 on the full image, feature loss on all queries.
  L_fut   for each h in forecast_horizons and t + h < T: D_hat(t + h), made at t from D_t, the C of t's
          segment and (posed views) the pose of view t + h, is decoded at frame t + h with the zc of t's
          segment: feature loss (w_fut) + rgb L1 (w_fut_rgb, only when > 0). Horizons weigh the same.
          fut_frames_per_horizon > 0: only that many random source frames t per horizon per step (the same
          frames for every clip of the batch); 0 decodes every pair.
  L_warp  (w_warp > 0, video / views) non-key frames decoded with the gate locked to 1, so only C read at
          (u, v) + Δ reaches the output: rgb L1. Forces Δ to carry the motion. warp_frames > 0: only that
          many random non-key frames per step.
  L_lpips / L_gram  (w_lpips / w_gram > 0) on at most lpips_max_frames full reconstructed frames, one VGG forward.
  gate    gate_loss "mean": (1 - o).mean() (v0, drives o to 1); "hinge": relu(gate_floor - mean o).
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

GRAM_DIST = "l1"          # distance between Gram matrices: "l1" | "mse"


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


def _psnr(pred: torch.Tensor, tgt: torch.Tensor, B: int) -> torch.Tensor:
    """Rows [B * T', ...] (b-major) -> mean over the T' frames of PSNR (range 2), MSE pooled over the batch."""
    mse = (pred.detach().float().clamp(-1, 1) - tgt.float()).pow(2).flatten(1).mean(1)
    mse = mse.view(B, -1).mean(0)
    return (10 * torch.log10(4.0 / mse.clamp_min(1e-10))).mean()


def _split(x: Optional[torch.Tensor], B: int, sizes: list) -> list:
    """Rows [B * sum(sizes), ...] (b-major) -> one [B * n, ...] tensor per size (None -> Nones)."""
    if x is None:
        return [None] * len(sizes)
    return [c.flatten(0, 1) for c in x.unflatten(0, (B, sum(sizes))).split(sizes, 1)]


def _subset(n: int, k: int, device) -> torch.Tensor:
    """Sorted random subset of k of range(n) (all of it when k <= 0 or k >= n). Drawn once per step and shared
    by every clip of the batch, so the decoded row counts (and the compiled shapes) are the same every step."""
    if k <= 0 or k >= n:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:k].sort().values


class MAVTDinoV0(nn.Module):
    def __init__(self, cfg: DinoV0Config, use_checkpoint: bool = False):
        super().__init__()
        assert cfg.gate_loss in ("mean", "hinge"), f"unknown gate_loss {cfg.gate_loss!r}"
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg)
        self.trunk = Trunk(cfg, use_checkpoint=use_checkpoint)
        self.cd = CDState(cfg)
        self.latent = LatentHeads(cfg)
        self.pool = SemanticPool(cfg)
        self.feat_dim = self.encoder.dino.out_dim
        self.decoder = QueryDecoder(cfg, feat_dim=self.feat_dim)
        self.lpips = L.LPIPSLoss() if cfg.w_lpips > 0 or cfg.w_gram > 0 else None
        if self.lpips is not None:                                      # frozen VGG: not in the state_dict
            self.register_load_state_dict_post_hook(L.ignore_lpips_keys)

    # ------------------------------------------------------------------ encode
    def _coords(self, T: int, modality: str, device) -> tuple:
        G, N = self.cfg.grid, self.cfg.n_patches
        coords = torch.stack([grid_coords(G, float(t), device=device) for t in range(T)])   # [T,N,4]
        mask = modality_mask(modality, N, device=device).unsqueeze(0).expand(T, -1, -1)
        return coords, mask

    def encode(self, data: torch.Tensor, modality: str, cam: Optional[dict] = None) -> dict:
        """Tiers A-D. Returns frame targets, C/D states and their latents.

        Sequences also get zd [B, T, M, zr_d], zd_next [B, T, M, zr_d] (one-step forecast made at t, v0)
        and zd_fut {h: [B, T - h, M, zr_d]} (index t = forecast made at t for frame t + h, horizons h < T;
        zd_fut[1] is zd_next[:, :T - 1], the same sample)."""
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
            out["zd_fut"] = {h: out["zd_next"][:, :T - 1] if h == 1
                             else self.latent.encode_d(D_h.flatten(0, 1))[0].unflatten(0, (B, T - h))
                             for h, D_h in st["D_fut"].items()}
        return out

    def semantic(self, e: dict) -> torch.Tensor:
        """z_s per frame [B, T, zs_dim] from Z_t = C_k U up(D_t)."""
        st = e["state"]
        B, T = e["frames"].shape[:2]
        C_t = st["C"][:, st["seg"]]                                     # [B,T,N,dim]
        D_up = None if st["D"] is None else self.cd.d_state.up(st["D"]).flatten(0, 1)
        return self.pool(C_t.flatten(0, 1), D_up).unflatten(0, (B, T))

    # ------------------------------------------------------------------ decode
    def decode_frame(self, zc: torch.Tensor, zd: Optional[torch.Tensor], t, modality: str,
                     idx: Optional[torch.Tensor] = None, rays: Optional[torch.Tensor] = None,
                     warp_only: bool = False, zd_len: Optional[torch.Tensor] = None) -> dict:
        """Decode B rows in ONE decoder call; a row is one frame, rows may come from different frames / clips.

        zc [B,N,cc], zd [B,M,zr_d] | None, t: float (every row) or tensor [B] (frame time of each row),
        idx [B,Q] | None (all N queries in row-major order; the conv rgb head needs None),
        rays [B,N,6] of the target view (posed views) | None, warp_only: gate locked to 1 (C read at
        (u, v) + Δ only), zd_len [B] long | None: each row reads only its first zd_len D tokens (nested
        dropout; 0 needs cfg.d_nested). Returns the decoder dict; "image" [B,3,H,W] is the conv head's
        output, or the patches put together when idx is None (linear head), else None."""
        mm = _mm(modality)
        B = zc.shape[0]
        if idx is not None and self.cfg.rgb_head == "conv":
            raise ValueError("rgb_head 'conv' decodes the full grid: idx must be None")
        if torch.is_tensor(t):
            qc = query_coords(self.cfg.grid, 0.0, B, idx, device=zc.device)
            qc[..., 0] = t.to(qc.device, qc.dtype).reshape(-1, 1)
        else:
            qc = query_coords(self.cfg.grid, float(t), B, idx, device=zc.device)
        qm = modality_mask(mm, qc.shape[1], device=zc.device).unsqueeze(0).expand(B, -1, -1)
        if rays is not None and idx is not None:
            rays = _gather(rays, idx)
        o = self.decoder(zc, zd, qc, qm, mm, q_rays=rays, warp_only=warp_only, zd_len=zd_len)
        if o.get("image") is None:
            o["image"] = patches_to_image(o["rgb"], self.cfg.grid, self.cfg.patch) if idx is None else None
        return o

    def _decode_set(self, e: dict, src: torch.Tensor, dst: torch.Tensor, zd: Optional[torch.Tensor],
                    subset: bool, warp_only: bool = False, zd_len: Optional[torch.Tensor] = None) -> tuple:
        """Decode frames dst [T'] of every clip in ONE decoder call, from the zc of src's [T'] segments and
        zd [B, T', M, zr_d] | None, at query time dst. subset (linear head only): a random subset of
        train_queries_per_frame queries per row (v0 training), else all N queries. zd_len [B] | None: the
        D prefix each clip reads (every frame of a clip the same).
        -> (decoder dict, targets dict(img [B*T',3,H,W], rgb [B*T',Q,p*p*3], feat [B*T',Q,F])), rows b-major."""
        cfg = self.cfg
        B, N = e["frames"].shape[0], cfg.n_patches
        zc = e["zc"][:, e["state"]["seg"][src]].flatten(0, 1)
        zd = None if zd is None else zd.flatten(0, 1)
        rows_len = None if zd_len is None or zd is None else zd_len.repeat_interleave(int(dst.numel()))
        rays = None if e["rays"] is None else e["rays"][:, dst].flatten(0, 1)
        img = e["frames"][:, dst].flatten(0, 1)
        feat = e["feat_target"][:, dst].flatten(0, 1)
        idx = None
        if subset and cfg.rgb_head == "linear" and cfg.train_queries_per_frame < N:
            idx = torch.rand(zc.shape[0], N, device=zc.device).argsort(-1)[:, :cfg.train_queries_per_frame]
        o = self.decode_frame(zc, zd, dst.float().repeat(B), e["modality"], idx, rays, warp_only, rows_len)
        rgb = image_to_patches(img, cfg.patch)
        if idx is not None:
            rgb, feat = _gather(rgb, idx), _gather(feat, idx)
        return o, dict(img=img, rgb=rgb, feat=feat)

    def _geo(self, o: dict, src: torch.Tensor, dst: torch.Tensor, geo: dict) -> tuple:
        """Decoder-shift loss of rows (b, src, dst), b-major, against geo['corr'] of keyframe (src // K) at frame dst
        (omniworld.py), plus diags (cells): epe on static / dynamic valid cells, valid shares."""
        cfg, G = self.cfg, self.cfg.grid
        k = torch.div(src, cfg.keyframe_every, rounding_mode="floor")
        tgt = geo["corr"][:, k, dst].flatten(0, 1).flatten(1, 2)                 # [B*T', N, 2]
        val = geo["valid"][:, k, dst].flatten(0, 1).flatten(1, 2)
        dyn = geo["dyncell"][:, dst].flatten(0, 1).flatten(1, 2)
        loss = L.geo_delta(o["delta"], tgt, val, G, cfg.geo_huber)
        with torch.no_grad():
            err = ((o["delta"].float() - tgt.float()) * G).norm(dim=-1)
            st, dy = val & ~dyn, val & dyn
            mean = lambda m: (err * m).sum() / m.sum().clamp_min(1)              # noqa: E731
            diag = dict(epe_static=mean(st), epe_dyn=mean(dy), valid=val.float().mean(), valid_dyn=dy.float().mean(),
                        target_cells=(tgt.float().norm(dim=-1) * G * val).sum() / val.sum().clamp_min(1))
        return loss, diag

    def _depth_loss(self, o: dict, geo: dict, B: int, T: int) -> torch.Tensor:
        """Scale-invariant loss of the depth head on the reconstruction rows against geo['depth'] [B,T,h,h]."""
        G, sub = self.cfg.grid, self.cfg.patch // 4
        gt = geo["depth"].unflatten(-2, (G, sub)).unflatten(-1, (G, sub))        # [B,T,G,sub,G,sub]
        gt = gt.permute(0, 1, 2, 4, 3, 5).reshape(B * T, -1)                    # rows, (cell, sub-cell) row-major
        return L.silog(o["depth"].flatten(1), gt)

    @staticmethod
    def _rgb_pair(o: dict, tg: dict) -> tuple:
        """(prediction, target) of the rgb L1: full images when decoded, else the query patches."""
        return (o["image"], tg["img"]) if o["image"] is not None else (o["rgb"], tg["rgb"])

    @torch.no_grad()
    def reconstruct(self, data: torch.Tensor, modality: str, cam: Optional[dict] = None) -> torch.Tensor:
        """Full reconstruction in the input layout, in [-1, 1]; all frames in one decoder call."""
        e = self.encode(data, modality, cam)
        B, T = e["frames"].shape[:2]
        tt = torch.arange(T, device=data.device)
        o, _ = self._decode_set(e, tt, tt, e.get("zd"), subset=False)
        rec = o["image"].unflatten(0, (B, T)).clamp(-1, 1)
        return rec[:, 0] if _mm(modality) == "image" else rec.permute(0, 2, 1, 3, 4)

    # ------------------------------------------------------------------ train
    def forward(self, data: torch.Tensor, modality: str, cam: Optional[dict] = None,
                geo: Optional[dict] = None) -> dict:
        """geo (modality 'world', omniworld.py): corr [B,nK,T,G,G,2], valid [B,nK,T,G,G], dyncell [B,T,G,G],
        depth [B,T,h,h]; drives the w_geo / w_depth losses when they are on."""
        cfg = self.cfg
        e = self.encode(data, modality, cam)
        frames, st = e["frames"], e["state"]
        B, T = frames.shape[:2]
        seq = e["modality"] != "image"
        tt = torch.arange(T, device=data.device)
        key = tt % cfg.keyframe_every == 0
        parts, diag = {}, {}

        # ---- nested dropout: one D prefix length per clip and step, used by every decode of the step
        zd_len = None
        if cfg.d_nested and self.training and seq and e.get("zd") is not None:
            zd_len = torch.randint(0, cfg.d_tokens + 1, (B,), device=data.device)
            mu = e["kl_d"][0].detach().float()                                      # [B*T, M, zr_d]
            en, q = mu.pow(2).mean((0, 2)), max(1, cfg.d_tokens // 4)             # energy by token order
            diag["zd_energy_head"], diag["zd_energy_tail"] = en[:q].mean(), en[-q:].mean()

        # ---- reconstruction: every frame from (C of its keyframe, D_t)
        o, tg = self._decode_set(e, tt, tt, e.get("zd"), subset=self.training, zd_len=zd_len)
        parts["rgb"] = L.rgb_l1(*self._rgb_pair(o, tg))
        parts["feat"] = L.feat_loss(o["feat"], tg["feat"])
        full = o["delta"].shape[1] == cfg.n_patches                            # every query decoded
        if geo is not None and seq and full:
            parts["geo"], d = self._geo(o, tt, tt, geo)
            diag.update({f"geo_{k}": v for k, v in d.items()})
        if geo is not None and "depth" in o:
            parts["depth"] = self._depth_loss(o, geo, B, T)

        # ---- forecasts: D_hat(t + h) made at t, decoded at frame t + h with the zc of t's segment
        if seq and e.get("zd_fut"):
            hs = list(e["zd_fut"])
            srcs = {h: _subset(T - h, cfg.fut_frames_per_horizon, data.device) for h in hs}   # source frames t
            sizes = [int(s.numel()) for s in srcs.values()]
            src = torch.cat([srcs[h] for h in hs])
            dst = torch.cat([srcs[h] + h for h in hs])
            zd_f = torch.cat([e["zd_fut"][h][:, srcs[h]] for h in hs], 1)
            of, tf = self._decode_set(e, src, dst, zd_f, subset=self.training, zd_len=zd_len)
            pred_f, tgt_f = self._rgb_pair(of, tf)
            fut, fut_rgb = [], []
            per_h = zip(hs, *(_split(x, B, sizes) for x in (of["feat"], tf["feat"], pred_f, tgt_f)))
            for h, f, f_t, p, p_t in per_h:
                fut.append(L.feat_loss(f, f_t))
                fut_rgb.append(L.rgb_l1(p, p_t))
                diag[f"psnr_fut_h{h}"] = _psnr(p, p_t, B)
            parts["fut"] = torch.stack(fut).mean()
            if geo is not None and of["delta"].shape[1] == cfg.n_patches:
                parts["geo_fut"], d = self._geo(of, src, dst, geo)
                diag["geo_fut_epe_static"], diag["geo_fut_epe_dyn"] = d["epe_static"], d["epe_dyn"]
            if cfg.w_fut_rgb > 0:
                parts["fut_rgb"] = torch.stack(fut_rgb).mean()

        # ---- warp-only: non-key frames with the gate locked to 1 (D acts through Δ alone)
        if seq and cfg.w_warp > 0 and not bool(key.all()):
            nk = tt[~key]
            nk = nk[_subset(int(nk.numel()), cfg.warp_frames, data.device)]
            ow, tw = self._decode_set(e, nk, nk, e["zd"][:, nk], subset=self.training, warp_only=True,
                                      zd_len=zd_len)
            pred_w, tgt_w = self._rgb_pair(ow, tw)
            parts["warp"] = L.rgb_l1(pred_w, tgt_w)
            diag["psnr_warp"] = _psnr(pred_w, tgt_w, B)

        # ---- LPIPS / Gram on a random subset of full reconstructed frames, one VGG forward
        if self.lpips is not None:
            sel = torch.randperm(B * T, device=data.device)[:cfg.lpips_max_frames]
            if o["image"] is not None:
                pred_img, tgt_img = o["image"][sel], tg["img"][sel]
            else:                                                       # query subset: decode those frames in full
                b, t = sel // T, sel % T
                zd = e["zd"][b, t] if "zd" in e else None
                rays = None if e["rays"] is None else e["rays"][b, t]
                pred_img = self.decode_frame(e["zc"][b, st["seg"][t]], zd, t.float(), modality, rays=rays)["image"]
                tgt_img = frames[b, t]
            lp, gr = self.lpips.lpips_gram(pred_img, tgt_img, with_gram=cfg.w_gram > 0, dist=GRAM_DIST)
            if cfg.w_lpips > 0:
                parts["lpips"] = lp
            else:
                diag["lpips"] = lp.detach()
            if gr is not None:
                parts["gram"] = gr

        kl = L.kl(*e["kl_c"])
        if "kl_d" in e:
            kl = kl + L.kl(*e["kl_d"])
        parts["kl"] = kl
        parts["var"] = L.var_cov(st["C"].flatten(0, 1).float())
        zs = self.semantic(e)
        parts["sem"] = L.sem_loss(zs.flatten(0, 1).float(), e["cls"].flatten(0, 1).float())
        if seq:
            parts["gate"] = (L.gate_hinge(o["gate"], cfg.gate_floor) if cfg.gate_loss == "hinge"
                             else L.gate_reg(o["gate"]))
        rel = self.trunk.relation_stats() if cfg.rgat_relations > 0 else None
        if rel is not None:                                             # R2 anti-collapse, weight inside
            parts["rel"] = -cfg.rgat_entropy_w * rel["marginal_entropy"]

        w = dict(rgb=cfg.w_rgb, feat=cfg.w_feat, fut=cfg.w_fut, fut_rgb=cfg.w_fut_rgb, warp=cfg.w_warp,
                 lpips=cfg.w_lpips, gram=cfg.w_gram, kl=cfg.w_kl, var=cfg.w_var, sem=cfg.w_sem,
                 gate=cfg.w_gate, rel=1.0, geo=cfg.w_geo, geo_fut=cfg.w_geo, depth=cfg.w_depth)
        loss = sum(w[k] * v for k, v in parts.items())
        with torch.no_grad():
            diag["psnr_queries"] = _psnr(o["rgb"], tg["rgb"], B)
            if o["image"] is not None:                                  # full frames were decoded
                diag["psnr"] = _psnr(o["image"], tg["img"], B)
            diag["mean_energy"] = e["mean_energy"].float()
            diag["gate_mean"] = o["gate"].float().mean()
            if rel is not None:
                diag["rel_marginal_entropy"] = rel["marginal_entropy"].detach().float()
                diag["rel_edge_entropy"] = rel["edge_entropy"].float()
            dcell = o["delta"].float().abs().mean(dim=(1, 2)) * cfg.grid        # per row [B*T]
            diag["delta_abs_cells"] = dcell.mean()
            if seq:
                k_rows = key.repeat(B)
                diag["delta_abs_cells_key"] = dcell[k_rows].mean()
                if not bool(key.all()):
                    diag["delta_abs_cells_nonkey"] = dcell[~k_rows].mean()
        return dict(loss=loss, parts={k: v.detach() for k, v in parts.items()}, diag=diag)
