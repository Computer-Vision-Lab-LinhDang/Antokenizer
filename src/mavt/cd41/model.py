"""MAVT C-D 4.1: unified tokenizer with geometric transport and Δ-Tucker (design of 26/10).

forward(x, modality, cam) -> dict(loss, parts, diag[, outputs]), the interface of the v0 trainer:

  image   [B, 3, H, W]      anchor only: c, rho, kappa_0 (estimated, fills the schema; no warp)
  video   [B, 3, T, H, W]   segments of K frames; frame 0 is the anchor; camera (xi, kappa) and motion M
                            estimated from the observation and the anchor state; median anchor depth 1
  threed  [B, 3, V, H, W]   static_views: kappa and xi = log(E_i E_0^-1) copied from cam, M = 0 exactly
  world   [B, 3, T, H, W]   clips with known cameras (OmniWorld): coded as video; the given cameras and depth
                            only enter the loss (world_camera = "given": copied into the state instead)

Encoder side (may read the observations): FrameEncoder (frozen DINOv3 + pixel stem, from cd32), one trunk
transformer for the anchor and for Enc_D, and Enc_R. Decoder side (reads only the state Z and the shared
weights): AnchorSynth (H_0 = F_C(c), U = F_U(c), beta(c)), the transport W_i, the Tucker core G and channel
map Q, Dec_R and Dec_RGB. `decode(Z)` runs the decoder side alone; forward uses the same functions.

    H_i^base = W_i H_0 + (W_i U) B(a_i) Q,   B(a)[p, q] = sum_b G[p, b, q] a[b]
    e^F = F_i - sg(H^base),  e^RGB = X_i - sg(Dec_RGB(H^base)),  r_i = Enc_R(e^F, e^RGB, sg(H^base), s_i)
    F^_i = H^base + Dec_R(H^base, s_i, r_i[:L])   (Dec_R = 0 exactly when L = 0),  X^_i = Dec_RGB(F^_i)

Targets of a segment are coded independently given the anchor state (a special case of the causal order:
the code of target i reads only the anchor's state and observation i).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (gradient checkpointing of the trunk)

from ..cd32 import geometry as geo
from ..cd32.depth import cell_log_inv_depth, silog
from ..cd32.encoder import FrameEncoder
from ..cd32.layers import MLP, StreamBlock
from ..cd32.losses import edge_aware_smooth, psnr, rgb_l1
from ..cd32.regs import _pearson
from ..cd32.vanilla import DecBlock
from ..dino_v0.camera import ZeroLinear
from ..dino_v0.losses import ignore_lpips_keys
from . import state as st
from . import transport as tr
from .config import CD41Config

MODALITIES = ("image", "video", "threed", "world")
T_CELL, T_CTX, T_KAPPA, T_POSE, T_MOTION, T_A = range(6)


def _blocks(dim: int, heads: int, n: int, mlp_ratio: float = 4.0) -> nn.ModuleList:
    return nn.ModuleList([StreamBlock(dim, heads, mlp_ratio, ctx=False) for _ in range(n)])


def _run(blocks: nn.ModuleList, h: torch.Tensor, checkpoint: bool = False) -> torch.Tensor:
    for blk in blocks:
        if checkpoint and torch.is_grad_enabled():
            h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
        else:
            h = blk(h)
    return h


class Trunk(nn.Module):
    """Encoder transformer shared by the anchor encoder and Enc_D (token types tell the roles apart)."""

    def __init__(self, cfg: CD41Config):
        super().__init__()
        d, N = cfg.dim, cfg.n_cells
        p = lambda *s: nn.Parameter(torch.randn(*s) * 0.02)
        self.type_emb, self.pos_emb = p(6, d), p(1, N, d)
        self.kappa_q, self.pose_q, self.a_q = p(1, 1, d), p(1, 1, d), p(1, 1, d)
        self.motion_q = p(1, cfg.n_motion, d)
        self.blocks = _blocks(d, cfg.heads, cfg.layers, cfg.mlp_ratio)
        self.norm = nn.LayerNorm(d)
        self.use_checkpoint = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.norm(_run(self.blocks, h, self.use_checkpoint and self.training))


class CameraHead(nn.Module):
    """Token -> kappa (bounded FOV, square pixels, principal point within pp_range of the centre)."""

    def __init__(self, cfg: CD41Config):
        super().__init__()
        self.cfg = cfg
        self.lin = ZeroLinear(cfg.dim, 3)
        span = cfg.fov_max - cfg.fov_min
        p0 = (cfg.fov_init - cfg.fov_min) / span if span > 1e-6 else 0.5
        p0 = min(max(p0, 1e-4), 1 - 1e-4)
        self.bias = math.log(p0 / (1 - p0))

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        o = self.lin(tok).float()
        fov = cfg.fov_min + (cfg.fov_max - cfg.fov_min) * torch.sigmoid(o[..., 0] + self.bias)
        return tr.kappa_from_fov(fov, cfg.pp_range * torch.tanh(o[..., 1:]))


class AnchorSynth(nn.Module):
    """Decoder side, from c only: H_0 = F_C(c) [B, N, d], U = F_U(c, grid) [B, N, R_xy], beta(c) [B, N, n_motion]."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        D = cfg.dec_dim
        self.inp = nn.Linear(cfg.d_c, D)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.n_cells, D) * 0.02)
        self.blocks = _blocks(D, cfg.dec_heads, cfg.syn_blocks)
        self.norm = nn.LayerNorm(D)
        self.h0 = nn.Linear(D, feat_dim)
        self.u = nn.Linear(D, cfg.rank_xy)
        self.beta = nn.Linear(D, cfg.n_motion)

    def forward(self, c: torch.Tensor) -> dict:
        h = self.norm(_run(self.blocks, self.inp(c) + self.pos_emb))
        return dict(H0=self.h0(h), U=self.u(h), beta=torch.softmax(self.beta(h).float(), -1))


class RGBDecoder(nn.Module):
    """Dec_RGB: features [B', N, d] -> image [B', 3, H, W] (cells, then unpatchify + 3 x 3 conv)."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        D, p = cfg.dec_dim, cfg.patch
        self.cfg = cfg
        self.inp = nn.Linear(feat_dim, D)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.n_cells, D) * 0.02)
        self.blocks = _blocks(D, cfg.dec_heads, cfg.rgb_blocks)
        self.norm = nn.LayerNorm(D)
        self.pix = nn.Linear(D, p * p * cfg.pix_out)
        self.pix_conv = nn.Conv2d(cfg.pix_out, 3, 3, padding=1)

    def forward(self, Fe: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        G, p, B = cfg.grid, cfg.patch, Fe.shape[0]
        h = self.norm(_run(self.blocks, self.inp(Fe) + self.pos_emb))
        x = self.pix(h).view(B, G, G, p, p, cfg.pix_out).permute(0, 5, 1, 3, 2, 4).reshape(B, cfg.pix_out, G * p, G * p)
        return self.pix_conv(F.gelu(x))


class InnovationEncoder(nn.Module):
    """Enc_R: [e^F ; PatchEmbed(e^RGB) ; sg(H^base) ; support] on the grid, read by n_slots ordered learned
    queries -> r [B', n_slots, d_r]."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        D, N = cfg.dec_dim, cfg.n_cells
        self.ef, self.hb = nn.Linear(feat_dim, D), nn.Linear(feat_dim, D)
        self.patch = nn.Conv2d(3, D, cfg.patch, stride=cfg.patch)
        self.pos_emb = nn.Parameter(torch.randn(1, N, D) * 0.02)
        self.sup_emb = nn.Parameter(torch.randn(D) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, cfg.n_slots, D) * 0.02)
        self.blocks = _blocks(D, cfg.dec_heads, cfg.enc_r_layers)
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, cfg.d_r)

    def forward(self, eF, eRGB, Hb, support) -> torch.Tensor:
        B, N = eF.shape[:2]
        cells = (self.ef(eF) + self.hb(Hb) + self.patch(eRGB).flatten(2).transpose(1, 2) + self.pos_emb
                 + support.unsqueeze(-1).to(eF.dtype) * self.sup_emb)
        h = torch.cat([cells, self.queries.expand(B, -1, -1).to(cells.dtype)], 1)
        h = self.norm(_run(self.blocks, h))
        return self.out(h[:, N:])


class ResidualDecoder(nn.Module):
    """Dec_R: cells built from H^base (+ support) cross-attend to [null ; r[:L]] -> ΔF [B', N, d]; exactly 0
    when L = 0 (the output is masked, so no bias leaks in)."""

    def __init__(self, cfg: CD41Config, feat_dim: int):
        super().__init__()
        D, N = cfg.dec_dim, cfg.n_cells
        self.cfg = cfg
        self.inp = nn.Linear(feat_dim, D)
        self.pos_emb = nn.Parameter(torch.randn(1, N, D) * 0.02)
        self.sup_emb = nn.Parameter(torch.randn(D) * 0.02)
        self.tok_in = nn.Linear(cfg.d_r, D)
        self.slot_emb = nn.Parameter(torch.randn(1, cfg.n_slots, D) * 0.02)
        self.blocks = nn.ModuleList([DecBlock(D, cfg.dec_heads) for _ in range(cfg.dec_r_blocks)])
        self.norm = nn.LayerNorm(D)
        self.out = ZeroLinear(D, feat_dim)

    def forward(self, Hb, support, r, L) -> torch.Tensor:
        x = self.inp(Hb) + self.pos_emb + support.unsqueeze(-1).to(Hb.dtype) * self.sup_emb
        tok = self.tok_in(r.to(x.dtype)) + self.slot_emb
        keep = torch.arange(self.cfg.n_slots, device=L.device).view(1, -1) < L.view(-1, 1)
        for blk in self.blocks:
            x = blk(x, tok, keep)
        out = self.out(self.norm(x))
        return out * (L > 0).view(-1, 1, 1).to(out.dtype)


def feat_terms(pred, target) -> torch.Tensor:
    """(1 - cos) + MSE per cell [..., N]. Norms below 1 are clamped in the cosine: a cell without support has
    H^base = 0 exactly, and the cosine of a zero vector has an unbounded gradient (teacher features have
    norm ~ sqrt(d))."""
    p, t = pred.float(), target.float()
    cos = (p * t).sum(-1) / (p.norm(dim=-1).clamp_min(1.0) * t.norm(dim=-1).clamp_min(1.0))
    return 1 - cos + (p - t).pow(2).mean(-1)


def masked_feat_loss(pred, target, mask) -> torch.Tensor:
    """feat_terms averaged over the cells where mask is True. [B, N, d], mask [B, N]."""
    w = mask.float()
    return (feat_terms(pred, target) * w).sum() / w.sum().clamp_min(1.0)


class MAVTCD41(nn.Module):
    COMPILE_REGIONS = ("StreamBlock", "DecBlock")
    MODALITIES = MODALITIES

    def __init__(self, cfg: CD41Config, use_checkpoint: bool = False):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg)
        d = self.feat_dim = self.encoder.dino.out_dim
        D = cfg.dim
        self.trunk = Trunk(cfg)
        self.trunk.use_checkpoint = use_checkpoint
        self.c_head = nn.Linear(D, cfg.d_c)
        self.rho_head = MLP(D, 1.0, out=1)
        nn.init.zeros_(self.rho_head.fc2.weight)
        nn.init.zeros_(self.rho_head.fc2.bias)
        self.cam_head = CameraHead(cfg)
        self.ctx_in = nn.Linear(d, D)
        self.pose_head, self.motion_head = ZeroLinear(D, 6), ZeroLinear(D, 3)       # identity, no motion at init
        self.a_head = ZeroLinear(D, cfg.rank_a) if cfg.tucker else None
        self.synth = AnchorSynth(cfg, d)
        if cfg.tucker:
            R, Ra, Rc = cfg.rank_xy, cfg.rank_a, cfg.rank_c
            self.core = nn.Parameter(torch.randn(R, Ra, Rc) / math.sqrt(Ra))
            self.Q = nn.Parameter(torch.randn(Rc, d) / math.sqrt(Rc * R))
        self.enc_r = InnovationEncoder(cfg, d)
        self.dec_r = ResidualDecoder(cfg, d)
        self.dec_rgb = RGBDecoder(cfg, d)
        self.lpips = None
        if cfg.w_lpips > 0:
            from ..dino_v0.losses import LPIPSLoss
            self.lpips = LPIPSLoss()
        self.register_load_state_dict_post_hook(ignore_lpips_keys)

    # ------------------------------------------------------------------------------ encoder side
    @staticmethod
    def to_frames(x: torch.Tensor, modality: str) -> torch.Tensor:
        return x.unsqueeze(1) if modality == "image" else x.permute(0, 2, 1, 3, 4)

    def _encode_frames(self, frames: torch.Tensor) -> tuple:
        """frames [B, n, 3, H, W] -> (tokens [B, n, N, dim], teacher features F [B, n, N, d])."""
        B, n = frames.shape[:2]
        enc = self.encoder(frames.flatten(0, 1))
        tok = enc["x0"] + enc["mem"].sum(1)
        return tok.view(B, n, *tok.shape[1:]), enc["feat_target"].view(B, n, *enc["feat_target"].shape[1:])

    def _anchor(self, tok0: torch.Tensor) -> tuple:
        """tok0 [B, N, dim] -> (c [B, N, d_c], rho_raw [B, N], kappa0 estimated [B, 4])."""
        t = self.trunk
        B, N = tok0.shape[:2]
        h = torch.cat([tok0 + t.pos_emb + t.type_emb[T_CELL], (t.kappa_q + t.type_emb[T_KAPPA]).expand(B, -1, -1)], 1)
        h = t(h)
        return self.c_head(h[:, :N]).float(), self.rho_head(h[:, :N]).squeeze(-1).float(), self.cam_head(h[:, N])

    def _enc_d(self, tok: torch.Tensor, H0: torch.Tensor) -> dict:
        """Enc_D of the targets. tok [B, T, N, dim] observations, H0 [B, N, d] the anchor's decoded features
        (from the state) -> xi [B, T, 6], kappa [B, T, 4], M [B, T, n_motion, 3], a [B, T, R_a] or None."""
        cfg, t = self.cfg, self.trunk
        B, T, N = tok.shape[:3]
        nm = cfg.n_motion
        ctx = (self.ctx_in(H0.detach()) + t.pos_emb + t.type_emb[T_CTX]).unsqueeze(1).expand(B, T, N, -1)
        q = torch.cat([t.pose_q + t.type_emb[T_POSE], t.kappa_q + t.type_emb[T_KAPPA],
                       t.motion_q + t.type_emb[T_MOTION], t.a_q + t.type_emb[T_A]], 1)       # [1, 3 + nm, dim]
        h = torch.cat([tok + t.pos_emb + t.type_emb[T_CELL], ctx.to(tok.dtype),
                       q.unsqueeze(0).expand(B, T, -1, -1).to(tok.dtype)], 2)
        h = t(h.flatten(0, 1)).view(B, T, -1, cfg.dim)
        o = 2 * N
        out = dict(xi=self.pose_head(h[:, :, o]).float(), kappa=self.cam_head(h[:, :, o + 1]),
                   M=self.motion_head(h[:, :, o + 2:o + 2 + nm]).float(), a=None)
        if self.a_head is not None:
            out["a"] = self.a_head(h[:, :, o + 2 + nm]).float()
        return out

    # ------------------------------------------------------------------------------ decoder side
    def tucker_term(self, U: torch.Tensor, W: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """[(W_i U) B(a_i)] Q for every target. U [B, N, R_xy], W [B, T, N, N], a [B, T, R_a] -> [B, T, N, d]."""
        WU = torch.einsum("btij,bjr->btir", W.to(U.dtype), U)
        Bm = torch.einsum("pkq,btk->btpq", self.core.to(U.dtype), a.to(U.dtype))
        return torch.einsum("btnp,btpq->btnq", WU, Bm) @ self.Q.to(U.dtype)

    def base(self, syn: dict, Z: dict) -> dict:
        """H^base of every target from the state. -> dict(Hb [B, T, N, d], support [B, T, N], W, warp
        [B, T, N, d] (W H_0), delta [B, T, N, d] or None, disp [B, T, N, 3], p, ok)."""
        cfg = self.cfg
        H0, U, beta = syn["H0"], syn["U"], syn["beta"]
        B, N = H0.shape[:2]
        T = Z["xi"].shape[1]
        dev = H0.device
        if not cfg.geometry:
            W = torch.eye(N, device=dev).expand(B, T, N, N)
            support = torch.ones(B, T, N, dtype=torch.bool, device=dev)
            disp, p, ok = torch.zeros(B, T, N, 3, device=dev), None, None
        else:
            R, t = geo.pose_matrix(Z["xi"].flatten(0, 1))
            disp = tr.blend_motion(beta.unsqueeze(1).expand(B, T, N, -1).flatten(0, 1), Z["M"].flatten(0, 1))
            rep = lambda v: v.unsqueeze(1).expand(B, T, *v.shape[1:]).flatten(0, 1)
            out = tr.transport(rep(Z["rho"]), rep(Z["kappa0"]), R, t, disp, Z["kappa"].flatten(0, 1), cfg.grid,
                               cfg.knn, cfg.r_cut, cfg.sigma_u, cfg.tau_z, cfg.z_eps)
            W, support = out["W"].view(B, T, N, N), out["support"].view(B, T, N)
            disp, p, ok = disp.view(B, T, N, 3), out["p"].view(B, T, N, 2), out["ok"].view(B, T, N)
        warp = torch.einsum("btij,bjd->btid", W.to(H0.dtype), H0)
        delta = None
        if cfg.tucker and Z.get("a") is not None:
            delta = self.tucker_term(U, W, Z["a"])
        Hb = warp if delta is None else warp + delta
        return dict(Hb=Hb, support=support, W=W, warp=warp, delta=delta, disp=disp, p=p, ok=ok)

    def finish(self, syn: dict, bs: Optional[dict], Z: dict) -> dict:
        """Dec_R and Dec_RGB: -> dict(F_hat [B, n, N, d], rgb [B, n, 3, H, W], dF [B, T, N, d] or None)."""
        H0 = syn["H0"]
        B = H0.shape[0]
        feats, dF = [H0.unsqueeze(1)], None
        if bs is not None:
            T = bs["Hb"].shape[1]
            dF = self.dec_r(bs["Hb"].flatten(0, 1), bs["support"].flatten(0, 1), Z["r"].flatten(0, 1),
                            Z["L"].flatten()).view(B, T, *H0.shape[1:])
            feats.append(bs["Hb"] + dF)
        Fh = torch.cat(feats, 1)
        rgb = self.dec_rgb(Fh.flatten(0, 1))
        return dict(F_hat=Fh, rgb=rgb.view(B, Fh.shape[1], *rgb.shape[1:]), dF=dF)

    def decode(self, Z: dict) -> dict:
        """Decoder side alone: state Z (see state.py) -> dict(F_hat, rgb, Hb, support)."""
        syn = self.synth(Z["c"])
        bs = self.base(syn, Z) if Z["xi"].shape[1] > 0 else None
        out = self.finish(syn, bs, Z)
        out.update(Hb=None if bs is None else bs["Hb"], support=None if bs is None else bs["support"])
        return out

    # ------------------------------------------------------------------------------ helpers
    def sample_L(self, shape: tuple, device) -> torch.Tensor:
        cfg = self.cfg
        if not (self.training and cfg.nested):
            return torch.full(shape, cfg.n_slots, dtype=torch.long, device=device)
        choices = torch.tensor(cfg.nested_L, device=device)
        return choices[torch.randint(len(cfg.nested_L), shape, device=device)]

    def _rec(self, rgb, img, Fh, Fs, parts, diag) -> torch.Tensor:
        """Reconstruction of every observation: L1 + LPIPS (random subset) on pixels, cosine + MSE on features."""
        cfg = self.cfg
        rgb, im = rgb.flatten(0, 1), img.flatten(0, 1)
        l1, fl = rgb_l1(rgb, im), feat_terms(Fh, Fs).mean()
        parts["rec_rgb"], parts["rec_feat"] = l1.detach(), fl.detach()
        loss = cfg.w_rgb * l1 + cfg.w_feat * fl
        if self.lpips is not None:
            n = cfg.lpips_max_frames
            sel = torch.randperm(rgb.shape[0], device=rgb.device)[:n] if 0 < n < rgb.shape[0] else None
            lp = self.lpips(rgb, im) if sel is None else self.lpips(rgb[sel], im[sel])
            parts["lpips"] = lp.detach()
            loss = loss + cfg.w_lpips * lp
        return loss

    # ------------------------------------------------------------------------------ one segment
    def _segment(self, frames, mode: str, cam: Optional[dict], parts, diag, L_given=None, gt_cam=False):
        """frames [B, n, 3, H, W] (observation 0 is the anchor); mode image | video | static_views;
        cam dict(c2w [B, n, 4, 4], K [B, n, 3, 3] [, depth, dyn]) or None; gt_cam: the cameras of a video
        segment enter the loss only (world clips)."""
        cfg = self.cfg
        B, n = frames.shape[:2]
        S, dev = cfg.image_size, frames.device
        T = n - 1
        tok, Fs = self._encode_frames(frames)
        c, rho_raw, kappa_est = self._anchor(tok[:, 0])
        given = mode == "static_views" or (gt_cam and cfg.world_camera == "given")
        kappa_cam = xi_cam = None
        if cam is not None:
            kappa_cam = tr.kappa_from_K(cam["K"].float(), S, S)                              # [B, n, 4]
            xi_cam = tr.se3_log(tr.relative_pose(cam["c2w"]))                                # [B, n, 6]
        kappa0 = kappa_cam[:, 0] if given else kappa_est
        rho = rho_raw if mode == "static_views" else geo.median_normalise(rho_raw)
        syn = self.synth(c)
        Z = dict(c=c, rho=rho, kappa0=kappa0, xi=torch.zeros(B, 0, 6, device=dev))
        loss, bs = 0.0, None
        if T > 0:
            ed = self._enc_d(tok[:, 1:], syn["H0"])
            if mode == "static_views":
                Z.update(xi=xi_cam[:, 1:], kappa=kappa_cam[:, 1:], M=torch.zeros(B, T, cfg.n_motion, 3, device=dev))
            elif given:
                Z.update(xi=xi_cam[:, 1:], kappa=kappa_cam[:, 1:], M=ed["M"])
            else:
                Z.update(xi=ed["xi"], kappa=ed["kappa"], M=ed["M"])
            Z["a"] = ed["a"]
            bs = self.base(syn, Z)
            Hb, sup = bs["Hb"], bs["support"]
            Ft, Xt = Fs[:, 1:], frames[:, 1:]
            with torch.no_grad():
                rgb_b = self.dec_rgb(Hb.detach().flatten(0, 1)).view(B, T, *frames.shape[2:])
            r = self.enc_r((Ft - Hb.detach()).flatten(0, 1), (Xt - rgb_b).flatten(0, 1),
                           Hb.detach().flatten(0, 1), sup.flatten(0, 1)).view(B, T, cfg.n_slots, cfg.d_r).float()
            L = self.sample_L((B, T), dev) if L_given is None else L_given.to(dev).long().view(B, T)
            keep = torch.arange(cfg.n_slots, device=dev).view(1, 1, -1) < L.unsqueeze(-1)
            Z.update(r=r * keep.unsqueeze(-1), L=L)
            bl = masked_feat_loss(Hb.flatten(0, 1), Ft.flatten(0, 1), sup.flatten(0, 1))
            parts["base_feat"] = bl.detach()
            loss = loss + cfg.w_base * bl
            if mode != "static_views":
                fl = Z["M"].abs().mean()
                parts["flow_l1"] = fl.detach()
                loss = loss + cfg.w_flow * fl
            self._target_diag(Z, bs, syn, Fs, frames, rgb_b, cam, diag)
        out = self.finish(syn, bs, Z)
        loss = loss + self._rec(out["rgb"], frames, out["F_hat"], Fs, parts, diag)
        if gt_cam and cam is not None and cfg.world_camera == "estimated" and T > 0:
            loss = loss + self._camera_loss(Z, kappa_cam, xi_cam, parts, diag)
        if cfg.w_smooth > 0 and mode != "image":
            s = edge_aware_smooth(rho, Fs[:, 0], cfg.grid)
            parts["smooth"] = s.detach()
            loss = loss + cfg.w_smooth * s
        if cam is not None and cam.get("depth") is not None:
            tgt, mask = cell_log_inv_depth(cam["depth"].float()[:, 0], cam["depth"][:, 0] > 0, cfg.grid)
            dg = silog(rho, tgt, mask, lam=cfg.depth_gt_lam)
            parts["depth_gt"] = dg.detach()
            diag["rho_gt_corr"] = _pearson(rho.detach(), tgt, mask).mean()
            if cfg.w_depth_gt > 0:
                loss = loss + cfg.w_depth_gt * dg
        if mode != "image":
            diag["rho_std"] = rho.detach().std(-1).mean()
        diag["fov_deg"] = torch.rad2deg(2 * torch.atan(0.5 / Z["kappa0"][:, 0].detach().clamp_min(1e-6))).mean()
        rgb = out["rgb"].detach()
        diag["psnr"] = psnr(rgb.flatten(0, 1), frames.flatten(0, 1))
        diag["psnr_anchor"] = psnr(rgb[:, 0], frames[:, 0])
        L_t = Z["L"] if T > 0 else torch.zeros(B, 0, dtype=torch.long, device=dev)
        Z["h"] = st.make_header(cfg, mode, n, L_t, cam_given=given)
        diag["numbers"] = st.payload_numbers(cfg, Z).float().mean()
        if T > 0:
            diag["psnr_target"] = psnr(rgb[:, 1:].flatten(0, 1), frames[:, 1:].flatten(0, 1))
            diag["feat_cos"] = F.cosine_similarity(out["F_hat"][:, 1:].detach().float(), Fs[:, 1:].float(), -1).mean()
        return loss, dict(Z=Z, rgb=out["rgb"], F_hat=out["F_hat"], Hb=None if bs is None else bs["Hb"],
                          support=None if bs is None else bs["support"], F=Fs)

    def _target_diag(self, Z, bs, syn, Fs, frames, rgb_b, cam, diag) -> None:
        cfg = self.cfg
        Ft, Hb, sup = Fs[:, 1:].float(), bs["Hb"].detach().float(), bs["support"]
        cos = F.cosine_similarity(Hb, Ft, -1)
        diag["support_frac"] = sup.float().mean()
        if sup.any():
            diag["base_cos_sup"] = cos[sup].mean()
        if (~sup).any():
            diag["base_cos_new"] = cos[~sup].mean()
        warp = bs["warp"].detach().float()
        diag["warp_cos_sup"] = F.cosine_similarity(warp, Ft, -1)[sup].mean() if sup.any() else torch.tensor(0.0)
        if bs["delta"] is not None:
            diag["tucker_rel"] = bs["delta"].detach().float().norm(dim=-1).mean() / warp.norm(dim=-1).mean().clamp_min(1e-6)
            diag["a_norm"] = Z["a"].detach().norm(dim=-1).mean()
        diag["psnr_base"] = psnr(rgb_b.flatten(0, 1), frames[:, 1:].flatten(0, 1))
        diag["psnr_copy"] = psnr(frames[:, :1].expand_as(frames[:, 1:]).flatten(0, 1), frames[:, 1:].flatten(0, 1))
        R, _ = geo.pose_matrix(Z["xi"].detach())
        diag["xi_rot_deg"] = geo.rotation_angle_deg(R).mean()
        diag["xi_trans"] = Z["xi"][..., 3:].detach().norm(dim=-1).mean()
        if cam is not None and cam.get("dyn") is not None:
            share = F.adaptive_avg_pool2d(cam["dyn"].float()[:, 0].unsqueeze(1), cfg.grid).flatten(1)   # [B, N]
            mov = bs["disp"].detach().norm(dim=-1).mean(1)
            dyn = share >= 0.5
            diag["dyn_frac"] = dyn.float().mean()
            if dyn.any():
                diag["flow_dyn"] = mov[dyn].mean()
            if (~dyn).any():
                diag["flow_static"] = mov[~dyn].mean()

    def _camera_loss(self, Z, kappa_cam, xi_cam, parts, diag) -> torch.Tensor:
        """Estimated cameras of a video segment against the given ones (the decoder never sees the latter)."""
        R, t = geo.pose_matrix(Z["xi"])
        Rg, tg = geo.pose_matrix(xi_cam[:, 1:])
        lr = (R - Rg).abs().mean()
        lt = (t - tg).abs().mean()
        lk = (Z["kappa"] - kappa_cam[:, 1:]).abs().mean() + (Z["kappa0"] - kappa_cam[:, 0]).abs().mean()
        cl = lr + lt + lk
        parts["cam_gt"] = cl.detach()
        with torch.no_grad():
            diag["cam_rot_err_deg"] = geo.rotation_angle_deg(R.transpose(-1, -2) @ Rg).mean()
            diag["cam_trans_err"] = (t - tg).norm(dim=-1).mean()
            diag["cam_trans_gt"] = tg.norm(dim=-1).mean()
        return self.cfg.w_cam_gt * cl

    # ------------------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, modality: str, cam: Optional[dict] = None,
                L: Optional[torch.Tensor] = None, return_outputs: bool = False) -> dict:
        """x in [-1, 1]; L (eval) innovation lengths [B * segments, K - 1] or None (n_slots)."""
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")
        cfg = self.cfg
        frames = self.to_frames(x, modality)
        B, n_all = frames.shape[:2]
        parts, diag = {}, {}
        if modality in ("threed", "world") and cam is None:
            raise ValueError(f"{modality} needs cam = dict(c2w, K)")
        if modality == "image":
            loss, outputs = self._segment(frames, "image", None, parts, diag)
        elif modality == "threed":
            if n_all > cfg.keyframe_every:
                raise ValueError(f"{n_all} views: a segment holds at most keyframe_every={cfg.keyframe_every}")
            loss, outputs = self._segment(frames, "static_views", cam, parts, diag, L)
        else:
            K = cfg.keyframe_every if n_all > cfg.keyframe_every else n_all
            if n_all % K:
                raise ValueError(f"clip length {n_all} must be a multiple of keyframe_every={cfg.keyframe_every}")
            S = n_all // K
            seg = lambda t: t.reshape(B * S, K, *t.shape[2:])
            cs = None if cam is None else {k: seg(v) for k, v in cam.items() if v is not None}
            loss, outputs = self._segment(seg(frames), "video", cs, parts, diag, L, gt_cam=modality == "world")
            outputs["rgb"] = outputs["rgb"].reshape(B, n_all, *outputs["rgb"].shape[2:])
        dev = x.device
        diag = {k: (v if torch.is_tensor(v) else torch.tensor(float(v))).to(dev).float().detach() for k, v in diag.items()}
        out = dict(loss=loss, parts=parts, diag=diag)
        if return_outputs:
            out["outputs"] = outputs
        return out


def load_cd41(ckpt_path: str, device="cpu", strict: bool = True, **overrides):
    """Checkpoint of scripts/train_cd41.py -> (model in eval mode, cfg, step)."""
    from .config import config_from_dict
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck.get("cfg", {}), **{"w_lpips": 0.0, **overrides})
    model = MAVTCD41(cfg)
    sd = ck["model"] if "model" in ck else ck
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.dino.model.")]
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return model.to(device).eval(), cfg, int(ck.get("step", -1))
