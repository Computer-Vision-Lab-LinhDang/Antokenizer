"""Public decoder (section 12): coarse estimate, then grain evidence, then innovation; one operator for every
query type, with the same weights in encoder synthesis and in public decoding.

    a^(0)   = MLP_a[PE_type(q), CA(PE_type(q), [c_0; z_i]), PoolGeom(C, D_i)]            (query_init cond)
    k_ik    = W_K[f~_ik; PE(P^can_ik); PE(u_k)],  v_ik = W_V f~_ik
    Q^(l)   = W_Q LN(a^(l)) + W_PE PE_type(q),    e^(l) = Attn(Q^(l), K_i, V_i; B^geom)
    a^(l+1) = a^(l) + MLP_ref[a^(l), e^(l), PE_type(q)],     F^base = W_out a^(2),  F^init = W_out a^(0)
    B^geom  = log nu,  nu_qk ∝ exp(-|16 (u_q - u^_k)|^2 / (2 sigma_k^2)) exp(-d_k / tau_d) over grains in front
    dF      = 1[L > 0] f_out(CA(f_q[a^(2), PE(q)], Embed(r_i); B^r)),  B^r = -|16 (u_q - u_ij)|^2 / (2 sigma_ij^2)
    RGB     = sigmoid(conv(up x2 conv ... (W F)))  on the 16 x 16 x D_t feature grid

2026-10 flags (config.py, all default off; see README "Flags 2026-10"):
    read_coarse_kv   K = [K_fine ; K_coarse (16 slots of [c_0; z_i] + PE_slot, own LN / W_K / W_V) ; sink],
                     logit_coarse = exp(tau_c) q.k_c / sqrt(d_h) + beta_c - proximity(u_q): the slots attract
                     only where no grain covers the query (proximity ~ 0 under a grain, << 0 elsewhere).
    read_relpos_value v_j(q) = W f_j + sum_{r<=R} phi_r(D_qj) taper(|D_qj|) W_r f_j, D_qj = grid (u_q - u^_j) / sigma_j,
                     phi: R/2 Fourier directions x (sin, cos) at |omega| = pi/2, taper = Wendland C2 to 0 at 3 sigma.
                     Low rank: R attention-weighted sums with weights a_qj phi_r(D_qj); nothing of size [B, Q, G, d].
    rgb_coarse_start X~0 = pyr_up(sigmoid(RGBLow(a0))) from the cell grid; X^ = clamp(sg[X~0] + R(H)), R zero-init.
    rgb_head=pointwise  1x1 MLP blocks (per-point LayerNorm) on the query grid, PixelShuffle x2 + conv3x3 stages,
                     receptive field <= 16 px at the image (see PointwiseRGB).
    read_split_layer2 the last ReadLayer exists twice: a_F (feature grid) and a_X (RGB grid, `branch="X"`).
"""
from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as g
from .config import GrainPriorConfig
from .layers import BIAS_FLOOR, CrossAttn, attend, mlp2

N_POOLGEOM = 18


def local_support(uq, uhat, sigma, valid, grid):
    """Diagnostic v2: on-screen, finite front-facing grain within 3 kernel radii.

    Evidence locality is not opacity or calibrated confidence, and never an eval mask.
    """
    on_screen = valid & torch.isfinite(uhat).all(-1) & ((uhat >= 0) & (uhat <= 1)).all(-1)
    d2 = ((grid * (uq.float().unsqueeze(2) - uhat.float().unsqueeze(1))) ** 2).sum(-1)
    return (on_screen.unsqueeze(1) & (d2 <= 9 * sigma.float().unsqueeze(1).square())).any(-1)


def proximity(uq, uhat, sigma, valid, grid) -> torch.Tensor:
    """log sum_k exp(-|u_q - u^_k|^2 / 2 s_k^2) over valid grains, capped at 0 [B, Q]: about 0 where a grain
    covers the query, far below where none does (no valid grain: -1e4). Added to the normalised geometry bias it
    lets the CRSR sink compare against the grain mass at the query, independent of the scene depth scale."""
    with g.fp32(uq.device):
        d2 = ((grid * (uq.float().unsqueeze(2) - uhat.float().unsqueeze(1))) ** 2).sum(-1)
        ln = (-d2 / (2 * sigma.float().unsqueeze(1) ** 2)).masked_fill(~valid.unsqueeze(1), float("-inf"))
        lse = torch.logsumexp(ln, -1)
        return torch.where(torch.isfinite(lse), lse.clamp_max(0.0), torch.full_like(lse, -1e4))


def projected_radius(sigma, d_anchor, z_target, kappa0, kappa, clip: float = 0.0) -> torch.Tensor:
    """Grain radius on the target image: sigma (anchor cells) x (d_anchor / z_target) x (f_target / f_anchor);
    clip > 0 bounds that scale to [1 / clip, clip] (a grain near the target camera would cover the image)."""
    f = lambda k: k[:, :2].float().prod(-1).sqrt().unsqueeze(-1)
    s = d_anchor.float() / z_target.float().clamp_min(1e-6) * f(kappa) / f(kappa0)
    return sigma.float() * (s.clamp(1 / clip, clip) if clip > 0 else s)


def pool_geom(xi, kappa, kappa0, rho, M, valid) -> torch.Tensor:
    """PoolGeom(C, D_i): pooled geometric metadata of the state at observation i [B, 18] (no target data)."""
    return torch.cat([xi, kappa, kappa0, rho.mean(-1, keepdim=True), rho.std(-1, keepdim=True),
                      M.norm(dim=-1).mean(-1, keepdim=True), valid.float().mean(-1, keepdim=True)], -1).float()


def geom_bias(u_q: torch.Tensor, u_hat: torch.Tensor, sigma: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor,
              grid: int, tau: float) -> torch.Tensor:
    """log nu [B, Q, G]: normalised over the valid grains of each query (rows without one: 0)."""
    with g.fp32(u_q.device):
        d2 = ((grid * (u_q.float().unsqueeze(2) - u_hat.float().unsqueeze(1))) ** 2).sum(-1)
        ln = -d2 / (2 * sigma.float().unsqueeze(1) ** 2) - depth.float().unsqueeze(1) / tau
        ln = ln.masked_fill(~valid.unsqueeze(1), float("-inf"))
        lse = torch.logsumexp(ln, -1, keepdim=True)
        ok = torch.isfinite(lse)
        return torch.where(ok & valid.unsqueeze(1), ln - torch.where(ok, lse, torch.zeros_like(lse)),
                           torch.zeros_like(ln))


def relpos_basis(u_q: torch.Tensor, u_hat: torch.Tensor, sigma: torch.Tensor, valid: torch.Tensor, grid: int,
                 R: int) -> torch.Tensor:
    """2.2: phi [B, Q, G, R] in fp32. D = grid (u_q - u^_j) / sigma_j (patches over radii); for k < R / 2 the
    direction omega_k = (pi / 2) (cos pi k / (R/2), sin pi k / (R/2)); phi_k = sin(omega_k . D), phi_{R/2 + k} =
    cos(omega_k . D); every basis is multiplied by the Wendland C2 taper (1 - rho / 3)^4 (1 + 4 rho / 3), rho = |D|,
    which is 0 beyond 3 sigma. Invalid grains give phi = 0."""
    with g.fp32(u_q.device):
        d = grid * (u_q.float().unsqueeze(2) - u_hat.float().unsqueeze(1)) / sigma.float().unsqueeze(1).unsqueeze(-1)
        d = torch.where(valid.unsqueeze(1).unsqueeze(-1), d, torch.zeros_like(d))
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        rho = (d.norm(dim=-1) / 3.0).clamp_max(1.0)
        taper = (1 - rho) ** 4 * (1 + 4 * rho) * valid.unsqueeze(1).float()
        k = torch.arange(R // 2, device=u_q.device, dtype=torch.float32) * math.pi / (R // 2)
        omega = (math.pi / 2) * torch.stack([k.cos(), k.sin()], -1)                        # [R/2, 2]
        proj = d @ omega.t()                                                               # [B, Q, G, R/2]
        return torch.cat([proj.sin(), proj.cos()], -1) * taper.unsqueeze(-1)


def pyr_down(x: torch.Tensor, n: int) -> torch.Tensor:
    """Gaussian pyramid: n times (binomial 5-tap [1 4 6 4 1] / 16, reflect padding, stride 2)."""
    k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=torch.float32) / 16
    k = (k1[:, None] * k1[None, :]).view(1, 1, 5, 5).expand(x.shape[1], 1, 5, 5)
    x = x.float()
    for _ in range(n):
        x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k, stride=2, groups=x.shape[1])
    return x


def pyr_up(x: torch.Tensor, n: int) -> torch.Tensor:
    """Gaussian pyramid expand with the same kernel: n times (zero insertion x2, conv 4 [1 4 6 4 1] / 16 x same)."""
    k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device, dtype=torch.float32) / 16
    k = 4 * (k1[:, None] * k1[None, :]).view(1, 1, 5, 5).expand(x.shape[1], 1, 5, 5)
    x = x.float()
    for _ in range(n):
        B, C, H, W = x.shape
        z = x.new_zeros(B, C, 2 * H, 2 * W)
        z[:, :, ::2, ::2] = x
        x = F.conv2d(F.pad(z, (2, 2, 2, 2), mode="reflect"), k, groups=C)
    return x


def pyramid_target(X: torch.Tensor, n: int) -> torch.Tensor:
    """2.3 target of X~0: pyr_up(pyr_down(X, n), n), the same up operator as X~0 itself."""
    return pyr_up(pyr_down(X, n), n)


class ReadLayer(nn.Module):
    def __init__(self, cfg: GrainPriorConfig, d_pe: int):
        super().__init__()
        d = cfg.dim
        self.heads = cfg.heads
        self.norm = nn.LayerNorm(d)
        self.wq, self.wpe = nn.Linear(d, d), nn.Linear(d_pe, d, bias=False)
        self.nk, self.nv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wk, self.wv, self.wo = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ref = mlp2(3 * d, d, cfg.mlp_hidden)
        self.strict = cfg.readout_version != "legacy" or cfg.read_strict
        sink_on = cfg.read_sink or (cfg.read_coarse_kv and cfg.coarse_sink)
        if self.strict:
            self.temperature = nn.Parameter(torch.zeros(()))
        elif cfg.read_qk_norm:
            self.qn, self.kn = nn.LayerNorm(d // cfg.heads), nn.LayerNorm(d // cfg.heads)
        if sink_on:                               # zero key and value, logit learned: starts at a few % of the read
            self.sink = nn.Parameter(torch.tensor(-4.0))
        self.coarse_kv = cfg.read_coarse_kv
        if self.coarse_kv:                        # 2.1: own projections, learned temperature and offset of the slots
            self.nk_c, self.nv_c = nn.LayerNorm(d), nn.LayerNorm(d)
            self.wk_c, self.wv_c = nn.Linear(d, d), nn.Linear(d, d)
            self.tau_c = nn.Parameter(torch.zeros(()))
            self.beta_c = nn.Parameter(torch.tensor(-4.0))
        self.relpos = cfg.read_relpos_value
        if self.relpos:                           # 2.2: R value projections W_r, zero at init (= the plain read)
            self.R = cfg.relpos_rank
            self.wv_rel = nn.Linear(d, self.R * d, bias=False)
            nn.init.zeros_(self.wv_rel.weight)
        self.collect_diag = False                 # set by Readout on the last layer: shares / effective support
        self.last_diag = None

    def _head_norm(self, x, ln):
        return ln(x.unflatten(-1, (self.heads, -1))).flatten(-2) if hasattr(self, "qn") else x

    def forward(self, a, pe, ktok, vtok, bias, valid, ctok=None, cbias=None, phi=None):
        """a, pe [B, Q, d]; grain tokens ktok / vtok [B, G, d]; bias [B, 1, Q, G] or None; valid [B, G];
        ctok [B, n_c, d] + cbias [B, Q, n_c] (2.1); phi [B, Q, G, R] (2.2)."""
        Q = self._head_norm(self.wq(self.norm(a)) + self.wpe(pe), getattr(self, "qn", None))
        K = self._head_norm(self.wk(self.nk(ktok)), getattr(self, "kn", None))
        V = self.wv(self.nv(vtok))
        n_fine = K.shape[1]
        if self.strict:
            norm = lambda x: F.normalize(x.unflatten(-1, (self.heads, -1)), dim=-1).flatten(-2)
            Q = norm(Q) * (8 * self.temperature.sigmoid()) * (Q.shape[-1] // self.heads) ** .5
            K = norm(K)
        Vr = self.wv_rel(self.nv(vtok)) if self.relpos else None
        n_coarse = 0
        if self.coarse_kv:
            if ctok is None or bias is None:
                raise ValueError("read_coarse_kv needs the coarse tokens and the geometry bias")
            Kc = self.wk_c(self.nk_c(ctok))
            Kc = (F.normalize(Kc.unflatten(-1, (self.heads, -1)), dim=-1).flatten(-2) if self.strict else Kc)
            Kc = Kc * torch.exp(self.tau_c).to(Kc.dtype)
            Vc = self.wv_c(self.nv_c(ctok))
            n_coarse = Kc.shape[1]
            K, V = torch.cat([K, Kc.to(K.dtype)], 1), torch.cat([V, Vc.to(V.dtype)], 1)
            cb = (cbias.float() + self.beta_c).unsqueeze(1).expand(-1, bias.shape[1], -1, -1)
            bias = torch.cat([bias.float(), cb], -1)
            valid = torch.cat([valid, valid.new_ones(valid.shape[0], n_coarse)], -1)
        has_sink = hasattr(self, "sink")
        if has_sink:
            K, V, bias, valid = self._with_sink(K, V, bias, valid)
            if getattr(self, "track_sink", False):
                self.last_sink_share = self._sink_share(Q, K, bias, valid)
        strict = self.strict or has_sink or self.coarse_kv
        if self.relpos:
            e, p = self._attend_explicit(Q, K, V, bias, valid, strict, phi, Vr, n_fine)
        else:
            e = attend(Q, K, V, self.heads, bias, valid, strict=strict)
            p = None
        e = self.wo(e)
        if strict and not (has_sink or self.coarse_kv):
            e = e * valid.any(-1)[:, None, None].to(e.dtype)
        if self.collect_diag:
            self.last_diag = self._diag(Q, K, bias, valid, strict, p, n_fine, n_coarse, has_sink)
        return a + self.ref(torch.cat([a, e, pe], -1))

    def _logits(self, Q, K, bias, valid, strict):
        """Explicit logits [B, H, nq, ne] in fp32 with the semantics of layers.attend (mask, floor, has)."""
        B, nq, d = Q.shape
        dh = d // self.heads
        sh = lambda t: t.float().unflatten(-1, (self.heads, dh)).transpose(1, 2)
        logit = sh(Q) @ sh(K).transpose(-1, -2) / math.sqrt(dh)
        ok = valid.view(B, 1, 1, -1)
        has = ok.any(-1, keepdim=True)
        ok = ok | ~has
        if bias is not None:
            b = bias.float() if strict else bias.float().clamp_min(BIAS_FLOOR)
            if strict:
                b = torch.where(has, b, torch.zeros_like(b))
            logit = logit + b
        return logit.masked_fill(~ok, float("-inf")), has

    def _attend_explicit(self, Q, K, V, bias, valid, strict, phi, Vr, n_fine):
        """2.2 read with position-dependent values. a = softmax(logits) [B, H, nq, ne];
        e = a V + sum_r (a[:, :, :, :G] * phi_r) V_r with V_r = W_r f (heads split): R weighted sums, each
        temporary of size [B, H, nq, G]; no tensor of size [B, nq, G, d] exists."""
        B, nq, d = Q.shape
        dh = d // self.heads
        logit, has = self._logits(Q, K, bias, valid, strict)
        p = logit.softmax(-1)
        sh = lambda t: t.float().unflatten(-1, (self.heads, dh)).transpose(1, 2)
        out = p @ sh(V)
        if Vr is not None and phi is not None:
            pf = p[..., :n_fine]
            for r in range(self.R):
                vr = sh(Vr[..., r * d:(r + 1) * d])
                out = out + (pf * phi[..., r].unsqueeze(1)) @ vr
        out = out * has
        return out.transpose(1, 2).reshape(B, nq, d).to(V.dtype), p

    @torch.no_grad()
    def _diag(self, Q, K, bias, valid, strict, p, n_fine, n_coarse, has_sink) -> dict:
        """3.1: per-query effective support 1 / sum_j a_j^2 and the fine / coarse / sink shares (mean over heads),
        per-grain attention mass (sum over queries); fp32 probabilities [B, H, Q, ne] without gradient."""
        if p is None:
            p = self._logits(Q, K, bias, valid, strict)[0].softmax(-1)
        p = p.float()
        n_c = n_fine + n_coarse
        return dict(eff_support_q=(1.0 / p.pow(2).sum(-1).clamp_min(1e-12)).mean(1),
                    fine_share_q=p[..., :n_fine].sum(-1).mean(1),
                    coarse_share_q=p[..., n_fine:n_c].sum(-1).mean(1),
                    sink_share_q=(p[..., n_c:].sum(-1).mean(1) if has_sink else torch.zeros_like(p[:, 0, :, 0])),
                    grain_mass=p[..., :n_fine].mean(1).sum(1))

    def _with_sink(self, K, V, bias, valid):
        """Append the sink: zero key (content logit 0 after the q/k normalisation) and zero value, logit = sink."""
        B, Q = K.shape[0], (bias.shape[-2] if bias is not None else None)
        z = K.new_zeros(B, 1, K.shape[-1])
        K, V = torch.cat([K, z], 1), torch.cat([V, z.to(V.dtype)], 1)
        valid = torch.cat([valid, valid.new_ones(B, 1)], -1)
        if bias is None:
            raise ValueError("read_sink needs the geometry bias")
        bias = torch.cat([bias, self.sink.to(bias.dtype).expand(*bias.shape[:-1], 1)], -1)
        return K, V, bias, valid

    @torch.no_grad()
    def _sink_share(self, Q, K, bias, valid) -> float:
        """Mean attention weight on the sink (diagnostic; explicit softmax, so only on small batches)."""
        sh = lambda t: t.float().unflatten(-1, (self.heads, -1)).transpose(1, 2)
        logit = sh(Q) @ sh(K).transpose(-1, -2) / math.sqrt(Q.shape[-1] // self.heads) + bias.float()
        logit = logit.masked_fill(~valid[:, None, None, :], float("-inf"))
        return logit.softmax(-1)[..., -1].mean().item()


class PointLN(nn.Module):
    """LayerNorm over the channel axis of an NCHW map at every point (no spatial statistics)."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PointMLP(nn.Module):
    """x + W_2 GELU(W_1 LN(x)) with 1 x 1 convolutions."""

    def __init__(self, c: int, ratio: float):
        super().__init__()
        h = int(c * ratio)
        self.norm, self.fc1, self.fc2 = PointLN(c), nn.Conv2d(c, h, 1), nn.Conv2d(h, c, 1)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class PointwiseRGB(nn.Module):
    """3.3: query-grid state [B, dim, s, s] -> residual image [B, 3, S, S] (no sigmoid; the caller composes).

    dim-channel map at the query grid -> 4 x PointMLP (1 x 1, per-point LN) -> for each x2 stage k (n_up stages):
    PixelShuffle(2) (channels / 4, pointwise) -> conv3x3 (k = 0: one; k >= 1: two) -> conv1x1 -> 3 (zero-init).
    Reference (dim 512, grid 64, image 256): 512 @64 -> 128 @128 (1 conv3x3) -> 32 @256 (2 conv3x3) -> 3.
    Receptive field at the output: the 1x1 blocks and PixelShuffle mix nothing spatially; conv3x3 @128 spans
    +-1 px @128 = +-2 px @256; two conv3x3 @256 span +-2 px: total +-4 px, a 9 px window <= 16 px.
    Capacity sits in the 1x1 MLPs (ratio 4: 4 x 2 x 512 x 2048 = 8.4 M of the ~8.8 M parameters).
    """

    def __init__(self, dim: int, n_up: int, ratio: float):
        super().__init__()
        self.blocks = nn.Sequential(*[PointMLP(dim, ratio) for _ in range(4)])
        stages, c = [], dim
        for k in range(n_up):
            c = c // 4
            convs = [nn.PixelShuffle(2), nn.Conv2d(c, c, 3, padding=1), nn.GELU()]
            if k >= 1:
                convs += [nn.Conv2d(c, c, 3, padding=1), nn.GELU()]
            stages.append(nn.Sequential(*convs))
        self.stages = nn.Sequential(*stages)
        self.final = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        return self.final(self.stages(self.blocks(x)))


class RGBHead(nn.Module):
    """Feature grid [B, G*G, D_t] -> image [B, 3, 16 G, 16 G] in [0, 1]; on cfg.query_grid Gq: one stride-2
    upsampling per factor of 2 up to the image (the first rgb_channels).

    rgb_coarse_start (2.3): forward(Fh, low=low_image(a0)) returns clamp(sg[X~0] + R(Fh), 0, 1) where R is the
    head's last layer output (zero at init, no sigmoid). rgb_head=pointwise (3.3) swaps the conv trunk for
    PointwiseRGB. rgb_in_feat (3.3): the head input is H + W up_bilinear(sg[F^]) (F^ on the feature grid)."""

    def __init__(self, cfg: GrainPriorConfig, teacher_dim: int):
        super().__init__()
        self.cfg = cfg
        self.side = cfg.query_grid or cfg.grid
        n_up = (cfg.image_size // self.side).bit_length() - 1
        self.proj = nn.Linear(teacher_dim, cfg.dim)
        self.pointwise = cfg.rgb_head == "pointwise"
        if self.pointwise:
            self.pw = PointwiseRGB(cfg.dim, n_up, cfg.rgb_pw_mlp_ratio)
        else:
            ch = [cfg.dim] + list(cfg.rgb_channels)[:n_up]
            self.convs = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 3, padding=1) for i in range(n_up)])
            self.final = nn.Conv2d(ch[-1], 3, 3, padding=1)
        self.coarse = cfg.rgb_coarse_start
        if self.coarse:
            self.low = mlp2(cfg.dim, 3, cfg.mlp_hidden)                 # RGBLow on a^(0) of the cell grid
            self.n_pyr = (cfg.image_size // cfg.grid).bit_length() - 1
            last = self.final if not self.pointwise else None
            if last is not None:                                         # pointwise: zero-init already
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        if cfg.rgb_in_feat:
            self.in_feat = nn.Linear(teacher_dim, cfg.dim)
            nn.init.zeros_(self.in_feat.weight)
            nn.init.zeros_(self.in_feat.bias)

    def low_image(self, a0: torch.Tensor) -> torch.Tensor:
        """2.3: X~0 [B, 3, S, S] in (0, 1) = pyr_up(sigmoid(RGBLow(a0)) on the cell grid)."""
        B = a0.shape[0]
        G = self.cfg.grid
        x = torch.sigmoid(self.low(a0.float()).float()).transpose(1, 2).reshape(B, 3, G, G)
        return pyr_up(x, self.n_pyr).clamp(0, 1)

    def residual(self, Fh: torch.Tensor, feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pre-activation output R [B, 3, S, S] of the trunk (legacy: before the sigmoid)."""
        B, N, _ = Fh.shape
        G = self.side
        x = self.proj(Fh)
        if feat is not None and hasattr(self, "in_feat"):
            Gf = int(round(math.sqrt(feat.shape[1])))
            f = self.in_feat(feat.detach().to(Fh.dtype)).transpose(1, 2).reshape(B, -1, Gf, Gf)
            f = F.interpolate(f, size=(G, G), mode="bilinear", align_corners=False)
            x = x + f.flatten(2).transpose(1, 2)
        x = x.transpose(1, 2).reshape(B, -1, G, G)
        if self.pointwise:
            return self.pw(x).float()
        for conv in self.convs:
            x = F.gelu(conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)))
        return self.final(x).float()

    def forward(self, Fh: torch.Tensor, low: Optional[torch.Tensor] = None, feat: Optional[torch.Tensor] = None):
        r = self.residual(Fh, feat)
        if self.coarse:
            if low is None:
                raise ValueError("rgb_coarse_start needs low = low_image(a0)")
            return (low.detach().float() + r).clamp(0, 1)
        return torch.sigmoid(r)


class Readout(nn.Module):
    def __init__(self, cfg: GrainPriorConfig, teacher_dim: int):
        super().__init__()
        self.cfg = cfg
        d, pf, dc = cfg.dim, cfg.pe_freq, cfg.d_coarse
        pe2, pe3 = g.pe_dim(2, pf), g.pe_dim(3, pf)
        self.q_pe = nn.Linear(pe2, d)
        self.q_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)                        # pixel query type
        # ---- a^(0)
        if cfg.read_coarse_init:                  # CRSR: bilinear read of c_0 and z_i at u_q
            self.coarse_in = nn.Linear(2 * dc, d)
        else:
            self.mem_tok = nn.Linear(dc, d)
            self.mem_type = nn.Parameter(torch.randn(2, d) * 0.02)                     # c_0, z_i
            self.mem_slot = nn.Parameter(torch.randn(1, cfg.n_coarse, d) * 0.02)
            self.mem_grid = nn.Linear(pe2, d)
            self.init_ca = CrossAttn(d, cfg.heads)
        self.poolgeom = mlp2(N_POOLGEOM, d, cfg.mlp_hidden)
        self.mlp_a = mlp2(3 * d, d, cfg.mlp_hidden)
        self.uncond = nn.Parameter(torch.randn(1, cfg.n_cells, d) * 0.02)
        # ---- grain reads
        self.ktok = nn.Linear(cfg.d_f + pe3 + pe2, d)
        self.vtok = nn.Linear(cfg.d_f, d)
        self.layers = nn.ModuleList([ReadLayer(cfg, d) for _ in range(cfg.read_layers)])
        if cfg.read_split_layer2:                 # 3.4: second copy of the last layer for the RGB-grid read
            self.last_x = ReadLayer(cfg, d)
        if cfg.read_coarse_kv:                    # 2.1: coarse slot tokens [c_0; z_i] + slot PE, shared by the layers
            self.coarse_kv_tok = nn.Linear(2 * dc, d)
            self.coarse_kv_pe = nn.Linear(pe2, d)
        self.w_out = nn.Linear(d, teacher_dim)
        self.depth = mlp2(d, 1, cfg.mlp_hidden)
        # ---- innovation read
        self.f_q = mlp2(2 * d, d, cfg.mlp_hidden)
        self.r_tok = nn.Linear(cfg.d_e + pe2 + 2, d)
        self.r_ca = CrossAttn(d, cfg.heads)
        self.f_out = nn.Linear(d, teacher_dim, bias=False)
        self.register_buffer("uq", g.cell_centres(cfg.grid), persistent=False)
        self.register_buffer("ucoarse", g.cell_centres(cfg.coarse_grid), persistent=False)

    def query_pe(self, B: int, uq=None) -> torch.Tensor:
        if uq is None:
            return (self.q_pe(g.fourier(self.uq, self.cfg.pe_freq)).unsqueeze(0) + self.q_type).expand(B, -1, -1)
        return self.q_pe(g.fourier(uq, self.cfg.pe_freq)) + self.q_type

    def init(self, pe, c0, z, pg, uq=None) -> torch.Tensor:
        cfg = self.cfg
        B = pe.shape[0]
        if cfg.query_init == "uncond":
            uncond = self.uncond.expand(B, -1, -1)
            return (uncond if uq is None else g.read_coarse(uncond, uq, cfg.grid)) + pe
        pg_q = self.poolgeom(pg).to(pe.dtype).unsqueeze(1).expand(-1, pe.shape[1], -1)
        if cfg.read_coarse_init:
            u = self.uq.expand(B, -1, -1) if uq is None else uq
            cz = torch.cat([g.read_coarse(c0.float(), u, cfg.coarse_grid), g.read_coarse(z.float(), u, cfg.coarse_grid)], -1)
            return self.mlp_a(torch.cat([pe, self.coarse_in(cz).to(pe.dtype), pg_q], -1))
        grid = self.mem_grid(g.fourier(self.ucoarse, cfg.pe_freq)).unsqueeze(0)
        tok = torch.cat([self.mem_tok(c0.float()) + self.mem_type[0], self.mem_tok(z.float()) + self.mem_type[1]], 1)
        tok = tok + (self.mem_slot + grid).repeat(1, 2, 1)
        ca = self.init_ca(pe, tok.to(pe.dtype))
        return self.mlp_a(torch.cat([pe, ca, pg_q], -1))

    def base(self, f_t, P_can, u_k, u_hat, z_cam, sigma, valid, scale, c0, z, pg, uq=None, branch="F",
             grain_keep=None) -> dict:
        """Grains at observation i (f~ [B, G, d_f], P^can, anchor u_k, projection u^ / z / valid, sigma), depth
        scale of the gauge [B], c_0, z_i, PoolGeom -> dict(a0, a, F_base, F_init, depth_grain, depth_dense,
        support [B, Q], diag (3.1: eff_support_q, fine/coarse/sink_share_q [B, Q], grain_mass [B, G])).
        branch "X" with read_split_layer2 runs the RGB copy of the last layer; grain_keep [B, G] bool (evaluation:
        drop grains) is intersected with valid."""
        cfg = self.cfg
        B = f_t.shape[0]
        if grain_keep is not None:
            valid = valid & grain_keep
        pe = self.query_pe(B, uq)
        a0 = self.init(pe, c0, z, pg, uq)
        sc = scale.float().view(B, 1, 1)
        kt = self.ktok(torch.cat([f_t.float(), g.fourier(P_can.float() / sc, cfg.pe_freq),
                                  g.fourier(u_k.float(), cfg.pe_freq)], -1)).to(a0.dtype)
        vt = self.vtok(f_t.float()).to(a0.dtype)
        u_q = self.uq.expand(B, -1, -1) if uq is None else uq
        dd = z_cam.float() / sc.view(B, 1) if cfg.tau_d_mode == "median" else z_cam.float()
        lnu = geom_bias(u_q, u_hat, sigma, dd, valid, cfg.grid, cfg.tau_d)              # [B, Q, G]
        sink_on = cfg.read_sink or (cfg.read_coarse_kv and cfg.coarse_sink)
        prox = proximity(u_q, u_hat, sigma, valid, cfg.grid) if (sink_on or cfg.read_coarse_kv) else None
        if sink_on:                               # the sink competes with the grain mass at the query
            lnu = lnu + prox.unsqueeze(-1)
        bias = lnu.unsqueeze(1) if cfg.geom_bias else None
        ctok = cbias = phi = None
        if cfg.read_coarse_kv:                    # 2.1: logit_c = beta_c - proximity(u_q) (+ content), 16 slots
            ctok = (self.coarse_kv_tok(torch.cat([c0.float(), z.float()], -1))
                    + self.coarse_kv_pe(g.fourier(self.ucoarse, cfg.pe_freq)).unsqueeze(0)).to(a0.dtype)
            cbias = (-prox).clamp(max=50.0).unsqueeze(-1).expand(-1, -1, cfg.n_coarse)
        if cfg.read_relpos_value:                 # 2.2
            phi = relpos_basis(u_q, u_hat, sigma, valid, cfg.grid, cfg.relpos_rank)
        layers = list(self.layers)
        if branch == "X" and cfg.read_split_layer2:
            layers[-1] = self.last_x
        a = a0
        for i, layer in enumerate(layers):
            layer.collect_diag = i == len(layers) - 1
            a = layer(a, pe, kt, vt, bias, valid, ctok=ctok, cbias=cbias, phi=phi)
        diag = layers[-1].last_diag
        with g.fp32(a.device):
            nu = lnu.exp() * valid.unsqueeze(1)
            support = valid.any(-1, keepdim=True).expand(-1, lnu.shape[1])
            far = z_cam.float().masked_fill(~valid, 0).amax(-1, keepdim=True).clamp_min(1.0)
            d_grain = torch.where(support, (nu * z_cam.float().unsqueeze(1)).sum(-1), far.expand_as(support).float())
            d_dense = torch.exp(self.depth(a.float()).squeeze(-1).float().clamp(-20, 20)) * scale.float().view(B, 1)
        support_v2 = local_support(u_q, u_hat, sigma, valid, cfg.grid)
        return dict(a0=a0, a=a, pe=pe, F_base=self.w_out(a).float(), F_init=self.w_out(a0).float(),
                    depth_grain=d_grain, depth_dense=d_dense,
                    support=support if cfg.measurement_version == 1 else support_v2,
                    support_v1=support, support_v2=support_v2, diag=diag)

    def innovation(self, a: torch.Tensor, pe: torch.Tensor, r: torch.Tensor, L: torch.Tensor,
                   return_hidden: bool = False, uq=None):
        """dF [B, Q, D_t] from the own-view innovation slots r [B, n_slots, d_r] (prefix L [B]); exactly 0 at L = 0.
        return_hidden: also the state increment dH [B, Q, dim] (the input of f_out), 0 at L = 0."""
        cfg = self.cfg
        B = a.shape[0]
        rf = r.float()
        tok = self.r_tok(torch.cat([rf[..., 4:], g.fourier(rf[..., :2], cfg.pe_freq), rf[..., 2:4]], -1)).to(a.dtype)
        ok = torch.arange(cfg.n_slots, device=r.device).view(1, -1) < L.view(-1, 1)
        sig = torch.where(ok, rf[..., 3], torch.ones_like(rf[..., 3])).clamp_min(cfg.sigma_min)   # slots past L: 0
        with g.fp32(a.device):
            u_q = self.uq.expand(B,-1,-1) if uq is None else uq
            d2 = ((cfg.grid * (u_q.unsqueeze(2) - rf[..., :2].unsqueeze(1))) ** 2).sum(-1)
            br = (-d2 / (2 * sig.unsqueeze(1) ** 2)).unsqueeze(1)
        q = self.f_q(torch.cat([a, pe], -1))
        on = (L > 0).float().view(B, 1, 1)
        dH = self.r_ca(q, tok, br, ok, strict=cfg.readout_version != "legacy")
        dF = self.f_out(dH).float() * on
        return (dF, dH * on.to(dH.dtype)) if return_hidden else dF
