"""Time-indexed scene state: S_t is updated only from the evidence of step t and read out only through z_t.

Per step t (attention never crosses steps, so the model is causal by construction):
    S⁻_t = S_{t-1} + predict(S_{t-1}, Δt)                      predict is zero-initialised: identity
    S_t  = S⁻_t                                                if no view of step t is visible
         = LN(blocks([observation tokens of step t ; S⁻_t])[-K:])  otherwise; hidden views are mask tokens
    z_t  = VAE(S_t)
    rgb_t, log_depth_t = readout(z_t, camera geometry of step t)   no path from the images to the readout
                                                                  except through z_t
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mavt.model.latent_heads import VAEHead
from mavt4d.geometry.camera import patch_ray_dirs
from mavt4d.model.stems import CameraStem

GEOM_FREQS = 4  # Fourier octaves for ray direction + camera centre
DT_FREQS = 6    # Fourier octaves for the time gap (periods 2 s .. 1/16 s)
LOGIT_SCALE_INIT, LOGIT_SCALE_MAX = 10.0, 100.0


@dataclass(frozen=True)
class StateConfig:
    dim: int = 256
    heads: int = 8
    depth: int = 4           # transformer blocks of one update step
    n_state: int = 16        # K state tokens; keep K below the number of patches per step
    z_dim: int = 32
    patch: int = 16
    readout_depth: int = 2
    mlp_ratio: float = 4.0
    kl_weight: float = 1e-4


@dataclass
class StateOutput:
    rgb: Tensor        # (B, V, T, 3, H, W) in [-1, 1]
    log_depth: Tensor  # (B, V, T, 1, H, W), log metres
    states: Tensor     # (B, T, K, dim)
    z: Tensor          # (B, T, K, z_dim)
    kl: Tensor         # scalar, already weighted


def fourier(x: Tensor, n: int) -> Tensor:
    """(..., c) -> (..., c * (2n + 1)): x itself plus sin/cos at octave frequencies 2^k * pi."""
    xf = x[..., None] * (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype) * math.pi)
    return torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], -1)


def patch_geometry(K: Tensor, c2w: Tensor, Hp: int, Wp: int, patch: int) -> Tensor:
    """(B, V, Hp*Wp, 6): world ray direction through each patch centre and camera centre in units of 10 m."""
    dirs = patch_ray_dirs(K, c2w, Hp, Wp, patch).flatten(-3, -2)
    return torch.cat([dirs, (c2w[..., None, :3, 3] / 10).expand_as(dirs)], -1)


def _mlp(d_in: int, d: int, d_out: int | None = None) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d_out or d))


class Attention(nn.Module):
    """Multi-head attention on per-head normalised q, k with a learnable logit scale s (init 10, at most 100):
    logits = s * cos(q, k).

    Plain dot-product attention trained from scratch starts near-uniform here (entropy within 1% of log N in
    both the update and the readout): every state token reads the mean of all patches and training sits on
    the scene-independent mean image for 1000-1500 steps before it escapes. Random cosines at head width 32
    have std ~0.18, so s = 10 makes attention selective from step 0. On 8-scene memorisation this shortened
    the plateau and reached a box-pixel PSNR of 19.1 dB at 3000 steps against 16.4 dB with plain attention.
    The cap keeps attention from saturating to one-hot, which is how slot attention died."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.log_scale = nn.Parameter(torch.full((heads, 1, 1), math.log(LOGIT_SCALE_INIT)))

    def _split(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)        # (B, H, N, d_head)

    def _qkv(self, x: Tensor, mem: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        k, v = self.kv(mem).chunk(2, -1)
        q = F.normalize(self._split(self.q(x)), dim=-1) * self.log_scale.exp().clamp(max=LOGIT_SCALE_MAX)
        return q, F.normalize(self._split(k), dim=-1), self._split(v)

    def weights(self, x: Tensor, mem: Tensor) -> Tensor:
        """(B, H, N, M) attention probabilities, for tests and diagnostics."""
        q, k, _ = self._qkv(x, mem)
        return (q @ k.transpose(-1, -2)).softmax(-1)

    def forward(self, x: Tensor, mem: Tensor) -> Tensor:
        q, k, v = self._qkv(x, mem)
        return self.out(F.scaled_dot_product_attention(q, k, v, scale=1.0).transpose(1, 2).flatten(2))


class Block(nn.Module):
    """Pre-LN block: attention of x to ``mem`` (cross) or to itself, then an MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, cross: bool = False) -> None:
        super().__init__()
        self.norm_x, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim) if cross else None
        self.attn = Attention(dim, heads)
        self.mlp = _mlp(dim, round(dim * mlp_ratio), dim)

    def forward(self, x: Tensor, mem: Tensor | None = None) -> Tensor:
        xn = self.norm_x(x)
        x = x + self.attn(xn, xn if self.norm_mem is None else self.norm_mem(mem))
        return x + self.mlp(self.norm_mlp(x))


class StateModel(nn.Module):
    def __init__(self, cfg: StateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, n_geo = cfg.dim, 6 * (2 * GEOM_FREQS + 1)
        self.stem = CameraStem(d, cfg.patch, n_registers=0)
        self.obs_geometry = _mlp(n_geo, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.state_init = nn.Parameter(torch.randn(cfg.n_state, d) * d ** -0.5)
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.state_norm = nn.LayerNorm(d)
        self.predict = _mlp(d + 2 * DT_FREQS + 1, d)
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)
        self.vae = VAEHead(d, cfg.z_dim, cfg.kl_weight)
        self.z_in = nn.Linear(cfg.z_dim, d)
        self.query_geometry = _mlp(n_geo, d)
        self.readout_blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp_ratio, cross=True)
                                            for _ in range(cfg.readout_depth))
        self.readout_norm = nn.LayerNorm(d)
        self.to_pixels = nn.Linear(d, cfg.patch * cfg.patch * 4)

    def readout(self, z: Tensor, K: Tensor, c2w: Tensor, H: int, W: int) -> tuple[Tensor, Tensor]:
        """z (B, K, z_dim), K (B, V, 3, 3), c2w (B, V, 4, 4) -> rgb (B, V, 3, H, W), log_depth (B, V, 1, H, W).

        Patch queries carry only camera geometry; everything else comes from z."""
        B, V = K.shape[:2]
        p = self.cfg.patch
        Hp, Wp = H // p, W // p
        q = self.query_geometry(fourier(patch_geometry(K, c2w, Hp, Wp, p), GEOM_FREQS))
        x, mem = q.flatten(0, 1), self.z_in(z)[:, None].expand(B, V, -1, -1).flatten(0, 1)
        for blk in self.readout_blocks:
            x = blk(x, mem)
        x = self.to_pixels(self.readout_norm(x)).view(B, V, Hp, Wp, p, p, 4).permute(0, 1, 6, 2, 4, 3, 5).reshape(B, V, 4, H, W)
        return x[:, :, :3].tanh(), x[:, :, 3:]

    def forward(self, images: Tensor, K: Tensor, c2w: Tensor, times: Tensor, visible: Tensor) -> StateOutput:
        """images (B, V, T, 3, H, W), K (B, V, 3, 3), c2w (B, V, T, 4, 4), times (B, T) seconds,
        visible (B, V, T) bool: which views the update of step t may read."""
        B, V, T, _, H, W = images.shape
        p, n = self.cfg.patch, self.cfg.n_state
        Hp, Wp = H // p, W // p
        tokens = self.stem(images.flatten(0, 2))[0].view(B, V, T, Hp * Wp, -1)
        tokens = torch.where(visible[..., None, None], tokens, self.mask_token)
        S = self.state_init.expand(B, -1, -1)
        states, zs, rgbs, depths, kl = [], [], [], [], images.new_zeros(())
        for t in range(T):
            if t > 0:
                dt = fourier((times[:, t] - times[:, t - 1])[:, None, None], DT_FREQS).expand(-1, n, -1)
                S = S + self.predict(torch.cat([S, dt], -1))
            geo = self.obs_geometry(fourier(patch_geometry(K, c2w[:, :, t], Hp, Wp, p), GEOM_FREQS))
            x = torch.cat([(tokens[:, :, t] + geo).flatten(1, 2), S], 1)
            for blk in self.blocks:
                x = blk(x)
            S = torch.where(visible[:, :, t].any(1)[:, None, None], self.state_norm(x[:, -n:]), S)
            z, _, _, kl_t = self.vae(S)
            rgb, log_depth = self.readout(z, K, c2w[:, :, t], H, W)
            states.append(S), zs.append(z), rgbs.append(rgb), depths.append(log_depth)
            kl = kl + kl_t / T
        return StateOutput(rgb=torch.stack(rgbs, 2), log_depth=torch.stack(depths, 2),
                           states=torch.stack(states, 1), z=torch.stack(zs, 1), kl=kl)


def state_loss(out: StateOutput, images: Tensor, depth: Tensor, valid: Tensor, cells: Tensor | None = None,
               depth_weight: float = 0.5) -> dict[str, Tensor]:
    """L1 on RGB and, where the GT surface exists (not sky), on log depth, over the supervised ``cells``
    (B, V, T) bool (default: all), plus the weighted KL. Supervising cells the update did not see is what
    forces the state to remember."""
    c = torch.ones(images.shape[:3], dtype=torch.bool, device=images.device) if cells is None else cells
    w = c[:, :, :, None, None, None].to(images.dtype)
    rgb = ((out.rgb - images).abs() * w).sum() / (w.sum() * images[0, 0, 0].numel()).clamp(min=1)
    m = (valid & c[..., None, None])[:, :, :, None].to(out.log_depth.dtype)
    err = (out.log_depth - torch.log(depth.clamp(min=1e-3))[:, :, :, None]).abs()
    d = (err * m).sum() / m.sum().clamp(min=1)
    return {"loss": rgb + depth_weight * d + out.kl, "rgb": rgb, "depth": d, "kl": out.kl}
