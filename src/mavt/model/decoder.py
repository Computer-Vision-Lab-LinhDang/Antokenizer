"""Stage 5: Two decoder heads on the shared latent z.

  AsymmetricDecoder     — z → pixel reconstruction (recon head)
  UnderstandingDecoder  — z → semantic vector aligned with vision teacher
                          (understanding head)

Both heads sit downstream of z so the latent must preserve enough information
for pixel-faithful reconstruction AND semantic alignment simultaneously.

UnifiedDetailExpander    — cross-attends from target positions into z
PixelShuffleCNNDecoder   — 4-stage PixelShuffle CNN (16× spatial upsample)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mavt.model.transformer import StandardTransformerBlock


# --------------------------------------------------------------------------- #
#  Position-encoded query generator                                             #
# --------------------------------------------------------------------------- #

class FourDQueryEncoding(nn.Module):
    """Learnable 4D position encoding for decoder queries."""

    def __init__(self, dim: int, max_t: int = 16, max_x: int = 64,
                 max_y: int = 64, max_z: int = 64):
        super().__init__()
        self.embed_t = nn.Embedding(max_t, dim // 4)
        self.embed_x = nn.Embedding(max_x, dim // 4)
        self.embed_y = nn.Embedding(max_y, dim // 4)
        self.embed_z = nn.Embedding(max_z, dim // 4)
        self.proj = nn.Linear(dim, dim)

    def forward(self, positions: torch.Tensor, B: int) -> torch.Tensor:
        """positions: (N, 4) → queries (B, N, dim)"""
        t = positions[:, 0].clamp(0, self.embed_t.num_embeddings - 1)
        x = positions[:, 1].clamp(0, self.embed_x.num_embeddings - 1)
        y = positions[:, 2].clamp(0, self.embed_y.num_embeddings - 1)
        z = positions[:, 3].clamp(0, self.embed_z.num_embeddings - 1)
        pe = torch.cat([self.embed_t(t), self.embed_x(x),
                        self.embed_y(y), self.embed_z(z)], dim=-1)  # (N, dim)
        pe = self.proj(pe).unsqueeze(0).expand(B, -1, -1)           # (B, N, dim)
        return pe


# --------------------------------------------------------------------------- #
#  UnifiedDetailExpander                                                        #
# --------------------------------------------------------------------------- #

class UnifiedDetailExpander(nn.Module):
    """Inverts C-D Split: cross-attends from target grid to compressed z.

    Uses 2 cross-attention layers from position-encoded queries into the
    compressed VAE latent representation. When latent positions are supplied,
    local residual detail tokens receive positional embeddings and a distance
    bias so each output position prefers nearby detail while content stays
    globally addressable.
    """

    def __init__(self, latent_dim: int = 32, dec_dim: int = 768,
                 num_heads: int = 8, num_layers: int = 2,
                 local_detail_bias: float = 0.25):
        super().__init__()
        self.query_enc = FourDQueryEncoding(dec_dim)
        self.kv_pos_enc = FourDQueryEncoding(latent_dim)
        self.token_type_embed = nn.Embedding(2, latent_dim)
        nn.init.zeros_(self.token_type_embed.weight)
        self.kv_pos_scale = nn.Parameter(torch.tensor(0.1))
        self.token_type_scale = nn.Parameter(torch.tensor(0.1))
        self.norm_kv = nn.LayerNorm(latent_dim)
        self.local_detail_bias = local_detail_bias
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm_q':  nn.LayerNorm(dec_dim),
                'norm_ff': nn.LayerNorm(dec_dim),
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=dec_dim, num_heads=num_heads,
                    kdim=latent_dim, vdim=latent_dim,
                    batch_first=True, bias=True,
                ),
                'ff': nn.Sequential(
                    nn.Linear(dec_dim, dec_dim * 4),
                    nn.GELU(),
                    nn.Linear(dec_dim * 4, dec_dim),
                ),
            })
            for _ in range(num_layers)
        ])

    def _detail_distance_bias(
        self,
        target_positions: torch.Tensor,
        latent_positions: Optional[torch.Tensor],
        latent_token_types: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if latent_positions is None or latent_token_types is None:
            return None
        detail_mask = latent_token_types == 1
        if not bool(detail_mask.any()):
            return None

        q_pos = target_positions.float()
        kv_pos = latent_positions.float()
        dist = (q_pos[:, None, :] - kv_pos[None, :, :]).abs().sum(dim=-1)
        bias = torch.zeros_like(dist, dtype=dtype)
        bias[:, detail_mask] = -self.local_detail_bias * dist[:, detail_mask].to(dtype)
        return bias

    def forward(
        self,
        z: torch.Tensor,
        target_positions: torch.Tensor,
        latent_positions: Optional[torch.Tensor] = None,
        latent_token_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        z               : (B, N_c+N_d, latent_dim)
        target_positions: (N_target, 4)

        Returns expanded : (B, N_target, dec_dim)
        """
        B = z.shape[0]
        q = self.query_enc(target_positions, B)   # (B, N_target, dec_dim)
        kv = z
        if latent_positions is not None:
            latent_positions = latent_positions.to(device=z.device, dtype=torch.long)
            kv = kv + (
                self.kv_pos_scale.to(kv.dtype)
                * self.kv_pos_enc(latent_positions, B).to(kv.dtype)
            )
        if latent_token_types is not None:
            latent_token_types = latent_token_types.to(device=z.device, dtype=torch.long)
            kv = kv + (
                self.token_type_scale.to(kv.dtype)
                * self.token_type_embed(latent_token_types).unsqueeze(0).to(kv.dtype)
            )
        kv = self.norm_kv(kv)                       # (B, N_z, latent_dim)

        attn_mask = self._detail_distance_bias(
            target_positions.to(z.device),
            latent_positions,
            latent_token_types,
            q.dtype,
        )

        for layer in self.layers:
            q_n = layer['norm_q'](q)
            out, _ = layer['cross_attn'](q_n, kv, kv, attn_mask=attn_mask)
            q = q + out
            q = q + layer['ff'](layer['norm_ff'](q))

        return q   # (B, N_target, dec_dim)


# --------------------------------------------------------------------------- #
#  PixelShuffle CNN decoder                                                     #
# --------------------------------------------------------------------------- #

class ResBlock2D(nn.Module):
    """GroupNorm-GELU pre-activation residual block."""

    def __init__(self, dim: int):
        super().__init__()
        groups = min(32, dim)
        self.norm1 = nn.GroupNorm(groups, dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class WindowedSelfAttn2D(nn.Module):
    """Pre-LN windowed self-attention + MLP for 2D feature maps.

    Partitions (B, C, H, W) into non-overlapping spatial windows of size
    `window_size × window_size`. Attention runs INSIDE each window only, so
    cost is O(B · nW · ws² · C) — feasible at 32²–64² where global attention
    would be too heavy.
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 window_size: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            batch_first=True, bias=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        # Zero-init last MLP linear so block starts as identity (residual only).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        # Pad spatial dims if not divisible by window_size.
        Hp = (H + ws - 1) // ws * ws
        Wp = (W + ws - 1) // ws * ws
        # Permute to channels-last for layernorm/attention.
        xp = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        if (Hp, Wp) != (H, W):
            xp = F.pad(xp, (0, 0, 0, Wp - W, 0, Hp - H))
        # Window partition: (B, nH, ws, nW, ws, C) → (B*nH*nW, ws*ws, C)
        nH, nW = Hp // ws, Wp // ws
        xp = xp.reshape(B, nH, ws, nW, ws, C)
        xp = xp.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        # Pre-LN attention + MLP, both residual.
        h = self.norm1(xp)
        out, _ = self.attn(h, h, h)
        xp = xp + out
        xp = xp + self.mlp(self.norm2(xp))
        # Reverse window partition.
        xp = xp.reshape(B, nH, nW, ws, ws, C)
        xp = xp.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        return xp[..., :H, :W]


class PixelShuffleCNNDecoder(nn.Module):
    """4-stage progressive upsampler with CNN + windowed attention refinement.

    Input : (B, in_channels, H_grid, W_grid)   e.g. (B, 768, 16, 16)
    Output: (B, 3, H_out, W_out)               e.g. (B, 3, 256, 256), in [-1, 1]

    Each stage upsamples 2× via Conv → PixelShuffle → GELU. Between stages we
    insert ResBlock2D + WindowedSelfAttn2D so the model can refine features at
    32², 64², 128² instead of leaving all high-freq generation to the final
    Conv → PS layer alone (which previously had ~LPIPS bottleneck).

    Param overhead vs flat CNN: ~+8 M (~30 M total).
    """

    def __init__(self, in_channels: int = 768):
        super().__init__()
        # 16×16 → 32×32
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 512 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.refine1 = nn.Sequential(
            ResBlock2D(512),
            WindowedSelfAttn2D(512, num_heads=8, window_size=8),
        )
        # 32×32 → 64×64
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.refine2 = nn.Sequential(
            ResBlock2D(256),
            WindowedSelfAttn2D(256, num_heads=8, window_size=8),
        )
        # 64×64 → 128×128 (no attn here — 16×16 windows = 256² tokens × 128 ch
        # would dominate compute. ResBlock only.)
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.refine3 = ResBlock2D(128)
        # 128×128 → 256×256 (output, bounded to [-1, 1]).
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 3 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )
        self._icnr_init()

    def _icnr_init(self) -> None:
        """ICNR (Aitken et al., 2017): initialise the r² sub-pixel filters of
        each conv-before-PixelShuffle block identically so init acts like
        nearest-neighbour upsample → no checkerboard artifact early on.
        """
        r = 2  # PixelShuffle factor used in every block
        for module in [self.up1, self.up2, self.up3, self.up4]:
            for m in module:
                if isinstance(m, nn.Conv2d) and m.out_channels % (r * r) == 0:
                    ni = m.in_channels
                    no = m.out_channels // (r * r)
                    kh, kw = m.kernel_size
                    kernel = m.weight.new_empty(no, ni, kh, kw)
                    nn.init.kaiming_normal_(kernel, nonlinearity='relu')
                    m.weight.data.copy_(kernel.repeat_interleave(r * r, dim=0))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.refine1(x)
        x = self.up2(x)
        x = self.refine2(x)
        x = self.up3(x)
        x = self.refine3(x)
        x = self.up4(x)
        return x


# --------------------------------------------------------------------------- #
#  AsymmetricDecoder                                                            #
# --------------------------------------------------------------------------- #

class AsymmetricDecoder(nn.Module):
    """Full asymmetric decoder: expander → self-attention blocks → CNN upsample.

    Handles all three modalities (image, video, 3D) with shared weights.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        dec_dim: int = 768,
        num_attn_blocks: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        t_patch: int = 2,
    ):
        super().__init__()
        self.t_patch = t_patch
        self.expander = UnifiedDetailExpander(latent_dim, dec_dim, num_heads=num_heads)
        self.self_attn_blocks = nn.ModuleList([
            StandardTransformerBlock(dec_dim, num_heads, mlp_ratio)
            for _ in range(num_attn_blocks)
        ])
        # Temporal expansion: one decoded chunk feature -> t_patch per-frame features.
        # Initialised as a stacked identity so every sub-frame starts equal to the
        # chunk decode (the previous T/τ behaviour), and the temporal detail is learned.
        self.temporal_expand = nn.Linear(dec_dim, dec_dim * t_patch)
        with torch.no_grad():
            self.temporal_expand.weight.copy_(torch.eye(dec_dim).repeat(t_patch, 1))
            self.temporal_expand.bias.zero_()
        self.cnn = PixelShuffleCNNDecoder(in_channels=dec_dim)

    def _decode_grid(
        self,
        z: torch.Tensor,           # (B, Nz, latent_dim)
        positions: torch.Tensor,   # (N_grid, 4)
        H_grid: int,
        W_grid: int,
        latent_positions: Optional[torch.Tensor] = None,
        latent_token_types: Optional[torch.Tensor] = None,
        t_expand: int = 1,
    ) -> torch.Tensor:
        """Decode one 2D grid → (B, 3, H_out, W_out), or with t_expand>1 the
        t_expand frames of a temporal chunk → (B, 3, t_expand, H_out, W_out)."""
        B = z.shape[0]
        expanded = self.expander(
            z, positions, latent_positions, latent_token_types
        )  # (B, N_grid, dec_dim)
        for blk in self.self_attn_blocks:
            expanded = blk(expanded)

        if t_expand == 1:
            feat = expanded.transpose(1, 2).reshape(B, -1, H_grid, W_grid)
            return self.cnn(feat)   # (B, 3, 16*H_grid, 16*W_grid)

        assert t_expand == self.t_patch, (t_expand, self.t_patch)
        N = expanded.shape[1]
        sub = self.temporal_expand(expanded).view(B, N, t_expand, -1)      # (B, N, τ, dec_dim)
        frames = []
        for k in range(t_expand):
            feat = sub[:, :, k].transpose(1, 2).reshape(B, -1, H_grid, W_grid)
            frames.append(self.cnn(feat))                                  # (B, 3, H, W)
        return torch.stack(frames, dim=2)                                  # (B, 3, τ, H, W)

    def forward(
        self,
        z: torch.Tensor,                # (B, Nz, latent_dim)
        target_positions: torch.Tensor, # (N_target, 4)
        modality: str,
        grid_shape: tuple,              # (H_grid, W_grid) for image/3D; (Tp, H, W) for video
        latent_positions: Optional[torch.Tensor] = None,
        latent_token_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode z → reconstructed pixel-space tensor."""
        if modality == 'image':
            H, W = grid_shape
            out = self._decode_grid(
                z, target_positions, H, W,
                latent_positions=latent_positions,
                latent_token_types=latent_token_types,
            )
            return out  # (B, 3, H_out, W_out)

        elif modality == 'video':
            Tp, Hg, Wg = grid_shape
            N_frame = Hg * Wg
            # Decode per temporal chunk, expanding each chunk to t_patch frames so the
            # output covers every input frame (T = Tp * t_patch).
            chunks = []
            for t in range(Tp):
                pos_t = target_positions[t * N_frame:(t + 1) * N_frame]
                chunk = self._decode_grid(
                    z, pos_t, Hg, Wg,
                    latent_positions=latent_positions,
                    latent_token_types=latent_token_types,
                    t_expand=self.t_patch,
                )  # (B, 3, τ, H, W)
                chunks.append(chunk)
            return torch.cat(chunks, dim=2)     # (B, 3, T, H, W)

        elif modality == 'threed':
            N_plane = target_positions.shape[0] // 3
            Hg, Wg = grid_shape
            planes_out = []
            for p in range(3):
                pos_p = target_positions[p * N_plane:(p + 1) * N_plane]
                plane = self._decode_grid(
                    z, pos_p, Hg, Wg,
                    latent_positions=latent_positions,
                    latent_token_types=latent_token_types,
                )  # (B, 3, H, W)
                planes_out.append(plane)
            return torch.stack(planes_out, dim=1)  # (B, 3, 3, H, W)

        else:
            raise ValueError(f"Unknown modality: {modality}")


# --------------------------------------------------------------------------- #
#  UnderstandingDecoder                                                         #
# --------------------------------------------------------------------------- #

class UnderstandingDecoder(nn.Module):
    """Decode latent z → global semantic vector aligned with vision teacher.

    Mirror of AsymmetricDecoder but for the understanding output. Operating on
    z (the bottleneck) — not on the encoder's pre-VAE features — forces the
    latent to preserve enough semantic information to recover a SigLIP-aligned
    representation. This is the "understanding head" of the unified tokenizer.

    Architecture:
        z (B, Nz, latent_dim)
          → Linear(latent_dim → dec_dim) + LayerNorm
          → N self-attention blocks (refine token interactions)
          → attention pool with single learnable query → (B, dec_dim)
          → LayerNorm + Linear(dec_dim → semantic_dim)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        dec_dim: int = 768,
        semantic_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        content_only: bool = False,
    ):
        super().__init__()
        # content_only: read only the content slots (token_type 0). The 2026-09-02 probe
        # showed the head pooling over 256 non-semantic detail tokens loses information
        # (kNN 0.378 on content-mu → 0.335 at the head output).
        self.content_only = content_only
        self.in_proj = nn.Linear(latent_dim, dec_dim)
        self.norm_in = nn.LayerNorm(dec_dim)

        self.self_attn_blocks = nn.ModuleList([
            StandardTransformerBlock(dec_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.query = nn.Parameter(torch.randn(1, 1, dec_dim) * (dec_dim ** -0.5))
        self.pool = nn.MultiheadAttention(
            embed_dim=dec_dim, num_heads=num_heads,
            batch_first=True, bias=True,
        )
        self.norm_out = nn.LayerNorm(dec_dim)
        self.proj = nn.Linear(dec_dim, semantic_dim)

    def forward(self, z: torch.Tensor, token_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z: (B, Nz, latent_dim), token_types: (Nz,) 0=content/1=detail → semantic: (B, semantic_dim)"""
        if self.content_only and token_types is not None:
            keep = token_types == 0
            if keep.any():
                z = z[:, keep]
        x = self.norm_in(self.in_proj(z))
        for blk in self.self_attn_blocks:
            x = blk(x)
        B = x.shape[0]
        q = self.query.expand(B, 1, -1)
        pooled, _ = self.pool(q, x, x)
        return self.proj(self.norm_out(pooled.squeeze(1)))
