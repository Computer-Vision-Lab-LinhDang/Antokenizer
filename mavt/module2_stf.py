"""Module 2: Space-Time-Frequency Transform (STF).

Enriches each token with a 15-dim frequency profile:
  - Spatial  (Haar DWT 2-level):  7 features, always active
  - Temporal (STFT along t-axis): 4 features, video only
  - Depth    (STFT along z-axis): 4 features, 3D only

Output: freq_embed (B, N, d_freq_embed=128) + freq_raw (B, N, 15)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import STFConfig
from .types import PatchifyOutput, STFOutput


class HaarDWT2D(nn.Module):
    """2-level Haar DWT energy extractor (no learnable params)."""

    def __init__(self, levels: int = 2) -> None:
        super().__init__()
        self.levels = levels

    @staticmethod
    def _haar_level(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Single Haar DWT level: (B,C,H,W) → (LL, LH, HL, HH) each (B,C,H/2,W/2)."""
        B, C, H, W = x.shape
        x_rs = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x00 = x_rs[:, :, :, 0, :, 0]
        x01 = x_rs[:, :, :, 0, :, 1]
        x10 = x_rs[:, :, :, 1, :, 0]
        x11 = x_rs[:, :, :, 1, :, 1]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 + x01 - x10 - x11) * 0.5   # horizontal edges
        hl = (x00 - x01 + x10 - x11) * 0.5   # vertical edges
        hh = (x00 - x01 - x10 + x11) * 0.5   # diagonal
        return ll, lh, hl, hh

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, C, p, p) → [LL2, LH2, HL2, HH2, LH1, HL1, HH1]"""
        ll1, lh1, hl1, hh1 = self._haar_level(x)
        ll2, lh2, hl2, hh2 = self._haar_level(ll1)
        return [ll2, lh2, hl2, hh2, lh1, hl1, hh1]


class SpaceTimeFrequencyTransform(nn.Module):
    """Unified 4D frequency analysis module."""

    def __init__(self, cfg: Optional[STFConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or STFConfig()
        self.dwt = HaarDWT2D(levels=self.cfg.dwt_levels)

        self.register_buffer(
            "hann_window",
            torch.hann_window(self.cfg.stft_window, periodic=False),
        )

        self.freq_proj = nn.Sequential(
            nn.Linear(self.cfg.freq_raw_dim, self.cfg.d_freq_embed),
            nn.LayerNorm(self.cfg.d_freq_embed),
            nn.GELU(),
            nn.Linear(self.cfg.d_freq_embed, self.cfg.d_freq_embed),
        )

    def forward(self, patch_out: PatchifyOutput) -> STFOutput:
        B, N = patch_out.f_spatial.shape[:2]
        device = patch_out.f_spatial.device

        spatial_freq = self._spatial_freq(patch_out.raw_patches)  # (B, N, 7)

        if patch_out.raw_patches_temporal is not None:
            # temporal_freq: (B, N_spatial, 4) — broadcast to all temporal chunks
            temporal_freq_sp = self._temporal_freq(
                patch_out.raw_patches_temporal, patch_out.f_spatial
            )
            N_spatial = temporal_freq_sp.shape[1]
            n_t = N // N_spatial
            temporal_freq = (
                temporal_freq_sp
                .unsqueeze(1)
                .expand(-1, n_t, -1, -1)
                .reshape(B, N, self.cfg.n_temporal_feats)
            )
        else:
            temporal_freq = torch.zeros(B, N, self.cfg.n_temporal_feats, device=device)

        if patch_out.depth_signal is not None:
            # depth_freq: (B, N_xy, 4) — XY-plane tokens; zeros for XZ/YZ
            depth_freq_xy = self._depth_freq(patch_out.depth_signal)
            N_xy = depth_freq_xy.shape[1]
            depth_freq = torch.zeros(B, N, self.cfg.n_depth_feats, device=device)
            depth_freq[:, :N_xy] = depth_freq_xy
        else:
            depth_freq = torch.zeros(B, N, self.cfg.n_depth_feats, device=device)

        freq_raw = torch.cat([spatial_freq, temporal_freq, depth_freq], dim=-1)  # (B, N, 15)
        freq_embed = self.freq_proj(freq_raw)
        return STFOutput(freq_embed=freq_embed, freq_raw=freq_raw)

    def _spatial_freq(self, raw_patches: torch.Tensor) -> torch.Tensor:
        """raw_patches: (B, N, C, p, p) → (B, N, 7) normalized DWT energies."""
        B, N, C, p, _ = raw_patches.shape
        x = raw_patches.reshape(B * N, C, p, p)
        subbands = self.dwt(x)
        energies = [(s ** 2).mean(dim=(1, 2, 3)) for s in subbands]  # each (B*N,)
        freq = torch.stack(energies, dim=-1)                          # (B*N, 7)
        freq = freq / (freq.sum(dim=-1, keepdim=True) + 1e-8)        # normalize
        return freq.reshape(B, N, 7)

    def _temporal_freq(
        self,
        raw_patches_temporal: torch.Tensor,
        f_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal STFT features per spatial position.

        Args:
            raw_patches_temporal: (B, N_spatial, T, C, p, p)
            f_spatial:            (B, n_t * N_spatial, D)

        Returns:
            (B, N_spatial, 4)  [mean_DC, mean_F1, mean_F2, var_DC]
        """
        B, N_spatial, T, C, p, _ = raw_patches_temporal.shape
        n_t = f_spatial.shape[1] // N_spatial

        if n_t < 2:
            return torch.zeros(B, N_spatial, 4, device=f_spatial.device)

        # Build scalar signal from f_spatial L2 norm per chunk per position
        f_chunked = f_spatial.view(B, n_t, N_spatial, -1)  # (B, n_t, N_spatial, D)
        signal = f_chunked.norm(dim=-1).permute(0, 2, 1)   # (B, N_spatial, n_t)

        return self._apply_stft(signal, is_depth=False)

    def _depth_freq(
        self,
        depth_signal: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute depth STFT features per XY patch.

        Args:
            depth_signal: (depth_xz, depth_yz) each (B, n_s, S_z, C)

        Returns:
            (B, N_xy, 4)  [mean_DC, mean_F1, mean_F2, max_F2]
        """
        depth_xz, depth_yz = depth_signal   # each (B, n_s, S_z, C)
        B, n_s, S_z, C = depth_xz.shape
        N_xy = n_s * n_s

        # Cross-plane depth for each (px, py):
        # (B, n_s, 1, S_z, C) + (B, 1, n_s, S_z, C) → (B, n_s, n_s, S_z, C)
        depth_xy = (
            depth_xz.unsqueeze(2) + depth_yz.unsqueeze(1)
        ) * 0.5                                              # (B, n_s, n_s, S_z, C)
        depth_xy = depth_xy.view(B, N_xy, S_z, C)

        # L2 norm along feature dim → scalar signal along depth
        signal = depth_xy.norm(dim=-1)                      # (B, N_xy, S_z)

        return self._apply_stft(signal, is_depth=True)

    def _apply_stft(self, signal: torch.Tensor, *, is_depth: bool) -> torch.Tensor:
        """Apply STFT and aggregate into 4 features.

        Args:
            signal:   (B, N, T)
            is_depth: if True, use max_F2 as 4th feature; else use var_DC

        Returns:
            (B, N, 4)
        """
        B, N, T = signal.shape
        W = self.cfg.stft_window
        H = self.cfg.stft_hop
        n_fft = self.cfg.stft_n_fft    # 4 → 3 positive freq bins

        if T < W:
            return torch.zeros(B, N, 4, device=signal.device)

        # Normalize
        mu = signal.mean(dim=-1, keepdim=True)
        std = signal.std(dim=-1, keepdim=True) + 1e-8
        signal = (signal - mu) / std

        # Sliding windows via unfold: (B, N, n_windows, W)
        windows = signal.unfold(dimension=-1, size=W, step=H)

        hann = self.hann_window.to(signal.device, signal.dtype)
        windowed = windows * hann                                    # (B, N, n_w, W)

        fft_out = torch.fft.rfft(windowed, n=n_fft, dim=-1)         # (B, N, n_w, n_freq)
        magnitude = fft_out.abs()

        mean_per_freq = magnitude.mean(dim=2)                        # (B, N, n_freq=3)

        if is_depth:
            stat = magnitude[:, :, :, -1].amax(dim=2, keepdim=True) # max high-freq
        else:
            dc = magnitude[:, :, :, 0]  # (B, N, n_windows)
            stat = dc.var(dim=2, correction=min(1, dc.shape[2] - 1), keepdim=True)  # var DC

        return torch.cat([mean_per_freq, stat], dim=-1)              # (B, N, 4)


__all__ = ["HaarDWT2D", "SpaceTimeFrequencyTransform"]
