"""Wavelet Enrichment: dual-domain fusion + content-dynamics split + selection.

Three roles in one module:
  1. ENRICH: every token gets f_sem + gated(f_freq) — dual-domain representation
  2. SPLIT:  frame 0 = content, frames 1+ = dynamics candidates (video/3D)
  3. SELECT: keep dynamics patches where DWT(residual) detail energy is highest

HaarDWT2D is reused from mavt/module2_stf.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import WaveletConfig
from .types import Modality, PatchifyOutput, WaveletOutput


class HaarDWT2D(nn.Module):
    """2-level Haar DWT energy extractor (no learnable params)."""

    def __init__(self, levels: int = 2) -> None:
        super().__init__()
        self.levels = levels

    @staticmethod
    def _haar_level(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        B, C, H, W = x.shape
        x_rs = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x00 = x_rs[:, :, :, 0, :, 0]
        x01 = x_rs[:, :, :, 0, :, 1]
        x10 = x_rs[:, :, :, 1, :, 0]
        x11 = x_rs[:, :, :, 1, :, 1]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, C, p, p) -> [LL2, LH2, HL2, HH2, LH1, HL1, HH1]"""
        ll1, lh1, hl1, hh1 = self._haar_level(x)
        ll2, lh2, hl2, hh2 = self._haar_level(ll1)
        return [ll2, lh2, hl2, hh2, lh1, hl1, hh1]


class WaveletEnrich(nn.Module):
    """Dual-domain enrichment + content-dynamics split + wavelet-guided selection."""

    def __init__(
        self, cfg: Optional[WaveletConfig] = None, d_model: int = 1152,
    ) -> None:
        super().__init__()
        self.cfg = cfg or WaveletConfig()
        self.d_model = d_model
        self.dwt = HaarDWT2D(levels=self.cfg.dwt_levels)

        # Frequency enrichment: 7 subband energies -> d_model
        self.freq_proj = nn.Sequential(
            nn.Linear(7, self.cfg.d_freq),
            nn.GELU(),
            nn.Linear(self.cfg.d_freq, d_model),
        )
        # Gating: model learns how much freq info to inject per token
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, patch_out: PatchifyOutput) -> WaveletOutput:
        B, N_total = patch_out.f_spatial.shape[:2]
        device = patch_out.f_spatial.device

        # 1. Subband energy for ALL patches
        energy = self._subband_energy(patch_out.raw_patches)  # (B, N_total, 7)

        # 2. Enrich: dual-domain fusion
        enriched = self._enrich(patch_out.f_spatial, energy)  # (B, N_total, d_model)

        # 3. Split + select (video and 3D triplane)
        if patch_out.modality == "image":
            mask = torch.ones(B, N_total, dtype=torch.bool, device=device)
            return WaveletOutput(
                features=enriched, positions=patch_out.positions,
                subband_energy=energy, content_mask=mask,
                n_content=N_total, n_dynamics=0,
            )

        # Video: frame 0 = content, frames 1+ = dynamics
        # 3D triplane: XY (t=0) = content, XZ (t=1) + YZ (t=2) = dynamics
        return self._split_and_select(
            enriched, patch_out.positions, patch_out.raw_patches, energy,
        )

    # ── Step 1: subband energy ──

    def _subband_energy(self, raw_patches: torch.Tensor) -> torch.Tensor:
        """raw_patches (B, N, C, p, p) -> normalized subband energies (B, N, 7)."""
        B, N, C, p, _ = raw_patches.shape
        x = raw_patches.reshape(B * N, C, p, p)
        subbands = self.dwt(x)
        energies = torch.stack(
            [(s ** 2).mean(dim=(1, 2, 3)) for s in subbands], dim=-1,
        )  # (B*N, 7)
        energies = energies / (energies.sum(dim=-1, keepdim=True) + 1e-8)
        return energies.reshape(B, N, 7)

    # ── Step 2: dual-domain enrichment ──

    def _enrich(
        self, f_sem: torch.Tensor, energy: torch.Tensor,
    ) -> torch.Tensor:
        """Gated fusion: enriched = f_sem + gate * f_freq."""
        f_freq = self.freq_proj(energy)                               # (B, N, d_model)
        gate = self.gate(torch.cat([f_sem, f_freq], dim=-1))          # (B, N, d_model)
        return f_sem + gate * f_freq

    # ── Step 3: content-dynamics split + wavelet selection ──

    def _split_and_select(
        self,
        enriched: torch.Tensor,
        positions: torch.Tensor,
        raw_patches: torch.Tensor,
        energy: torch.Tensor,
    ) -> WaveletOutput:
        """Video/3D: key frame = content, other frames = wavelet-selected dynamics."""
        B, N_total, D = enriched.shape
        device = enriched.device

        # Determine spatial size from t=0 tokens
        N_spatial = int((positions[0, :, 0] == 0).sum().item())
        n_frames = N_total // N_spatial

        # Content: frame 0 (all patches)
        content_feat = enriched[:, :N_spatial]         # (B, N_sp, D)
        content_pos = positions[:, :N_spatial]         # (B, N_sp, 4)
        content_raw = raw_patches[:, :N_spatial]       # (B, N_sp, C, p, p)

        # Dynamics: select top-k patches per frame by residual detail energy
        k = max(
            int(self.cfg.min_keep_ratio * N_spatial),
            min(int(self.cfg.max_keep_ratio * N_spatial), N_spatial),
        )
        dyn_feats, dyn_positions = [], []
        total_dyn = 0

        for i in range(1, n_frames):
            s, e = i * N_spatial, (i + 1) * N_spatial

            # DWT on pixel residual
            residual = raw_patches[:, s:e] - content_raw
            detail_e = self._residual_detail_energy(residual)   # (B, N_sp)

            _, topk_idx = detail_e.topk(k, dim=-1)              # (B, k)

            # Gather features and positions
            idx_f = topk_idx.unsqueeze(-1).expand(-1, -1, D)
            idx_p = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
            dyn_feats.append(torch.gather(enriched[:, s:e], 1, idx_f))
            dyn_positions.append(torch.gather(positions[:, s:e], 1, idx_p))
            total_dyn += k

        # Concatenate
        if dyn_feats:
            features = torch.cat([content_feat] + dyn_feats, dim=1)
            pos = torch.cat([content_pos] + dyn_positions, dim=1)
        else:
            features = content_feat
            pos = content_pos
            total_dyn = 0

        content_mask = torch.cat([
            torch.ones(B, N_spatial, dtype=torch.bool, device=device),
            torch.zeros(B, total_dyn, dtype=torch.bool, device=device),
        ], dim=1)

        return WaveletOutput(
            features=features, positions=pos,
            subband_energy=energy, content_mask=content_mask,
            n_content=N_spatial, n_dynamics=total_dyn,
        )

    def _residual_detail_energy(self, residual: torch.Tensor) -> torch.Tensor:
        """Sum of non-LL subband energies for each patch residual.

        residual: (B, N, C, p, p) -> (B, N) scalar detail energy.
        """
        B, N, C, p, _ = residual.shape
        x = residual.reshape(B * N, C, p, p)
        subbands = self.dwt(x)
        # Sum energy of detail subbands (indices 1..6), skip LL2 (index 0)
        detail = sum((s ** 2).mean(dim=(1, 2, 3)) for s in subbands[1:])
        return detail.reshape(B, N)


__all__ = ["HaarDWT2D", "WaveletEnrich"]
