"""Per-modality metrics must never be logged with sync_dist=True: under DDP the ranks hold different
modalities at the same step, so key sets differ and the collectives mismatch (deadlock at the first
validation of the 3-modality run, 2026-09-02)."""
from __future__ import annotations
import torch
from mavt.training.lightning_module import MAVTLightningModule

TINY = dict(embed_dim=64, num_heads=4, num_blocks=2, patch_size=16, t_patch=2, latent_dim=8,
            semantic_dim=16, dec_dim=64, num_dec_attn_blocks=1, use_gradient_checkpointing=False,
            init_siglip2=False, use_semantic_distill=False, use_lpips=False, training_stage=1)


def test_modality_specific_keys_are_logged_rank_locally():
    m = MAVTLightningModule(**TINY)
    calls = {}
    m.log = lambda name, value, **kw: calls.__setitem__(name, kw)      # capture instead of Lightning
    losses = {"loss": torch.tensor(1.0), "loss_l1": torch.tensor(0.5), "loss_kl": torch.tensor(0.1), "loss_temp": torch.tensor(0.0)}
    m._log_losses(losses, {"slot_diversity": torch.tensor(0.2)}, "video", "train")
    assert calls["train/loss"]["sync_dist"] is True and calls["train/loss_l1"]["sync_dist"] is True
    for k, kw in calls.items():
        if k.endswith("_video") or "/cd_" in k or "/modality_" in k:
            assert kw["sync_dist"] is False, f"{k} must be rank-local"
    assert "train/loss_l1_video" in calls and "train/loss_video" in calls
