"""LR schedule must follow the *actual* run length (trainer.max_steps), not a stale hparam.

Resuming the 5k Stage-1 run with --ckpt_path and STEPS=10000 kept total_steps=5000 from the
checkpoint's hparams, so training continued at lr ~1e-9 (tail of the old cosine)."""
from __future__ import annotations
import math
from types import SimpleNamespace
import pytest
import torch
from mavt.training.lightning_module import MAVTLightningModule

TINY = dict(embed_dim=64, num_heads=4, num_blocks=2, patch_size=16, t_patch=2, latent_dim=8,
            semantic_dim=16, dec_dim=64, num_dec_attn_blocks=1, use_gradient_checkpointing=False,
            init_siglip2=False, use_semantic_distill=False, use_lpips=False, training_stage=1,
            lr=2e-4, warmup_steps=500, total_steps=5000)


def _lambda(module):
    cfg = module.configure_optimizers()
    sched = cfg["lr_scheduler"]["scheduler"]
    return sched.lr_lambdas[0]


def test_schedule_uses_trainer_max_steps_when_attached():
    m = MAVTLightningModule(**TINY)
    m.trainer = SimpleNamespace(max_steps=10000)           # what the CLI passed for this run
    f = _lambda(m)
    expected = 0.5 * (1 + math.cos(math.pi * (5009 - 500) / (10000 - 500)))
    assert abs(f(5009) - expected) < 1e-6, f(5009)
    assert f(5009) > 0.5, "half-way through a 10k run the lr must still be ~half of base"
    assert f(9999) < 1e-5 and f(250) == pytest.approx(0.5)  # end of cosine, mid-warmup


def test_schedule_falls_back_to_hparam_without_trainer_or_with_unbounded_trainer():
    m = MAVTLightningModule(**TINY)
    f = _lambda(m)
    assert f(4999) < 1e-5                                    # total_steps=5000 honoured
    m.trainer = SimpleNamespace(max_steps=-1)                # Lightning's "not set" sentinel
    assert _lambda(m)(4999) < 1e-5
