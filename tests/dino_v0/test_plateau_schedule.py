"""--schedule plateau: warmup, constant lr until the train loss stops improving over a window, then a cosine
decay of --decay-steps and stop (--steps stays the cap)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import train_dino_v0 as T  # noqa: E402


def _args(**kw):
    a = T.build_parser().parse_args(["--schedule", "plateau", "--warmup", "10", "--decay-steps", "100",
                                     "--min-lr-ratio", "0.1", "--steps", "100000"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_plateau_lr_is_flat_until_the_decay_then_cosine_to_the_floor():
    a, plat = _args(), dict(decay_start=-1)
    _, sched = T.build_optim(torch.nn.Linear(2, 2), a, plat=plat)
    f = sched.lr_lambdas[0]
    assert f(0) < f(9) <= 1.0 and f(10) == 1.0 and f(50_000) == 1.0
    plat["decay_start"] = 1000
    assert f(1000) == 1.0 and abs(f(1050) - 0.55) < 1e-6 and abs(f(1100) - 0.1) < 1e-9 and f(5000) == 0.1


def test_plateau_trigger():
    assert not T.plateau_reached(None, 1.0, 0.01)
    assert not T.plateau_reached(1.0, 0.95, 0.01)          # still 5 % better per window
    assert T.plateau_reached(1.0, 0.995, 0.01)
    assert not T.plateau_reached(1.0, 1.02, 0.01)          # got worse: a new stage's data mix, not convergence
    assert T.plateau_reached(1.0, 1.005, 0.01)             # flat within tol either way


def test_cosine_schedule_is_unchanged_by_default():
    a = T.build_parser().parse_args(["--warmup", "10", "--steps", "110", "--min-lr-ratio", "0.1"])
    assert a.schedule == "cosine"
    _, sched = T.build_optim(torch.nn.Linear(2, 2), a)
    f = sched.lr_lambdas[0]
    assert f(10) == 1.0 and abs(f(60) - 0.55) < 1e-6 and abs(f(110) - 0.1) < 1e-9
