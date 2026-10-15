"""CameraStem: patch tokens + registers per frame (Plan 4D stem, camera part only)."""
from __future__ import annotations

import pytest
import torch

from mavt4d.model.stems import CameraStem


def test_camera_stem_nonsquare():
    stem = CameraStem(dim=32, patch=16, n_registers=4)
    tokens, registers = stem(torch.randn(5, 3, 64, 112))
    assert tokens.shape == (5, 4 * 7, 32)
    assert registers.shape == (5, 4, 32)


def test_camera_stem_rejects_size_not_multiple_of_patch():
    stem = CameraStem(dim=32, patch=16)
    with pytest.raises(ValueError):
        stem(torch.randn(1, 3, 64, 100))
