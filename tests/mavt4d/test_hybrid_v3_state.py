"""v3 contracts: config, state and readout code with innovation/appearance."""

from dataclasses import fields, replace

import pytest
import torch

from mavt4d.hybrid.config import HybridConfig, HybridV3Config, is_v3
from mavt4d.hybrid.state import CDACode, HybridCDAKVState, HybridCDKVState
from mavt4d.hybrid_pilot.objectives import slice_state

WEIGHTS = "42" * 32
NEW = ("appearance", "appearance_valid", "prev_tokens", "prev_seen")


def v3(**kw):
    return HybridV3Config.tiny(**{"innovation": True, "appearance_tokens": 2, **kw})


def state(config=None, batch=2):
    return HybridCDAKVState.initial(config or v3(), batch_size=batch, device="cpu",
                                    last_time=torch.zeros(batch, dtype=torch.float64))


def test_v2_fingerprints_unchanged():
    assert HybridConfig().fingerprint() == "cc08e2c847a982f03d9c582bfd5d458debff8689f60d1a243c4ef0e73111b376"
    assert HybridConfig.tiny().fingerprint() == "e41a5d597856790e4396c1efec71725d0b687804c8c28dca1f4a165e4fd39992"


def test_v3_config_validates_and_fingerprints():
    cfg = HybridV3Config(innovation=True)
    assert is_v3(cfg) and not is_v3(HybridConfig())
    assert cfg.fingerprint() != HybridConfig().fingerprint()
    assert cfg.fingerprint() != HybridV3Config(innovation=True, appearance_tokens=32).fingerprint()
    with pytest.raises(ValueError):
        HybridV3Config(appearance_tokens=16)
    with pytest.raises(ValueError):
        HybridV3Config()
    assert HybridV3Config.tiny(innovation=True).innovation_grid == (4, 6)


def test_v3_state_initial_shapes_and_bytes():
    cfg = HybridV3Config(innovation=True, appearance_tokens=32)
    s = HybridCDAKVState.initial(cfg, batch_size=2, device="cpu", last_time=0.0)
    assert s.prev_tokens.shape == (2, 1, 8, 12, 384) and s.prev_seen.shape == (2, 1, 8, 12)
    assert s.appearance.shape == (2, 32, 32) and s.appearance_valid.shape == (2,)
    assert (s.payload_nbytes - s.HEADER_NBYTES) // 2 == 299_905
    assert s.code().appearance.shape == (2, 32, 32) and isinstance(s.code(), CDACode)
    off = HybridCDAKVState.initial(HybridV3Config(appearance_tokens=32), batch_size=1,
                                   device="cpu", last_time=0.0)
    assert off.prev_tokens.numel() == 0 and off.prev_seen.numel() == 0
    off = HybridCDAKVState.initial(HybridV3Config(innovation=True), batch_size=1,
                                   device="cpu", last_time=0.0)
    assert off.appearance.numel() == 0


def test_v3_payload_roundtrip_and_schema_isolation():
    cfg, s = v3(), state()
    s.appearance.fill_(1.5); s.prev_tokens.fill_(-2.); s.prev_seen.fill_(True)
    payload = s.to_payload(weights_hash=WEIGHTS)
    back = HybridCDAKVState.from_payload(payload, config=cfg, weights_hash=WEIGHTS)
    for name in HybridCDAKVState.STATE_FIELDS:
        assert torch.equal(getattr(back, name), getattr(s, name))
    with pytest.raises(ValueError):
        HybridCDKVState.from_payload(payload, config=cfg, weights_hash=WEIGHTS)
    v2 = HybridCDKVState.initial(HybridConfig.tiny(), batch_size=1, device="cpu", last_time=0.0)
    with pytest.raises(ValueError):
        HybridCDAKVState.from_payload(v2.to_payload(weights_hash=WEIGHTS), config=cfg, weights_hash=WEIGHTS)


def test_slice_clone_detach_keep_v3_fields():
    s = state(batch=3)
    assert {f.name for f in fields(s)} >= set(NEW)
    sl = slice_state(s, 1, 3)
    c = s.clone(detach=True)
    for out, b in ((sl, 2), (c, 3), (s.snapshot(), 3)):
        for name in NEW:
            assert getattr(out, name).shape[0] == b
    assert type(sl) is HybridCDAKVState and type(c) is HybridCDAKVState
