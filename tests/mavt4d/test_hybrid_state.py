"""Strict restart payload and memory accounting for the hybrid reference."""

from dataclasses import replace
import io

import pytest
import torch

from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.state import CDCode, HybridCDKVState, weights_fingerprint


WEIGHTS = "42" * 32
FIELDS = (
    "c", "d", "k", "v", "last_time", "step_count", "write_index",
    "steps_filled", "slot_times", "slot_step_ids", "token_valid",
)


def initial(config=None, batch=1):
    return HybridCDKVState.initial(
        config or HybridConfig.tiny(), batch_size=batch,
        device="cpu", last_time=torch.zeros(batch, dtype=torch.float64),
    )


def occupied(count=5, batch=2):
    state = initial(batch=batch)
    w = state.config.cache_steps
    ids = torch.arange(count - w, count, dtype=torch.int64)
    for step in ids.tolist():
        slot = step % w
        state.slot_step_ids[:, slot] = step
        state.slot_times[:, slot] = float(step + 1)
        state.token_valid[:, :, slot] = True
        state.k[:, :, slot] = float(step + 1)
        state.v[:, :, slot] = -float(step + 1)
    return replace(
        state, step_count=torch.full((batch,), count, dtype=torch.int64),
        write_index=torch.full((batch,), count % w, dtype=torch.int64),
        steps_filled=torch.full((batch,), w, dtype=torch.int64),
        last_time=torch.full((batch,), float(count), dtype=torch.float64),
    )


def test_default_config_and_numeric_budget_match_preset():
    cfg = HybridConfig()
    assert (cfg.stage_count, cfg.state_tokens, cfg.state_dim) == (2, 32, 32)
    assert (cfg.spatial_blocks, cfg.spatial_width, cfg.spatial_heads) == (8, 384, 6)
    assert (cfg.summary_tokens, cfg.width, cfg.q_heads, cfg.kv_heads) == (16, 256, 8, 2)
    assert (cfg.patch_size, cfg.cache_steps, cfg.decoder_blocks, cfg.decoder_tile) == (16, 16, 4, 8)
    state = initial(cfg, batch=2)
    assert state.cd_nbytes == 2 * 16384
    assert state.kv_nbytes == 2 * 131072
    assert state.metadata_nbytes == 2 * 800
    assert state.total_nbytes == 2 * 148256
    assert state.retained_storage_nbytes == state.total_nbytes
    assert state.code().total_nbytes == 2 * (16384 + 8)
    payload = state.to_payload(weights_hash=WEIGHTS)
    header = sum(payload[k].numel() * payload[k].element_size() for k in (
        "schema_version", "config_sha256", "weights_sha256",
    ))
    assert header == 68
    assert state.payload_nbytes == state.total_nbytes + 68


@pytest.mark.parametrize("kwargs", [
    {"stage_count": 1}, {"state_tokens": 0}, {"cache_steps": -1},
    {"width": 17}, {"q_heads": 3}, {"kv_heads": 3},
    {"spatial_width": 25}, {"decoder_tile": 0}, {"patch_size": True},
])
def test_config_rejects_invalid_dimensions(kwargs):
    with pytest.raises((TypeError, ValueError)):
        HybridConfig.tiny(**kwargs)


def test_config_fingerprint_is_stable_and_covers_geometry():
    cfg = HybridConfig.tiny()
    assert len(cfg.fingerprint()) == 64
    assert cfg.fingerprint() == HybridConfig.tiny().fingerprint()
    assert cfg.fingerprint() != replace(cfg, cache_steps=cfg.cache_steps + 1).fingerprint()
    assert cfg.fingerprint() != replace(cfg, decoder_tile=cfg.decoder_tile + 1).fingerprint()


def test_feedback_behavior_is_part_of_restart_identity():
    cfg = HybridConfig.tiny()
    assert cfg.feedback is True
    disabled = replace(cfg, feedback=False)
    assert cfg.fingerprint() != disabled.fingerprint()
    state = initial(cfg)
    payload = state.to_payload(weights_hash=WEIGHTS)
    with pytest.raises(ValueError, match="config_sha256"):
        HybridCDKVState.from_payload(payload, config=disabled, weights_hash=WEIGHTS)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_feedback_configuration_requires_a_boolean(value):
    with pytest.raises(ValueError, match="feedback"):
        HybridConfig.tiny(feedback=value)


def test_initial_state_has_canonical_empty_ring_and_explicit_time():
    state = initial(batch=2)
    assert state.k.dtype == state.v.dtype == torch.bfloat16
    assert state.c.dtype == state.d.dtype == torch.float32
    assert state.last_time.dtype == state.slot_times.dtype == torch.float64
    assert bool((state.slot_step_ids == -1).all())
    assert not bool(state.token_valid.any())
    assert not bool(state.k.any()) and not bool(state.v.any())
    assert not bool(state.slot_times.any())
    state.validate()
    with pytest.raises((TypeError, ValueError)):
        HybridCDKVState.initial(state.config, batch_size=0, device="cpu", last_time=0.0)


@pytest.mark.parametrize("name,value", [
    ("c", float("nan")), ("d", float("inf")),
    ("last_time", float("nan")), ("slot_times", float("inf")),
])
def test_nonfinite_state_rejected(name, value):
    state = initial()
    altered = getattr(state, name).clone()
    altered.flatten()[0] = value
    with pytest.raises(ValueError):
        replace(state, **{name: altered})


def test_wrong_shape_dtype_and_extra_payload_are_rejected():
    state = initial()
    for name, altered in (
        ("c", state.c.double()), ("k", state.k.float()),
        ("token_valid", state.token_valid.long()),
        ("step_count", state.step_count.float()),
        ("d", state.d[..., :-1]),
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(state, **{name: altered})
    payload = state.to_payload(weights_hash=WEIGHTS)
    payload["unreported_history"] = torch.zeros(1)
    with pytest.raises(ValueError, match="keys"):
        HybridCDKVState.from_payload(payload, config=state.config, weights_hash=WEIGHTS)


@pytest.mark.parametrize("corrupt", ["write", "filled", "old_id", "duplicate_id", "time", "last_time", "negative_count"])
def test_ring_metadata_is_checked_before_resume(corrupt):
    state = occupied()
    values = {name: getattr(state, name).clone() for name in FIELDS}
    if corrupt == "write":
        values["write_index"][:] = 0
    elif corrupt == "filled":
        values["steps_filled"][:] = 1
    elif corrupt == "old_id":
        values["slot_step_ids"][0, 0] -= state.config.cache_steps
    elif corrupt == "duplicate_id":
        values["slot_step_ids"][0, 0] = values["slot_step_ids"][0, 1]
    elif corrupt == "time":
        values["slot_times"][0, 0] = 100
    elif corrupt == "last_time":
        values["last_time"][:] += 1
    else:
        values["step_count"][:] = -1
    with pytest.raises(ValueError):
        HybridCDKVState(**values, config=state.config)


def test_invalid_and_empty_cache_values_must_be_zero():
    state = initial()
    k = state.k.clone()
    k.flatten()[0] = 1
    with pytest.raises(ValueError, match="invalid|empty"):
        replace(state, k=k)
    times = state.slot_times.clone()
    times[0, 0] = 1
    with pytest.raises(ValueError, match="empty"):
        replace(state, slot_times=times)
    state = occupied()
    valid = state.token_valid.clone()
    valid[0, 1, 0, 0] = False
    with pytest.raises(ValueError, match="invalid"):
        replace(state, token_valid=valid)


def test_full_missing_occupied_slots_are_valid_and_same_timestamp_allowed():
    state = occupied()
    valid = torch.zeros_like(state.token_valid)
    state = replace(state, token_valid=valid, k=torch.zeros_like(state.k), v=torch.zeros_like(state.v))
    state = replace(state, slot_times=state.last_time[:, None].expand_as(state.slot_times).clone())
    state.validate()


def test_batched_streams_have_independent_ring_positions_and_occupancy():
    state = initial(batch=3)
    counts = torch.tensor([0, 1, 5], dtype=torch.int64)
    w = state.config.cache_steps
    values = {name: getattr(state, name).clone() for name in FIELDS}
    values["step_count"] = counts
    values["steps_filled"] = counts.clamp_max(w)
    values["write_index"] = counts.remainder(w)
    values["last_time"] = counts.double()
    for batch, count in enumerate(counts.tolist()):
        for step in range(max(0, count - w), count):
            slot = step % w
            values["slot_step_ids"][batch, slot] = step
            values["slot_times"][batch, slot] = step + 1
    state = HybridCDKVState(**values, config=state.config)
    assert torch.equal(state.steps_filled, torch.tensor([0, 1, 3]))
    assert not bool(state.token_valid.any())
    state.validate()


def test_payload_roundtrip_is_strict_independent_and_weights_only_loadable():
    state = occupied()
    payload = state.to_payload(weights_hash=WEIGHTS)
    assert set(payload) == set(FIELDS) | {"schema_version", "config_sha256", "weights_sha256"}
    assert payload["schema_version"].dtype == torch.int32
    assert payload["schema_version"].item() == 2
    assert payload["config_sha256"].dtype == payload["weights_sha256"].dtype == torch.uint8
    stream = io.BytesIO()
    torch.save(payload, stream)
    assert stream.tell() > state.payload_nbytes
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    restored = HybridCDKVState.from_payload(loaded, config=state.config, weights_hash=WEIGHTS)
    for name in FIELDS:
        assert torch.equal(getattr(state, name), getattr(restored, name))
        assert getattr(state, name).data_ptr() != getattr(restored, name).data_ptr()
    restored.c.add_(1)
    assert not torch.equal(restored.c, state.c)


@pytest.mark.parametrize("kind", ["config", "weights", "version", "header_type", "missing"])
def test_payload_rejects_wrong_identity_or_header(kind):
    state = initial()
    payload = state.to_payload(weights_hash=WEIGHTS)
    config, weights = state.config, WEIGHTS
    if kind == "config":
        config = replace(config, cache_steps=config.cache_steps + 1)
    elif kind == "weights":
        weights = "ab" * 32
    elif kind == "version":
        payload["schema_version"] = torch.tensor(1, dtype=torch.int32)
    elif kind == "header_type":
        payload["schema_version"] = 2
    else:
        del payload["k"]
    with pytest.raises((TypeError, ValueError)):
        HybridCDKVState.from_payload(payload, config=config, weights_hash=weights)


def test_cd_code_is_decode_only_and_exact_resume_never_fills_cache():
    state = occupied()
    code = state.code()
    assert isinstance(code, CDCode)
    assert code.c is state.c and code.d is state.d
    assert not hasattr(code, "k")
    with pytest.raises((TypeError, ValueError)):
        HybridCDKVState.from_payload(code, config=state.config, weights_hash=WEIGHTS)


def test_snapshot_compacts_backing_storage_and_detaches_all_float_memory():
    state = initial()
    backing = torch.zeros((100, *state.c.shape[1:]), requires_grad=True)
    k_backing = torch.zeros((100, *state.k.shape[1:]), dtype=torch.bfloat16, requires_grad=True)
    state = replace(state, c=backing[:1], k=k_backing[:1])
    assert state.retained_storage_nbytes > state.total_nbytes
    snap = state.snapshot()
    assert snap.retained_storage_nbytes == snap.total_nbytes
    assert not snap.c.requires_grad and not snap.k.requires_grad
    assert snap.c.is_contiguous() and snap.k.is_contiguous()
    copied = state.clone(detach=False)
    assert copied.c.requires_grad and copied.k.requires_grad
    (copied.c.sum() + copied.k.float().sum()).backward()
    assert backing.grad is not None and k_backing.grad is not None
    detached = state.detach()
    assert not detached.c.requires_grad and not detached.k.requires_grad
    assert detached.c.data_ptr() == state.c.data_ptr()


def test_validation_runs_again_after_mutating_a_frozen_container_tensor():
    state = initial()
    state.k.flatten()[0] = 1
    with pytest.raises(ValueError):
        state.to_payload(weights_hash=WEIGHTS)
    with pytest.raises(ValueError):
        state.code()


def test_weights_fingerprint_is_order_independent_and_value_sensitive():
    tensors = {"a": torch.arange(8, dtype=torch.bfloat16), "b": torch.tensor(2, dtype=torch.int64)}
    first = weights_fingerprint(tensors)
    assert len(first) == 64
    assert first == weights_fingerprint(dict(reversed(list(tensors.items()))))
    assert first != weights_fingerprint({"a": tensors["a"].float(), "b": tensors["b"]})
    tensors["a"][0] += 1
    assert first != weights_fingerprint(tensors)


@pytest.mark.parametrize("value", ["short", "xx" * 32, None])
def test_weights_identity_requires_sha256_hex(value):
    with pytest.raises((TypeError, ValueError)):
        initial().to_payload(weights_hash=value)
