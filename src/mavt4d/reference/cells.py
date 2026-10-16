"""Small, explicit recurrent references with C/D as their entire memory.

The affine references implement an associative prefix scan in ordinary PyTorch.
This is an algebra/gradient reference, not a custom Mamba kernel or a speed
claim. Parameters are shared across streams; no sequence values are cached in
the module. Only float32/float64 are supported in this reference implementation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mavt4d.contracts import CDState, CDStateSpec


def affine_prefix_scan(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Chronological inclusive scan of block maps ``h -> a @ h + b``.

    Inputs have shape ``[T,...,N,N]`` and ``[T,...,N]``. The Hillis--Steele
    reference uses log(T) tensor stages, with no in-place writes. Combining a
    later map 2 with an earlier map 1 gives ``(A2 A1, A2 b1 + b2)``.
    """
    if a.ndim < 3 or a.shape[-1] != a.shape[-2] or b.shape != a.shape[:-1]:
        raise ValueError("affine scan: expected [T,...,N,N] and [T,...,N]")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("affine scan: operators and bias must share dtype/device")
    prefix_a, prefix_b = a, b
    offset = 1
    while offset < a.shape[0]:
        later_a = prefix_a[offset:]
        composed_a = later_a @ prefix_a[:-offset]
        composed_b = (later_a @ prefix_b[:-offset].unsqueeze(-1)).squeeze(-1) + prefix_b[offset:]
        prefix_a = torch.cat((prefix_a[:offset], composed_a), dim=0)
        prefix_b = torch.cat((prefix_b[:offset], composed_b), dim=0)
        offset *= 2
    return prefix_a, prefix_b


def _diagonal_prefix_scan(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Elementwise affine specialization with the same chronological order."""
    offset = 1
    while offset < a.shape[0]:
        next_b = a[offset:] * b[:-offset] + b[offset:]
        next_a = a[offset:] * a[:-offset]
        a = torch.cat((a[:offset], next_a), dim=0)
        b = torch.cat((b[:offset], next_b), dim=0)
        offset *= 2
    return a, b


def _tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name}: expected Tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name}: expected dense strided Tensor")


def _compact_state(c: Tensor, d: Tensor, time: Tensor, spec: CDStateSpec) -> CDState:
    # clone(), not contiguous(): a contiguous time slice can still retain the
    # complete sequence allocation. Cloning preserves autograd while detaching
    # storage ownership; inference snapshots explicitly detach separately.
    return CDState(
        c.clone(memory_format=torch.contiguous_format),
        d.clone(memory_format=torch.contiguous_format),
        time.clone(memory_format=torch.contiguous_format), spec=spec,
    )


class _ReferenceCell(nn.Module):
    """Validation and sequential reference; subclasses own all learned maps."""

    def __init__(self, spec: CDStateSpec, evidence_dim: int) -> None:
        super().__init__()
        if not isinstance(spec, CDStateSpec):
            raise TypeError("spec: expected CDStateSpec")
        if type(evidence_dim) is not int or evidence_dim <= 0:
            raise ValueError("evidence_dim: expected positive integer")
        self.spec = spec
        self.evidence_dim = evidence_dim

    def _check_evidence(self, evidence: Tensor, observed: Tensor, delta_time: Tensor) -> Tensor:
        for name, value in (("evidence", evidence), ("observed", observed), ("delta_time", delta_time)):
            _tensor(name, value)
        if evidence.ndim != 3:
            raise ValueError("evidence: expected [B,Kc+Kd,E]")
        batch = evidence.shape[0]
        if evidence.shape != (batch, self.spec.c_tokens + self.spec.d_tokens, self.evidence_dim):
            raise ValueError("evidence: expected [B,Kc+Kd,E]")
        if observed.shape != (batch,) or observed.dtype != torch.bool:
            raise ValueError("observed: expected bool [B]")
        if delta_time.shape != (batch,) or delta_time.dtype != torch.float64:
            raise ValueError("delta_time: expected float64 [B] seconds")
        parameter = next(self.parameters())
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError("reference cells support only float32 and float64")
        if evidence.dtype != parameter.dtype:
            raise ValueError("evidence and cell must share float32/float64 dtype")
        if any(value.device != parameter.device for value in (evidence, observed, delta_time)):
            raise ValueError("evidence, observed, delta_time and cell must share device")
        if not bool(torch.isfinite(delta_time).all()) or bool((delta_time < 0).any()):
            raise ValueError("delta_time: expected finite nonnegative seconds")
        clean = torch.where(observed[:, None, None], evidence, torch.zeros_like(evidence))
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("evidence: observed values must be finite")
        return clean

    def _check_state(self, state: CDState) -> None:
        if not isinstance(state, CDState):
            raise TypeError("state: expected CDState")
        state.validate()
        if state.spec != self.spec:
            raise ValueError("state spec must match cell spec")
        parameter = next(self.parameters())
        if state.c.dtype != parameter.dtype or state.c.device != parameter.device:
            raise ValueError("state and cell must share dtype/device")

    def _step_inputs(
        self, state: CDState, evidence: Tensor, observed: Tensor, end_time: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._check_state(state)
        _tensor("end_time", end_time)
        if end_time.shape != state.last_time.shape or end_time.dtype != torch.float64:
            raise ValueError("end_time: expected float64 [B]")
        if end_time.device != state.last_time.device:
            raise ValueError("end_time and state must share device")
        delta_time = end_time - state.last_time
        clean = self._check_evidence(evidence, observed, delta_time)
        if clean.shape[0] != state.c.shape[0]:
            raise ValueError("evidence batch must match state batch")
        return clean, delta_time

    def _sequence_inputs(
        self, state: CDState, evidence: Tensor, observed: Tensor, times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._check_state(state)
        for name, value in (("evidence", evidence), ("observed", observed), ("times", times)):
            _tensor(name, value)
        if evidence.ndim != 4:
            raise ValueError("evidence: expected [T,B,Kc+Kd,E]")
        length, batch = evidence.shape[:2]
        if batch != state.c.shape[0]:
            raise ValueError("evidence batch must match state batch")
        if observed.shape != (length, batch) or times.shape != (length, batch):
            raise ValueError("observed and times: expected [T,B]")
        if times.dtype != torch.float64 or times.device != state.last_time.device:
            raise ValueError("times: expected float64 on state device")
        previous = torch.cat((state.last_time[None], times[:-1]), dim=0) if length else times
        delta_time = times - previous
        clean = self._check_evidence(
            evidence.reshape(length * batch, *evidence.shape[2:]),
            observed.reshape(-1), delta_time.reshape(-1),
        )
        return clean.reshape_as(evidence), delta_time

    def scan(
        self, state: CDState, evidence: Tensor, observed: Tensor, times: Tensor,
        method: str = "sequential",
    ) -> tuple[CDState, ...]:
        """Return every state on the supplied physical step schedule.

        Splitting this schedule into chunks changes neither updates nor values.
        Splitting one missing interval into additional steps is a different
        schedule and is not claimed equivalent to the normalized operator.
        """
        if method not in ("sequential", "associative"):
            raise ValueError("method: expected sequential or associative")
        if method == "associative" and isinstance(self, GRUCDCell):
            raise ValueError("GRUCDCell has no associative scan")
        clean, delta_time = self._sequence_inputs(state, evidence, observed, times)
        if evidence.shape[0] == 0:
            return ()
        if method == "associative":
            return self._associative(state, clean, observed, times, delta_time)
        outputs = []
        for index in range(evidence.shape[0]):
            state = self.step(state, clean[index], observed[index], times[index])
            outputs.append(state)
        return tuple(outputs)


class CoupledAffineCell(_ReferenceCell):
    """Per-coordinate 2x2 C/D blocks, with coefficients independent of state.

    The observation operator is ``(1-alpha) I + alpha M`` with alpha initialized
    to 0.05. Raw M is ``I + 0.1*tanh(head([eC,eD,log1p(dt)]))`` and bias is
    bounded by 0.1. Missing dynamics use ``exp(dt G)`` where the shared learned
    generator is ``G = skew(omega) - L L^T``. Every raw operator uses the common
    normalizer ``max(1, ||raw||2, ||mask_CD(raw)||2)``. Full and feedback-off
    variants with equal weights differ only at destination-C/source-D.
    """

    def __init__(self, spec: CDStateSpec, evidence_dim: int, feedback: bool = True) -> None:
        super().__init__(spec, evidence_dim)
        if spec.c_tokens != spec.d_tokens:
            raise ValueError("coupled blocks require equal C and D token counts")
        if type(feedback) is not bool:
            raise ValueError("feedback: expected bool")
        self.feedback = feedback
        self.operator_head = nn.Linear(2 * evidence_dim + 1, spec.token_dim * 4)
        self.bias_head = nn.Linear(2 * evidence_dim + 1, spec.token_dim * 2)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))
        self.missing_rotation = nn.Parameter(torch.full((spec.c_tokens, spec.token_dim), 0.03))
        damping = 0.05 * torch.eye(2).expand(spec.c_tokens, spec.token_dim, 2, 2).clone()
        self.missing_damping = nn.Parameter(damping)

    def _normalize(self, raw: Tensor) -> Tensor:
        mask = raw.new_tensor([[1, 0], [1, 1]])
        cut = raw * mask
        scale = torch.maximum(
            torch.linalg.matrix_norm(raw, ord=2), torch.linalg.matrix_norm(cut, ord=2),
        ).clamp_min(1)
        return (raw if self.feedback else cut) / scale[..., None, None]

    def coefficients(self, evidence: Tensor, observed: Tensor, delta_time: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``A[B,Kc,d,2,2], b[B,Kc,d,2]`` without reading C/D."""
        clean = self._check_evidence(evidence, observed, delta_time)
        batch = evidence.shape[0]
        k, d = self.spec.c_tokens, self.spec.token_dim
        dt = delta_time.to(dtype=clean.dtype)
        paired = torch.cat((clean[:, :k], clean[:, k:], torch.log1p(dt)[:, None, None].expand(-1, k, 1)), dim=-1)
        identity = torch.eye(2, dtype=clean.dtype, device=clean.device)
        raw = identity + 0.1 * torch.tanh(self.operator_head(paired).reshape(batch, k, d, 2, 2))
        alpha = torch.sigmoid(self.alpha_logit)
        observed_a = (1 - alpha) * identity + alpha * self._normalize(raw)
        observed_b = 0.1 * torch.tanh(self.bias_head(paired).reshape(batch, k, d, 2))
        omega = self.missing_rotation
        zero = torch.zeros_like(omega)
        skew = torch.stack((zero, omega, -omega, zero), dim=-1).reshape(k, d, 2, 2)
        generator = skew - self.missing_damping @ self.missing_damping.transpose(-1, -2)
        missing_a = self._normalize(torch.matrix_exp(dt[:, None, None, None, None] * generator))
        a = torch.where(observed[:, None, None, None, None], observed_a, missing_a)
        b = torch.where(observed[:, None, None, None], observed_b, torch.zeros_like(observed_b))
        return a, b

    def step(self, state: CDState, evidence: Tensor, observed: Tensor, end_time: Tensor) -> CDState:
        clean, dt = self._step_inputs(state, evidence, observed, end_time)
        a, b = self.coefficients(clean, observed, dt)
        previous = torch.stack((state.c, state.d), dim=-1)
        updated = (a @ previous.unsqueeze(-1)).squeeze(-1) + b
        return _compact_state(updated[..., 0], updated[..., 1], end_time, self.spec)

    def _associative(self, state, evidence, observed, times, delta_time):
        length, batch = evidence.shape[:2]
        a, b = self.coefficients(evidence.flatten(0, 1), observed.flatten(), delta_time.flatten())
        a = a.reshape(length, batch, *a.shape[1:])
        b = b.reshape(length, batch, *b.shape[1:])
        prefix_a, prefix_b = affine_prefix_scan(a, b)
        initial = torch.stack((state.c, state.d), dim=-1)
        values = (prefix_a @ initial[None, ..., None]).squeeze(-1) + prefix_b
        return tuple(_compact_state(values[i, ..., 0], values[i, ..., 1], times[i], self.spec)
                     for i in range(length))


class TwoScanCell(_ReferenceCell):
    """C affine scan, followed by D affine scan conditioned on computed C.

    C sees only observation evidence and time. D sees current evidence and the
    mean of the just-computed C slots. Missing snapshots freeze C exactly;
    ``aD=exp(-softplus(rate(C))*dt)`` and
    ``bD=0.1*tanh(dynamics(C))*(1-exp(-dt))``. This guarantees aD(0)=1 and
    bD(0)=0 without requiring a learned loss. No prior D can influence C.
    """

    def __init__(self, spec: CDStateSpec, evidence_dim: int) -> None:
        super().__init__(spec, evidence_dim)
        d = spec.token_dim
        self.c_head = nn.Linear(evidence_dim + 1, 2 * d)
        self.d_head = nn.Sequential(nn.Linear(evidence_dim + d + 1, 2 * d), nn.SiLU(), nn.Linear(2 * d, 2 * d))
        self.missing_rate = nn.Linear(d, d)
        self.missing_dynamics = nn.Linear(d, d)

    def _c_coefficients(self, evidence, observed, dt):
        k = self.spec.c_tokens
        time_feature = torch.log1p(dt.to(evidence.dtype))[:, None, None].expand(-1, k, 1)
        raw_a, raw_b = self.c_head(torch.cat((evidence[:, :k], time_feature), dim=-1)).chunk(2, dim=-1)
        a = torch.sigmoid(raw_a + 3)
        b = 0.1 * torch.tanh(raw_b)
        return (torch.where(observed[:, None, None], a, torch.ones_like(a)),
                torch.where(observed[:, None, None], b, torch.zeros_like(b)))

    def _d_coefficients(self, evidence, observed, dt, current_c):
        k = self.spec.d_tokens
        context = current_c.mean(dim=-2, keepdim=True).expand(-1, k, -1)
        physical_dt = dt.to(evidence.dtype)[:, None, None]
        time_feature = torch.log1p(physical_dt).expand(-1, k, 1)
        features = torch.cat((evidence[:, self.spec.c_tokens:], context, time_feature), dim=-1)
        raw_a, raw_b = self.d_head(features).chunk(2, dim=-1)
        observed_a, observed_b = torch.sigmoid(raw_a + 3), 0.1 * torch.tanh(raw_b)
        missing_a = torch.exp(-F.softplus(self.missing_rate(context)) * physical_dt)
        missing_b = 0.1 * torch.tanh(self.missing_dynamics(context)) * (-torch.expm1(-physical_dt))
        return (torch.where(observed[:, None, None], observed_a, missing_a),
                torch.where(observed[:, None, None], observed_b, missing_b))

    def step(self, state: CDState, evidence: Tensor, observed: Tensor, end_time: Tensor) -> CDState:
        clean, dt = self._step_inputs(state, evidence, observed, end_time)
        ac, bc = self._c_coefficients(clean, observed, dt)
        c = ac * state.c + bc
        ad, bd = self._d_coefficients(clean, observed, dt, c)
        d = ad * state.d + bd
        return _compact_state(c, d, end_time, self.spec)

    def _associative(self, state, evidence, observed, times, delta_time):
        length, batch = evidence.shape[:2]
        flat_e, flat_m, flat_dt = evidence.flatten(0, 1), observed.flatten(), delta_time.flatten()
        ac, bc = self._c_coefficients(flat_e, flat_m, flat_dt)
        ac, bc = ac.reshape(length, batch, *ac.shape[1:]), bc.reshape(length, batch, *bc.shape[1:])
        prefix_ac, prefix_bc = _diagonal_prefix_scan(ac, bc)
        c = prefix_ac * state.c[None] + prefix_bc
        ad, bd = self._d_coefficients(flat_e, flat_m, flat_dt, c.flatten(0, 1))
        ad, bd = ad.reshape(length, batch, *ad.shape[1:]), bd.reshape(length, batch, *bd.shape[1:])
        prefix_ad, prefix_bd = _diagonal_prefix_scan(ad, bd)
        d = prefix_ad * state.d[None] + prefix_bd
        return tuple(_compact_state(c[i], d[i], times[i], self.spec) for i in range(length))


class GRUCDCell(_ReferenceCell):
    """GRU baseline whose hidden vector is exactly flattened C followed by D.

    Input is masked evidence, log1p(dt), and availability. Missing updates mix
    the proposed GRU state with the previous state by ``1-exp(-dt)``; zero-time
    missing steps are exact identity and positive-time steps can evolve. This
    nonlinear baseline does not support associative scan.
    """

    def __init__(self, spec: CDStateSpec, evidence_dim: int) -> None:
        super().__init__(spec, evidence_dim)
        slots = spec.c_tokens + spec.d_tokens
        self.gru = nn.GRUCell(slots * evidence_dim + 2, spec.num_state_values)

    def step(self, state: CDState, evidence: Tensor, observed: Tensor, end_time: Tensor) -> CDState:
        clean, dt = self._step_inputs(state, evidence, observed, end_time)
        batch = clean.shape[0]
        physical_dt = dt.to(clean.dtype)
        features = torch.cat((clean.flatten(1), torch.log1p(physical_dt)[:, None], observed.to(clean.dtype)[:, None]), dim=-1)
        hidden = torch.cat((state.c.flatten(1), state.d.flatten(1)), dim=-1)
        proposal = self.gru(features, hidden)
        amount = torch.where(observed, torch.ones_like(physical_dt), -torch.expm1(-physical_dt))
        updated = hidden + amount[:, None] * (proposal - hidden)
        c_size = self.spec.c_tokens * self.spec.token_dim
        c = updated[:, :c_size].reshape(batch, self.spec.c_tokens, self.spec.token_dim)
        d = updated[:, c_size:].reshape(batch, self.spec.d_tokens, self.spec.token_dim)
        return _compact_state(c, d, end_time, self.spec)
