"""Composable reference pipeline whose only recurrent input is ``CDState``.

Encoding is local to each physical observation step. Encoding a whole sequence
before a scan therefore changes scheduling, not its information boundary. The
module has no stream cache, feature queue, convolution history, or hidden state.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real

import torch
from torch import Tensor, nn

from mavt4d.contracts import CDState, CDStateSpec, DecoderQuery, ObservationStep

from .cells import CoupledAffineCell, GRUCDCell, TwoScanCell
from .decoder import StructuredQueryDecoder
from .observation import ObservationEncoder


PhysicalStep = ObservationStep | Sequence[ObservationStep]


class ReferenceCDModel(nn.Module):
    """Shared observation/readout path for three explicit-state cell references.

    ``feedback=False`` is the destination-C/source-D coupling ablation of the
    coupled cell only. ``two_scan`` and ``gru`` keep their own fixed dependency
    structures. The separate ``CTemporalHead`` can be attached for a particular
    training objective; this general model does not assume event class labels.

    Parallel scan means algebraic equivalence on the *same physical steps*.
    Repartitioning frames into different tubelets changes observations and is
    not a supported equivalence. No speed advantage is asserted here.
    """

    def __init__(
        self,
        spec: CDStateSpec,
        cell_kind: str = "coupled",
        feedback: bool = True,
        embed_dim: int = 32,
        patch_size: int = 4,
        modulation_dim: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(spec, CDStateSpec):
            raise TypeError("spec: expected CDStateSpec")
        if cell_kind not in ("coupled", "two_scan", "gru"):
            raise ValueError("cell_kind: expected coupled, two_scan, or gru")
        if type(feedback) is not bool:
            raise ValueError("feedback: expected bool")
        if not feedback and cell_kind != "coupled":
            raise ValueError("feedback=False applies only to the coupled cell")
        self.spec = spec
        self.cell_kind = cell_kind
        self.embed_dim = embed_dim
        self.observer = ObservationEncoder(spec, embed_dim, patch_size)
        if cell_kind == "coupled":
            self.cell = CoupledAffineCell(spec, embed_dim, feedback=feedback)
        elif cell_kind == "two_scan":
            self.cell = TwoScanCell(spec, embed_dim)
        else:
            self.cell = GRUCDCell(spec, embed_dim)
        self.decoder = StructuredQueryDecoder(spec, hidden_dim=embed_dim,
                                               modulation_dim=modulation_dim)
        self.temporal_head = None

    def initial_state(self, batch_size: int = 1, last_time: float = 0.0) -> CDState:
        """Create compact zeros on the model device/dtype, with exact float64 time.

        All streams initially share the supplied physical timestamp. For streams
        with different starting times, construct the declared ``CDState`` with
        an explicit float64 ``last_time[B]`` instead of storing a side cache.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size: expected positive integer")
        if isinstance(last_time, bool) or not isinstance(last_time, Real):
            raise ValueError("last_time: expected finite scalar seconds")
        if not math.isfinite(last_time):
            raise ValueError("last_time: expected finite scalar seconds")
        parameter = next(self.parameters())
        if parameter.dtype not in (torch.float32, torch.float64):
            raise ValueError("reference model supports only float32 and float64")
        return CDState(
            parameter.new_zeros((batch_size, self.spec.c_tokens, self.spec.token_dim)),
            parameter.new_zeros((batch_size, self.spec.d_tokens, self.spec.token_dim)),
            torch.full((batch_size,), float(last_time), dtype=torch.float64,
                       device=parameter.device), spec=self.spec,
        )

    def _validate_state(self, state: CDState) -> None:
        if not isinstance(state, CDState):
            raise TypeError("state: expected CDState")
        state.validate()
        if state.spec != self.spec:
            raise ValueError("state spec must match model spec")
        parameter = next(self.parameters())
        if state.c.dtype != parameter.dtype or state.c.device != parameter.device:
            raise ValueError("state and model must share dtype/device")

    def _encode_step(
        self, previous_times: Tensor, observation: PhysicalStep, modality: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a current batch independently, without consulting recurrent C/D."""
        if isinstance(observation, ObservationStep):
            observations = (observation,)
        elif isinstance(observation, Sequence) and not isinstance(observation, (str, bytes)):
            observations = observation
        else:
            raise TypeError("observation: expected ObservationStep or a sequence for the batch")
        if len(observations) != previous_times.shape[0]:
            raise ValueError("observation batch must match state batch")
        evidence, available, times = [], [], []
        for index, item in enumerate(observations):
            if not isinstance(item, ObservationStep):
                raise TypeError("observation batch entries must be ObservationStep")
            item.validate_against(self.spec)
            if item.frame_times.device != previous_times.device:
                raise ValueError("observation and state must share device")
            # Check every actual frame, not only max(frame_times): accepting a
            # new maximum would otherwise silently re-assimilate stale frames.
            if bool((item.frame_times < previous_times[index]).any()):
                raise ValueError("frame_times: stale frame precedes the previous state time")
            encoded, observed = self.observer(item, modality=modality)
            evidence.append(encoded)
            available.append(torch.as_tensor(observed, dtype=torch.bool,
                                             device=encoded.device))
            times.append(item.end_time)
        return torch.stack(evidence), torch.stack(available), torch.stack(times)

    def step(
        self, state: CDState, observation: PhysicalStep, modality: str = "camera",
    ) -> CDState:
        """Assimilate one physical observation per stream and return all memory."""
        self._validate_state(state)
        evidence, observed, times = self._encode_step(state.last_time, observation, modality)
        return self.cell.step(state, evidence, observed, times)

    def forward_sequence(
        self,
        state: CDState,
        steps: Sequence[PhysicalStep],
        method: str = "sequential",
        modality: str = "camera",
    ) -> tuple[CDState, ...]:
        """Process a sequence, returning one compact state per physical step.

        A batch is a sequence of ``ObservationStep`` values *inside* each step.
        The outer sequence always denotes time. Chunk continuation feeds the
        returned final state into the next call with the remaining same steps.
        """
        self._validate_state(state)
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise TypeError("steps: expected a sequence of physical observations")
        evidence, observed, times = [], [], []
        previous_times = state.last_time
        for observation in steps:
            encoded, available, current_times = self._encode_step(previous_times, observation, modality)
            evidence.append(encoded)
            observed.append(available)
            times.append(current_times)
            previous_times = current_times
        if not evidence:
            batch = state.c.shape[0]
            empty_evidence = state.c.new_empty((0, batch,
                                               self.spec.c_tokens + self.spec.d_tokens,
                                               self.embed_dim))
            empty_observed = torch.empty(0, batch, dtype=torch.bool, device=state.c.device)
            empty_times = state.last_time.new_empty((0, batch))
            return self.cell.scan(state, empty_evidence, empty_observed, empty_times, method=method)
        return self.cell.scan(state, torch.stack(evidence), torch.stack(observed),
                              torch.stack(times), method=method)

    def rollout(
        self, state: CDState, end_times: Tensor, method: str = "sequential",
    ) -> tuple[CDState, ...]:
        """Advance the same cell with no observation on explicit float64 [T,B] times.

        Generated pixels never become observed evidence. Repeated times are
        valid no-observation identity steps; decreasing times are invalid.
        """
        self._validate_state(state)
        if not isinstance(end_times, Tensor):
            raise TypeError("end_times: expected Tensor")
        batch = state.c.shape[0]
        if end_times.ndim != 2 or end_times.shape[1] != batch:
            raise ValueError("end_times: expected shape [T,B]")
        if end_times.dtype != torch.float64 or end_times.device != state.c.device:
            raise ValueError("end_times: expected float64 on the state device")
        evidence = state.c.new_zeros((end_times.shape[0], batch,
                                     self.spec.c_tokens + self.spec.d_tokens, self.embed_dim))
        observed = torch.zeros(end_times.shape, dtype=torch.bool, device=state.c.device)
        return self.cell.scan(state, evidence, observed, end_times, method=method)

    def decode(self, state: CDState, query: DecoderQuery) -> Tensor:
        """Read C/D only; a future timestamp must first be reached via rollout."""
        self._validate_state(state)
        if not isinstance(query, DecoderQuery):
            raise TypeError("query: expected DecoderQuery")
        query.validate()
        if query.times.device != state.last_time.device:
            raise ValueError("query and state must share device")
        if bool((query.times[None, :] > state.last_time[:, None]).any()):
            raise ValueError("future query time requires an explicit rollout before decode")
        return self.decoder(state, query)
