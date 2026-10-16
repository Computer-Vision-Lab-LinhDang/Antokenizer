"""Two temporal GQA/C-D stages with complete explicit restart state."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import Tensor, nn

from mavt4d.contracts import DecoderQuery, ObservationStep

from .attention import TemporalGQA
from .cells import HybridCoupledCell, StageMixer
from .config import HybridConfig
from .decoder import HybridQueryDecoder
from .observation import HybridObservationEncoder
from .state import CDCode, HybridCDKVState


PhysicalStep = ObservationStep | Sequence[ObservationStep]


class HybridCDModel(nn.Module):
    """Reference hybrid with no module-owned causal cache.

    ``forward_encoded`` computes GQA1 -> complete scan1 -> complete mixer ->
    GQA2 -> complete scan2. It never disguises a network step loop as a parallel
    scan. Temporal attention uses bounded reference gathers; no speed claim is
    made for this implementation.
    """

    def __init__(self, config: HybridConfig, feedback: bool | None = None) -> None:
        super().__init__()
        if not isinstance(config, HybridConfig):
            raise TypeError("config: expected HybridConfig")
        if feedback is not None:
            if type(feedback) is not bool:
                raise ValueError("feedback: expected bool or None")
            config = replace(config, feedback=feedback)
        self.config = config
        self.observer = HybridObservationEncoder(config)
        self.attentions = nn.ModuleList(TemporalGQA(config) for _ in range(config.stage_count))
        self.cells = nn.ModuleList(HybridCoupledCell(config, feedback=config.feedback) for _ in range(config.stage_count))
        self.mixer = StageMixer(config)
        self.decoder = HybridQueryDecoder(config)

    def initial_state(self, batch_size: int = 1, last_time: float | Tensor = 0.0) -> HybridCDKVState:
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError("hybrid reference model requires FP32 parameters")
        return HybridCDKVState.initial(self.config, batch_size=batch_size,
                                       device=parameter.device, last_time=last_time)

    def _validate_state(self, state: HybridCDKVState) -> None:
        if not isinstance(state, HybridCDKVState):
            raise TypeError("exact continuation requires HybridCDKVState, not CDCode")
        state.validate()
        if state.config != self.config:
            raise ValueError("state config must match model config")
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32 or state.c.device != parameter.device:
            raise ValueError("state and FP32 model must share device")

    def _prepare_step(self, previous_times: Tensor,
                      observation: PhysicalStep) -> tuple[Sequence[ObservationStep], Tensor]:
        """Normalize one physical batch while checking each stream's time."""
        if isinstance(observation, ObservationStep):
            batch = (observation,)
        elif isinstance(observation, Sequence) and not isinstance(observation, (str, bytes)):
            batch = observation
        else:
            raise TypeError("observation must be ObservationStep or a batch sequence")
        if len(batch) != len(previous_times):
            raise ValueError("observation batch must match state batch")
        times = []
        for index, item in enumerate(batch):
            if not isinstance(item, ObservationStep):
                raise TypeError("observation batch entries must be ObservationStep")
            if item.frame_times.device != previous_times.device:
                raise ValueError("observation and state must share device")
            if bool((item.frame_times < previous_times[index]).any()):
                raise ValueError("frame_times: stale frame precedes state time")
            times.append(item.end_time)
        return batch, torch.stack(times)

    def _encode_step(self, previous_times: Tensor, observation: PhysicalStep,
                     modality: str) -> tuple[Tensor, Tensor, Tensor]:
        batch, times = self._prepare_step(previous_times, observation)
        evidence, observed = self.observer.forward_batch(batch, modality=modality)
        return evidence, observed, times

    def step(self, state: HybridCDKVState, observation: PhysicalStep,
             modality: str = "camera") -> HybridCDKVState:
        self._validate_state(state)
        evidence, observed, times = self._encode_step(state.last_time, observation, modality)
        return self.forward_encoded(state, evidence[None], observed[None], times[None])[0]

    def forward_sequence(self, state: HybridCDKVState, steps: Sequence[PhysicalStep],
                         method: str = "sequential", modality: str = "camera") -> tuple[HybridCDKVState, ...]:
        self._validate_state(state)
        if method not in ("sequential", "associative"):
            raise ValueError("method: expected sequential or associative")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise TypeError("steps: expected sequence of physical observations")
        observations, times = [], []
        previous_times = state.last_time
        for item in steps:
            batch, current_times = self._prepare_step(previous_times, item)
            observations.extend(batch)
            times.append(current_times)
            previous_times = current_times
        if not times:
            return ()
        # Spatial encoding has no recurrent state. Flatten time into its batch
        # axis so equal shapes share kernels, while spatial attention continues
        # to isolate every physical observation (including its own views).
        # Restore [T,B,...] before any causal temporal computation.
        evidence, observed = self.observer.forward_batch(observations, modality=modality)
        length, batch_size = len(times), len(state.last_time)
        evidence = evidence.reshape(length, batch_size, *evidence.shape[1:])
        observed = observed.reshape(length, batch_size)
        return self.forward_encoded(state, evidence, observed,
                                    torch.stack(times), method=method)

    def forward_encoded(self, state: HybridCDKVState, evidence: Tensor,
                        observed: Tensor, times: Tensor,
                        method: str = "sequential") -> tuple[HybridCDKVState, ...]:
        """Explicit post-encoder API, useful for testing temporal mathematics.

        Evidence is ``[T,B,P,width]`` FP32; masks bool and timestamps FP64.
        A missing zero-time entry is not an accepted step and changes no state.
        Returned tensors own compact storage; gradients remain connected until
        callers explicitly detach the complete C/D+KV state.
        """
        self._validate_state(state)
        if method not in ("sequential", "associative"):
            raise ValueError("method: expected sequential or associative")
        cfg = self.config
        if evidence.ndim != 4 or evidence.shape[1:] != (state.c.shape[0], cfg.summary_tokens, cfg.width):
            raise ValueError("evidence: expected [T,B,P,width]")
        length, batch = evidence.shape[:2]
        if observed.shape != (length, batch) or observed.dtype != torch.bool:
            raise ValueError("observed: expected bool [T,B]")
        if times.shape != (length, batch) or times.dtype != torch.float64:
            raise ValueError("times: expected FP64 [T,B]")
        if evidence.dtype != torch.float32 or any(x.device != state.c.device for x in (evidence, observed, times)):
            raise ValueError("evidence must be FP32 and all inputs on state device")
        previous_times = torch.cat((state.last_time[None], times[:-1])) if length else times
        delta = times - previous_times
        if not bool(torch.isfinite(times).all()) or bool((delta < 0).any()):
            raise ValueError("times must be finite and monotonic/nonnegative relative to state")
        clean = torch.where(observed[..., None, None], evidence, torch.zeros_like(evidence))
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("observed evidence must be finite")
        if length == 0:
            return ()
        accepted = observed | (delta > 0)
        counts = state.step_count[None] + accepted.to(torch.int64).cumsum(dim=0)
        step_ids = torch.where(accepted, counts - 1, torch.full_like(counts, -1))

        lower, k1, v1 = self.attentions[0].forward_sequence(
            clean, state.k[:, 0], state.v[:, 0], state.token_valid[:, 0],
            state.slot_times, state.slot_step_ids, observed, times, step_ids)
        c1, d1 = self.cells[0].scan(state.c[:, 0], state.d[:, 0], lower, observed, delta, method=method)
        mixed = self.mixer(lower, c1, d1, observed)
        upper, k2, v2 = self.attentions[1].forward_sequence(
            mixed, state.k[:, 1], state.v[:, 1], state.token_valid[:, 1],
            state.slot_times, state.slot_step_ids, observed, times, step_ids)
        c2, d2 = self.cells[1].scan(state.c[:, 1], state.d[:, 1], upper, observed, delta, method=method)
        c = torch.stack((c1, c2), dim=2)
        d = torch.stack((d1, d2), dim=2)
        new_k = torch.stack((k1, k2), dim=2)
        new_v = torch.stack((v1, v2), dim=2)

        results = []
        previous = state
        slots = torch.arange(cfg.cache_steps, device=state.c.device)[None]
        for t in range(length):
            write = accepted[t, :, None] & (slots == previous.write_index[:, None])
            kv_write = write[:, None, :, None, None, None]
            k = torch.where(kv_write, new_k[t, :, :, None], previous.k)
            v = torch.where(kv_write, new_v[t, :, :, None], previous.v)
            valid = torch.where(write[:, None, :, None],
                                observed[t, :, None, None, None].expand(batch, cfg.stage_count, cfg.cache_steps, cfg.summary_tokens),
                                previous.token_valid)
            slot_times = torch.where(write, times[t, :, None], previous.slot_times)
            slot_ids = torch.where(write, step_ids[t, :, None], previous.slot_step_ids)
            # Preserve exact carry on per-stream no-ops, including cancellation
            # roundoff in an associative prefix that includes earlier updates.
            current_c = torch.where(accepted[t, :, None, None, None], c[t], previous.c)
            current_d = torch.where(accepted[t, :, None, None, None], d[t], previous.d)
            previous = replace(
                state, c=current_c.clone(), d=current_d.clone(), k=k.clone(), v=v.clone(),
                last_time=times[t].clone(), step_count=counts[t].clone(),
                write_index=(counts[t] % cfg.cache_steps).clone(),
                steps_filled=counts[t].clamp_max(cfg.cache_steps).clone(),
                slot_times=slot_times.clone(), slot_step_ids=slot_ids.clone(),
                token_valid=valid.clone())
            results.append(previous)
        return tuple(results)

    def rollout(self, state: HybridCDKVState, end_times: Tensor,
                method: str = "sequential") -> tuple[HybridCDKVState, ...]:
        self._validate_state(state)
        if not isinstance(end_times, Tensor) or end_times.ndim != 2 or end_times.shape[1] != len(state.last_time):
            raise ValueError("end_times: expected FP64 [T,B]")
        shape = (*end_times.shape, self.config.summary_tokens, self.config.width)
        evidence = state.c.new_zeros(shape)
        observed = torch.zeros(end_times.shape, dtype=torch.bool, device=state.c.device)
        return self.forward_encoded(state, evidence, observed, end_times, method=method)

    def decode(self, state: HybridCDKVState | CDCode, query: DecoderQuery) -> Tensor:
        if isinstance(state, HybridCDKVState):
            self._validate_state(state)
            state = state.code()
        if not isinstance(state, CDCode):
            raise TypeError("decode requires CDCode or HybridCDKVState")
        return self.decoder(state, query)
