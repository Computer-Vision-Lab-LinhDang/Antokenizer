"""Four-block composition with one explicit state for filtering/readout/rollout."""

from dataclasses import replace

import torch
from torch import Tensor, nn

from .config import ObjectStateConfig
from .dynamics import SelectiveDynamics
from .resampler import ObjectResampler
from .state import Evidence, ObjectState


class ObjectStateModel(nn.Module):
    def __init__(
        self, config: ObjectStateConfig | None = None, *, observer=None, decoder=None
    ):
        super().__init__()
        self.config = config or ObjectStateConfig()
        if observer is None:
            from .observation import SparseObservationEncoder

            observer = SparseObservationEncoder(self.config)
        if decoder is None:
            from .decoder import CanonicalReadout

            decoder = CanonicalReadout(self.config)
        self.observer = observer
        self.decoder = decoder
        self.dynamics = SelectiveDynamics(self.config)
        self.resampler = ObjectResampler(self.config)

    def initial_state(self, batch_size: int, *, device=None, time=0.0, frame_id=0):
        if device is None:
            device = next(self.parameters()).device
        return ObjectState.initial(
            self.config, batch_size, device=device, time=time, frame_id=frame_id
        )

    def encode_observation(self, observation):
        evidence = self.observer(observation)
        evidence.validate(self.config)
        return evidence

    def predict(
        self, state: ObjectState, dt: Tensor | float, action: Tensor | None = None
    ):
        with torch.autocast(device_type=state.c.device.type, enabled=False):
            return self.dynamics(state, dt, action)

    def correct(
        self,
        state: ObjectState,
        evidence: Evidence,
        *,
        return_diagnostics=False,
        candidate_budget=None,
    ):
        state.validate(self.config)
        evidence.validate(self.config)
        budget = (
            self.config.candidates if candidate_budget is None else candidate_budget
        )
        if type(budget) is not int or budget < 0:
            raise ValueError("candidate_budget must be a nonnegative integer or None")
        if (
            evidence.tokens.shape[0] != state.c.shape[0]
            or evidence.tokens.device != state.c.device
        ):
            raise ValueError("evidence batch/device must match state")
        if not torch.allclose(evidence.time.double(), state.t, rtol=0, atol=1e-9):
            raise ValueError(
                "correction timestamp must match predicted state timestamp"
            )
        available = evidence.valid.any(-1)
        indices = available.nonzero(as_tuple=False).flatten()
        b, n = evidence.valid.shape
        k = self.config.slots
        if not len(indices):
            details = {
                "ownership": state.c.new_zeros(b, n, k + 1),
                "birth_proposals": state.c.new_zeros(b, n, k + 1),
                "candidate_uid": state.slot_uid.clone(),
                "candidate_indices": torch.zeros(
                    b,
                    k,
                    n if budget == 0 else min(n, budget),
                    device=state.c.device,
                    dtype=torch.int64,
                ),
                "support": state.c.new_zeros(b, k),
                "births": torch.zeros_like(state.allocated),
                "observed": torch.zeros_like(state.allocated),
                "alpha": state.c.new_zeros(b, k, self.config.content_tokens, 1),
            }
            return (state, details) if return_diagnostics else state
        selected = state.select(indices)
        selected_evidence = evidence.select(indices)
        with torch.autocast(device_type=state.c.device.type, enabled=False):
            _, relations = self.dynamics.relations(selected)
            updated, details = self.resampler(
                selected,
                selected_evidence,
                self.decoder.controls,
                relations,
                candidate_budget=candidate_budget,
            )
        values = {
            name: value.index_copy(0, indices, updated.tensor_dict()[name])
            for name, value in state.tensor_dict().items()
        }
        result = replace(state, **values)
        full_details = {}
        for name, value in details.items():
            if name == "candidate_uid":
                default = state.slot_uid.clone()
            else:
                default = value.new_zeros(b, *value.shape[1:])
            full_details[name] = default.index_copy(0, indices, value)
        return (result, full_details) if return_diagnostics else result

    def read(self, state: ObjectState, query):
        state.validate(self.config)
        return self.decoder(state, query)

    def rollout(self, state: ObjectState, dts: Tensor, actions: Tensor | None = None):
        elapsed = torch.as_tensor(dts, device=state.c.device)
        if elapsed.ndim != 2 or elapsed.shape[1] != state.c.shape[0]:
            raise ValueError("rollout dts must be [H,B]")
        if actions is not None and actions.shape != (
            len(elapsed),
            state.c.shape[0],
            self.config.action_dim,
        ):
            raise ValueError("rollout actions must be [H,B,A]")
        outputs = []
        for index, delta in enumerate(elapsed):
            state = self.predict(
                state, delta, None if actions is None else actions[index]
            )
            outputs.append(state)
        return tuple(outputs)

    def forward(
        self,
        state: ObjectState,
        observation,
        action: Tensor | None = None,
        *,
        return_diagnostics=False,
        candidate_budget=None,
    ):
        evidence = self.encode_observation(observation)
        prior = self.predict(state, evidence.time.double() - state.t, action)
        return self.correct(
            prior,
            evidence,
            return_diagnostics=return_diagnostics,
            candidate_budget=candidate_budget,
        )
