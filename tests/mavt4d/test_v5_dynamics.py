from dataclasses import replace
import torch
from mavt4d.object_state.v5.config import V5Config
from mavt4d.object_state.v5.state import V5State
from mavt4d.object_state.v5.dynamics import CConditionedDynamics


def test_frozen_generator_semigroup_and_zero_limit():
    h = torch.randn(2, 3, 2, 6)
    omega = torch.randn(2, 3, 2, 3)
    kappa = torch.rand_like(omega)
    forcing = torch.randn_like(h)
    step = CConditionedDynamics.frozen_step
    torch.testing.assert_close(step(step(h, omega, kappa, forcing, .2), omega, kappa, forcing, .3), step(h, omega, kappa, forcing, .5), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(step(h, omega * 0, kappa * 0, forcing, .5), h + .5 * forcing)


def test_coefficients_no_raw_dynamic_dependency_and_prediction_preserves_content():
    cfg = V5Config(slots=3, width=12, content_tokens=2, dynamic_tokens=2, resampler_heads=3, evidence_width=12, spatial_heads=3, spatial_blocks=2)
    net = CConditionedDynamics(cfg)
    state = V5State.initial(cfg, 2)
    state = replace(state, c=torch.randn_like(state.c), d=torch.randn_like(state.d), slot_status=torch.ones_like(state.slot_status))
    controls = torch.zeros(2, 3, 28)
    first = net.coefficients(state, controls)
    second = net.coefficients(replace(state, d=state.d + 123), controls)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    future = net(state, .5, controls)
    assert torch.equal(future.c, state.c)
    assert torch.equal(future.support_bits, state.support_bits)
    assert torch.equal(future.last_seen, state.last_seen)
