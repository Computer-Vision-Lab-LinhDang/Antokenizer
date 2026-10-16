# Explicit C/D reference model

`mavt4d.reference` implements a small, deterministic observation → C/D → RGB
pipeline against the [state contract](event_fixture_contract.md). It is a
reference for memory boundaries, recurrence and scan equivalence. It does not
establish learned event semantics, reconstruction quality, or a speed advantage.
The existing tokenizer and geometry interfaces remain unchanged.

## Shared pipeline

1. `ObservationEncoder` applies the availability mask before a shared Conv3d
   tubelet projection. Image, camera-video and triplane inputs use the same
   projection weights. Short steps are left-padded with unavailable frames;
   there is no retained frame buffer. Spatial dimensions must be divisible by
   the patch size.
2. Per-patch coverage, physical frame offsets and available camera geometry
   accompany the features. Local Q/K/V attention gathers a spatial stencil
   directly, costing O(Nk) attention scores. Learned, fixed C/D queries pool
   current evidence at O(KN). This is not an N×N attention matrix with a mask,
   and the pooling cost is not claimed independent of the state-token count.
3. The selected cell updates the supplied `CDState`. No history, KV cache,
   convolution cache or larger hidden state is stored on the module.
4. `StructuredQueryDecoder` reads C/D and a `DecoderQuery` only. C provides
   content values; D supplies bounded routing and multiplicative modulation
   through a small bottleneck. There is no D/query residual directly into RGB.

Camera views are treated as a set: permuting inputs with their calibration
preserves evidence within numerical tolerance. Cross-view fusion occurs in
the learned-query pooling; local attention stays within each view. There is
no projection-edge graph in this reference. Triplane slots have fixed XY/XZ/YZ
meaning and require three planes, one frame and no camera calibration.

`observed=False` denotes unavailable data. Natural occluder pixels remain
valid observations; this interface does not infer an amodal object mask.

## Cells and execution

| `cell_kind` | State update | Associative execution |
|---|---|---|
| `coupled` | Per-coordinate 2×2 affine blocks coupling previous C and D | One chronological block-affine prefix scan |
| `two_scan` | Diagonal C update, then diagonal D update conditioned on current C | C prefix scan followed by D prefix scan |
| `gru` | One GRU whose hidden vector is exactly flattened C followed by D | Unsupported; requesting it raises an error |

All variants share the observation/readout path and declared state budget.
Their parameter counts and computation are different. Coupled blocks require
equal C/D token counts; the other cells allow unequal counts. Reference model
arithmetic supports float32 and float64, not half precision or mixed precision.
This module provides no custom Mamba kernel.

Coupled observed transitions use `A=(1-alpha)I+alpha*M`, with alpha initialized
to 0.05, raw `M=I+0.1*tanh(head(e,log1p(dt)))` and bias bounded by 0.1. The
normalizer is shared by full and feedback-off variants:

```text
s = max(1, spectral_norm(raw), spectral_norm(mask_CD(raw)))
M_full = raw / s
M_without_D_to_C = mask_CD(raw) / s
```

`feedback=False` changes only the destination-C/source-D entry for identical
weights and inputs; the other three entries and write bias remain equal.
Coefficients do not read previous C/D. Missing dynamics use a learned fixed
generator `G=skew(omega)-L@L.T`, matrix exponential `exp(dt*G)`, the same shared
full/masked normalizer, and zero bias. Both operators have spectral norm ≤1
within floating-point tolerance. This controls propagation, not the growth
caused by repeated observation writes.

Two-scan missing steps freeze C. D uses C-conditioned decay and dynamics:
`a=exp(-softplus(rate(C))*dt)` and
`b=0.1*tanh(dynamics(C))*(1-exp(-dt))`. The GRU missing step interpolates its
proposal with the previous state by `1-exp(-dt)`. All three cells therefore
have exact identity when both observation is missing and dt=0. An observed
step at dt=0 can still assimilate evidence.

The PyTorch prefix implementation uses out-of-place Hillis–Steele stages.
Composition is chronological: `(A2,b2) o (A1,b1) = (A2@A1, A2@b1+b2)`.
It is an algebra and gradient reference, not an optimized throughput benchmark.
Every materialized state clones C, D and timestamps into compact storage, so
a state does not retain the backing allocation of a whole sequence. Training
autograd graphs remain live until deliberately detached.

## Model API

```python
import torch
from mavt4d.contracts import CDState, CDStateSpec
from mavt4d.reference import ReferenceCDModel

spec = CDStateSpec(c_tokens=8, d_tokens=8, token_dim=16)
model = ReferenceCDModel(spec, cell_kind="coupled", embed_dim=32,
                         patch_size=4, modulation_dim=8)
state = model.initial_state(batch_size=1, last_time=0.0)

# `observations` is a sequence of ObservationStep values, not an entire
# EventFixture, its labels, manifest records or target geometry.
prefix = model.forward_sequence(state, observations[:2], method="associative")
suffix = model.forward_sequence(prefix[-1], observations[2:], method="associative")
rgb = model.decode(suffix[-1], query)  # [B,Q,3], unconstrained RGB prediction

saved = suffix[-1].to_payload()
restored = CDState.from_payload(saved)
next_state = model.step(restored, next_observation)
future_states = model.rollout(next_state, future_end_times)  # float64 [T,B]
```

This example assumes a nonempty prefix and suffix and externally supplied
observations/query. `step` accepts one `ObservationStep` for batch size one, or
a sequence of B observations. `forward_sequence` accepts an outer sequence
over time; each entry contains the current observations for that batch.
An empty sequence returns an empty tuple without updating the supplied state.
`initial_state` takes one scalar start time; construct `CDState` explicitly
when different streams need different initial float64 timestamps.

Every real frame timestamp must be ≥ the previous state timestamp. Checking
only a new maximum would permit older frames inside an asynchronous step.
Equality is allowed for explicit same-time assimilation; the caller is
responsible for not unintentionally replaying an observation.

Chunk/step/scan comparisons use **the same physical observation steps**.
Splitting one two-frame Conv3d tubelet into two padded one-frame steps changes
the model input and is not an equivalence. Likewise, inserting extra missing
steps changes the physical schedule; the normalized missing operator is not
claimed to form a semigroup. TBPTT may call `state.detach()` at a chunk boundary;
that preserves values and intentionally truncates gradients.

Restart needs the same model configuration and trained weights, plus only the
declared C/D payload and subsequent external inputs. Weights are not
sequence-dependent memory. Calibration must be supplied for each observation
or query; the model does not cache it between calls.

## Decoder and C-only head

The decoder supports `image_pixel`, `world_xyz` and `triplane_uv` query spaces.
Camera IDs index an explicit calibration table when one is supplied; without
it they have no inferred geometry. A query batch is shared by the B states;
use separate decode calls for different per-stream query tables. There is no
query-to-query attention, so splitting/reordering queries preserves point
readouts within numerical tolerance.

D passes through a tanh bottleneck, bounded routing, and content gain in
`[0.9,1.1]`. Content values are bias-free projections of C. Setting C to zero
therefore makes output independent of D, even though an output-head bias may
remain. This structural restriction is only an inductive bias: nonzero C
still permits D to communicate through routing/modulation, so empirical
disentanglement tests are necessary.

`decode` rejects a query timestamp later than any corresponding state time.
First advance the same cell with `rollout`, then decode the resulting state.
Past queries are permitted but are not a guarantee that the state retained
every detail of the past. Predicted pixels are never fed back as observations.

`CTemporalHead(spec, output_dim)` accepts only the C tensor. It is exported
separately, and the general model does not attach a label-specific head or
provide a training objective.

## Verification scope

The reference tests cover chronological noncommuting affine composition;
step/chunk/scan outputs and gradients; clean D→C ablation; missing-step identity
and coupled missing propagation bounds; mask isolation and zero hidden-pixel
gradients; view permutation; state storage; future-read rejection; and a fresh
Python process resuming from weights plus the declared state payload.

Tests also exercise a synthetic fixture through the observation-only adapter.
No training run, natural-video evaluation, geometry readout quality, mixed
precision, GPU speed or semantic C/D separation is established by these gates.
