# Hybrid C/D reference v2

This module implements a two-stage causal hybrid with explicit recurrent state.
It is a correctness reference, not a trained model or an optimized Mamba kernel.
The existing `mavt4d.reference` model and synthetic pilot v1 are unchanged.

## Configuration and data flow

`configs/mavt4d/hybrid_cd2_kv_v2.json` contains the default architecture. Load it
with `HybridConfig(**json.load(...))`. `HybridConfig.tiny()` changes dimensions
for CPU tests while preserving the same update algorithms.

The observer applies availability before shared `Conv3d` Patchify, then eight
spatial Transformer blocks (width 384, six heads). Blocks 4/8 attend globally
within the physical observation step; other blocks use shifted local windows.
Fixed learned queries pool the result to 16 tokens of width 256.

The temporal path is:

```text
evidence -> GQA1 -> coupled C/D scan1 -> current-state readout/mixer
         -> GQA2 -> coupled C/D scan2 -> C/D-only tile decoder
```

Each GQA block has eight query heads and two KV heads. Its window contains the
current step and at most 15 preceding accepted steps. All summary slots within
one step are available together. K/V projections use normalized layer inputs,
not outputs of the same attention block, and are rounded to BF16 before use in
both sequence and streaming execution.

Each scan stores C32x32 and D32x32 directly as FP32. Its per-coordinate 2x2
affine transition includes both C-to-D and D-to-C coupling. Coefficients do not
depend on the previous state of the same layer. Layer two may depend nonlinearly
on the already computed current state of layer one. `feedback=False` removes
the direct intra-cell D-to-C entry only; it does not remove D1-to-mixer-to-C2.
The feedback policy is part of the immutable configuration and its fingerprint.

## Complete state and restart

`CDCode` contains C/D plus the readout timestamp. It is accepted by the decoder
and rejected by the continuation API. `HybridCDKVState` additionally contains
both KV rings, cache validity, accepted-step IDs, timestamps and ring cursors.
The module holds no stream-specific cache of its own.

For the default configuration, each stream allocates:

| Payload | Bytes |
|---|---:|
| C/D FP32 | 16,384 |
| K/V BF16 | 131,072 |
| Clock, IDs, validity and cursors | 800 |
| Total stream tensors | **148,256** |

The shared identity header adds 68 bytes: schema version, config SHA-256 and
weights SHA-256. A batch of B streams has `B * 148256 + 68` numeric payload bytes.
Serialization container overhead, model parameters, attention workspace and
training autograd graphs are separate. An empty ring allocates its full capacity.

```python
import torch
from mavt4d.hybrid.config import HybridConfig
from mavt4d.hybrid.model import HybridCDModel
from mavt4d.hybrid.state import HybridCDKVState, weights_fingerprint

config = HybridConfig()
model = HybridCDModel(config).eval()
state = model.initial_state(batch_size=1, last_time=0.0)
# state = model.step(state, observation)
identity = weights_fingerprint(model.state_dict())
torch.save(state.to_payload(weights_hash=identity), "stream.pt")

# In a new process, instantiate the same config and load the same weights first.
payload = torch.load("stream.pt", map_location="cpu", weights_only=True)
restored = HybridCDKVState.from_payload(
    payload, config=model.config,
    weights_hash=weights_fingerprint(model.state_dict()),
)
```

Recompute the weights fingerprint after changing parameters. Deserialization
checks the exact field whitelist, hashes, shape/dtype, finite values and ring
consistency. It does not synthesize a missing cache. `snapshot()` and
`to_payload()` detach and compact all tensors; `clone(detach=False)` preserves
gradients. `detach()` truncates both C/D and KV paths for TBPTT without resetting
their values. Inference continuation does not require optimizer state; resuming
training would additionally require a separate training checkpoint.

## Execution and causality

`ObservationStep` has images `[V,F,3,H,W]`, availability `[V,F,H,W]` and FP64
frame timestamps `[V,F]`. A video step has at most two frames; a single image
uses one real frame and internal validity padding. The state becomes available
at the final frame timestamp, so the model is causal at the tubelet boundary.
Triplane planes are distinct views, not successive physical times.
Camera rows have learned index embeddings even when calibration is supplied.
Keep the same camera row order across the stream and use those row indices in
`DecoderQuery.view_ids`. Camera identity and optional camera geometry are
separate inputs; an uncalibrated row index does not imply a known pose. Triplane
inputs use their own plane embeddings instead of the camera embedding bank.

`step(state, observation)` accepts one observation per stream. For a batch, pass
a sequence of observations inside that physical step. The outer sequence in
`forward_sequence(state, steps, method=...)` denotes time. Both `sequential` and
`associative` methods use the same physical tubelets and return one state per
step. Chunk continuation passes the final state into the next call.

The associative method genuinely computes GQA1, a complete affine prefix scan,
the mixer, GQA2 and a second complete prefix scan. This is layerwise parallel
algebra, not a single associative scan of the whole network. The reference
attention implementation uses Python window gathers; searching candidate IDs
and producing all requested per-step snapshots is not an optimized linear-time
implementation. Returning T complete states intentionally retains T outputs;
one continuation state has the fixed budget above.

Missing observations are not generated evidence:

- Partial availability masks hidden RGB, including NaNs, before Patchify.
- Whole-step missing with positive elapsed time advances the same cells with
  zero evidence bias and the dissipative missing operator. It inserts an invalid,
  zero K/V slot, aging the cache by one accepted step.
- Whole-step missing with zero elapsed time is a no-op for every state field.
- Observed steps with the same timestamp are additional accepted assimilations.
- `rollout(state, end_times)` takes explicit FP64 `[T,B]` times. One large time
  interval is not a substitute for several physical steps when comparing caches.

Naturally occluding surfaces remain valid RGB observations. Availability is not
an oracle mask of hidden objects.

## C/D-only decoder

`decode(state_or_code, DecoderQuery(...))` returns `[B,Q,3]` for `image_pixel`
queries. Future query timestamps require rollout first. World-coordinate and
normalized triplane queries need separate adapters and are not silently treated
as image-pixel tiles.

Four query blocks read C values, with width 256 and eight heads. Stage embeddings
affect keys/routing only. D is compressed to 16 values and controls bounded query
offsets and gains. Raw C value projections, the final attention output projection
and RGB head are bias-free, preserving exact `C == 0 => RGB == 0` independently
of D. This is a structural constraint, not proof of semantic disentanglement.

Each aligned query returns an 8x8 RGB tile through a 256-to-192 head. Channel
index `c*64 + dy*8 + dx` maps to channel c and pixel `(8*i+dx, 8*j+dy)` in tile
column i, row j. A 256x256 image needs 1,024 tile queries. `decode_tiles`,
`tile_grid_query`, `tiles_to_image` and `render` expose tile/grid operations.
Boundary tiles are cropped explicitly; query chunks are independent. Arbitrary
pixel queries select their aligned tile and then the containing raster pixel
using floor, including for continuous subpixel coordinates. No decoder cache
persists across calls. Decoder FFNs prepare each block's routing query before
its C cross-attention, so each FFN participates in the final output.

## Verification scope

The hybrid tests cover strict payload identity/bytes, fresh-process continuation
before and after cache wrap, future perturbations, missing/no-op behavior,
full/chunk/step/associative value and gradient parity, BF16 use during attention,
NaN isolation, spatial padding, and tile readout. Floating-point reduction orders
can differ; tests declare tolerances rather than claiming bitwise scan equality.
The reference requires FP32 parameters/C/D and BF16 K/V; mixed-precision module
conversion and optimized CUDA kernels are outside this implementation.

These checks contain no optimizer steps and establish technical correctness
only. Event/configuration learning, reconstruction quality and throughput remain
unmeasured; a training benchmark needs its own frozen protocol.
