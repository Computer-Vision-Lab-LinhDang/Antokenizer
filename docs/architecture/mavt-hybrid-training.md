# Hybrid trainer and recovery architecture

The [model architecture](mavt-hybrid-v2.html) and C/D+KV state contract remain
the reference. This diagram adds the batch and training runtime around them.

```mermaid
flowchart TD
  A[Audited source-group fixtures] --> B[Deterministic rank and update schedule]
  B --> C[Shared anisotropic mask and photometric views]
  C --> D[Batched Patchify and 8 spatial Transformer blocks]
  D --> E[GQA1 and coupled C1/D1 scan]
  E --> F[GQA2 and coupled C2/D2 scan]
  K[Bounded KV1 and KV2] <--> E
  K <--> F
  F --> G[C/D-only tile decoder]
  F --> H[C-only history and semantic heads]
  G --> I[Observed, missing and future RGB losses]
  H --> J[History, invariance and noncollapse losses]
  I --> L[DDP gradient accumulation and AdamW]
  J --> L
  L --> M[Atomic training checkpoint]
  M --> N[Fresh process resume with optimizer and rank RNG]
  N --> B
  L --> O[Rank-zero online W&B]
  P[Reduced progress and device telemetry] --> O
  Q[Collective deadline flag] --> M
  Q --> R[Orderly stop then per-rank hard timeout]
  M --> S[Final-only held-out frozen-feature evaluation]
  S --> O
```

Spatial batching keeps attention isolated between examples. Selective BF16
applies to spatial compute; FP32 evidence enters the original temporal pipeline.
Each source keeps its camera calibration when decoding, even when multiple
sources share a spatial batch. C/D are the recurrent scene variables; KV is
explicit additional memory. Optimizer/RNG/checkpoint data belong to training
recovery and are not added to the inference scene-state contract.

See the [frozen pilot protocol](../hybrid_synthetic_pilot_v2.md) for budgets,
losses, controls and the boundary between this pilot and paper benchmarks.
