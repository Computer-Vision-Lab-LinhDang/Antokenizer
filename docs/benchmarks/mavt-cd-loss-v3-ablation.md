# MAVT loss v3 — 2×2 ablation and evaluation protocol

Last updated: **2026-10-17**. Status: **2×2 comparison planned; B objective/evaluator/runner implemented and CPU checked; no GPU trial or new scientific results yet**.

[Loss v3 design](../plans/mavt-cd-loss-v3.md) ·
[Frozen hybrid benchmark](mavt-hybrid-v2-protocol.md) ·
[Same-frame diagnostic](../cd_diagnostic.md#same-frame-d-path-diagnostic-v2) ·
[Single-GPU B trial](../runbooks/hybrid-loss-v3-trial.md)

## Comparison and purpose

| Arm | Representation objective | Decoder D control | Status / question |
|---|---|---|---|
| A | Frozen v2 six-component objective | Current shared 16-D modulation | Matched baseline for the new comparison |
| **B** | **V3 event relation, full-C consistency, region balance and reciprocal swap** | **Current shared 16-D modulation** | **Implemented for the first trial: is the objective sufficient?** |
| C | Frozen v2 six-component objective | Proposed query-conditioned D readout | Planned: does decoder access alone help? |
| D | V3 objective | Proposed query-conditioned D readout | Planned: is the combination needed? |

All four keep the same Patchify, spatial/temporal backbone, C/D/KV state contract
and state budget. C/D are direct recurrent state. The proposed decoder for C/D
would read `m_q=Read(q,D_t)` to supply routing/gain controls for each query, with
16 control values/query initially. C remains the content-value source. It does
not add a D-only RGB branch, persistent decoder state or change parallel scan.
Its parameterization, initialization and cost must be separately specified and
tested before C or D is runnable. Neither is implemented by this document.

The comparison estimates the effect of the v3 **objective bundle**, including
its additional structured synthetic pair supervision. It does not isolate one
loss term, and does not show a gain under the same supervision as RGB-only v2.
If B/D helps, a later frozen within-v3 ablation can remove one component at a
time. Do not add such experiments after seeing held-out scores under this lock.

## Data, exposure and initialization controls

- Freeze source manifests and train/dev/canonical-held-out partition hashes.
  Keep all eight crossed variants of a source in one partition. Internal dev
  comes from training sources; held-out sources are never used for selection.
- Use the same source-group ordering, mask-family draws, augmentation draws,
  frame endpoints, batch, gradient accumulation and optimizer-update budget for
  all arms. Count clip presentations, source presentations and rendered queries.
  New v3 loss-side pairs use those same inputs; report extra paired renders.
- Use seeds **17, 29, 43** for the planned multi-seed comparison. Seed 17 is the
  first implementation/optimization check, not a substitute for the full table.
  Copy shared model weights by tensor name from one canonical initialization;
  seed equality alone is insufficient when different modules consume RNG.
- Freeze optimizer, precision and schedule in a new run lock before launch.
  The first [B trial](../runbooks/hybrid-loss-v3-trial.md) uses the v2 AdamW
  defaults with its own budget and single-GPU runner; it is not the full 2×2
  comparison. Resource allocation and bounded execution follow that runbook.
- Compare equal data exposure and update budgets. Do **not** call this equal
  compute: v3 needs extra paired renders, and query-conditioned D can add cost.
  Report parameter counts (training heads separately), estimated/measured FLOPs,
  wall time, clips/s, peak allocated/reserved memory and full restart-state bytes.
  A later equal-FLOP comparison needs its own predeclared budget.

Historical runs with other batches/exposures are context, not cells in this
matched table. Preserve their original scores and selection criteria.

## Common checkpoint criterion

Training losses differ between arms and cannot select directly comparable
checkpoints. The following **new common dev criterion for all A–D** is implemented
in the [v3 evaluator](../../src/mavt4d/hybrid_loss_v3/evaluation.py) and used by
the B trial runner. It does not relabel old checkpoints or scores; any later
A/C/D comparison must use this same criterion.

Evaluate the complete fixed dev source set at initialization, every **200**
updates and the final completed update. Use deterministic clean inputs for the
three selection components below. This selects clean factorization; separately
report the existing fixed anisotropic-mask reconstruction suite, without using
it to silently change the selection rule.

1. **E_C:** `0.5 * [(1 - direction_accuracy) + (1 - core4_accuracy)]`. Give
   direction error and four-way core error equal weight so selection can detect
   both reversal failure and collapse across motion families. Use the existing
   mean-Gram ridge probe definitions: direction accuracy averages the two binary
   reversal tasks equally; core4 accuracy comes from the four-class core probe.
   Fit on optimization-train source representations only, reading flattened
   final C1+C2. Freeze the fitting procedure: train-only mean/std (floor `1e-4`),
   ridge `1.0`, FP64 solve and unpenalized intercept. Refit the statistics and
   probes on that checkpoint's optimization-train features; never fit on dev.
   Use final frame 11, where the paired suffix is identical. Report both
   accuracies separately. Do not use a training head whose meaning differs
   between A and B.
2. **E_D:** region-balanced paired-target L1 for reciprocal D swaps at frame 3,
   with C, D, queries and targets all at that frame. Use the fixed target support
   and eligibility in the v3 plan. Average both directions, four cores and then
   sources equally. This is target accuracy, not merely output sensitivity.
3. **E_R:** region-balanced own-target L1, averaged equally across the six clean
   endpoints, eight variants and then sources. This guards against improving a
   swap score while losing ordinary reconstruction.

Minimize `S_dev = (E_C + E_D/2 + E_R/2) / 3`. Division by 2 is the fixed RGB target
range, not an empirical standard deviation or per-run normalization. Predictions
remain unclipped, so RGB terms may exceed one. Report all three components and
S_dev; an aggregate win is not enough to claim both C and D improved. If an
expected dev pair or component is invalid, mark that evaluation incomplete and
do not select a checkpoint. Ties keep the earlier step.

Keep only atomically written `best.pt` for the new run, with optimizer/RNG/config
and restart metadata. The checkpoint step can precede the final step. Log curves
and step-labeled inference images; identify images from current weights versus
the selected checkpoint. No incomplete dev pass can replace best. This entire
criterion is implemented for the B trial; no measured improvement is claimed.

## Required result tables

All cells remain **unmeasured** until a separately authorized run produces them.

| Table | Purpose | Required rows and columns |
|---|---|---|
| Learning comparison | Separate objective and decoder effects | A/B/C/D; seed; best step; common dev components; held-out C direction/core4 and D configuration accuracy; self/swap RGB errors |
| Factor interventions | Test functional use rather than decodability alone | Each arm; D swap and zero at frame 3; raw-D/modulation/routing/gain distances; output change; signed paired-target gain; region sizes |
| C history controls | Rule out terminal appearance and orderless shortcuts | Trained C, untrained matched C, complete-state reset before suffix, orderless observer, D-only direction; identical source split and ridge setup |
| Factor leakage | Describe where information remains | C/D × direction/core/configuration probes; both stages combined primary, each stage secondary |
| Missing evidence | Separate recall from conditional reconstruction | Clean, wide, tube, snapshot, never-seen; observed/missing errors; available-history condition; no unsupported multi-view claim |
| Cost | Expose supervision/compute differences | Parameters, optimizer steps, clip/source presentations, decoder query count, FLOPs, runtime/throughput, memory, C/D bytes and complete C/D+KV bytes |

Preserve the frozen held-out direction/core/configuration probe definitions and
reset/orderless/untrained controls. Fit probes/scalers only on optimization-train
sources. Evaluate selected checkpoints on canonical held-out sources only after
the run's decisions are frozen. Equal-weight source aggregation and paired
source bootstrap use **1000 resamples, seed 31415**. Report each training seed,
mean and sample standard deviation; source-bootstrap intervals do not replace
between-seed uncertainty.

The current fixture has six physical steps within its KV window. Success cannot
be called memory beyond KV or general event understanding. Long-lag evidence
requires the separately versioned bridge/cache interventions in the frozen
hybrid benchmark. Do not infer a natural-video result from synthetic labels.

## Same-frame mechanism interpretation

Measure D → 16-D modulation → per-block routing / final gain → RGB using one
state endpoint and one matched query. Keep full-image and target-only moving
region metrics; report empty regions as invalid, not successful zero error.

| Pattern | Supported hypothesis / limitation |
|---|---|
| Small paired D difference | The code may not distinguish configuration at this endpoint |
| D differs, modulation nearly equal | The bottleneck may suppress the distinction |
| Modulation differs, routing/gain nearly equal | The control projections/nonlinearities may suppress it |
| Controls differ, output nearly equal | Attention/content/readout may be insensitive on these queries |
| Output changes, paired-target gain ≤ 0 | D affects predictions but does not transfer the intended configuration |
| Positive paired-target gain | Functional evidence for these pairs/endpoints; still check self error and cross-core portability |

These are localized hypotheses, not causal proofs from magnitude ratios. Raw D,
modulation and RGB have different dimensions/units. Do not compare a final-state
D distance with an early-frame swap and claim attenuation along one path.
Zero D can produce nonzero modulation because the bottleneck has biases.

For cross-core C swaps or same-configuration D donors, report target validity and
target RGB separation first. Common-suffix cores deliberately share RGB. Only
generator-audited portable combinations with distinguishable targets can support
a functional core-transfer claim; otherwise retain the history probe and mark
that image intervention not applicable. Never continue a swapped state with a
mismatched KV history.

## Optimization checks versus held-out claims

A clean tiny source can test whether v3 gradients fit known paired examples;
it cannot select a paper winner or establish generalization. CPU checks cover
loss normalization, pair gates, no-future dependency and a tiny optimizer step;
the unchanged decoder/state contracts retain their existing correctness tests.
The [trial runbook](../runbooks/hybrid-loss-v3-trial.md) lists the focused test
command. Read-only checkpoint diagnostics may
inspect old weights without replacing their heads, logs or checkpoints.

The old jointly trained C-only RGB-history MSE is not a comparable v3 optimization
metric: B removes that head and its target conflicts with configuration
invariance. If a new post-hoc RGB-history readout is desired, fit it train-only
and identically for every arm, label it a new diagnostic, and do not use it for
selection. Historical RGB-history measurements remain as originally reported.
