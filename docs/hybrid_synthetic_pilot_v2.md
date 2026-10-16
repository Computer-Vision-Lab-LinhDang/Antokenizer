# Hybrid synthetic pilot v2

Frozen before learning on 2026-10-17. This is a single-seed engineering pilot
for the hybrid C/D+KV model, separate from the frozen three-cell v1 experiment.
It does not complete the long-lag or baseline matrix of the
[paper benchmark protocol](benchmarks/mavt-hybrid-v2-protocol.md).

## Model and data

The complete scientific configuration is
[`hybrid_synthetic_pilot_v2.json`](../configs/mavt4d/hybrid_synthetic_pilot_v2.json).
Its canonical SHA256, dataset manifest hashes and source revision accompany
every checkpoint. The architecture uses the existing default hybrid: eight
spatial Transformer blocks at width 384; two GQA/coupled-C/D stages at width
256; C and D each 32×32 values per stage; two bounded 16-step KV rings; and
four decoder blocks. Persistent C/D are FP32, KV BF16, clocks FP64.

Render `event-cd-v1` into a **new directory** with 128 sources (seeds 0–127),
12 frames, 128×192 pixels and dt=1/12. Keep all eight core×configuration
variants of each source together. Export audits must confirm exact common
final-two-frame equality, different histories, reversal frame-multiset equality
and target visibility before training. The higher-resolution render has its
own render fingerprint; no previous fixture directory is overwritten.

The existing group split uses `mavt-split-v1`, validation fraction 0.2; the
training-source dev split uses `mavt-pilot-dev-v1`, fraction 0.1. Neither source
IDs, configuration/core labels nor geometry targets enter the representation
model. Evaluation labels are read only by the frozen-feature evaluator.

## Training lock

| Setting | Value |
|---|---|
| Seed / model | 17 / one coupled hybrid |
| Tiny phase | First training source, eight variants, 200 optimizer updates, clean observations |
| Main phase | Fresh matched initialization, 1200 optimizer updates |
| Effective batch | 32 source groups = 256 clips per update across eight ranks |
| Per-rank batch | Four source groups; microbatch 1/2/4 groups with accumulation 4/2/1 |
| Optimizer | AdamW, lr 3e-4, betas (0.9, 0.95), weight decay 1e-4, gradient clip 1 |
| Reconstruction | Full image at each of six tubelet endpoints, 384 tiles of 8×8 pixels per endpoint |
| Checkpoints | Every 100 updates or 600 seconds, at phase boundaries and orderly interruption |
| Selection | Final checkpoint only; no held-out checkpoint selection |

Microbatch selection uses training-only timing and memory measurements. It may
change accumulation, but not the effective batch, source schedule, objectives
or optimizer update count. Tiny batches intentionally repeat the same eight
examples; this is an optimization diagnostic, not additional unique data.
Spatial compute may use selective BF16 autocast after correctness checks.
Parameters, pooling, evidence and recurrent C/D remain FP32. Blanket module
conversion or autocast of the recurrent pipeline is unsupported.

The loss is explicitly adopted for this new configuration:

\[
L=L_{observed}+L_{missing}+0.5L_{future}+L_{history}
  +0.1L_{invariance}+0.05L_{noncollapse}.
\]

Normalize observed and missing RGB channels separately; an empty subset
contributes zero. Average each source-group objective equally before averaging
over ranks. Changing microbatch grouping must preserve this objective.

The C-only history head reads flattened C1+C2 and predicts the signed pooled
RGB differences for frame pairs (2,7) and (3,6), pooled to 8×12 with RMS floor
0.02. Its target changes sign under reversal. A separate 16-D semantic
projection of C receives the existing photometric invariance/noncollapse loss,
with std floor 0.1. No event class labels supervise these heads.

Future targets are frames 7 and 9 after observing the prefix ending at frame 5;
the same recurrent cell rolls forward without observations. Future target RGB
never enters the prefix encoder or recurrent update. Main training uses the
five existing mask families with equal probability: clean, wide, tube,
snapshot and never-seen. The eight variants in a source share the mask;
photometric views share temporal/availability support. Mask extents follow the
new resolution, not target boxes.

Tiny diagnostics adopt final/initial clean reconstruction L1 ≤0.75 and history
MSE/zero-predictor MSE ≤0.8. Report absolute values as well. These thresholds
mean optimization progress, not perfect memorization. Finite failure does not
grant extra updates or change the main loss. Nonfinite runs fail explicitly.

## Evaluation and interpretation

Log fixed dev progress while training. Only a completed main budget is eligible
for the final held-out evaluation; an interrupted run is reported as partial.
Fit feature normalization and ridge probes on train sources only: FP64,
lambda=1, Gram normalized by sample count, unpenalized intercept, std floor
1e-4. Use 1000 source-group bootstrap samples, seed 31415; these intervals are
conditional on one trained model and do not measure across-seed uncertainty.
The hybrid evaluator solves the smaller dual ridge system when feature width
exceeds the number of training examples. This is algebraically equivalent to
the same primal objective; parity tests compare weights and predictions. The
shared probe API accepts an optional fitter, while v1 keeps its original default.

Primary C score is the mean binary direction accuracy of the two reversal
pairs, chance 50%, using flattened C1+C2. Primary D score is compact/wide
configuration accuracy using D1+D2, also reported per core. Report cross-probes
and controls: full-state reset before the common final tubelet, sorted
orderless single-frame features, and the matched untrained initialization.
Above-chance C alone does not establish learning if the untrained control is
equivalent. Full-state reset includes C/D **and KV**.

Reconstruction checks use four fixed held-out source groups, the five mask
families and seed 2718. Report observed/missing counts and errors separately;
retain the C-zero and D-zero decoder checks. Swap D only between the paired
configurations of the same source and core at the same time, and report error
against both the original and paired configuration. Full-image training is
a declared change from v1's sampled pixel queries. Do not compare the two runs
as matched compute. This fixture does not establish natural-event semantics,
amodal reasoning, multi-view recovery or 64-step memory retention. Those remain
separate benchmarks with versioned fixtures and cache interventions.

## Runtime and recovery

The [training architecture](architecture/mavt-hybrid-training.md) keeps the
training wrapper inside DDP. Gradient accumulation uses `no_sync()` only for
nonfinal microbatches. One online W&B run on rank zero records train losses,
gradient norm, progress, throughput, summed and maximum-rank allocated/peak memory, available
AMD device utilization/power/VRAM counters, evaluation progress and results.
Missing telemetry is marked unavailable rather than replaced by zero.
W&B owns its transport history index, including after resume. Charts use the
explicit `progress/logical_step` axis, while `progress/optimizer_step` and phase
identify the actual learning update. The local journal keeps a monotonic event
counter; it is not used to force W&B history indices.

Atomic training checkpoints contain model and auxiliary heads, optimizer,
phase/next-step cursor, frozen configuration and dataset identities, each rank's
RNG state, revision and W&B resume ID. This is distinct from a scene-state
snapshot. Each example is an independent complete clip; a checkpoint is taken
at an optimizer boundary, with no unrecorded live recurrent stream. Any future
TBPTT mode must also checkpoint its full C/D+KV state and stream cursor.

Operator-supplied UTC deadlines stay outside the scientific config. The trainer
stops at a collective optimizer boundary, saves before slow network flushing,
and rejects expired deadlines. A separate per-rank hard timeout bounds workers
even if the trainer hangs. Deadline interruption is recorded with its actual
completed budget; it does not authorize a run to resume beyond its reservation.

Before a pilot: run batching/causality/parity tests, finite optimizer and
accumulation tests, checkpoint/resume tests, mocked W&B tests, deadline tests,
then a bounded eight-rank optimizer/checkpoint smoke on the target device.
