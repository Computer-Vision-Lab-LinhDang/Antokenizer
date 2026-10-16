# Synthetic C/D learning pilot v1

This pilot tests whether the existing direct C/D state can learn temporal order
and preserve configuration. It is a controlled learning experiment, not a claim
of event semantics or disentanglement. The machine-readable lock is
`configs/mavt4d/synthetic_pilot_v1.json`; every output records its SHA-256.
Do not change the loss, schedule, threshold or metric after observing validation
results. A revised experiment must have a new config hash and output directory.

## Data and budget

The existing `event-cd-v1` renderer generates 128 sources (seeds 0–127), each
crossed with four cores and two configurations: 1,024 clips, 12 frames at
32×48. A new artifact directory preserves the earlier higher-resolution fixture.
The renderer audits identical final two frames across cores for every source
and configuration, and identical frame multisets within reversal pairs.

Source split uses the existing `mavt-split-v1` salt with validation fraction 0.2.
An independent `mavt-pilot-dev-v1` hash reserves 10% of training sources for dev.
The fixed run does not use dev for tuning or checkpoint selection. It evaluates
held-out sources, including texture and camera variation; it does not hold out
core×configuration combinations. All eight variants of a source stay together.
Training receives RGB, availability, timestamps and calibration only. IDs and
labels remain outside model inputs; core/configuration labels are read only by
evaluation after all representation training has finished.

| Setting | Locked value |
|---|---|
| Cells | Coupled affine, two scans C→D, GRU |
| State | C: 8×16, D: 8×16; 256 float32 values + float64 timestamp |
| State bytes | 1,032 per sample, including timestamp |
| Observer / decoder | Shared architecture; embed 32, patch 4, D modulation 8 |
| Batch | One whole source group, eight clips |
| Physical update | Two frames; six updates per clip |
| RGB queries | 256 per update, at its last frame |
| Tiny overfit | First training source, 300 steps, seed 17, each cell |
| Main training | 1,200 steps, each cell × seeds 17, 29, 43 |
| Optimizer | AdamW, lr 0.001, betas (0.9, 0.95), decay 0.0001, clip norm 1 |
| Precision / execution | float32, sequential reference recurrence for all cells |
| Selection | Final checkpoint only; no validation-based early stopping |

The state, examples, optimizer updates and query counts are matched. Parameter
counts and computation are **not** matched: flattening the same state into a GRU
uses many more parameters. Results report these counts and elapsed time; this
reference implementation is not a fused-kernel throughput comparison. Shared
observer, decoder and auxiliary heads are copied from the same initialization,
because a common seed alone does not align modules constructed after different
cells. Source sampling and objective RNG are independent of model RNG.

## Training objectives

No core/configuration supervision or pretrained teacher is used to learn the
representation. The total objective is

`L = L_observed + L_missing + 0.5 L_future + L_history
     + 0.1 L_invariance + 0.05 L_noncollapse`.

- Observed and missing reconstruction use L1, normalized by valid RGB channels
  separately. Both read the same C/D-only decoder. Empty subsets contribute zero.
- Future L1 starts from the masked prefix through frame 5 and rolls the same cell
  without observations to frames 7 and 9. Future RGB is a loss target only.
- The history head reads final C only. It predicts concatenated adaptive-average
  pooled differences `x[2]−x[7]` and `x[3]−x[6]`, each 3×8×12. Per-clip RMS
  normalization has floor 0.02. Reversing the moving segment flips the target's
  sign. The common final tubelet cannot determine this target.
- Invariance compares normalized 16-dimensional semantic projections of final C
  from the same temporal/availability support, with a photometric gain in
  [0.9,1.1] and bias in [−0.05,0.05]. It does not force all raw C dimensions to
  discard appearance needed by reconstruction.
- Noncollapse regularizes the raw semantic projection: standard-deviation floor
  0.1 and off-diagonal covariance, both views differentiable. For each view it is
  mean `ReLU(0.1−sqrt(mean(centered²)+1e-4))` plus mean squared off-diagonal
  sample covariance (denominator B−1); average the two view penalties.
  It is not a supervised class loss.

Tiny overfit uses clean observations. Main masks are drawn independently of RGB,
labels and target geometry, with one shared mask for all eight source variants.
The five families each have probability 0.2: clean; wide rectangle H/2×W/2 over
one or two tubelets; H/2×W/4 tube from frame 4 onward; missing frames 4–7; and an
H/2×W/2 rectangle missing for the whole clip. Masking happens before Patchify.
Pixels hidden from the student may be used only as targets, never as evidence.

The overfit diagnostic passes when final/initial clean reconstruction L1 ≤0.75
and history MSE / zero-predictor MSE ≤0.8. A finite failure is reported and still
followed by the fixed main budget for all cells. Nonfinite runs fail visibly.
No extra steps are added to rescue one cell. LPIPS is omitted for this small
pilot, consistently across cells. RGB targets are in [−1,1]; PSNR uses range 2.
Passing these thresholds demonstrates optimization progress on the tiny set,
not exact memorization; report the absolute errors as well as these ratios.

## Frozen evaluation

After every main model has finished, freeze all weights and fit ridge probes on
training-source features only. Use raw flattened C and D. Standardization uses
training mean/std with std floor 0.0001. The ridge parameter is 1.0, with
mean-normalized Gram matrix `XᵀX/n`, float64 solve and unpenalized intercept.
Validation labels cannot affect feature normalization, fit, checkpoint or loss.

The primary C metric averages two binary direction probes: left-then-right vs
right-then-left, and clockwise vs counterclockwise. Chance is 50%. Four-way core
accuracy is secondary: an order-blind feature can reach 50% by recognizing the
motion family. D is tested by configuration balanced accuracy, overall and for
each core. Cross-probes C→configuration and D→core/direction expose information
overlap without assuming it must be zero.
Per-core configuration scores stratify the same globally fitted D probe; they
do not refit a different probe for each core.

Required controls are state reset before the final tubelet, temporal mean of
**single-frame** observer features, and untrained recurrent C/D from the matched
initialization. The single-frame control has no time-position input; sorting each
feature coordinate before averaging ensures exact reversal invariance rather
than floating-point summation cues. It has a different feature dimension from C
or D, which is reported. The matched suffix and orderless controls must remain
at chance on balanced reversal direction. A trained probe above chance alone
does not demonstrate learning if its untrained control already does as well.

Report each seed, mean and sample standard deviation over three seeds. Bootstrap
source groups (1,000 resamples, seed 31415), retaining all variants together and
using paired source resamples across cells. Source intervals are conditional on
the fitted models, not a substitute for uncertainty across training seeds.

Decoder evaluation uses the first four sorted held-out source groups, seed 2718,
with a separate result for each of the five mask families. Snapshot is read at
frame 7 while the snapshot remains unavailable; other families are read at
frame 11, and the evaluation wide mask covers that endpoint. Report observed and
missing RGB channel counts, L1 and PSNR from aggregate squared error (range 2,
without clipping). Empty regions have null metrics; exact zero error has null
PSNR and an explicit `perfect` flag rather than an arbitrary numerical cap.

Configuration is visible in the terminal frame because it changes radius/depth.
Successful D→configuration only establishes retention of these two settings;
it does not establish phase, velocity or semantic disentanglement. Comparing
the three cells alone cannot attribute a gain specifically to D→C feedback.
That requires a subsequent matched feedback-off ablation. This pilot also does
not establish recovery of previously seen 3D surfaces from pixel mask history.

## Execution and artifacts

Build the versioned fixture with `scripts/mavt4d/build_event_suite.py`, then run
`scripts/mavt4d/run_synthetic_pilot.py --config
configs/mavt4d/synthetic_pilot_v1.json --dataset <fixture-directory>
--output <new-run-directory> --device cuda` in an allocated GPU job. The output
directory must be new. The runner records config/data identity, source revision,
initialization hashes, parameter/state budgets, checkpoints, per-step metrics,
probe predictions and a report. Production CLI requires CUDA; CPU is reserved
for bounded correctness tests, not training this experiment.

Technical gates before pilot execution include prior C/D-only restart, causal
prefix and step/chunk/scan parity tests, plus pilot config/data/loss/probe and
runner tests. A submitted or queued job is not evidence of any learning result.
