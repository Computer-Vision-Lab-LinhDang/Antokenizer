# Bounded C/D optimization diagnostics

This protocol is separate from the frozen `hybrid-synthetic-pilot-v2` experiment.
It does not overwrite its runs, snapshots, results, configuration, losses, or
held-out measurements. The short experiments diagnose optimization mechanisms;
they are not a new generalization benchmark.

## Questions and concrete mechanisms

The pilot's tiny phase improved image reconstruction but did not pass its ordered
history target gate. Its D-swap evaluation changed reconstruction errors only
slightly. Three mechanisms need distinguishing before extending training:

1. The recurrent write probability starts near 0.01. There are only six physical
   steps per clip, so C/D magnitudes and order-sensitive writes can initially be
   small. This is an initialization hypothesis, not a demonstrated root cause.
2. D passes through a 16-dimensional modulation bottleneck. Both its query-routing
   and content-gain contributions are multiplied by 0.1; the corresponding
   projections start with standard deviation 0.02. This may weaken D gradients.
3. Whole-image L1 is dominated by static background. A small **error difference**
   after swapping D does not alone imply small **prediction change**, nor prove
   that D has no effect on moving objects. Report both explicitly.
4. The frozen objective supervises ordered history through C but has no D-only
   configuration loss or counterfactual swap constraint. C may retain
   configuration, and reconstruction may use little D. Increasing D sensitivity
   does not by itself make the intended C/D factorization identifiable.

The labels used below arrange diagnostic pairs only. They never enter the model
or any optimization loss. No held-out source is used for tuning or arm selection.

## Locked short experiment

| Arm | Updates | Change from fresh canonical initialization | Purpose |
|---|---:|---|---|
| `baseline` | 200 | None | Reproduce the short optimization behavior |
| `write_gate` | 200 | Initial gate bias changes from logit(0.01) to logit(0.1) | Test attenuation of recurrent writes |
| `d_strength` | 200 | D routing and gain multiplier changes from 0.1 to 1.0 | Test decoder sensitivity to D |

Each arm runs independently on one allocated GPU. Two GPUs can run baseline and
D-strength concurrently; write-gate follows baseline on its GPU. Every arm starts
from the same seed and original weights except its declared intervention. It uses
the first verified training source's eight crossed variants, clean observations,
the same six RGB-only losses, AdamW settings, and spatial BF16 policy as the pilot.

Each update contains one source group (eight clips), compared with 32 repeated
groups in the original tiny phase. Therefore this baseline is a matched baseline
for the new arms, not a claim of numerically reproducing the old tiny run. The
photometric augmentation sample count changes; all three arms share the new
schedule. An optional explicitly locked `--steps` cap is for bounded smoke tests.

The CLI writes an immutable `run_lock.json` before work. It records the scientific
config, diagnostic hash, source revision, dataset hash, precision, budget, and
deadlines. Old tiny/main checkpoints can be inspected read-only using repeated
`--audit-checkpoint` arguments, but are never used to initialize the new arms.

## Measurements

Training curves log all six objective components and the optimizer step every
five updates. Detailed probes run at initialization and every 50 updates:

- Ordered-history MSE, its zero-predictor baseline, their ratio, and cosine.
- Clean reconstruction L1 across all six physical endpoints.
- Separate history/reconstruction gradient L2 and RMS for the observer, temporal
  attention, mixer, cells, history head, decoder D path, and decoder content path.
  Cell output rows also separate C writes, D writes, and gate logits. These are
  component-gradient diagnostics, not the gradient of the full weighted loss.
- Final C/D RMS, across-variant standard deviation, matched configuration
  differences, and matched reversed-history differences.
- Actual gate means/ranges by stage; D modulation magnitude, configuration
  difference, and multiplicative gain deviation from one.
- At the fixed early endpoint frame 3: D-zero and paired D-swap **prediction L1
  change**, both over all pixels and moving pixels; error to original targets and
  improvement toward the paired configuration target. Moving pixels are defined
  from temporal RGB changes in the training source with threshold 1e-5.
- Target configuration and reversed-history distances expose whether the chosen
  view/time makes the intended change visible. Exact C-zero output is checked.

Small panels at initialization and every 100 updates show two matched
configurations. Columns are masked input at frame 3, masked reconstruction,
target, future prediction at frame 7 using only the prefix through frame 5,
future target, and the D-swapped reconstruction at frame 3. Metrics use unclipped
predictions; displayed images are clipped to [-1,1]. The fixed rectangular mask
is only for illustration; it does not change the clean optimization objective.

## Checkpoints, selection, and stopping

Keep exactly one `best.pt` per arm, atomically replaced only when the clean tiny
history MSE improves at a scheduled probe. This is training-source selection,
**not a validation/generalization claim**. Ties keep the earlier snapshot.
Checkpoint contents include model and heads, optimizer, RNG state, arm/config
hashes, source cursor, next optimizer step, and W&B run ID. The final selected
snapshot is uploaded to W&B once; intermediate snapshots are not uploaded.

The controller stops optimizer updates at the supplied optimization deadline,
uses the separate work deadline for final diagnostics, and responds to TERM/INT
at an optimizer boundary. Deployment must additionally enforce an external hard
deadline. A stopped result records the actual completed step and the last probe
step; its retained best checkpoint can be earlier than the stop. The runner
refuses an existing output directory. Errors leave the last valid best snapshot.

Interpret relative improvements against the matched baseline and measured
gradients. A larger D prediction change alone is not success: it must also help
the paired configuration target without destroying reconstruction. A faster tiny
history fit does not establish useful event semantics or long-term memory.

## Subsequent two-GPU run

`run_hybrid_cd_longrun.py` requires an explicit reviewed arm. It never selects an
arm automatically from the diagnostic or launches from a previous run. Start
fresh from the canonical seed, applying only the selected arm's intervention.
The original six-component objective and train/dev source split stay fixed.

The separate long-run lock uses two DDP ranks, one source group per rank, thus
16 clips per optimizer update. The base budget is 10,000 updates. At a completed
budget boundary it may extend to 15,000 and then 20,000 only when measured update
time plus amortized dev/checkpoint time, with a 15% margin, fits before the
optimization cutoff. Extension decisions never inspect losses or quality.

This changes effective batch size from the original pilot's 256 clips. Therefore
10,000 updates means 160,000 clip presentations, compared with 307,200 during
the original 1,200-update main phase; 20,000 updates means 320,000 presentations.
More updates alone must not be described as more total data exposure or as a
controlled matched-budget comparison to the original pilot.

Every 200 updates, evaluate the **complete fixed dev partition**, with fixed
source-specific mask and augmentation seeds and equal source weighting. Select
`best.pt` by the unchanged weighted dev objective; report history and
reconstruction components alongside it. Both ranks contribute their RNG state to
the atomic best checkpoint. The checkpoint's model/optimizer step can be earlier
than the final training step. Only this final selected artifact is sent to W&B.
Training curves log every ten updates, and small fixed-source inference panels
log at initialization, every 1,000 updates, and finalization when time permits.

There is no held-out evaluation or retuning interface in this run. The same
optimizer/work/external hard deadlines apply. A partial result records its actual
step and most recent complete dev evaluation. Board utilization/power counters
are filtered to the explicitly assigned physical GPU IDs and are snapshots,
not integrated energy measurements; unavailable counters are not zero-filled.
