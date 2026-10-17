# MAVT C/D — labelled upper bound for C

Last updated: **2026-10-17**. Status: **design locked before any GPU run; no learning result yet**.

[Loss v3 plan](mavt-cd-loss-v3.md) ·
[Single-GPU loss v3 runbook](../runbooks/hybrid-loss-v3-trial.md) ·
[Existing backbone](mavt-hybrid-v2-design.md)

## Question

In loss v3 trials, C is collapsed. It is almost identical across the eight variants
of a source. The order loss sits exactly at its margin (0.2 per eligible source) and
the variance guard at about 0.999. Dev ridge probes on final C are at chance.

This experiment asks one question: **can the existing architecture carry event
history into final C at all, if C is supervised directly with the synthetic core
label?**

- If it cannot, even with labels, the bottleneck is architectural.
- If it can, the unlabelled loss is the bottleneck.

### Why a labelled head

At fresh initialization, the within-source variant contrast
`||X − mean_variants(X)|| / ||X||` was measured on CPU for three sources. It falls
from 0.14–0.17 in pixels to about 0.02 after the spatial blocks, about 0.002 after
pooling to 16 summary tokens, and 0.001–0.0006 in C.

The current C objectives are relative: cosine distances and a variance hinge. Their
gradient scales with the tiny differences, so it vanishes as C collapses. A linear
cross-entropy head gives every clip its own target. Its gradient `Wᵀ(p − y)` stays
nonzero and differs between variants even when all C rows are identical.

## What changes

Arm **U** adds exactly one training-only module:

```
logits = Linear(2048 → 4)(states[-1].c.flatten(1))
L_U = L_v3 + 1.0 · CE(logits, core)
```

- **Input:** the head reads the same final-endpoint C1+C2 features as the dev
  ridge probe (`states[-1].c`, frames 10–11).
- **Labels:** within a source and configuration, all four cores share the final
  two frames, so the label can only come from history. Labels enter the loss only.
  They are never model inputs, and the head is not part of inference or decoding.
- **Reduction:** cross-entropy is averaged over the eight clips of a source, then
  over sources, like every v3 term.
- **Masks:** the v3 masks are unchanged. Train accuracy is logged for all clips,
  for fully observed sources (`clean`) and for masked sources.
- **Unchanged:** Patchify, spatial blocks, temporal attention, C/D cells, KV, the
  decoder, the v3 terms and weights, the data, the split, the optimizer, the seed
  and the selection score.

Arm and head weight derive from the trial config version. Existing configs keep
their hashes and their locks are unchanged. Warm-starting or resuming U from a
B checkpoint is rejected with an explicit arm mismatch, so U always starts fresh.

## Profiles

| Profile | Config | Data per update | Updates | Masks | Purpose |
|---|---|---|---|---|---|
| U-tiny | `hybrid_loss_v3_upper_bound_tiny.json` | first training source, 8 variants | 200 | clean | Can the model fit 8 labelled clips? |
| U-main | `hybrid_loss_v3_upper_bound.json` | 8 random training sources = 64 clips | 1000 | five v3 families | Does supervised history reach C and generalize? Logs every update |
| B0 (matched) | `hybrid_loss_v3_trial.json` (existing) | same schedule as U-main | 1000 | same | Same budget, no head |

U-main and B0 share seed 17, the source/mask schedule, the learning rate
(AdamW 3e-4, constant), the dev cadence (every 200 updates) and the number of
updates. B0 is needed because no v3 run has a complete dev evaluation beyond
step 0.

## Measurements

- **Train (every update for U):**
  - `core_ce`, `core_accuracy`;
  - `core_accuracy_{clean,masked}_sum` with `active_source_fraction/core_{clean,masked}`.
    These are source means that aggregate exactly across microbatches. The
    per-update ratio is logged as `core_accuracy_{clean,masked}` (0 when no such
    source). Over a window, accuracy = Σ sums / Σ fractions;
  - `c_variant_contrast` of final C (both arms).
- **Dev (complete fixed partition, unchanged evaluator):**
  - ridge probes on final C and D (`c_core4`, `c_direction`, …);
  - a 95% source-bootstrap interval for each probe (1000 resamples, seed 31415);
  - the closed-form ridge fit on the training sources (`<probe>_train`), as a
    capacity control that does not depend on SGD. Features (2048) outnumber
    training clips (752), so a high value is weak evidence. A value near chance
    means the information is absent;
  - `c_variant_contrast` and `d_variant_contrast`;
  - `S_dev`.

  Dev has 9 sources (72 clips). Decisions use the interval, never a point
  estimate alone.
- Canonical held-out sources stay untouched.

## Decision rules (fixed before running)

**U-tiny.** A = mean train `core_accuracy` over the last 20 updates (chance 0.25).

| A | Verdict |
|---|---|
| ≥ 0.90 | Fits: proceed to U-main |
| ≤ 0.50 and dev-pass `c_core4_train` ≤ 0.50 | **Blocked**: neither the SGD head nor a closed-form ridge fit separates cores from final C; go to architecture changes, skip U-main |
| ≤ 0.50 but `c_core4_train` > 0.50 | Optimization/scale limit of the head, not evidence about the architecture; report, decide with the user |
| otherwise | Inconclusive: report, decide with the user |

**U-main.** Inputs: Acc_clean = Σ `core_accuracy_clean_sum` / Σ
`active_source_fraction/core_clean` over the last 100 updates; dev `c_core4` and
its interval at the last complete dev pass.

| Outcome | Condition | Meaning |
|---|---|---|
| Carries and generalizes | Acc_clean ≥ 0.60 and dev `c_core4` ≥ 0.50 with interval low > 0.25 | With direct supervision the architecture routes history into final C, and it transfers to new sources. The unlabelled objective becomes the main suspect, to be tested next; this result alone does not prove it |
| Carries, does not generalize | Acc_clean ≥ 0.60, dev interval low ≤ 0.25 | History reaches C for training sources only; data or regularization limits |
| Blocked | Acc_clean ≤ 0.35 and `c_core4_train` ≤ 0.50 | Neither supervision nor a closed-form fit finds history in C at this scale; architectural bottleneck |
| Inconclusive | anything else, including Acc_clean in (0.35, 0.60) or dev `c_core4` < 0.50 with interval low > 0.25 | Report with all numbers |

The head's effect is attributed only if U's dev `c_core4` interval lies entirely
above B0's. Contrast metrics are reported for both arms. They are diagnostics,
not decision criteria.

## Known confounds

- Gradient clipping is global: the head's gradients share the norm budget with
  the v3 terms, so U is not a pure addition to B0. Gradient norms in earlier v3
  runs stayed below the clip value 1. Report how often U reaches it.
- AdamW weight decay also applies to the head.
- `c_variant_contrast` is now also logged for B runs. It is detached and does not
  change B's loss.
- Both arms must run with the same `--microbatch-groups`; check the two run locks.
- U is never resumed in this protocol. Resume of U runs is not separately tested.

## Limitations

- Each clip has six steps, inside the 16-step KV window. Final C can receive history
  through the last temporal-attention write, so this tests whether information can
  **reach** C, not memory beyond the KV window.
- With labels this is an **upper bound only**. It must never be reported as
  unsupervised C learning or as event semantics.
- One seed; dev has 9 sources.
- The synthetic events are four motion cores. Success does not transfer to natural
  video by itself.

## Budget

- U-tiny: about 200 single-source updates plus two dev passes.
- U-main and B0: about 1000 updates each. Earlier single-GPU measurements give
  about 4 s per 8-source update, plus six complete dev passes of 30–45 s each,
  so about 70 minutes per arm. The two arms can run concurrently on two
  single-GPU allocations.
- Train and dev progress, images and the selected best artifact go to the
  experiment tracker, as in the loss v3 runbook.
- Launch requires explicit approval. Order: U-tiny, then U-main and B0 only if
  U-tiny does not end blocked.

The runner and CLI are the loss v3 ones (`scripts/mavt4d/run_hybrid_loss_v3.py`),
with fresh outputs and the deadlines from the runbook:

```bash
# U-tiny: one source, clean, microbatch 1 (no capacity probe needed)
python scripts/mavt4d/run_hybrid_loss_v3.py --base-config configs/mavt4d/hybrid_synthetic_pilot_v2.json \
  --config configs/mavt4d/hybrid_loss_v3_upper_bound_tiny.json --microbatch-groups 1 \
  --dataset <event-cd-v1 export> --output <new dir> --git-revision <sha> \
  --stop-optimization-at <utc> --stop-work-at <utc>
# U-main and B0: same flags with hybrid_loss_v3_upper_bound.json / hybrid_loss_v3_trial.json
# and the same --microbatch-groups for both arms
```
