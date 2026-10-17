# Direction B: single-GPU loss v3 trial

The implementation uses the existing hybrid backbone/decoder and a separate
`hybrid_loss_v3` objective/runner. The v2 configuration and previous results are
unchanged. This is a seed-17 optimization check, not the four-arm paper result.

## Frozen trial

- [`hybrid_loss_v3_trial.json`](../../configs/mavt4d/hybrid_loss_v3_trial.json):
  1,000 maximum updates, eight source groups (64 clips) per update. The existing
  v2 JSON supplies architecture, fixture/split and AdamW defaults; its old loss
  weights, world size, update budget and auxiliary heads do not govern v3.
- One visible allocated GPU, one process, device `cuda:0`; production rejects
  other counts. The physical GPU mapping is a launcher responsibility.
- BF16 is limited to the existing spatial observer path. Recurrent state,
  attention/decoder and objectives retain their existing FP32 behavior.
- Microbatch candidates are 1, 2, 4 or 8 source groups. Accumulation preserves
  eight effective source groups and identical step-indexed source/mask draws.
  This candidate cap is not a claim of the hardware's maximum batch size.
- [`loss v3`](../plans/mavt-cd-loss-v3.md) replaces the old objective. No old
  history/semantic auxiliary head is instantiated. Structured pair labels and
  target support enter losses/measurement only.

## Capacity then fresh training

Use `scripts/mavt4d/run_hybrid_loss_v3.py` with `--capacity-only` in a fresh
output directory before the actual run. Supply `--base-config`, `--config`,
`--dataset`, `--output`, `--git-revision`, `--stop-optimization-at` and
`--stop-work-at`. Both deadlines must include an explicit timezone; the work
cutoff must be later. The caller must enforce the allocated job's hard cutoff.

Capacity uses disposable independently initialized models, one clean warmup
update and two timed complete effective main-mask updates per candidate. The
clean warmup exercises all eligible pair renders, conservatively covering their
memory cost; timing uses the same fixed step-1 source/mask schedule for all candidates. Capacity
writes `capacity.json` locally, creates no W&B run and saves no model checkpoint.
Among complete candidates it chooses the fastest measured update retaining at
least 10% VRAM headroom. Incomplete and OOM candidates cannot win. Stop without
training if no candidate completes. A different main-training mask mix can
change measured throughput; report actual training throughput separately.

Launch the actual trial in another fresh output directory with
`--microbatch-groups` set to the selected value. It starts from the canonical
seed-17 initialization, never from capacity/old trial weights. W&B project is
`semantic-router`, display name **semantic router**, with a fresh run ID. Secrets
come only through the existing SDK environment/credential mechanism, never the
scientific config or committed command examples.

For ROCm deployments, compute visibility environment variables alone can leave
DRM handles open on other devices. A strict one-GPU scope requires the allocated
render node and KFD in the container/device allowlist. Before training, verify
the physical PCI/UUID mapping, Torch device count of one, an actual backward
pass, host file descriptors and scheduler claims. The visible device ordinal
may be renumbered inside the container; it is not a physical-device identity.
Run child timeouts **inside** the container and bound PID1 by the original
absolute hard deadline. A timeout on the Docker client alone does not ensure
the workload stops, and restarting must not extend the allocation window.

## Selection, evidence and stopping

The immutable run lock records both configurations, objective constants, source
partition/hash, manifest hash, code revision, precision, packing and deadlines.
Canonical held-out sources are not evaluated or used for tuning.

The complete fixed-dev partition is evaluated at initialization, every 200
updates and the final completed update. Probe means/std and ridge parameters
are refit solely on that checkpoint's optimization-train features. The frozen
selection score is exactly [the 2×2 protocol](../benchmarks/mavt-cd-loss-v3-ablation.md):

`S_dev = (E_C + E_D/2 + E_R/2)/3`, where
`E_C = ((1-C_direction_accuracy)+(1-C_core4_accuracy))/2`.

E_D is frame-3 reciprocal D-swap paired-target balanced L1. E_R is balanced
self-reconstruction across all six endpoints. Every expected source/pair must
be valid; partial evaluations never select a checkpoint. A tie keeps the earlier
step. Selection reads clean inputs; it does not substitute a training head's
RGB-history error. The broader anisotropic-mask, controls and held-out paper
suite remains a separate evaluation task.

Only `best.pt` is retained atomically, with optimizer, RNG and exact restart
metadata. Best may be step zero if no later full dev improves. One selected
artifact is sent to W&B at completion; earlier runs/artifacts are preserved.
Training curves log objective terms, eligibility counts, source/clip exposure,
throughput and available assigned-GPU counters. Same-frame diagnostic curves
track D → modulation → routing/gain → RGB without auxiliary heads.

Inference panels are explicitly labeled **current weights**, separate from
best-checkpoint identity. At frame 3 they show own target, reconstruction,
D-swap output, donor target and D-zero output for both configurations. Only the
image display is clipped; selection metrics use unclipped predictions.

Optimization checks deadlines between accumulated microbatches and before each
optimizer step; an interrupted partial accumulation is discarded. The runner
reserves measured time for final full-dev and checkpoint work, handles SIGINT/
SIGTERM cooperatively, and also requires an external hard job deadline.

## Local correctness

`PYTHONPATH=src python -m pytest tests/mavt4d/test_hybrid_loss_v3_objectives.py tests/mavt4d/test_hybrid_loss_v3_runner.py -q`

CPU mode is restricted to explicit `--allow-cpu` correctness runs. Tests include
one tiny real optimizer update; no CPU fallback is permitted in production.
The small test architecture is not a training preset.
