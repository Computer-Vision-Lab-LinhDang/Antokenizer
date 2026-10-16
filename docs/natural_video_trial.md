# Natural-video hybrid feasibility trial v1

This independent trial tests whether the existing HybridCDModel can optimize on real egocentric RGB video. It does not replace the frozen synthetic pilot and does not establish that C represents event semantics or that D uniquely captures configuration. No synthetic C/D labels are assigned to natural clips.

## Data contract

The bounded export reads OmniWorld EPIC-KITCHENS RGB archives and their metadata. Up to 16 original recordings are selected by a fixed hash, with up to eight video segments per recording. The source group is `epic:Pxx/Pxx_recording`; all segments from that recording stay together. The canonical `mavt-split-v1` SHA-1 bucket with video validation fraction 0.02 excludes validation recordings entirely. A second frozen SHA-256 bucket assigns 20% of the remaining recording groups to development. Development is used for checkpoint selection; there is no final held-out evaluation in this trial.

Each clip contains 12 distinct frames, selected approximately uniformly from the available frame indices in the first two seconds. Timestamps are actual frame indices divided by the metadata FPS and shifted to start at zero. Missing or invalid FPS is excluded, including `X`; accepted FPS must be finite and in (0,240]. Frame spacing can vary and is not replaced with a nominal FPS. Counts, skipped clips, selected indices, FPS provenance, input hashes, metadata hash, and manifest hash are recorded in the export. Dataset outputs are immutable and existing destinations are rejected.

The exporter streams compressed archives and decodes only selected PNGs. It can terminate an archive early only after obtaining every integer frame index in the bounded time interval for every selected clip; otherwise it reads to the end. It does not assume contiguous archive member ordering. Full archive checksums are not claimed. RGB is resized to 128×192, changing the original aspect ratio. Intrinsics and poses are unavailable and omitted rather than invented. This is a visual sequence trial, not a geometry benchmark.

## Frozen model and objective

The model uses the exact default hybrid architecture: eight stateless spatial Transformer blocks at width 384; two C/D recurrent stages at width 256; 32×32 C and D state values per stage; 16 summary tokens; 8 query and 2 KV heads; a 16-step KV window; and the four-block C/D-only tile decoder. Each recurrent step observes two frames. Spatial computation uses BF16 on GPU; recurrent state, model parameters, history head, and decoder remain FP32. Fresh initialization uses seed 23; synthetic-pilot weights are not loaded.

The batch contains 48 clips across six ranks. The initial packing is two clips per microbatch and four gradient accumulation steps per rank. Per-clip normalization makes the loss independent of microbatch packing. AdamW uses learning rate 3e-4, betas (0.9,0.95), weight decay 1e-4, and gradient clipping at 1. The nominal budget is 600 optimizer updates, always bounded by external optimization/work deadlines.

The objective is:

`L = observed_RGB_L1 + missing_RGB_L1 + 0.5 future_RGB_L1 + 0.2 ordered_RGB_history_MSE`.

Training chooses clean observations, a persistent wide horizontal band mask, or a persistent vertical tube mask with equal probability. Mask locations are seeded per clip and optimizer update. Missing pixels are zeroed before Patchify and carried separately as availability. Observed and missing RGB errors are normalized within each clip over six tubelet endpoints, then averaged over clips; the missing term is zero for a fully observed clip. Future targets are frames 7 and 9, decoded after observationless rollout from the state that has seen only frames 0–5.

The history target concatenates pooled signed RGB differences `(frame7−frame2, frame6−frame3)`, with 8×12 pooling and an RMS normalization floor of 0.02. A linear head reads C from both stages. This is an order-sensitive RGB surrogate, not an action, event, object, or configuration label. Camera motion, lighting, and appearance may influence this target. No event-semantic conclusion follows from a lower value.

## Selection, logs, and restart

Every 25 optimizer updates, all development clips are evaluated with fixed per-clip masks and seeds. Equal per-clip mean total dev loss is the selection criterion, fixed before training. Every rank checks the work deadline at the same development-round boundaries. An interrupted development pass never selects a checkpoint.

Only `best.pt` is retained. A lower complete dev loss creates a temporary snapshot, fsyncs it, then atomically replaces `best.pt`; there are no periodic or final-last checkpoint copies. The snapshot includes model and history head, optimizer, every rank's RNG, optimizer step and next step, source cursor, configuration/data hashes, source revision, W&B run ID, and dev selection score. Resume validates these fields and records explicitly when restoring the best step rewinds a later observed optimizer step. Only the final best checkpoint is submitted as a W&B artifact; online upload success remains subject to the account quota.

W&B and a local JSONL journal record train/dev curves, optimizer progress, synchronized update time, throughput, memory, and available GPU utilization/power snapshots restricted to the allocated devices. These snapshots are not energy integrals. Every 100 updates, a small fixed development image panel shows masked last input, reconstruction, last-frame ground truth, a future prediction from the six-frame prefix, and its target. Local PNGs remain available if online media upload fails.

The final result reports either completion of all 600 updates or a visible bounded stop. An earlier best checkpoint can remain the final saved model. It must not be confused with the last optimizer state. Deadline termination, data limitations, and the absence of held-out semantic C/D labels are part of the reported result.

## Checks before GPU launch

Tests cover recording-disjoint splits, exclusion of canonical validation, invalid FPS, interleaved archive ordering, timestamps from exported pixels, immutable exports, missing-pixel causality, order-sensitive history targets with identical final frames, finite backward, mixed-mask microbatch loss/gradient parity, atomic best replacement, and a CPU fresh-run/resume round trip. Existing hybrid restart, causality, and step/chunk/scan parity tests remain unchanged.
