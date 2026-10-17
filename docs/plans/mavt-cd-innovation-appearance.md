# Innovation write and appearance bank — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop C from collapsing into a shared background code.
- C writes only new evidence: what changed in each image patch since that patch was last seen, or a patch seen for the first time.
- A separate write-once appearance bank A gives the decoder the background, so C no longer has to carry it.

**Architecture:** Two opt-in changes to the existing hybrid. Both are off by default; the v2 model, state, fingerprints and checkpoints stay byte-identical.
- **Innovation write.** Patch tokens are compared with each patch's last fully observed token, then pooled by their own attention. The result drives only the C write value. The gate, the transition matrix and the D write keep reading the full evidence.
- **Appearance bank.** A (32×32 FP32) is written once, at the first observed step, and the decoder reads it beside C.
- Both run as arms of the existing loss-v3 trainer. Loss, data, split and selection are unchanged.

**Tech Stack:** PyTorch (existing shared env, no new packages), pytest.

**Spec:** this document. Its evidence comes from the C-collapse analysis summarized below and the [labelled upper bound](mavt-cd-upper-bound.md).

## Why (evidence this plan argues from)

- **At fresh init, variant information is diluted about 150× before C.** Within-source variant contrast `||X − mean_variants X|| / ||X||` was measured on three synthetic sources:

  | Stage | Contrast |
  |---|---|
  | pixels | 0.14–0.17 |
  | after the spatial blocks | 0.02 |
  | after pooling to 16 summary tokens | 0.002 |
  | C1 | 1.0e-3 → 6.4e-4 over six steps |

  The largest single drop is the pooling, which averages roughly 96 patches. The moving box covers about 6% of the image.
- **Training makes it worse.** Loss-v3 runs grow a large shared C component. The order loss sits at its 0.2 margin, the variance guard at about 0.999, and dev probes on final C are at chance.
- **Architectural cause.** The decoder takes all content from C, so reconstruction makes C carry the background. Cell writes are the full pooled observation, so the common part is rewritten and accumulated every step.
- **Labelled upper bound (CPU, one source × 8 clean clips, 200 updates).** A labelled linear head on final C stayed at chance until update 116, then reached 75%. C contrast went from 5.6e-4 to 0.58 and the order loss left its margin. History can reach C, but only after a long, unstable escape. This plan removes the shared component at the write instead.

## Design decisions

1. **Availability-aware, patch-level innovation.** Let `x_t,p` be the spatial-block output for patch p (FP32, `[V, hp, wp, spatial_width]`). A patch counts as observed when its tubelet coverage equals 1. The state carries each patch's last observed token `prev_tokens` and a flag `prev_seen`. Per patch:

   | Patch now | Seen before | δ_p | prev update |
   |---|---|---|---|
   | observed | yes | `x_t,p − prev_p` | `prev_p ← x_t,p` |
   | observed | no | `x_t,p` (new evidence), or `0` at the stream's first observed step when the appearance bank is on, because A holds it | `prev_p ← x_t,p`, `seen ← True` |
   | not fully observed | — | `0` | unchanged |

   Consequences:
   - A mask turning on or off is **not** a change.
   - A revealed patch is compared with its own last observed content, so a static background gives 0.
   - Only patches never seen before add content to C.
   - A whole missing step leaves everything unchanged.
2. **Pooling of change.** `InnovationPool` turns the δ patches into 16 tokens `[16, width]`:
   - its own learned queries;
   - keys `Linear(spatial_width → width)`;
   - values `Linear(spatial_width → width, bias=False)`, so all-zero δ pools to exactly 0;
   - coverage-masked attention.

   The pooled result is then normalized with a floor: `δ̂ = pool / (rms(pool) + κ·rms(e))` per token, where e is the observer's summary evidence and `κ = 1e-3`. A static scene gives `δ̂ = 0`, not amplified noise.
3. **Only the C write value reads δ̂.**
   - `b[..., 0] = α · tanh(innovation_head(innovation_attention(write_queries, δ̂, δ̂)))`. `innovation_head` is `Linear(width → state_dim, bias=False)`, so with no change there is no C write.
   - The gate α, the 2×2 matrix and the D write `b[..., 1]` read the full evidence as before. D carries configuration, which is visible in every frame.
   - Coefficients depend only on observations and carried buffers, never on C/D, so the affine associative scan stays exact.
4. **Write-once appearance bank.**
   - `A ∈ [B, appearance_tokens = state_tokens, state_dim]` FP32 is written at the first observed step of each stream by `AppearanceWriter`: learned queries, LayerNorm(e), attention, then `Linear(width → state_dim)`.
   - A is never rewritten, so later motion has to come from C.
   - A belongs to both the restart state and the readout code.
5. **The decoder reads A.**
   - Memory tokens are `[C (2×state_tokens slots); A (state_tokens slots)]`.
   - A has its own identity keys `appearance_keys [state_tokens, width]` and shares `content_keys`/`content_values`.
   - Contract: `C = 0 and A = 0 → RGB = 0`.
6. **D→C feedback is kept, and its leakage is measured.**
   - The arms keep `feedback=True`, so the comparison with B changes one factor.
   - The evaluator logs `dc_feedback_ratio` per stage: rms of the in-cell D→C term `a01·d_prev` over rms of the C write `b_c` at dev.
   - Rule: if the winning arm's ratio exceeds 1 at the end of Stage 1, Stage 2 adds the same arm with `feedback=False`.
7. **Versioned contracts; v2 untouched.**
   - `HybridV3Config(HybridConfig)` adds `innovation: bool`, `appearance_tokens: int` (0 or `state_tokens`), `innovation_floor: float` and `innovation_grid: tuple[int, int]`. The grid is the patch grid `(hp, wp)`, which fixes the input resolution; `(8, 12)` for 128×192.
   - Fingerprint schema is `mavt-hybrid-cd2-kv-v3`.
   - `HybridCDAKVState(HybridCDKVState)` adds `appearance`, `appearance_valid`, `prev_tokens` and `prev_seen` (`SCHEMA_VERSION = 3`). `CDACode(CDCode)` adds `appearance`.
   - New modules exist only when the flags ask for them, so v2 `state_dict` keys are unchanged.
8. **Restart payload.** For 128×192 with one view and both flags on, it grows from 148,256 B to 299,905 B per stream:
   - `prev_tokens` 8×12×384 FP32 = 147,456 B;
   - `prev_seen` 96 B;
   - A 4,096 B;
   - `appearance_valid` 1 B.

   This is the price of mask-safe innovation. A summary-level difference was rejected because a mask turning on or off would count as change, and all eight variants share their mask.
9. **Trainer arms.**
   - B (existing), I (innovation), A (appearance) and IA (both).
   - Each arm is selected by the trial-config version, like arm U. Base config and its hash are unchanged; only the new arms write `objective.architecture_variant` into the lock.
   - Loss, data, split, optimizer, seed and selection are identical across arms.
10. **Memory controls in the evaluator (v3 arms).**
    - `c_core4_lastwrite`: probe final C after C/D were zeroed before the last step, with KV, `prev_tokens` and A kept. This separates history carried in C from history re-derived at the last write.
    - `a_core4`: probe A. First frames differ by core start position, so core information could live in A.

## Global Constraints

- The v2 fingerprints stay exact: `HybridConfig().fingerprint() == "cc08e2c847a982f03d9c582bfd5d458debff8689f60d1a243c4ef0e73111b376"` and `HybridConfig.tiny().fingerprint() == "e41a5d597856790e4396c1efec71725d0b687804c8c28dca1f4a165e4fd39992"`.
- Existing trial configs keep their pinned sha256 values (`test_hybrid_loss_v3_upper_bound.py`), and the base config hash stays `ea21c934…`.
- C, D, A and `prev_tokens` are FP32. Only spatial compute may use BF16, and KV stays BF16.
- No `pip install`. Tests run with the shared env: `PYTHONPATH=src:tests/mavt4d python -m pytest … -m "not server"`.
- Labels never enter model inputs.
- The repo is public: no host, account or path names in code or docs.
- GPU runs are capped at **one hour** of wall-clock per allocation, including dev passes. Optimizer, work and hard deadlines all fall inside it.
- TDD: write the test, see it fail for the right reason, implement, then commit and push after each task.

## Review Focus

- **The first observation arrives after missing steps.** A and `prev_seen` are written at the first *observed* step, not at step 0. Owned by Task 5.
- **A fully static clip.** No NaN or Inf, `δ̂` exactly 0 after the first step, and C keeps only its transition. Owned by Task 2.
- **A missing step with `dt = 0` and with `dt > 0`.** A, `prev_tokens` and the flags stay unchanged, and `dt = 0` remains an exact no-op for the whole payload. Owned by Task 5.
- **Streams at different progress in one batch.** Per-stream flags decide, never batch-wide ones. Owned by Task 5.
- **A mask that turns on or off over a static scene.** `δ̂ = 0` at both transitions. A patch hidden at the first step and revealed later counts as first seen when revealed: it writes C, not A. Owned by Task 2 for δ and Task 5 for C/A.

---

### Task 1: v3 config, state and code (contracts only)

**Files:**
- Modify: `src/mavt4d/hybrid/config.py` (add `HybridV3Config` and `is_v3`)
- Modify: `src/mavt4d/hybrid/state.py` (add a `STATE_FIELDS` ClassVar used by every method in place of the module tuple; add `CDACode` and `HybridCDAKVState`)
- Test: `tests/mavt4d/test_hybrid_v3_state.py`

**Interfaces:**
- Produces:
  - `HybridV3Config(HybridConfig)`: frozen dataclass with fields `innovation: bool = False`, `appearance_tokens: int = 0` (0 or `state_tokens`), `innovation_floor: float = 1e-3` and `innovation_grid: tuple[int, int] = (8, 12)`.
    - It requires `innovation or appearance_tokens`.
    - `fingerprint()` uses schema `"mavt-hybrid-cd2-kv-v3"` with all fields.
    - `HybridV3Config.tiny(**overrides)` uses the v2 tiny dimensions with `innovation_grid=(4, 6)`.
  - `is_v3(config) -> bool`.
  - `CDACode(CDCode)`: adds `appearance [B, appearance_tokens, state_dim]` FP32.
  - `HybridCDAKVState(HybridCDKVState)`, with `SCHEMA_VERSION = 3`, adds four fields:
    - `appearance [B, appearance_tokens, state_dim]` FP32;
    - `appearance_valid [B]` bool;
    - `prev_tokens [B, reference_views, hp, wp, spatial_width]` FP32;
    - `prev_seen [B, reference_views, hp, wp]` bool.

    `initial()` sets them to zeros and False, and `code()` returns a `CDACode`. With a flag off, its tensors have zero length in the token, grid or patch dimension.

- [ ] **Step 1: Write the failing tests**
  - `test_v2_fingerprints_unchanged`: the two hashes in Global Constraints.
  - `test_v3_config_validates_and_fingerprints`: the v3 fingerprint differs from v2; `appearance_tokens=16` raises with the default dimensions; both flags off raises.
  - `test_v3_state_initial_shapes_and_bytes`: with default dimensions, both flags on and grid `(8, 12)`, `payload_nbytes − HEADER_NBYTES == 299,905` per stream.
  - `test_v3_payload_roundtrip_and_schema_isolation`: the round trip is exact, a v2 payload is rejected for a v3 config, and vice versa.
  - `test_slice_clone_detach_keep_v3_fields`: `slice_state`, `clone(detach=True)` and `snapshot` keep all four new fields.
- [ ] **Step 2: Run, expect FAIL** (`ImportError: HybridV3Config`).
- [ ] **Step 3: Implement.** In base methods, `_STATE_FIELDS` becomes `type(self).STATE_FIELDS`. The subclasses extend it and override `validate`, `initial` and `code`.
- [ ] **Step 4: Run Task 1 tests, then `test_hybrid_state.py` and `test_hybrid_restart.py`.** Expect all PASS.
- [ ] **Step 5: Commit** `feat(mavt4d): v3 config, state and readout code for innovation/appearance`.

### Task 2: Observer patches and patch-level innovation

**Files:**
- Modify: `src/mavt4d/hybrid/observation.py` (`forward_batch(..., return_patches: bool = False)`, plus the matching return from `_encode_group`)
- Create: `src/mavt4d/hybrid/innovation.py`
- Test: `tests/mavt4d/test_hybrid_innovation.py`

**Interfaces:**
- Consumes: `HybridV3Config` (Task 1).
- Produces:
  - `forward_batch(observations, *, modality, return_patches=False)`. With `True` it returns `(evidence [N,P,W], available [N], patches [N,V,hp,wp,S] FP32, coverage [N,V,hp,wp])`. Patches are the post-spatial-block tokens the pool reads, and rows that aren't available are zero. With `False` the return is unchanged.
  - `patch_innovation(patches [T,B,V,hp,wp,S], coverage [T,B,V,hp,wp], prev_tokens [B,V,hp,wp,S], prev_seen [B,V,hp,wp], stream_first_observed [T,B] bool, *, appearance_on: bool) -> (delta [T,B,V,hp,wp,S], prev_tokens, prev_seen)` follows the table in Design decision 1.
  - `InnovationPool(config)` with `forward(delta [N,V,hp,wp,S], evidence [N,P,W]) -> δ̂ [N,P,W]`, as in Design decision 2.

- [ ] **Step 1: Write the failing tests**
  - `test_static_clip_gives_zero_after_first_step`: δ is 0 at steps ≥ 1, and `InnovationPool` output is exactly 0 there.
  - `test_mask_on_off_is_not_change` (Review Focus 5): a static scene where a rectangle is masked at steps 1–2 and visible at 3. δ is 0 at every step ≥ 1 and the masked patches' prev is unchanged during the mask.
  - `test_first_seen_patch_is_new_evidence`: a patch hidden at step 0 and visible at step 2 gives `δ = x` there, even when `appearance_on=True`.
  - `test_first_step_zero_with_appearance`: at the stream's first observed step with `appearance_on=True`, δ is 0 and prev is written.
  - `test_chunked_equals_full`: threading prev through two chunks equals the single call bitwise.
  - `test_floor_bounds_small_changes`: for a pooled change of `1e-6·rms(e)`, `rms(δ̂) ≤ 1e-3 + 1e-9`.
  - `test_return_patches_false_is_unchanged`: the default call returns the same tensors as before.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Loop over T (T ≤ 16) with per-patch `torch.where`.
- [ ] **Step 4: Run, plus `test_hybrid_batched_observer.py` and `test_hybrid_spatial_sequence_batch.py`.** Expect PASS.
- [ ] **Step 5: Commit** `feat(mavt4d): availability-aware patch innovation and change pooling`.

### Task 3: Cell reads innovation for the C write

**Files:**
- Modify: `src/mavt4d/hybrid/cells.py` (`HybridCoupledCell.__init__`, `coefficients`, `scan`)
- Test: `tests/mavt4d/test_hybrid_innovation_cell.py`

**Interfaces:**
- Consumes: `is_v3` (Task 1), and `δ̂ [T,B,P,W]` from Task 2.
- Produces: `coefficients(evidence, observed, delta_time, innovation: Tensor | None = None)` and `scan(..., innovation: Tensor | None = None, method=...)`.
  - Under `config.innovation`, `innovation` is required and `b[..., 0]` follows Design decision 3. `b[..., 1]`, α and the matrix are unchanged.
  - New modules `innovation_attention` (`nn.MultiheadAttention(width, q_heads)`) and `innovation_head` (`nn.Linear(width, state_dim, bias=False)`) exist only under `config.innovation`.
  - Helper `dc_feedback_ratio(a [.., 2, 2], previous [.., 2], b [.., 2]) -> Tensor` returns `rms(a[...,0,1]·previous[...,1]) / rms(b[...,0])`, for the evaluator.

- [ ] **Step 1: Write the failing tests**
  - `test_v2_cell_has_no_new_parameters`.
  - `test_c_write_ignores_evidence_given_innovation`: perturb evidence with `innovation` fixed. `b[..., 0]/α` is unchanged and `b[..., 1]` changes.
  - `test_zero_innovation_gives_zero_c_write`.
  - `test_innovation_required_when_enabled`.
  - `test_scan_parity_with_innovation`: values and gradients match within rtol 2e-5, atol 2e-6.
  - `test_dc_feedback_ratio_matches_definition`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Missing rows keep the existing missing transition.
- [ ] **Step 4: Run, plus `test_hybrid_attention.py` and `test_hybrid_cell_missing_efficiency.py`.** Expect PASS.
- [ ] **Step 5: Commit** `feat(mavt4d): innovation-driven C write in coupled cells`.

### Task 4: Appearance writer and decoder

**Files:**
- Create: `src/mavt4d/hybrid/appearance.py`
- Modify: `src/mavt4d/hybrid/decoder.py` (`HybridQueryDecoder.__init__`, `decode_tiles`)
- Test: `tests/mavt4d/test_hybrid_appearance.py`

**Interfaces:**
- Consumes: `CDACode`, `HybridV3Config` (Task 1).
- Produces:
  - `AppearanceWriter(config)` with `forward(evidence [B,P,W]) -> [B, appearance_tokens, state_dim]`.
  - `write_once(appearance, appearance_valid, written, observed) -> (appearance, appearance_valid)`.
  - The decoder creates `appearance_keys [appearance_tokens, width]` only when `appearance_tokens > 0`. For a `CDACode`, memory is `cat(c.flatten(1, 2), appearance)` and identity is `cat(stage+slot identity, appearance_keys)`.

- [ ] **Step 1: Write the failing tests**
  - `test_write_once_keeps_first_observation`.
  - `test_unobserved_step_does_not_write`.
  - `test_decoder_zero_codes_render_zero`: `C = 0`, `A = 0` gives RGB exactly 0.
  - `test_decoder_uses_appearance`: `C = 0` with A nonzero gives RGB that isn't all zero, and the gradient reaches `AppearanceWriter`.
  - `test_v2_decoder_unchanged`: same `state_dict` keys, and `CDCode` still decodes.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, plus `test_hybrid_io.py`.** Expect PASS.
- [ ] **Step 5: Commit** `feat(mavt4d): write-once appearance bank read by the decoder`.

### Task 5: Model integration, restart and causality

**Files:**
- Modify: `src/mavt4d/hybrid/model.py` (`__init__`, `initial_state`, `step`, `forward_sequence`, `forward_encoded`, `rollout`)
- Test: `tests/mavt4d/test_hybrid_v3_model.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: with `HybridCDModel(HybridV3Config(...))`, every entry point returns `HybridCDAKVState`.
  - `forward_encoded(state, evidence, observed, times, method=..., patches=None, coverage=None)` requires `patches`/`coverage` when `config.innovation` is set. `step` and `forward_sequence` get them from `forward_batch(return_patches=True)`.
  - Rollout leaves A, `prev_tokens` and both flags unchanged.
  - The v2 path never calls `patch_innovation`, `InnovationPool`, `AppearanceWriter` or `write_once`.

- [ ] **Step 1: Write the failing tests** (tiny v3, both flags on, plus each flag alone):
  - `test_v3_restart_in_fresh_process`: before and after a ring wrap, payload only. Continuation is exact.
  - `test_v3_step_chunk_scan_parity`: values and gradients within rtol 3e-5, atol 3e-6.
  - `test_v3_causality`: perturbing future RGB leaves prefix states, A and prev unchanged.
  - `test_first_observation_after_missing_writes_A_and_prev` (Review Focus 1).
  - `test_missing_dt_zero_is_exact_noop_and_dt_positive_keeps_A_prev` (Review Focus 3).
  - `test_streams_at_different_progress` (Review Focus 4).
  - `test_revealed_patch_writes_C_not_A` (Review Focus 5).
  - `test_v2_path_never_calls_v3_components`: monkeypatch the four v3 functions to raise, then run a v2 tiny model forward and decode.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** Branch on `is_v3(self.config)` only.
- [ ] **Step 4: Run the Task 5 tests and all of `tests/mavt4d -m "not server"`.** Expect all PASS.
- [ ] **Step 5: Commit** `feat(mavt4d): integrate innovation write and appearance bank into the hybrid`.

### Task 6: Trainer arms, evaluator controls, init probe and configs

**Files:**
- Modify: `src/mavt4d/hybrid_loss_v3/config.py` (versions and the `architecture_variant` property)
- Modify: `src/mavt4d/hybrid_loss_v3/objectives.py` (`LossV3Module(cfg, core_head_weight=0.0, architecture_variant=None)` builds `HybridCDModel(HybridV3Config(**asdict(cfg.architecture), innovation_grid=(cfg.H // patch, cfg.W // patch), **variant))`)
- Modify: `src/mavt4d/hybrid_loss_v3/runner.py` (`initialize(..., architecture_variant=None)` at both call sites; arm names; `lock["objective"]["architecture_variant"]` for new arms only)
- Modify: `src/mavt4d/hybrid_loss_v3/evaluation.py` (for v3 arms only: `c_core4_lastwrite` features, `a_core4`, `dc_feedback_ratio`)
- Create: `configs/mavt4d/hybrid_cd_arch_{innovation,appearance,innovation_appearance}{,_tiny}.json`. Each copies the existing trial and upper-bound tiny settings except `version`, with `log_every = 1`.
- Create: `scripts/mavt4d/probe_variant_contrast.py`. It reports per-stage within-source contrast for arms B/I/A/IA on N synthetic sources at fresh init (seed 17), as JSON.
- Test: `tests/mavt4d/test_hybrid_arch_arms.py`

**Interfaces:**
- Produces:
  - Versions `hybrid-cd-arch-{innovation,appearance,innovation-appearance}-v1` and `…-tiny-v1`, with arms `I`, `A` and `IA`. `architecture_variant` returns `{"innovation": bool, "appearance_tokens": 0 | state_tokens}`, or `None` for B and U.
  - `last_write_only_states(model, batch) -> HybridCDAKVState` runs steps 0…T−2, zeroes C/D, then runs the last step.
- [ ] **Step 1: Write the failing tests**
  - `test_existing_trial_hashes_unchanged`.
  - `test_arm_versions_select_variants`.
  - `test_b_lock_unchanged_and_variant_lock_for_new_arms`.
  - `test_last_write_only_zeroes_cd_before_last_step`.
  - `test_v3_dev_reports_controls`: a tiny IA CPU trial with two updates gives a complete dev with `c_core4_lastwrite`, `a_core4` and `dc_feedback_ratio` logged.
  - `test_probe_script_runs_on_tiny_config`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, plus all `tests/mavt4d -m "not server"`.** Expect PASS.
- [ ] **Step 5: Commit, push the work branch, then pull and push `HEAD:feat/mavt4d`.**

## Experiments (after all tasks; rules fixed before running)

All arms use seed 17, the loss-v3 objective and `event-cd-v1` at 128×192: 94 train / 9 dev sources, with held-out untouched.

**Stage 0 — init probe (CPU, no training).** Run `probe_variant_contrast.py` on 3 sources.
- Pass for I and IA: final-step C1 contrast ≥ 0.05 (B measured 6.4e-4), and `rms(δ̂)` in [0.1, 10] at steps 1–5.
- If it fails, fix the change path before any training.

**Stage 1 — tiny, unlabelled (CPU allowed).** One source × 8 clean clips, 200 updates per arm.
- Per arm, report:
  - order loss over the last 20 updates;
  - final train C contrast;
  - reconstruction L1;
  - the first update where the order loss drops below 0.2;
  - `dc_feedback_ratio` at the final dev.
- An arm escapes collapse if:
  - its last-20 order loss is < 0.15;
  - its final C contrast is ≥ 0.05;
  - its reconstruction L1 is at most B's + 0.02.
- B is expected to fail, as in every earlier run.

**Stage 2 — main, GPU, one-hour allocation.** Run B and the best Stage-1 arm (plus its `feedback=False` twin under Design decision 6, or all arms if Stage 1 has no single winner) concurrently on the allocated GPU.
- Setup:
  - 8 sources/update;
  - identical schedule;
  - the same `--microbatch-groups` for every arm;
  - `steps = 1000` as a ceiling;
  - optimizer, work and hard deadlines inside the hour;
  - online tracker;
  - pre-run GPU check: on one clean source, log `innovation_rms` at steps 1–5 through the main path and through `last_write_only_states`. Static patches must give δ̂ ≈ 0; flag the run if the last-write path exceeds 1e-2 where the main path is ~0.
- Primary measure: dev `c_core4` with its 95% source-bootstrap interval at the last dev step that both B and the arm completed (`dev_every = 200` gives common boundaries); the arms spend more per dev pass and may stop earlier than B.
- An arm improves on B only if all three hold:
  - its `c_core4` interval low exceeds B's interval high;
  - its `c_core4` interval low exceeds its own `c_core4_lastwrite` interval high, so the win is history carried in C;
  - its `E_R` is no worse than B's by more than 0.02.
- Also report:
  - `c_direction`, `E_D`, `c_variant_contrast`, `c_core4_train`;
  - for A arms, `a_core4` and the A-only render error (C zeroed).
- A deadline-limited run is reported as partial, with its actual step.

**Limitations.**
- Six steps fit inside the KV window, and KV can resupply history at the last write. The last-write control bounds this but does not test memory beyond KV.
- Write-once A and a fixed patch grid suit static-camera synthetic clips. Camera motion needs a refresh rule, deferred until a synthetic win exists.
- `κ = 1e-3` is a fixed design value, checked by Stage 0 and not tuned on dev.
