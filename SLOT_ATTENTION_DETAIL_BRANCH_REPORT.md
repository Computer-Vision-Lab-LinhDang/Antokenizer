# Content-Detail Split: Slot Attention vs. Window-Centered Detail Branch

_Technical report for MAVT architecture review — focused on content/detail decomposition, reconstruction quality, and 3D object failure modes._

---

## 📋 Executive summary

The original content-detail split design used slot-style global attention as the natural pooling mechanism for both the content branch and the detail branch. That design is coherent for semantic compression, but it is weak for detail reconstruction. Slot attention is global, permutation-tolerant, and learns soft ownership over the whole token set. Those properties are useful for content tokens, but they are misaligned with high-frequency residuals, which require spatial anchoring, local coverage, and predictable reconstruction paths.

The revised direction keeps slot attention for the content branch and moves the detail branch toward a window-centered local mechanism. In the current implementation, detail tokens are produced by pooling residual tokens inside coordinate windows, preserving the center position of each window. The decoder then uses position-aware detail bias so each target token prefers nearby detail tokens. This makes the detail branch function like a local high-frequency correction field instead of another set of positionless global slots.

The 3D object issue should be treated separately from the detail-branch issue. Current 3D inputs are triplane RGB renders, not voxel grids. However, the RGAT geometry assumes 4D coordinate relationships that are closer to voxel-like or true spatial grids. This mismatch can create incorrect relations across planes and should be investigated as its own failure mode.



## 🔍 Problem: using Slot Attention for both content and detail

### Content and detail have different invariances

Content tokens are expected to summarize global semantics: object identity, layout, scene type, and low-frequency structure. A global slot pooler is appropriate here because the model should be free to aggregate information from anywhere in the input sequence.

Detail tokens have the opposite requirement. They need to preserve local corrections: edges, texture, small motion, patch-level alignment, and residual information that the content approximation cannot explain. A global slot pooler asks the detail branch to learn both "what detail exists" and "where it belongs" without a stable spatial anchor. That makes the decoder solve a harder inverse problem.

### Slot attention can erase spatial ownership

Slot attention produces learned slots by cross-attending from a fixed set of learned queries into the entire token sequence. Unless strong positional metadata is preserved downstream, each slot is a global mixture. This is acceptable for content, but risky for detail because high-frequency residuals are not interchangeable across positions.

In practice, this creates several failure modes:

| Failure mode | Why it happens | Expected symptom |
| --- | --- | --- |
| Position ambiguity | Detail slots summarize residuals without a guaranteed local center | Texture appears in the wrong region |
| Detail collapse | Multiple detail slots attend to similar salient residuals | Low diversity and missing fine structure |
| Content-detail leakage | Detail branch learns semantic/global information because global slots allow it | Content branch becomes less cleanly separated |
| Decoder burden | Decoder must infer local ownership from weak or implicit signals | Blurry reconstruction or unstable artifacts |
| Poor temporal locality | Video detail slots can mix unrelated frame-local residuals | Flicker, inconsistent motion edges |

### Global detail slots are not coverage guarantees

For reconstruction, the model needs coverage. Every local region should have access to a nearby residual carrier. Global slot attention does not guarantee this. It may allocate several slots to visually salient regions and underrepresent low-salience but reconstruction-critical regions such as background texture, object boundaries, and thin structures.

This is especially problematic because detail tokens are supposed to encode residuals after content approximation. Residuals are often sparse, local, and high-frequency. A global slot pooler is free to treat them as unordered evidence; reconstruction needs them as local corrections.

## 🪟 Why switch detail to a window-centered mechanism

### The detail branch should preserve local residual geometry

The revised design treats detail as a local residual field. After content slots approximate the original feature sequence, the residual is grouped by coordinate windows. Each group produces one detail token and one center position. This gives the decoder a stable answer to the question: "which part of the target grid does this detail token belong to?"

Current implementation behavior:

| Step | Mechanism | Effect |
| --- | --- | --- |
| Content extraction | Global slot pooler | Preserves semantic and low-frequency information |
| Residual computation | `R = x - x_approx` | Isolates content-unexplained information |
| Detail extraction | Coordinate-window residual pooling | Produces local detail tokens with explicit centers |
| Decoder expansion | Distance-biased cross-attention | Makes each target prefer nearby detail tokens |

This gives content and detail different inductive biases instead of forcing both through the same global slot abstraction.

### Window-centered detail reduces decoder ambiguity

The decoder receives target positions and latent positions. For detail tokens, the distance bias penalizes attending to far-away detail positions. Content tokens remain globally available, while detail tokens become local correction candidates. This makes the decoding problem more constrained and easier to learn.

The expected benefits are:

- Cleaner separation between semantic content and local detail
- Better high-frequency reconstruction without relying on global slots
- More predictable detail coverage across the image or frame grid
- Lower risk of texture teleportation across regions
- Easier visual debugging because each detail token has a center

### This does not remove attention from detail

The design shift is not "attention vs. no attention." It is a shift from global, positionless detail slots to local, position-aware detail carriers. The decoder still uses attention to query latent tokens, but the detail branch now exposes locality explicitly instead of hoping the decoder can discover it.

## 📊 Before/after results

_Fill this table after running the controlled evaluation. Keep the same checkpoint stage, sample set, resolution, and decoding settings across both variants._

| Category | Metric | Before: content slot + detail slot | After: content slot + window-centered detail | Delta | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Image reconstruction | PSNR | 28.97 | 32.1 | TBD |  |
| Image reconstruction | SSIM | 0.813 | 0.93 | TBD |  |
| Video reconstruction | PSNR | 21.37 | 27.16 | TBD |  |
| Video reconstruction | SSIM | 0.751 | 0.86 | TBD |  |
| Video reconstruction | LPIPS | 0.452 | 0.121 | TBD |  |

## 🧪 Ablations

| Ablation | Purpose | Expected interpretation |
| --- | --- | --- |
| Content slot + detail slot | Baseline | Tests the original global/global split |
| Content slot + local detail pooling | Current direction | Tests whether local residual anchoring helps |
| Content slot + local detail with no distance bias | Decoder ablation | Separates benefit of local pooling from decoder bias |
| Content slot + window detail at different window sizes | Capacity/locality sweep | Finds tradeoff between compression and detail fidelity |
| Detail branch disabled | Lower bound | Confirms how much reconstruction depends on detail |

## 🧊 3D object issue 1: representation mismatch

The current 3D data path is triplane-based. Each object is represented as three RGB planes, typically `XY`, `XZ`, and `YZ`. This is not the same as a voxel grid. A voxel representation has an actual volumetric coordinate for every occupied or sampled cell. A triplane representation has three projected planes where one axis is missing in each plane.

This matters because the current architecture uses 4D positions `(t, x, y, z)`. For triplanes, missing axes are filled with zero. For example:

| Plane | Encoded position |
| --- | --- |
| `XY` | `(0, x, y, 0)` |
| `XZ` | `(0, x, 0, z)` |
| `YZ` | `(0, 0, y, z)` |

Those zeros do not mean real spatial coordinates. They mean "axis not present in this plane." If later modules treat them as real coordinates, the graph can learn incorrect geometric relations.

The key conclusion is that RGAT-style 4D geometric relations are better aligned with voxel-like tokens than with raw triplane RGB planes. For triplanes, the graph must be plane-aware. Otherwise, the model may overconnect tokens that only appear to share coordinates because a missing axis was encoded as zero.

## 🕸️ 3D object issue 2: graph and decoder semantics

The RGAT adjacency currently defines spatial, temporal, depth, and cross-plane relations. Depth is reserved and unused, but the spatial and cross-plane rules can still become problematic for 3D triplanes.

Potential failure modes:

| Area | Current behavior | Risk |
| --- | --- | --- |
| Temporal edges | Triplane tokens all use `t=0` | Temporal relation can become meaningless for 3D |
| Spatial edges | Spatial rule assumes same `z` and local `x/y` | `XZ` and `YZ` planes do not fit this rule cleanly |
| Cross-plane edges | Tokens connect if they share any coordinate value | Missing-axis zeros can create false cross-plane links |
| Decoder distance bias | Bias uses 4D positions but not explicit plane identity | Detail from another plane can appear artificially close |

This should be treated as a separate investigation from the detail-branch change. Even if the detail branch improves image/video reconstruction, 3D object reconstruction can still fail if the triplane graph injects wrong geometry before the content-detail split.

Recommended 3D-specific experiments:

| Experiment | Goal |
| --- | --- |
| Disable RGAT for `threed` only | Check whether RGAT is the source of 3D shattering |
| Disable temporal edges for `threed` | Remove meaningless `t=0` temporal relation |
| Make spatial edges plane-specific | Use `XY`, `XZ`, `YZ` local axes correctly |
| Restrict cross-plane edges to shared real axes | Avoid links caused by missing-axis zeros |
| Add explicit `plane_id` embedding | Let encoder and decoder distinguish planes |
| Prototype small voxel input | Test whether RGAT behaves better with true 3D coordinates |

## 🧩 Implementation references

| File | Relevant behavior |
| --- | --- |
| `src/mavt/model/content_detail_split.py` | Current content slot extraction, residual computation, and local detail pooling |
| `src/mavt/model/decoder.py` | Position-aware detail distance bias in `UnifiedDetailExpander` |
| `src/mavt/model/patchify.py` | Image/video/triplane tokenization and 4D position construction |
| `src/mavt/model/rgat.py` | RGAT adjacency construction and typed edge masks |
| `src/mavt/model/mavt.py` | Content/detail ratios and end-to-end forward path |

## 🪟 Choosing `local_detail_window_size`: why 1 instead of 2

### Where the variable lives

`local_detail_window_size` is defined in `src/mavt/model/content_detail_split.py:101` (default `1`). It is consumed inside `_local_detail_pool()` (lines 184–189) as a *coordinate divisor* that groups residual tokens into windows before mean-pooling:

```python
t_win = max(1, int(self.local_detail_temporal_window_size))
s_win = max(1, int(self.local_detail_window_size))
grouped[:, 0] = grouped[:, 0] // t_win     # temporal axis
grouped[:, 1] = grouped[:, 1] // s_win     # spatial axis (H)
grouped[:, 2] = grouped[:, 2] // s_win     # spatial axis (W)
grouped[:, 3] = grouped[:, 3] // s_win     # depth axis (3D triplane)
```

After the integer division, `torch.unique` collapses identical coordinate rows into a single group, and the detail token of that group becomes the mean of all residuals that fell inside the window.

### How `window_size=1` differs from `window_size=2`

| Aspect | `window_size=1` (chosen) | `window_size=2` |
|---|---|---|
| Grouping rule | `pos // 1 = pos` → every coordinate is its own group | `pos // 2` → adjacent 2×2 (or 2×2×2 for 3D) patches collapse into one group |
| Detail tokens / 16×16 image | 256 — full residual carrier, no compression | 64 — 4× fewer |
| Detail tokens / 8×16×16 video | 2048 (with `t_win=1`) | 256 (with `t_win=2`); 512 (with `t_win=1`) |
| High-frequency fidelity | Maximum — no averaging across positions | Reduced — each detail token is a blur over 3 neighbours, attenuating edges and texture |
| Decoder load | Higher (must consume `N_c + N` latents) | Lower by ~4× on the detail side |
| Gradient routing | Every residual position receives its own gradient → fine-grained learning signal | Gradients are averaged across the window → weaker signal per source token |
| Spatial anchor | One detail center per source token → exact local correspondence | One center per 4 (or 8) source tokens → coarser anchor; the distance-bias decoder loses locality precision |

### Why `=1` is the right default at this stage

1. **The detail branch already starts from a residual.** Content slots (N_c ≈ 25 % N) carry the global semantic. The detail branch exists specifically to recover what content slots could not approximate. Pooling that residual a second time with `window_size > 1` re-introduces the very loss the residual was created to avoid — it contradicts the purpose of the branch.
2. **Stage 2 is reconstruction-driven.** L1 and LPIPS depend directly on retaining high-frequency information. With `window_size=2`, detail tokens are blurred *before* the decoder ever sees them, so the decoder has no way to recover sharp edges, fine textures, or thin structures.
3. **Compression is not the bottleneck here.** Total post-split sequence length is `N_c + N ≈ 1.25 N`, which fits within the current compute budget. The real bottlenecks at this stage are data IO and corrupt video shards, not decoder FLOPs.
4. **It preserves the content/detail asymmetry.** With `content_ratio = 0.25`, switching detail to `window_size=2` would shrink the detail branch down to roughly the same token count as the content branch, collapsing the design into two parallel coarse-scale views instead of a "global semantic + local correction" split.

### When `window_size=2` would become defensible

- A much larger dataset where decoder compute becomes the genuine bottleneck.
- A pure representation-learning regime where downstream tasks are semantic (classification, retrieval) rather than pixel-faithful reconstruction.
- An explicit regularization study where forcing the detail branch through a narrower bottleneck is the experimental goal — but this should be gated on a paired LPIPS/PSNR ablation, not adopted by default.

### Caveat: runtime drift

Section 4 of `STAGE2_ANALYSIS.md` documents that the current training run was launched with `local_detail_window_size=2`, while the working tree default is `1`. The live run is therefore producing ~4× fewer detail tokens than the design intends and is a plausible contributor to the stalled validation loss observed in section 2.1 of that report.

## 🎞️ The temporal pathway for video

The "temporal pathway" is not a single mechanism. It is the composition of **two independent components** that must agree, or the supervision becomes meaningless. They are easy to conflate because they live in different files and use overlapping vocabulary.

### (A) `local_detail_temporal_window_size` — temporal pooling inside cd_split

`content_detail_split.py:184` applies `t_win = local_detail_temporal_window_size` to the temporal axis (`positions[:, 0]`) using the same integer-division grouping as the spatial window.

| Value | Behaviour on an 8-frame video |
|---|---|
| `t_win=1` (LightningModule default) | Each frame produces its own detail tokens → up to `8 × (H/s × W/s)` detail tokens |
| `t_win=2` (legacy default in `mavt.py`) | Adjacent frame pairs are mean-pooled → 4 temporal slabs, motion attenuated |
| `t_win=T` (= number of frames) | Entire clip pooled to a single temporal slab → temporal information destroyed |

Choosing `t_win=1` keeps per-frame detail tokens intact, which is a **prerequisite** for any frame-difference supervision: if the detail branch has already averaged neighbouring frames, the reconstructed `pred[t+1] - pred[t]` is artificially smooth before the loss ever sees it.

**Hidden mismatch**: `mavt.py` declares the default as `2`, while `lightning_module.py` declares it as `1`. The LightningModule's argument propagates downward and overrides MAVT's default, so the effective runtime value is `1`. The mismatch is benign at runtime but extremely misleading when reading the code. Both files should be unified on `1`.

### (B) `temporal_consistency_loss` (`w_temp`) — motion supervisor

`src/mavt/losses/losses.py:104–113`:

```python
def temporal_consistency_loss(pred, target):
    pred_diff   = pred[:, :, 1:] - pred[:, :, :-1]     # frame-to-frame motion
    target_diff = target[:, :, 1:] - target[:, :, :-1]
    return F.l1_loss(pred_diff, target_diff)
```

This compares the **temporal gradient** of the reconstruction against the temporal gradient of the ground truth. The model is explicitly penalized for producing the wrong motion, not just the wrong per-frame appearance. This addresses the three classic failure modes of video autoencoders:

- **Flickering** — each frame looks fine in isolation but inter-frame consistency is poor.
- **Motion smearing** — the network minimizes per-frame L1 by outputting a temporal average, blunting moving edges.
- **Static fallback** — the network copies a previous frame as a low-cost shortcut.

Activation guard at `losses.py:245`:

```python
if self.w_temp > 0.0 and pred.ndim == 5 and pred.shape[2] > 1:
```

The loss is computed only when (1) the weight is positive, (2) the prediction is a 5-D tensor `(B, C, T, H, W)`, and (3) there are at least two frames.

### Runtime implication: `w_temp = 0` silently disables (B)

The current run launches with `w_temp = 0` despite `stage2_universal.yaml` declaring `0.1` and commit `c202c1e` adding the loss in the first place. The motion supervisor is therefore inactive across the steps already trained. Expected consequences:

- Per-frame PSNR may look healthy while playback exhibits flicker.
- `val/loss_video` reflects only L1 and LPIPS per frame, so motion artefacts are invisible to the validation metric.
- Re-enabling `w_temp` on resume injects an unoptimized term; expect a transient spike in total loss. Mitigate with a short linear warmup of `w_temp` from `0 → 0.1` over the first 1–2 k steps after resume, or start at `0.05`.

### How (A) and (B) interact

| Scenario | `t_win` | `w_temp` | Outcome |
|---|---|---|---|
| Intended target state | 1 | 0.1 | Per-frame detail preserved *and* motion is supervised → sharp, temporally consistent video |
| Current runtime | 1 | 0 | Detail intact but no motion signal → flicker risk, no validation visibility |
| Logically inconsistent | 2 | 0.1 | Detail tokens are temporally blurred before loss is computed → `pred_diff` is pre-smoothed; loss term wastes compute and gradient signal is degraded |
| Worst case | 2 | 0 | No per-frame detail *and* no motion supervisor → reconstructions degrade across both axes |

**Recommendation when resuming Stage 2.** Before re-enabling `w_temp`, confirm `local_detail_temporal_window_size = 1` in both `mavt.py` and `lightning_module.py`. Then launch with `--model.w_temp 0.1`; if total loss spikes hard, drop to `0.05` and warm `w_temp` linearly from `0` to its target across ~2000 steps. Only after this sequence is in place should the per-modality `val/loss_video` curve be trusted as a video-quality signal.

## ✅ Decision statement

Slot attention should remain the content branch mechanism because the content branch needs global semantic compression. The detail branch should not use the same global slot abstraction by default. Detail should be local, residual-centered, and position-aware, because reconstruction quality depends on assigning high-frequency information to the correct target region.

The 3D object issue should be handled as a parallel track. Triplane data does not expose true depth in the same way as voxel data, so RGAT's geometric prior must be made plane-aware or ablated for 3D before attributing all 3D artifacts to the content-detail split.
