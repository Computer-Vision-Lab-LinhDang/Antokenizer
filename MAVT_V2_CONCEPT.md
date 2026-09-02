# MAVT v2: Two-Axis Unified Tokenizer
## Core Philosophy Redesign

*Date: 2026-05-27*
*Status: Concept Design (pre-implementation)*

---

## 0. Executive Summary

**Problem:** Current MAVT suffers from a fundamental architectural conflict — a single latent space must simultaneously satisfy semantic objectives (teacher alignment) and reconstruction objectives (pixel fidelity). This gradient conflict manifests as semantic degradation through stages (82.7% → 82.2% ImageNet zero-shot) and forces trade-offs that hurt both goals.

**Solution:** Decompose the latent space into **two orthogonal axes** based on invariance properties:

| Axis | Invariant (z_inv) | Variant (z_var) |
|------|-------------------|-----------------|
| **What it captures** | Content that persists across the "temporal-like" axis | What changes along that axis |
| **Objective** | Semantic → distillation loss | Reconstruction + motion understanding |
| **Natural axis** | Time (video), Scale (image), View (3D) | Deviation from mean |

**Key insight:** The tax problem disappears when losses target different latent branches — no gradient conflict because semantic loss and recon loss operate on different parameters.

---

## 1. Why Two Axes Instead of 4D

### 1.1 The Tax Problem, Dissected

Current MAVT:
```
z = [content_tokens; detail_tokens]  # Same objective space
L_total = L_semantic(z) + L_recon(z)  # Gradient conflict!
```

The gradient from `L_semantic` pulls `z` toward the teacher's representation.
The gradient from `L_recon` pulls `z` toward pixel-space structure.

When one wins, the other loses. Empirically, recon wins (rFID improves) and semantic degrades.

### 1.2 The New Philosophy

```
z_inv = Pool_k(feat)           # Semantic: mean across temporal axis
z_var = feat - Broadcast(z_inv) # Temporal: deviation from mean

L_semantic → z_inv ONLY         # No conflict
L_recon    → z_var + residual   # Separate objective space
```

**Now the gradient conflict is structurally impossible** — semantic and recon losses target different parameter subsets, not the same numbers.

### 1.3 Why Not Keep Content-Detail Split?

Current C-D split has a self-cancellation problem:

```
At window=1 (default):
  grouped[:, i] = positions[:, i] // 1 = positions[:, i]
  → Each token is its own group
  → Mean-pool over 1 element = identity
  → Detail branch = Affine(x) with cosine 0.998 to input
```

The "detail" branch doesn't split anything — it's just an affine copy of the input. The split only works if `window > 1`, but that creates its own problems (residual reconfounds what it was designed to avoid).

**Two-axis approach defines splitting structurally:**
- `z_inv` is *mathematically* invariant by construction (mean across axis)
- `z_var` is *mathematically* variant by construction (deviation from mean)
- No pooling parameters that can degenerate

---

## 2. The Two Axes in Detail

### 2.1 Invariant Axis (z_inv)

**What:** Content that persists across the temporal-like dimension.

**How to compute:**
```python
z_inv = mean(feat, dim=k)  # k = temporal axis
```

**Properties:**
- Naturally 98% of video energy (measured in audit6)
- Low frequency, semantic
- Semantic loss should attach HERE

**For each modality:**

| Modality | "Temporal" axis | What z_inv captures |
|----------|----------------|---------------------|
| Video (T frames) | Time (k=T) | Scene structure, objects, semantics |
| Image (K scales) | Scale pyramid (k=K) | Shape, composition |
| 3D (V views) | View axis (k=V) | 3D shape, structure |

### 2.2 Variant Axis (z_var)

**What:** Deviation from the invariant — motion, detail, scale-dependent appearance.

**How to compute:**
```python
z_var = feat - broadcast(z_inv)  # Broadcast along k axis
```

**Properties:**
- Only 2% of video energy (measured)
- BUT: 54% of its energy is LOW frequency (not high-freq texture!)
- Contains motion (frame-to-frame differences) and multi-scale detail

**Important correction from audit6:**
> "Temporal ≠ high frequency. 54% of temporal energy is LOW frequency — moving structures."

This means:
- `z_var` captures both motion (low-freq, structural) AND texture (high-freq)
- Both belong to the "variant" axis
- We don't need a separate "detail" branch

### 2.3 The Residual Question

**Do we still need a third branch?**

Current thinking: The two-axis split is sufficient:
- `z_inv` → semantic
- `z_var` → recon + motion

But empirically from audit6:
- `z_var` has 54% low-freq (motion) + 5% high-freq (texture)
- Current "detail" concept wanted high-freq texture

**If high-freq texture matters for rFID/LPIPS**, we might still need:
- `z_inv` → semantic
- `z_var` → low-freq motion
- `z_detail` → high-freq texture

However, this is 3 branches. The claim of the redesign is that 2 is enough if we:
1. Let recon loss operate on the FULL `z_var` (not just a fraction)
2. Let the compression ratio handle the detail implicitly

**Decision: Start with 2 branches, profile, then decide on 3.**

---

## 3. Handling All Three Modalities

### 3.1 Video

**Natural fit — axis IS time.**

```
Input: (B, 3, T, H, W)  # T frames
Patchify → feat: (B, N, T, D)  # N patches × T frames
```

**Two-axis extraction:**
```python
z_inv  = mean(feat, dim=t)      # (B, N, D) — temporal mean
z_var  = feat - z_inv[:,:,None,:]  # (B, N, T, D) — deviations
```

**Compression:**
- `z_inv`: 1 frame worth of tokens → aggressive compression OK
- `z_var`: T frames worth → also compress, but different compression

**Token budget:**
- Budget for `z_inv`: determined by entropy of invariant signal
- Budget for `z_var`: determined by entropy of variant signal

### 3.2 Image

**Need a synthetic "temporal" axis.**

Scale pyramid is the closest analog:

```
Input: (B, 3, H, W)
Scale levels: [1x, 2x, 4x, 8x]  # K=4 levels
Patchify at each scale → concat → (B, N, K, D)
```

**Two-axis extraction:**
```python
z_inv  = mean(feat, dim=k)      # Scale-invariant structure
z_var  = feat - z_inv[:,:,None,:]  # Scale-dependent detail
```

**Why scale pyramid? (Verified in audit6)**
- Energy in variant branch: 1.4% (vs 2.4% for video's temporal)
- Gini: 0.59 (vs 0.76 for video)
- Closest match to video's statistics among all candidates

**Alternative considered but rejected:**
- Frequency bands: 66.7% energy in variant (28× mismatch)
- Spatial neighbors: 43.9% energy in variant (18× mismatch)

### 3.3 3D Object

**Axis = view (triplane projections)**

```
Input: Objaverse mesh/voxel
Project to XY, XZ, YZ planes
Voxelize → feat: (B, N, V, D)  # V=3 views
```

**Two-axis extraction:**
```python
z_inv  = mean(feat, dim=v)      # 3D shape structure
z_var  = feat - z_inv[:,:,None,:]  # View-dependent appearance
```

**Note:** This is cleaner than current triplane approach because:
- Current: 3 planes processed independently, then concatenated
- New: Views are samples from the same 3D space, so mean IS the 3D structure

### 3.4 Unified Encoding

All three modalities now share the same coordinate system:

```
positions = (p, k)
p = patch index within sample
k = position on temporal-like axis

Video:  p ∈ [0, N-1],  k ∈ [0, T-1]
Image:  p ∈ [0, N-1],  k ∈ [0, K-1]  # K=scale levels
3D:     p ∈ [0, N-1],  k ∈ [0, V-1]  # V=views
```

This is the key unification — not the architecture, but the **coordinate system**.

---

## 4. RGAT Redesign: From 4D to 2D

### 4.1 Current RGAT Problems

Current RGAT uses 4D coordinates `(t, x, y, z)` with 4 edge types:

| Edge Type | Meaning | Usage |
|-----------|---------|-------|
| Type 0 | Self (Manhattan=0) | Always |
| Type 1 | Face-adjacent | Sometimes |
| Type 2 | Depth-adjacent | **NEVER** (zeros) |
| Type 3 | Cross-plane | 64% fake (zero-padding) |

**Cost:**
- 3.54M extra parameters
- 23% of backbone
- 589× more expensive than relative position bias with same coverage

### 4.2 New RGAT: Two Edge Types

New coordinate: `(p, k)` — patch index + temporal position

```
Edge Type 0: Same p, different k  (temporal edge)
Edge Type 1: Same k, different p  (spatial edge)
```

**Why this works:**
- Video: "same spatial position, different time" = motion
- Image: "same spatial position, different scale" = scale-dependent detail
- 3D: "same spatial position, different view" = view-dependent appearance

**All modalities use the SAME edge types with the SAME meaning.**

### 4.3 Parameter Comparison

| Component | Old (4D RGAT) | New (2D RGAT) | Savings |
|-----------|--------------|---------------|---------|
| Edge type embeddings | 4 × proj_dim | 2 × proj_dim | 50% |
| k_proj (depth) | 3.54M (unused) | 0 | 100% |
| Total RGAT overhead | 3.54M | ~1.18M | **67%** |

### 4.4 The Remaining Question: Is RGAT Still Needed?

Even with 2D, RGAT is 196× more expensive than relative position bias (1.18M vs 6K).

**Case FOR RGAT:**
- Different edge types get different K/V projections
- Can learn transformations specific to edge type
- More expressive than additive bias

**Case FOR relative position bias:**
- 196× cheaper
- Proven effective in modern transformers (ViT, Llama)

**Decision framework:**
> RGAT is justified if and only if the edge types carry REAL information.
> In 3D, they do (depth edge type is real for voxels).
> In 2D (image/video), it's less clear.

**Recommendation:** Start with 2D relative position bias (cheap). If 3D voxel branch needs RGAT, add it only for that branch.

---

## 5. Adaptive Token Budget

### 5.1 The Problem with Fixed Ratios

Current approach: `content_ratio = 0.25` fixed for all samples.

Measured reality (from audit6):

| Clip | Energy in variant | Tokens needed for 90% | Fixed budget (64) |
|------|------------------|---------------------|-------------------|
| 00013001 | 2.4% | 78 | Too few (miss 14 tokens) |
| 00013006 | 1.2% | 114 | Too many (waste 50 tokens) |

**One fixed ratio cannot fit both.**

### 5.2 Entropy-Based Budget

**Signal:** Entropy of `z_var` distribution across patches

```python
def compute_budget(z_var, base_budget=64):
    # z_var shape: (B, N, K, D)
    # Compute variance/entropy per patch
    variance = z_var.var(dim=[k, d])  # (B, N)
    entropy = -torch.log(variance + eps)  # Higher variance = higher entropy

    # Budget proportional to entropy
    total_entropy = entropy.sum(dim=1)  # (B,)
    budget = base_budget * (entropy / total_entropy)[:, None]  # (B, N)

    return budget.clamp(min=8, max=128)  # Sanity bounds
```

**Measured properties (from audit6):**
- Gini coefficient: 0.62–0.76 (highly skewed — good for budget allocation)
- Spearman correlation: 0.93–0.95 between halves of same clip (stable!)

### 5.3 Implementation Considerations

**Variable token count breaks batching:**
- Need padding or packing
- Need attention mask updates

**Breaking the cache:**
- Current: `adjacency_cache[(modality, N)]`
- Problem: Different `N` per sample when budget varies
- Fix: Key cache by `hash(positions)` instead of `(modality, N)`

This is a prerequisite change — must fix before adaptive budget.

---

## 6. Loss Function Redesign

### 6.1 New Loss Structure

```
L_total = λ_semantic * L_semantic(z_inv)
         + λ_recon * L_recon(z_var + z_detail)
         + λ_kl * L_kl(z_inv, z_var)
         + λ_diversity * L_diversity(z_inv)
```

**Key differences from current:**
- Semantic loss ONLY on `z_inv` (not on full latent)
- Reconstruction loss on variant branch (not splitting between branches)
- No conflict because losses target different latent spaces

### 6.2 Why This Reduces the Tax

**Current:**
```
Gradient L_semantic → entire z → conflicts with → Gradient L_recon → entire z
```

**New:**
```
Gradient L_semantic → z_inv parameters only
Gradient L_recon    → z_var parameters only
Backbone is shared, but gradient targets are disjoint
```

The backbone learns a joint representation that satisfies BOTH objectives through their respective branches — that's the NATURAL function of a shared encoder. No tax because there's no conflict.

---

## 7. Architectural Summary

```
INPUT
  ↓
PATCHIFY
  positions = (p, k)  # Unified coordinate for all modalities
  p = patch index
  k = temporal-like position (time/scale/view)
  ↓
SHARED BACKBONE (Transformer blocks)
  2D relative position bias OR 2-edge-type RGAT
  Edge 0: same p, different k (temporal)
  Edge 1: same k, different p (spatial)
  ↓
TWO-AXIS SPLIT
  z_inv  = mean(feat, dim=k)      # Invariant branch
  z_var  = feat - z_inv          # Variant branch
  ↓
COMPRESSION
  Adaptive budget based on entropy
  z_inv → semantic latent
  z_var → temporal latent
  ↓
DECODERS
  Recon decoder ← z_var
  Semantic head  ← z_inv
  ↓
LOSSES
  L_semantic → z_inv ONLY
  L_recon    → z_var
```

---

## 8. Modality-Specific Instantiations

### 8.1 Video Pipeline

```
Input: (B, 3, T, H, W)
  ↓ Patchify
feat: (B, N, T, D)  where T = num_frames
  ↓ Two-axis
z_inv:  (B, N, D)      # Temporal mean
z_var:  (B, N, T, D)   # Frame deviations
  ↓ Compress
z_inv → semantic tokens
z_var → temporal tokens
  ↓ Decode + Loss
L_semantic(z_inv) + L_recon(z_var)
```

### 8.2 Image Pipeline

```
Input: (B, 3, H, W)
  ↓ Multi-scale patchify (K=4 levels)
feat: (B, N, K, D)  where K = scale levels
  ↓ Two-axis
z_inv:  (B, N, D)      # Scale-invariant structure
z_var:  (B, N, K, D)   # Scale deviations
  ↓ Compress
z_inv → semantic tokens
z_var → temporal tokens
  ↓ Decode + Loss
L_semantic(z_inv) + L_recon(z_var)
```

### 8.3 3D Pipeline

```
Input: 3D mesh/voxel
  ↓ Voxelize + project
feat: (B, N, V, D)  where V = views (3 planes)
  ↓ Two-axis
z_inv:  (B, N, D)      # View-invariant shape
z_var:  (B, N, V, D)   # View-dependent appearance
  ↓ Compress
z_inv → semantic tokens
z_var → temporal tokens
  ↓ Decode + Loss
L_semantic(z_inv) + L_recon(z_var)
```

---

## 9. Open Questions & Validation Needed

### 9.1 Must Validate Before Commit

| Question | Method | Acceptance Criteria |
|----------|--------|---------------------|
| Scale pyramid for images | Measure on ≥100 images | Energy in variant ≤ 5%, Gini ≥ 0.5 |
| Entropy-based budget | Simulate on video clips | Budget predicts true information |
| 2-edge RGAT vs RPB | Ablation on video | RPB within 1% of RGAT |

### 9.2 Design Decisions to Make

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Number of branches | 2 vs 3 | Start with 2, profile, add if needed |
| RGAT vs RPB | Full RGAT, RPB only, hybrid | RPB first, add RGAT only for 3D |
| Scale levels for images | 3, 4, or 5 | K=4 (tested in audit6) |

### 9.3 Prerequisite Fixes

These MUST be done before implementing the new architecture:

1. **Cache adjacency key:** Change from `(modality, N)` to `hash(positions)`
2. **slot_diversity_loss:** Remove from `no_grad()` context
3. **Evaluation noise:** Use `μ` instead of sampling for eval metrics

---

## 10. Comparison: Current vs Redesigned

| Aspect | Current MAVT | Redesigned MAVT-v2 |
|--------|--------------|-------------------|
| **Coordinate system** | 4D (t, x, y, z) | 2D (p, k) |
| **Edge types** | 4 (1 unused) | 2 (all used) |
| **RGAT overhead** | 3.54M params | ~1.18M or RPB |
| **Latent branches** | 2 (content + detail) | 2 (invariant + variant) |
| **Split mechanism** | Slot attention + pooling | Mean + deviation |
| **Tax problem** | Present (gradient conflict) | Absent (separate objectives) |
| **Image axis** | None (2D only) | Scale pyramid |
| **3D bias** | Fake (zero-padding) | Real (view mean = shape) |
| **Token budget** | Fixed ratio | Entropy-adaptive |

---

## 11. Implementation Roadmap

### Phase 1: Validate (Weeks 1-2)
- [ ] Measure scale pyramid on ≥100 images
- [ ] Verify entropy-stability on more video clips
- [ ] Ablation: 2-edge RGAT vs RPB

### Phase 2: Prerequisite Fixes (Week 2)
- [ ] Fix cache adjacency key
- [ ] Fix slot_diversity_loss no_grad
- [ ] Fix eval sampling noise

### Phase 3: Core Implementation (Weeks 3-6)
- [ ] Implement two-axis split (video first)
- [ ] Implement scale pyramid for images
- [ ] Implement view-axis for 3D
- [ ] Implement adaptive token budget

### Phase 4: Training & Eval (Weeks 7-10)
- [ ] Train video baseline
- [ ] Train image + 3D
- [ ] Measure rFID, linear probe
- [ ] Compare vs current MAVT

---

*Next: Detailed implementation specification*
