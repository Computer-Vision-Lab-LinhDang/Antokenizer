# Scale Pyramid Analysis for Image Tokenizer
## MAVT v2 Component Deep Dive

*Date: 2026-05-27*
*Status: Needs Validation*

---

## 1. Problem Statement

MAVT v2 sử dụng **scale pyramid** như synthetic "temporal axis" cho images. Cần phân tích:

1. Scale pyramid là gì, tại sao nó phù hợp?
2. So sánh với các alternatives
3. Validation criteria
4. Implementation considerations

---

## 2. Background: Scale Pyramid

### 2.1 Traditional Scale Pyramid (Laplacian Pyramid)

```python
# Gaussian Pyramid
G_0 = image  # Level 0
G_1 = downsample(G_0)        # 2x smaller
G_2 = downsample(G_1)        # 4x smaller
G_3 = downsample(G_2)        # 8x smaller

# Laplacian Pyramid
L_0 = G_0 - upsample(G_1)   # Detail at scale 0
L_1 = G_1 - upsample(G_2)   # Detail at scale 1
L_2 = G_2 - upsample(G_3)   # Detail at scale 2
L_3 = G_3                    # Coarsest (residual)
```

**Properties:**
- Mỗi level chứa frequency content khác nhau
- Coarser levels = lower frequency
- Finer levels = higher frequency
- Energy thường tập trung ở coarse levels

### 2.2 Scale Pyramid vs Temporal Axis (Video)

| Property | Video Temporal | Image Scale Pyramid |
|----------|----------------|---------------------|
| Axis meaning | Time (causal) | Scale (spatial frequency) |
| Energy distribution | 98% in mean (temporal mean) | **Need measurement** |
| Variant component | Motion (frame differences) | **Scale-dependent detail** |
| Semantic axis | Scene structure (mean) | **Should be scale-invariant** |

**Key Question:** Does "mean across scales" capture semantic content?

---

## 3. Literature Review

### 3.1 Multi-Scale in Vision (2024-2026)

| Paper | Approach | Relevance |
|-------|---------|-----------|
| **UDT** (2026) | Token merging in DiT | Multi-scale token reduction |
| **PyramidMamba** (2024) | Dense spatial pyramid pooling | Multi-scale fusion |
| **FlowAR** (2024) | Scale-wise AR generation | Simple doubling scales |
| **MSDS** (2026) | Multi-scale DeepSSIM | Scale importance in similarity |
| **VAR/VQ-VAE** | Multi-scale tokenization | Scale decomposition |

### 3.2 VAR-style Multi-Scale Tokenization

**Key Paper:** Visual Autoregressive Modeling (VAR)

```python
# VAR's scale hierarchy
scale_0: 1x1 tokens (1 token)
scale_1: 2x2 tokens (4 tokens)
scale_2: 4x4 tokens (16 tokens)
scale_3: 8x8 tokens (64 tokens)
...
```

**Insight:** VAR đã dùng multi-scale decomposition cho tokenization.
- Coarse scales chứa structure (semantic)
- Fine scales chứa detail (texture)

**But:** VAR không dùng "mean across scales" — nó dùng hierarchical prediction.

### 3.3 Key Finding from MSDS (2026)

> "Deep-feature similarity at different scales provides complementary information, and multi-scale fusion improves perceptual quality."

**Implication:** Different scales DO carry different information — validates our axis hypothesis.

---

## 4. Hypothesis Analysis

### 4.1 Main Hypothesis

```
z_inv (scale-invariant) = mean(features across scales)
z_var (scale-variant)    = features - mean(features)
```

**Expected properties:**
- `z_inv`: Scale-invariant structure, semantic content
- `z_var`: Scale-dependent detail, texture

### 4.2 Predicted Statistics

Based on video measurements (audit6):

| Statistic | Video Temporal | Predicted for Image Scale |
|-----------|----------------|---------------------------|
| Energy in z_inv | 97.6-98.8% | **~95-98%** |
| Energy in z_var | 1.2-2.4% | **~2-5%** |
| Gini coefficient | 0.62-0.76 | **~0.5-0.7** |

**Uncertainty:** Image statistics could differ significantly from video.

---

## 5. Comparison with Alternatives

### 5.1 Alternatives Considered (from audit6)

| Alternative | Energy in Variant | Match to Video |
|-------------|------------------|----------------|
| Frequency bands | 66.7% | ❌ 28× mismatch |
| Spatial neighbors | 43.9% | ❌ 18× mismatch |
| **Scale pyramid** | ~1.4% (measured) | ✅ Closest |

### 5.2 Why Scale Pyramid Wins

1. **Structural analogy:** Both decompose by "resolution"
2. **Measured match:** ~1.4% variant energy vs video's ~2.4%
3. **Multi-scale is standard:** FPN, U-Net, etc. all use this
4. **Interpretable:** Scales = different frequency bands spatially

### 5.3 Alternative: No Axis (Single Image)

**Option:** Don't use any "temporal axis" for images.

**Pros:**
- Simpler architecture
- No artificial decomposition

**Cons:**
- Inconsistent with video/3D pipeline
- Can't share two-axis design across modalities

**Verdict:** Scale pyramid is better for unified design.

---

## 6. Validation Plan

### 6.1 Measurement Protocol

```python
def measure_scale_pyramid(image, K=4):
    """
    K = number of scale levels
    Returns statistics for validating scale pyramid approach
    """
    # Build feature pyramid
    features = []
    for k in range(K):
        scale = 2 ** k  # 1x, 2x, 4x, 8x
        feat_k = encode_at_scale(image, scale)
        features.append(feat_k)

    # Stack: (B, N, K, D)
    features = torch.stack(features, dim=2)

    # Compute invariant = mean across scales
    z_inv = features.mean(dim=2)  # (B, N, D)

    # Compute variant = deviation from invariant
    z_var = features - z_inv.unsqueeze(2)  # (B, N, K, D)

    # Measure statistics
    energy_inv = (z_inv ** 2).sum() / (features ** 2).sum()
    energy_var = (z_var ** 2).sum() / (features ** 2).sum()

    # Per-patch variance (for Gini)
    var_per_patch = z_var.var(dim=[2, 3])  # (B, N)
    gini = compute_gini(var_per_patch)

    return {
        'energy_in_invariant': energy_inv.item(),
        'energy_in_variant': energy_var.item(),
        'gini_coefficient': gini,
        'per_scale_energy': [(f**2).sum() / (features**2).sum()).item()
                            for f in features]
    }
```

### 6.2 Validation Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Energy in z_inv | ≥90% | Most information preserved |
| Energy in z_var | ≤10% | Small fraction for detail |
| Gini coefficient | 0.4-0.8 | Stable allocation |
| Per-scale energy | Decreasing | More energy in coarse scales |

### 6.3 Dataset Requirements

- **Minimum:** 100 diverse images (ImageNet val subset)
- **Recommended:** 1000 images across categories
- **Diversity:** Natural scenes, objects, textures, synthetic

---

## 7. Design Decisions

### 7.1 Number of Scale Levels (K)

| K | Spatial Resolution | Total Tokens | Trade-off |
|---|-------------------|--------------|-----------|
| 2 | 1x, 2x | N + N/4 | Too coarse |
| 3 | 1x, 2x, 4x | N + N/4 + N/16 | Minimal |
| **4** | 1x, 2x, 4x, 8x | N + N/4 + N/16 + N/64 | **Recommended** |
| 5 | 1x-16x | N + ... + N/256 | Too fine |

**Current decision:** K=4 (from audit6 measurement)

**TODO:** Ablate K=3,4,5

### 7.2 Scale Factor

**Option A:** Doubling (1x, 2x, 4x, 8x) — standard pyramid
**Option B:** Fixed patch size (16×16 at all scales)

**Decision:** Doubling is standard for pyramids, but may need patch-equalized.

```python
# Option A: Doubling resolution
feat_k = encoder(resize(image, scale=2**k))

# Option B: Equal patch count
feat_k = encoder(image)  # All at native, but pool differently
```

### 7.3 Patch Alignment

**Problem:** Different scales → different patch counts

```python
# Scale 1: (B, N, D) — 256 patches
# Scale 2: (B, N/4, D) — 64 patches
# Scale 3: (B, N/16, D) — 16 patches

# Need to align for two-axis split
```

**Solutions:**

1. **Interpolate to common resolution:**
```python
z_var_scaled = F.interpolate(
    z_var.permute(0,3,1,2),  # (B, D, N/4, K)
    size=N, mode='bilinear'
).permute(0,2,3,1)
```

2. **Upsample coarse scales:**
```python
z_var_upsampled = F.interpolate(
    z_var, size=feat_0.shape[1:3], mode='bilinear'
)
```

3. **Use position-aware pooling:**
```python
# Pool fine patches to match coarse
z_var_pooled = adaptive_avg_pool2d(z_var, output_size=(H_coarse, W_coarse))
```

**Decision:** Option 3 (upsample coarse to match fine) — preserves fine detail.

---

## 8. Potential Issues

### 8.1 Semantic Content in `z_inv`

**Concern:** Does "mean across scales" capture semantic content?

**Analysis:**
- Coarse scales: Global structure, shapes, objects
- Fine scales: Local textures, edges, details
- Mean across scales: Weighted toward coarse (more area)

**Expected:** `z_inv` should capture object-level semantics.

**Risk:** If scales have conflicting semantics (e.g., texture vs shape), mean could blur.

**Mitigation:** Measure semantic quality of `z_inv` via linear probe on ImageNet.

### 8.2 Information Loss in `z_var`

**Concern:** `z_var` might lose critical detail needed for reconstruction.

**Analysis:**
- Images: ~5% energy in variant (our hypothesis)
- Video: ~2% energy in variant

**Risk:** If rFID degrades significantly, may need separate detail branch.

**Mitigation:** Compare:
- Full features → decoder
- z_inv only → decoder
- z_var only → decoder
- z_inv + partial z_var → decoder

### 8.3 Scale Selection Bias

**Concern:** Which scales to include?

**Analysis:**
- Too few scales: May not capture sufficient variance
- Too many scales: Diminishing returns, increased complexity

**Current:** K=4 (1x, 2x, 4x, 8x)

**TODO:** Profile per-scale contribution to z_var energy.

---

## 9. Implementation Sketch

### 9.1 Encoder with Scale Pyramid

```python
class ScalePyramidEncoder(nn.Module):
    def __init__(self, encoder, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.encoder = encoder

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, N, K, D) feature tensor
        """
        features = []
        for k in range(self.num_scales):
            scale = 2 ** k
            if scale > 1:
                x_scaled = F.interpolate(x, scale_factor=1/scale, mode='bilinear')
            else:
                x_scaled = x

            feat = self.encoder(x_scaled)  # (B, D, H', W')
            feat = feat.flatten(2).transpose(1, 2)  # (B, N', D)

            # Upsample to common resolution
            feat = F.interpolate(
                feat.transpose(1, 2).reshape(B, D, H', W'),
                size=(x.shape[2]//16, x.shape[3]//16),
                mode='bilinear'
            ).flatten(2).transpose(1, 2)  # (B, N, D)

            features.append(feat)

        return torch.stack(features, dim=2)  # (B, N, K, D)
```

### 9.2 Two-Axis Split

```python
class TwoAxisSplit(nn.Module):
    def forward(self, feat):
        """
        feat: (B, N, K, D)
        Returns: z_inv, z_var
        """
        # Invariant: mean across scales
        z_inv = feat.mean(dim=2)  # (B, N, D)

        # Variant: deviation from invariant
        z_var = feat - z_inv.unsqueeze(2)  # (B, N, K, D)

        return z_inv, z_var
```

### 9.3 Compressed Token Budget

```python
def compute_token_budget(z_var, base_budget=64):
    """
    z_var: (B, N, K, D)
    Returns: per-patch token budget
    """
    # Compute variance per patch
    variance = z_var.var(dim=[2, 3])  # (B, N)

    # Normalize to budget
    total_var = variance.sum(dim=1, keepdim=True)  # (B, 1)
    budget = (variance / total_var * base_budget).long()  # (B, N)

    return budget.clamp(min=8, max=128)
```

---

## 10. Validation Checklist

### 10.1 Before Implementation

- [ ] Measure scale pyramid statistics on ≥100 images
- [ ] Confirm energy in z_inv ≥90%
- [ ] Confirm Gini coefficient 0.4-0.8
- [ ] Profile per-scale energy contribution

### 10.2 During Training

- [ ] Monitor z_inv semantic quality (linear probe)
- [ ] Monitor z_var reconstruction quality (rFID)
- [ ] Compare with single-branch baseline

### 10.3 Ablation Studies

- [ ] K=3 vs K=4 vs K=5
- [ ] Scale pyramid vs spatial neighbors vs frequency bands
- [ ] z_inv only vs z_var only vs both

---

## 11. Conclusion

### 11.1 Assessment

**Scale Pyramid for Images:** ⚠️ **PROMISING but UNVALIDATED**

**Strengths:**
- Structural analogy with video temporal axis
- Multi-scale is well-studied in vision
- Preliminary measurement shows ~1.4% variant energy
- Matches video statistics better than alternatives

**Weaknesses:**
- No direct precedent for this specific use case
- Semantic content in mean-across-scales is uncertain
- Scale alignment adds complexity

### 11.2 Recommendations

1. **VALIDATE FIRST:** Measure on ≥100 images before committing

2. **START SIMPLE:** Use K=4, standard doubling scales

3. **MONITOR SEMANTIC:** Check z_inv quality via linear probe

4. **BE READY TO PIVOT:** If scale pyramid doesn't match video statistics:
   - Try frequency bands (but expect 28× mismatch)
   - Try spatial neighbors (but expect 18× mismatch)
   - Consider single-branch for images (inconsistent but simpler)

### 11.3 Next Steps

1. Run scale pyramid measurement on 100+ diverse images
2. Compare statistics with video temporal axis
3. If match: Proceed with implementation
4. If mismatch: Investigate cause, adjust K or method

---

## References

1. **Laplacian Pyramid** — Burt & Adelson (1983)
2. **VAR** — Visual Autoregressive Modeling (2024)
3. **MSDS** — Multi-scale DeepSSIM (2026)
4. **FlowAR** — Scale-wise AR + Flow Matching (2024)
5. **PyramidMamba** — Multi-scale for segmentation (2024)

---

*Analysis completed: 2026-05-27*
*Validation required before implementation*
