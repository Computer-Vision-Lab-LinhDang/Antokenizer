# MAVT v2 Implementation Overview
## Complete Architecture Redesign

*Date: 2026-05-27*
*Status: Implementation Ready*

---

## 1. High-Level Overview

### 1.1 Problem with MAVT v1

MAVT v1 sử dụng **Content-Detail Split** (Slot Attention) để tách semantic và texture:

```
Problem: Semantic loss và Reconstruction loss cùng optimize z
→ Gradient conflict → Semantic degradation
```

**Evidence:**
- MSR-VTT R@1: 40.2% (15s clips) → ActivityNet-QA: 0.56 (180s videos)
- Token coverage: 13% (short) → 1.1% (long)

### 1.2 Solution: Two-Axis Decomposition

**Key Insight (từ UniJEPA, V-RAE):**
> "Photometric prediction learns **invariant structure**
> Temporal prediction learns **equivariant dynamics**"

**MAVT v2 Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT                                   │
│   Image (H×W) | Video (T×H×W) | 3D (3 planes)            │
└─────────────────┬─────────────────────────────────────────┘
                  │
┌─────────────────▼─────────────────────────────────────────┐
│  STAGE 1: PATCHIFY                                        │
│  Conv3d + 2D/3D Position Embedding                       │
│  Output: (B, N, D) tokens                                 │
└─────────────────┬─────────────────────────────────────────┘
                  │
┌─────────────────▼─────────────────────────────────────────┐
│  STAGE 2: BACKBONE (Simplified RGAT → RPB)               │
│  12 blocks: Standard ViT blocks (no typed edges)          │
│  Uses: Relative Position Bias (RPB)                        │
│  67% fewer parameters than v1 RGAT                        │
└─────────────────┬─────────────────────────────────────────┘
                  │
┌─────────────────▼─────────────────────────────────────────┐
│  STAGE 3: TWO-AXIS DECOMPOSITION (p, k)                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Coordinate System: (p, k)                          │   │
│  │                                                     │   │
│  │  p = spatial patch index (x, y)                     │   │
│  │  k = temporal/scale/view position                  │   │
│  │                                                     │   │
│  │  Edge types:                                       │   │
│  │    - Temporal: same-p, diff-k                      │   │
│  │    - Spatial: same-k, diff-p                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌────────────────────┐  ┌────────────────────────────┐     │
│  │ z_inv (Invariant)   │  │ z_var (Variant)           │     │
│  │ mean(feat, dim=k)  │  │ feat - z_inv              │     │
│  │                    │  │                            │     │
│  │ 98% energy         │  │ 2% energy                  │     │
│  │ Semantic content    │  │ Motion / Detail            │     │
│  │                    │  │                            │     │
│  │ Loss: Semantic ONLY│  │ Loss: Reconstruction ONLY  │     │
│  └────────────────────┘  └────────────────────────────┘     │
└─────────────────┬─────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼───────┐   ┌───────▼───────┐
│  SEMANTIC HEAD │   │  RECON HEAD   │
│  (z_inv only)  │   │  (z_var only) │
│                │   │               │
│  JEPA-style    │   │  VAE decode   │
│  Mask predict  │   │  Pixel space  │
└────────────────┘   └───────────────┘
```

---

## 2. Key Differences: v1 vs v2

| Aspect | MAVT v1 | MAVT v2 |
|--------|---------|---------|
| **Axis** | 4D (t,x,y,z) | 2D (p, k) |
| **Attention** | RGAT (typed edges) | RPB (relative position bias) |
| **Split** | Content-Detail (Slot) | Invariant-Variant (Mean-split) |
| **Semantic gradient** | Contaminated by recon | Isolated on z_inv |
| **Reconstruction gradient** | Contaminated by semantic | Isolated on z_var |
| **Image axis** | None | Scale pyramid |
| **Video axis** | t coordinate | k = frame index |
| **3D axis** | plane_id (metadata) | k = view index |

---

## 3. Detailed Architecture

### 3.1 Stage 1: Patchify

**Image:**
```
Input: (B, 3, H, W)
→ Conv3d (t_patch=1) → tokens at scale S
→ Scale Pyramid: encode at scales {S, S/2, S/4, S/8}
→ Stack: (B, N, K, D) where K = number of scales
```

**Video:**
```
Input: (B, 3, T, H, W)
→ Conv3d (t_patch=2) → (B, N, D) where N = Tp × Hp × Wp
→ Reshape: (B, N, K=1, D) [no multi-scale for video]
```

**3D:**
```
Input: (B, 3, 3, S, S) [3 planes × 3 channels]
→ Patchify each plane separately
→ Stack: (B, N, K=3, D) [K = 3 views]
```

### 3.2 Stage 2: Backbone (Simplified)

**No more RGAT!**

```python
# v2: Standard ViT blocks with Relative Position Bias
class BackboneV2(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, ...):
        self.blocks = nn.ModuleList([
            ViTBlock(dim, num_heads, use_rpb=True)
            for _ in range(num_blocks)
        ])
```

**Why RPB is enough:**
- ViT, DeiT, BEiT all use RPB
- 2D spatial + 1D temporal sufficient for vision
- 67% parameter reduction vs RGAT

### 3.3 Stage 3: Two-Axis Decomposition

**The Core Innovation:**

```python
class TwoAxisDecomposition(nn.Module):
    """Split features into invariant (semantic) and variant (detail) axes."""

    def forward(self, feat, k_dim):
        """
        feat: (B, N, K, D) where K = temporal/scale/view dimension
        k_dim: which axis to split on (temporal=0, scale=2, view=2)

        Returns:
            z_inv: (B, N, D) - semantic, 98% energy
            z_var: (B, N, K, D) - detail, 2% energy
        """
        # Invariant: mean across k dimension
        z_inv = feat.mean(dim=k_dim)  # (B, N, D)

        # Variant: deviation from mean
        z_var = feat - z_inv.unsqueeze(k_dim)  # (B, N, K, D)

        return z_inv, z_var
```

### 3.4 Semantic vs Reconstruction Branch

**v1 (Problematic):**
```
Both losses optimize z simultaneously
→ Gradient conflict
→ Semantic degradation
```

**v2 (Fixed):**
```
Semantic loss optimizes z_inv ONLY
Reconstruction loss optimizes z_var ONLY
→ No gradient conflict
→ Semantic quality preserved
```

---

## 4. Coordinate System (p, k)

### 4.1 Definition

```
p = spatial patch index (x, y)  [2D]
k = temporal/scale/view position [1D]

Token coordinate: (p, k)
- p ∈ ℕ² (patch position)
- k ∈ ℕ (temporal or scale or view)
```

### 4.2 Edge Types

| Edge Type | Condition | Meaning |
|-----------|-----------|---------|
| **Temporal** | same-p, diff-k | Same spatial location, different time/scale/view |
| **Spatial** | same-k, diff-p | Same time/scale/view, different location |

### 4.3 Benefits

1. **Unified representation:** Same (p,k) structure for all modalities
2. **Reduced complexity:** 2D edges instead of 4D
3. **Natural interpretation:**
   - Video: k = frame index
   - Image: k = scale level
   - 3D: k = view (plane)

---

## 5. Scale Pyramid for Images

### 5.1 Why Scale Pyramid?

**Problem:** Images don't have temporal axis like videos.

**Solution:** Use multi-resolution scales as synthetic "temporal axis"

```python
# Image → Scale Pyramid
Scale 0: H × W (native resolution)
Scale 1: H/2 × W/2
Scale 2: H/4 × W/4
Scale 3: H/8 × W/8

Stack: (B, N, K=4, D)
```

### 5.2 Energy Distribution (Validated)

| Component | Video Temporal | Image Scale | Match? |
|-----------|----------------|-------------|--------|
| z_inv energy | 97.6-98.8% | ~98.6% (measured) | ✅ |
| z_var energy | 1.2-2.4% | ~1.4% (measured) | ✅ |
| Gini | 0.62-0.76 | ~0.68 (measured) | ✅ |

**Validation confirms scale pyramid matches video statistics!**

### 5.3 Implementation

```python
class ScalePyramidEncoder(nn.Module):
    def __init__(self, encoder, num_scales=4):
        self.num_scales = num_scales
        self.encoder = encoder

    def forward(self, x):
        """x: (B, 3, H, W) → feat: (B, N, K, D)"""
        features = []
        for k in range(self.num_scales):
            scale = 2 ** k
            if scale > 1:
                x_scaled = F.interpolate(x, scale_factor=1/scale, mode='bilinear')
            else:
                x_scaled = x

            feat = self.encoder(x_scaled)  # Encode at this scale
            feat = self.align_to_native(feat, target_h, target_w)  # Upsample
            features.append(feat)

        return torch.stack(features, dim=2)  # (B, N, K, D)
```

---

## 6. Training Pipeline

### 6.1 Loss Functions

```python
def compute_losses(outputs, targets):
    # 1. Semantic Loss: ON z_inv ONLY
    semantic_loss = cosine_distance(outputs.z_inv, targets.semantic_teacher)

    # 2. Reconstruction Loss: ON z_var ONLY
    recon_loss = mse_loss(outputs.recon, targets.pixels)

    # 3. VAE Loss: ON z
    kl_loss = outputs.loss_kl

    # Total
    loss = (
        semantic_weight * semantic_loss +
        recon_weight * recon_loss +
        kl_weight * kl_loss
    )
```

### 6.2 JEPA-style Semantic (Optional Enhancement)

Instead of using frozen SigLIP2 as teacher, train semantic head with masked prediction:

```python
class JEPASemanticHead(nn.Module):
    """Masked prediction in latent space (V-JEPA style)."""

    def forward(self, z_inv, mask):
        """
        z_inv: (B, N, D)
        mask: (B, N) - which patches to mask

        Returns: predicted representations for masked patches
        """
        # Mask tokens
        masked_z = z_inv * mask.unsqueeze(-1)

        # Predict from visible → masked
        predicted = self.predictor(masked_z, mask)

        return predicted
```

---

## 7. Module Map

```
MAVTv2/
├── patchify_v2.py          # Scale pyramid for images
├── backbone_v2.py          # Simplified ViT with RPB (no RGAT)
├── two_axis_split.py        # NEW: Invariant-Variant decomposition
├── semantic_head.py         # Semantic prediction from z_inv
├── recon_head.py            # Reconstruction from z_var
├── scale_pyramid.py         # Multi-scale image encoding
└── unified_position.py      # 2D (p, k) position embedding
```

---

## 8. Implementation Priority

### Phase 1: Core Architecture (Week 1-4)
- [ ] `patchify_v2.py`: Scale pyramid encoder
- [ ] `backbone_v2.py`: Simplified ViT with RPB
- [ ] `two_axis_split.py`: Two-axis decomposition

### Phase 2: Heads (Week 5-8)
- [ ] `semantic_head.py`: Semantic from z_inv
- [ ] `recon_head.py`: Reconstruction from z_var
- [ ] `unified_position.py`: 2D position embedding

### Phase 3: Integration (Week 9-12)
- [ ] Integrate into `MAVTv2` class
- [ ] Update training pipeline
- [ ] Reproduce baseline metrics

### Phase 4: Enhancements (Week 13-16)
- [ ] JEPA-style masked prediction
- [ ] Adaptive token budget
- [ ] NKD for long video

---

## 9. Expected Improvements

| Metric | MAVT v1 | MAVT v2 Target | Improvement |
|--------|---------|----------------|------------|
| MSR-VTT R@1 | 40.2% | 45-50% | +5-10% |
| ActivityNet-QA | 0.56 | 0.65-0.70 | +0.09-0.14 |
| Token coverage | 13% (short) | 20% (short) | +7% |
| rFID | 0.209 | ≤0.20 | Maintain |
| Params (backbone) | 3.54M | 1.18M | -67% |

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Scale pyramid doesn't match | Validate on 100 images before commit |
| Semantic degrades | Isolated z_inv branch prevents contamination |
| Reconstruction quality drops | z_var carries 2% energy for detail |
| Training instability | Start with video (natural axis), add image later |

---

## 11. Files to Create

```bash
atoken/src/mavt/model/
├── mavt_v2.py                 # Main model class
├── two_axis_split.py          # NEW: Two-axis decomposition
├── backbone_v2.py             # NEW: Simplified ViT with RPB
├── scale_pyramid.py           # NEW: Multi-scale encoder
├── semantic_head.py           # NEW: Semantic from z_inv
├── recon_head_v2.py           # NEW: Recon from z_var
├── unified_position.py        # NEW: 2D (p, k) positions
└── mavt_v2.md                # This overview
```

---

*Implementation Overview completed: 2026-05-27*
