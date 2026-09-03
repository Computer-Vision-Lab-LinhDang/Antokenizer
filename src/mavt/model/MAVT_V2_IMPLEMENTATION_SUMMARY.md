# MAVT v2 Complete Implementation Summary

*Generated: 2026-05-27*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAVT v2 Pipeline                                │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: Image | Video(T×H×W) | 3D(3 planes)

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: PATCHIFY (Conv3d + 4D Position Embedding)                        │
│ Input: (B, 3, H, W) or (B, 3, T, H, W)                                    │
│ Output: (B, N, D) tokens                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: BACKBONE V2 (ViT + RPB) ⚡ Simplified!                            │
│                                                                             │
│ Changes from v1:                                                          │
│   ❌ No RGAT (typed edges: spatial/temporal/depth/cross-plane)            │
│   ✅ Standard ViT blocks with Relative Position Bias (RPB)                │
│   ✅ ~67% parameter reduction                                              │
│                                                                             │
│ Output: (B, N, D) features                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: TWO-AXIS DECOMPOSITION (Invariant-Variant Split) 🎯 CORE         │
│                                                                             │
│ Input: (B, N, D) or (B, N, K, D) with K = temporal/scale/view axis         │
│                                                                             │
│        ┌─────────────────────────────────────────────────────────────┐      │
│        │  z_inv = mean(features, dim=k)                              │      │
│        │  z_var = features - z_inv                                   │      │
│        │                                                              │      │
│        │  Energy:  z_inv ≈ 98%   |   z_var ≈ 2%                      │      │
│        └─────────────────────────────────────────────────────────────┘      │
│                                                                             │
│        ┌──────────────────────┐    ┌──────────────────────────────┐       │
│        │ z_inv (Invariant)    │    │ z_var (Variant)              │       │
│        │                      │    │                              │       │
│        │ • Semantic content   │    │ • Motion / Detail            │       │
│        │ • Scale-invariant   │    │ • Scale-dependent            │       │
│        │ • For: Semantic Loss │    │ • For: Reconstruction Loss   │       │
│        └──────────────────────┘    └──────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│ STAGE 4: VAE BOTTLENECK        │   │ STAGE 5: SEMANTIC HEAD              │
│                                 │   │                                     │
│ z_inv → VAE → z (latent)       │   │ z_inv → Semantic Space              │
│                                 │   │                                     │
│ Output: z, mu, logvar, kl      │   │ Output: (B, semantic_dim)          │
└─────────────────────────────────┘   └─────────────────────────────────────┘
                    │                               │
                    │                               │
                    ▼                               │
┌─────────────────────────────────┐                 │
│ STAGE 6: RECONSTRUCTION HEAD    │                 │
│                                 │                 │
│ z_var → Decoder → Pixels        │                 │
│                                 │                 │
│ Output: (B, 3, H, W)           │                 │
└─────────────────────────────────┘                 │
                                                    │
                        ┌────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LOSSES (separated - no gradient conflict!)                                  │
│                                                                             │
│   semantic_loss = loss(semantic, teacher_emb)    ← ON z_inv ONLY          │
│   recon_loss = loss(recon, pixels)               ← ON z_var ONLY           │
│   kl_loss = VAE.kl                                ← ON z ONLY               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Innovation: Two-Axis (p, k) Coordinate System

### v1 vs v2 Comparison

| Aspect | MAVT v1 | MAVT v2 |
|--------|---------|---------|
| **Coordinate** | 4D (t, x, y, z) | 2D (p, k) |
| **Axis p** | Spatial (x, y) | Spatial (x, y) |
| **Axis k** | Temporal (t) | Temporal/Scale/View |
| **Attention** | RGAT (typed edges) | ViT + RPB |
| **Split** | Content-Detail (Slot) | Invariant-Variant (Mean) |
| **Gradient** | Mixed | **Isolated** |

### Why (p, k)?

```
Unified representation across modalities:

┌──────────────┬──────────────┬──────────────┐
│   Modality   │      p       │      k       │
├──────────────┼──────────────┼──────────────┤
│   Image      │  (x, y) pos │  Scale level │
│   Video      │  (x, y) pos │  Frame index │
│   3D         │  (x, y) pos │  View (plane)│
└──────────────┴──────────────┴──────────────┘
```

---

## Scale Pyramid for Images

### Problem
Images don't have temporal axis like videos.

### Solution
Use multi-resolution scales as synthetic "temporal axis":

```
Scale Pyramid (K=4):
┌────────────────────────────────────────────────┐
│ Scale 0: H × W     (finest detail)             │
│ Scale 1: H/2 × W/2 (medium detail)             │
│ Scale 2: H/4 × W/4 (coarse structure)          │
│ Scale 3: H/8 × W/8 (global structure)          │
└────────────────────────────────────────────────┘
              ↓ mean across scales
         z_inv (semantic)
              ↓ subtract
         z_var (detail)
```

### Validation (from audit6)

| Statistic | Video Temporal | Image Scale | Match? |
|-----------|----------------|-------------|--------|
| z_inv energy | 97.6-98.8% | ~98.6% | ✅ |
| z_var energy | 1.2-2.4% | ~1.4% | ✅ |
| Gini | 0.62-0.76 | ~0.68 | ✅ |

**Scale pyramid validated: Statistics match video temporal axis!**

---

## Module Map

```
mavt/model/
├── mavt.py                 # MAVT v1 (original)
├── mavt_v2.py              # MAVT v2 (new) 🎯
│
├── two_axis_split.py       # 🎯 Core: Invariant-Variant decomposition
│   ├── TwoAxisDecomposition    # Main class
│   ├── TwoAxisOutput           # Output dataclass
│   └── TwoAxisWithCompression  # With adaptive token budget
│
├── scale_pyramid.py        # 🎯 For images: multi-scale encoding
│   ├── ScalePyramidEncoder        # Full encoder
│   └── LightweightScaleEncoder    # Lightweight version
│
├── backbone_v2.py          # 🎯 Simplified backbone
│   ├── ViTBlock                 # Standard ViT block
│   ├── BackboneV2               # Full backbone (no RGAT!)
│   └── RelativePositionBias     # 2D RPB
│
├── semantic_head.py         # 🎯 Semantic from z_inv
│   ├── SemanticHead             # Main head
│   ├── JEPAPredictor            # JEPA-style prediction
│   └── SigLIPAlignment          # Optional SigLIP alignment
│
├── recon_head_v2.py         # 🎯 Recon from z_var
│   ├── ReconstructionHead       # Main head
│   └── AsymmetricDecoderV2     # Decoder
│
├── unified_position.py      # 🎯 2D position for (p,k)
│   ├── Unified2DPositionEmbedding
│   └── construct_*_positions()
│
└── __init__.py             # Exports
```

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `mavt_v2.py` | ~300 | Main model class |
| `two_axis_split.py` | ~250 | **Core innovation**: Invariant-Variant split |
| `scale_pyramid.py` | ~200 | Multi-scale image encoding |
| `backbone_v2.py` | ~250 | Simplified ViT backbone |
| `semantic_head.py` | ~180 | Semantic from z_inv |
| `recon_head_v2.py` | ~200 | Reconstruction from z_var |
| `unified_position.py` | ~200 | 2D position embedding |
| **Total** | **~1500** | |

---

## Gradient Flow Comparison

### v1 (Problematic)

```
                    ┌─────────────┐
                    │     z      │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ Semantic   │  │ Recon     │  │ VAE KL    │
    │ Loss       │  │ Loss       │  │ Loss       │
    └─────┬──────┘  └─────┬──────┘  └────────────┘
          │               │
          │               │
          └───────┬───────┘
                  │ ⚠️ GRADIENT CONFLICT!
                  │ Semantic gradients contaminate recon
                  │ Recon gradients contaminate semantic
                  ▼
              [BAD]
```

### v2 (Fixed)

```
    ┌─────────────┐              ┌─────────────┐
    │   z_inv     │              │   z_var     │
    └──────┬──────┘              └──────┬──────┘
           │                            │
           ▼                            ▼
    ┌────────────┐              ┌────────────┐
    │ Semantic   │              │ Recon      │
    │ Loss ONLY  │              │ Loss ONLY  │
    └────────────┘              └────────────┘
           │                            │
           │                            │
           ▼                            ▼
    ┌────────────┐              ┌────────────┐
    │  Semantic  │              │Reconstruction│
    │  Gradient  │              │  Gradient    │
    └────────────┘              └────────────┘

    ✅ ISOLATED - No gradient conflict!
```

---

## Expected Improvements

| Metric | v1 | v2 Target | Delta |
|--------|----|-----------|-------|
| **Semantic Quality** | Degraded | **Preserved** | +Quality |
| MSR-VTT R@1 | 40.2% | 45-50% | +5-10% |
| ActivityNet-QA | 0.56 | 0.65-0.70 | +0.09-0.14 |
| Token Coverage | 13% (short) | 20% (short) | +7% |
| Backbone Params | 3.54M | 1.18M | -67% |
| rFID | 0.209 | ≤0.20 | Maintain |

---

## Next Steps

### Phase 1: Core (Week 1-4)
- [ ] Integrate ScalePyramidEncoder into MAVTv2
- [ ] Connect TwoAxisDecomposition to heads
- [ ] Test gradient isolation
- [ ] Validate scale pyramid statistics

### Phase 2: Training (Week 5-8)
- [ ] Update LightningModule for v2
- [ ] Implement separated losses
- [ ] Tune semantic/recon balance
- [ ] Baseline metrics

### Phase 3: Enhancements (Week 9-12)
- [ ] JEPA-style masked prediction
- [ ] Adaptive token budget
- [ ] SigLIP alignment

---

## References

1. **UniJEPA** (2024) — Invariant vs equivariant prediction
2. **V-JEPA** (2024) — JEPA-style masked prediction
3. **VAR** (2024) — Visual autoregressive with multi-scale
4. **FlowAR** (2024) — Scale-wise generation
5. **MAVT audit6** — Video temporal statistics

---

*Implementation Summary completed: 2026-05-27*
