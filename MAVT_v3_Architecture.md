# MAVT v3: Adaptive Decoupled 4D Visual Tokenizer
## Architecture Plan — March 2026

---

## 1. Motivation & Positioning

### 1.1 Problem Statement

Unified visual tokenizers aim to encode image, video, and 3D assets into a shared token space for both reconstruction and understanding. AToken (Apple, Sep 2025) demonstrated this is feasible using a pure transformer with 4D RoPE in a shared latent space. However, AToken treats all tokens equally — every video frame receives the same full spatial encoding (256 tokens per frame), regardless of whether the frame is nearly identical to the previous one.

This is wasteful. In a 16-frame video at 256×256 resolution, AToken produces 2048 tokens. But most consecutive frames share 90%+ of their spatial content — background, lighting, static objects. Only a small fraction of each frame contains novel information (moving objects, camera motion, scene changes).

### 1.2 Core Insight

All visual data has two components:

- **Content**: What the scene looks like (spatial appearance, texture, layout). Stable across time/viewpoint.
- **Dynamics**: What changed (object motion, camera movement, viewpoint shift, depth variation). Sparse, varies per frame/view.

By decoupling these two components and adaptively allocating tokens based on the magnitude of change, we can achieve the same reconstruction quality with significantly fewer tokens — or better quality at the same token budget.

### 1.3 Contribution Summary

| # | Contribution | Novelty Source |
|---|---|---|
| C1 | Content-dynamics decoupling in unified 4D space | VTok did video-only; we extend to unified image+video+3D |
| C2 | Adaptive dynamics allocation (variable K per frame/view) | AdapTok did uniform tokens; we do it on dynamics tokens only |
| C3 | Lightweight delta scorer predicting optimal K from feature distance | New module, <1M params, enables content-aware budget |
| C4 | Redundancy penalty loss ensuring dynamics tokens are informative | New loss component preventing dynamics from duplicating content |

### 1.4 Landscape Comparison

| System | Modalities | Decoupled | Adaptive | Recon+Understand | Token Types |
|---|---|---|---|---|---|
| AToken (Apple) | Img+Vid+3D | No | No | Both | Continuous+Discrete |
| COSMOS (NVIDIA) | Img+Vid | No | No | Recon only | Both |
| UniTok (NeurIPS'25) | Img only | No | No | Both | Discrete |
| VTok (Feb 2026) | Vid only | Yes | No | Both | Continuous |
| OmniTokenizer | Img+Vid | Yes | No | Recon only | Both |
| AdapTok (May 2025) | Vid only | No | Yes | Recon only | Discrete |
| EvoTok (Mar 2026) | Img only | Residual | No | Both | Discrete |
| **MAVT v3 (Ours)** | **Img+Vid+3D** | **Yes** | **Yes** | **Both** | **Continuous+Discrete** |

No existing system combines decoupled tokenization + adaptive allocation + unified multi-modal + dual task support.

---

## 2. 4D Coordinate System

All modalities map into a shared 4D space (t, x, y, z). Content tokens occupy t=0 (reference). Dynamics tokens occupy t>0 (deltas).

### 2.1 Image

```
Content tokens: (t=0, x=col, y=row, z=0)     256 tokens for 256×256
Dynamics tokens: none                          0 tokens
Total: 256 tokens
```

Image is the trivial case — only content, no dynamics. Identical to AToken.

### 2.2 Video (T frames × H × W)

```
Content tokens:  (t=0, x=col, y=row, z=0)     256 tokens (key frame 0, full spatial)
Dynamics tokens: (t=i, x=0, y=0, z=0)         K_i tokens per delta frame i=1..T-1
Total: 256 + Σ K_i tokens

Example 16f×256×256:
  AToken: 16 × 256 = 4096 tokens (flat, all equal)
  Ours fixed K=1: 256 + 15 = 271 tokens (7.5× reduction)
  Ours adaptive:  256 + ~44 = ~300 tokens (content-aware)
```

### 2.3 3D Object (V views)

```
Content tokens:  (t=0, x=col, y=row, z=0)     256 tokens (canonical view 0, full spatial)
Dynamics tokens: (t=v, x=0, y=0, z=0)         K_v tokens per delta view v=1..V-1
Total: 256 + Σ K_v tokens

Example 8-view, 256×256:
  AToken: 8 × 256 = 2048 tokens (each view full)
  Ours fixed K=4: 256 + 28 = 284 tokens (7.2× reduction)
  Ours adaptive:  256 + ~20 = ~276 tokens (content-aware)
```

Key design choice: 3D views are treated identically to video frames — "viewpoint change" is analogous to "temporal change". The encoder sees no difference between video and 3D; it only sees a dense spatial cluster at t=0 and sparse dynamics tokens at t>0.

---

## 3. Architecture Overview

```
Raw Input (Image / Video / 3D)
    │
    ▼
╔═══════════════════════════════════════════════════╗
║  STEP 1: SigLIP2 Patchify (all frames/views)     ║
║  Each frame/view → (N_patch, 1152)                ║
║  Frozen or progressive unfreeze                   ║
╚═══════════════════════════════════════════════════╝
    │
    ▼
╔═══════════════════════════════════════════════════╗
║  STEP 2: Content-Dynamics Split                   ║
║  Frame 0 → keep all N_patch tokens (content)      ║
║  Frame i → cross-attn pool into K_i tokens (dyn)  ║
║  K_i determined by delta scorer                   ║
╚═══════════════════════════════════════════════════╝
    │
    ▼
╔═══════════════════════════════════════════════════╗
║  STEP 3: 4D Position Assignment + Concatenation   ║
║  Content: (t=0, x, y, z=0)                       ║
║  Dynamics: (t=i, 0, 0, 0)                        ║
║  → Single token sequence per sample               ║
╚═══════════════════════════════════════════════════╝
    │
    ▼
╔═══════════════════════════════════════════════════╗
║  STEP 4: Transformer Encoder                      ║
║  Pure self-attention + 4D RoPE (AToken-style)     ║
║  All tokens attend all tokens within same sample  ║
║  NaViT packing with block-diagonal mask           ║
╚═══════════════════════════════════════════════════╝
    │
    ▼
╔═══════════════════════════════════════════════════╗
║  STEP 5: Dual Projection                         ║
║  z_recon: (N_total, d_latent) for reconstruction  ║
║  z_understand: (768) global vector for alignment  ║
║  Optional: FSQ for discrete tokens (Stage 4)      ║
╚═══════════════════════════════════════════════════╝
    │
    ▼
╔═══════════════════════════════════════════════════╗
║  STEP 6: Decoder                                  ║
║  Content z → base canvas reconstruction           ║
║  Dynamics z → per-frame/view delta application    ║
║  Task-specific heads: pixel / video / 3D gaussian ║
╚═══════════════════════════════════════════════════╝
    │
    ▼
Reconstructed Output (all frames/views)
```

---

## 4. Module Details

### 4.1 Step 1: SigLIP2 Patchify

Identical to AToken. SigLIP2-SO400M patch embedding extended to 4D with:
- Patch size: 16×16 spatial, temporal patch τ=2 (for video)
- 4D RoPE applied per attention layer
- Zero-initialized temporal weights preserve image features
- Modality embedding added (learned, 3 classes)

```
Input per frame/view: (B, 3, H, W)
Output per frame/view: (B, N_patch, 1152)
  where N_patch = (H/16) × (W/16)
  e.g., 256×256 → N_patch = 256
```

Progressive freeze schedule (following AToken):
- Stage 1: SigLIP2 fully frozen
- Stage 2: Last 4 layers unfrozen (LR × 0.1)
- Stage 3: Fully unfrozen (LR × 0.3)

### 4.2 Step 2: Content-Dynamics Split

This is the core novel module.

#### 4.2.1 Content Extraction

Frame/view index 0 is designated as the **key frame**. Its SigLIP2 features are kept as-is:

```
content_tokens = siglip2(frame_0)    # (N_patch, 1152)
```

No compression, no pooling. Full spatial detail preserved.

#### 4.2.2 Delta Scorer

For each subsequent frame i, a lightweight scorer estimates "how different is this frame from the key frame" and outputs an optimal token count K_i.

```python
class DeltaScorer(nn.Module):
    """
    Predicts optimal K for each delta frame/view.
    
    Input: content features (N, D) and frame_i features (N, D)
    Output: K_i ∈ {1, 2, 4, 8}
    
    Parameters: ~0.5M (negligible)
    """
    def __init__(self, d_model=1152, K_options=[1, 2, 4, 8]):
        super().__init__()
        self.K_options = K_options
        
        # Compute per-patch feature distance then aggregate
        self.score_mlp = nn.Sequential(
            nn.Linear(3, 64),    # 3 aggregate stats as input
            nn.GELU(),
            nn.Linear(64, len(K_options)),  # logits over K options
        )
    
    def forward(self, content_feat, frame_i_feat):
        """
        content_feat:  (N, D) — key frame SigLIP2 features
        frame_i_feat:  (N, D) — delta frame SigLIP2 features
        
        Returns: K_i (int), score (float for logging)
        """
        # Per-patch cosine distance
        cos_sim = F.cosine_similarity(content_feat, frame_i_feat, dim=-1)  # (N,)
        
        # Aggregate statistics
        stats = torch.stack([
            cos_sim.mean(),           # average similarity (low = high change)
            cos_sim.std(),            # variance (high = mixed static+motion)
            (cos_sim < 0.8).float().mean(),  # fraction of changed patches
        ])  # (3,)
        
        # Predict K
        logits = self.score_mlp(stats)  # (len(K_options),)
        
        if self.training:
            # Gumbel-softmax for differentiable selection
            idx = F.gumbel_softmax(logits, tau=1.0, hard=True).argmax()
        else:
            idx = logits.argmax()
        
        K_i = self.K_options[idx]
        return K_i, cos_sim.mean()
```

#### 4.2.3 Cross-Attention Pooling

Given K_i from the scorer, compress frame_i's 256 spatial features into K_i dynamics tokens:

```python
class DynamicsPooler(nn.Module):
    """
    Compress delta frame features into K compact dynamics tokens.
    
    Uses learnable query tokens that cross-attend to
    concat(content_features, frame_i_features).
    
    The queries learn to extract "what is new in frame_i
    that content doesn't already contain."
    
    Parameters: ~5M
    """
    def __init__(self, d_model=1152, n_heads=16, max_K=8):
        super().__init__()
        self.max_K = max_K
        
        # Learnable query tokens (max_K, shared across all frames)
        self.queries = nn.Parameter(torch.randn(max_K, d_model) * 0.02)
        
        # Cross-attention: Q=queries, KV=concat(content, frame_i)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
    
    def forward(self, content_feat, frame_i_feat, K_i):
        """
        content_feat:  (N, D) — key frame features
        frame_i_feat:  (N, D) — delta frame features
        K_i:           int — number of dynamics tokens to produce
        
        Returns: dynamics_tokens (K_i, D)
        """
        # Select first K_i queries
        Q = self.queries[:K_i].unsqueeze(0)  # (1, K_i, D)
        
        # KV = concat content + frame_i (both provide context)
        KV = torch.cat([content_feat, frame_i_feat], dim=0)  # (2N, D)
        KV = KV.unsqueeze(0)  # (1, 2N, D)
        
        # Cross-attention
        out, _ = self.cross_attn(Q, KV, KV)  # (1, K_i, D)
        out = self.norm(out + Q)
        
        # FFN
        out = self.norm_ffn(out + self.ffn(out))
        
        return out.squeeze(0)  # (K_i, D)
```

#### 4.2.4 Full Split Procedure

```python
def content_dynamics_split(frames, siglip2, scorer, pooler):
    """
    frames: list of (3, H, W) tensors, length T
    
    Returns:
        content_tokens:  (N_patch, D)
        dynamics_tokens: (Σ K_i, D)
        positions:       (N_patch + Σ K_i, 4)
        token_types:     (N_patch + Σ K_i,) — 0=content, 1=dynamics
    """
    # Encode all frames with SigLIP2
    all_features = [siglip2(f) for f in frames]  # list of (N, D)
    
    # Content = frame 0
    content = all_features[0]  # (N, D)
    N = content.shape[0]
    n_h = int(N ** 0.5)  # assume square
    
    # Content positions: (t=0, x, y, z=0)
    gy, gx = torch.meshgrid(torch.arange(n_h), torch.arange(n_h), indexing='ij')
    content_pos = torch.zeros(N, 4)
    content_pos[:, 1] = gx.flatten().float()
    content_pos[:, 2] = gy.flatten().float()
    
    # Dynamics = frames 1..T-1
    all_dynamics = []
    all_dyn_pos = []
    
    for i in range(1, len(frames)):
        # Score: how different is frame i?
        K_i, score = scorer(content, all_features[i])
        
        # Pool into K_i tokens
        dyn_tokens = pooler(content, all_features[i], K_i)  # (K_i, D)
        all_dynamics.append(dyn_tokens)
        
        # Dynamics positions: (t=i, x=0, y=0, z=0)
        dyn_pos = torch.zeros(K_i, 4)
        dyn_pos[:, 0] = float(i)  # temporal position
        all_dyn_pos.append(dyn_pos)
    
    # Concatenate
    if all_dynamics:
        dynamics = torch.cat(all_dynamics, dim=0)
        dynamics_pos = torch.cat(all_dyn_pos, dim=0)
        
        tokens = torch.cat([content, dynamics], dim=0)
        positions = torch.cat([content_pos, dynamics_pos], dim=0)
        token_types = torch.cat([
            torch.zeros(N), torch.ones(dynamics.shape[0])
        ])
    else:
        # Image: no dynamics
        tokens = content
        positions = content_pos
        token_types = torch.zeros(N)
    
    return tokens, positions, token_types
```

### 4.3 Step 3: 4D Position Assignment

Content tokens get dense spatial positions at t=0. Dynamics tokens get sparse temporal positions. RoPE handles the rest — it naturally encodes:
- Content-content: close in (x,y), attend strongly → spatial layout
- Dynamics-dynamics: close in t, attend strongly → temporal trajectory
- Content-dynamics: far in t, moderate attention → context

No special position encoding module needed. Standard 4D RoPE (same as AToken) handles everything.

### 4.4 Step 4: Transformer Encoder

Pure self-attention + 4D RoPE. Identical to AToken's architecture.

```
Architecture: SigLIP2-SO400M transformer (~400M frozen + trainable layers)
Layers: 27 transformer blocks (following SigLIP2)
Attention: Standard multi-head self-attention
Position: 4D RoPE applied per layer
Sequence: NaViT-packed, block-diagonal attention mask
```

Key advantage: because content-dynamics split reduces token count drastically (2048 → ~300 for video), the same encoder processes the same information much faster. Attention is O(N²), so 300 vs 2048 tokens = ~46× fewer attention operations.

### 4.5 Step 5: Dual Projection

Following AToken's design exactly:

```
Reconstruction path:
  encoder_output (N_total, 1152) → Linear → z_recon (N_total, d_latent)
  d_latent = 48 (continuous) or FSQ quantized (discrete)

Understanding path:
  encoder_output (N_total, 1152) → Attention pooling → z_global (1, 1152)
  z_global → Linear → z_understand (768) aligned with text embeddings

KL divergence loss on z_recon (VAE-style)
```

Optional Stage 4: FSQ quantization for discrete tokens (autoregressive compatibility).

### 4.6 Step 6: Decoder

The decoder receives z_recon which contains both content and dynamics latents. It must reconstruct ALL frames/views.

```
Decoder architecture: Transformer (trained from scratch), asymmetric
  - Deeper than encoder (following ViTok/GigaTok findings)
  - Input: z_recon (N_total, d_latent)
  - 4D RoPE for position awareness
  - Token types (content vs dynamics) available as conditioning

Reconstruction approach:
  1. Project z_recon back to d_model: Linear(d_latent → 1152)
  2. Self-attention across all tokens (content + dynamics interact)
  3. For each output frame i:
     - Content tokens → base spatial reconstruction
     - Dynamics tokens at t=i → delta modifiers
     - Combine: frame_i = base + delta_i
  4. Output heads:
     - Image/Video: pixel space (3, H, W) per frame
     - 3D: Gaussian splatting parameters per view
```

---

## 5. Loss Functions

### 5.1 Reconstruction Losses (following AToken)

```
L_recon = L1(pred, target)                           w = 1.0
L_perceptual = LPIPS(pred, target)                   w = 1.0
L_gram = GramMatrix(pred_features, target_features)  w = 1.0
L_kl = KL(q(z|x) || p(z))                           w = 0.001
```

GAN-free — following AToken's finding that Gram matrix loss handles the covariance component (86.6% of rFID) without adversarial instability.

### 5.2 Understanding Loss

```
L_understand = ContrastiveLoss(z_understand, text_embedding)  w = 0.3
L_vf = DINOv2_alignment(encoder_features, dino_features)     w = 0.5
```

### 5.3 Novel: Redundancy Penalty (C4)

Ensure dynamics tokens encode genuinely new information, not repeat content:

```python
def redundancy_loss(content_tokens, dynamics_tokens):
    """
    Penalize dynamics tokens that are too similar to content tokens.
    Pushes dynamics to encode only the delta information.
    
    content_tokens:  (N_content, D) — from encoder
    dynamics_tokens: (N_dynamics, D) — from encoder
    """
    # Cosine similarity between each dynamics token and all content tokens
    content_norm = F.normalize(content_tokens, dim=-1)
    dynamics_norm = F.normalize(dynamics_tokens, dim=-1)
    
    sim = torch.mm(dynamics_norm, content_norm.T)  # (N_dyn, N_content)
    max_sim = sim.max(dim=-1).values               # (N_dyn,)
    
    # Penalize high similarity (dynamics should be different from content)
    loss = F.relu(max_sim - 0.5).mean()  # threshold at 0.5 cosine sim
    
    return loss
```

```
L_redundancy = redundancy_loss(content_z, dynamics_z)  w = 0.1
```

### 5.4 Novel: Scorer Supervision

Train the delta scorer to predict K_i that minimizes reconstruction error under budget:

```python
def scorer_loss(predicted_K, reconstruction_errors, budget):
    """
    predicted_K: (T-1,) — predicted K per frame
    reconstruction_errors: (T-1,) — per-frame recon loss
    budget: int — total dynamics token budget
    
    Goal: minimize total recon error subject to Σ K_i ≤ budget
    """
    # Rate loss: encourage efficiency
    L_rate = F.relu(predicted_K.sum() - budget)
    
    # Quality loss: high-error frames should get more tokens
    # (frames with high recon error should have high K)
    correlation = -torch.corrcoef(
        torch.stack([predicted_K.float(), reconstruction_errors])
    )[0, 1]
    L_quality = F.relu(correlation)  # penalize negative correlation
    
    return L_rate + L_quality
```

### 5.5 Total Loss

```
L_total = L_recon + L_perceptual + L_gram + L_kl
        + L_understand + L_vf
        + L_redundancy
        + L_scorer
```

---

## 6. Training Pipeline

### 6.1 Progressive Curriculum (following AToken)

| Stage | Steps | Modalities | Resolution | SigLIP2 | Dynamics | Key Changes |
|---|---|---|---|---|---|---|
| 1 | 200k | Image only | 64-512 | Frozen | None (content only) | Learn spatial reconstruction |
| 2 | 200k | Image + Video | Img 64-1024, Vid 64-512×4-16f | Last 4 unfrozen | Video dynamics active | + temporal decoupling, + understanding loss |
| 3 | 100k | Img + Vid + 3D | Img 64-2048, Vid 64-1024×4-32f | Fully unfrozen | Video + 3D dynamics | + 3D views as dynamics, + depth reasoning |
| 4 | 50k | All | All | Fully unfrozen | All + adaptive K | + FSQ discrete tokens, + scorer training |

### 6.2 Stage-Specific Details

**Stage 1 — Image Foundation:**
- Content-dynamics split not needed (images have no dynamics)
- Train encoder + decoder on image reconstruction only
- Loss: L_recon + L_perceptual + L_gram + L_kl
- This establishes spatial encoding quality (target: rFID ≤ 0.25)

**Stage 2 — Video Dynamics:**
- Introduce content-dynamics split with fixed K=1 per frame
- Delta scorer not active yet — all frames get equal allocation
- Loss: + L_understand + L_vf + L_redundancy
- Key metric: rFVD on video, check image rFID doesn't degrade

**Stage 3 — 3D Geometry:**
- Add 3D objects, treat views as temporal steps
- Still fixed K — establish that 3D views work with decoupling
- Loss: + 3D reconstruction loss (PSNR on rendered views)
- Key finding to validate: cross-modal enhancement (AToken showed 19% image improvement)

**Stage 4 — Adaptive + Discrete:**
- Activate delta scorer, train with scorer_loss
- Allow K ∈ {1, 2, 4, 8} per frame/view
- Add optional FSQ quantization for discrete tokens
- Fine-tune entire model end-to-end

### 6.3 Optimizer

```
Optimizer:     AdamW
Betas:         (0.9, 0.95)
Weight decay:  0.05
Grad clip:     1.0
Precision:     bfloat16

Learning rates per stage:
  Stage 1: 1e-4, warmup 10k steps, cosine decay
  Stage 2: 5e-5, warmup 5k steps
  Stage 3: 3e-5, warmup 2k steps
  Stage 4: 1e-5, warmup 1k steps

Parameter groups:
  SigLIP2:         0.0 → 0.1 → 0.3 (across stages)
  Dynamics pooler:  1.0
  Delta scorer:     1.0 (active from Stage 4)
  Decoder:          1.0
  Latent projection: 1.0
```

### 6.4 Data

```
Images:  DFN-2B, OpenImages v7, LAION-aesthetic (filtered)
Video:   WebVid-10M, Panda-70M, InternVid-10M
3D:      Objaverse 800K + Cap3D captions

NaViT packing: L=4096 max sequence length per pack
  Image 256²: 256 content tokens → 16 images per pack
  Video 16f×256²: ~300 tokens → 13 videos per pack
  3D 8-view: ~284 tokens → 14 objects per pack
```

---

## 7. Ablation Plan

Each contribution is independently ablatable:

| Experiment | Config | Expected Result |
|---|---|---|
| Baseline | AToken reproduction (flat 4D, all tokens equal) | rFID ~0.21, rFVD ~3.0 |
| + C1 | Decoupled, fixed K=1 | Similar quality, ~7× fewer video tokens |
| + C1 + K=4 | Decoupled, fixed K=4 | Better video quality than K=1, more tokens |
| + C1 + C2 | Decoupled, adaptive K (1-8) | Best quality-efficiency tradeoff |
| + C1 + C2 + C3 | Learned scorer vs heuristic threshold | Scorer allocates better than cosine heuristic |
| + C1 + C2 + C3 + C4 | + redundancy loss | Dynamics tokens more informative, slight quality gain |
| Discrete variant | + FSQ at Stage 4 | Competitive with AToken discrete |

### 7.1 Key Metrics

```
Image:  rFID (↓), ImageNet zero-shot accuracy (↑)
Video:  rFVD (↓), MSRVTT retrieval R@1 (↑)
3D:     PSNR (↑), classification accuracy (↑)

Efficiency: tokens per sample, throughput (samples/sec), GPU memory

Cross-modal: does adding video/3D improve image quality? (AToken showed yes)
```

---

## 8. Estimated Parameters

| Component | Parameters | Notes |
|---|---|---|
| SigLIP2 encoder (frozen) | ~400M | Pretrained, progressively unfrozen |
| Dynamics pooler | ~5M | Cross-attention + FFN |
| Delta scorer | ~0.5M | Lightweight MLP |
| Latent projection | ~5M | Linear layers (recon + understand) |
| Decoder | ~400M | Asymmetric, from scratch |
| **Total trainable** | **~410M** | Without SigLIP2 frozen |
| **Total with frozen** | **~810M** | Comparable to AToken ~800M |

---

## 9. Comparison with AToken

| Aspect | AToken | MAVT v3 |
|---|---|---|
| Encoder | SigLIP2 + 4D RoPE | Same (identical) |
| Tokenization | Flat: every frame full spatial | Decoupled: content full + dynamics compact |
| Token count (16f video) | 2048 | ~300 (adaptive) |
| Attention cost (16f video) | O(2048²) = 4.2M | O(300²) = 90K (~46× less) |
| Adaptive allocation | No | Yes (delta scorer) |
| 3D handling | Voxel grid 64³ | Multi-view decoupled (same as video) |
| Discrete support | FSQ (Stage 4) | FSQ (Stage 4), same |
| Training | Progressive 4-stage | Progressive 4-stage, same |
| Loss | GAN-free (Gram + perceptual) | Same + redundancy penalty |
| Novel modules | 4D RoPE, Gram loss | Content-dynamics split, delta scorer, redundancy loss |

---

## 10. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| K=1 too lossy for complex motion | Medium | Adaptive K (C2) handles this |
| Cross-attention pooling loses spatial detail | Medium | Increase max K; ablate K vs quality |
| Delta scorer hard to train | Low | Start with heuristic (cosine threshold), add learned scorer later |
| 3D views as temporal steps may not work | Medium | Ablate 3D separately; fall back to AToken-style if needed |
| Redundancy loss too aggressive | Low | Tune weight; threshold at 0.5 prevents over-penalizing |
| Reviewer says "just AToken + VTok" | High | Emphasize adaptive K (novel), redundancy loss (novel), unified 3D (novel combination) |

---

## 11. Paper Structure (Draft)

```
Title: "Adaptive Decoupled Tokenization for Unified Visual Representation"

Abstract: Unified visual tokenizers encode images, videos, and 3D assets
into a shared space, but treat all tokens equally regardless of information
content. We introduce content-dynamics decoupling — separating stable
appearance from temporal/viewpoint changes — combined with adaptive token
allocation that assigns more dynamics tokens to high-change frames and
fewer to static ones. On the same token budget, our approach achieves
X% better reconstruction; on the same quality, uses Y% fewer tokens.

1. Introduction
2. Related Work
   - Unified tokenizers (AToken, UniTok, COSMOS)
   - Spatial-temporal decoupling (VTok, OmniTokenizer, VidTwin)
   - Adaptive allocation (AdapTok, ElasticTok, ALIT)
3. Method
   3.1 Content-Dynamics Split in Unified 4D Space
   3.2 Adaptive Dynamics Allocation (Delta Scorer)
   3.3 Cross-Attention Dynamics Pooling
   3.4 Redundancy Penalty Loss
   3.5 Training Pipeline
4. Experiments
   4.1 Image Reconstruction & Understanding
   4.2 Video Reconstruction & Understanding
   4.3 3D Reconstruction & Understanding
   4.4 Ablation Study
   4.5 Efficiency Analysis
   4.6 Downstream Applications
5. Discussion & Conclusion
```

---

*Document generated: March 2026*
*Status: Architecture plan — pre-implementation*
