# MAVT (AnTokenizer) — Model & Pipeline

> Unified vision tokenizer cho image / video / 3D, học chung 1 latent space.
> Train pipeline 3-stage curriculum (image → +video → +3D).
> Code chính: `src/mavt/model/` + `src/mavt/training/lightning_module.py`.

---

## 1. Pipeline tổng quan (7 stages)

```
                 ┌─────────────────┐
   raw input ──▶ │ 1. Patchify     │  Conv3d, modality-specific
                 │   (multi-modal) │  → tokens (B,N,D), positions (N,4), plane_ids (N,)
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ 2. Hybrid       │  Transformer + RGAT, 12 blocks, dim=768
                 │   Backbone      │  → features (B,N,D)
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ 3. C-D Split    │  GLOBAL content (slot attn) + LOCAL detail (window pool)
                 │   (cd-split)    │  → compressed (B, N_c+N_d, D), latent_positions, token_types
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ 4. VAE bottleneck│  N×D → N×latent_dim (32), reparametrize z
                 │   + KL loss     │  → z, mu, logvar, loss_kl
                 └────────┬────────┘
                          │
              ┌───────────┴───────────┐
       ┌──────▼──────┐         ┌──────▼──────┐
       │ 5a. Recon   │         │ 5b. Underst.│
       │   Decoder   │         │   Decoder   │
       │ (z → pixel) │         │ (z → semantic, distill từ SigLIP2)
       └──────┬──────┘         └──────┬──────┘
              │                        │
              ▼                        ▼
        recon (B,3,H,W)         semantic (B, 768)
              │                        │
              └─────────┬──────────────┘
                        │
              ┌─────────▼─────────┐
              │ 6. Loss (MAVTLoss)│  l1 + lpips + KL + sem distill + temporal (video) + slot div
              │   per-modality EMA│  proportional weighting
              └─────────┬─────────┘
                        │
                        ▼
              7. Outputs (`MAVTOutput` dataclass)
```

---

## 2. Stage chi tiết

### Stage 1 — Patchify (`patchify.py`)
- **Image** `(B, 3, H, W)`: Conv3d với causal-pad 1 frame ảo → `(B, D, 1, Hp, Wp)` → flatten `(B, Hp·Wp, D)`. Position `(0, i, j, 0)`, plane_id=−1.
- **Video** `(B, 3, T, H, W)`: Conv3d giảm temporal `t_patch=2`, spatial `patch=16` → `(B, D, Tp, Hp, Wp)` → `(B, Tp·Hp·Wp, D)`. Position `(t, i, j, 0)`, plane_id=−1.
- **Threed** `(B, 3, 3, S, S)` (3 planes XY/XZ/YZ): Conv3d riêng cho mỗi plane → concat `(B, 3·Hp·Wp, D)`. Position encode plane-specific axes; plane_id ∈ {0,1,2}.
- Tất cả share Conv3d weight → unified patch embedding qua modality.
- Kèm 4D `pos_embed` (Fourier features, dim D) cộng vào tokens.

### Stage 2 — Hybrid Backbone (`backbone.py`)
- 12 layers xen kẽ Transformer blocks + RGAT (Relational Graph Attention).
- RGAT là attention dạng graph với:
  - `r_s=2` neighborhoods spatial (window-based)
  - `r_t=1` neighborhood temporal
- `use_gradient_checkpointing=true`: tradeoff compute/memory (đang OFF cho run tuned để tăng tốc).
- SigLIP2 weights được load vào last 4 transformer blocks (Stage 1: frozen, Stage 2: unfrozen).

### Stage 3 — Content-Detail Split ⭐ **(deep dive section 3)**

### Stage 4 — VAE bottleneck (`latent_heads.py`)
- `VAEHead`: Linear(D=768 → 2·latent_dim=64) → split mu, logvar → reparametrize trick.
- KL loss: `KL(N(mu, σ²) || N(0, I))`, scale bằng `kl_weight=1e-4` (built-in).
- Output `z` shape `(B, N_c+N_d_local, latent_dim=32)`.
- Note: post `42c622b update`, `w_kl=1.0` ở MAVTLoss = passthrough — đã pre-scale 1e-4 trong VAEHead → tránh double-scale.

### Stage 5 — Decoders (`decoder.py`)

**5a. Reconstruction `AsymmetricDecoder`:**
- `UnifiedDetailExpander` cross-attention từ target grid positions vào latent z (dim 32).
- 4 self-attention blocks dim 768.
- Pixel projection → `(B, 3, H, W)` cho image, frame-by-frame cho video.
- **Mới (cd-split)**: nhận `latent_positions` + `latent_token_types` để áp distance-bias attention (xem section 3).

**5b. Understanding `UnderstandingDecoder`:**
- 2 cross-attn layers + linear proj → `(B, semantic_dim=768)`.
- Trained để khớp với SigLIP2 teacher's `pooler_output` qua cosine loss.

### Stage 6 — Loss (`losses.py`)

```
L_total = w_l1 · L1(pred, target)
        + w_lpips · LPIPS(pred, target)         # AlexNet/VGG perceptual
        + w_kl · L_KL                            # đã pre-scaled ở VAEHead
        + w_sem · (1 - cos(MAVT.semantic, teacher.pooler))
        + w_temp · L1(Δ_t pred, Δ_t target)     # chỉ video, T>1
        + w_aux · slot_diversity_penalty
```
- Mỗi modality scale bằng `ModalityEMAWeighter.weight(modality)`:
  - `weight(m) = ema_m / mean(ema_active)` — modality có loss CAO được boost (cross-stage gradient flow vào branch chưa train)
  - Sau commit `42c622b update`. Trước đó là 1/ema (logic ngược).

---

## 3. ⭐ C-D Split (sau commit `6368dfb cd-split`)

### Ý tưởng cốt lõi

Phân chia input tokens thành 2 kênh có **đặc tính khác nhau**, encode bằng **cơ chế khác nhau**:

| Kênh | Bản chất | Cơ chế | Position info |
|---|---|---|---|
| **Content** | Semantic / low-freq / global | Slot cross-attention (toàn ảnh) | Không có (slot là global summary) |
| **Detail** | Residual / high-freq / local | Coordinate window pooling | **Có** (window center) |

### Tại sao Detail cần local + position?

**Trước cd-split** (slot pooler global cho cả detail):
```
detail = SlotPooler(N_d=25 slots)(Residual)
       (slots tự học pool ở đâu, không có vị trí)
```
- Decoder cross-attend vào detail slots không biết slot này ứng với patch nào → phải reconstruct texture từ "positionless global slots" → khó.
- High-freq (texture, edges) cần spatial precision → mất khi pool global.

**Sau cd-split** (windowed pool + position):
```python
# Group residual tokens theo coordinate window
group_key = (plane_id, t // t_win, x // s_win, y // s_win, z // s_win)
# Mean-pool tokens trong cùng window
detail_token[g] = mean(residual[token] for token in window g)
detail_position[g] = mean(positions[token] for token in window g) + 0.5 # window center
```
- **Mỗi detail token có toạ độ rõ ràng** → decoder biết detail thuộc patch nào.
- Decoder dùng **distance bias** (Manhattan) trong cross-attn để mỗi pixel ưu tiên detail token gần.
- Compression vẫn tốt: window 2×2 → 4 token → 1 token (75% giảm), tổng compression vẫn ~50% (bằng N_c + N_d_local).

### Architecture sau update

```
                    ┌─ slot attn ──▶  C  (B, N_c, D)        [global, positionless]
features ──┬─▶ ─────┤
(B,N,D)    │        └─ approx via inverse softmax weights:
           │           x_approx = softmax(C @ xᵀ / √D)ᵀ @ C
           │
           └─▶ R = x - x_approx (residual)
                    │
                    ▼
              ┌──────────────────────────────────────┐
              │   _local_detail_pool(R, pos, plane)  │
              │                                       │
              │  group_key = (plane, t/1, i/2, j/2, k/2)
              │  D_tokens[g] = mean(R[t] for t in g)  │
              │  D_pos[g]    = mean(pos[t]) + 0.5     │
              │  D_tokens   ← detail_proj(detail_norm(.))
              └──────────┬───────────────────────────┘
                         │
                         ▼
                    detail tokens (B, N_d_local, D)  + detail_positions (N_d_local, 4)

compressed = concat([C, detail_tokens])  (B, N_c + N_d_local, D)
latent_positions  = concat([zeros(N_c, 4), detail_positions])
latent_token_types = concat([zeros(N_c), ones(N_d_local)])
```

### Decoder sử dụng metadata thế nào?

```python
# UnifiedDetailExpander forward
kv = z + kv_pos_scale * kv_pos_enc(latent_positions)        # add 4D Fourier pos
       + token_type_scale * token_type_embed(latent_token_types)  # +0/+1 embedding

# Distance bias chỉ apply cho detail keys
dist_manhattan = |query_pos - kv_pos|
attn_bias[detail_keys] = -local_detail_bias * dist  # local_detail_bias=0.25

cross_attn(query, kv, kv, attn_mask=attn_bias)
```
→ Pixel position xa detail position thì attention bị penalize logarithmically (softmax-scale).
→ Content tokens KHÔNG bị penalize → decoder vẫn dùng được toàn bộ semantic info.

### Worked example: image 256×256

```
Input: x  shape = (B, 3, 256, 256)
Patch_size=16  →  Hp=Wp=16  →  N=256 tokens
positions = [(0,i,j,0) for i,j in 16×16]  → (256, 4)
plane_ids = [-1] * 256

content_ratio=0.25, detail_ratio=0.25 → N_c=64, N_d_key=64 (key naming only)
```

**Stage 3a: Content slot pool**
```
slot_pooler = SlotPooler(num_slots=64, dim=768, num_heads=8, num_layers=2)
C = slot_pooler(features)  # 2 cross-attn layers
   shape (B, 64, 768)
```
- 64 learned slots cross-attend toàn 256 tokens → mỗi slot là weighted summary toàn ảnh.

**Stage 3b: Approximate + residual**
```
weights = softmax(C @ features.T / sqrt(768), dim=-1)  # (B, 64, 256)
x_approx = weights.T @ C                                # (B, 256, 768)
R = features - x_approx                                 # (B, 256, 768)  — high-freq
```

**Stage 3c: Local detail pool (window=2)**
```
group_key[token_n] = (plane_id=-1, t=0, i//2, j//2, z=0)
                   = (-1, 0, i//2, j//2, 0)

i=0,j=0 → key (-1,0,0,0,0)   group 0
i=0,j=1 → key (-1,0,0,0,0)   group 0  (same window 2×2)
i=0,j=2 → key (-1,0,0,1,0)   group 1
i=0,j=3 → key (-1,0,0,1,0)   group 1
...
i=1,j=0 → key (-1,0,0,0,0)   group 0
i=1,j=1 → key (-1,0,0,0,0)   group 0
...
```
→ 4 token (vd i=0..1, j=0..1) gộp vào group 0.
→ Tổng số group = 8×8 = **64 detail tokens**.

```
counts[0] = 4 (i=0,1; j=0,1)
D_token[0] = mean(R[0], R[1], R[16], R[17])     # 4 token trong window 2×2
D_token[0] = detail_proj(detail_norm(D_token[0]))

D_pos[0] = mean([(0,0,0,0), (0,0,1,0), (0,1,0,0), (0,1,1,0)]) + 0.5 = (0, 1, 1, 0)
                                                                  → window center floor
```

**Output**:
- `compressed` = concat(C, D_tokens) shape `(B, 128, 768)`
- `latent_positions` shape `(128, 4)`:
  - First 64 rows: `(0,0,0,0)` (content, positionless)
  - Last 64 rows: window centers like `(0, 1, 1, 0)`, `(0, 1, 3, 0)`, ...
- `latent_token_types` shape `(128,)`: `[0]*64 + [1]*64`

### Worked example: video 256×256, 16 frames

```
T=16, t_patch=2 → Tp=8
N = 8 × 16 × 16 = 2048 tokens

content_ratio=0.25 → N_c=512
detail_ratio=0.25  → N_d_key=512 (naming)

Detail windows (s_win=2, t_win=1):
  group_key = (plane=-1, t//1=t, i//2, j//2, 0)
  t in [0..7]:    8 unique
  i//2 in [0..7]: 8 unique
  j//2 in [0..7]: 8 unique
  → 8 × 8 × 8 = 512 detail windows
```
- Compressed: 512 + 512 = **1024 tokens** (vs 2048 raw → 2× compression)
- Mỗi detail token gồm 1 temporal × 4 spatial residual (tổng 4 raw tokens).

### Compare số token: image 256² (3 modality khác nhau)

| Modality | N raw | N_c (content) | N_d_local (detail, win=2) | Total | Compression |
|---|---:|---:|---:|---:|---:|
| Image | 256 | 64 | **64** | **128** | 2× |
| Video | 2048 | 512 | **512** | **1024** | 2× |
| Threed | 768 | 268 | **192** | **460** | 1.67× |

### Hyperparameters (configurable qua CLI hoặc yaml)

| Param | Default | Tác động |
|---|---:|---|
| `local_detail_window_size` | 1 (sau user update) | Kích thước window spatial. 1 = không pool (mỗi token 1 group), 2 = 2×2 windows |
| `local_detail_temporal_window_size` | 1 | Window temporal. 1 = mỗi frame riêng |
| `content_ratio` (modality-specific) | 0.25 (img/vid), 0.35 (3D) | N_c = N × ratio |
| `detail_ratio` (key naming only) | 0.25 | Không ảnh hưởng số detail token thực |
| `local_detail_bias` (decoder) | 0.25 | Hệ số distance bias trong cross-attn. Lớn = ép detail mạnh hơn |
| `kv_pos_scale` (decoder, learnable) | init 0.1 | Trọng số position encoding cộng vào KV |
| `token_type_scale` (decoder, learnable) | init 0.1 | Trọng số token type embed cộng vào KV |

### Monitoring metrics

| Metric | Ý nghĩa | Target |
|---|---|---|
| `slot_diversity` | mean pairwise cos sim giữa các content slots | ≤ 0.5 (slots khác nhau) |
| `residual_ratio` | `‖R‖ / ‖x‖` | 0.3–0.5 (content giữ phần lớn signal) |
| `detail_token_count` | số detail token thực | = số window distinct |
| `detail_avg_window_tokens` | trung bình tokens per window | ≈ s_win² · t_win nếu density đều |

### Tại sao đổi từ slot pooler global → window pool cho detail?

| Khía cạnh | Slot pooler (cũ) | Window pool (mới) |
|---|---|---|
| Position info | ❌ (global) | ✅ (window center) |
| Compression | tốt (25 slots cho 256 token) | tốt (64 windows cho 256 token với win=2) |
| High-freq detail | hạn chế (slot abstract) | tốt (mean trong window nhỏ giữ texture) |
| Inductive bias | không có spatial prior | có (locality assumption) |
| Tham số | trainable slot params + 2 layer cross-attn | non-trainable scatter_add + 1 LayerNorm + 1 Linear |
| Compute | cao (slot attn O(N·N_d)) | thấp (scatter O(N)) |

→ Chuyển sang windowed pool cho detail là **lossless về expressive power** với inductive bias hợp lý cho high-freq, lại **rẻ hơn** về tham số/compute.

---

## 4. Curriculum 3 stage

| Stage | Modalities | SigLIP2 unfreeze | LR | Purpose |
|---|---|---|---:|---|
| 1 | image only | hoàn toàn frozen | 1e-4 | Học image tokenizer + distill semantic |
| 2 | image + video | last 4 blocks unfrozen | 5e-5 | Thêm video poolers, fine-tune backbone cho temporal |
| 3 | + threed | toàn bộ unfrozen | 2e-5 | Thêm 3D, polish toàn bộ |

Cross-stage transfer: `--model.init_from_ckpt <prev_stage_ckpt>` (strict=False, weights only). Lightning module `setup('fit')`:
1. `_prepare_cd_split_poolers()` — eagerly tạo content pooler cho mỗi modality active (bắt buộc trước `configure_optimizers`)
2. `_sync_ema_modalities()` — đồng bộ active_modalities từ DataModule vào EMA weighter
3. `load_siglip2_weights()` — load HF weights nếu `init_siglip2=true`
4. `_load_semantic_teacher()` — load frozen SigLIP2 vision tower nếu `use_semantic_distill=true`
5. `_load_weights_from_ckpt(init_from_ckpt)` — load prev-stage weights cuối cùng để override init

---

## 5. Latent space chính thức

Sau VAE bottleneck:
- Image: 128 token × 32 dim = **4096 floats** (vs raw 196,608 → 48× compression)
- Video: 1024 token × 32 dim = **32,768 floats** (vs raw 3,145,728 → 96× compression)
- Threed: 460 token × 32 dim = **14,720 floats** (vs raw 589,824 → 40× compression)

So với baselines (theo `results.md`):
- AToken-So/C Stage 1: (1, 16, 16) = 256 token × 32 ch = 8192 floats / image (chúng tôi 4096 — gấp đôi compression nhờ slot ratio 0.25)
- Cosmos-CI16×16: 256 × 16 = 4096 (same as ours per token volume, nhưng arch khác)

---

## 6. Reference files

| File | Content |
|---|---|
| `src/mavt/model/patchify.py` | Stage 1 |
| `src/mavt/model/backbone.py` | Stage 2 (Transformer + RGAT) |
| `src/mavt/model/content_detail_split.py` | Stage 3 ⭐ |
| `src/mavt/model/latent_heads.py` | Stage 4 (VAEHead) |
| `src/mavt/model/decoder.py` | Stage 5 (AsymmetricDecoder, UnderstandingDecoder, UnifiedDetailExpander) |
| `src/mavt/losses/losses.py` | Stage 6 (MAVTLoss, ModalityEMAWeighter, temporal_consistency_loss) |
| `src/mavt/training/lightning_module.py` | Curriculum, optimizer, logging |
| `src/mavt/model/mavt.py` | End-to-end MAVT module |
| `configs/model/mavt_base.yaml` | Hyperparams arch |
| `configs/train/universal_data/stage{1,2,3}_universal.yaml` | Stage curriculum |
