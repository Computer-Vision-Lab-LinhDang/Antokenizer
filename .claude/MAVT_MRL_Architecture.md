# MAVT + Matryoshka Representation Learning — Architecture Spec

Áp dụng **Matryoshka Representation Learning (MRL, Kusupati et al. 2022)** vào MAVT: sau khối Content-Detail Split, mỗi token `(B, N, D=1152)` được cắt prefix theo trục channel thành các nested dims `{8, 64, 256, 512, 1152}`. Mỗi prefix đi qua **2 nhánh song song** — Reconstruction và Understanding — và được supervise đồng thời. Inference có thể chọn `d_k` tuỳ ngân sách mà không cần retrain.

---

## Pipeline tổng quan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT (modality-specific)                           │
│   image: (B,3,H,W)   video: (B,3,T,H,W)   threed: (B,3planes,3,S,S)         │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
        ┌──────────────────────────▼──────────────────────────┐
        │  STAGE 1 — PatchifyEncoder      (patchify.py)        │
        │  • Conv3d(3→D, k=t_p×p×p, s=t_p×p×p)  unified        │
        │  • + FourDPositionEmbedding(t,x,y,z)                 │
        │  output: tokens (B,N,D=1152), positions (N,4),       │
        │          plane_ids (N,)                              │
        └──────────────────────────┬───────────────────────────┘
                                   │
        ┌──────────────────────────▼───────────────────────────┐
        │  STAGE 2 — HybridBackbone   (12 blocks)              │
        │  Transformer (×10) + RGAT4DBlock @ idx {4,8}         │
        │  output: features (B, N, D=1152)                     │
        └──────────────────────────┬───────────────────────────┘
                                   │
        ┌──────────────────────────▼───────────────────────────┐
        │  STAGE 3 — ContentDetailSplit                        │
        │       C = ContentSlotPooler(features) (B,N_c,D)      │
        │       R = features − softmax(C·xᵀ/√D)·C              │
        │       D = DetailSlotPooler(R)         (B,N_d,D)      │
        │       compressed = concat(C, D)       (B, N, D=1152) │
        │       N = N_c + N_d                                  │
        └──────────────────────────┬───────────────────────────┘
                                   │  (B, N, D=1152)
        ┌──────────────────────────▼───────────────────────────┐
        │  STAGE 4 — MATRYOSHKA HEAD (per-token nested dims)   │
        │                                                      │
        │   For each token t_i ∈ R^1152, expose nested prefixes│
        │   t_i[:d_k]  với  d_k ∈ {8, 64, 256, 512, 1152}      │
        │                                                      │
        │   ┌───────────────────────────────────────────────┐  │
        │   │ slice 0 :  z⁽⁸⁾    = compressed[..., :8   ]   │  │
        │   │ slice 1 :  z⁽⁶⁴⁾   = compressed[..., :64  ]   │  │
        │   │ slice 2 :  z⁽²⁵⁶⁾  = compressed[..., :256 ]   │  │
        │   │ slice 3 :  z⁽⁵¹²⁾  = compressed[..., :512 ]   │  │
        │   │ slice 4 :  z⁽¹¹⁵²⁾ = compressed[..., :1152]   │  │
        │   └───────────────────────────────────────────────┘  │
        │                                                      │
        │   Tổng loss = Σ_k α_k · L_task( head_k(z⁽ᵈₖ⁾) )      │
        │   (mỗi prefix phải tự đủ tốt cho cả recon &          │
        │    understanding → buộc backbone học nested feature) │
        └────────────┬─────────────────────────┬───────────────┘
                     │                         │
        ┌────────────▼────────────────┐   ┌────▼─────────────────────────────┐
        │  BRANCH A — RECONSTRUCTION  │   │  BRANCH B — UNDERSTANDING        │
        │                             │   │                                  │
        │  Per-prefix d_k:            │   │  Per-prefix d_k:                 │
        │                             │   │                                  │
        │  ┌─────────────────────┐    │   │   tokens_k = compressed[...,:dk] │
        │  │ VAEHead_k           │    │   │       │                          │
        │  │ Linear(d_k → 2·L)   │    │   │       ▼                          │
        │  │ → (μ_k, logσ²_k)    │    │   │  Attention Pool                  │
        │  │ z_k = μ_k + σ_k·ε   │    │   │   • 1 learnable query q_k        │
        │  │ loss_kl_k           │    │   │   • MHAttn(q_k, kv=tokens_k)     │
        │  └──────────┬──────────┘    │   │       │                          │
        │             │ (B, N, L)     │   │       ▼ (B, d_k)                 │
        │             ▼               │   │  LayerNorm                       │
        │  AsymmetricDecoder_k        │   │       │                          │
        │   (UnifiedDetailExpander    │   │       ▼   global vector g_k      │
        │    → 4× SelfAttn            │   │                                  │
        │    → PixelShuffleCNN ×16)   │   │  ┌──────────────────────────┐    │
        │             │               │   │  │ Multi-task heads on g_k  │    │
        │             ▼               │   │  │  • cls_head    → logits  │    │
        │   reconstruction (pixel)    │   │  │  • retr_head   → embed.  │    │
        │   per modality:             │   │  │  • cap_head    → tokens  │    │
        │     image  : (B,3,H,W)      │   │  │  • (sem distill SigLIP)  │    │
        │     video  : (B,3,Tp,H,W)   │   │  └──────────────────────────┘    │
        │     threed : (B,3,3,H,W)    │   │                                  │
        └─────────────┬───────────────┘   └────────────────┬─────────────────┘
                      │                                    │
                      ▼                                    ▼
            { recon_k }_{k∈K}                    { logits_k, emb_k, cap_k }_{k∈K}
```

---

## Chi tiết các stage

### Stage 1–3: giữ nguyên kiến trúc cũ
- **Patchify (`patchify.py`)** — `Conv3d` unified cho cả 3 modality, cộng `FourDPositionEmbedding(t,x,y,z)`.
- **HybridBackbone (`backbone.py`)** — 12 blocks, RGAT4D ở idx `{4, 8}`, còn lại là `StandardTransformerBlock` (SigLIP2-init).
- **ContentDetailSplit (`content_detail_split.py`)** — slot-attention pooler tách `C` và `D` từ residual `R`. Output `(B, N, D=1152)` với `N = N_c + N_d`.

### Stage 4 — Matryoshka Head (mới)

Cắt prefix trực tiếp trên `compressed`, **không** cần Linear projection cho mỗi `d_k` (đặc trưng cốt lõi của MRL):

```python
K = (8, 64, 256, 512, 1152)
slices = {d_k: compressed[..., :d_k] for d_k in K}   # nested prefixes
```

- Mỗi `slice` có shape `(B, N, d_k)` và **chia sẻ cùng tokens** — chỉ khác về số channel exposed.
- Backbone không cần biết về MRL; gradient từ tất cả `d_k` đẩy thông tin quan trọng nhất về các channel đầu của `D`.

### Stage 5a — Reconstruction Branch (per prefix `d_k`)

```
tokens_k (B, N, d_k)
   │
   ▼
VAEHead_k:    Linear(d_k → 2·latent_dim)   → (μ_k, logσ²_k)
              z_k = μ_k + σ_k · ε          (latent_dim = 32)
              loss_kl_k = β_k · KL(N(μ_k,σ²_k) ∥ N(0,1))
   │
   ▼ z_k (B, N, latent_dim=32)
AsymmetricDecoder_k:
   • UnifiedDetailExpander: cross-attn từ FourDQueryEncoding(target_pos) vào z_k
   • 4× StandardTransformerBlock (dec_dim=768)
   • Reshape (B, dec_dim, H_grid, W_grid)
   • PixelShuffleCNNDecoder: 768 → 512 → 256 → 128 → 3, ×16 spatial
   │
   ▼
recon_k:
   image  : (B, 3, H, W)
   video  : (B, 3, Tp, H, W)
   threed : (B, 3, 3, H, W)
```

**Tuỳ chọn capacity:**
- **Per-prefix VAE, shared decoder** *(khuyến nghị)*: `VAEHead_k` riêng (Linear in-dim khác nhau), **`AsymmetricDecoder` dùng chung** cho mọi prefix vì input của decoder đã được normalize về `latent_dim=32`.
- **Per-prefix VAE + decoder**: nếu muốn tách hoàn toàn — tốn ≈5× tham số decoder.

### Stage 5b — Understanding Branch (per prefix `d_k`)

```
tokens_k (B, N, d_k)
   │
   ▼
Attention Pool_k:
   • Learnable query  q_k ∈ R^{1×d_k}
   • MultiheadAttention(query=q_k, key=tokens_k, value=tokens_k)
   │
   ▼ pooled_k (B, 1, d_k)
LayerNorm_k → squeeze → g_k ∈ R^{d_k}     ← GLOBAL REPRESENTATION
   │
   ▼
Multi-task heads (mỗi head riêng cho từng prefix d_k):
   • cls_head_k    : Linear(d_k → num_classes)         → CE loss
   • retr_head_k   : Linear(d_k → retr_dim) + L2-norm  → InfoNCE loss
   • cap_head_k    : Linear(d_k → vocab) hoặc autoregressive decoder
   • sem_head_k    : Linear(d_k → 768)                 → cosine distill SigLIP2
```

`g_k` là vector toàn cục dùng cho downstream tasks. **Không** đi qua VAE bottleneck → semantic pathway tách khỏi reconstruction, tránh mất thông tin do KL.

---

## Tổng loss

```
L_total = Σ_{k ∈ K}  α_k · [
              w_l1   · L1(recon_k, x)
            + w_lpips · LPIPS(recon_k, x)        // chỉ cho image / per-frame video
            + w_kl   · KL_k                       // β_k = w_kl · (d_k / D_max)
            + w_cls  · CE(cls_k,    y_class)
            + w_retr · InfoNCE(retr_k, retr_pos)
            + w_cap  · CE(cap_k,    y_caption)
            + w_sem  · (1 − cos(sem_k, SigLIP2_teacher(x)))
          ]
        + w_aux · slot_diversity                  // monitoring, áp 1 lần
```

### Trọng số Matryoshka `α_k`

Default — nested-weighted, prefix lớn ưu tiên hơn nhưng prefix nhỏ vẫn nặng đủ để buộc compactness:

| `d_k`  | `α_k` (default) |
|--------|-----------------|
| 8      | 1.0             |
| 64     | 1.0             |
| 256    | 1.0             |
| 512    | 1.0             |
| 1152   | 1.0             |

Có thể thử *log-uniform* hoặc *uniform* — Kusupati et al. cho thấy uniform thường đủ.

### Trọng số per-modality
`ModalityEMAWeighter` (giữ nguyên từ kiến trúc cũ) áp lên `(w_l1·L1 + w_lpips·LPIPS)` *trước khi* nhân với `α_k`.

### KL scaling theo prefix
`β_k = w_kl · (d_k / 1152)`. Lý do: prefix nhỏ vốn đã rất compact → KL loss với cùng `kl_weight` sẽ áp đặt mạnh hơn so với capacity → cần down-scale để tránh posterior collapse.

---

## Kỹ thuật triển khai

### Thay đổi cụ thể trên codebase

| File | Thay đổi |
|------|----------|
| `src/mavt/model/mavt.py` | Bỏ gọi `vae_head` / `understanding_decoder` đơn lẻ; gọi `MatryoshkaHead(compressed, modality, …)` trả về dict per-prefix outputs. |
| `src/mavt/model/matryoshka_head.py` *(mới)* | `class MatryoshkaHead(nn.Module)` chứa `nn.ModuleDict` các `VAEHead_k`, pool-query `q_k`, multi-task heads. |
| `src/mavt/model/decoder.py` | `AsymmetricDecoder` không đổi (input vẫn là `z` 32-d). Loại bỏ class `UnderstandingDecoder` (thay bằng pool trong MRL head). |
| `src/mavt/losses/losses.py` | `MAVTLoss.forward` loop qua `K`, accumulate `α_k · L_task_k`. |
| `src/mavt/training/lightning_module.py` | `setup('fit')` pre-build mọi `VAEHead_k` & pool query để optimizer pick up. |
| `configs/model/mavt_base.yaml` | Thêm `matryoshka_dims: [8, 64, 256, 512, 1152]`, `matryoshka_weights: [1,1,1,1,1]`. |

### Pseudocode `MatryoshkaHead`

```python
class MatryoshkaHead(nn.Module):
    def __init__(self, dims=(8,64,256,512,1152), latent_dim=32,
                 num_classes=None, retr_dim=512, vocab_size=None,
                 semantic_dim=768, num_heads=8):
        super().__init__()
        self.dims = tuple(dims)

        # Per-prefix recon
        self.vae_heads = nn.ModuleDict({
            str(d): VAEHead(in_dim=d, latent_dim=latent_dim)
            for d in dims
        })

        # Per-prefix understanding pool
        self.pool_queries = nn.ParameterDict({
            str(d): nn.Parameter(torch.randn(1, 1, d) * d**-0.5)
            for d in dims
        })
        self.pool_attn = nn.ModuleDict({
            str(d): nn.MultiheadAttention(d, num_heads=min(num_heads, d),
                                          batch_first=True)
            for d in dims
        })
        self.pool_norm = nn.ModuleDict({
            str(d): nn.LayerNorm(d) for d in dims
        })

        # Per-prefix downstream heads (đặt None nếu không dùng)
        self.cls_heads  = nn.ModuleDict({str(d): nn.Linear(d, num_classes)  for d in dims}) if num_classes else None
        self.retr_heads = nn.ModuleDict({str(d): nn.Linear(d, retr_dim)     for d in dims})
        self.sem_heads  = nn.ModuleDict({str(d): nn.Linear(d, semantic_dim) for d in dims})
        # cap_heads: tuỳ chọn, có thể là 1 Transformer decoder share weight + per-prefix in-proj

    def forward(self, compressed):                      # (B, N, D_max)
        out = {}
        for d in self.dims:
            tokens_k = compressed[..., :d]              # nested prefix
            # --- Recon branch ---
            z_k, mu, logvar, kl = self.vae_heads[str(d)](tokens_k)
            # --- Understanding branch ---
            B = tokens_k.shape[0]
            q = self.pool_queries[str(d)].expand(B, -1, -1)
            pooled, _ = self.pool_attn[str(d)](q, tokens_k, tokens_k)
            g = self.pool_norm[str(d)](pooled.squeeze(1))   # (B, d)

            out[d] = {
                'z': z_k, 'mu': mu, 'logvar': logvar, 'kl': kl,
                'g': g,
                'cls':  self.cls_heads[str(d)](g)  if self.cls_heads  else None,
                'retr': F.normalize(self.retr_heads[str(d)](g), dim=-1),
                'sem':  self.sem_heads[str(d)](g),
            }
        return out
```

### Tích hợp với decoder
`AsymmetricDecoder` có thể được **share** giữa các prefix vì input là `z_k ∈ R^{latent_dim=32}` (đã normalize qua `VAEHead_k`). Loop trong `MAVT.forward`:

```python
mrl_out = self.matryoshka_head(compressed)
recon_per_dim = {}
for d in self.matryoshka_head.dims:
    recon_per_dim[d] = self.decoder(mrl_out[d]['z'], positions, modality, grid_shape)
```

---

## Ưu điểm của thiết kế

1. **Inference flexibility** — chọn `d_k` theo ngân sách (latency / storage / bandwidth) mà **không retrain**. Ví dụ: classification trên edge có thể dùng `d=64`; retrieval server dùng `d=1152`.
2. **Tách biệt recon vs understanding** — semantic không bị "nén ép" qua VAE bottleneck, trong khi recon vẫn hưởng lợi từ `latent_dim=32` cho hiệu quả lưu trữ token.
3. **Backbone học nested representations** — channel đầu chứa thông tin global / abstract; channel sau bổ sung detail. Phù hợp với linh cảm thông tin trong slot-attention output `[C; D]`.
4. **Multi-task supervision đồng thời** — backbone nhận gradient từ cả recon, classification, retrieval, captioning, semantic distill → representation tổng quát hơn.
5. **Backward-compatible với code cũ** — Stage 1–3 không đổi, chỉ thay phần head; có thể giữ nhánh `decode=False` cho encoder-only inference.

---

## Pitfalls & cảnh báo

- **Optimizer registration.** Mọi `VAEHead_k`, `pool_query_k`, `head_k` phải được build trong `__init__` (hoặc qua `prepare_for_modalities`) **trước** `configure_optimizers`. Lười tạo trong `forward` sẽ dẫn tới params không nằm trong param-groups (giống bug slot pooler hiện tại).
- **KL scaling.** Áp `β_k = w_kl · (d_k / D_max)` để tránh prefix nhỏ bị posterior collapse.
- **`num_heads` cho MultiheadAttention.** Khi `d_k=8`, `num_heads=8` chỉ có 1 dim/head. Dùng `num_heads = min(8, d_k // head_dim_min)` hoặc fallback `num_heads=1` cho prefix nhỏ.
- **Cost của recon nhiều prefix.** 5 prefix × decoder pass = 5× compute decoder. Có thể train với *random-prefix sampling* mỗi step (chỉ 1 trong 5 active) để giảm cost — Kusupati et al. gọi là "MRL-E" (efficient).
- **Gradient interference.** Prefix nhỏ và prefix lớn đôi khi cạnh tranh. Theo dõi `loss_recon_k` và `loss_sem_k` riêng cho mỗi `k` trong WandB.
- **Slot pooler vẫn áp dụng cho `D=1152`.** Slot pooler không bị ảnh hưởng — MRL chỉ cắt sau khi đã pool xong.
- **Nếu drop `latent_dim=32` để recon thẳng từ `d_k`**, sẽ phá tính chất "per-token VAE latent 32-d" của tokenizer gốc — không khuyến nghị.

---

## Chiến lược curriculum (đề xuất)

| Stage | Modality | Trọng số `α_k` | Active prefixes |
|-------|----------|----------------|-----------------|
| 1 | image | uniform | tất cả `K` |
| 2 | + video | uniform | tất cả `K` |
| 3 | + threed | tăng dần `α_k` cho `d_k` lớn | tất cả `K` |
| 3+ (fine-tune) | per-task | chỉ task-specific `α_k` | tuỳ task |

Có thể warm-up bằng cách chỉ active `d_k ∈ {1152}` ở 5K step đầu rồi mở dần các prefix nhỏ — giúp backbone ổn định trước khi bị MRL ép.

---

## Inference

```python
# Lúc inference, chọn d_k theo ngân sách:
out = model(x, modality)
g_64 = out.understanding[64]['g']      # global vector 64-d cho retrieval edge
recon = out.reconstruction[1152]       # full-fidelity recon
```

Một backbone — N prefix — N điểm hoạt động trên đường cong accuracy/cost. Đó là toàn bộ tinh thần của MRL.
