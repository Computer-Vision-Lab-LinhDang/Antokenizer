# AToken Benchmark Tables: Image & Video Modalities

> **Nguồn:** Lu et al., *AToken: A Unified Tokenizer for Vision* (arXiv:2509.14476v2, Sep 2025)
> **Tổng hợp từ:** Table 4 (Image Recon), Table 5 (Image Und), Table 6 (Video Recon), Table 7 (Video Und)
> **Quy ước ký hiệu:** ↑ cao hơn tốt hơn; ↓ thấp hơn tốt hơn; — không có dữ liệu; vùng tô xám = discrete tokenizer

---

## 📊 Bảng 1 — Image Modality

### 1.1. Image Reconstruction (ImageNet-1K & COCO, 256×256)

| Method | Comp. Ratio | Latent Ch. | Token Type | ImageNet PSNR↑ | ImageNet SSIM↑ | ImageNet LPIPS↓ | ImageNet rFID↓ | COCO PSNR↑ | COCO SSIM↑ | COCO LPIPS↓ | COCO rFID↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **— Continuous Latent —** | | | | | | | | | | | |
| SD-VAE | (1, 8, 8) | 4 | VAE | 26.26 | 0.745 | 0.133 | 0.606 | 25.99 | 0.759 | 0.130 | 4.142 |
| SD3-VAE | (1, 8, 8) | 16 | VAE | 31.29 | 0.886 | 0.059 | 0.201 | 31.18 | 0.894 | 0.056 | 1.671 |
| FLUX.1 [dev] | (1, 8, 8) | 16 | VAE | 32.86 | 0.917 | 0.044 | **0.176** | 32.73 | 0.923 | 0.041 | **1.343** |
| Qwen-Image | (1, 8, 8) | 16 | VAE | 32.18 | 0.899 | 0.053 | 1.459 | 32.01 | 0.908 | 0.050 | 4.618 |
| Cosmos-0.1-CI8×8 | (1, 8, 8) | 16 | AE | 32.25 | 0.902 | 0.064 | 1.031 | 32.08 | 0.909 | 0.061 | 3.844 |
| Cosmos-0.1-CI16×16 | (1, 16, 16) | 16 | AE | 25.07 | 0.700 | 0.167 | 0.959 | 24.74 | 0.711 | 0.165 | 5.063 |
| VAVAE | (1, 16, 16) | 32 | VAE | 27.70 | 0.798 | 0.096 | 0.279 | 27.50 | 0.811 | 0.093 | 2.709 |
| OmniTokenizer | (4, 8, 8) | 8 | VAE | 26.74 | 0.824 | 0.101 | 1.023 | 26.44 | 0.833 | 0.099 | 4.687 |
| Hunyuan | (4, 8, 8) | 16 | VAE | **33.32** | 0.916 | 0.053 | 0.670 | **33.25** | 0.924 | 0.050 | 2.597 |
| Wan2.1 | (4, 8, 8) | 16 | VAE | 31.34 | 0.886 | 0.058 | 0.945 | 31.19 | 0.895 | 0.055 | 3.449 |
| Wan2.2 | (4, 16, 16) | 48 | VAE | 31.25 | 0.878 | 0.057 | 0.749 | 31.10 | 0.888 | 0.054 | 3.279 |
| **AToken-So/C — Stage 1** | (1, 16, 16) | 32 | VAE | 28.77 | 0.814 | 0.099 | 0.258 | 28.66 | 0.829 | 0.096 | 2.336 |
| **AToken-So/C — Stage 2** | (4, 16, 16) | 48 | VAE | 29.55 | 0.845 | 0.087 | 0.246 | 29.49 | 0.858 | 0.083 | 2.180 |
| **AToken-So/C — Stage 3** | (4, 16, 16) | 48 | VAE | 29.72 | 0.848 | 0.085 | 0.209 | 29.67 | 0.861 | 0.081 | 2.026 |
| **— Discrete Latent —** | | | | | | | | | | | |
| Cosmos-0.1-DI8×8 | (1, 8, 8) | 6 | FSQ | 25.87 | 0.750 | 0.155 | 0.867 | 25.54 | 0.760 | 0.153 | 5.016 |
| GigaTok-B-L | (1, 16, 16) | 8 | VQ | 21.87 | 0.591 | 0.200 | 0.507 | 21.42 | 0.596 | 0.202 | 5.565 |
| GigaTok-XL-XXL | (1, 16, 16) | 8 | VQ | 22.42 | 0.613 | 0.189 | 0.795 | 22.03 | 0.620 | 0.191 | 5.757 |
| VILA-U | (1, 16, 16) | 16 | RQ | 22.24 | 0.612 | 0.228 | 4.231 | 21.89 | 0.620 | 0.227 | 10.997 |
| UniTok | (1, 16, 16) | 64 | MCQ | 25.34 | 0.742 | 0.132 | 0.362 | 24.95 | 0.750 | 0.131 | 3.918 |
| OmniTokenizer | (4, 8, 8) | 8 | VQ | 24.69 | 0.771 | 0.138 | 1.411 | 24.31 | 0.779 | 0.137 | 6.292 |
| **AToken-So/D** | (4, 16, 16) | 48 | FSQ | **27.14** | **0.801** | **0.119** | **0.379** | **27.00** | **0.815** | **0.115** | **3.270** |

**Ghi chú:**
- Tất cả baselines được tác giả re-evaluate bằng official implementation, ảnh resize + center-crop về 256×256 → kết quả có thể khác paper gốc, nhưng đảm bảo so sánh công bằng.
- AToken-So/C đạt rFID 0.209 trên ImageNet với 16×16 compression — outperform mọi 16×16 baseline (VAVAE 0.279, Cosmos-CI16×16 0.959).
- Quan sát quan trọng: rFID cải thiện qua các stage (0.258 → 0.246 → 0.209), tức multimodal training **giúp** (chứ không tổn hại) image reconstruction.
- AToken-So/D là **discrete tokenizer 16×16 đầu tiên** đạt PSNR > 27 trên ImageNet, đánh bại UniTok (25.34) và GigaTok-XL-XXL (22.42).

---

### 1.2. Image Understanding (Zero-shot Classification & Cross-modal Retrieval)

| Resolution | Seq Len | Method | ImageNet val↑ | ImageNet v2↑ | COCO T→I↑ | COCO I→T↑ | Flickr T→I↑ | Flickr I→T↑ |
|---|---|---|---|---|---|---|---|---|
| **224** | **196** | CLIP | 68.3 | 61.9 | 33.1 | 52.4 | 62.1 | 81.9 |
| | | MetaCLIP | 72.4 | 65.1 | 48.9 | — | 77.1 | — |
| | | EVA-CLIP | 74.7 | 67.0 | 42.2 | 58.7 | 71.2 | 85.7 |
| | | DFN | 76.2 | 68.2 | 51.9 | — | 77.3 | — |
| **256** | **256** | SigLIP | 80.8 | 74.1 | 49.4 | 68.6 | 80.0 | 92.1 |
| | | SigLIP 2 | **83.4** | **77.8** | **55.4** | **71.5** | **84.4** | **94.2** |
| | | AToken-So/C — Stage 1 | 82.7 | 76.7 | 54.1 | 70.4 | 81.3 | 93.1 |
| | | AToken-So/C — Stage 2 | 82.3 | 76.4 | 53.8 | 70.6 | 80.7 | 93.0 |
| | | AToken-So/C — Stage 3 | 82.2 | 76.1 | 53.7 | 70.5 | 80.5 | 93.2 |
| | | AToken-So/D | 82.2 | 76.2 | 53.8 | 70.1 | 80.9 | **93.5** |
| **384** | **576** | SigLIP 2 | **84.1** | **78.4** | **56.0** | 71.2 | **85.3** | **95.9** |
| | | AToken-So/C — Stage 1 | 83.4 | 77.6 | 54.8 | 70.4 | 81.7 | 93.8 |
| | | AToken-So/C — Stage 2 | 82.9 | 77.1 | 54.7 | 71.1 | 81.9 | 93.9 |
| | | AToken-So/C — Stage 3 | 82.9 | 76.8 | 54.6 | **71.3** | 81.9 | 93.5 |
| | | AToken-So/D | 82.8 | 76.6 | 54.4 | 70.9 | 81.9 | 93.5 |
| **512** | **1024** | SigLIP 2 | **84.3** | **79.1** | **56.0** | **71.3** | **85.5** | **95.4** |
| | | AToken-So/C — Stage 1 | 83.5 | 77.8 | 54.7 | 71.1 | 82.1 | 94.1 |
| | | AToken-So/C — Stage 2 | 83.1 | 77.3 | 54.7 | 71.3 | 82.2 | 93.6 |
| | | AToken-So/C — Stage 3 | 82.9 | 77.2 | 54.7 | 71.1 | 82.3 | 93.6 |
| | | AToken-So/D | 82.9 | 77.0 | 54.7 | 71.2 | 82.3 | 93.5 |

**Ghi chú:**
- AToken cách SigLIP 2 (teacher) chỉ ~1.2% trên ImageNet ở res 256 (82.2% vs 83.4%) — thu hẹp đáng kể gap so với UniTok (78.6%) và VILA-U (78.0%).
- Discrete quantization (FSQ) **không** làm giảm semantic performance — AToken-So/D giữ nguyên 82.2% accuracy.
- Điểm đáng chú ý: degradation qua các stage rất nhỏ (Stage 1 → Stage 3: chỉ giảm 0.5%), chứng minh joint training không phá hủy semantic priors.

---

## 🎬 Bảng 2 — Video Modality

### 2.1. Video Reconstruction (DAVIS 1080p & TokenBench 720p)

| Method | Comp. Ratio | Latent Ch. | Token Type | DAVIS PSNR↑ | DAVIS SSIM↑ | DAVIS LPIPS↓ | DAVIS rFVD↓ | TokenBench PSNR↑ | TokenBench SSIM↑ | TokenBench LPIPS↓ | TokenBench rFVD↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **— Continuous Latent —** | | | | | | | | | | | |
| Cosmos-0.1-CV4×8×8 | (4, 8, 8) | 16 | AE | 32.25 | 0.894 | 0.219 | 19.15 | 34.33 | 0.924 | 0.155 | 8.34 |
| OmniTokenizer | (4, 8, 8) | 8 | VAE | 21.06 | 0.800 | 0.315 | 206.34 | 19.39 | 0.782 | 0.275 | 173.48 |
| Hunyuan | (4, 8, 8) | 16 | VAE | 32.33 | **0.907** | 0.194 | 22.94 | 36.37 | **0.944** | 0.129 | 3.78 |
| Wan2.1 | (4, 8, 8) | 16 | VAE | **33.50** | 0.884 | **0.164** | 17.75 | 36.11 | 0.940 | **0.128** | 3.21 |
| Wan2.2 | (4, 16, 16) | 48 | VAE | 33.06 | **0.907** | 0.184 | 12.65 | **36.39** | 0.942 | 0.126 | 3.19 |
| **AToken-So/C — Stage 2** | (4, 16, 16) | 48 | VAE | 32.29 | 0.902 | 0.196 | 13.50 | 35.63 | 0.937 | 0.139 | 3.63 |
| **AToken-So/C — Stage 3** | (4, 16, 16) | 48 | VAE | 33.11 | 0.907 | 0.189 | **10.76** | 36.07 | 0.940 | 0.135 | **3.01** |
| **— Discrete Latent —** | | | | | | | | | | | |
| OmniTokenizer | (4, 8, 8) | 8 | VQ | 20.62 | 0.770 | 0.346 | 240.20 | 19.89 | 0.787 | 0.293 | 202.46 |
| Cosmos-0.1-DV4×8×8 | (4, 8, 8) | 6 | FSQ | 27.26 | 0.798 | 0.310 | 110.33 | 31.20 | 0.892 | **0.190** | 25.94 |
| **AToken-So/D** | (4, 16, 16) | 48 | FSQ | **29.75** | **0.846** | **0.288** | **41.42** | **33.12** | **0.913** | 0.193 | **22.16** |

**Ghi chú:**
- AToken-So/C Stage 3 đạt **best rFVD trên cả 2 benchmark** (DAVIS 10.76, TokenBench 3.01) — vượt cả các tokenizer chuyên cho video như Wan2.2 (12.65, 3.19) và Hunyuan (22.94, 3.78).
- Cross-modal benefit rõ rệt: thêm 3D ở Stage 3 → TokenBench PSNR tăng 35.63 → 36.07 (+0.44).
- AToken-So/D **chiến thắng tuyệt đối** ở mảng discrete video tokenization: PSNR 29.75 vs Cosmos-DV 27.26 trên DAVIS, vượt hơn 2 dB.
- OmniTokenizer là cảnh báo cho transformer-based video tokenizers — ~21 PSNR cho thấy adversarial training instability nghiêm trọng.

---

### 2.2. Video Understanding — Zero-shot Text-Video Retrieval (MSR-VTT 1K-A & MSVD)

| Method | Res. | MSRVTT T→V R@1↑ | MSRVTT T→V R@5↑ | MSRVTT T→V R@10↑ | MSRVTT V→T R@1↑ | MSRVTT V→T R@5↑ | MSRVTT V→T R@10↑ | MSVD T→V R@1↑ | MSVD T→V R@5↑ | MSVD T→V R@10↑ | MSVD V→T R@1↑ | MSVD V→T R@5↑ | MSVD V→T R@10↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP-ViT-B/32 | 224 | 31.2 | 53.7 | 63.3 | 26.4 | 49.9 | 61.7 | 36.4 | 63.3 | 73.1 | 57.8 | 84.1 | 90.7 |
| SigLIP2-So400m | 256 | 41.9 | 66.3 | 75.7 | 32.4 | 55.4 | 65.9 | **55.5** | **81.2** | 87.8 | 72.7 | 91.7 | 96.1 |
| VideoPrism-g | 288 | **52.7** | **77.2** | — | **51.7** | **75.2** | — | — | — | — | — | — | — |
| PE-Core-B16 | 224 | 45.8 | 70.1 | 78.1 | 45.5 | 70.9 | 80.0 | 48.7 | 75.5 | 84.1 | 79.1 | 96.7 | 98.8 |
| PE-Core-L14 | 336 | 49.1 | 73.3 | **81.6** | 50.9 | 74.4 | **82.7** | 54.4 | **81.2** | **88.4** | **82.5** | **98.2** | **99.4** |
| AToken-So/C — Stage 1 | 224 | 40.8 | 65.3 | 75.2 | 31.0 | 55.0 | 63.7 | 53.9 | 79.9 | 87.3 | 72.4 | 93.0 | 95.4 |
| AToken-So/C — Stage 2 | 224 | 40.1 | 64.9 | 75.2 | 30.9 | 53.7 | 64.0 | 53.4 | 79.6 | 87.1 | 71.6 | 91.9 | 95.5 |
| AToken-So/C — Stage 3 | 224 | 40.2 | 64.9 | 75.2 | 30.5 | 53.1 | 63.2 | 53.5 | 79.5 | 87.1 | 72.4 | 91.6 | 95.4 |
| AToken-So/D | 224 | 40.3 | 65.0 | 74.6 | 30.3 | 51.8 | 61.7 | 53.8 | 79.7 | 87.2 | 71.5 | 91.8 | 95.2 |

**Ghi chú:**
- VideoPrism-g (chuyên video, không tái tạo) thắng tuyệt đối ở MSRVTT — cho thấy **cái giá** của joint reconstruction + understanding training.
- AToken giữ R@1 ~40% trên MSRVTT — gần SigLIP2 (41.9%), nhỉnh hơn CLIP (31.2%) đáng kể, nhưng kém PE-Core và VideoPrism vốn được train với video-text data quy mô lớn.
- Trên MSVD, AToken (~53.5% T→V R@1) competitive với PE-Core-L14 (54.4%) và chỉ kém SigLIP2 ~2%.
- Hạn chế đáng chú ý theo paper: data video-text trong training của AToken nhỏ hơn nhiều so với các video-only encoder → gap có thể đóng được bằng cách scale data.

---

## 🎯 Tổng kết quan trọng cho MAVT/AnTokenizer

**Image — AToken-So/C ở Stage 3 đặt baseline:**
- Reconstruction: **rFID 0.209** trên ImageNet (16×16, 48-ch latent)
- Understanding: **82.2%** zero-shot ImageNet @ 256px
- Discrete: AToken-So/D đạt **27.14 PSNR + 82.2% acc** — best discrete unified tokenizer ở thời điểm này

**Video — AToken-So/C ở Stage 3 đặt baseline:**
- Reconstruction: **PSNR 33.11 / rFVD 10.76** trên DAVIS, **36.07 / 3.01** trên TokenBench
- Understanding: **40.2% MSRVTT R@1**, **53.5% MSVD R@1**
- Discrete: **29.75 PSNR DAVIS** — duy nhất ở segment này

**Để vượt AToken trên benchmark P0:**
- Image rFID 16×16: cần đạt < 0.209 (gating threshold)
- Video rFVD TokenBench 720p: cần đạt < 3.01
- Image zero-shot acc: cần đạt ≥ 82.2% với 16× compression
- MSRVTT T→V R@1: cần đạt ≥ 40.2%

---

## 🧪 AnTokenizer (Ours) — Stage 1 cd-split @ step 40k

> **Ckpt:** `checkpoints/stage1_3/balanced/mavt-stage1_3-balanced-step=0040000-val/loss=0.1962.ckpt`
> **Arch:** embed_dim=768, num_heads=16, 12 blocks, latent_dim=32, patch=16, t_patch=2
> **C-D Split:** local detail pooling, window=2 (default), N_c=64 + N_d_local=64 = **128 tokens** cho image 256² (compression 1×16×16, 32 ch latent — tương đương AToken-So/C Stage 1 setting)
> **Train state:** Stage 1 only image, step 40k / 200k (~20%). Distillation từ SigLIP2-base-patch16-224.
> **Eval set:** 1024 images sample từ `dataset/image10k/train` (WDS .tar shards) — NOT ImageNet/COCO
> **Eval files:** `results/infer_demo_stage1_3/`

### Image Reconstruction (eval_image.py, 1024 samples @ 256×256, batch=16, bf16)

| Metric | **Ours @ 40k** | AToken-So/C Stage 1 (ImageNet ref) | Gap |
|---|---:|---:|---|
| PSNR ↑ | **17.41** | 28.77 | -11.4 |
| SSIM ↑ | **0.483** | 0.814 | -0.331 |
| LPIPS (AlexNet) ↓ | **0.462** | 0.099 | +0.363 |
| FID (Inception-2048) ↓ | **168.66** | 0.258 (rFID) | rất xa |
| cos_sim teacher ↑ | **0.815** | — (no direct equivalent) | proxy understanding |

**Ghi chú** (số quá thấp so với AToken):
1. **Dữ liệu**: chỉ 10K image (image10k) vs ImageNet-1K (1.28M) → train set quá nhỏ
2. **Steps**: mới 40k/200k (chưa converge)
3. **Stage 1 only**: AToken Stage 1 cũng chỉ image; joint training Stage 2/3 cải thiện thêm (rFID 0.258 → 0.246 → 0.209)
4. **FID metric**: AToken `rFID` paper-specific impl; ta dùng `torchmetrics.FrechetInceptionDistance` Inception-V3 pool3 — cùng spirit, absolute value KHÔNG so trực tiếp
5. cos_sim teacher = 0.81 → distillation đang work (matches training `loss_sem=0.18 ≈ 1 - 0.82`)

### Image Understanding (proxy: cosine similarity to SigLIP2 teacher)

| Method | Eval set | Metric | Value |
|---|---|---|---|
| **Ours @ 40k** | 1024 image10k samples | mean cos_sim_teacher | **0.815** |
| SigLIP2 (teacher) | — | self-cos | 1.000 |

→ Distillation fidelity ~0.81 (target 0.99 để competitive). Cần tăng training steps + data hoặc `w_sem`.
→ Để eval ImageNet zero-shot acc cần thêm linear probe / CLIP-style text encoder pipeline (chưa có trong eval_image.py).

### Video Reconstruction & Understanding

⚠️ **Stage 1 ckpt KHÔNG có video pooler trained** (`active_modalities=['image']` → chỉ tạo `_content_poolers.64_64`). Khi forward video, lazy-create pooler `512_512` với random init → recon là noise.

**Single-sample infer demo (`Fun clown - 3d animation`):**

| Metric | Value | Note |
|---|---:|---|
| Input shape | (1, 3, 16, 256, 256) | 16 frames @ 256² |
| Recon shape | (1, 3, 8, 256, 256) | Tp = T/t_patch = 8 |
| PSNR (single) | 7.57 | random output |
| L1 (single) | 0.268 | random output |
| cos_sim teacher (mid-frame) | 0.674 | non-zero — backbone vẫn extract semantic; VAE-bottleneck path bị nhiễu vì content slot không thấy video distribution |

**Video benchmark** (DAVIS / TokenBench): **N/A** — cần stage 2+ để có video pooler trained.

### Demo files (`results/infer_demo_stage1_3/`)

| File | Mô tả |
|---|---|
| `image_input.png` | GT image (lady on swim ring float) |
| `image_recon.png` | Recon — blurry, color/structure preserved |
| `image_side_by_side.png` | GT \| recon side-by-side |
| `video_input_strip.png` | GT 8 frames evenly sampled (clown 3D animation) |
| `video_gt_subsampled_strip.png` | GT subsampled to 8 frames (target Tp) |
| `video_recon_strip.png` | Recon — random noise pattern (random pooler) |
| `summary.json` | Per-sample metrics + paths |
| `eval_image.json` | Full 1024-sample eval output |

### Tóm tắt diagnostic

| Component | State | Diag |
|---|---|---|
| Image recon (l1, lpips) | training, slope ~-0.001/1k | val/recon=0.143, val/lpips=0.305 (training metrics) |
| Image semantic | training, cos_sim 0.81 / target 0.99 | val/sem=0.20 ≈ 1 - cos_sim |
| Image structural (SSIM) | 0.48 — chưa tốt | cần thêm step + data |
| Video (any) | random init | chờ stage 2 |

**Đường đi tiếp:**
1. Stage 1 đến step ~80-100k để val/lpips ~0.25
2. Stage 2 init từ stage1 step ≥80k (cùng arch: embed_dim=768, num_heads=16, win=2)
3. Eval trên benchmark chuẩn (cần download ImageNet val + DAVIS để so apples-to-apples)
4. Cân nhắc tăng `w_sem` weight để pull cos_sim_teacher từ 0.81 → 0.95+