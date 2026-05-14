# MAVT — Báo cáo kết quả training Stage 1 & Stage 2

*Tổng hợp: 2026-05-13 · Project: `banalaxis93/mavt` (wandb) · Branch: `rgat-cdsplit`*

---

## 1. Tổng quan model & pipeline

- Kiến trúc: **MAVT** (Multimodal Audio-Visual Tokenizer) — VAE-style với content/detail split, decoder dùng triplane, semantic distillation từ SigLIP2.
- Tham số: **210 M (model) + 14.7 M (loss) + 92.9 M (SigLIP teacher, frozen)** → 167 M trainable, 150 M non-trainable.
- Pipeline 3 giai đoạn:
  - **Stage 1** — image-only, học content/detail VAE + semantic distillation. Khoá pooler video.
  - **Stage 2** — bật thêm video stream, init từ best Stage 1 ckpt. Train cả image + video.
  - **Stage 3** — multimodal / 3D triplane (chưa chạy).
- Loss: `loss = w_l1·L1 + w_lpips·LPIPS + w_kl·KL + w_sem·CosSimDistill (+ w_temp·temporal — tắt)`.

---

## 2. Stage 1 — Image tokenizer

### 2.1 Best run cuối cùng (sử dụng init cho Stage 2)

| Mục | Giá trị |
|---|---|
| Wandb run | `stage1_cdsplit_v2_win1` (id `w9g5hf2j`) |
| Created | 2026-05-07 03:55 UTC |
| Runtime | 64.3 h |
| Status | `crashed` ở step 144 449 / 200 000 (~72 %) |
| Batch size | 64 (image only) |
| Loss weights | `w_l1=1, w_lpips=0.25, w_sem=0.3, w_kl=1, w_temp=0` |
| Best ckpt (dùng làm init Stage 2) | `checkpoints/stage1_5/balanced/mavt-stage1_5-balanced-step=0120000-val/loss=0.1238.ckpt` |

### 2.2 Loss tại lần log cuối

| Split | total | recon | sem (cos-dist) | lpips | kl |
|---|---|---|---|---|---|
| train | **0.111** | 0.0755 | 0.1157 | 0.1245 | 6.1e-4 |
| val (image) | **0.126** | 0.0817 | 0.1470 | 0.1406 | — |

### 2.3 Eval metrics (image, n=512–1024 samples từ `image10k`)

| Ckpt | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | cos_sim_teacher ↑ |
|---|---|---|---|---|---|
| stage1 step=15000 (sớm) | 23.57 | 0.683 | 0.305 | 71.4 | — |
| stage1_3 step=40000 (mid) | 17.41 | 0.483 | 0.462 | 168.7 | 0.815 |
| **stage1_5 step=120000** (final) ¹ | **28.41** | **0.857** | **0.122** | **26.66** | **0.874** |

¹ Eval của ckpt này được chạy gián tiếp qua "stage2_3 step=125" — về bản chất là stage1_5 ckpt vừa load vào trainer stage2 (chưa update gradient), nên các con số đại diện cho Stage 1 cuối cùng.

### 2.4 Lịch sử ckpt (recon-loss trên val) — chọn lọc

| Run | Step | val/loss | val/loss_recon | val/loss_sem |
|---|---|---|---|---|
| stage1 (raw) | 5 000 | 1.016 | 0.732 | 0.520 |
| stage1 | 30 000 | 0.369 | 0.106 | 0.211 |
| stage1_2 | 60 000 | 0.194 | 0.138 | 0.186 |
| stage1_3 | 40 000 | 0.196 | 0.143 | 0.175 |
| stage1_4 | 140 000 | 0.121 | 0.078 | — |
| **stage1_5** | **120 000** | **0.124** | **0.079** | **0.147** |

### 2.5 Demo định tính

- Input: 1 ảnh "maillot" (256×256).
- Reconstruction (stage1_5): PSNR_single = **31.81**, L1 = **0.0156**, cos_sim_teacher = **0.868**.
- Demo trước (stage1_3 step=40000): PSNR_single = 17.77 → cải thiện rõ rệt theo training.
- File: `results/eval_stage2_decoder_v2_step125/image_side_by_side.png` (effectively stage1_5).

---

## 3. Stage 2 — Image + Video

### 3.1 Lịch sử các attempt

| Run id | Tên | Init từ | Steps đạt | Status | batch | Ghi chú |
|---|---|---|---|---|---|---|
| `0os87ltw` | v3-prop_ema_no_temp | stage1 step=15000 (loss 0.232) | 40 977 / 200 000 | crashed | 16 | val/loss=0.546 — diverge video |
| `r86y0kkw` | decoder_v2 (attempt 1) | stage1_5 step=120000 | 460 687 epoch-step (logging quirk) | crashed | 4 | val/loss=**0.174** — tốt nhất |
| **`mivnmjrd`** | **decoder_v2_w2 (latest)** | stage1_5 step=120000 | **120 269 / 200 000** | **crashed/cancelled** | 8 | val/loss=0.199 — đang chạy thì job 8581 bị cancel |

### 3.2 Cấu hình run mới nhất `mivnmjrd`

- Created 2026-05-12 08:31 UTC, runtime **23.3 h**, **cancelled lúc 2026-05-13 14:48** (job 8581 hết slot / bị cancel).
- `init_from_ckpt = checkpoints/stage1_5/.../step=0120000-val/loss=0.1238.ckpt`
- `active_modalities=['image','video']`, `video_frames=16`, `video_resolution=256`, `t_patch=2` → recon Tp=8.
- Loss weights: `w_l1=1, w_lpips=0.3, w_sem=0.3, w_kl=1, w_temp=0`.
- Total steps target: 200 000 (chưa đạt).

### 3.3 Loss cuối cùng (`mivnmjrd`, step 120 269)

| Split | total | recon | sem | lpips | kl |
|---|---|---|---|---|---|
| train | **0.223** | 0.168 | 0.167 | 0.311 | 4.5e-4 |
| val/image | 0.199 | 0.149 | — | — | — |
| val/video | 0.227 | 0.170 | — | — | — |

→ Loss video cao hơn image ~14 % (recon) → video pooler vẫn đang học.

### 3.4 Eval metrics (ckpt stage2_3 step=125 ≈ stage1_5 init, chưa cập nhật stage2)

> Note: chưa có eval của ckpt stage2 sau khi train. Mới chỉ eval ckpt step=125 (~ngay sau init).

| Modal | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID/rFVD ↓ | cos_sim_teacher ↑ |
|---|---|---|---|---|---|---|
| **Image (256²)** | 512 | **28.41** | **0.857** | **0.122** | FID **26.66** | **0.874** |
| **Video (16f, 256²)** | 128 | 15.88 | 0.598 | 0.377 | rFVD **111.5** | 0.721 (midframe) |

Demo định tính (cùng ckpt step=125):
- Image: PSNR_single=31.81, L1=0.0156, cos_sim=0.868.
- Video ("Fun clown - 3d animation"): PSNR_single=**5.11** — *(caveat từ infer log: "stage1 ckpt has no trained video pooler — recon is roughly random")*.

### 3.5 So sánh so với baseline trước

| Ckpt | image PSNR | image SSIM | image LPIPS | video PSNR | video LPIPS | rFVD |
|---|---|---|---|---|---|---|
| stage1 step=15000 | 23.57 | 0.683 | 0.305 | — | — | — |
| stage2_recon step=10000 (cũ, từ stage1 step=15000) | — | — | — | **15.97** | 0.630 | 191.4 |
| **stage1_5 / stage2 init (step=125)** | **28.41** | **0.857** | **0.122** | **15.88** | **0.377** | **111.5** |

→ Việc thay init từ stage1 step=15000 → stage1_5 step=120000 cải thiện rõ: image PSNR +4.8 dB, LPIPS giảm ~60 %; video LPIPS giảm ~40 %, rFVD giảm ~42 %.

---

## 4. Vấn đề & rủi ro hiện tại

### 4.1 Stage 2 dừng đột ngột
- Job 8581 (`mavt`) **CANCELLED** lúc 2026-05-13 14:48 — không rõ do timeout hay cancel manual; ExitCode 0:0.
- Wandb run `mivnmjrd` last update 2026-05-12 21:47 (wandb daemon chết trước job).
- **Không có ckpt stage2 step >15000** trong `checkpoints/stage2_3/` → mất ~105 000 steps training (≈ 18 giờ GPU).

### 4.2 Dataset video bị hỏng nặng
- Scan `dataset/dataset_10m/` (614 478 MP4): **318 923 file hỏng (51.9 %)**.
  - `invalid_data` (moov atom missing / corrupt): 311 509.
  - `ffmpeg: End of file` (truncated): 7 414.
- 199 shards có ≥ 900/1000 file lỗi (đã wipe 2026-05-13, đang re-download).
- Toàn bộ 615 shards đều có ít nhất 1 file hỏng.
- Loader đang silently skip → batch_size effective dao động, ảnh hưởng độ ổn định loss.

### 4.3 Môi trường training bị break
- Tạm thời: torch trong `.venv` đã bị downgrade `2.7.x+cu118 → 2.1.2+cu121` khi cài `video2dataset`. Driver NVIDIA 470.141 (CUDA 11.4) **không chạy được cu121**.
- Phải reinstall torch trước khi resume training:
  ```bash
  VIRTUAL_ENV=$PWD/.venv uv pip install \
      torch==2.7.0 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu118
  ```

---

## 5. Plan tiếp theo (theo thứ tự)

1. **Hoàn tất re-download WebVid** (đang chạy, PID 2876388) → 3576 CSV partitions → ~hàng triệu MP4.
2. **Re-scan integrity** sau download (`scripts/scan_broken_videos.py`).
3. **Khôi phục torch cu118** trong `.venv`.
4. **Resume Stage 2** từ checkpoint stage1_5 step=120000 (vì stage2_3 step=15000 đã save nhưng đang nghi ngờ tính nhất quán) hoặc từ stage2_3 step=15000 nếu xác nhận healthy.
5. **Hoàn tất Stage 2** (mục tiêu 200 000 steps) → eval đầy đủ image + video bằng `eval_image.py` / `eval_video.py`.
6. **Stage 3** — chưa bắt đầu (chỉ có thư mục rỗng `checkpoints/stage3/`).

---

## 6. Tham chiếu file & nguồn

| Loại | Đường dẫn |
|---|---|
| Wandb runs | `https://wandb.ai/banalaxis93/mavt` |
| Eval JSON (stage2 init) | `results/eval_stage2_decoder_v2_step125/eval_image.json`, `eval_video.json`, `summary.json` |
| Eval JSON (stage1_3 demo) | `results/infer_demo_stage1_3/eval_image.json`, `summary.json` |
| Eval JSON (early stage1/stage2) | `logs/eval_image_8581.json`, `logs/eval_s2_v4.json` |
| Training logs | `logs/stage1_cdsplit_v2_8573.{log,err}`, `logs/stage2_decoder_v2_8581.{log,err}` |
| Best stage1 ckpt | `checkpoints/stage1_5/balanced/mavt-stage1_5-balanced-step=0120000-val/loss=0.1238.ckpt` |
| Best stage2 ckpt (so far) | `checkpoints/stage2_3/balanced/mavt-stage2_3-balanced-step=0015000-val/loss=0.2223.ckpt` |
| Broken videos list | `logs/broken_videos.txt`, `logs/broken_videos.summary.json` |
