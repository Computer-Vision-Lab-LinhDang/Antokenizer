# Stage 2 Training Branch — Phân tích vấn đề

*Branch: `rgat-cdsplit` · Commit head: `6368dfb` (cd-split) · Working tree có uncommitted changes*

## Mục lục
1. Cấu trúc branch & các thay đổi gần đây
2. **🔴 Critical bugs (root cause của các vấn đề đã thấy)**
3. ⚠️ Cấu hình rủi ro / suboptimal
4. 🟡 Drift giữa config & runtime
5. Khuyến nghị fix theo độ ưu tiên

---

## 1. Cấu trúc branch

Branch `rgat-cdsplit` là biến thể của `main` với thay đổi lớn:
- Refactor toàn bộ thư mục `train/` → `src/mavt/training/` (Lightning-based).
- Thêm `ContentDetailSplit` (cd-split) module + `UnifiedDetailExpander` decoder.
- Thêm temporal loss (`w_temp`).

10 commit gần nhất:
```
6368dfb cd-split                      ← HEAD
42c622b update
9fd10da hot fix video loader
c202c1e add temporal loss
576b193 fix loss activate modalities
65b57e9 change val for video
5e0dd75 fix ema
d1794b6 slot evict
06d2cd6 update lpips
c9c1d5d Merge pull request #5
```

Working tree có **modifications chưa commit** ở 6 file (`eval_image.py`, `datamodule.py`, `content_detail_split.py`, `decoder.py`, `mavt.py`, `lightning_module.py`, `train.py`) — đa số là follow-up fixes cho cd-split commit.

---

## 2. 🔴 Critical bugs

### 2.1 Mất ~105 000 training steps vì checkpoint config (`save_top_k=1` + không `save_last`)

**File**: `configs/train/universal_data/stage2_3_paths.yaml` (và `stage2_universal.yaml`)

3 callbacks ModelCheckpoint đều dùng:
```yaml
save_top_k: 1
monitor: val/loss     # hoặc val/loss_recon, val/loss_sem
mode: min
every_n_train_steps: 5000
# ❌ KHÔNG có save_last
# ❌ KHÔNG có save_top_k=-1 (giữ tất cả)
```

**Hậu quả thực tế**: Run `mivnmjrd` chạy tới step 120 269, nhưng ckpt cuối cùng được lưu chỉ là **step=15000 (val/loss=0.2223)**. Khi run bị cancel, **không thể resume** trừ khi resume từ stage1_5 init hoặc ckpt step=15000 → mất 105k steps GPU (~18h).

**Nguyên nhân chuỗi**:
1. `save_top_k=1` + `monitor=val/loss` (epoch-aggregated): callback **chỉ save khi val/loss epoch cải thiện**.
2. Cross-run state: `ModelCheckpoint` reset `best_metric=+inf` khi run bắt đầu → ckpt đầu tiên luôn được save. Ckpt sau chỉ save nếu val/loss thấp hơn ckpt đầu.
3. Stage 2 có video noise nhiều → val/loss epoch dao động, có thể không bao giờ thấp hơn baseline step=15000.
4. Không có `save_last=True` → khi job bị kill, không có "last known good" để resume.

**Bằng chứng**: 
- `checkpoints/stage2_3/balanced/` chỉ có `step=0000125` (May 11, run cũ) và `step=0015000` (May 13, run mới).
- Wandb summary cho thấy val/loss_step=0.199 ở step 120k → có cải thiện theo step nhưng val/loss EPOCH không match.

### 2.2 Race giữa wandb daemon và process training

**Symptom**: Wandb run `mivnmjrd` last update 2026-05-12 21:47, nhưng training thực sự chạy tới ~14:48 ngày 13/05 (job 8581 cancel time). → **17 giờ training không log lên wandb**.

**Nguyên nhân**: 
- File `wandb-debug.log` cho thấy filestream sync vẫn đang chạy đến 21:47.
- Sau đó wandb daemon im lặng → có thể wandb subprocess bị OOM-killed (heavy log volume với 60 modules in eval mode + per-modality logging × val_check_interval=2000).
- Process training tiếp tục chạy nhưng metrics không upload.

### 2.3 `init_from_ckpt` không restore optimizer state → "soft resume" cold start

**File**: `src/mavt/training/lightning_module.py` `_load_weights_from_ckpt()`

```python
def _load_weights_from_ckpt(self, path: str) -> None:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    missing, unexpected = self.load_state_dict(sd, strict=False)
    # ❌ KHÔNG restore: optimizer state, scheduler state, current_step, EMA state
```

**Hậu quả khi resume Stage 2 từ Stage 1 ckpt**:
- Optimizer state Adam (momentum + variance) = zero → bước đầu rất noisy.
- LR scheduler restart từ 0 → re-warmup 500 steps ở LR cực thấp → loss tăng đột ngột.
- EMA modality weights (`ema_image`, `ema_video`, `ema_threed`) trong `MAVTLoss` reset → loss weighting sai vài trăm steps đầu.

Đây là intended design cho cross-stage transfer, nhưng phải lưu ý: **NÊN dùng `--ckpt_path` cho cùng-stage resume** (Lightning native full restore), chỉ dùng `init_from_ckpt` khi transfer giữa các stage.

### 2.4 Silent corrupt-data skipping → effective batch size bất định

`src/mavt/data/datasets.py` `ShardVideoDataset` không có integrity check trước khi return → khi PyAV gặp `moov atom not found`, dataloader có thể return tensor zeros / partial / raise → DataLoader có thể skip hoặc batch lỗi.

51.9% video bị hỏng (đã scan): mỗi epoch effective batch size phụ thuộc shard nào đang đọc → loss noise tăng → val/loss không stable → checkpoint không save (cf. 2.1).

---

## 3. ⚠️ Cấu hình rủi ro / suboptimal

### 3.1 `find_unused_parameters=true` trong DDPStrategy
`configs/train/universal_data/stage2_universal.yaml`:
```yaml
strategy:
  init_args:
    find_unused_parameters: true   # slow (~20-30% overhead)
    gradient_as_bucket_view: true
```
Stage 2 active modalities = `[image, video]`, nhưng `cd_split` pool video pooler nếu video không xuất hiện trong batch hiện tại → có params unused. `find_unused_parameters=true` cần nhưng nên cố reduce. Hiện chạy single-GPU → không ảnh hưởng (DDP không activate), nhưng nếu mở rộng multi-GPU sẽ chậm.

### 3.2 `w_temp` không khớp giữa config và run
- `stage2_universal.yaml`: `w_temp: 0.1`
- Wandb run `mivnmjrd` config: `w_temp: 0` 
→ Có người override khi launch nhưng commit message `c202c1e` là "add temporal loss". **Temporal loss đã bị tắt trong run thực tế** → mất hiệu lực fix temporal consistency cho video.

### 3.3 `accumulate_grad_batches=2` + `batch_size=8` = effective batch 16
- Effective batch = 16 (image) hoặc 16 (video × 16 frames = 256 frames/optim step).
- Stage 1 dùng batch=64 → effective batch giảm 4× ở Stage 2 → cập nhật optim mỗi step chứa ít thông tin hơn → cần nhiều steps hơn để converge.

### 3.4 `val_check_interval=2000` cộng `every_n_train_steps=5000` mismatch
- Val chạy mỗi 2000 steps (step 2000, 4000, ...).
- Checkpoint save check mỗi 5000 steps (5000, 10000, 15000, ...).
- Tại step 5000, val gần nhất ở step 4000 → metric stale 1000 steps. Chấp nhận được, nhưng tạo **gap 4000 → 6000 = 2000 steps mới có val tiếp theo** → có thể bỏ lỡ điểm tối ưu cục bộ.

### 3.5 `weights_only=False` khi load ckpt
`_load_weights_from_ckpt`:
```python
ckpt = torch.load(path, map_location='cpu', weights_only=False)
```
Không security-critical ở đây (mình tự tạo ckpt), nhưng anti-pattern. Lightning warning sẽ trigger trong torch >= 2.6.

### 3.6 `train_stage2.sh` có cấu hình outdated
```bash
STAGE1_CKPT="checkpoints/stage1/last.ckpt"       # ❌ Không tồn tại
--data.batch_size 16                              # ❌ Thực tế chạy với 8
--data.video_max_shards 50                        # ⚠️ Chỉ dùng 50/615 shards
```
Run thực tế không launch từ script này (job 8581 dùng holder + interactive srun) → script lỗi thời.

### 3.7 `video_max_shards=50` chỉ dùng 50/615 shards
Train với ~50 000 video thay vì ~600 000 → kém đa dạng, val/train split khả năng overlap nếu shuffle cùng seed.

---

## 4. 🟡 Drift giữa code, config, runtime

| Tham số | Default (code) | Config yaml | Runtime (wandb) |
|---|---|---|---|
| `local_detail_window_size` | 1 (working tree) | — | **2** |
| `local_detail_temporal_window_size` | 2 (mavt.py) / 1 (lightning_module.py) | — | **1** |
| `embed_dim` | 768 (cd_split default, đã edit) | — | **768** |
| `w_temp` | 0.0 | **0.1** | 0 |
| `batch_size` | 8 (datamodule) | 16 (yaml) | 8 |
| `init_siglip2` | True | False (train_stage2.sh) | True |

→ Có **mâu thuẫn `local_detail_temporal_window_size`**: `mavt.py` default = 2 nhưng `lightning_module.py` default = 1. Lightning module truyền giá trị của nó xuống MAVT → final value **= 1** (khớp runtime).

→ Có `data.active_modalities` qua command line `'["image", "video"]'` — phải parse JSON string. `train.py` cần xử lý đúng.

---

## 5. Khuyến nghị fix (ưu tiên giảm dần)

### P0 — Phải fix ngay trước khi resume training
1. **Thêm `save_last: true` + `save_top_k: -1`** (hoặc ≥5) vào ModelCheckpoint:
   ```yaml
   - class_path: lightning.pytorch.callbacks.ModelCheckpoint
     init_args:
       dirpath: checkpoints/stage2_3/last
       filename: mavt-stage2_3-last-{step:07d}
       save_top_k: -1                # giữ tất cả
       every_n_train_steps: 5000
       save_last: true
   ```
2. **Wipe video bị hỏng + re-download** (đang chạy).
3. **Restore torch cu118** (đã bị downgrade do cài v2d).

### P1 — Fix tuần này
4. Đồng bộ default `local_detail_temporal_window_size` giữa `mavt.py` (=2) và `lightning_module.py` (=1). Quyết định giá trị canonical → fix cả 2.
5. Re-enable temporal loss: launch run với `--model.w_temp 0.1` (hoặc lower 0.05 nếu sợ unstable).
6. Trong `_load_weights_from_ckpt`, log rõ "optimizer/scheduler will be reset" + suggest dùng `--ckpt_path` cho same-stage resume.
7. Tăng `video_max_shards` từ 50 lên 615 (hoặc bỏ giới hạn) sau khi dataset sạch.
8. Update `train_stage2.sh`: ckpt path đúng, batch_size match thực tế, document holder-job pattern.

### P2 — Cải thiện
9. Add wandb resilience: log `os.kill(wandb_pid)` warning nếu daemon dies. Hoặc dùng `WANDB_MODE=offline` + sync sau.
10. Add `EarlyStopping` callback (patience=20 val cycles) để tránh waste GPU khi diverge.
11. Filter corrupt video at dataset level (mtime + ffprobe quick check) để tránh silent skip.
12. Đổi `weights_only=False` → load với explicit allowlist hoặc khẳng định trust source bằng comment.
13. EMA modality state nên persist trong ckpt (đã có `ema_image/video/threed` buffers, OK nếu Lightning save state_dict đầy đủ).

### P3 — Nice to have
14. Document training launch command thực tế ở README (vì không qua sbatch).
15. Cleanup branch zoo: `rgat-fix`, `rgat-hihi`, `rgat-hotfix`, `rgat-merge`, `rgat-head`, `rgat-dgx`, `rgat-cdsplit` — quá nhiều branch song song dễ gây drift configs.

---

## Tóm tắt 1 dòng

> **Stage 2 đang chạy trên dataset 52% hỏng, với checkpoint config chỉ giữ `save_top_k=1` không `save_last`, dẫn đến 105k steps GPU bị mất khi job 8581 bị cancel. Trước khi re-launch, cần thêm `save_last: true`, fix corrupt videos, và restore torch cu118.**
