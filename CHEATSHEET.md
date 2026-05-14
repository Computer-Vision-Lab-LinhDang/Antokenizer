# Cheatsheet — Lệnh thường dùng

> Reference cho cluster DGX-A100 (defq partition), training MAVT/AnTokenizer.

---

## 1. Slurm — Quản lý job

### Submit job mới
```bash
sbatch train_stage1.sh                          # train stage 1 (default config)
sbatch train_stage2.sh                          # train stage 2
sbatch --gres=gpu:1 --cpus-per-task=12 --mem=60G --time=14-00:00:00 \
       --job-name=mavt --output=logs/holder_%j.log \
       --wrap='echo "holder up"; sleep infinity'   # holder để giữ GPU slot
```

### Xem queue
```bash
squeue -u "$USER"                               # tất cả job của mình
squeue -u "$USER" -o "%.8i %.10j %.8T %.10M %R"   # định dạng gọn
squeue -j 8581                                  # 1 job cụ thể
squeue -j 8581 -s                               # xem STEPS trong job (.batch, .0, .1, ...)
sinfo -N -O "NodeList:10,Gres:25,GresUsed:25,StateLong:12"   # GPU usage cluster
scontrol show node dgx01                        # chi tiết node
scontrol show job 8581                          # chi tiết job (priority, StartTime, TRES)
```

### Cancel
```bash
scancel 8581                                    # cancel toàn bộ job
scancel 8581.5                                  # cancel CHỈ step .5 (giữ holder .batch)
# ⚠️ KHÔNG dùng `scancel --batch 8581` để gửi signal — nó kill cả job
```

### Lịch sử
```bash
sacct -u "$USER" --starttime=now-1day --format=JobID,JobName,State,ExitCode,Start,End,Elapsed -X
sacct -j 8581 --format=JobID,State,ExitCode,Reason,Elapsed
```

---

## 2. srun --overlap — Chạy thêm step trong job đang sống

> Pattern: giữ holder job qua đêm, srun các process khác vào để dùng GPU mà không xếp hàng lại.

### Test connectivity
```bash
srun --mpi=none --overlap --jobid=8581 hostname
srun --mpi=none --overlap --jobid=8581 nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Interactive shell trên GPU node
```bash
srun --mpi=none --overlap --jobid=8581 --pty bash
# bên trong: cd /home/.../Antokenizer; source .venv/bin/activate; ...
```

### Launch script trong job
```bash
srun --mpi=none --overlap --jobid=8581 \
     --output=logs/<name>.log --error=logs/<name>.err \
     bash -c "
set -euo pipefail
cd /home/user02/linhdang/Antokenizer
source .venv/bin/activate
export PYTHONPATH=\$PWD/src
python train.py fit --config configs/model/mavt_base.yaml ...
"
```

### Swap training trong cùng job (không mất allocation)
```bash
# 1. Đảm bảo có holder step (sleep infinity) trong job
# 2. Kill training step hiện tại
scancel <JOB>.<STEP>              # KHÔNG scancel cả <JOB>

# 3. Launch training mới qua srun --overlap (như ví dụ trên)
```

**Lưu ý**: srun --overlap có thể bị Slurm rate-limit khi step hiện tại nặng. Đợi 1-3 phút.

---

## 3. Training (Lightning CLI)

### Stage 1 (image only)
```bash
PYTHONPATH=src .venv/bin/python train.py fit \
    --config configs/model/mavt_base.yaml \
    --config configs/train/universal_data/stage1_universal.yaml \
    --config configs/train/universal_data/stage1_3_paths.yaml \
    --data.universal_data_root dataset/image10k/train \
    --data.active_modalities '["image"]' \
    --model.active_modalities '["image"]' \
    --data.batch_size 64 --data.num_workers 8 \
    --data.persistent_workers true --data.prefetch_factor 4 \
    --model.training_stage 1 --model.use_lpips true \
    --model.use_gradient_checkpointing false \
    --model.local_detail_window_size 2 \
    --trainer.devices 1 --trainer.precision bf16-mixed \
    --trainer.max_steps 200000 --trainer.val_check_interval 2000 \
    --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
    --trainer.logger.init_args.project mavt \
    --trainer.logger.init_args.name stage1_run
```

### Stage 2 (image + video, init from stage 1 ckpt)
```bash
CKPT="checkpoints/stage1_5/balanced/.../loss=0.1238.ckpt"

python train.py fit \
    --config configs/model/mavt_base.yaml \
    --config configs/train/universal_data/stage2_universal.yaml \
    --config configs/train/universal_data/stage2_3_paths.yaml \
    --data.image_shards_dir $PWD/dataset/image10k/train \
    --data.video_shards_dir $PWD/dataset/dataset_10m \
    --data.video_max_shards 50 \
    --data.active_modalities '["image", "video"]' \
    --model.active_modalities '["image", "video"]' \
    --data.batch_size 8 --data.num_workers 4 \
    --model.training_stage 2 \
    --model.local_detail_window_size 2 \
    --model.w_temp 0.0 \
    --model.init_from_ckpt "$CKPT" \
    --trainer.accumulate_grad_batches 1 \
    --trainer.val_check_interval 1000 \
    --trainer.logger.init_args.name stage2_run
```

### Override key flags
| Flag | Tác động |
|---|---|
| `--data.batch_size N` | batch (effective × accum) |
| `--data.num_workers N` | dataloader workers (≥8 cần `mp.set_sharing_strategy('file_system')` trong train.py) |
| `--model.local_detail_window_size N` | 1 = không pool detail, 2 = 2×2 window pool (nhỏ 4× detail tokens) |
| `--model.use_gradient_checkpointing true/false` | true = tiết kiệm VRAM, chậm 25% |
| `--model.w_temp 0.1` | temporal consistency loss (chỉ video). Past run: 0.1 đã gây diverge → start 0 |
| `--trainer.accumulate_grad_batches N` | effective batch = batch×N. N=1 chạy nhanh hơn 8× về optim steps |
| `--trainer.val_check_interval N` | val mỗi N steps (mặc định 2000) |
| `--trainer.max_steps N` | giới hạn tổng steps |
| `--trainer.devices N` | số GPU (auto = all) |

---

## 4. Eval

### Image eval (PSNR/SSIM/LPIPS/FID + understanding cos_sim teacher)
```bash
PYTHONPATH=src .venv/bin/python eval_image.py \
    --ckpt checkpoints/stage1_5/balanced/.../loss=0.1238.ckpt \
    --image_shards_dir dataset/image10k/train \
    --max_images 1024 --batch_size 16 --num_workers 4 \
    --output results/eval_image_<name>.json
```

### Video eval (PSNR/SSIM/LPIPS per-frame + rFVD clip-level)
```bash
PYTHONPATH=src .venv/bin/python eval_video.py \
    --ckpt checkpoints/stage2_3/balanced/.../loss=0.1044.ckpt \
    --video_shards_dir dataset/dataset_10m \
    --max_videos 256 --max_shards 4 \
    --batch_size 4 --num_workers 2 \
    --output results/eval_video_<name>.json
```

### Infer demo (1 image + 1 video → save PNG strips + JSON)
```bash
PYTHONPATH=src .venv/bin/python infer_demo.py \
    --ckpt <ckpt> \
    --image_shards_dir dataset/image10k/train \
    --video_shards_dir dataset/dataset_10m \
    --video_max_shards 2 \
    --outdir results/infer_demo_<name>
```

---

## 5. Monitoring training qua wandb

### CLI query (không cần wandb dashboard)
```bash
.venv/bin/python -c "
import wandb
api = wandb.Api(timeout=60)
r = api.run('banalaxis93/mavt/<RUN_ID>')
print('state:', r.state, 'runtime:', r.summary.get('_runtime', 0)/3600, 'h')
for k in ['trainer/global_step','train/loss_step','val/loss_step','val/loss_recon_step','val/loss_lpips_step']:
    v = r.summary.get(k)
    if v is not None: print(f'  {k}: {v:.4f}' if isinstance(v,(int,float)) else f'  {k}: {v}')
"
```

### List recent runs
```bash
.venv/bin/python -c "
import wandb
api = wandb.Api()
for r in list(api.runs('banalaxis93/mavt', order='-created_at'))[:8]:
    print(f'{r.id} {r.state:>9} {r.name}  step={r.summary.get(\"trainer/global_step\",\"?\")}')
"
```

### Loss curve theo bucket steps
```bash
.venv/bin/python << 'PY'
import wandb
api = wandb.Api(timeout=60)
r = api.run('banalaxis93/mavt/<RUN_ID>')
tr = [h for h in r.scan_history(page_size=2000)
      if h.get('trainer/global_step') is not None and h.get('train/loss_step') is not None]
buckets = {}
for h in tr:
    b = int(h['trainer/global_step'] // 5000) * 5000
    buckets.setdefault(b, []).append(h)
print(f"{'step':>6} {'total':>7} {'l1':>7} {'lpips':>7} {'sem':>7}")
for b in sorted(buckets):
    vs = buckets[b]
    avg = lambda k: sum((v.get(k) or 0) for v in vs)/len(vs)
    print(f"{b:>6} {avg('train/loss_step'):>7.4f} {avg('train/loss_l1_step'):>7.4f} {avg('train/loss_lpips_step'):>7.4f} {avg('train/loss_sem_step'):>7.4f}")
PY
```

### Theo dõi log file
```bash
tail -f logs/stage1_<job>.log                   # stdout
tail -f logs/stage1_<job>.err                   # stderr
```

---

## 6. Checkpoint inspection

### List ckpts theo thời gian
```bash
find checkpoints/stage1_5 -name '*.ckpt' -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort -r | head -10
```

### Đọc hparams trong ckpt
```bash
.venv/bin/python -c "
import torch
ckpt = torch.load('<ckpt>', map_location='cpu', weights_only=False)
hp = ckpt.get('hyper_parameters', {})
for k in ['embed_dim','num_heads','t_patch','training_stage','w_lpips','w_sem','w_temp','local_detail_window_size']:
    print(f'{k}: {hp.get(k)}')
"
```

### Liệt kê content/detail poolers trong ckpt
```bash
.venv/bin/python -c "
import torch
ckpt = torch.load('<ckpt>', map_location='cpu', weights_only=False)
keys = set()
for k in ckpt['state_dict'].keys():
    if 'cd_split._content_poolers' in k:
        shape = k.split('.')[3]
        keys.add(shape)
print('content pooler keys:', sorted(keys))
"
```

---

## 7. Env / network

### Activate env
```bash
source /home/user02/linhdang/Antokenizer/.venv/bin/activate    # .venv (PyTorch 2.7.1+cu118)
# Hoặc conda base:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
```

### Network test
```bash
ping -c 3 8.8.8.8                               # basic internet
curl -sI -m 8 https://download.pytorch.org/     # PyTorch CDN
curl -o /dev/null -w 'speed=%{speed_download} B/s\n' \
    https://download.pytorch.org/models/r3d_18-b3b3357e.pth
```

### Resume download (slow link)
```bash
for i in 1 2 3 4 5; do
    timeout 120 curl -sSL --retry 5 -C - -o <dest> <url>
    sz=$(stat -c%s <dest> 2>/dev/null || echo 0)
    [ "$sz" -ge <expected_size> ] && break
done
```

---

## 8. Hay quên / pitfalls

### Slurm
- `scancel --batch <JOB>` **giết cả job**, không gửi signal — dùng `scancel <JOB>.<STEP>` cho từng step
- `srun --overlap` cần đợi Slurm step creation rate-limit (1-3 phút khi job đang nặng)
- Holder pattern: `sbatch --wrap='sleep infinity'` để giữ slot, srun --overlap launch real work vào

### Training
- `--data.num_workers >= 8` cần `mp.set_sharing_strategy('file_system')` trong train.py — không thì FD exhaustion sau vài giờ
- Stage 2 với `w_temp=0.1` từng diverge → start `w_temp=0.0`, bật sau khi val ổn định
- `local_detail_window_size=1` cho video → 2560 latent tokens → có thể OOM, cần gradient_checkpointing + batch nhỏ
- `init_from_ckpt` strict=False — shape mismatch (embed_dim, num_heads khác) sẽ silent drop. Verify hparams ckpt trước
- ckpt dir trùng tên có thể bị overwrite — dùng `stage1_2_paths.yaml`, `stage1_3_paths.yaml` ... cho từng run

### Eval
- `eval_image.py` cần `--semantic` (default) để load SigLIP2 teacher (~370MB)
- `eval_video.py` rFVD dùng `r3d_18` Kinetics — KHÔNG so trực tiếp được với paper rFVD I3D
- Stage 1 ckpt eval video sẽ ra rác (video pooler random init)

### Wandb
- `trainer/global_step` trong summary đôi khi bị wandb _step counter ghi đè — không tin số đó. Dùng `scan_history` lấy max.
- Val rows log `trainer/global_step` cùng tên nhưng giá trị khác → bucket theo từng row.

---

## 9. Path quan trọng

| Loại | Path |
|---|---|
| Project root | `/home/user02/linhdang/Antokenizer` |
| Code chính | `src/mavt/{model,training,data,losses}/` |
| Config | `configs/model/mavt_base.yaml`, `configs/train/universal_data/stage{1,2,3}_universal.yaml` + `stage*_paths.yaml` overrides |
| Stage 1 ckpts | `checkpoints/stage1_3/`, `stage1_5/` (cd-split + decoder v2) |
| Stage 2 ckpts | `checkpoints/stage2_3/` |
| Dataset image | `dataset/image10k/train/*.tar` (609 WDS shards) |
| Dataset video | `dataset/dataset_10m/NNNNN/<id>.{mp4,json,txt}` (~1800 shard dirs) |
| Logs | `logs/<run_name>.log` + `.err` + `wandb/run-<date>-<id>/` |
| Eval results | `results/<eval_name>/{eval_image.json, eval_video.json, summary.json, *.png}` |
| Docs | `MODEL.md`, `results.md`, `CHEATSHEET.md` (this file), `logs/training_report_*.md` |
