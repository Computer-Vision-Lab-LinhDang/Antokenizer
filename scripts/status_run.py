#!/usr/bin/env python3
"""One-line status of a run dir: last step + per-modality train losses + last val + ckpts + gates."""
import csv, glob, json, os, sys
run = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(run)
files = sorted(glob.glob(f"{run}/lightning_logs/*/metrics.csv"), key=os.path.getmtime)
out = []
if files:
    rows = list(csv.DictReader(open(files[-1])))
    def last(k):
        v = [r[k] for r in rows if r.get(k)]; return float(v[-1]) if v else None
    def lastrow(k):
        v = [r for r in rows if r.get(k)]; return v[-1] if v else None
    st = lastrow("train/loss_step")
    if st:
        parts = [f"step={st['step']}", f"loss={float(st['train/loss_step']):.3f}"]
        for name, key in (("img", "train/loss_image_step"), ("vid", "train/loss_video_step"), ("3d", "train/loss_threed_step"),
                          ("sem", "train/loss_sem_step"), ("temp", "train/loss_temp_step"), ("lr", "lr-AdamW")):
            v = last(key); parts.append(f"{name}={v:.3g}" if v is not None else f"{name}=-")
        out.append(" ".join(parts))
    vr = lastrow("val/loss_epoch")
    if vr:
        vp = [f"val@{vr['step']} loss={float(vr['val/loss_epoch']):.3f}"]
        for name, key in (("img", "val/loss_image_epoch"), ("vid", "val/loss_video_epoch"), ("3d", "val/loss_threed_epoch")):
            v = vr.get(key); vp.append(f"{name}={float(v):.3f}" if v else f"{name}=-")
        out.append(" ".join(vp))
log = f"{run}/train.log"
if os.path.exists(log):
    with open(log, "rb") as f:
        tail = f.read()[-400000:].decode("utf-8", "ignore").splitlines()
    err = [l for l in tail if ("Traceback" in l or "out of memory" in l or "_EXIT=" in l) and "Warning" not in l]
    out += [e[:200] for e in err[-2:]]
out.append(f"ckpts={len(glob.glob(f'{run}/checkpoints/*.ckpt'))}")
for j in sorted(glob.glob(f"runs/gates/deepseek_{tag}__*.json")):
    d = json.load(open(j)); s = d["student"]; step = j.split("step=")[1].split(".")[0].lstrip("0")
    out.append("GATE %s eff_rank=%.0f pair_cos=%.3f align=%.3f knn=%.3f psnr=%.2f lpips=%.3f pass=%s" % (
        step, s["eff_rank"], s["pair_cos"], s["align_cos_centered"], s["knn_top1"], s["psnr"], s["lpips"],
        ",".join(k for k, v in d["gates"].items() if v["pass"]) or "-"))
print("\n".join(out))
