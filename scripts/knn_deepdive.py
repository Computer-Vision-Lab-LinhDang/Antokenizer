"""Tách hai giả thuyết về kNN: (A) protocol/low-res, (B) tín hiệu teacher — vs (C) thiếu step.
Đo: (1) kNN in-distribution theo checkpoint, (2) sweep độ phân giải student vs teacher, (3) noise floor."""
import sys, inspect, math, torch, torchvision, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT
from mavt.evaluation.gates import knn_top1
dev = torch.device("cuda")

def build(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); hp = ck["hyper_parameters"]
    sig = inspect.signature(MAVT.__init__).parameters
    m = MAVT(**{k: v for k, v in hp.items() if k in sig})
    m.prepare_for_modalities([{"modality": "image", "resolution": 256}])
    m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True)
    return m.to(dev).eval()

from transformers import AutoModel
sig2 = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384")
T = sig2.vision_model.to(dev).eval(); TS = int(sig2.config.vision_config.image_size)

tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(256),
                                     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5]*3, [0.5]*3)])
def woof(split, n, seed=0):
    ds = torchvision.datasets.ImageFolder(f"data/eval/imagewoof2-320/{split}", transform=tf)
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
tr, te = woof("train", 3000), woof("val", 1500)

@torch.no_grad()
def feats(model, ds, res=None, teacher=False):
    E, Y = [], []
    for x, y in DataLoader(ds, batch_size=48, num_workers=6):
        x = x.to(dev)
        if res and res != 256:                       # hạ rồi nâng lại: mô phỏng ảnh low-res
            x = F.interpolate(F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False),
                              size=(256, 256), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = T(pixel_values=F.interpolate(x, size=(TS, TS), mode="bilinear", align_corners=False)).pooler_output if teacher \
                else model(x, "image", decode=False).semantic
        E.append(e.float().cpu()); Y.append(y)
    return torch.cat(E), torch.cat(Y)

CKPTS = [("s3@%dk" % (s // 1000), f"runs/deepseek/s3/checkpoints/mavt-v3-s3-step={s:07d}.ckpt") for s in (1000, 5000, 10000, 15000)]

print("### 1. kNN in-distribution (Imagewoof @256, 3000/1500) theo checkpoint — gate CIFAR để so sánh")
print("%-8s %10s %10s" % ("ckpt", "woof kNN", "CIFAR gate"))
cifar_gate = {"s3@1k": .336, "s3@5k": .358, "s3@10k": .387, "s3@15k": .378}
woof_trend = {}
for name, p in CKPTS:
    m = build(p)
    a, ya = feats(m, tr); b, yb = feats(m, te)
    k = knn_top1(a, ya, b, yb); woof_trend[name] = k
    print("%-8s %10.3f %10.3f" % (name, k, cifar_gate[name]))
    del m; torch.cuda.empty_cache()

print("\n### 2. Sweep độ phân giải trên s3@15k — student vs teacher (cùng ảnh, chỉ đổi độ nét đầu vào)")
m = build(CKPTS[-1][1])
print("%-14s %10s %10s %10s" % ("input res", "student", "teacher", "gap"))
for res in (32, 64, 128, 256):
    a, ya = feats(m, tr, res); b, yb = feats(m, te, res); ks = knn_top1(a, ya, b, yb)
    ta, _ = feats(None, tr, res, teacher=True); tb, _ = feats(None, te, res, teacher=True); kt = knn_top1(ta, ya, tb, yb)
    print("%-14s %10.3f %10.3f %10.3f" % (f"{res}→256", ks, kt, kt - ks))

print("\n### 3. Noise floor của phép đo kNN")
n_test = 1500
for p in (0.38, 0.72):
    print(f"  p={p:.2f}, n_test={n_test}: 1σ = {math.sqrt(p*(1-p)/n_test):.4f}  → 2σ = ±{2*math.sqrt(p*(1-p)/n_test):.3f}")
print(f"  gate CIFAR dùng n_test=1000 → 1σ ≈ {math.sqrt(.38*.62/1000):.4f}; dao động quan sát 0.336–0.391 = {(0.391-0.336)/math.sqrt(.38*.62/1000):.1f}σ")
print("DEEPDIVE_DONE")
