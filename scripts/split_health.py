"""Nhánh content có thực sự xấp xỉ được feature không? Nếu ||R|| >= ||x|| thì phép tách vô nghĩa."""
import inspect, torch, torchvision, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT
dev = torch.device("cuda")
ck = torch.load("runs/deepseek/s3/checkpoints/mavt-v3-s3-step=0015000.ckpt", map_location="cpu", weights_only=False)
hp = ck["hyper_parameters"]; sig = inspect.signature(MAVT.__init__).parameters
m = MAVT(**{k: v for k, v in hp.items() if k in sig})
m.prepare_for_modalities([{"modality": "image", "resolution": 256}])
m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True)
m = m.to(dev).eval()
tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(256),
                                     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5]*3, [0.5]*3)])
ds = torchvision.datasets.ImageFolder("data/eval/imagenette2-320/val", transform=tf)
g = torch.Generator().manual_seed(0); ds = Subset(ds, torch.randperm(len(ds), generator=g)[:128].tolist())
S = {k: [] for k in ("nx","na","nr","cos_xa","slot_div","slot_cos")}
with torch.no_grad():
    for x, _ in DataLoader(ds, batch_size=16, num_workers=4):
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok, pos, pid = m.patchify(x, "image"); feat = m.backbone(tok, pos, pid, "image")
        feat = feat.float(); B, N, D = feat.shape
        N_c = max(1, int(N*0.25)); C = m.cd_split._content_poolers[f"{N_c}_{N_c}"].to(dev)(feat).float()
        w = F.softmax((C @ feat.transpose(-1,-2)) / D**0.5, dim=-1)
        xa = w.transpose(-1,-2) @ C; R = feat - xa
        S["nx"].append(feat.norm(dim=-1).mean().item()); S["na"].append(xa.norm(dim=-1).mean().item())
        S["nr"].append(R.norm(dim=-1).mean().item())
        S["cos_xa"].append(F.cosine_similarity(feat, xa, dim=-1).mean().item())
        Cn = F.normalize(C, dim=-1); sim = Cn @ Cn.transpose(-1,-2)
        off = sim[:, ~torch.eye(N_c, dtype=torch.bool, device=dev)].mean().item()
        S["slot_cos"].append(off)
        S["slot_div"].append(float((w.max(dim=-2).values > 0.5).float().mean()))
a = {k: sum(v)/len(v) for k, v in S.items()}
print(f"||x||            = {a['nx']:.3f}   (feature từ backbone)")
print(f"||x_approx||     = {a['na']:.3f}   (dựng lại từ 64 content slot)")
print(f"||R|| = ||x-x̂||  = {a['nr']:.3f}   → tỉ lệ residual/gốc = {a['nr']/a['nx']:.3f}")
print(f"cos(x, x_approx) = {a['cos_xa']:.3f}   (1.0 = xấp xỉ đúng hướng)")
print(f"cos trung bình giữa các content slot = {a['slot_cos']:.3f}   (1.0 = mọi slot giống hệt nhau)")
print(f"tỉ lệ patch bị 1 slot chiếm > 50% trọng số attention = {a['slot_div']*100:.1f}%")
print()
print("Diễn giải: nếu ||R||/||x|| ≈ 1 và cos(x,x̂) nhỏ, nhánh content KHÔNG giảm được gánh nặng cho nhánh detail.")
print("SPLIT_DONE")
