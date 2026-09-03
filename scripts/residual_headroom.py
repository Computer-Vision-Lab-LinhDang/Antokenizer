"""Điều kiện cần cho cấp phát token theo nội dung: năng lượng residual có lệch giữa các vùng không?
Nếu phân bố phẳng → cấp phát thích nghi vô ích. Nếu lệch mạnh → có dư địa ở cùng ngân sách."""
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
g = torch.Generator().manual_seed(0)
ds = Subset(ds, torch.randperm(len(ds), generator=g)[:256].tolist())

energies = []
with torch.no_grad():
    for x, _ in DataLoader(ds, batch_size=16, num_workers=4):
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok, pos, pid = m.patchify(x, "image")
            feat = m.backbone(tok, pos, pid, "image")
        feat = feat.float()
        B, N, D = feat.shape
        N_c = max(1, int(N * 0.25)); key = f"{N_c}_{max(1,int(N*0.25))}"
        C = m.cd_split._content_poolers[key].to(dev)(feat)          # (B, N_c, D)
        w = F.softmax((C @ feat.transpose(-1, -2)) / D**0.5, dim=-1)
        R = feat - w.transpose(-1, -2) @ C                           # residual mỗi patch
        energies.append(R.norm(dim=-1).cpu())                        # (B, N)
E = torch.cat(energies)                                              # (256 ảnh, 256 patch)
print(f"residual norm mỗi patch: {E.shape[0]} ảnh x {E.shape[1]} patch")

# mức lệch trong từng ảnh
Es, _ = E.sort(dim=-1, descending=True)
tot = (Es**2).sum(-1, keepdim=True)
for frac in (0.10, 0.25, 0.50):
    k = int(frac * E.shape[1])
    share = ((Es[:, :k]**2).sum(-1) / tot.squeeze(-1)).mean()
    print(f"  {int(frac*100):>2d}% patch có residual lớn nhất mang {share*100:5.1f}% tổng năng lượng residual")
gini = []
for row in Es:
    v = row.sort().values; n = len(v); idx = torch.arange(1, n+1, dtype=torch.float32)
    gini.append(float(((2*idx - n - 1) * v).sum() / (n * v.sum())))
print(f"  hệ số Gini trung bình trong một ảnh: {sum(gini)/len(gini):.3f}  (0 = đều tuyệt đối, 1 = dồn hết vào một patch)")
print(f"  tỉ lệ patch cao nhất / thấp nhất: {(Es[:,0]/Es[:,-1]).median():.1f}x (trung vị)")
print(f"  độ lệch giữa các ảnh (std/mean của năng lượng tổng): {(E.pow(2).sum(-1).std()/E.pow(2).sum(-1).mean()):.3f}")

# mô phỏng: giữ top-k% window, bỏ phần còn lại → mất bao nhiêu năng lượng
print("\nMô phỏng cấp phát thích nghi ở CÙNG ngân sách 128 token (64 content + 64 detail):")
print("  hiện tại: 64 window cố định, mỗi window gộp 4 patch liền kề")
grid = E.view(-1, 16, 16)
pooled_fixed = grid.view(-1, 8, 2, 8, 2).pow(2).sum((2, 4))          # gộp 2x2 cố định
keep_top = torch.topk(E.pow(2), 64, dim=-1).values.sum(-1) / E.pow(2).sum(-1)
print(f"  giữ 64 patch có residual lớn nhất (không gộp): giữ được {keep_top.mean()*100:.1f}% năng lượng residual")
print(f"  gộp 2x2 cố định thành 64 window: giữ 100% năng lượng nhưng trung bình hoá trong mỗi window")
print("HEADROOM_DONE")
