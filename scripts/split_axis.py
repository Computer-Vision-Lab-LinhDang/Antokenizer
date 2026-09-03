"""64 slot có chứa thông tin không, hay chỉ cách đọc ra bị hỏng?
So 3 cách dựng lại x từ CÙNG bộ slot C đã train: softmax theo patch (code hiện tại),
softmax theo slot (chuẩn), và chiếu bình phương tối thiểu (trần lý thuyết của 64 slot)."""
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
                                     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5]*3,[0.5]*3)])
ds = torchvision.datasets.ImageFolder("data/eval/imagenette2-320/val", transform=tf)
g = torch.Generator().manual_seed(0); ds = Subset(ds, torch.randperm(len(ds), generator=g)[:128].tolist())
acc = {k: [] for k in ("cur_cos","cur_rr","slot_cos","slot_rr","ls_cos","ls_rr","rank")}
with torch.no_grad():
    for x, _ in DataLoader(ds, batch_size=8, num_workers=4):
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok, pos, pid = m.patchify(x, "image"); feat = m.backbone(tok, pos, pid, "image")
        f = feat.float(); B, N, D = f.shape; N_c = int(N*0.25)
        C = m.cd_split._content_poolers[f"{N_c}_{N_c}"].to(dev)(f).float()
        logits = (C @ f.transpose(-1,-2)) / D**0.5                      # (B, N_c, N)
        for tag, w in (("cur", F.softmax(logits, dim=-1)), ("slot", F.softmax(logits, dim=-2))):
            xa = w.transpose(-1,-2) @ C
            acc[f"{tag}_cos"].append(F.cosine_similarity(f, xa, dim=-1).mean().item())
            acc[f"{tag}_rr"].append(((f-xa).norm(dim=-1)/f.norm(dim=-1)).mean().item())
        # trần: chiếu f lên span(C) bằng bình phương tối thiểu
        sol = torch.linalg.lstsq(C.transpose(-1,-2), f.transpose(-1,-2)).solution   # (B, N_c, N)
        xls = (C.transpose(-1,-2) @ sol).transpose(-1,-2)
        acc["ls_cos"].append(F.cosine_similarity(f, xls, dim=-1).mean().item())
        acc["ls_rr"].append(((f-xls).norm(dim=-1)/f.norm(dim=-1)).mean().item())
        acc["rank"].append(float(torch.linalg.matrix_rank(C).float().mean()))
a = {k: sum(v)/len(v) for k, v in acc.items()}
print("%-42s %8s %10s" % ("cách dựng lại x từ 64 content slot", "cos(x,x̂)", "||R||/||x||"))
print("%-42s %8.3f %10.3f" % ("softmax theo PATCH — code hiện tại", a["cur_cos"], a["cur_rr"]))
print("%-42s %8.3f %10.3f" % ("softmax theo SLOT — chuẩn", a["slot_cos"], a["slot_rr"]))
print("%-42s %8.3f %10.3f" % ("bình phương tối thiểu — trần của 64 slot", a["ls_cos"], a["ls_rr"]))
print(f"\nhạng của ma trận 64 slot: {a['rank']:.1f}/64  (thấp = slot suy biến, cao = span đủ rộng)")
print("AXIS_DONE")
