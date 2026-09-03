"""content_err kẹt ở 1.0 dù bị ép. Nghi vấn cơ chế: logit attention tính trên x CHƯA trừ trung bình,
mà 99.9% năng lượng của x nằm ở trung bình chung → mọi patch có similarity gần như nhau
→ softmax gần đều → x_approx gần như hằng số theo patch → không bắt được dao động nào."""
import inspect, sys, torch, torchvision, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT
dev = torch.device("cuda")
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); hp = dict(ck["hyper_parameters"]); hp["local_detail_window_size"]=2
sig = inspect.signature(MAVT.__init__).parameters
m = MAVT(**{k:v for k,v in hp.items() if k in sig}); m.prepare_for_modalities([{"modality":"image","resolution":256}])
m.load_state_dict({k[6:]:v for k,v in ck["state_dict"].items() if k.startswith("model.")}, strict=True); m=m.to(dev).eval()
tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(256),
                                     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5]*3,[0.5]*3)])
ds = torchvision.datasets.ImageFolder("data/eval/imagenette2-320/val", transform=tf)
g = torch.Generator().manual_seed(0); ds = Subset(ds, torch.randperm(len(ds), generator=g)[:64].tolist())
A = {k: [] for k in ("w_spread","logit_spread","logit_mean_share","xa_spread","x_spread","err_now","err_centered_logits")}
with torch.no_grad():
    for x,_ in DataLoader(ds, batch_size=8, num_workers=4):
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok,pos,pid = m.patchify(x,"image"); feat = m.backbone(tok,pos,pid,"image")
            C = m.cd_split._get_content_pooler(64,64).to(dev)(feat)
        f = feat.float(); Cf = C.float(); D = f.shape[-1]
        logits = (Cf @ f.transpose(-1,-2)) / D**0.5            # (B, N_c, N)
        w = F.softmax(logits, dim=-2)
        xa = w.transpose(-1,-2) @ Cf
        # logit tính trên feature ĐÃ trừ trung bình
        dv = f - f.mean(1, keepdim=True); Cd = Cf - Cf.mean(1, keepdim=True)
        w2 = F.softmax((Cd @ dv.transpose(-1,-2)) / D**0.5, dim=-2)
        xa2 = w2.transpose(-1,-2) @ Cf
        def err(a):
            return float((f-a).pow(2).sum(-1).mean() / dv.pow(2).sum(-1).mean())
        A["w_spread"].append(float(w.std(dim=-1).mean() / w.mean()))          # dao động của trọng số theo patch
        A["logit_spread"].append(float(logits.std(dim=-1).mean()))
        A["logit_mean_share"].append(float(logits.mean(dim=-1).abs().mean() / logits.abs().mean()))
        A["xa_spread"].append(float((xa - xa.mean(1,keepdim=True)).norm(dim=-1).mean()))
        A["x_spread"].append(float(dv.norm(dim=-1).mean()))
        A["err_now"].append(err(xa)); A["err_centered_logits"].append(err(xa2))
a = {k: sum(v)/len(v) for k,v in A.items()}
print(f"logit: |trung bình| / |giá trị| = {a['logit_mean_share']:.4f}   (1.0 = logit gần như hằng số, không phân biệt patch)")
print(f"độ lệch chuẩn của logit theo patch = {a['logit_spread']:.4f}")
print(f"trọng số attention: std/mean theo patch = {a['w_spread']:.4f}   (0 = softmax đều, mọi patch nhận cùng tổ hợp slot)")
print()
print(f"biên độ dao động THẬT của feature quanh trung bình : {a['x_spread']:.3f}")
print(f"biên độ dao động mà x_approx tạo ra                : {a['xa_spread']:.3f}   ({100*a['xa_spread']/a['x_spread']:.1f}% của thật)")
print()
print(f"content_err hiện tại (logit trên x thô)            : {a['err_now']:.3f}")
print(f"content_err nếu logit tính trên feature đã trừ TB  : {a['err_centered_logits']:.3f}   ← phép thử cách sửa")
print("WHY_DONE")
