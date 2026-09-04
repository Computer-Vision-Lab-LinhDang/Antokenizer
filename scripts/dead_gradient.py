"""Mắt xích trung tâm: softmax bão hoà → gradient theo logit ≈ 0 → phép gán patch→slot
không bao giờ học được, nên thêm loss bao nhiêu cũng vô ích. Đo trực tiếp."""
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
g = torch.Generator().manual_seed(0); ds = Subset(ds, torch.randperm(len(ds), generator=g)[:32].tolist())
x, _ = next(iter(DataLoader(ds, batch_size=8, num_workers=2))); x = x.to(dev)
with torch.no_grad():
    tok,pos,pid = m.patchify(x,"image"); feat = m.backbone(tok,pos,pid,"image").float()
    C = m.cd_split._get_content_pooler(64,64).to(dev)(feat).float()
D = feat.shape[-1]; den = (feat-feat.mean(1,keepdim=True)).pow(2).sum(-1).mean()

print("%-30s %10s %12s %14s %10s" % ("thang logit","|logit| TB","entropy softmax","‖∂L/∂logit‖","content_err"))
for scale, tag in ((1.0,"gốc (÷√D)"), (0.1,"÷10"), (0.01,"÷100"), (0.001,"÷1000")):
    logits = ((C @ feat.transpose(-1,-2)) / D**0.5 * scale).detach().requires_grad_(True)
    w = F.softmax(logits, dim=-2)
    xa = w.transpose(-1,-2) @ C
    loss = (feat-xa).pow(2).sum(-1).mean() / den
    loss.backward()
    ent = -(w.clamp_min(1e-12)*w.clamp_min(1e-12).log()).sum(-2).mean()      # entropy trên trục slot
    print("%-30s %10.1f %12.4f %14.3e %10.3f" % (tag, logits.abs().mean().item(), ent.item(),
          logits.grad.norm().item(), loss.item()))
print()
print("entropy tối đa nếu softmax đều trên 64 slot = %.4f" % torch.log(torch.tensor(64.0)).item())
print("→ entropy ≈ 0 nghĩa là one-hot cứng; gradient theo logit teo đi cùng lúc.")
print("DEADGRAD_DONE")
