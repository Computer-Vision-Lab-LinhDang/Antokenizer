"""Softmax bão hoà vì logit quá lớn. Thử các cách đọc khác trên CÙNG bộ slot đã train,
và tính trần tuyệt đối (bình phương tối thiểu). Mọi số chuẩn hoá theo phương sai quanh trung bình."""
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
res = {}
def add(k, v): res.setdefault(k, []).append(v)
with torch.no_grad():
    for x,_ in DataLoader(ds, batch_size=8, num_workers=4):
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok,pos,pid = m.patchify(x,"image"); feat = m.backbone(tok,pos,pid,"image")
            C = m.cd_split._get_content_pooler(64,64).to(dev)(feat)
        f = feat.float(); Cf = C.float(); D = f.shape[-1]
        den = (f - f.mean(1,keepdim=True)).pow(2).sum(-1).mean()
        def err(a): return float((f-a).pow(2).sum(-1).mean()/den)
        add("hiện tại: softmax((C·x)/√D)", err(F.softmax((Cf@f.transpose(-1,-2))/D**0.5, dim=-2).transpose(-1,-2)@Cf))
        cn, fn = F.normalize(Cf,dim=-1), F.normalize(f,dim=-1)
        for tau in (0.1, 0.05, 0.01):
            w = F.softmax((cn@fn.transpose(-1,-2))/tau, dim=-2)
            add(f"cosine + nhiệt độ {tau}", err(w.transpose(-1,-2)@Cf))
        ln = F.layer_norm(f, (D,)); lc = F.layer_norm(Cf, (D,))
        add("LayerNorm cả hai rồi dot", err(F.softmax((lc@ln.transpose(-1,-2))/D**0.5, dim=-2).transpose(-1,-2)@Cf))
        sol = torch.linalg.lstsq(Cf.transpose(-1,-2), f.transpose(-1,-2)).solution
        add("TRẦN: bình phương tối thiểu (tuyến tính, không ràng buộc)", err((Cf.transpose(-1,-2)@sol).transpose(-1,-2)))
        # trần tuyệt đối: 64 hướng PCA tốt nhất của chính feature đó
        dv = f - f.mean(1,keepdim=True)
        U,S,V = torch.linalg.svd(dv, full_matrices=False)
        rec = (U[:,:,:64]*S[:,None,:64]) @ V[:,:64]
        add("TRẦN TUYỆT ĐỐI: 64 hướng PCA tốt nhất", float((dv-rec).pow(2).sum(-1).mean()/den))
print("%-56s %s" % ("cách đọc slot", "content_err (1.0 = vô dụng, 0 = hoàn hảo)"))
for k, v in res.items(): print("%-56s %.3f" % (k, sum(v)/len(v)))
print("SALVAGE_DONE")
