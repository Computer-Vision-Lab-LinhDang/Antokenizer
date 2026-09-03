"""Điểm mới trên đường cong nén-chất lượng: rd10k@10000 tại 4,096 float, cùng bộ ảnh đã dùng cho SD-VAE."""
import inspect, sys, torch, torchvision, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT, _MODALITY_RATIOS
from mavt.evaluation.gates import psnr_from_minus1_1
import lpips as lpips_lib
dev = torch.device("cuda")
tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(256),
                                     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5]*3,[0.5]*3)])
ds = torchvision.datasets.ImageFolder("data/eval/imagenette2-320/val", transform=tf)
g = torch.Generator().manual_seed(0); ds = Subset(ds, torch.randperm(len(ds), generator=g)[:512].tolist())
LP = lpips_lib.LPIPS(net="vgg").to(dev).eval()
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); hp = dict(ck["hyper_parameters"]); hp["local_detail_window_size"]=2
sig = inspect.signature(MAVT.__init__).parameters
m = MAVT(**{k:v for k,v in hp.items() if k in sig}); m.prepare_for_modalities([{"modality":"image","resolution":256}])
mi,un = m.load_state_dict({k[6:]:v for k,v in ck["state_dict"].items() if k.startswith("model.")}, strict=True)
m = m.to(dev).eval()
ps=l1=lp=0.0; n=0; abl={"no_content":0.0,"no_detail":0.0}
with torch.no_grad():
    for x,_ in DataLoader(ds, batch_size=16, num_workers=6):
        x = x.to(dev); B = x.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok,pos,pid = m.patchify(x,"image"); feat = m.backbone(tok,pos,pid,"image")
            cr,dr = _MODALITY_RATIOS["image"]
            comp,cdm,lpos,ltyp = m.cd_split(feat, positions=pos, plane_ids=pid, content_ratio=cr, detail_ratio=dr, return_metadata=True)
            z,mu,lv,kl = m.vae_head(comp); gs = m._grid_shape("image", x)
            r = m.decoder(z,pos,"image",gs,latent_positions=lpos,latent_token_types=ltyp).float().clamp(-1,1)
            for name, mask in (("no_content",ltyp==0),("no_detail",ltyp==1)):
                zz=z.clone(); zz[:,mask]=0
                abl[name] += psnr_from_minus1_1(m.decoder(zz,pos,"image",gs,latent_positions=lpos,latent_token_types=ltyp).float().clamp(-1,1), x)*B
        ps += psnr_from_minus1_1(r,x)*B; l1 += (r-x).abs().mean().item()*B; lp += LP(r,x).mean().item()*B; n += B
print("%-34s %7s %8s %8s %11s" % ("tokenizer (4,096 float/anh 256²)","PSNR","L1","LPIPS","nen"))
print("%-34s %7.2f %8.4f %8.4f %10.1fx" % ("MAVT rd10k@10000 (MOI)", ps/n, l1/n, lp/n, 196608/4096))
print("%-34s %7.2f %8.4f %8.4f %10.1fx" % ("MAVT s1@10k (baseline cu)", 23.92, 0.0837, 0.2618, 48.0))
print("%-34s %7.2f %8.4f %8.4f %10.1fx" % ("SD-VAE f8 ft-mse", 27.44, 0.0587, 0.1186, 48.0))
print("%-34s %7.2f %8.4f %8.4f %10.1fx" % ("SDXL-VAE f8", 27.75, 0.0564, 0.1156, 48.0))
print()
print("Ablation tren checkpoint moi: dong gop content %.2f dB | detail %.2f dB" % (ps/n-abl["no_content"]/n, ps/n-abl["no_detail"]/n))
print("FINAL_DONE")
