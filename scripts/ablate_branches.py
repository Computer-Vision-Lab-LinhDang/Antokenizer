"""Thông tin nằm ở nhánh nào? Xoá content slot hoặc detail token khỏi z rồi decode."""
import inspect, sys, torch, torchvision, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT, _MODALITY_RATIOS
from mavt.evaluation.gates import psnr_from_minus1_1
dev = torch.device("cuda")
tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(256),
                                     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5]*3,[0.5]*3)])
ds = torchvision.datasets.ImageFolder("data/eval/imagenette2-320/val", transform=tf)
g = torch.Generator().manual_seed(0); ds = Subset(ds, torch.randperm(len(ds), generator=g)[:256].tolist())

def run(ckpt, window, tag):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); hp = dict(ck["hyper_parameters"])
    hp["local_detail_window_size"] = window
    sig = inspect.signature(MAVT.__init__).parameters
    m = MAVT(**{k: v for k, v in hp.items() if k in sig})
    m.prepare_for_modalities([{"modality":"image","resolution":256}])
    m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True)
    m = m.to(dev).eval()
    res = {k: 0.0 for k in ("full","no_content","no_detail")}; n = 0; shapes = None
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=16, num_workers=4):
            x = x.to(dev); B = x.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok, pos, pid = m.patchify(x, "image")
                feat = m.backbone(tok, pos, pid, "image")
                cr, dr = _MODALITY_RATIOS["image"]
                comp, cdm, lpos, ltyp = m.cd_split(feat, positions=pos, plane_ids=pid,
                                                   content_ratio=cr, detail_ratio=dr, return_metadata=True)
                z, mu, lv, kl = m.vae_head(comp)
                gs = m._grid_shape("image", x)
                for name, mask in (("full", None), ("no_content", ltyp==0), ("no_detail", ltyp==1)):
                    zz = z.clone()
                    if mask is not None: zz[:, mask] = 0
                    r = m.decoder(zz, pos, "image", gs, latent_positions=lpos, latent_token_types=ltyp)
                    res[name] += psnr_from_minus1_1(r.float().clamp(-1,1), x) * B
            if shapes is None:
                shapes = (int((ltyp==0).sum()), int((ltyp==1).sum()))
            n += B
    nc, nd = shapes
    print(f"\n{tag}  (window={window}: {nc} content + {nd} detail = {nc+nd} token, {(nc+nd)*32} float)")
    print(f"  z đầy đủ                       : {res['full']/n:6.2f} dB")
    print(f"  xoá {nc:>3d} content slot          : {res['no_content']/n:6.2f} dB   (mất {res['full']/n-res['no_content']/n:5.2f} dB khi bỏ {100*nc/(nc+nd):.0f}% ngân sách)")
    print(f"  xoá {nd:>3d} detail token          : {res['no_detail']/n:6.2f} dB   (mất {res['full']/n-res['no_detail']/n:5.2f} dB khi bỏ {100*nd/(nc+nd):.0f}% ngân sách)")
    del m; torch.cuda.empty_cache()

run("runs/deepseek/s3/checkpoints/mavt-v3-s3-step=0015000.ckpt", 1, "MAVT s3@15k")
run("runs/deepseek/s1/checkpoints/mavt-v3-s1-step=0010000.ckpt", 2, "MAVT s1@10k")
print("\nABLATE_DONE")
