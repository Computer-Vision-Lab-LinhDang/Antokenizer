"""Thước đo đúng cho một tokenizer: chất lượng tái tạo tại cùng ngân sách nén.
So MAVT (window 1 = 320 token, và window 2 = 128 token) với SD-VAE f8 trên cùng bộ ảnh held-out."""
import inspect, torch, torchvision, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT
from mavt.evaluation.gates import psnr_from_minus1_1
import lpips as lpips_lib
dev = torch.device("cuda")

tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(256),
                                     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5]*3, [0.5]*3)])
ds = torchvision.datasets.ImageFolder("data/eval/imagenette2-320/val", transform=tf)   # held-out, chưa từng train
g = torch.Generator().manual_seed(0)
ds = Subset(ds, torch.randperm(len(ds), generator=g)[:512].tolist())
LP = lpips_lib.LPIPS(net="vgg").to(dev).eval()

def score(fn, tag, floats):
    ps = l1 = lp = n = 0.0
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=16, num_workers=6):
            x = x.to(dev); r = fn(x).float().clamp(-1, 1)
            b = x.shape[0]
            ps += psnr_from_minus1_1(r, x) * b; l1 += (r - x).abs().mean().item() * b
            lp += LP(r, x).mean().item() * b; n += b
    px = 3 * 256 * 256
    print("%-26s %8.2f %8.4f %8.4f %10d %9.1fx" % (tag, ps/n, l1/n, lp/n, floats, px/floats))

def mavt(ckpt, window):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); hp = dict(ck["hyper_parameters"])
    hp["local_detail_window_size"] = window
    sig = inspect.signature(MAVT.__init__).parameters
    m = MAVT(**{k: v for k, v in hp.items() if k in sig})
    m.prepare_for_modalities([{"modality": "image", "resolution": 256}])
    m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True)
    m = m.to(dev).eval()
    def f(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x, "image", decode=True).reconstruction
    return f, m

print("%-26s %8s %8s %8s %10s %10s" % ("tokenizer", "PSNR", "L1", "LPIPS", "floats/img", "nén"))
f, m = mavt("runs/deepseek/s3/checkpoints/mavt-v3-s3-step=0015000.ckpt", 1)
score(f, "MAVT s3@15k (w=1)", 320*32)
del m; torch.cuda.empty_cache()
f, m = mavt("runs/deepseek/s1/checkpoints/mavt-v3-s1-step=0010000.ckpt", 2)
score(f, "MAVT s1@10k (w=2)", 128*32)
del m; torch.cuda.empty_cache()

from diffusers import AutoencoderKL
for repo, tag, ch in (("stabilityai/sd-vae-ft-mse", "SD-VAE f8 (4ch, ft-mse)", 4),
                      ("stabilityai/sdxl-vae", "SDXL-VAE f8 (4ch)", 4)):
    try:
        vae = AutoencoderKL.from_pretrained(repo, torch_dtype=torch.float32).to(dev).eval()
        def f(x, vae=vae):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return vae.decode(vae.encode(x).latent_dist.mean).sample
        score(f, tag, 32*32*ch)
        del vae; torch.cuda.empty_cache()
    except Exception as e:
        print(f"{tag}: bỏ qua ({type(e).__name__}: {str(e)[:70]})")
print("BENCH_DONE")
