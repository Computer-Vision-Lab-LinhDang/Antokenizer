"""Inference demo for a MAVT checkpoint: reconstructions (image / video / 3D) + semantic retrieval sheets."""
import sys, os, json, inspect, random, torch, torch.nn.functional as F, torchvision
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT
from mavt.data.datasets import ManifestVideoDataset, UniversalThreeDDataset
from mavt.evaluation.gates import psnr_from_minus1_1
ckpt, out_dir = sys.argv[1], sys.argv[2]; os.makedirs(out_dir, exist_ok=True)
dev = torch.device("cuda"); torch.manual_seed(0); random.seed(0)
ck = torch.load(ckpt, map_location="cpu", weights_only=False); hp = ck["hyper_parameters"]; sig = inspect.signature(MAVT.__init__).parameters
m = MAVT(**{k: v for k, v in hp.items() if k in sig})
m.prepare_for_modalities([{"modality": "image", "resolution": 256}, {"modality": "video", "resolution": 256, "frames": 8}, {"modality": "threed", "resolution": 256}])
m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True); m = m.to(dev).eval()

def to_img(t):  # (3,H,W) in [-1,1] -> PIL
    return Image.fromarray(((t.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
def label(im, text):
    d = ImageDraw.Draw(im); d.rectangle([0, 0, 8 + 7 * len(text), 14], fill=(0, 0, 0)); d.text((4, 1), text, fill=(255, 255, 0)); return im
def sheet(rows, cell=256):
    W = max(len(r) for r in rows) * cell; H = len(rows) * cell; s = Image.new("RGB", (W, H), "white")
    for r, row in enumerate(rows):
        for c, im in enumerate(row): s.paste(im.resize((cell, cell)), (c * cell, r * cell))
    return s

tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(256), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)])
woof = torchvision.datasets.ImageFolder("data/eval/imagewoof2-320/val", transform=tf)
# ---- images: 6 held-out natural images ----
idx = random.sample(range(len(woof)), 6); x = torch.stack([woof[i][0] for i in idx]).to(dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): o = m(x, "image", decode=True)
r = o.reconstruction.float().clamp(-1, 1)
rows = []
for i in range(6):
    p = psnr_from_minus1_1(r[i:i+1], x[i:i+1]); err = ((r[i] - x[i]).abs().mean(0, keepdim=True).repeat(3, 1, 1) * 4 - 1).clamp(-1, 1)
    rows.append([label(to_img(x[i]), "input"), label(to_img(r[i]), f"recon {p:.1f} dB"), label(to_img(err), "|error| x4")])
sheet(rows).save(f"{out_dir}/01_image_recon.png")
print("image psnr:", [round(float(psnr_from_minus1_1(r[i:i+1], x[i:i+1])), 1) for i in range(6)])
# ---- video: 2 clips, 8 frames ----
vid = ManifestVideoDataset("data/manifests/openvid_v2.jsonl", n_frames=8, resolution=256, frame_stride=2)
rows = []
for j in [len(vid) - 3, len(vid) - 17]:
    s = vid[j]; xv = s["data"].unsqueeze(0).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): ov = m(xv, "video", decode=True)
    rv = ov.reconstruction.float().clamp(-1, 1)[0]; p = psnr_from_minus1_1(rv.unsqueeze(0), xv)
    rows.append([label(to_img(xv[0, :, t]), f"in t{t}") for t in range(8)]); rows.append([label(to_img(rv[:, t]), f"recon t{t}" + (f" {p:.1f}dB" if t == 0 else "")) for t in range(8)])
    print("video", s["id"][:20], "psnr", round(float(p), 2), "| caption:", s["caption"][:70])
sheet(rows, cell=192).save(f"{out_dir}/02_video_recon.png")
# ---- 3D: 3 objects ----
thr = UniversalThreeDDataset("data/datasets/objaverse_lvis", 256)
rows = []
for j in random.sample(range(len(thr)), 3):
    s = thr[j]; x3 = s["data"].unsqueeze(0).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): o3 = m(x3, "threed", decode=True)
    r3 = o3.reconstruction.float().clamp(-1, 1)[0]; p = psnr_from_minus1_1(r3.unsqueeze(0), x3)
    rows.append([label(to_img(x3[0, k]), f"in {n}") for k, n in enumerate(("front", "top", "side"))] + [label(to_img(r3[k]), f"recon {n}" + (f" {p:.1f}dB" if k == 0 else "")) for k, n in enumerate(("front", "top", "side"))])
    print("3d", s["id"][:8], "psnr", round(float(p), 2), "| caption:", s["caption"][:70])
sheet(rows, cell=192).save(f"{out_dir}/03_threed_recon.png")
# ---- semantic retrieval on 600 held-out Imagewoof images ----
pool_idx = random.sample(range(len(woof)), 600); embs = []; ys = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for b in DataLoader(Subset(woof, pool_idx), batch_size=50, num_workers=4):
        embs.append(F.normalize(m(b[0].to(dev), "image", decode=False).semantic.float(), dim=-1).cpu()); ys.append(b[1])
E = torch.cat(embs); Y = torch.cat(ys); sim = E @ E.T; sim.fill_diagonal_(-2)
top = sim.topk(5, dim=-1).indices
print("retrieval top-1 same-class rate (600 held-out Imagewoof):", round(float((Y[top[:, 0]] == Y).float().mean()), 3), "| top-5 majority:", round(float(((Y[top] == Y[:, None]).float().mean(1) >= 0.6).float().mean()), 3))
rows = []
for q in random.sample(range(600), 4):
    row = [label(to_img(woof[pool_idx[q]][0]), f"query: {woof.classes[Y[q]][:12]}")]
    for k in range(4):
        n = top[q, k].item(); row.append(label(to_img(woof[pool_idx[n]][0]), f"#{k+1} {woof.classes[Y[n]][:12]} {'OK' if Y[n]==Y[q] else 'x'}"))
    rows.append(row)
sheet(rows, cell=192).save(f"{out_dir}/04_semantic_retrieval.png")
print("DEMO_DONE")
