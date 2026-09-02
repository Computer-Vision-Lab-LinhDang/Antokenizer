#!/usr/bin/env python3
"""Where does class information get lost along the student pipeline? CIFAR-100 kNN@20 (5k/1k,
same protocol as the S1 gates) measured on features hooked at each stage of MAVT plus the SigLIP2
teacher pooled / patch-mean features, z noise level, and linear-map ceilings.
Run on the GPU server from the repo root: PYTHONPATH=src python scripts/probe_semantic_stages.py"""
import torch, torch.nn.functional as F, torchvision, inspect
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT
from mavt.evaluation.gates import knn_top1
dev = torch.device("cuda")
import sys, argparse, os
_ap = argparse.ArgumentParser()
_ap.add_argument("ckpt", nargs="?", default="runs/deepseek/s1_w1/checkpoints/mavt-v3-s1-step=0010000.ckpt")
_ap.add_argument("--imagefolder", default="", help="comma-separated ImageFolder roots with train/ and val/ (in-distribution kNN, e.g. imagenette2-320,imagewoof2-320)")
_ap.add_argument("--openimages", default="data/manifests/openimages_v2.jsonl", help="manifest for the label-free teacher-neighbourhood overlap (first 1024 images)")
_ap.add_argument("--no-cifar", action="store_true")
ARGS = _ap.parse_args()
CKPT = ARGS.ckpt
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
hp = ck["hyper_parameters"]; sig = inspect.signature(MAVT.__init__).parameters
m = MAVT(**{k: v for k, v in hp.items() if k in sig}); m.prepare_for_modalities([{"modality": "image", "resolution": 256}])
m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True); m = m.to(dev).eval()
from transformers import AutoModel
sig2 = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384"); T = sig2.vision_model.to(dev).eval()
tsize = int(sig2.config.vision_config.image_size)
print("teacher image_size:", tsize, "| window:", hp["local_detail_window_size"], "| latent_dim:", hp["latent_dim"], "| kl_weight:", hp["kl_weight"], "| w_sem:", hp.get("w_sem"), "| w_vic:", hp.get("w_vic"))
tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)])
tr = Subset(torchvision.datasets.CIFAR100("data/eval/cifar100", train=True, download=True, transform=tf), range(5000))
te = Subset(torchvision.datasets.CIFAR100("data/eval/cifar100", train=False, download=True, transform=tf), range(1000))
feats = {}
m.backbone.register_forward_hook(lambda mod, i, o: feats.__setitem__("backbone", o if torch.is_tensor(o) else o[0]))
m.cd_split.register_forward_hook(lambda mod, i, o: feats.__setitem__("compressed", o[0]))
m.vae_head.register_forward_hook(lambda mod, i, o: feats.update(z=o[0], mu=o[1], logvar=o[2]))
KEYS = ("sem", "bb_mean", "content_mean", "detail_mean", "mu_content_flat", "mu_mean", "t_pool", "t_patch_mean")

@torch.no_grad()
def extract(ds):
    out = {k: [] for k in KEYS}; ys = []; snr = []
    for x, y in DataLoader(ds, batch_size=50, num_workers=4):
        x = F.interpolate(x.to(dev), size=(256, 256), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(x, "image", decode=False)
            to = T(pixel_values=F.interpolate(x, size=(tsize, tsize), mode="bilinear", align_corners=False))
        bb = feats["backbone"].float(); comp = feats["compressed"].float(); mu = feats["mu"].float(); lv = feats["logvar"].float()
        Nc = 64
        out["sem"].append(o.semantic.float().cpu()); out["bb_mean"].append(bb.mean(1).cpu())
        out["content_mean"].append(comp[:, :Nc].mean(1).cpu()); out["detail_mean"].append(comp[:, Nc:].mean(1).cpu())
        out["mu_content_flat"].append(mu[:, :Nc].flatten(1).cpu()); out["mu_mean"].append(mu.mean(1).cpu())
        out["t_pool"].append(to.pooler_output.float().cpu()); out["t_patch_mean"].append(to.last_hidden_state.float().mean(1).cpu()); ys.append(y)
        snr.append(((0.5 * lv).exp() / (mu.abs() + 1e-6)).median().item())
    return {k: torch.cat(v) for k, v in out.items()}, torch.cat(ys), sum(snr) / len(snr)

def report(A, ya, B, yb, title):
    print(f"--- {title}: kNN@20 per stage ---")
    print("%-18s %6s %7s %8s" % ("feature", "dim", "kNN@20", "eff_rank"))
    for k in KEYS:
        print("%-18s %6d %7.3f %8.1f" % (k, A[k].shape[1], knn_top1(A[k], ya, B[k], yb), er(A[k])))
    s_ = B["sem"] - B["sem"].mean(0); t_ = B["t_pool"] - B["t_pool"].mean(0)
    print("align_cos_centered (test):", round(float(F.cosine_similarity(s_, t_, dim=-1).mean()), 3))

def er(e):
    e = F.normalize(e.float(), dim=-1); e = e - e.mean(0, keepdim=True); s = torch.linalg.svdvals(e); p = s ** 2 / (s ** 2).sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))

if not ARGS.no_cifar:
    A, ya, snr = extract(tr); B, yb, _ = extract(te)
    print(f"z noise: median std/|mu| = {snr:.2f}  (head trained on z=mu+std*eps, evaluated on mu)")
    report(A, ya, B, yb, "CIFAR-100 32px→256 (gate protocol, OFF-distribution)")
    for name, lam in (("sem", 1e-2), ("content_mean", 1e-1), ("bb_mean", 1e-1)):
        X = A[name]; Y = A["t_pool"]; Xc = torch.cat([X, torch.ones(len(X), 1)], 1)
        W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(Xc.shape[1]), Xc.T @ Y)
        pred = torch.cat([B[name], torch.ones(len(B[name]), 1)], 1) @ W
        print("kNN after linear map %s -> teacher (fit on 5k train): %.3f" % (name, knn_top1(A["t_pool"], ya, pred, yb)))

# ---------------- in-distribution kNN: natural images at 256 px (ImageFolder train/val) ----------------
tf_nat = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(256),
                                         torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)])
for root in [r for r in ARGS.imagefolder.split(",") if r]:
    trn = torchvision.datasets.ImageFolder(os.path.join(root, "train"), transform=tf_nat)
    val = torchvision.datasets.ImageFolder(os.path.join(root, "val"), transform=tf_nat)
    g = torch.Generator().manual_seed(0)
    trn = Subset(trn, torch.randperm(len(trn), generator=g)[:5000].tolist())
    val = Subset(val, torch.randperm(len(val), generator=g)[:2000].tolist())
    A2, y2a, _ = extract(trn); B2, y2b, _ = extract(val)
    report(A2, y2a, B2, y2b, f"{os.path.basename(root.rstrip('/'))} @256 (IN-distribution, {len(trn)} train / {len(val)} val)")

# ---------------- label-free: teacher-neighbourhood overlap on Open Images ----------------
if ARGS.openimages and os.path.exists(ARGS.openimages):
    import json
    from mavt.data.datasets import ManifestImageDataset
    ds = Subset(ManifestImageDataset(ARGS.openimages, 256), range(1024))
    E = {k: [] for k in ("sem", "content_mean", "t_pool")}
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, num_workers=4):
            x = b["data"].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(x, "image", decode=False)
                to = T(pixel_values=F.interpolate(x, size=(tsize, tsize), mode="bilinear", align_corners=False))
            E["sem"].append(o.semantic.float().cpu()); E["content_mean"].append(feats["compressed"][:, :64].float().mean(1).cpu()); E["t_pool"].append(to.pooler_output.float().cpu())
    E = {k: F.normalize(torch.cat(v), dim=-1) for k, v in E.items()}
    def nn_idx(e, k=10):
        sim = e @ e.T; sim.fill_diagonal_(-2); return sim.topk(k, dim=-1).indices
    tk = nn_idx(E["t_pool"])
    for k in ("sem", "content_mean"):
        sk = nn_idx(E[k]); ov = torch.tensor([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(sk, tk)]).float().mean() / 10
        print(f"Open Images 1024 (IN-distribution, label-free): top-10 neighbour overlap with teacher — {k}: {ov:.3f}")

