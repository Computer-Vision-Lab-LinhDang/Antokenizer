"""Reconstruction quality per modality for a Stage-3 checkpoint (image / video / 3D triplane).
3D held-out = objects rendered after the run started (sorted uid index >= --seen-3d)."""
import sys, json, inspect, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT
from mavt.data.datasets import ManifestImageDataset, ManifestVideoDataset, UniversalThreeDDataset
from mavt.evaluation.gates import psnr_from_minus1_1
ckpt, seen3d = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
ck = torch.load(ckpt, map_location="cpu", weights_only=False); hp = ck["hyper_parameters"]; sig = inspect.signature(MAVT.__init__).parameters
m = MAVT(**{k: v for k, v in hp.items() if k in sig})
m.prepare_for_modalities([{"modality": "image", "resolution": 256}, {"modality": "video", "resolution": 256, "frames": 8}, {"modality": "threed", "resolution": 256}])
mi, un = m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True); m = m.to(dev).eval()
try:
    import lpips; lp = lpips.LPIPS(net="vgg").to(dev).eval()
except Exception: lp = None

@torch.no_grad()
def run(ds, modality, n, bs, per_plane=False):
    ds = Subset(ds, list(range(len(ds) - n, len(ds))) if modality != "threed" else list(range(len(ds) - n, len(ds))))
    ps, l1s, lps, planes = [], [], [], [[], [], []]
    for b in DataLoader(ds, batch_size=bs, num_workers=4):
        x = b["data"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, modality, decode=True)
        r = out.reconstruction.float().clamp(-1, 1)
        ps.append(psnr_from_minus1_1(r, x) * x.shape[0]); l1s.append((r - x).abs().mean().item() * x.shape[0])
        if lp is not None:
            if modality == "image": lps.append(lp(r, x).mean().item() * x.shape[0])
            elif modality == "video": lps.append(lp(r[:, :, r.shape[2] // 2], x[:, :, x.shape[2] // 2]).mean().item() * x.shape[0])
            else: lps.append(lp(r[:, 0], x[:, 0]).mean().item() * x.shape[0])
        if per_plane:
            for p in range(3): planes[p].append(psnr_from_minus1_1(r[:, p], x[:, p]) * x.shape[0])
    N = len(ds); res = {"n": N, "psnr": sum(ps) / N, "l1": sum(l1s) / N, "lpips": (sum(lps) / N) if lps else None}
    if per_plane: res["psnr_planes(oxoy,oxoz,oyoz)"] = [round(sum(pp) / N, 2) for pp in planes]
    return res

img = ManifestImageDataset("data/manifests/openimages_v2.jsonl", 256)
vid = ManifestVideoDataset("data/manifests/openvid_v2.jsonl", n_frames=8, resolution=256, frame_stride=2)
thr = UniversalThreeDDataset("data/datasets/objaverse_lvis", 256)
print("3D objects now:", len(thr), "| seen during training:", seen3d, "| held-out available:", len(thr) - seen3d)
print("image (last 256 of manifest):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in run(img, "image", 256, 32).items()}))
print("video (last 64 clips, 8 frames):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in run(vid, "video", 64, 8).items()}))
r3 = run(thr, "threed", 192, 8, per_plane=True)
print("3D held-out (192 unseen objects):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r3.items()}))
# also seen 3D for comparison (first 192)
seen = Subset(thr, list(range(192))); ps = 0
with torch.no_grad():
    for b in DataLoader(seen, batch_size=8, num_workers=4):
        x = b["data"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = m(x, "threed", decode=True)
        ps += psnr_from_minus1_1(out.reconstruction.float().clamp(-1, 1), x) * x.shape[0]
print("3D seen (first 192 objects): psnr", round(ps / 192, 2))
