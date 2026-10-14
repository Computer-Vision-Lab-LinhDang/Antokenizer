"""Reconstruction quality per modality for a Stage-3 checkpoint (image / video / 3D triplane).
Held-out = the datamodule's group_hash val split (mavt.data.splits, default fractions and salt):
no clip shares a source video with training. Usage: eval_modalities.py <ckpt>"""
import sys, json, inspect, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mavt.model.mavt import MAVT
from mavt.data.datasets import ManifestImageDataset, ManifestVideoDataset, UniversalThreeDDataset
from mavt.evaluation.gates import psnr_from_minus1_1
from mavt.data.splits import DEFAULT_VAL, group_key, item_ids, split_indices
ckpt = sys.argv[1]
dev = torch.device("cuda")
ck = torch.load(ckpt, map_location="cpu", weights_only=False); hp = ck["hyper_parameters"]; sig = inspect.signature(MAVT.__init__).parameters
m = MAVT(**{k: v for k, v in hp.items() if k in sig})
m.prepare_for_modalities([{"modality": "image", "resolution": 256}, {"modality": "video", "resolution": 256, "frames": 8}, {"modality": "threed", "resolution": 256}])
mi, un = m.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}, strict=True); m = m.to(dev).eval()
try:
    import lpips; lp = lpips.LPIPS(net="vgg").to(dev).eval()
except Exception: lp = None

@torch.no_grad()
def run(ds, modality, idx, bs, per_plane=False):
    ds = Subset(ds, idx)
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
split = {name: split_indices([group_key(mod, i) for i in item_ids(ds)], DEFAULT_VAL[mod])
         for name, mod, ds in (("img", "image", img), ("vid", "video", vid), ("thr", "threed", thr))}
print("val split sizes:", {k: len(v["val"]) for k, v in split.items()})
print("image (val split, 256):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in run(img, "image", split["img"]["val"][:256], 32).items()}))
print("video (val split, 64 clips, 8 frames):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in run(vid, "video", split["vid"]["val"][:64], 8).items()}))
r3 = run(thr, "threed", split["thr"]["val"][:192], 8, per_plane=True)
print("3D val split (192 objects):", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r3.items()}))
r3s = run(thr, "threed", split["thr"]["train"][:192], 8)   # seen objects, for comparison
print("3D train split (192 objects): psnr", round(r3s["psnr"], 2))
