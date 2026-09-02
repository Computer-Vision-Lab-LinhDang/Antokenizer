#!/usr/bin/env python3
"""Evaluate S1 acceptance gates for a checkpoint (or the SigLIP2-initialised student).

Measures student AND frozen teacher on the same Open Images subset + CIFAR-100 kNN
probe, reports student-teacher alignment (plain and batch-centered), writes JSON.
"""
from __future__ import annotations
import argparse, inspect, json, os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mavt.model.mavt import MAVT
from mavt.data.datasets import ManifestImageDataset
from mavt.evaluation.gates import (S1_GATES, effective_rank, evaluate_gates, knn_top1,
                                   mean_feature_std, mean_pairwise_cosine, psnr_from_minus1_1)


def build_model(args, dev):
    kw = dict(embed_dim=1152, num_heads=16, num_blocks=12, mlp_ratio=args.mlp_ratio, semantic_dim=args.semantic_dim,
              local_detail_window_size=args.window, rgat_impl=args.rgat_impl, use_gradient_checkpointing=False)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        hp = ck.get("hyper_parameters", {})
        sig = inspect.signature(MAVT.__init__).parameters
        kw.update({k: v for k, v in hp.items() if k in sig})
        kw["use_gradient_checkpointing"] = False
        model = MAVT(**kw)
        model.prepare_for_modalities([{"modality": "image", "resolution": args.resolution}])
        sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[ckpt] loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    else:
        model = MAVT(**kw)
        model.prepare_for_modalities([{"modality": "image", "resolution": args.resolution}])
        if args.init_siglip2:
            model.load_siglip2_weights(args.teacher, freeze_stages=0, strict=True, init_patchify=args.init_patchify)
    return model.to(dev).eval()


def load_teacher(name, dev):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name)
    t = m.vision_model.to(dev).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t, int(m.config.vision_config.image_size)


@torch.no_grad()
def teacher_pool(teacher, x, size):
    xi = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return teacher(pixel_values=xi).pooler_output


@torch.no_grad()
def run_openimages(model, teacher, tsize, args, dev):
    from torch.utils.data import DataLoader, Subset
    ds = ManifestImageDataset(args.manifest, args.resolution)
    ds = Subset(ds, range(min(args.n_images, len(ds))))
    dl = DataLoader(ds, batch_size=args.batch, num_workers=4)
    lp = None
    try:
        import lpips; lp = lpips.LPIPS(net="vgg").to(dev).eval()
    except Exception: pass
    s_emb, t_emb, psnr_sum, lpips_sum, n = [], [], 0.0, 0.0, 0
    for b in dl:
        x = b["data"].to(dev)
        out = model(x, "image", decode=True)
        s_emb.append(out.semantic.float().cpu()); t_emb.append(teacher_pool(teacher, x, tsize).float().cpu())
        psnr_sum += psnr_from_minus1_1(out.reconstruction, x) * x.shape[0]
        if lp is not None:
            lpips_sum += lp(out.reconstruction.clamp(-1, 1), x).mean().item() * x.shape[0]
        n += x.shape[0]
    return {"student": torch.cat(s_emb), "teacher": torch.cat(t_emb), "psnr": psnr_sum / n,
            "lpips": (lpips_sum / n) if lp is not None else None}


@torch.no_grad()
def run_cifar(model, teacher, tsize, args, dev):
    import torchvision
    from torchvision import transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    tr = torchvision.datasets.CIFAR100(args.cifar_root, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(args.cifar_root, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(0)
    tr_idx = torch.randperm(len(tr), generator=g)[:args.cifar_train]; te_idx = torch.randperm(len(te), generator=g)[:args.cifar_test]
    def enc(ds, idx):
        from torch.utils.data import DataLoader, Subset
        se, tee, ys = [], [], []
        for x, y in DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch, num_workers=4):
            x = F.interpolate(x.to(dev), size=(args.resolution, args.resolution), mode="bilinear", align_corners=False)
            se.append(model(x, "image", decode=False).semantic.float().cpu()); tee.append(teacher_pool(teacher, x, tsize).float().cpu()); ys.append(y)
        return torch.cat(se), torch.cat(tee), torch.cat(ys)
    s_tr, t_tr, y_tr = enc(tr, tr_idx); s_te, t_te, y_te = enc(te, te_idx)
    return {"student": knn_top1(s_tr, y_tr, s_te, y_te, args.knn_k), "teacher": knn_top1(t_tr, y_tr, t_te, y_te, args.knn_k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--manifest", default="data/manifests/openimages.jsonl")
    ap.add_argument("--n-images", type=int, default=1024); ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--teacher", default="google/siglip2-so400m-patch16-384")
    ap.add_argument("--init-siglip2", action="store_true"); ap.add_argument("--init-patchify", action="store_true")
    ap.add_argument("--semantic-dim", type=int, default=1152); ap.add_argument("--mlp-ratio", type=float, default=4304 / 1152)
    ap.add_argument("--window", type=int, default=2); ap.add_argument("--rgat-impl", default="dense")
    ap.add_argument("--cifar-root", default=os.environ.get("MAVT_CIFAR_ROOT", "data/eval/cifar100"))
    ap.add_argument("--cifar-train", type=int, default=5000); ap.add_argument("--cifar-test", type=int, default=1000)
    ap.add_argument("--knn-k", type=int, default=20); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model = build_model(args, dev); teacher, tsize = load_teacher(args.teacher, dev)
    print(f"[setup] student semantic_dim={model.understanding_decoder.proj.out_features} teacher img={tsize} ({time.time()-t0:.0f}s)", flush=True)
    oi = run_openimages(model, teacher, tsize, args, dev)
    print(f"[openimages] n={oi['student'].shape[0]} psnr={oi['psnr']:.2f} lpips={oi['lpips']} ({time.time()-t0:.0f}s)", flush=True)
    knn = run_cifar(model, teacher, tsize, args, dev)
    print(f"[cifar100] knn student={knn['student']:.3f} teacher={knn['teacher']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    same_dim = oi["student"].shape[1] == oi["teacher"].shape[1]
    align = F.cosine_similarity(oi["student"], oi["teacher"], dim=-1).mean().item() if same_dim else None
    align_c = None
    if same_dim:
        sc = oi["student"] - oi["student"].mean(0, keepdim=True); tc = oi["teacher"] - oi["teacher"].mean(0, keepdim=True)
        align_c = F.cosine_similarity(sc, tc, dim=-1).mean().item()
    student = {"eff_rank": effective_rank(oi["student"]), "pair_cos": mean_pairwise_cosine(oi["student"]),
               "feat_std": mean_feature_std(oi["student"]), "knn_top1": knn["student"], "psnr": oi["psnr"],
               "lpips": oi["lpips"], "align_cos_to_teacher": align, "align_cos_centered": align_c}
    ref = {"eff_rank": effective_rank(oi["teacher"]), "pair_cos": mean_pairwise_cosine(oi["teacher"]),
           "feat_std": mean_feature_std(oi["teacher"]), "knn_top1": knn["teacher"]}
    gates = evaluate_gates(student, S1_GATES)
    for k, g in gates.items():
        r = ref.get(k); rs = f"{r:.3f}" if isinstance(r, float) else "  -"
        print(f"{k:<20s} {g['value'] if g['value'] is None else round(g['value'], 3):>8}  {g['op']} {g['threshold']:<7} {'PASS' if g['pass'] else 'FAIL'}   ref={rs}")
    print(f"  align(student,teacher) cos = {align}   centered = {align_c}")
    ok = all(g["pass"] for g in gates.values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ckpt": args.ckpt, "student": student, "teacher_reference": ref, "gates": gates, "all_pass": ok,
               "seconds": round(time.time() - t0, 1)}, open(args.out, "w"), indent=2)
    print(f"ALL_GATES_PASS={ok}  -> {args.out}"); print("GATES_DONE", flush=True)


if __name__ == "__main__":
    main()
