"""S1 acceptance-gate metrics for the pooled semantic embedding and reconstruction.

Thresholds were calibrated against the frozen SigLIP2-SO400M teacher measured the
same way (Open Images 1024 imgs, CIFAR-100 kNN 5k/1k): teacher eff_rank 532/1152,
pair_cos 0.525, kNN 0.791; untrained student eff_rank 90, pair_cos 0.90, kNN 0.015.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F

S1_GATES: Dict[str, Dict] = {
    "eff_rank": {"op": ">=", "threshold": 200.0,
                 "desc": "effective rank of centered pooled embeddings (init 90, teacher 532)"},
    "pair_cos": {"op": "<=", "threshold": 0.60,
                 "desc": "mean pairwise cosine of pooled embeddings (teacher 0.525)"},
    "align_cos_centered": {"op": ">=", "threshold": 0.60,
                 "desc": "cosine after removing each side's batch mean — what centered distillation optimises"},
    "knn_top1": {"op": ">=", "threshold": 0.40,
                 "desc": "cosine kNN top-1 on CIFAR-100 (teacher 0.791)"},
    "psnr": {"op": ">=", "threshold": 26.0, "desc": "reconstruction PSNR @256 on [0,1]"},
}


def mean_pairwise_cosine(e: torch.Tensor) -> float:
    e = F.normalize(e.float(), dim=-1)
    n = e.shape[0]
    sim = e @ e.T
    mask = ~torch.eye(n, dtype=torch.bool, device=e.device)
    return sim[mask].mean().item()


def effective_rank(e: torch.Tensor) -> float:
    c = e.float() - e.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(c)
    s = s[s > 0]
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def mean_feature_std(e: torch.Tensor) -> float:
    return e.float().std(dim=0, unbiased=False).mean().item()


def psnr_from_minus1_1(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = (pred.float().clamp(-1, 1) + 1) / 2
    t = (target.float().clamp(-1, 1) + 1) / 2
    mse = ((p - t) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def knn_top1(train_e, train_y, test_e, test_y, k: int = 20) -> float:
    tr = F.normalize(train_e.float(), dim=-1)
    te = F.normalize(test_e.float(), dim=-1)
    idx = (te @ tr.T).topk(k, dim=-1).indices
    votes = train_y[idx]
    n_cls = int(max(train_y.max(), test_y.max()).item()) + 1
    pred = F.one_hot(votes, n_cls).sum(1).argmax(-1)
    return (pred == test_y).float().mean().item()


def evaluate_gates(metrics: Dict[str, float], gates: Dict[str, Dict] = S1_GATES) -> Dict[str, Dict]:
    out = {}
    for name, g in gates.items():
        v = metrics.get(name)
        if v is None:
            out[name] = {"value": None, "threshold": g["threshold"], "op": g["op"], "pass": False}
            continue
        ok = (v >= g["threshold"]) if g["op"] == ">=" else (v <= g["threshold"])
        out[name] = {"value": float(v), "threshold": g["threshold"], "op": g["op"], "pass": bool(ok)}
    return out


__all__ = ["S1_GATES", "mean_pairwise_cosine", "effective_rank", "mean_feature_std",
           "psnr_from_minus1_1", "knn_top1", "evaluate_gates"]
