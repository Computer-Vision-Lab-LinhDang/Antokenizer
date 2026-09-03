"""Metric contracts for the S1 acceptance gates."""
import torch
import pytest
from mavt.evaluation.gates import (
    mean_pairwise_cosine, effective_rank, mean_feature_std,
    psnr_from_minus1_1, knn_top1, evaluate_gates, S1_GATES,
)


def test_pairwise_cosine_identical_is_one_and_orthogonal_is_zero():
    assert abs(mean_pairwise_cosine(torch.ones(8, 16)) - 1.0) < 1e-6
    assert abs(mean_pairwise_cosine(torch.eye(8, 16))) < 1e-6


def test_effective_rank_bounds():
    torch.manual_seed(0)
    collapsed = torch.randn(64, 1) @ torch.randn(1, 32) + 1e-4 * torch.randn(64, 32)
    spread = torch.randn(64, 32)
    er_c, er_s = effective_rank(collapsed), effective_rank(spread)
    assert er_c < 3.0 and er_s > 20.0 and er_s <= 32.0


def test_mean_feature_std_zero_for_constant():
    assert mean_feature_std(torch.ones(10, 5)) == 0.0


def test_psnr_perfect_and_known_value():
    x = torch.rand(2, 3, 8, 8) * 2 - 1
    assert psnr_from_minus1_1(x, x) > 80
    y01 = ((x + 1) / 2 + 0.1).clamp(0, 1); y = y01 * 2 - 1
    assert 19.0 < psnr_from_minus1_1(y, x) < 21.5


def test_knn_top1_trivially_separable():
    train = torch.cat([torch.randn(50, 8) + 5, torch.randn(50, 8) - 5])
    ytr = torch.tensor([0] * 50 + [1] * 50)
    test = torch.cat([torch.randn(10, 8) + 5, torch.randn(10, 8) - 5])
    yte = torch.tensor([0] * 10 + [1] * 10)
    assert knn_top1(train, ytr, test, yte, k=5) == 1.0


def test_evaluate_gates_reports_pass_fail_per_gate():
    m = {"eff_rank": 250, "pair_cos": 0.2, "align_cos_centered": 0.8, "knn_top1": 0.5, "psnr": 27.0}
    out = evaluate_gates(m, S1_GATES)
    assert all(v["pass"] for v in out.values())
    m["pair_cos"] = 0.9
    out = evaluate_gates(m, S1_GATES)
    assert out["pair_cos"]["pass"] is False and out["eff_rank"]["pass"] is True
