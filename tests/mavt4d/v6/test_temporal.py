"""Order-free KDA write: bit-identity without evidence, order invariance, the v6 C rule, the delta rule, forgetting."""
import torch
import torch.nn.functional as F

from mavt4d.v6.temporal import forget_factor, read, set_write


def _case(seed=0, b=2, n=7, k=5, d=3):
    g = torch.Generator().manual_seed(seed)
    mem = torch.randn(b, k, d, generator=g)
    lam = torch.rand(b, k, generator=g) * 3
    keys = torch.randn(b, n, k, generator=g).softmax(-1)
    values = torch.randn(b, n, d, generator=g)
    beta = torch.rand(b, n, generator=g)
    return mem, lam, keys, values, beta


def test_no_evidence_and_no_time_leaves_memory_bit_identical():
    mem, lam, keys, values, beta = _case()
    alpha = forget_factor(torch.randn(2, 5), torch.zeros(2, 1))
    out, out_lam = set_write(mem, lam, keys, values, torch.zeros_like(beta), alpha)
    assert torch.equal(out, mem) and torch.equal(out_lam, lam)


def test_write_does_not_depend_on_token_order():
    mem, lam, keys, values, beta = _case(1)
    perm = torch.randperm(keys.shape[1])
    a, a_lam = set_write(mem, lam, keys, values, beta)
    b, b_lam = set_write(mem, lam, keys[:, perm], values[:, perm], beta[:, perm])
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a_lam, b_lam)


def test_one_hot_keys_give_the_v6_weighted_mean_rule():
    mem, lam, _, values, beta = _case(2)
    rows = torch.tensor([[0, 0, 1, 3, 3, 3, 4], [2, 2, 2, 2, 0, 1, 1]])
    keys = F.one_hot(rows, 5).float()
    out, out_lam = set_write(mem, lam, keys, values, beta)
    for b in range(2):
        for k in range(5):
            mine = rows[b] == k
            total = beta[b, mine].sum()
            mean = (beta[b, mine, None] * values[b, mine]).sum(0) / total.clamp_min(1e-12)
            expect = mem[b, k] + total / (lam[b, k] + total) * (mean - mem[b, k]) if mine.any() else mem[b, k]
            torch.testing.assert_close(out[b, k], expect)
            torch.testing.assert_close(out_lam[b, k], lam[b, k] + total)


def test_one_token_with_a_unit_key_is_the_delta_rule():
    mem, _, _, values, beta = _case(3, n=1)
    key = F.normalize(torch.randn(2, 1, 5), dim=-1)
    lam = torch.full((2, 5), .7)
    out, _ = set_write(mem, lam, key, values, beta)
    step = (beta / (.7 + beta))[:, :, None]                                   # [B,1,1]
    expect = mem + step * key.transpose(-1, -2) @ (values - key @ mem)
    torch.testing.assert_close(out, expect)


def test_forgetting_composes_over_time_and_keeps_slow_rows():
    rate = torch.tensor([[-30., 0., 3.]])
    a1, a2 = forget_factor(rate, torch.tensor([[.4]])), forget_factor(rate, torch.tensor([[.6]]))
    torch.testing.assert_close(a1 * a2, forget_factor(rate, torch.tensor([[1.]])))
    assert a1[0, 0] > 1 - 1e-6 and a1[0, 2] < a1[0, 1] < 1


def test_strong_writes_with_orthonormal_keys_are_recalled():
    keys = torch.linalg.qr(torch.randn(8, 8))[0][:4][None]                    # 4 orthonormal unit keys
    values = torch.randn(1, 4, 6)
    out, _ = set_write(torch.zeros(1, 8, 6), torch.zeros(1, 8), keys, values, torch.full((1, 4), 1e3))
    torch.testing.assert_close(read(out, keys), values, atol=1e-5, rtol=1e-5)


def test_gradients_reach_keys_values_and_strengths():
    mem, lam, keys, values, beta = _case(4)
    logits = torch.randn_like(keys, requires_grad=True)
    values.requires_grad_(True)
    beta.requires_grad_(True)
    out, _ = set_write(mem, lam, logits.softmax(-1), values, beta)
    out.square().sum().backward()
    for t in (logits, values, beta):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
