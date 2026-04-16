import torch

from mavt.latent.discrete_fsq import DiscreteFSQHead


def test_fsq_has_straight_through_gradient():
    head = DiscreteFSQHead(embed_dim=8, levels=(8, 8, 8, 5, 5, 5))
    x = torch.randn(2, 4, 8, requires_grad=True)
    out = head(x)
    out["quantized"].sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
