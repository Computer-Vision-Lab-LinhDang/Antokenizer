import torch

from mavt.latent.discrete_vq import MultiCodebookVQ


def test_vq_uses_multiple_codes():
    vq = MultiCodebookVQ(embed_dim=16, num_codebooks=4, codebook_size=8)
    used = set()
    for _ in range(4):
        out = vq(torch.randn(2, 6, 16))
        used.update(out["indices"].reshape(-1).tolist())
    assert len(used) > 1
