import torch

from mavt.encoder.rope4d import RoPE4D, rotate_half


def test_rope4d_keeps_t_and_z_groups_static_when_zero():
    rope = RoPE4D(head_dim=16)
    q = torch.randn(2, 3, 5, 16)
    k = torch.randn(2, 3, 5, 16)
    positions = torch.zeros(2, 5, 4, dtype=torch.long)
    positions[:, :, 1] = torch.arange(5)
    positions[:, :, 2] = torch.arange(5)

    q_out, k_out = rope(q, k, positions)
    group_dim = rope.group_dim

    torch.testing.assert_close(q_out[..., :group_dim], q[..., :group_dim])
    torch.testing.assert_close(k_out[..., :group_dim], k[..., :group_dim])
    torch.testing.assert_close(q_out[..., -group_dim:], q[..., -group_dim:])
    torch.testing.assert_close(k_out[..., -group_dim:], k[..., -group_dim:])


def test_rope4d_matches_manual_axis_rotation():
    rope = RoPE4D(head_dim=16)
    q = torch.randn(1, 1, 4, 16)
    k = torch.randn(1, 1, 4, 16)
    positions = torch.zeros(1, 4, 4, dtype=torch.long)
    positions[0, :, 1] = torch.tensor([0, 1, 2, 3])

    q_out, _ = rope(q, k, positions)
    x_group = q[..., rope.group_dim : rope.group_dim * 2]
    freqs = positions[..., 1].float().unsqueeze(-1) * rope.inv_freq
    cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(1)
    expected = (x_group * cos) + (rotate_half(x_group) * sin)
    torch.testing.assert_close(q_out[..., rope.group_dim : rope.group_dim * 2], expected)
