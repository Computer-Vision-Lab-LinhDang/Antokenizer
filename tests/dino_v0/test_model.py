import torch

from mavt.dino_v0.config import tiny_config
from mavt.dino_v0.model import MAVTDinoV0


def _model(**kw):
    torch.manual_seed(0)
    return MAVTDinoV0(tiny_config(**kw))


def test_image_step_and_reconstruct():
    m = _model()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    out = m(x, "image")
    assert torch.isfinite(out["loss"])
    assert "fut" not in out["parts"] and "gate" not in out["parts"]
    out["loss"].backward()
    m.eval()
    assert m.reconstruct(x, "image").shape == x.shape


def test_video_step_and_reconstruct():
    m = _model()
    x = torch.rand(2, 3, 6, 64, 64) * 2 - 1
    out = m(x, "video")
    assert torch.isfinite(out["loss"])
    assert {"rgb", "feat", "fut", "kl", "var", "sem", "gate"} <= set(out["parts"])
    assert out["diag"]["delta_abs_cells"] == 0          # offset head starts at zero
    out["loss"].backward()
    assert all(p.grad is None for p in m.encoder.dino.parameters())
    m.eval()
    assert m.reconstruct(x, "video").shape == x.shape


def test_frozen_dino_untouched_by_optimizer():
    m = _model()
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert not any(n.startswith("encoder.dino.") for n in trainable)


def _orbit_cams(B, T, size=64):
    """Cameras on a circle of radius 2 looking at the origin, OpenCV axes."""
    import math
    c2w = torch.zeros(B, T, 4, 4)
    for t in range(T):
        a = 2 * math.pi * t / T
        o = torch.tensor([2 * math.cos(a), 2 * math.sin(a), 0.3])
        z = -o / o.norm()
        x = torch.linalg.cross(z, torch.tensor([0.0, 0.0, -1.0])); x = x / x.norm()
        y = torch.linalg.cross(z, x)
        c2w[:, t, :3, 0], c2w[:, t, :3, 1], c2w[:, t, :3, 2], c2w[:, t, :3, 3] = x, y, z, o
        c2w[:, t, 3, 3] = 1
    f = size / 2 / math.tan(0.3456)
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]]).expand(B, T, 3, 3).clone()
    return dict(c2w=c2w, K=K)


def test_posed_views_step():
    m = _model()
    x = torch.rand(2, 3, 5, 64, 64) * 2 - 1
    cam = _orbit_cams(2, 5)
    out = m(x, "threed", cam)
    assert torch.isfinite(out["loss"]) and "fut" in out["parts"]
    out["loss"].backward()
    # zero-init conditioning still receives gradient through its last layer
    assert m.decoder.ray_q.weight.grad is not None and m.decoder.ray_q.weight.grad.abs().sum() > 0
    assert m.cd.d_state.pose_proj[-1].weight.grad.abs().sum() > 0
    m.eval()
    assert m.reconstruct(x, "threed", cam).shape == x.shape


def test_rays_unit_directions():
    from mavt.dino_v0.camera import patch_rays
    cam = _orbit_cams(1, 2)
    r = patch_rays(cam["c2w"], cam["K"], 64, 4)
    assert r.shape == (1, 2, 16, 6)
    assert torch.allclose(r[..., :3].norm(dim=-1), torch.ones(1, 2, 16), atol=1e-5)
