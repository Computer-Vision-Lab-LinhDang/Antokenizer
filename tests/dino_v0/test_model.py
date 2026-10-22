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
