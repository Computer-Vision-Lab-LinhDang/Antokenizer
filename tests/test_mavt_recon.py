# tests/test_mavt_recon.py

import pytest
import torch

from mavt.mavt_recon import MAVTRecon


# Fixtures
@pytest.fixture(scope="module")
def model():
    return MAVTRecon()


@pytest.fixture
def image():
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def video():
    return torch.randn(1, 3, 4, 64, 64)


# IMAGE TESTS
def test_image_inference(model, image):
    model.eval()

    with torch.no_grad():
        output = model(image, compute_loss=False)

    assert output.recon.shape == image.shape
    assert output.loss is None
    assert "psnr" in output.logs
    assert "l1" in output.logs


def test_image_training(model, image):
    model.train()

    output = model(image, compute_loss=True)

    assert output.recon.shape == image.shape
    assert output.loss is not None
    assert torch.isfinite(output.loss).all()

    assert "psnr" in output.logs
    assert "l1" in output.logs


# VIDEO TESTS
def test_video_inference(model, video):
    model.eval()

    with torch.no_grad():
        output = model(video, compute_loss=False)

    assert output.recon.shape == video.shape
    assert output.loss is None
    assert "psnr" in output.logs
    assert "l1" in output.logs


def test_video_training(model, video):
    model.train()

    output = model(video, compute_loss=True)

    assert output.recon.shape == video.shape
    assert output.loss is not None
    assert torch.isfinite(output.loss).all()

    assert "psnr" in output.logs
    assert "l1" in output.logs


# MODE CONSISTENCY TEST
def test_train_vs_inference_consistency(model, image):
    """
    Ensure inference does not break shape / outputs vs training
    """
    model.train()
    train_out = model(image, compute_loss=True)

    model.eval()
    with torch.no_grad():
        eval_out = model(image, compute_loss=False)

    assert train_out.recon.shape == eval_out.recon.shape
    assert train_out.loss is not None
    assert eval_out.loss is None


# ENCODE / DECODE TEST
def test_encode_decode_image(model, image):
    model.eval()

    with torch.no_grad():
        z = model.encode(image)
        recon = model.decode(z)

    assert torch.is_tensor(z)
    assert recon.shape == image.shape


# LATENT TEST
def test_return_latent(model, image):
    model.eval()

    with torch.no_grad():
        output = model(image, return_latent=True)

    assert output.latent is not None
    assert torch.is_tensor(output.latent)


# NUMERICAL STABILITY
def test_no_nan(model, image):
    model.train()

    output = model(image, compute_loss=True)

    assert torch.isfinite(output.recon).all()
    assert torch.isfinite(output.loss).all()

    for v in output.logs.values():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all()