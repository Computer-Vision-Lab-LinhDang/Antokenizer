"""v3 pipeline contracts (TDD): behaviours the pilot showed missing or silently wrong."""
from __future__ import annotations
import json
import pytest
import torch
from conftest import requires_hf
from mavt.model.mavt import MAVT
from mavt.losses.losses import MAVTLoss

TINY = dict(embed_dim=128, num_heads=8, num_blocks=2, patch_size=16, t_patch=2,
            latent_dim=8, semantic_dim=32, dec_dim=128, num_dec_attn_blocks=1,
            use_gradient_checkpointing=False)


def _tiny_model() -> MAVT:
    torch.manual_seed(0)
    return MAVT(**TINY)


def test_video_recon_returns_all_T_frames():
    m = _tiny_model()
    m.prepare_for_modalities([{"modality": "video", "resolution": 64, "frames": 8}])
    x = torch.randn(1, 3, 8, 64, 64)
    out = m(x, "video", decode=True)
    assert out.reconstruction.shape == x.shape, (
        f"video recon must cover all T={x.shape[2]} frames, got {tuple(out.reconstruction.shape)}")


def test_3d_recon_returns_all_three_planes():
    m = _tiny_model()
    m.prepare_for_modalities([{"modality": "threed", "resolution": 64}])
    x = torch.randn(1, 3, 3, 64, 64)
    out = m(x, "threed", decode=True)
    assert out.reconstruction.shape == x.shape


def test_understanding_head_receives_gradient_with_distill():
    m = _tiny_model()
    m.prepare_for_modalities([{"modality": "image", "resolution": 64}])
    m.train()
    loss_fn = MAVTLoss(w_sem=0.5, use_lpips=False, active_modalities=("image",))
    x = torch.randn(2, 3, 64, 64)
    out = m(x, "image", decode=True)
    teacher = torch.randn(2, TINY["semantic_dim"])
    losses = loss_fn(pred=out.reconstruction, target=x, loss_kl=out.loss_kl,
                     slot_diversity=out.cd_metrics["slot_diversity"], modality="image",
                     semantic_embed=out.semantic, teacher_embed=teacher)
    losses["loss"].backward()
    grads = [p.grad for p in m.understanding_decoder.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads), "understanding head got no gradient"
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_loss_rejects_shape_mismatch():
    loss_fn = MAVTLoss(use_lpips=False, active_modalities=("video",))
    pred = torch.randn(1, 3, 4, 16, 16)
    target = torch.randn(1, 3, 8, 16, 16)
    with pytest.raises(ValueError, match="shape"):
        loss_fn(pred=pred, target=target, loss_kl=torch.tensor(0.0),
                slot_diversity=torch.tensor(0.0), modality="video")


def test_siglip2_strict_load_raises_on_unloadable_model():
    m = _tiny_model()
    with pytest.raises(RuntimeError, match="SigLIP2"):
        m.load_siglip2_weights("definitely/not-a-real-model-xyz", freeze_stages=0, strict=True)


def test_manifest_image_dataset_contract(tmp_path):
    from PIL import Image
    from mavt.data.datasets import ManifestImageDataset
    img_path = tmp_path / "a.jpg"
    Image.new("RGB", (40, 30), (255, 0, 0)).save(img_path)
    manifest = tmp_path / "images.jsonl"
    manifest.write_text(json.dumps({"path": str(img_path), "caption": "red"}) + "\n")
    ds = ManifestImageDataset(str(manifest), resolution=32)
    s = ds[0]
    assert s["data"].shape == (3, 32, 32)
    assert s["modality"] == "image" and s["caption"] == "red" and s["id"] == "a"
    assert -1.0 <= s["data"].min() and s["data"].max() <= 1.0


def test_manifest_video_dataset_fails_loud_on_broken_file(tmp_path):
    from mavt.data.datasets import ManifestVideoDataset
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(json.dumps({"path": str(bad)}) + "\n")
    ds = ManifestVideoDataset(str(manifest), n_frames=4, resolution=32)
    with pytest.raises(RuntimeError, match="Video load failed"):
        ds[0]


def test_vicreg_penalises_collapsed_embeddings_more_than_spread():
    from mavt.losses.losses import vicreg_loss
    torch.manual_seed(0)
    spread = torch.randn(64, 32)
    collapsed = torch.randn(64, 1).expand(64, 32) * 0.01 + 1.0
    assert vicreg_loss(collapsed) > vicreg_loss(spread) * 5


def test_centered_cosine_distill_ignores_shared_offset():
    from mavt.losses.losses import cosine_distill_loss
    torch.manual_seed(0)
    s, t = torch.randn(16, 8), torch.randn(16, 8)
    c = 100.0 * torch.ones(8)
    plain_shifted = cosine_distill_loss(s + c, t + c)
    centered_shifted = cosine_distill_loss(s + c, t + c, center=True)
    centered = cosine_distill_loss(s, t, center=True)
    assert plain_shifted < 0.05
    assert abs(centered_shifted - centered) < 1e-3 and centered > 0.5


def test_mavt_loss_exposes_vic_and_lr_override():
    from mavt.losses.losses import MAVTLoss
    lf = MAVTLoss(use_lpips=False, w_vic=0.1, distill_center=True, active_modalities=("image",))
    pred = torch.randn(4, 3, 16, 16); tgt = pred.clone()
    out = lf(pred=pred, target=tgt, loss_kl=torch.tensor(0.0), slot_diversity=torch.tensor(0.0),
             modality="image", semantic_embed=torch.randn(4, 8), teacher_embed=torch.randn(4, 8))
    assert "loss_vic" in out and out["loss_vic"].item() > 0
    from mavt.training.lightning_module import MAVTLightningModule
    import inspect
    sig = inspect.signature(MAVTLightningModule.__init__).parameters
    for k in ("lr", "w_vic", "distill_center", "init_siglip2_patchify", "rgat_impl"):
        assert k in sig, k


def test_openvid_manifest_builder_joins_captions(tmp_path):
    from mavt.data.manifest import build_openvid
    vids = tmp_path / "videos" / "sub"; vids.mkdir(parents=True)
    (vids / "clip_a.mp4").write_bytes(b"x"); (vids / "clip_b.mp4").write_bytes(b"x")
    csv_path = tmp_path / "OpenVid-1M.csv"
    csv_path.write_text("video,caption,frame,fps\nclip_a.mp4,\"a dog runs\",120,24\nclip_b.mp4,\"a cat sits\",96,24\n")
    out = tmp_path / "openvid.jsonl"
    n = build_openvid(str(tmp_path / "videos"), str(csv_path), str(out))
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 2 and {r["caption"] for r in recs} == {"a dog runs", "a cat sits"}
    assert all(r["path"].endswith(".mp4") for r in recs)
