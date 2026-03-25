"""Pytest suite for MAVT modules.

Tests are deliberately lightweight (small tensors, no SigLIP2 weights)
so they run quickly on CPU without network access.
"""
from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 2          # batch size used throughout
D = 64         # small embed_dim for fast tests
PATCH = 16     # spatial patch size
# IMG_H=64: (IMG_H//PATCH)^2 = 16 tokens > n_heads=8 (required by rope4d
# layout heuristic when seq_len is compared to num_heads).
# Decoder: (IMG_H//PATCH) * 2^4 = 4 * 16 = 64 = IMG_H ✓ (output = input).
IMG_H = 64
N_TOKENS = (IMG_H // PATCH) ** 2  # 16 — tokens per image frame


def _small_patchify_cfg():
    from mavt.config import PatchifyConfig
    return PatchifyConfig(
        patch_size_spatial=PATCH,
        patch_size_temporal=2,
        embed_dim=D,
        siglip2_model="google/siglip2-so400m-patch16-384",
        freeze_siglip2=False,  # won't load weights → fallback Conv2d
    )


def _small_encoder_cfg():
    from mavt.config import EncoderConfig
    return EncoderConfig(d_model=D, n_blocks_stage3=1, n_blocks_stage4=1)


def _small_latent_cfg():
    from mavt.config import LatentConfig
    return LatentConfig(d_encoder=D, latent_dim=8, d_understand=16)


def _small_decoder_cfg():
    from mavt.config import DecoderConfig
    # 4 PixelShuffle stages give 16× upscale; with PATCH=16 and IMG_H=64 this
    # produces 4*16=64 output pixels — matching the input for compute_loss.
    return DecoderConfig(
        latent_dim=8,
        d_model=D,
        n_attn_blocks=1,
        n_attn_heads=4,
        cnn_channels=(32, 16, 8, 4),
        out_channels=3,
        patch_size=PATCH,
    )


# Each PixelShuffle stage doubles spatial resolution; 4 stages → 2^4 = 16×.
# For IMG_H=64, PATCH=16: (IMG_H//PATCH) * _DEC_SCALE = 4 * 16 = 64 ✓
_DEC_SCALE = 2 ** 4  # 4 PixelShuffle stages → 16× upscale


def _small_mavt_cfg():
    from mavt.config import MAVTConfig
    return MAVTConfig(
        patchify=_small_patchify_cfg(),
        encoder=_small_encoder_cfg(),
        latent=_small_latent_cfg(),
        decoder=_small_decoder_cfg(),
    )


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestConfig:
    def test_mavt_config_defaults(self):
        from mavt.config import MAVTConfig
        cfg = MAVTConfig()
        assert cfg.d_model == cfg.patchify.embed_dim

    def test_mavt_config_d_model_property(self):
        from mavt.config import MAVTConfig, PatchifyConfig
        cfg = MAVTConfig(patchify=PatchifyConfig(embed_dim=512))
        assert cfg.d_model == 512

    def test_dataclass_fields_accessible(self):
        from mavt.config import EncoderConfig, LatentConfig, DecoderConfig
        enc = EncoderConfig()
        assert enc.n_blocks_stage3 + enc.n_blocks_stage4 > 0

        lat = LatentConfig()
        assert lat.latent_dim == 32

        dec = DecoderConfig()
        assert dec.out_channels == 3


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------

class TestTypes:
    def _make_patchify_output(self, B=2, N=4, D=64):
        from mavt.types import PatchifyOutput
        return PatchifyOutput(
            f_spatial=torch.randn(B, N, D),
            positions=torch.zeros(B, N, 4),
            raw_patches=torch.zeros(B, N, 3, PATCH, PATCH),
            modality="image",
        )

    def test_patchify_output_properties(self):
        out = self._make_patchify_output(B=3, N=7)
        assert out.batch_size == 3
        assert out.num_tokens == 7

    def test_patchify_output_optional_fields_default_none(self):
        out = self._make_patchify_output()
        assert out.raw_patches_temporal is None
        assert out.depth_signal is None

    def test_encoder_output_fields(self):
        from mavt.types import EncoderOutput
        enc = EncoderOutput(
            encoded=torch.randn(2, 4, 64),
            positions_out=torch.zeros(2, 4, 4),
        )
        assert enc.encoded.shape == (2, 4, 64)

    def test_latent_output_fields(self):
        from mavt.types import LatentOutput
        lat = LatentOutput(
            z=torch.randn(2, 4, 8),
            z_understand=torch.randn(2, 16),
            mu=torch.randn(2, 4, 8),
            log_var=torch.randn(2, 4, 8),
        )
        assert lat.z.shape == (2, 4, 8)

    def test_decoder_output_default_aux(self):
        from mavt.types import DecoderOutput
        dec = DecoderOutput(reconstruction=torch.randn(2, 3, 32, 32))
        assert dec.aux == {}


# ---------------------------------------------------------------------------
# module1_patchify.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patchify():
    from mavt.module1_patchify import Unified4DPatchify
    return Unified4DPatchify(_small_patchify_cfg()).eval()


class TestUnified4DPatchify:
    # --- detect_modality ---

    def test_detect_image(self, patchify):
        x = torch.randn(1, 3, 32, 32)
        assert patchify.detect_modality(x) == "image"

    def test_detect_video(self, patchify):
        x = torch.randn(1, 3, 4, 32, 32)
        assert patchify.detect_modality(x) == "video"

    def test_detect_invalid_rank_raises(self, patchify):
        with pytest.raises(ValueError):
            patchify.detect_modality(torch.randn(32))

    # --- image ---

    def test_image_output_shapes(self, patchify):
        x = torch.randn(B, 3, IMG_H, IMG_H)
        out = patchify(x, modality="image")
        N = N_TOKENS
        assert out.f_spatial.shape == (B, N, D)
        assert out.positions.shape == (B, N, 4)
        assert out.raw_patches.shape == (B, N, 3, PATCH, PATCH)
        assert out.modality == "image"

    def test_image_positions_z_zero(self, patchify):
        x = torch.randn(B, 3, IMG_H, IMG_H)
        out = patchify(x, modality="image")
        assert (out.positions[..., 0] == 0).all()  # t=0
        assert (out.positions[..., 3] == 0).all()  # z=0

    def test_image_auto_detection(self, patchify):
        x = torch.randn(B, 3, IMG_H, IMG_H)
        out = patchify(x)  # no modality arg
        assert out.modality == "image"

    # --- video ---

    def test_video_output_shapes(self, patchify):
        T, H = 4, IMG_H
        x = torch.randn(B, 3, T, H, H)
        out = patchify(x, modality="video")
        n_t = T // 2
        N_spatial = (H // PATCH) ** 2
        N = n_t * N_spatial
        assert out.f_spatial.shape == (B, N, D)
        assert out.positions.shape == (B, N, 4)
        assert out.raw_patches_temporal is not None
        assert out.modality == "video"

    def test_video_temporal_patches_shape(self, patchify):
        T, H = 4, IMG_H
        x = torch.randn(B, 3, T, H, H)
        out = patchify(x, modality="video")
        N_spatial = (H // PATCH) ** 2
        # Expected: (B, N_spatial, T_used, C, p, p)
        rpt = out.raw_patches_temporal
        assert rpt.shape == (B, N_spatial, T, 3, PATCH, PATCH)

    # --- 3D triplane ---

    def test_3d_output_shapes(self, patchify):
        S = IMG_H
        x = torch.randn(B, 3, 3, S, S)   # (B, n_planes=3, C_tri=3, S, S)
        out = patchify(x, modality="3d")
        N_per_plane = (S // PATCH) ** 2
        N_total = 3 * N_per_plane
        assert out.f_spatial.shape == (B, N_total, D)
        assert out.positions.shape == (B, N_total, 4)
        assert out.modality == "3d"
        assert out.depth_signal is not None

    def test_3d_wrong_planes_raises(self, patchify):
        x = torch.randn(B, 2, 3, IMG_H, IMG_H)
        with pytest.raises(ValueError, match="triplane"):
            patchify(x, modality="3d")

    # --- helpers ---

    def test_build_positions_grid_values(self, patchify):
        B_, n_t, n_h, n_w = 1, 2, 2, 3
        pos = patchify._build_positions(B_, n_t, n_h, n_w, device=torch.device("cpu"))
        assert pos.shape == (B_, n_t * n_h * n_w, 4)
        # z is always 0
        assert (pos[..., 3] == 0).all()
        # t spans [0, n_t)
        assert pos[0, :, 0].min().item() == 0.0
        assert pos[0, :, 0].max().item() == float(n_t - 1)
        # x spans [0, n_w), y spans [0, n_h)
        assert pos[0, :, 1].max().item() == float(n_w - 1)
        assert pos[0, :, 2].max().item() == float(n_h - 1)

    def test_extract_raw_patches(self, patchify):
        x = torch.randn(B, 3, 32, 32)
        patches = patchify._extract_raw_patches(x, 32 // PATCH, 32 // PATCH)
        assert patches.shape == (B, (32 // PATCH) ** 2, 3, PATCH, PATCH)


# ---------------------------------------------------------------------------
# module5_encoder.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder():
    from mavt.module5_encoder import MAVTEncoder
    return MAVTEncoder(_small_encoder_cfg()).eval()


@pytest.fixture(scope="module")
def sample_patch_output():
    """Synthetic PatchifyOutput with small D.

    N=16 (4×4 grid) ensures seq_len > n_heads (8 for D=64) so that
    rope4d's layout heuristic correctly identifies (B, seq, heads, head_dim).
    """
    from mavt.types import PatchifyOutput
    N = N_TOKENS  # 16
    return PatchifyOutput(
        f_spatial=torch.randn(B, N, D),
        positions=torch.zeros(B, N, 4),
        raw_patches=torch.zeros(B, N, 3, PATCH, PATCH),
        modality="image",
    )


class TestMAVTEncoder:
    def test_output_shape_matches_input(self, encoder, sample_patch_output):
        out = encoder(sample_patch_output)
        assert out.encoded.shape == sample_patch_output.f_spatial.shape
        assert out.positions_out.shape == sample_patch_output.positions.shape

    def test_output_is_encoder_output_type(self, encoder, sample_patch_output):
        from mavt.types import EncoderOutput
        out = encoder(sample_patch_output)
        assert isinstance(out, EncoderOutput)

    def test_encoder_changes_values(self, encoder, sample_patch_output):
        out = encoder(sample_patch_output)
        # encoded should differ from input (not identity)
        assert not torch.allclose(out.encoded, sample_patch_output.f_spatial, atol=1e-3)

    def test_encoder_no_nan(self, encoder, sample_patch_output):
        out = encoder(sample_patch_output)
        assert not torch.isnan(out.encoded).any()

    def test_different_seq_lengths(self, encoder):
        # N must exceed n_heads (8 for D=64) to avoid rope4d layout ambiguity.
        from mavt.types import PatchifyOutput
        for N in [9, 16, 25]:
            patch_out = PatchifyOutput(
                f_spatial=torch.randn(1, N, D),
                positions=torch.zeros(1, N, 4),
                raw_patches=torch.zeros(1, N, 3, PATCH, PATCH),
                modality="image",
            )
            out = encoder(patch_out)
            assert out.encoded.shape == (1, N, D)


# ---------------------------------------------------------------------------
# module6_latent.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def latent_proj():
    from mavt.module6_latent import ContinuousLatentProjection
    return ContinuousLatentProjection(_small_latent_cfg())


@pytest.fixture(scope="module")
def sample_encoder_output():
    from mavt.types import EncoderOutput
    N = N_TOKENS  # 16
    return EncoderOutput(
        encoded=torch.randn(B, N, D),
        positions_out=torch.zeros(B, N, 4),
    )


class TestContinuousLatentProjection:
    def test_output_shapes(self, latent_proj, sample_encoder_output):
        latent_proj.eval()
        out = latent_proj(sample_encoder_output)
        N = sample_encoder_output.encoded.shape[1]
        assert out.z.shape == (B, N, 8)          # latent_dim=8
        assert out.z_understand.shape == (B, 16)  # d_understand=16
        assert out.mu.shape == (B, N, 8)
        assert out.log_var.shape == (B, N, 8)

    def test_eval_mode_uses_mu(self, latent_proj, sample_encoder_output):
        latent_proj.eval()
        out = latent_proj(sample_encoder_output)
        # In eval mode reparameterize returns mu directly
        assert torch.allclose(out.z, out.mu)

    def test_train_mode_adds_noise(self, latent_proj, sample_encoder_output):
        latent_proj.train()
        out = latent_proj(sample_encoder_output)
        # z should differ from mu in training (stochastic)
        assert not torch.allclose(out.z, out.mu)
        latent_proj.eval()

    def test_kl_loss_scalar(self, latent_proj, sample_encoder_output):
        latent_proj.eval()
        out = latent_proj(sample_encoder_output)
        kl = latent_proj.kl_loss(out)
        assert kl.ndim == 0
        assert not torch.isnan(kl)

    def test_kl_loss_positive_for_non_unit_gaussian(self, latent_proj):
        from mavt.types import LatentOutput
        N = N_TOKENS
        # mu far from 0 → KL should be large
        mu = torch.full((B, N, 8), 10.0)
        log_var = torch.zeros(B, N, 8)
        out = LatentOutput(
            z=mu, z_understand=torch.zeros(B, 16), mu=mu, log_var=log_var
        )
        kl = latent_proj.kl_loss(out)
        assert kl.item() > 1.0

    def test_variational_projection_shapes(self):
        from mavt.module6_latent import VariationalProjection
        vp = VariationalProjection(d_encoder=D, latent_dim=8)
        x = torch.randn(B, 4, D)
        mu, lv = vp(x)
        assert mu.shape == (B, 4, 8)
        assert lv.shape == (B, 4, 8)


# ---------------------------------------------------------------------------
# module7_decoder.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def decoder():
    from mavt.module7_decoder import AsymmetricDecoder
    return AsymmetricDecoder(_small_decoder_cfg()).eval()


def _make_latent(B=B, N=None, latent_dim=8, d_understand=16):
    N = N if N is not None else N_TOKENS  # default: 16
    from mavt.types import LatentOutput
    return LatentOutput(
        z=torch.randn(B, N, latent_dim),
        z_understand=torch.randn(B, d_understand),
        mu=torch.randn(B, N, latent_dim),
        log_var=torch.randn(B, N, latent_dim),
    )


class TestAsymmetricDecoder:
    def test_image_decode_shape(self, decoder):
        N = N_TOKENS   # 16 tokens (4×4 grid)
        lat = _make_latent(N=N)
        pos = torch.zeros(B, N, 4)
        out = decoder(lat, pos, target_shape=(B, 3, IMG_H, IMG_H), modality="image")
        # 4 PixelShuffle stages → 16× → (IMG_H//PATCH) * 16 = IMG_H
        expected_out = (IMG_H // PATCH) * _DEC_SCALE
        assert out.reconstruction.shape == (B, 3, expected_out, expected_out)

    def test_video_decode_shape(self, decoder):
        T = 4
        n_t = T // 2
        N_spatial = N_TOKENS
        N = n_t * N_spatial
        lat = _make_latent(N=N)
        pos = torch.zeros(B, N, 4)
        out = decoder(lat, pos, target_shape=(B, 3, T, IMG_H, IMG_H), modality="video")
        expected_out = (IMG_H // PATCH) * _DEC_SCALE
        assert out.reconstruction.shape == (B, 3, n_t, expected_out, expected_out)

    def test_3d_decode_shape(self, decoder):
        S = IMG_H
        N_per_plane = (S // PATCH) ** 2
        N = 3 * N_per_plane
        lat = _make_latent(N=N)
        pos = torch.zeros(B, N, 4)
        out = decoder(lat, pos, target_shape=(B, 3, 3, S, S), modality="3d")
        # 3D decoder decodes only the XY plane (first N_per_plane tokens) as an image.
        expected_spatial = (S // PATCH) * _DEC_SCALE
        assert out.reconstruction.shape == (B, 3, expected_spatial, expected_spatial)

    def test_unknown_modality_raises(self, decoder):
        N = N_TOKENS
        lat = _make_latent(N=N)
        with pytest.raises(ValueError):
            decoder(lat, torch.zeros(B, N, 4), target_shape=(B, 3, IMG_H, IMG_H), modality="unknown")

    def test_no_nan_in_output(self, decoder):
        N = N_TOKENS
        lat = _make_latent(N=N)
        pos = torch.zeros(B, N, 4)
        out = decoder(lat, pos, target_shape=(B, 3, IMG_H, IMG_H), modality="image")
        assert not torch.isnan(out.reconstruction).any()

    def test_pixel_shuffle_upsample_doubles_resolution(self):
        from mavt.module7_decoder import PixelShuffleUpsample
        up = PixelShuffleUpsample(in_ch=D, out_ch=D // 2)
        x = torch.randn(B, D, 4, 4)
        out = up(x)
        assert out.shape == (B, D // 2, 8, 8)


# ---------------------------------------------------------------------------
# tokenizer.py  (end-to-end)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mavt():
    from mavt.tokenizer import MAVTokenizer
    return MAVTokenizer(_small_mavt_cfg()).eval()


class TestMAVTokenizer:
    # IMG_H=64: (IMG_H//PATCH)*_DEC_SCALE = 4*16 = 64 = IMG_H ✓
    H = IMG_H

    def test_encode_image(self, mavt):
        x = torch.randn(B, 3, self.H, self.H)
        latent = mavt.encode(x)
        N = (self.H // PATCH) ** 2
        assert latent.z.shape == (B, N, 8)
        assert latent.z_understand.shape == (B, 16)

    def test_forward_image_returns_reconstruction(self, mavt):
        x = torch.randn(B, 3, self.H, self.H)
        dec_out, lat_out = mavt(x)
        n_h = self.H // PATCH
        expected = n_h * _DEC_SCALE
        assert dec_out.reconstruction.shape == (B, 3, expected, expected)

    def test_forward_video(self, mavt):
        T = 4
        x = torch.randn(B, 3, T, self.H, self.H)
        dec_out, lat_out = mavt(x, modality="video")
        n_t = T // 2  # temporal chunks
        expected_spatial = (self.H // PATCH) * _DEC_SCALE
        assert dec_out.reconstruction.shape == (B, 3, n_t, expected_spatial, expected_spatial)

    def test_compute_loss_keys(self, mavt):
        # Make a target that matches decoder output shape so L1 loss works.
        x = torch.randn(B, 3, self.H, self.H)
        dec_out, lat_out = mavt(x)
        # Provide a target of the same spatial size as reconstruction.
        recon_shape = dec_out.reconstruction.shape
        target = torch.randn(*recon_shape)
        losses = mavt.compute_loss(target, dec_out, lat_out)
        assert "recon_loss" in losses
        assert "kl_loss" in losses
        assert "total_loss" in losses

    def test_compute_loss_values_are_scalars(self, mavt):
        x = torch.randn(B, 3, self.H, self.H)
        dec_out, lat_out = mavt(x)
        recon_shape = dec_out.reconstruction.shape
        target = torch.randn(*recon_shape)
        losses = mavt.compute_loss(target, dec_out, lat_out)
        for v in losses.values():
            assert v.ndim == 0

    def test_total_loss_sum(self, mavt):
        x = torch.randn(B, 3, self.H, self.H)
        dec_out, lat_out = mavt(x)
        recon_shape = dec_out.reconstruction.shape
        target = torch.randn(*recon_shape)
        kl_weight = 1e-4
        losses = mavt.compute_loss(target, dec_out, lat_out, kl_weight=kl_weight)
        expected = losses["recon_loss"] + kl_weight * losses["kl_loss"]
        assert torch.allclose(losses["total_loss"], expected)

    def test_gradients_flow(self):
        from mavt.tokenizer import MAVTokenizer
        model = MAVTokenizer(_small_mavt_cfg()).train()
        x = torch.randn(B, 3, self.H, self.H)
        dec_out, lat_out = model(x)
        recon_shape = dec_out.reconstruction.shape
        target = torch.randn(*recon_shape)
        losses = model.compute_loss(target, dec_out, lat_out)
        losses["total_loss"].backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_encode_explicit_modality(self, mavt):
        x = torch.randn(B, 3, self.H, self.H)
        latent = mavt.encode(x, modality="image")
        N = (self.H // PATCH) ** 2
        assert latent.z.shape == (B, N, 8)

    def test_no_nan_end_to_end(self, mavt):
        x = torch.randn(B, 3, self.H, self.H)
        dec_out, lat_out = mavt(x)
        assert not torch.isnan(dec_out.reconstruction).any()
        assert not torch.isnan(lat_out.z).any()
