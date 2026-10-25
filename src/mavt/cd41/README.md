# C-D 4.1: geometric transport and Δ-Tucker (prototype)

A prototype of the C-D 4.1 design (26/10): a unified tokenizer for images, video and posed views of a static
scene on a frozen DINOv3 ViT-L/16, in which a target observation is coded as

    H_i^base = W_i H_0 + (W_i U) B(a_i) Q          transport of the anchor + low-rank variation
    F^_i     = H_i^base + Dec_R(H_i^base, s_i, r_i[:L])   + ordered innovation slots

and every output is decoded from the serialized state Z alone. Status: implemented and unit-tested on CPU with
the tiny config; it covers steps P0 and P2 of the design (schema, ledger, serializer, tests; the synthesis head
on the current DINO encoder) and E1a. The causal prior (P3) is not implemented.

## State and ledger

| Part | Scalars | Notes |
|---|---:|---|
| header | 16 | mode, n_obs, time_offsets[4], view_ids[4], L_target[3], coordinate_mode, camera_mode, version |
| anchor `c` | 256 x 23 | content per cell, decoded by the nonlinear generator F_C (no spatial Tucker) |
| anchor `rho` | 256 | log inverse z-depth; median depth 1 on video, the data's units on posed views |
| `kappa_0` | 4 | fx / W, fy / H, cx / W, cy / H (edge convention); estimated on video and images |
| target `xi_i` | 6 | anchor camera -> camera i, (axis-angle, translation) |
| target `M_i` | 16 x 3 | displacements from the anchor, in the anchor frame, blended per cell by beta(c); 0 on posed views |
| target `kappa_i` | 4 | |
| target `a_i` | 32 | Tucker coefficients (`tucker: false` drops them) |
| target `r_i` | L x 24, L <= 80 | ordered slots; only the first L are sent |

Image 6 164, segment of 4 observations 12 194 (94 under the 12 288 ceiling, unallocated). Without Δ-Tucker,
`n_slots: 81` gives 12 170. `state.serialize` writes exactly these scalars per sample (no padding) and
`state.deserialize` reads them back from the header alone.

## Code

| File | What |
|---|---|
| `config.py` | `CD41Config` (extends the cd32 config for the DINO reader), ledger, `tiny_cd41_config` |
| `transport.py` | intrinsics conventions, SO(3) log, relative poses, the sparse transport `W_i` |
| `state.py` | header, `serialize`, `deserialize`, `payload_numbers` |
| `model.py` | `MAVTCD41`: encoder side (FrameEncoder, trunk, Enc_D, Enc_R) and decoder side (`decode(Z)`) |

Transport: anchor cell j is unprojected with z = exp(-rho_j), moved by the blended displacement, transformed by
exp(xi_i) and projected with kappa_i. Row u' of `W_i` is a softmax over at most 9 projections that are in front
of the camera, finite, inside the crop and within `r_cut` of u', with logits
`-|u' - u_ij|^2 / (2 sigma_u^2) - z_ij / tau_z`. An empty row is exactly zero and has support 0; Dec_R reads the
support, but the innovation is not gated by it. With `sigma_u = 0.35` a zero-motion warp keeps about 0.94 of
each cell on itself (the 9-neighbour kernel is not the identity unless `r_cut < 1`).

Modes: `image` (anchor only), `video` (camera and motion estimated by Enc_D from the observation and the anchor
state H_0), `threed` = static_views (kappa and xi copied from the cameras, M = 0 exactly). The trainer's
`world` clips (time + known cameras) are coded as video; the given cameras and depth enter only the loss
(`world_camera: given` copies them into the state instead).

Losses: L1 + LPIPS (subset of frames) on pixels and cosine + MSE on the teacher features for every observation,
the base loss on H^base over cells with support, |M|, and on world clips the camera against the given one.

## Deviations from the design text

- Targets of a segment are coded independently given the anchor state (Enc_D and Enc_R read observation i,
  H_0 and H_i^base, not S_{i-1}). This is a special case of the causal order and keeps prefixes fixed.
- `xi` is (axis-angle, translation), the parametrisation of `cd32.geometry.pose_matrix`, not the se(3) log.
- Time offsets in the header are in frame steps of the sampled clip: the loaders do not carry seconds.
- The feature cosine clamps norms below 1: an unsupported cell has H^base = 0 exactly, where the plain cosine
  has an unbounded gradient.
- The half-pixel convention of the OmniWorld pack intrinsics is not audited here (at most 1/32 of a cell).

## Commands

    python -m pytest tests/cd41 -q
    python scripts/train_cd41.py --dry-run --mix-modalities --out /tmp/x
    torchrun --standalone --nproc_per_node N scripts/train_cd41.py --config configs/cd41/base.yaml \
        --imagenet <root> --openvid <manifest> --objaverse <root> --objaverse-depth \
        --omniworld <pack> --omniworld-modality world --omniworld-scenes 30 --out <run>
    ... --set tucker=false --set n_slots=81                     # arm without Δ-Tucker, about the same payload
    python scripts/eval_cd41.py --ckpt <run>/last.pt --openvid <manifest> --objaverse <root> --objaverse-depth \
        --omniworld <pack> --omniworld-modality world --imagenet <root> --dino-repo <local> --out <run>/eval
    python scripts/e1a_tucker_fit.py --omniworld <pack> --dino-repo <local> --stride 8 --limit 64 --out <dir>

`eval_cd41.py` reports, per clip kind, the anchor and the targets decoded from Z, the targets with the
innovation cut to L slots (L = 0: transport + Δ-Tucker only), with a_i = 0 at decode, the copy-the-anchor
baseline, and the payload counted by serializing Z. `e1a_tucker_fit.py` is E1a: the relaxed Tucker fit of the
feature residual after an oracle warp (measured depth and given poses, static cells), with and without the warp.
