# C-D 4.1: geometric transport and Δ-Tucker (prototype)

A prototype of the C-D 4.1 design (26/10) and of its module spec: a unified tokenizer for images, video and
posed views of a static scene on a frozen DINOv3 ViT-L/16, in which a target observation is coded as

    H_i^base = W_i H_0 + [(W_i U) B(a_i)] Q        transport of the anchor + low-rank variation
    F^_i     = H_i^base + 1[L_i > 0] Dec_R(H_i^base, r_i[:L_i], meta_i)   + ordered innovation slots

and every output is decoded from the serialized state Z alone. Status: implemented and unit-tested on CPU
with the tiny config (P0 schema, ledger, serializer and tests; P2 synthesis head and encoder; E1a). The
causal prior (Module 5, P3) and the streaming AppendAndTrim are not implemented.

## State and ledger

| Part | Scalars | Notes |
|---|---:|---|
| header | 16 | mode, n_obs, time_offsets[4], view_ids[4], L_target[3], coordinate_mode, camera_mode, version |
| anchor `c` | 256 x 23 | content per cell, decoded by F_C (no spatial Tucker) |
| anchor `rho` | 256 | log inverse z-depth; median depth 1 on video, the data's units on posed views |
| `kappa_0` | 4 | fx / W, fy / H, cx / W, cy / H (edge convention); estimated on video and images |
| target `xi_i` | 6 | anchor camera -> camera i in se(3), (omega, v): R = exp(omega), t = V(omega) v |
| target `M_i` | 16 x 3 | displacements from the anchor, in the anchor frame, blended per cell by beta(c); 0 on posed views |
| target `kappa_i` | 4 | |
| target `a_i` | 32 | Tucker coefficients (`tucker: false` drops them) |
| target `r_i` | L x 24, L <= 80 | ordered slots; only the first L are sent |

Image 6 164, segment of 4 observations 12 194 (94 under the 12 288 ceiling, unallocated). Without Δ-Tucker,
`n_slots: 81` gives 12 170. `state.serialize` writes exactly these scalars per sample (no padding) and
`state.deserialize` reads them back from the header alone. An image has n_obs = 1 and no target at all.

## Modules

| Spec module | Code | What |
|---|---|---|
| 0 serializer | `state.py` | header, `serialize`, `deserialize`, `payload_numbers` |
| 1 transport | `transport.py` | `build_sparse_transport` -> `SparseTransportOp` (idx, w [B, 256, 9], support), `apply(V)` by gather in O(9 N D); se(3) exp / log; intrinsics conventions |
| 2 GTA synthesis | `model.py` `ConvSynth`, `target_op`, `target_base`, `tucker_term` | F_C: Linear(23 -> 256) -> depthwise-separable 3 x 3 -> GELU -> Linear(-> 1024); F_U: Linear(25 -> 128) -> 3 x 3 conv -> GELU -> Linear(-> 32); beta = softmax(Linear(23 -> 16)); G [32, 32, 32]; Q Linear(32 -> 1024, no bias). `synth: transformer` keeps the first prototype's generator |
| 3 encoder | `TwoStreamTrunk`, `Refiner`, `InnovationEncoder` | see below |
| 4 public decoder | `decode(Z)`, `residual`, `ResidualDecoder`, `RGBDecoder` | each target reads only C, z_i and the header |

Encoder side, per segment:

- `FrameEncoder` (from cd32): frozen DINOv3 read at blocks 6, 12, 18, 24 (Mem_i, 4 x 256 tokens, read-only keys
  in every trunk layer) and the pixel stem; the cells start from the last DINO layer and the stem.
- Stream C (6 layers) reads observation 0 and Mem_0 only -> c, rho, kappa_0.
- Stream D (6 layers, its own weights) reads observation i, Mem_i and context tokens decoded from the public
  state S_{i-1}: on video H_0 and the F^ of the earlier targets with their time offsets, so targets are coded
  in order; on static views H_0 alone with time offset 0, and the views are coded in parallel. No keys pass
  from stream C to stream D. Tokens: 256 cells + camera + 16 motion + a = 274; a known camera (static views,
  `world_camera: given`) is embedded into the camera token. Level 1 (xi, kappa, M, a) comes from these heads.
- Refiner (level 2, `refine_layers` = 2): the cells get the alignment feedback F_i - sg(H^base(1)); zero heads
  add increments to xi, the kappa logits, M and a. Static views and given cameras keep xi and kappa (and static
  views M = 0), so only a moves and W_i is built once.
- Enc_R (4 layers): fused cells LN(e^F) W_F + LN(PatchEmbed(e^RGB)) W_RGB + LN(sg H^base) W_H + MetaEmbed(h,
  xi, kappa) + grid, read by 80 ordered queries (cross-attention to the cells, self-attention among the slots
  with a lower-triangular mask: `slot_causal`) -> r_i [80, 24], cut to L_i.

Transport (section 7): anchor cell j is unprojected with z = exp(-rho_j), moved by the blended displacement,
transformed by exp(xi_i) and projected with kappa_i. Row u' keeps at most 9 projections that are finite, in
front (z > z_eps = 0.01), inside the crop and within r_cut of u', with logits
`-|u' - u_ij|^2 / (2 sigma_u^2) - z_ij / tau_z`; the max is taken over the kept entries only, so far scenes
(z / tau_z ~ 1e4) keep their rows. An empty row is exactly zero and has support 0. `use_smooth_cutoff` adds
`log max(eps, 1 - d^2 / r_cut^2)`, off by default. The dense [N, N] distance matrix exists only for the
neighbour search (no gradient through the selection); `apply` never forms W_i, and its backward loops over the
9 neighbours, so no [B, N, 9, D] tensor is saved.

Losses: L1 + LPIPS (subset of frames) on pixels and cosine + MSE on the teacher features for every observation,
the same on H^base over cells with support (`w_base`) and on the level-1 H^base (`w_refine1`), |M|, and on
world clips the camera against the given one.

## Choices beyond the spec

- LN(e^F) and LN(PatchEmbed(e^RGB)) drop the size of the residual of each cell; Enc_R and the refiner also get
  its log RMS, so that a cell that is already right can be told apart from one that is far off.
- Enc_R and Dec_R read from the header the mode, the time offset of target i, the coordinate and camera modes;
  not the view ids (order is not content) and not L (r[:L] must not depend on L).
- The refiner adds to the logits of the bounded camera head (kappa stays inside the FOV range), and its level 1
  is supervised too (`w_refine1`), as iterative refiners usually are.
- `z_eps` guards the projection instead of z > 0 exactly.
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
