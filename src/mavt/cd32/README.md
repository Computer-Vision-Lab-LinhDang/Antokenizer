# C-D tokenizer, bản 3.2, with the two-stream backbone

A tokenizer for images, video and posed views on a frozen DINOv3 ViT-L/16. It splits a clip into what
stays (C, at keyframes) and what changes (D, at the other frames), and makes the change pass through a
fixed geometric operator so that depth, pose and object motion keep their physical meaning.

Status: implemented and unit-tested on CPU with the tiny config. Not trained at scale yet.

## Latent

| Part | Where | Numbers | Meaning |
|---|---|---|---|
| `c_k` | keyframe, per cell | 47 (image) / 23 (video, views) | static content (Gaussian, KL) |
| `rho_k` | keyframe, per cell | 1 | log inverse depth; median depth 1 per keyframe on video |
| `f_k` | keyframe | 1 (video only) | focal length as a bounded FOV (30 to 120 degrees, init 50) |
| `xi_t` | other frames | 6 | pose of frame t against the keyframe (axis-angle + translation) |
| `M_t` | other frames | 4 x 4 x 3 | coarse 3D object flow, convex-upsampled guided by `c_k` |
| `D_innov` | other frames / views | L x 24, L <= L_max | residual tokens, prefix of random length (nested dropout) |
| link | keyframe k >= 1 | 7 | pose of keyframe k against k - 1 and a log scale ratio |

Budget with the default config: an image is 256 x 48 = 12 288 numbers. A segment of K = 4 frames is
256 x 24 + 1 + 7 + 3 x (54 + 82 x 24) = 12 218, so L_max = 82. Posed views cost 0 numbers of D_geom: the
poses are given and M = 0.

## Forward projection and decoder

Cell u of keyframe k lands in frame t at

    P_k(u) = exp(-rho_k(u)) K^-1 [u, 1]
    P'(u)  = R_t (P_k(u) + m_t(u)) + tau_t
    p(u)   = pi(P'(u)),  z(u) = P'_z(u)

Each query cell u' of frame t reads its k = 9 nearest projected cells. The attention logit bias is
`-lam_d d^2 - lam_z Δlog z + w_cut(d) + cull`, with `w_cut = log max(eps, 1 - d^2 / R^2)` and R the
distance to the 10th neighbour, so the read is smooth when a cell enters or leaves the neighbour set.
`Δlog z` is the depth of each neighbour against the soft-selected front surface (soft z-buffer), and
`cull` is a soft back-face term from the normals of the keyframe surface. Values are built from
`[c_k ; Fourier(u' - p) ; Fourier(t - t_k)]`.

Frames after the keyframe are coded in order. For each one:

1. **Predict.** Run the geometric read, then self-attention, to get `h_geom`. Add a 3 x 3 window read of
   the previous decoded frame to get `h_pred` (zero-init).
2. **Residual.** Compute `e = X_t - sg[feat(h_pred)]` in DINO space. `Enc_R` (slots reading
   `[e ; sg[X~] ; sg[c_k]]`) turns it into `D_innov`.
3. **Innovation.** Cross-attend from a query built on `[Fourier(u') ; sg[h_pred] ; d_min]` to
   `[null ; D_innov[:L]]`. The null token has value 0, and the branch is masked when L = 0.
4. **Combine.** `h = h_pred` when L = 0, with no stop-gradient. Otherwise `h = sg[h_pred] + h_innov`.
   DINO features come from a linear head and RGB from the v0 conv upsampler.

## Backbone

The backbone has two streams with separate weights (MM-DiT style). Each layer runs block C, then block D:

| Block | Queries | Keys and values |
|---|---|---|
| C | 256 cells + 1 focal token of the keyframe | own tokens + DINOv3 memory of the keyframe |
| D, frame t | 256 cells + 1 pose + 16 flow tokens | `[K_C ; K_D(t-1) ; K_D(t) ; memory(t)]` |
| link | 1 token per keyframe | `[K_C(k) ; K_C(k-1) ; own]` |

- **The mask is asymmetric.** C never reads D or other frames. D is sparse-causal (Tune-A-Video), with C
  as the sink. For t = 1 the previous frame is the keyframe, so its D slot is empty.
- **The memory is read-only.** It holds the DINOv3 patch features of layers 6, 12, 18 and 24, each with
  its own projection and layer embedding. Every block re-reads it in every layer.
- **Positions.** RoPE runs over (t, x, y) with 16 / 24 / 24 dims of each 64-dim head. t is the frame
  index in the clip. Posed views add a zero-init Plücker ray embedding instead of a view axis.
- **Heads.**
  - Content `mu` and `logvar`.
  - `rho`, which reads the stream and the 4 memory layers, like a DPT head.
  - FOV, from the focal token.
  - Pose, flow and link, all zero-init, so identity pose and no flow at the start.
  - On posed views, `rho` also gets a `tanh(alpha)`-gated cross-view refinement, with alpha = 0 at init.
- **Option B.** `shared_streams: true` gives the D blocks the C block weights.

The default config has about 102M trainable parameters: 59M backbone, 16M Enc_R and 20M decoder.

## Losses

| Name | What |
|---|---|
| `rec_feat`, `rec_rgb` | DINO (1 - cos + MSE) and RGB L1 on every decoded frame |
| `geo_feat`, `geo_rgb` | Charbonnier on the geometry-only decode `h_geom` of the frames after the keyframe |
| `pred_feat` | Charbonnier on `sg[h_geom] + h_prev`: trains the previous-frame read without bending geometry |
| `flow_l1` | mean of the 3D flow magnitude, so motion is explained by pose and depth first |
| `smooth` | edge-aware smoothness of `rho` at keyframes |
| `kl_c`, `kl_d` | KL of `c_k` and of the kept innovation tokens |
| `link` | Charbonnier: keyframe k decoded from `C_{k-1}` through the link pose and scale |
| `depth_prior` | optional, shift-invariant pull of `rho` to `sg[probe]`, a frozen depth probe trained on real depth labels |
| `depth_gt` | optional, SILog of `rho` (view 0 of posed views) against rendered depth, when the batch has it |
| `rho_distill` | optional, posed views: L1 of the single-view part of `rho` to `sg[rho]` (with the cross-view refinement) |

On still images `rho` is detached from `L_rec`, because without parallax it could only learn to carry
content.

## Decisions taken in this version

- **Backbone option A** (two streams). Option B is one flag away.
- **The innovation reference is the previous decoded frame** (DCVC-style conditional coding). It enters
  the prediction through the window read, so the residual and `Enc_R` see it.
- **`Δlog z` sits in the logit bias only**, never in the value path.
- **The soft z-buffer reference is a softmax-weighted mean** of the neighbours' `log z`, windowed by
  `w_cut`. The log-sum-exp form shifts by log(number of equal neighbours).
- **Posed views keep the image order of bản 3.2.** View 0 is the keyframe and the others are sorted by
  the angle of their optical axis to view 0. The previous view in that order is the reference.

## Not implemented yet

- 4D rigs: several cameras over time sharing one D_geom.
- The world-model forecast of D (`L_fut`).
- Choosing L by a distortion threshold at inference. `forward(..., L=...)` takes explicit lengths.
- LPIPS (`w_lpips > 0`) reuses the v0 module and has not been tested here.

## Vanilla variant (`vanilla.py`, `vanilla_model.py`, `scripts/train_cdv.py`)

The same latent, budget and forward projection on one plain transformer, with the reconstruction branch
on. It exists to test the idea quickly before the two-stream design is trained at scale.

| | two-stream (`model.py`) | vanilla (`vanilla_model.py`) |
|---|---|---|
| backbone | C and D blocks, DINO memory re-read per layer, 3-axis RoPE | one pre-LN transformer, learned positions, the 4 DINO layers summed into the input |
| tokens of a segment | C: 257; D per frame: 273 (+ link) | C: 257; D per frame: 256 cells + pose + 16 flow + L_max slots = 355 |
| the asymmetric mask | by construction of the key sets | an attention mask: C reads C; D_t reads C, D_t, D_{t-1} (`causal_d`) |
| innovation | Enc_R after a trial decode (residual) | the slot tokens of the same pass (mu, logvar), nested dropout |
| decoder | geometric read, prediction blocks, previous-frame read, innovation branch | geometric read, `dec_blocks` x (self-attention, cross-attention to [null ; r[:L]], MLP), all frames of a segment in one pass |
| pixels | conv upsampler | linear unpatchify + one 3 x 3 conv (`rgb_head: linear`), or the conv upsampler |
| losses | as above | `rec_rgb` L1 + `lpips` (0.5) + `rec_feat` (0.25), `geo_*` (L = 0 decode), `flow_l1`, `smooth`, `kl_*`, depth terms |
| not there | | link, L_pred, previous-frame read, cross-view rho refinement |

`geometry: false` (`--set geometry=false`) is the ablation at the same budget: identity projection for every
frame, so D_innov has to carry all change. `--init <ckpt>` starts a stage from the weights of the previous
one (optimizer and schedule fresh): images, then + video, then + posed views.

    python scripts/train_cdv.py --dry-run --out /tmp/cdv_dry
    torchrun --standalone --nproc_per_node <gpus> scripts/train_cdv.py --config configs/cd32/vanilla.yaml \
        --imagenet <parquet root> --out <stage 1>
    ... --imagenet <root> --openvid <manifest> --n-frames 4 --init <stage 1>/last.pt --out <stage 2>
    ... --objaverse <render root> --objaverse-depth --init <stage 2>/last.pt --out <stage 3>
    ... --omniworld <OmniWorld-Game pack root> --batch-world <n> --init <stage 3>/last.pt --out <stage 4>

**World clips** (`--omniworld`, modality `world`): posed dynamic video from the OmniWorld-Game pack
(`scripts/pack_omniworld_dino.py` on branch `feat/dino-omniworld-geo`, loader `dino_v0/omniworld.py`), one
segment of `keyframe_every` frames per clip. The poses and intrinsics come from the cameras (translations over
the keyframe's median depth, so they match the median-normalised `rho`); the object flow `M` and the innovation
tokens are predicted as for video. The metric depth of the keyframe is scored against `rho` (`rho_gt_corr`, and
the loss with `w_depth_gt > 0`).

## Depth

`rho` is log inverse depth per cell. Still images and video only know it up to a shift (median depth 1),
posed views in the units of the cameras (no normalisation). Two optional depth signals:

- **Prior from a probe.** `scripts/train_depth_probe.py` fits a linear probe on the frozen DINOv3 layers
  of `dino_layers` (each centred and layer-normed) to real depth labels, with SILog (lambda 0.85) plus a
  small ordinal term. The tokenizer loads it with `depth_probe=<file>` as non-persistent buffers (never in
  checkpoints) and pulls `rho` to `sg[probe]` with `w_depth_prior`, by default on images and video only.
  `depth_prior_kind`: `silog` (lambda 1: shift-invariant in log depth), `ssi` (scale and shift) or
  `ordinal`. With `w_depth_prior = 0` the loss and `rho_probe_corr` are only logged.
- **Rendered depth on posed views.** `--objaverse-depth` loads the Objaverse `_depth.png` maps
  (`v / 65535 * max_depth`, background invalid) and `w_depth_gt` scores `rho` of view 0 against them
  (SILog, `depth_gt_lam`).
- **Self-distillation on posed views.** `rho = rho_mono + tanh(alpha) CrossAttn_rho(...)`. With
  `w_rho_distill > 0`, `|rho_mono - sg[rho]|` makes the single-view part learn what the cross-view
  refinement adds, since images and video have no other views. Only cells with rendered depth count when
  the batch has it (background cells are not constrained by any view). The term is 0 while the gate is
  closed (`alpha = 0` at init).

Depth labels come as jsonl manifests (format in `depth_data.py`). `scripts/make_depth_manifest.py` builds
them from the packed OmniWorld-Game (dense metric depth), Argoverse 2 (LiDAR projected into the ring
cameras, sparse, real) and NYU Depth v2. `scripts/eval_cd32_depth.py` scores `rho`, the probe and a
constant depth with median scaling per image (AbsRel, SqRel, RMSE, RMSE log, SILog, delta < 1.25^k), per
cell and per pixel, and on held-out Objaverse objects also without scaling.

## Files

| File | Content |
|---|---|
| `config.py` | `CD32Config`, budget accounting (`L_max`, `segment_numbers`), `tiny_config()` |
| `geometry.py` | unproject / SE(3) / project, k-NN with cut radius, soft z-buffer, normals and cull, convex upsampling |
| `encoder.py` | frozen DINOv3 read at several layers, pixel stem, memory tokens |
| `layers.py` | 3-axis RoPE, stream block, window read |
| `backbone.py` | two-stream backbone, C heads (content, rho, FOV, cross-view rho), D heads (pose, flow, link) |
| `enc_r.py` | residual encoder |
| `decoder.py` | geometric read, prediction blocks, innovation branch, feature and RGB heads |
| `model.py` | `MAVTCD32.forward(x, modality, cam)`, the same interface as `MAVTDinoV0`; `load_model()` |
| `regs.py` | the C regularisers shared by both models: KL, smoothness, depth probe / rendered depth / distill |
| `vanilla.py` | `CDVConfig`, the one-transformer backbone with the mask, vanilla decoder, pixel head |
| `vanilla_model.py` | `MAVTCDV.forward(x, modality, cam)`, same interface; `load_cdv()` |
| `depth.py` | cell depth targets, SILog / SSI / ordinal losses, depth metrics, the depth probe |
| `depth_data.py` | depth manifests: reading depth files, crops that keep invalid pixels invalid |

## Commands

    python -m pytest tests/cd32 -q
    python scripts/train_cd32.py --dry-run --out /tmp/cd32_dry
    torchrun --nproc_per_node 2 scripts/train_cd32.py --dry-run --mix-modalities --out /tmp/cd32_ddp
    torchrun --standalone --nproc_per_node <gpus> scripts/train_cd32.py --config configs/cd32/base.yaml \
        --imagenet <parquet root> --openvid <manifest> --out <run dir>
    python scripts/make_depth_manifest.py omniworld --root <pack root> --split train --out <train.jsonl>
    python scripts/make_depth_manifest.py argoverse2 --root <av2 sensor root> --split val --out-dir <dir>
    python scripts/train_depth_probe.py --train <manifests> --val <manifests> --out <probe dir>
    python scripts/eval_cd32_depth.py --ckpt <run dir>/last.pt --probe <probe.pt> --depth <manifests> \
        [--objaverse <render root>] --out <json>

Overrides of single config fields: `--set KEY=VALUE` (repeatable), e.g.
`--set depth_probe=<probe.pt> --set w_depth_prior=0.1`.

`scripts/train_cd32.py` is the v0 trainer, so it has the same flags, data loaders, DDP and checkpoints.
Video clips must have a multiple of 4 frames. The link is trained only when a clip has at least 2
segments, for example `--n-frames 8`.
