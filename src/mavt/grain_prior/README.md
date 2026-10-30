# Grain-Prior C-D v1 (prototype)

A tokenizer for images, video and posed views of a static scene on a frozen DINOv3 ViT-L/16. The anchor
observation is coded as a set of grains (2.5D points with a feature) plus a coarse memory; every later
observation is coded from the previous closed state as a short record, in three steps:

1. **conditional initialisation**: a proposal of the record from the previous state alone (P_init, stream D);
2. **evidence refinement**: two passes that project the grains with the current estimate, read the current
   image around the projections and update camera, motion and memory (zero-initialised heads);
3. **closed state**: the record is appended to a queue of at most three; a fourth folds the oldest into the
   boundary memory z_b. Everything the public decoder needs is in the serialized state.

Status: implemented and unit-tested on CPU with the tiny config. Weights and temperatures are starting values.

## State and ledger

| Part | Scalars | Notes |
|---|---:|---|
| header | 32 | version, mode, n_targets, coordinate / camera mode, times, view ids, lengths L_i, size, provenance |
| grains G_0 | 64 x (2 + 1 + 1 + 60) | one per 2 x 2 stratum: u (soft location in the stratum), rho = log 1/d, sigma, f |
| coarse c_0 | 16 x 128 | 4 x 4 coarse slots |
| kappa_0 | 4 | fx / W, fy / H, cx / W, cy / H (edge convention); copied when the camera is known |
| boundary z_b | 16 x 128 | sent as soon as the state holds a target record |
| record xi_i | 6 | anchor camera -> camera i in se(3), ordered (v, omega) |
| record M_i | 16 x 3 | motion controls, blended per grain by beta(f, u, c_0); 0 on static views |
| record kappa_i | 4 | |
| record b_i | 16 x 2 | memory update of the diagonal recurrence z_i = U(z_{i-1}, b_i, dt) |
| record r_i | L x 24, L <= 52 | ordered innovation slots (u 2, rho, sigma, e 20); only the first L are sent |

Image 6 180 scalars; a state with three records at L = 52 is 12 242 (46 under the 12 288 ceiling).
`schema.serialize` writes exactly these scalars per sample, `schema.validate` checks one vector against its
header and `schema.deserialize` reads it back. Modes: image, video (anchor median depth 1, cameras estimated),
static_views (cameras copied, metric, M = 0, z_i = U(c_0, b_i, 1)), calibrated_video (cameras copied, metric).

## Modules

| Stage | Code | What |
|---|---|---|
| 1-2 features | `anchor.py` `Stems` | frozen DINOv3 last layer F_i (target and evidence), PixelStem (4 stride-2 convs), Y_i = LN(F W_F + P + PE), g_i coarse summary |
| 3 stream C | `anchor.py` `StreamC` | six blocks over Y_0 |
| 4A anchor | `anchor.py` `AnchorHeads` | c_0, kappa_0, grain location inside each stratum, conditional grain queries and two read-refine blocks with a Gaussian + exploration prior -> f, rho, sigma |
| 4B trunk | `trunk.py` `Bank`, `PInit`, `StreamD` | the bank rebuilt from the previous state (temporal or static view), the record proposal, stream D over 256 cells + 18 control tokens |
| 4B refinement | `refinement.py` `CorrRead`, `UpdateHead` | correspondence read over all 256 patches around the projected grains, increments of xi (left composition), kappa logits, M, b |
| memory | `memory.py` `Recurrence`, `Appearance` | diagonal recurrence in fp32; f~ = f + A(f, u, z) - A(f, u, c_0), so z = c_0 leaves the grains unchanged |
| 5 innovation | `innovation.py` `InnovationEncoder` | residual tokens from e^F = F - sg F^base and e^RGB = X - sg X^base, 52 slots with causal self-attention |
| 7 public decoder | `readout.py` `Readout`, `RGBHead`; `model.py` `decode(S)` | a^(0) from c_0, z_i and pooled geometry (or learned queries), two grain reads with the geometry bias log nu, F^base = W_out a^(2), innovation read (exactly 0 at L = 0), RGB head |
| 8 causal prior | `prior.py` `CausalPrior`; `model.py` `predict` | the next record from S_t alone; trained with the tokenizer frozen (`phase: prior`) |
| state | `schema.py`, `model.py` `append_and_fold`, `replay` | header, serializer, queue and fold |
| geometry | `geometry.py` | conventions, intrinsics bounds, se(3) with (v, omega), grain transport and projection |

Losses (`model.py`, `losses.py`): feature loss 1 - cos + 0.01 log-norm^2 on F^, L1 + 0.1 LPIPS on RGB, the same
on the base (weight 0.5) and on the initializer F^init (0.25), Huber log-depth of grains and readouts where depth
labels exist, camera loss on world clips coded with estimated cameras, small motion and code penalties.

## Arms

`--set` on the trainer: `geom_bias=false` (grain read without the geometry bias), `query_init=uncond` (learned
queries instead of the conditional a^(0)), `memory=false` (z_i = c_0), `tau_d_mode=absolute` (depth temperature
in scene units instead of anchor-median units).

## Flags 2026-10 (all off by default)

Every flag below is off at its default: the s1-s3 checkpoints keep their `state_dict` keys and behaviour
(`tests/test_state_dict_roundtrip.py`, `test_arch_flags.py::test_defaults_add_no_parameters`). None adds a number to
the state: the ledger and `schema.serialize` are unchanged. Tests: `tests/grain_prior/test_arch_flags.py` (one
section per item), `tests/grain_prior/test_evaluate_mechanism.py` (3.2).

| Item | Flag(s) | What |
|---|---|---|
| 1.1 | `innov_stopgrad_base` | samples with L_b > 0: F^ = sg[F_base] + dF, H^ = sg[a] + dH (the innovation query also reads sg[a]); identity at L_b = 0. Needs `w_base > 0`. |
| 1.2 | `w_geom_dense`, `w_geom_ramp_steps` | L_geom = (anchor + sum grain + w_dense sum dense) / (1 + n + w_dense n) (1.0: the old plain mean); weight w_geom min(1, step / ramp) from `set_step` (the v0 trainer calls it). |
| 1.3 | `GrainPriorConfig.full_budget(stratum)`, `full_budget_config` | largest d_f with header + C <= 12 288, counted by the ledger (155 at stratum 2). |
| 1.4 | `dino_multi_layer` | F_mid = W LN(concat DINO layers `dino_layers[:-1]`): second cross-attention of stream C and fourth source of the grain read. |
| 1.5 | `read_prev_frame`, `w_pred_frame` | a += Mix(9 bilinear taps of W sg[F^_{t-1}] at u_q - du_q + {-1, 0, 1}^2 / grid), du_q = Gaussian-weighted grain motion u^(t) - u^(t-1); zero-init, video / world only, also on the RGB grid. L_pred = w_pred_frame feat_loss(F_base, F). `decode(S)` reads the previous record of the queue (the anchor for the first): equal to the encoder while no record was folded into z_b; a streaming decoder keeps F^_{t-1}. (cd32's `prev_read` / `w_pred` are inherited and unused here.) |
| 2.1 | `read_coarse_kv`, `coarse_sink` | the 16 slots [c_0; z_i] + PE join the grain softmax: logit = exp(tau_c) q.k_c / sqrt(d_h) + beta_c - proximity(u_q); zero-value sink on by default. The depth weights nu stay normalised (the proximity shift enters the read only). |
| 2.2 | `read_relpos_value`, `relpos_rank` | v_j(q) = W f_j + sum_r phi_r(D_qj) taper(D_qj) W_r f_j, D = grid (u_q - u^_j) / sigma_j, R / 2 Fourier directions x (sin, cos), Wendland C2 taper to 0 at 3 sigma; W_r zero-init; R weighted sums, no [B, Q, G, d] tensor. |
| 2.3 | `rgb_coarse_start`, `w_rgb_coarse` | X^ = clamp(sg[X~0] + R, 0, 1), X~0 = pyr_up(sigmoid(RGBLow(a^(0)))), R zero-init; L1(X~0, pyr_up(pyr_down(X))) with the same binomial kernel; diag `psnr_coarse_rgb`. |
| 2.4 | `read_normalizer`, `read_topk`, `entmax_alpha_init`, `alpha_anneal_steps` | softmax (SDPA) / entmax15 (alpha-entmax by bisection, closed-form backward; alpha per head learned in (1, 2), or annealed init -> 1.5) / topk (the k grains of largest geometry bias per query, then the coarse slots and the sink, SDPA). Depth weights nu stay Gaussian. |
| 2.5 | `sem_target`, `w_sem`, `sem_dim` | 1 - cos(MLP(mean c_0), teacher(X_0)), training only. Teachers sit outside the module tree (`set_sem_teacher`); `stub` is built in, `dinov3txt` / `siglip2` need their weights and raise until a teacher callable is attached. |
| 3.1 | (always) | on the feature-grid read (an extra no-grad softmax [B, H, Q, G]): diag `eff_support` (1 / sum a^2), `fine_share` / `coarse_share` / `sink_share`, `init_cos_all`, error fractions by band, `fine_share_hf_corr`. |
| 3.3 | `rgb_head=pointwise`, `rgb_pw_mlp_ratio`, `rgb_in_feat` | 4 x 1 x 1 MLP blocks with per-point LayerNorm, PixelShuffle x2 + conv3x3 stages, 1 x 1 -> 3 zero-init (receptive field 9 px <= 16; needs image / side <= 4, i.e. `query_grid` >= image / 4); `rgb_in_feat`: H + W up(sg[F^]) with W from the DINO width. |
| 3.4 | `read_split_layer2` | the last ReadLayer exists twice: feature grid (F_base, depth) and RGB grid (needs `query_grid`). |
| 3.5 | `state_noise_bits`, `geom_noise_px` | training only: f, c_0 + U(-D/2, D/2), D = batch range / 2^bits; u + N(0, (px / image_size)^2). |
| 3.6 | `rgb_decoder=splat`, `splat_layers`, `splat_feat`, `splat_atoms`, `splat_window`, `splat_wmax` | RGB grid only (needs `query_grid` % n_strata = 0 and the legacy readout; use `rgb_head=pointwise`): a context transformer over the grains and the [c_0; z] slots (attention biased by the per-head scaled `geom_bias`), K frequency atoms per grain (amplitudes zero-init, omega = wmax tanh(.) cycles per cell), splatted over a (2w+1)^2 window of strata with depth-aware Gaussian weights; added to `readout.init` a^(0), then the RGB head. Replaces the read layers on the RGB grid (`splat.py`, `test_splat_decoder.py`). |
| 3.7 | `splat_pix_feat`, `splat_pix_atoms`, `splat_pix_wmax`, `rgb_pix_blocks`, `rgb_pix_width` | pixel-level decoder capacity for the pointwise head: `splat_pix_feat` > 0 evaluates a second set of atoms (amplitudes zero-init) at the image pixels with the splat's window weights (the query-grid atoms alias above 2 cycles per cell; at the pixels the limit is patch / 2) and adds the map after the last x2 stage (1 x 1, no bias); `rgb_pix_blocks` ResBlocks (3 x 3 -> GELU -> 3 x 3 zero-init, hidden max(stage width, `rgb_pix_width`)) after every x2 stage. Both reproduce a checkpoint trained without them at load. |
| 3.8 | `w_edge`, `feat_sg_f` | `w_edge` > 0: L1 of the Sobel gradients (both directions, / 8, reflect padding) of decoded vs target frames, added with L1 and LPIPS. `feat_sg_f`: the feature-grid read receives sg[f], so L_F (and L_init / L_base through it) no longer trains the grain features; the RGB read and the splat keep the live f. |
| 3.9 | `splat_index`, `splat_cap` | `splat_index=projected`: the splat window of an RGB query is built from the grains that project INTO the strata around the query on the decoded observation (per-sample buckets of at most `splat_cap` grains per stratum, `splat.bucket_index`; grains over the cap are dropped, their share is diag `splat_dropped_frac`), instead of the fixed table by ANCHOR stratum (`anchor`, the default: a grain that moves more than `splat_window` strata is no longer read where it lands, and queries whose neighbourhood emptied read nothing). Identical outputs when no grain leaves its stratum (images). `test_splat_projected.py`. |

3.2 (no training): `python -m mavt.grain_prior.evaluate_mechanism --ckpt <run>/last.pt --openvid <manifest> --out
<dir>` writes `mechanism.json` / `.md`: CD1 (D off), CD3 (C -> + xi -> + M -> + r(L)), CD5 (c_0 swapped: change by
band), CD6 (mean / random state), AT4 (grains of lowest / highest mass removed), AT2 (with `read_coarse_kv`), AT3.

## Commands

    python -m pytest tests/grain_prior -q
    python scripts/train_grain_prior.py --dry-run --mix-modalities --out /tmp/gp
    torchrun --standalone --nproc_per_node N scripts/train_grain_prior.py --config configs/grain_prior/base.yaml \
        --imagenet <parquet root> --openvid <manifest> --objaverse <render root> --objaverse-depth \
        --omniworld <pack> --omniworld-modality world --omniworld-scenes 30 --out <run>
    python scripts/eval_grain_prior.py --ckpt <run>/last.pt --imagenet <root> --openvid <manifest> \
        --objaverse <root> --objaverse-depth --omniworld <pack> --omniworld-modality world --out <run>/eval

Phase T3 (causal prior): `--set phase=prior --init <codec checkpoint>` on clips of at least five frames.
