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

## Commands

    python -m pytest tests/grain_prior -q
    python scripts/train_grain_prior.py --dry-run --mix-modalities --out /tmp/gp
    torchrun --standalone --nproc_per_node N scripts/train_grain_prior.py --config configs/grain_prior/base.yaml \
        --imagenet <parquet root> --openvid <manifest> --objaverse <render root> --objaverse-depth \
        --omniworld <pack> --omniworld-modality world --omniworld-scenes 30 --out <run>
    python scripts/eval_grain_prior.py --ckpt <run>/last.pt --imagenet <root> --openvid <manifest> \
        --objaverse <root> --objaverse-depth --omniworld <pack> --omniworld-modality world --out <run>/eval

Phase T3 (causal prior): `--set phase=prior --init <codec checkpoint>` on clips of at least five frames.
