# C/D state contract and temporal event fixtures

This first stage defines a compact recurrent state interface and a controlled
`core × configuration` diagnostic. It does not implement a recurrent cell,
decoder, learned C/D split, or training procedure. Passing these fixture tests
does not show that a model has learned an event representation.

## State and observation interfaces

The interfaces live in `mavt4d.contracts`. The intended model uses the same C/D
tensors for its next recurrent update and for readout. C should retain evidence
about an event's temporal structure; D should represent configuration relative
to that structure. These are learning objectives, not properties guaranteed by
a container's field names.

| Interface | Fields and shapes |
|---|---|
| `CDStateSpec` | `c_tokens=64`, `d_tokens=64`, `token_dim=32`, `frames_per_step=2`, `reference_views=1` |
| `CDState` | `c[B,Kc,d]`, `d[B,Kd,d]`, `last_time[B]` in float64 physical seconds, plus fixed `spec` |
| `ObservationStep` | `images[V,F,3,H,W]`, `observed[V,F,H,W]` bool, `frame_times[V,F]` float64 seconds, optional paired `intrinsics[V,3,3]` and `cam_to_world[V,F,4,4]` |
| `DecoderQuery` | `coordinates[Q,3]`, `view_ids[Q]` int64, `times[Q]` float64 seconds, `coordinate_space="image_pixel"`, optional paired camera table `intrinsics[V,3,3]` and `cam_to_world[V,4,4]` |

The default state contains 4,096 C/D scalar values per sample. Its logical tensor
payload is `4096 × dtype_bytes + 8` bytes per sample, including the timestamp:
16,392 bytes for float32 or 8,200 bytes for float16/bfloat16. This count excludes
serialization headers and allocator overhead. `CDStateSpec.num_state_values`
counts one sample; `CDState.num_state_values`, `state_nbytes`, `metadata_nbytes`,
and `total_nbytes` include the complete batch. Increasing the number of views
does not implicitly multiply the state budget.

`retained_storage_nbytes` also reports the actual backing storage retained by
the C/D/timestamp tensors, counting shared storage once and including unused
tails behind tensor views. It excludes allocator and autograd overhead. This
value can exceed the logical payload for sliced tensors. A compact snapshot
or export followed by restore has equal logical and retained-storage sizes.

`CDState.to_payload()` exports exactly `schema_version`, `spec`, `c`, `d`, and
`last_time`; `from_payload()` rejects missing or additional fields. The version-1
payload supports `torch.load(..., weights_only=True)`. Export and `snapshot()`
make detached compact copies. `detach()` preserves values but shares storage.
Containers are frozen, while PyTorch tensor contents remain mutable; export
revalidates their contents.

Any future recurrent implementation must put all persistent sequence memory in
this state. A hidden convolution cache, SSM cache, retained features, recurrent
RNG state, or observation buffer would need an explicit contract and budget
change. Fixed trained weights are not sequence memory. Query calibration is
external input on each query; if scene-dependent calibration is cached between
steps, it becomes additional persistent memory to declare and count. A future
save/restart test must compare both subsequent updates and
readouts after restoring only the declared state; no such cell test is claimed
by this first stage.

`ObservationStep.masked_images()` replaces unavailable pixels with zero before
Patchify and preserves the separate availability mask. Hidden NaNs are permitted
because they are never consumed; observed NaNs are rejected. Use
`validate_against(spec)` to check the fixed view count and that a step has between
one and `frames_per_step` real frames. An odd prefix ends in a partial step,
without invented padding timestamps. A full two-frame tubelet is available only
after both physical frames have arrived.

Query coordinates have an explicit `coordinate_space`:

- `image_pixel` (default): x right and y down in pixels, centers at `n + 0.5`,
  with z zero. Continuous subpixel queries are allowed; the interface does not
  infer image extents or enforce bounds without H/W.
- `world_xyz`: finite coordinates in metres in the declared scene world frame,
  matching camera poses.
- `triplane_uv`: normalized u/v in `[-1,1]`, z zero, with view IDs 0/1/2 for
  XY/XZ/YZ. Camera metadata is not allowed for this coordinate space.

Camera-view IDs have no implicit geometry. If calibration is supplied, IDs must
index that explicit camera table; moving-camera queries with different poses
require separate tables/calls. All calibration tensors must be float32 or
float64 independently of image, query or state dtype; float16 and bfloat16
calibration are rejected. A query timestamp alone does not perform a state
transition.

## What the synthetic crossing controls

`mavt4d.data.event_synthetic` renders a textured 3D box over a ground plane with a
fixed camera and exact geometry. Each source seed contributes all eight variants:

| Factor | Values | Meaning |
|---|---|---|
| Temporal core | `left_then_right`, `right_then_left` | Opposite ordering of a lateral excursion and return; left is image left |
| Temporal core | `clockwise`, `counterclockwise` | Opposite traversal of a world-horizontal loop; direction is defined from +z looking down |
| Configuration | `compact`, `wide` | Different path radii and terminal depths, crossed with all four cores |
| Source nuisance | Integer source seed | Appearance and camera nuisance shared by all eight variants |

`EventSceneConfig` defaults to `version="event-cd-v1"`, `T=12`, `H=64`, `W=96`,
`dt=1/12`, and `val_fraction=0.2`. T must be even and at least eight; H must be at
least 16 and W at least 24. Source seeds are integers in `[0, 2**63)`.

For a fixed source and configuration, **all four cores end with exactly the same
two RGB frames, positions, timestamps and camera calibration**. Their earlier
RGB histories differ. Each opposite-direction pair uses the same earlier poses
in reverse order and has exactly the same full-clip frame multiset. Consequently:

- The last two frames cannot identify any of the four cores within a matched
  source/configuration group.
- Order-invariant pooling of frames cannot distinguish either reversal pair.
- Colour, camera metadata and terminal pose cannot be shortcuts for the core.
- Configuration remains separately observable: the compact and wide terminal
  states differ even though the temporal core can be the same.

Every target has at least four visible pixels in every frame. This version has
no occluders, missing observations, flow or LiDAR. The geometry is useful for
evaluating known trajectories; it is never inserted into model inputs. The
four motion labels are a narrow temporal-memory diagnostic, not a definition
of the essence of arbitrary natural events, intention, or causality.

## In-memory use and causal prefixes

```python
from mavt4d.data.event_synthetic import EventSceneConfig, make_event_fixture

cfg = EventSceneConfig()
fixture = make_event_fixture(cfg, 17, "left_then_right", "compact")

# Frames 0 through 4 only: steps with 2, 2, and 1 real frames.
observations = fixture.observation_prefix(stop_frame=5)
for observation in observations:
    masked_rgb = observation.masked_images()  # Feed this to Patchify.
    # The later model consumes this observation and the previous CDState.

# Evaluation only; never pass this object to the observation encoder.
target = fixture.evaluation
```

`EventFixture.scene` is a camera-only `SceneSample` with RGB `[1,T,3,H,W]`,
timestamps `[1,T]`, intrinsics `[1,3,3]`, and camera-to-world transforms
`[1,T,4,4]`. `scene.boxes`, `scene.lidar_points`, and `scene.patch_flow` are `None`.
Fixture RGB is float32 in `[-1,1]`; the general observation interface does not
normalize or enforce this RGB range for other callers.
Use the prefix adapter rather than passing the entire scene object to a model:
scene IDs and other bookkeeping are not model features.

The separate `EventEvaluation` contains version, source seed, group, split, core,
configuration, target positions `[T,3]`, per-frame `Boxes3D`, and `SyntheticGT`
(`world_xyz`, `depth`, `instance`, `valid`). The box has `target_instance_id=1`.

`observation_prefix(stop_frame, availability=mask)` accepts a bool mask shaped
`[V,T,H,W]` or `[V,stop_frame,H,W]`. It returns only the observed prefix, zero-fills
missing RGB, and clones compact tensor storage. Neither future frames nor hidden
pixel values remain accessible through the returned tensors' backing storage.
The mask is an availability declaration; this initial suite does not create a
masking curriculum or supervise amodal reconstruction.

`EventSyntheticDataset(cfg, source_seeds, split)` constructs the complete crossing
for `"train"` or `"val"`. Its `.keys` are audit metadata, and
`.model_inputs(index, stop_frame, ...)` is the observation-only adapter. All
variants of a source use group key `synthetic-event:source-{seed}` and the existing
`mavt-split-v1` salted group hash. The group key excludes version, core and
configuration, so adding variants or reordering/expanding sources preserves
existing source assignments when the split fraction is unchanged. Small source
sets are not forced to contain a validation source.

## Reproducible export

Run the CPU exporter with the existing project environment:

```bash
PYTHONPATH=src python scripts/mavt4d/build_event_suite.py \
  --output data/eval/event_cd_v1 \
  --source-count 16 --seed-start 0 \
  --frames 12 --height 64 --width 96 --val-fraction 0.2
```

The CLI limits PyTorch to one CPU thread. It creates a new output directory and
refuses every existing path, including an empty directory; it never overwrites
or deletes a suite. Use a new output directory for each artifact version.

| Export | Contents |
|---|---|
| `manifest_v1.jsonl` | Opaque record ID, render-configuration fingerprint, group, split and relative input path |
| `inputs/<opaque-id>.pt` | Exactly `images`, `observed`, `frame_times`, `intrinsics`, `cam_to_world`; tensor values only |
| `eval_targets_v1.jsonl` | Evaluation labels, target ID and relative target path joined by record ID |
| `targets/<opaque-id>.pt` | Positions, per-frame boxes and pixel GT, all separate from inputs |
| `summary.json` | Configuration, sorted seeds, source/sample counts by split, observed audits and SHA256 checksums |
| `contact_sheet.png` | Four core rows for one source/configuration, with the shared final two frames outlined |
| `README.md` | Preview and input/target separation notes |

Only tensor values from the input whitelist are model observations. Do not encode
record IDs, group keys, split labels, file paths, evaluation labels or target
geometry as model features. Exported input files contain a complete clip for data
storage; a recurrent evaluator must still create causal prefixes. `torch.load`
with `weights_only=True` loads both input and target tensor dictionaries.

Every source is audited before its records are written: exact common suffix,
pairwise different histories, reversal equality and matching frame multiset,
core-independent camera metadata, grouped split, and target visibility. Failed
audits raise an error. `summary.json` is written last and marks a completed
export; a failure may leave a partial new directory with no summary. The exporter
does not remove or reuse that directory.

Sorted seeds and opaque names make manifests and tensor-file checksums repeatable
across output directories under the same software stack. Cross-version PyTorch
serialization bytes are not promised to match. The preview shows only one source;
the summary assertions cover all generated source/configuration combinations.

Each record identity includes a stable SHA256 fingerprint of the fixture version
and rendering parameters (T/H/W/dt). Changing these parameters changes record IDs
and the manifest hash while preserving source groups and split assignments at
the same split fraction. The fingerprint is included in both manifests and the
summary. It excludes output paths, Git state and the validation fraction; split
membership remains separately recorded in each manifest row.

## What to measure after a model exists

Keep the source-group split fixed. Read C and D after the common terminal
two-frame observation and compare a C-only core probe with a final-tubelet
baseline, an order-invariant baseline, and a D-only probe. Check both opposite
direction pairs separately: combining all four labels can hide an order-insensitive
model's ability to distinguish lateral motion from a loop. Probe configuration
separately and report the complete factor crossing. A successful probe alone
does not establish that D is dispensable or that the desired decomposition is
identifiable.

Later recurrence work still needs state-only restart equivalence, serial/parallel
update equivalence where applicable, explicit missing-observation behavior, and
decoder/rollout tests. Those checks are outside this fixture-and-contract stage.
