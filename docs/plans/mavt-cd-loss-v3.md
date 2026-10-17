# MAVT C/D loss v3 — direction B

Last updated: **2026-10-17**. Status: **accepted design; losses and training are not implemented or launched**.

[Ablation and evaluation](../benchmarks/mavt-cd-loss-v3-ablation.md) ·
[Same-frame diagnostic](../cd_diagnostic.md#same-frame-d-path-diagnostic-v2) ·
[Existing backbone](mavt-hybrid-v2-design.md)

## Decision and scope

Implement direction **B: new objectives with the existing decoder** first. Keep
Patchify, eight spatial Transformer blocks, two temporal attention/C/D scan
stages, their feedback, state sizes, KV cache, and four-block tile decoder.
No D-only RGB branch, extra persistent state, predictor/updater, or recurrent
feedback change is part of B. This document does not authorize optimizer steps
or a GPU launch. The separate same-frame diagnostic is read-only measurement.

The frozen v2 configuration, objective, checkpoints and historical measurements
remain unchanged. V3 requires a new config/version and run lock when implemented;
editing the v2 JSON in place is not an implementation of this plan.

## Representation contract

- **C_t** stores causal evidence about the event observed through t, together
  with source content needed to render it. C may change when new evidence arrives;
  it is not forced to be constant through an event or reduced to a class label.
- **D_t** stores configuration relative to that event, including within-event
  changes. D is not merely the difference between consecutive frames. Some
  temporal information in D is allowed; the goal is functional use of the factors.
- The decoder still reads C/D only. Continuation still needs C/D, KV and all
  clocks/metadata. Decode-only swaps do not create a valid recurrent history.

The current C-only head predicts signed spatial RGB differences. Changing
compact/wide configuration changes that target even for the same event. B removes
that head's loss from representation training and substitutes an event/order
relation objective. B does **not** add a C+D RGB-history auxiliary head.

The current normalized 16-D projection is insufficient to enforce invariance of
all decoder-facing C values. B uses consistency on the actual C bank. It does not
impose photometric invariance on raw C: appearance can be needed for rendering.
The frozen v2 projected invariance/noncollapse losses are replaced, not added
silently to the following objective.

## Sampling and eligibility

Use the existing source-group split and `event-cd-v1` four-core × two-configuration
fixtures. Keep all eight variants of a source together. A source is the unit of
loss aggregation, split and uncertainty. Metadata identifying source/core/config
is used by the loss-side pair builder only, never as model input.

**This is additional structured synthetic supervision relative to RGB-only v2.**
It tests whether the intended factorization can be learned under known pair
relations. It is not evidence of unsupervised factorization on natural video.

For each source, use endpoints t in `{1,3,5,7,9,11}` (zero-based frames), and the
same source-specific availability mask across its variants. Process only the
prefix through t before reading C_t/D_t. No future frame, availability or target
metadata may influence that state.

The initial B implementation uses a conservative eligibility policy:

1. Cross-configuration C consistency, event triplets and cross-reconstruction
   operate only when the entire prefix through t is observed in all involved
   variants. This is checked from input availability, not target visibility.
   The existing clean/wide/tube/snapshot/never-seen mask sampler stays unchanged.
   An early prefix before a mask starts can qualify; a partly hidden prefix cannot.
2. Positive pairs share source, core, endpoint/physical time and camera metadata;
   only compact/wide differs. There is no cross-source appearance invariance.
3. Negatives share source/configuration and differ in core: either the opposite
   direction within the motion family, or one of the two cores in the other
   family. For each anchor/negative core pair, both configurations must have a
   visible prefix witness by t: at least two distinct frames in the observed
   prefix contain at least four pixels whose maximum-channel paired RGB
   difference exceeds `1e-5`. The audit uses prefix pixels only. No negative is
   made merely from a future event label before evidence diverges.
4. A D-swap pair additionally requires at least four target pixels at t whose
   maximum-channel compact/wide RGB difference exceeds `1e-5`. Otherwise skip
   the functional swap term and report the count; a zero-distance target cannot
   establish useful D control.

Do not infer equivalent evidence from equal rectangular masks alone: objects can
occupy different masked regions. Relaxing the complete-prefix rule requires a
separately versioned evidence-equivalence policy. Masked self-reconstruction and
future prediction still train on every mask family. Thus B does not force a
confident event identity when all distinguishing observations were hidden.

## Proposed initial objective

These coefficients and normalizations are design defaults, to be frozen with
implementation tests before a future run. They are not measured improvements.

\[
L_{v3}=L_{obs}^{bal}+L_{miss}^{bal}+0.5L_{future}^{bal}
       +L_{order}+0.1L_{Cpair}+L_{swap}^{bal}+0.05L_{Cvar}.
\]

| Term | Initial weight | Definition and role |
|---|---:|---|
| Observed / missing RGB | 1 / 1 | Separately normalized, region-balanced self-reconstruction |
| Future RGB | 0.5 | Same prefix/rollout targets as v2; region-balanced error |
| Event/order relation | 1 | Prefix-qualified triplet on decoder-facing C |
| C consistency | 0.1 | Same-source/core/phase, different-configuration raw C agreement |
| Reciprocal D swap | 1 | Decode with C from one configuration and D from the other |
| C variance floor | 0.05 | Guard against collapse on eligible distinct event representatives |

For ordinary terms, first average valid variants/pairs and endpoints within each
source, then average sources equally. An empty source/term contributes zero and
its valid count is logged; do not divide by a global pixel count that lets one
source dominate. C variance has its explicitly defined group normalization below.
Log weighted and unweighted terms, active-source fraction and skipped-pair counts.
All masks and pair eligibility are detached loss-side targets.

### Region-balanced RGB

RGB remains in `[-1,1]`; losses use unclipped predictions. For each source define
a target-only support M as the union of (a) temporal RGB changes relative to frame
0 across every variant and (b) compact/wide RGB differences at matching frames.
Maximum channel difference greater than `1e-5` selects a pixel. This support is a
loss weight, not an encoder input, availability mask, query selector or visibility
oracle. It can use complete targets without allowing those targets into states.

For error `e=abs(prediction-target)` and a valid pixel subset V, average RGB
channels and pixels separately within `V∩M` and `V∩not-M`. Average the nonempty
region means equally. If both are empty, contribute zero and mark invalid.
Use V=observed pixels for L_obs and V=missing pixels for L_miss. Use the complete
image for future and swap terms. Report whole-image and both region errors too.
This changes weighting deliberately; the old whole-image metrics remain intact.

Future predictions still begin with the prefix through frame 5 and use
observationless rollout to frames 7 and 9. Targets are not assimilated. Missing
or future pixels can remain uncertain; reconstruction is not proof of recall.

### Full-C consistency and event relation

Flatten both C stages into `c` (2048 values under the current preset). For a
positive pair `(a,b)` use

\[
L_{Cpair}(a,b)=\operatorname{mean}((c_a-c_b)^2)/(0.1)^2.
\]

The scale `0.1` is a fixed unit, not a learned or batch-derived denominator. Both
sides receive gradients. The penalty covers every decoder-facing coordinate and
norm difference rather than only a learned projection. It is a soft constraint:
the decoder can still amplify small configuration differences. Report agreement,
variance and functional leakage per stage as diagnostics.

For event relations use `u=c/max(norm(c),1e-8)` and
`d(u,v)=(1-dot(u,v))/2`. With positive p from the opposite configuration and
eligible different-core negative n from the anchor configuration:

\[
L_{order}(a,p,n)=\max(0,\;0.2+d(u_a,u_p)-d(u_a,u_n)).
\]

For each anchor, allocate half the negative weight to its reversal and half to
the other motion family. Average eligible other-family negatives equally. If
only one category has a prefix witness, assign it the full weight; if neither
does, skip that anchor and log it. Then average eligible directed anchors and
both configurations equally within the source/endpoint. This prevents the
objective from treating two motion families with the same direction code as
an adequate separation, though it does not guarantee learned semantics.
The margin `0.2` is in the normalized distance's units. Do not mine negatives
using held-out scores or include unobserved-order labels. No learned projector
hides the constrained representation from the decoder. Normalization here does
not replace raw-C consistency above.

For each eligible source and endpoint where every core has both negative
categories qualified, average C over its two configurations to create four
event representatives.
For each C coordinate compute their population standard deviation
`sqrt(var + 1e-8)`. L_Cvar is `mean(relu(0.1-std)/0.1)`, then averaged over eligible
endpoints and sources (zero for a source with no valid endpoint). Config variants
are not counted as independent variance samples. This is a collapse guard, not
a proof of semantics; report raw RMS, per-coordinate std, norm distribution and
effective rank. Do not tune the floor after inspecting held-out results.

### Functional use of D

For an eligible same-event pair a/b, supervise both directions:

\[
R(C_{a,t},D_{b,t},q_t)\to x_{b,t},\qquad
R(C_{b,t},D_{a,t},q_t)\to x_{a,t}.
\]

Keep the receiver's C and explicit query metadata fixed; require matching
camera/time metadata and use the donor configuration's target. These are
decode-only interventions. Never feed a swapped code into a later recurrent
update with either original KV cache. A larger output change is useful only
when it improves the correct paired target, with self-reconstruction retained.

Reciprocal swaps within one core do not exclude D storing the core as well.
Keep cross-core/configuration portability checks as evaluation diagnostics:
hold configuration and swap C across cores, or use a D donor from a different
core but the same configuration. Use a target only where the generator contract
and a fixture audit establish the intended portable factors and corresponding
render. These are not additional B training losses by default. Mark unsupported
combinations unavailable rather than inventing targets.

At a common suffix, different cores have identical RGB by construction. Do not
require C swaps to produce different RGB there. Test C history with the frozen
direction probe; use distinguishable earlier endpoints for functional renders.
Likewise, C=0 giving RGB=0 is necessary for the existing content-path contract,
but does not rule out a constant C basis controlled almost entirely by D.

## Implementation gates and inference

Before any v3 training, tests must cover: pair eligibility and empty-set handling;
equal-source reduction; exact suffix/reversal fixture assertions; no future
dependency in prefix eligibility/state; correct reciprocal targets; gradients
to the actual C/D banks; parameter/state contract unchanged; deterministic
checkpoint/restart and step/chunk/scan parity. Loss tests may use backward;
this task does not authorize optimizer steps.

Inference is unchanged: availability zeros hidden pixels before Patchify, only
available observations update the causal state, and decoder queries read the
same C/D code. No pair labels or target-derived region masks are available at
inference. Recovery from history, recovery from another view and never-observed
content remain distinct evaluation conditions. The present synthetic fixture
does not establish multi-view or real-occluder performance.

The next implementation target is B, followed by its CPU correctness tests and
same-frame diagnostic. A training budget, resource allocation, run lock and
launch are separate work. The [2×2 protocol](../benchmarks/mavt-cd-loss-v3-ablation.md)
defines the comparison before any later measurements.
