# Diagnostic and natural-video training flows

Last updated: **2026-10-17**. The first diagrams document the existing protocols.
The loss v3 section now describes implemented B objectives with the same backbone
and decoder. CPU checks are complete; no GPU trial has launched at this update.

These runs use the existing hybrid C/D + KV architecture in independent model instances. They do not alter the core state contract or overwrite the original pilot. The diagram describes training and selection, not a new backbone.

```mermaid
flowchart TD
    subgraph S["Synthetic lane — two allocated GPUs"]
        S0["Audited core × configuration fixtures"] --> S1["Three fresh 200-update diagnostic arms:<br/>baseline, write_gate, d_strength<br/>At most two arms concurrently"]
        S1 --> S2["Explicit reviewed arm selection<br/>Training-source mechanism evidence only"]
        S2 --> S3["Fresh two-rank DDP run<br/>16 clips per update; original six losses"]
        S3 --> S4["10,000 updates<br/>May extend to 15,000 then 20,000<br/>Only when measured runtime fits the cutoff"]
        S4 --> S5["Complete fixed dev partition every 200 updates<br/>Select by unchanged weighted dev loss"]
    end
    subgraph N["Natural-video lane — six allocated GPUs"]
        N0["Bounded EPIC RGB export<br/>Recording-disjoint train/dev; canonical val excluded"] --> N1["Fresh six-rank DDP run<br/>48 clips per update; 600-update budget"]
        N1 --> N2["Observed/missing RGB + future RGB<br/>+ order-sensitive RGB history surrogate"]
        N2 --> N3["Complete fixed dev partition every 25 updates<br/>Select by fixed mean per-clip dev loss"]
    end
    S5 --> B["Per run: atomically replace only best.pt<br/>Retain model, optimizer, RNG and restart metadata"]
    N3 --> B
    B --> A["Upload only the final selected best artifact"]
    S3 --> W["W&B + local progress journal<br/>Train/dev curves, small inference panels,<br/>throughput, memory, utilization/power snapshots"]
    N1 --> W
    D["Optimizer cutoff → work cutoff → external hard stop<br/>All budgets remain bounded by these deadlines"] -.-> S4
    D -.-> N1
```

The short diagnostic arms each retain their best **training-source history-MSE** checkpoint; that selection is not validation. The later two-GPU run and the natural-video trial select by their own complete fixed dev partitions. An interrupted dev pass does not select a checkpoint. The retained best step may precede the final optimizer step. Runtime-only budget extensions never inspect loss or quality.

Each training run instantiates this same model structure, with its own weights and recurrent state:

```mermaid
flowchart LR
    X["RGB + availability + physical times"] --> P["Two-frame Patchify<br/>8 spatial Transformer blocks"]
    P --> T["Two temporal hybrid stages<br/>Causal attention + C/D scan"]
    R["Previous C/D + KV + clocks"] --> T
    T --> U["Updated C/D + KV + clocks<br/>Complete restart state"]
    U -->|"C/D only"| Q["4-block tile decoder<br/>Explicit space/time queries"]
    Q --> L["RGB reconstruction and future losses"]
    U -->|"C only"| H["Training-only ordered RGB history head"]
    U -. "next observation or observationless rollout" .-> R
```

KV participates in temporal updates and restart; the image decoder reads only C/D. Spatial BF16 does not change the FP32 C/D carries and decoder; KV retains its BF16 cache contract. This diagram shows the shared backbone and history head. The synthetic objective also uses a semantic projection for invariance and noncollapse, omitted here; the natural trial has no such projection head. A selected synthetic intervention changes only its declared write initialization or D modulation strength; the natural trial uses the default architecture and initialization policy specified in its protocol.

Neither objective guarantees C/D disentanglement. Improved history fitting, a larger D-swap prediction change, or lower natural-video reconstruction error does not alone demonstrate event semantics in C or configuration uniquely stored in D. The natural history target is an RGB-order surrogate, and no synthetic C/D labels are invented for natural clips. These runs do not use held-out results for tuning.

Detailed frozen protocols: [C/D diagnostics and subsequent two-GPU run](../cd_diagnostic.md), [natural-video trial](../natural_video_trial.md). See also the [hybrid training architecture](mavt-hybrid-training.md) for the original pilot flow.

## Same-frame read-only diagnostic

This diagram traces existing decoder operations at one endpoint. It adds
instrumentation, not another architecture or state transition. Legacy final-state
metrics remain separate from the new `aligned_d/frame_003/` measurements.

```mermaid
flowchart LR
    P["Observed prefix through frame 3<br/>Same source/core, two configurations"] --> S["Existing backbone<br/>C_a,D_a and C_b,D_b at frame 3"]
    S --> D["D_a versus D_b<br/>Raw paired RMS"]
    D --> M["Existing shared 16-D modulation<br/>Paired RMS"]
    M --> R["Per-block routing controls<br/>Paired RMS"]
    M --> G["Final gain controls<br/>Paired RMS / deviation from one"]
    S --> C["Fixed receiver C_a<br/>Same frame-3 queries and camera"]
    C --> O["Existing decoder<br/>Base, D swap, D zero"]
    R --> O
    G --> O
    O --> E["Output change and target errors<br/>Signed paired-target gain<br/>Full image + moving region / validity"]
    T["Frame-3 target pair<br/>Legacy target-only moving support"] --> E
```

Target support never enters the model. A swap changes D only for decoding;
it is not rolled forward with either source's KV. A final-state D distance cannot
be combined with this early-frame path to claim where information was lost.

## Implemented loss v3: B first

The [loss v3 implementation](../../src/mavt4d/hybrid_loss_v3/objectives.py) follows
the [plan](../plans/mavt-cd-loss-v3.md) and keeps the existing backbone and decoder.
It replaces the C-only spatial RGB-history and projected invariance
objectives with prefix-qualified event/order relations, consistency on actual C,
and a collapse guard. Region-balanced self/future RGB and reciprocal same-event
C/D cross-reconstruction retain rendering pressure. Pair metadata is additional
synthetic supervision and stays outside model inputs.

```mermaid
flowchart LR
    X["Same causal inputs and source splits"] --> B["Unchanged Patchify + Transformer<br/>+ two temporal C/D scan stages + KV"]
    B --> Z["Same decoder-facing C/D code"]
    Z --> R["Unchanged decoder<br/>Own C/D and decode-only paired C/D"]
    R --> RGB["Region-balanced RGB<br/>Self / missing / future / paired target"]
    Z --> C["Actual-C constraints<br/>Eligible event-order triplets<br/>Configuration consistency + variance floor"]
    P["Loss-side source/core/config relations<br/>Observed-prefix eligibility only"] --> RGB
    P --> C
```

The [2×2 ablation](../benchmarks/mavt-cd-loss-v3-ablation.md) compares old/new
objectives and shared/query-conditioned D controls. B has a separate
[single-GPU trial runner](../runbooks/hybrid-loss-v3-trial.md); query-conditioned
D remains a later decoder candidate, not part of the implemented architecture.
No new persistent memory is introduced by the B objective or proposed decoder
change. The complete 2×2 study and its scientific results remain pending.
