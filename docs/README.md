# MAVT documentation

Last updated: **2026-10-17**.

## C/D loss v3 — selected next direction

- [Loss v3 plan](plans/mavt-cd-loss-v3.md): implemented direction B objectives with the existing backbone and decoder; CPU correctness checks completed, no completed matched learning comparison reported.
- [2×2 ablation](benchmarks/mavt-cd-loss-v3-ablation.md): objective × decoder comparison, common dev selection, unchanged held-out probes, and supervision/compute accounting.
- [Single-GPU loss v3 trial](runbooks/hybrid-loss-v3-trial.md): capacity calibration, common dev selection, local-last recovery, selected-best W&B artifacts and bounded stopping.
- [Same-frame D-path diagnostic](cd_diagnostic.md#same-frame-d-path-diagnostic-v2): measure D, modulation, routing/gain and output at the same endpoint, alongside the preserved legacy metrics.

## Hybrid C/D state v2

- [Giải thích kiến trúc bằng tiếng Việt](architecture/mavt-hybrid-explained-vi.md): ý tưởng C/D, flow từng khối, tensor I/O, parallel scan, decoder, missing observations và vai trò loss v3.
- [Design plan](plans/mavt-hybrid-v2-design.md): architecture, state contract, update equations and implementation scope.
- [Interactive architecture](architecture/mavt-hybrid-v2.html) · [diagram source](architecture/mavt-hybrid-v2.json): code-backed data flow pinned to the reference implementation. Download the HTML and open it in a browser; the GitHub file view does not render the interactive diagram.
- [Benchmark protocol](benchmarks/mavt-hybrid-v2-protocol.md): event-history tests for C, configuration tests for D, cache interventions and prerequisites for training.
- [Proposed paper benchmark tables](benchmarks/mavt-paper-benchmark-tables.md): five main comparisons and three ablation groups, with unmeasured results explicitly left blank.
- [Reference API](reference_hybrid_cd_model.md): snapshot/restart, causal execution, missing observations and tile decoding.
- [Architecture preset](../configs/mavt4d/hybrid_cd2_kv_v2.json).
- [Hybrid synthetic pilot v2](hybrid_synthetic_pilot_v2.md): separate training lock, DDP, W&B and checkpoint/recovery protocol.
- [Training architecture](architecture/mavt-hybrid-training.md): batched observer, C/D losses, distributed updates and deadline handling.

The hybrid began as a correctness reference. Technical tests establish contract and execution behavior within their tested scope; earlier experiment artifacts have their own protocols and results. Loss v3 now has a separate objective, evaluator and single-GPU runner. CPU checks include a tiny optimizer step; no GPU trial or scientific v3 results are reported here. This index makes no learning or GPU-performance claim from technical tests.

## Related contracts and baselines

- [C/D reference v1](reference_cd_model.md)
- [Synthetic event fixture contract](event_fixture_contract.md)
- [Synthetic C/D pilot v1](synthetic_cd_pilot.md)
- [Data contract](data_contract.md)

The hybrid benchmark is a separate planned protocol. It does not replace the frozen v1 pilot configuration or imply that its training results exist.
