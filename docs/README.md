# MAVT documentation

## Hybrid C/D state v2

- [Design plan](plans/mavt-hybrid-v2-design.md): architecture, state contract, update equations and implementation scope.
- [Interactive architecture](architecture/mavt-hybrid-v2.html) · [diagram source](architecture/mavt-hybrid-v2.json): code-backed data flow pinned to the reference implementation. Download the HTML and open it in a browser; the GitHub file view does not render the interactive diagram.
- [Benchmark protocol](benchmarks/mavt-hybrid-v2-protocol.md): event-history tests for C, configuration tests for D, cache interventions and prerequisites for training.
- [Reference API](reference_hybrid_cd_model.md): snapshot/restart, causal execution, missing observations and tile decoding.
- [Architecture preset](../configs/mavt4d/hybrid_cd2_kv_v2.json).

The hybrid is a correctness reference. Technical tests have passed; event/configuration learning and GPU performance have not been measured.

## Related contracts and baselines

- [C/D reference v1](reference_cd_model.md)
- [Synthetic event fixture contract](event_fixture_contract.md)
- [Synthetic C/D pilot v1](synthetic_cd_pilot.md)
- [Data contract](data_contract.md)

The hybrid benchmark is a separate planned protocol. It does not replace the frozen v1 pilot configuration or imply that its training results exist.
