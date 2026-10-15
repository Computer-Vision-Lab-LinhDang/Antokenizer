"""Edge counts, mean in-degree and provenance per edge type, averaged over prepared scenes.

Each scene goes through the training preparation: hold-out (tier A) -> hubs -> build_graph.
    python scripts/mavt4d/graph_stats.py --source synthetic --n 4
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mavt4d.data.holdout import apply_holdout  # noqa: E402
from mavt4d.data.synthetic import SyntheticSceneConfig, SyntheticSceneDataset  # noqa: E402
from mavt4d.geometry.hubs import build_hubs  # noqa: E402
from mavt4d.graph.builder import GraphConfig, build_graph  # noqa: E402
from mavt4d.graph.types import EdgeType  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic"], default="synthetic")  # nuScenes: plan T13/T27
    ap.add_argument("--n", type=int, default=4, help="number of scenes")
    ap.add_argument("--seed", type=int, default=0, help="first scene seed")
    ap.add_argument("--track-mode", choices=["correspondence", "same_pixel"], default="correspondence")
    args = ap.parse_args(argv)

    cfg = GraphConfig(track_mode=args.track_mode)
    data = SyntheticSceneDataset(args.n, SyntheticSceneConfig(), base_seed=args.seed)
    edges, degree, prov = Counter(), Counter(), {et.name: Counter() for et in EdgeType}
    for i in range(len(data)):
        sample = data[i]
        if sample.tier == "A":
            sample = apply_holdout(sample)
        hubs = build_hubs(sample, cfg.hubs_K, cfg.hubs_M)
        _, stats = build_graph(sample, hubs, cfg)
        edges.update(stats.num_edges)
        degree.update(stats.mean_in_degree)
        for name, counts in stats.provenance.items():
            prov[name].update(counts)

    print(f"source={args.source} scenes={len(data)} track_mode={args.track_mode} (edges summed, degree averaged)")
    print(f"{'type':<9} {'edges':>8} {'mean_in_degree':>15}  provenance")
    for et in EdgeType:
        p = ", ".join(f"{k}={v}" for k, v in sorted(prov[et.name].items())) or "-"
        print(f"{et.name:<9} {edges[et.name]:>8} {degree[et.name] / len(data):>15.2f}  {p}")


if __name__ == "__main__":
    main()
