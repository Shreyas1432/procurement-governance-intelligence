"""
Shared read-only bipartite edge-list loader for the PGI dashboard.

Routing decision (Task 6, edge-list routing gate): `data/results/rq1_edges.parquet`
does not exist and is not produced by any compute pipeline step. The RQ1 compute
module (`src/rq1_supplier_network/rq1_network.py::build_bipartite_graph`) builds
the buyer-supplier bipartite graph in-memory from the contracts feature table
and never persists it as a standalone edge-list artifact.

Rather than editing a frozen compute file to add a `write_parquet` call (Option
A, would need explicit sign-off and a determinism note), this loader
reproduces the SAME deterministic aggregation read-only, from the existing
locked features parquet (Option B, no compute-file change):

    contracts.group_by(["buyer_id", "supplier_id"])
             .agg(sum(contract_value), count(*))

This mirrors `NetworkIntelligence.build_bipartite_graph()` exactly (same
group-by keys, same aggregations) and was already independently implemented
this way in `pages/05_network_resilience.py`. Extracted here so every
dashboard consumer of the bipartite graph shares one read-only, deterministic
source instead of copy-pasting the aggregation.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import polars as pl
import streamlit as st


@st.cache_data(show_spinner="Loading network topology...")
def load_bipartite_edges(root: Path) -> pl.DataFrame:
    """Read-only buyer-supplier edge aggregation from the locked features
    parquet, restricted to the training window (matches the RQ1 compute
    module's graph-construction window exactly). Never mutates source data,
    never calls .fit()/.predict(), no sklearn/xgboost import."""
    import sys
    sys.path.insert(0, str(root))
    from src.core.config import TRAIN_YEARS
    from dashboard._auth import DASHBOARD_DATA_FEATURES

    features_path = DASHBOARD_DATA_FEATURES / "rq1_network_features.parquet"
    contracts = pl.read_parquet(features_path)
    return (
        contracts.filter(pl.col("award_year").is_between(min(TRAIN_YEARS), max(TRAIN_YEARS)))
        .group_by(["buyer_id", "supplier_id"])
        .agg(
            pl.sum("contract_value").alias("contract_value"),
            pl.len().alias("contract_count"),
        )
        .sort("contract_value", descending=True)
    )


def neighbourhood_subgraph(
    graph: nx.Graph,
    seed_nodes,
    node_info: dict,
    cap: int = 150,
) -> dict:
    """Return the seed node(s) plus their highest-exposure 1-hop neighbours,
    capped at `cap` total nodes, as a plain nodes/links dict ready for a
    JSON payload. Deterministic: neighbours are ranked by descending
    exposed_contract_value with node_id as the tiebreak.

    `node_info` is node_id -> row dict from resilience_exposure_table.parquet
    (must carry exposed_contract_value; node_type, community_id, and
    is_articulation_point are copied through when present). This function
    never recomputes a metric, it only decides which already-computed nodes
    fit in the render budget.

    If the seed set alone is >= cap (e.g. a large Louvain community), the
    seed set itself is truncated to the top `cap` by exposed value and no
    neighbours are added.
    """
    if isinstance(seed_nodes, str):
        seed_nodes = [seed_nodes]
    seed_nodes = [n for n in dict.fromkeys(seed_nodes) if n in graph]
    if not seed_nodes:
        return {"nodes": [], "links": []}

    def _value(node_id: str) -> float:
        return float(node_info.get(node_id, {}).get("exposed_contract_value", 0.0))

    seed_sorted = sorted(seed_nodes, key=lambda n: (-_value(n), n))

    if len(seed_sorted) >= cap:
        kept = seed_sorted[:cap]
    else:
        neighbour_candidates: set = set()
        for n in seed_sorted:
            neighbour_candidates.update(graph.neighbors(n))
        neighbour_candidates -= set(seed_sorted)
        ranked_neighbours = sorted(neighbour_candidates, key=lambda n: (-_value(n), n))
        kept = seed_sorted + ranked_neighbours[: cap - len(seed_sorted)]

    kept_set = set(kept)
    seed_set = set(seed_nodes)
    links = [
        {"source": u, "target": v}
        for u, v in graph.edges(kept)
        if u in kept_set and v in kept_set
    ]
    nodes = []
    for n in kept:
        info = node_info.get(n, {})
        default_type = "buyer" if graph.nodes[n].get("bipartite") == 0 else "supplier"
        nodes.append({
            "id": n,
            "node_type": info.get("node_type", default_type),
            "degree": int(graph.degree(n)),
            "exposed_contract_value": _value(n),
            "community_id": info.get("community_id"),
            "is_articulation_point": bool(info.get("is_articulation_point", False)),
            "is_seed": n in seed_set,
        })
    return {"nodes": nodes, "links": links}
