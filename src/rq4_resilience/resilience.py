"""Network resilience / what-if analytics on a sampled subset of the RQ1 graph.

This module is additive and descriptive: it quantifies structural exposure
when a node (buyer or supplier) is removed from a value-stratified sample of
the 2014-2018 training-period bipartite graph. It does not re-fit or touch
any RQ1/RQ2/RQ3 locked artifact, and it never runs on the full graph.

Framing (see docs/THRESHOLD_JUSTIFICATION.md, resilience section): the
dataset carries award-level contract value only. There is no re-procurement
cost, substitution price, delay cost, quality effect, or supplier financial
standing in the data, so this module cannot and does not quantify a
monetary consequence of removing a node. `exposed_contract_value` is the
factual sum of award value on a node's incident edges -- the value
currently routed through that node, nothing more.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.core.config import (
    DATA_FEATURES,
    DATA_RESULTS,
    RESILIENCE_COMPONENT_SHARE_GATE_TOLERANCE,
    RESILIENCE_RANK_WEIGHT_EXPOSED_VALUE,
    RESILIENCE_RANK_WEIGHT_FRAGMENTATION,
    RESILIENCE_RANK_WEIGHT_ORPHANED,
    RESILIENCE_SAMPLE_FRAC,
    RESILIENCE_SEED,
    TRAIN_YEARS,
)
from src.common.utils import get_logger, timed, write_json

logger = get_logger("rq4_resilience")

CAVEAT = (
    "exposed value is the award value currently routed through a node, "
    "not an estimate of financial loss."
)

import networkx as nx
import community as community_louvain


# ---------------------------------------------------------------------------
# Part 1: sampled graph construction
# ---------------------------------------------------------------------------

def _value_stratified_sample(all_edges: pl.DataFrame, frac: float, seed: int) -> pl.DataFrame:
    """Sample edges proportionally within each contract-value decile.

    ``all_edges`` must already be sorted deterministically by
    (contract_value desc, buyer_id, supplier_id), matching
    src.rq1_supplier_network.rq1_network.build_bipartite_graph. Because the
    frame is sorted by value, contiguous positional deciles ARE value
    deciles -- no separate quantile-boundary computation is needed, and the
    heavy tail is preserved by sampling within each decile rather than
    globally at random.

    Each decile draws from an independent seeded RNG (seed + decile index)
    so the sample is fully deterministic and reproducible.
    """
    n = all_edges.height
    row_idx = np.arange(n)
    decile = np.minimum((row_idx * 10) // max(n, 1), 9)

    chosen_parts = []
    for d in range(10):
        idx_in_decile = row_idx[decile == d]
        if idx_in_decile.size == 0:
            continue
        take = round(idx_in_decile.size * frac)
        take = min(take, idx_in_decile.size)
        rng = np.random.default_rng(seed + d)
        if take >= idx_in_decile.size:
            chosen = idx_in_decile
        else:
            chosen = rng.choice(idx_in_decile, size=take, replace=False)
        chosen_parts.append(chosen)

    selected = np.sort(np.concatenate(chosen_parts)) if chosen_parts else np.array([], dtype=int)
    # Selecting rows in ascending original-index order preserves the
    # deterministic (contract_value desc, buyer_id, supplier_id) ordering.
    return all_edges[selected.tolist()]


def load_sample(
    seed: int = RESILIENCE_SEED,
    frac: float = RESILIENCE_SAMPLE_FRAC,
    features_path: Path | None = None,
    results_dir: Path | None = None,
):
    """Load the curated 2014-2018 edge list, sample it, and build a bipartite graph.

    Returns (graph, coverage_stats). The graph mirrors
    src.rq1_supplier_network.rq1_network.build_bipartite_graph node/edge
    schema exactly: nodes are "B_{buyer_id}" / "S_{supplier_id}" with a
    ``bipartite`` attribute (0 buyer, 1 supplier), edges carry ``weight``
    (contract value) and ``contracts`` (contract count). Buyer nodes also
    carry a ``cpv`` attribute (their most frequent cpv_division in the
    sample), used for the substitutability proxy in simulate_removal.

    ``features_path``/``results_dir`` default to the module's normal
    DATA_FEATURES/DATA_RESULTS (the real compute pipeline's locations) and
    exist only so the dashboard's public build can point this same function
    at data/public/ without this module importing anything dashboard-specific.
    """
    contracts_path = features_path or (DATA_FEATURES / "rq1_network_features.parquet")
    if not contracts_path.exists():
        raise FileNotFoundError("Run feature engineering before the resilience module")
    contracts = pl.read_parquet(contracts_path)
    train_contracts = contracts.filter(
        pl.col("award_year").is_between(min(TRAIN_YEARS), max(TRAIN_YEARS))
    )

    all_edges = (
        train_contracts
        .group_by(["buyer_id", "supplier_id"])
        .agg(
            pl.sum("contract_value").alias("contract_value"),
            pl.len().alias("contract_count"),
            pl.col("cpv_division").mode().first().alias("cpv_division"),
        )
        .sort(
            ["contract_value", "buyer_id", "supplier_id"],
            descending=[True, False, False],
        )
    )
    total_pairs = all_edges.height
    total_value = float(all_edges["contract_value"].sum())
    n_buyers_full = all_edges["buyer_id"].n_unique()
    n_suppliers_full = all_edges["supplier_id"].n_unique()

    sample_edges = _value_stratified_sample(all_edges, frac, seed)

    G = nx.Graph()
    for row in sample_edges.iter_rows(named=True):
        buyer = f"B_{row['buyer_id']}"
        supplier = f"S_{row['supplier_id']}"
        G.add_node(buyer, bipartite=0, raw_id=row["buyer_id"])
        G.add_node(supplier, bipartite=1, raw_id=row["supplier_id"])
        G.add_edge(
            buyer, supplier,
            weight=float(row["contract_value"]),
            contracts=int(row["contract_count"]),
        )
        if row["cpv_division"] is not None:
            # Kept as the buyer's dominant cpv across all its sampled edges;
            # overwritten deterministically in sorted edge order.
            G.nodes[buyer].setdefault("_cpv_votes", {})
            G.nodes[buyer]["_cpv_votes"][row["cpv_division"]] = (
                G.nodes[buyer]["_cpv_votes"].get(row["cpv_division"], 0) + 1
            )

    for node, data in G.nodes(data=True):
        if data.get("bipartite") == 0:
            votes = data.pop("_cpv_votes", {})
            data["cpv"] = max(votes, key=votes.get) if votes else None

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_buyers_sample = sum(1 for _, d in G.nodes(data=True) if d.get("bipartite") == 0)
    n_suppliers_sample = sum(1 for _, d in G.nodes(data=True) if d.get("bipartite") == 1)
    sample_value = float(sample_edges["contract_value"].sum())

    if n_nodes > 0:
        components = list(nx.connected_components(G))
        giant_share_sample = max(len(c) for c in components) / n_nodes
    else:
        giant_share_sample = 0.0

    giant_share_full = _full_graph_giant_share(results_dir)

    coverage = {
        "sample_config": {
            "frac": frac,
            "seed": seed,
            "source_years": [min(TRAIN_YEARS), max(TRAIN_YEARS)],
            "source_file": "rq1_network_features.parquet",
        },
        "coverage": {
            "edge_frac_retained": sample_edges.height / total_pairs if total_pairs else 0.0,
            "node_frac_retained_buyers": n_buyers_sample / n_buyers_full if n_buyers_full else 0.0,
            "node_frac_retained_suppliers": n_suppliers_sample / n_suppliers_full if n_suppliers_full else 0.0,
            "value_frac_retained": sample_value / total_value if total_value else 0.0,
            "n_buyers": n_buyers_sample,
            "n_suppliers": n_suppliers_sample,
            "n_edges": n_edges,
            "largest_component_share_sample": giant_share_sample,
            "largest_component_share_full_rq1": giant_share_full,
        },
    }

    if giant_share_full is not None:
        delta = abs(giant_share_sample - giant_share_full)
        if delta > RESILIENCE_COMPONENT_SHARE_GATE_TOLERANCE:
            logger.warning(
                "Sample largest-component share (%.4f) differs from the full "
                "RQ1 training graph (%.4f) by %.4f, exceeding the %.2f gate "
                "tolerance; report this rather than treating the sample as "
                "structurally representative.",
                giant_share_sample, giant_share_full, delta,
                RESILIENCE_COMPONENT_SHARE_GATE_TOLERANCE,
            )

    logger.info(
        "Resilience sample: %d nodes (%d buyers, %d suppliers), %d edges, "
        "%.1f%% of edges / %.1f%% of value retained, giant component %.2f%% "
        "(full graph %.2f%%)",
        n_nodes, n_buyers_sample, n_suppliers_sample, n_edges,
        coverage["coverage"]["edge_frac_retained"] * 100.0,
        coverage["coverage"]["value_frac_retained"] * 100.0,
        giant_share_sample * 100.0,
        (giant_share_full or 0.0) * 100.0,
    )

    return G, coverage


def _full_graph_giant_share(results_dir: Path | None = None):
    """Read the already-characterised full-graph giant-component share from
    the locked RQ1 metrics, rather than rebuilding the ~296K-edge graph here.
    Returns None if RQ1 has not been run yet."""
    path = (results_dir or DATA_RESULTS) / "rq1_success_metrics.json"
    if not path.exists():
        return None
    import json
    with open(path) as handle:
        metrics = json.load(handle)
    return metrics.get("giant_component_frac")


# ---------------------------------------------------------------------------
# Part 2: node-removal what-if engine
# ---------------------------------------------------------------------------

def _normalize_removal_set(graph: "nx.Graph", nodes_to_remove) -> set:
    if isinstance(nodes_to_remove, str):
        nodes_to_remove = [nodes_to_remove]
    removal_set = set(nodes_to_remove)
    if not removal_set:
        raise ValueError("nodes_to_remove must be non-empty")
    missing = removal_set - set(graph.nodes())
    if missing:
        raise ValueError(f"Node(s) not found in graph: {sorted(missing)}")
    return removal_set


def _sole_source_buyer_count(graph: "nx.Graph") -> int:
    return sum(
        1 for n, d in graph.nodes(data=True)
        if d.get("bipartite") == 0 and graph.degree(n) == 1
    )


def _dependency_ratio(graph: "nx.Graph", buyer: str) -> float:
    """Max single-supplier value share of a buyer's incident edges. 0.0 if isolated."""
    neighbors = list(graph.neighbors(buyer))
    if not neighbors:
        return 0.0
    weights = [graph[buyer][s].get("weight", 0.0) for s in neighbors]
    total = sum(weights)
    return max(weights) / total if total else 0.0


def simulate_removal(graph: "nx.Graph", nodes_to_remove, community_partition: dict | None = None) -> dict:
    """Return structural what-if metrics for removing ``nodes_to_remove`` from ``graph``.

    Pure function of (graph, removal set, optional precomputed community
    partition): does not mutate ``graph`` and produces identical output for
    identical inputs. Raises ValueError if a node in the removal set is not
    present in the graph.

    ``substitutability`` counts suppliers active in the same CPV family as an
    affected buyer; it is an EXISTENCE proxy only -- it says nothing about
    supplier capacity, price competitiveness, or quality, and none of those
    are observable in this dataset.
    """
    removal_set = _normalize_removal_set(graph, nodes_to_remove)

    exposed_contract_value = sum(
        d.get("weight", 0.0) for _, _, d in graph.edges(removal_set, data=True)
    )

    neighbor_candidates = set()
    for n in removal_set:
        neighbor_candidates.update(graph.neighbors(n))
    neighbor_candidates -= removal_set

    orphaned_counterparties = sorted(
        c for c in neighbor_candidates
        if not (set(graph.neighbors(c)) - removal_set)
    )
    stranded_buyers = sorted(
        c for c in orphaned_counterparties if graph.nodes[c].get("bipartite") == 0
    )

    H = graph.copy()
    H.remove_nodes_from(removal_set)

    sole_source_before = _sole_source_buyer_count(graph)
    sole_source_after = _sole_source_buyer_count(H)

    affected_buyers = sorted(c for c in neighbor_candidates if graph.nodes[c].get("bipartite") == 0)
    dependency_shift = {}
    for buyer in affected_buyers:
        before = _dependency_ratio(graph, buyer)
        after = _dependency_ratio(H, buyer) if buyer in H else 0.0
        dependency_shift[buyer] = {"before": before, "after": after}

    total_nodes_before = graph.number_of_nodes()
    if total_nodes_before:
        components_before = list(nx.connected_components(graph))
        largest_component_share_before = max(len(c) for c in components_before) / total_nodes_before
    else:
        components_before = []
        largest_component_share_before = 0.0

    total_nodes_after = H.number_of_nodes()
    if total_nodes_after:
        components_after = list(nx.connected_components(H))
        largest_component_share_after = max(len(c) for c in components_after) / total_nodes_after
    else:
        largest_component_share_after = 0.0

    # Articulation-point check, once per DISTINCT component touched by the
    # removal set rather than once per removed node: nx.articulation_points
    # is O(V+E) on that component, and a multi-hundred-node removal set
    # (e.g. a whole Louvain community) would otherwise repeat that full-cost
    # scan hundreds of times on the same giant component.
    membership: dict = {}
    for idx, comp in enumerate(components_before):
        for node in comp:
            membership[node] = idx
    touched_comp_ids = {membership[n] for n in removal_set if n in membership}
    articulation_by_comp: dict = {}
    for comp_id in touched_comp_ids:
        comp = components_before[comp_id]
        if len(comp) < 3:
            articulation_by_comp[comp_id] = set()
            continue
        subgraph = graph.subgraph(comp)
        articulation_by_comp[comp_id] = set(nx.articulation_points(subgraph))
    is_articulation_point = {
        n: n in articulation_by_comp.get(membership.get(n), set()) for n in removal_set
    }

    if community_partition is None:
        community_partition = community_louvain.best_partition(graph, weight="weight", random_state=42)

    community_internal_value: dict = {}
    for u, v, d in graph.edges(data=True):
        cu, cv = community_partition.get(u), community_partition.get(v)
        if cu is not None and cu == cv:
            community_internal_value[cu] = community_internal_value.get(cu, 0.0) + d.get("weight", 0.0)

    removed_internal_value: dict = {}
    for u, v, d in graph.edges(removal_set, data=True):
        cu, cv = community_partition.get(u), community_partition.get(v)
        if cu is not None and cu == cv:
            removed_internal_value[cu] = removed_internal_value.get(cu, 0.0) + d.get("weight", 0.0)

    touched_ids = sorted({community_partition[n] for n in removal_set if n in community_partition})
    communities_touched = {
        cid: {
            "internal_value_removed_share": (
                removed_internal_value.get(cid, 0.0) / community_internal_value[cid]
                if community_internal_value.get(cid) else 0.0
            ),
        }
        for cid in touched_ids
    }

    substitutability = {}
    if affected_buyers:
        # Build cpv -> active-supplier-set once (O(V+E)) instead of rescanning
        # every supplier node for every affected buyer (O(affected_buyers *
        # n_suppliers)): the latter is fine for Phase A's single-node examples
        # but becomes tens of seconds for a multi-hundred-buyer removal set.
        cpv_to_suppliers: dict = {}
        for s, data in graph.nodes(data=True):
            if data.get("bipartite") != 1:
                continue
            for cpv in {graph.nodes[b].get("cpv") for b in graph.neighbors(s)}:
                if cpv is not None:
                    cpv_to_suppliers.setdefault(cpv, set()).add(s)

        for buyer in affected_buyers:
            cpv = graph.nodes[buyer].get("cpv")
            if cpv is None:
                substitutability[buyer] = 0
                continue
            substitutability[buyer] = len(cpv_to_suppliers.get(cpv, set()) - removal_set)

    return {
        "removed_nodes": sorted(removal_set),
        "exposed_contract_value": float(exposed_contract_value),
        "orphaned_counterparties": orphaned_counterparties,
        "sole_source_before": sole_source_before,
        "sole_source_after": sole_source_after,
        "dependency_shift": dependency_shift,
        "largest_component_share_before": largest_component_share_before,
        "largest_component_share_after": largest_component_share_after,
        "is_articulation_point": is_articulation_point,
        "communities_touched": communities_touched,
        "substitutability": substitutability,
        "stranded_buyers": stranded_buyers,
        "caveat": CAVEAT,
    }


# ---------------------------------------------------------------------------
# Part 3: per-node exposure table and systemic-importance ranking
# ---------------------------------------------------------------------------

def _fragment_sizes_via_block_cut_tree(graph: "nx.Graph", articulation_points: set) -> dict:
    """For every articulation point, compute the size of the largest fragment
    left WITHIN ITS OWN CONNECTED COMPONENT if that single node were removed.

    A naive approach (graph.copy() + remove_node + connected_components per
    articulation point) is O(n_articulation * (V+E)); in this bipartite
    hub-and-spoke procurement graph, any hub buyer with degree-1 supplier
    leaves is automatically a cut vertex, so n_articulation can be a large
    fraction of all nodes -- the naive approach does not finish in
    reasonable time. Instead this builds the block-cut tree once (biconnected
    components + articulation points, both O(V+E) via networkx) and does a
    single rooted subtree-weight pass over it: removing an articulation
    point v disconnects the block-cut tree at v into one branch per
    surviving block, and the resulting fragment sizes are exactly the
    subtree weights of those branches. Total cost is O(V+E) regardless of
    how many articulation points exist.
    """
    blocks = list(nx.biconnected_components(graph))
    bct = nx.Graph()
    block_weight: dict = {}
    for i, block in enumerate(blocks):
        bid = ("B", i)
        bct.add_node(bid)
        block_weight[bid] = sum(1 for n in block if n not in articulation_points)
        for n in block:
            if n in articulation_points:
                bct.add_edge(bid, ("A", n))

    fragment_after: dict = {}
    for comp_nodes in nx.connected_components(bct):
        root = next(iter(comp_nodes))
        parent = {root: None}
        children: dict = {}
        order = [root]
        for u, v in nx.bfs_edges(bct, root):
            parent[v] = u
            children.setdefault(u, []).append(v)
            order.append(v)

        own_weight = {nid: (1 if nid[0] == "A" else block_weight[nid]) for nid in comp_nodes}
        subtree_weight = dict(own_weight)
        for nid in reversed(order):
            p = parent.get(nid)
            if p is not None:
                subtree_weight[p] += subtree_weight[nid]

        total = subtree_weight[root]
        for nid in comp_nodes:
            if nid[0] != "A":
                continue
            node = nid[1]
            child_sizes = [subtree_weight[c] for c in children.get(nid, [])]
            upward = total - own_weight[nid] - sum(child_sizes)
            pieces = child_sizes + ([upward] if upward > 0 else [])
            fragment_after[node] = max(pieces) if pieces else 0

    return fragment_after


def build_exposure_table(graph: "nx.Graph") -> pl.DataFrame:
    """Precompute per-node structural exposure rows for every node in ``graph``.

    Uses an O(V+E) algebraic shortcut for non-articulation-point nodes
    (removing a node that doesn't disconnect its component always shrinks
    the giant component by exactly 1, with no further fragmentation -- if a
    node had a degree-1 leaf whose removal WOULD isolate that leaf, that
    node is itself an articulation point by definition, so it is never
    misclassified here) and the block-cut-tree pass above for articulation
    points, instead of recomputing connected components once per node,
    which would be O(n * (V+E)) and does not finish at sample scale.
    """
    total_nodes = graph.number_of_nodes()
    community_partition = community_louvain.best_partition(graph, weight="weight", random_state=42)

    components = list(nx.connected_components(graph))
    component_sizes = [len(c) for c in components]
    giant_size = max(component_sizes) if component_sizes else 0
    membership: dict = {}
    comp_size_of: dict = {}
    for idx, comp in enumerate(components):
        for node in comp:
            membership[node] = idx
        comp_size_of[idx] = len(comp)
    giant_comp_id = component_sizes.index(giant_size) if component_sizes else None
    comp_sizes_sorted = sorted(component_sizes, reverse=True)
    second_largest_other = comp_sizes_sorted[1] if len(comp_sizes_sorted) > 1 else 0

    articulation_points: set = set()
    for comp in components:
        if len(comp) < 3:
            continue
        subgraph = graph.subgraph(comp)
        articulation_points.update(nx.articulation_points(subgraph))

    fragment_after = _fragment_sizes_via_block_cut_tree(graph, articulation_points) if articulation_points else {}

    rows = []
    for node, data in graph.nodes(data=True):
        degree = graph.degree(node)
        exposed_value = graph.degree(node, weight="weight")
        orphaned_count = sum(1 for nb in graph.neighbors(node) if graph.degree(nb) == 1)
        is_ap = node in articulation_points
        remaining = total_nodes - 1

        if is_ap:
            comp_id = membership[node]
            if comp_id == giant_comp_id:
                new_giant = max(fragment_after[node], second_largest_other)
            else:
                new_giant = giant_size
        else:
            comp_id = membership[node]
            comp_size = comp_size_of[comp_id]
            if comp_size <= 1:
                new_giant = giant_size
            elif comp_id == giant_comp_id:
                new_giant = giant_size - 1
            else:
                new_giant = giant_size

        share_after = new_giant / remaining if remaining else 0.0

        rows.append({
            "node_id": node,
            "node_type": "buyer" if data.get("bipartite") == 0 else "supplier",
            "degree": int(degree),
            "exposed_contract_value": float(exposed_value),
            "orphaned_counterparties_count": int(orphaned_count),
            "is_articulation_point": bool(is_ap),
            "largest_component_share_after": float(share_after),
            "community_id": int(community_partition.get(node, -1)),
        })

    df = pl.DataFrame(rows)
    return _add_systemic_importance_rank(df)


def _add_systemic_importance_rank(df: pl.DataFrame) -> pl.DataFrame:
    """Composite systemic_importance_rank: weighted sum of percentile ranks of
    exposed value, orphaned count, and fragmentation impact (1 - share_after).

    Weights (RESILIENCE_RANK_WEIGHT_*) are documented in
    docs/THRESHOLD_JUSTIFICATION.md. This is an audit-prioritisation ordering,
    not a risk, corruption, or wrongdoing claim.
    """
    n = df.height
    if n == 0:
        return df.with_columns(pl.lit(None).alias("systemic_importance_rank"))
    df = df.with_columns(
        (pl.col("exposed_contract_value").rank(method="average") / n).alias("_pct_value"),
        (pl.col("orphaned_counterparties_count").rank(method="average") / n).alias("_pct_orphaned"),
        (1.0 - pl.col("largest_component_share_after")).alias("_fragmentation_impact"),
    )
    df = df.with_columns(
        (pl.col("_fragmentation_impact").rank(method="average") / n).alias("_pct_fragmentation"),
    )
    df = df.with_columns(
        (
            RESILIENCE_RANK_WEIGHT_EXPOSED_VALUE * pl.col("_pct_value")
            + RESILIENCE_RANK_WEIGHT_ORPHANED * pl.col("_pct_orphaned")
            + RESILIENCE_RANK_WEIGHT_FRAGMENTATION * pl.col("_pct_fragmentation")
        ).alias("systemic_importance_rank")
    )
    return df.drop(["_pct_value", "_pct_orphaned", "_fragmentation_impact", "_pct_fragmentation"])


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_resilience() -> bool:
    with timed(logger, "load resilience sample"):
        graph, coverage = load_sample()

    with timed(logger, "build exposure table"):
        exposure_table = build_exposure_table(graph)
        exposure_table.write_parquet(DATA_RESULTS / "resilience_exposure_table.parquet")

    n_articulation = int(exposure_table["is_articulation_point"].sum())
    n_stranded_candidates = int(
        exposure_table.filter(
            (pl.col("node_type") == "buyer") & (pl.col("orphaned_counterparties_count") == 0) & (pl.col("degree") == 1)
        ).height
    )
    n_communities = exposure_table["community_id"].n_unique()

    metrics = dict(coverage)
    metrics["rank_weights"] = {
        "exposed_value": RESILIENCE_RANK_WEIGHT_EXPOSED_VALUE,
        "orphaned_count": RESILIENCE_RANK_WEIGHT_ORPHANED,
        "fragmentation_impact": RESILIENCE_RANK_WEIGHT_FRAGMENTATION,
    }
    metrics["global_summary"] = {
        "n_articulation_points": n_articulation,
        "n_stranded_buyer_candidates_degree1": n_stranded_candidates,
        "n_communities": n_communities,
    }
    metrics["caveat"] = CAVEAT
    write_json(DATA_RESULTS / "resilience_metrics.json", metrics)

    logger.info(
        "Resilience run complete: %d articulation points, %d communities, "
        "exposure table written to resilience_exposure_table.parquet",
        n_articulation, n_communities,
    )
    print(
        f"Resilience sample: {coverage['coverage']['n_buyers']} buyers, "
        f"{coverage['coverage']['n_suppliers']} suppliers, "
        f"{coverage['coverage']['n_edges']} edges "
        f"({coverage['coverage']['edge_frac_retained']:.1%} of edges, "
        f"{coverage['coverage']['value_frac_retained']:.1%} of value)"
    )
    print(f"Articulation points: {n_articulation}; communities touched: {n_communities}")
    print(CAVEAT)
    return True


if __name__ == "__main__":
    run_resilience()
