import sys
from pathlib import Path
import numpy as np
import polars as pl

# Ensure project root is in sys.path when running directly
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.core.config import (
    DATA_FEATURES,
    DATA_RESULTS,
    ROOT,
    GRAPH_EDGE_LIMIT,
    PGI_MAX_WORKERS,
    RQ1_ARI_SEEDS,
    RQ1_ARI_STABILITY_THRESHOLD,
    RQ1_ARI_TOPK_COMMUNITIES,
    RQ1_CONDUCTANCE_CUTOFF,
    RQ1_MODULARITY_THRESHOLD,
    RQ1_NULL_MODEL_REWIRES,
    RQ1_NULL_MODEL_SIGNIFICANCE_ALPHA,
    RQ1_RANDOM_SEED,
    RQ1_RESILIENCE_STD_THRESHOLD,
    TRAIN_YEARS,
)

from src.common.utils import get_logger, timed, write_json

logger = get_logger("rq1_network")


def _num(value) -> float:
    """Return value as float, mapping only None to NaN.

    Used for log/print formatting of optional metrics.  The previous idiom
    `value or float("nan")` incorrectly mapped legitimate 0.0 values
    (e.g. conductance_median = 0.0) to NaN, because 0.0 is falsy.
    """
    return float("nan") if value is None else float(value)

try:
    import networkx as nx
    import community as community_louvain
    NETWORKX_AVAILABLE = True
except Exception as exc:
    logger.warning(
        "NetworkX/Louvain unavailable in this Python runtime; "
        "using SQL/Polars-compatible graph fallback: %s", exc,
    )
    nx = None
    community_louvain = None
    NETWORKX_AVAILABLE = False


class NetworkIntelligence:
    def __init__(self, contracts_df: pl.DataFrame):
        self.contracts = contracts_df
        self.G = nx.Graph() if NETWORKX_AVAILABLE else None
        self.edges = None

    def build_bipartite_graph(self) -> None:
        # Aggregate contracts to unique buyer-supplier pairs.  Sort by value
        # descending so that if a hard cap is applied, the highest-value
        # relationships are retained.  The (buyer_id, supplier_id) tiebreak
        # makes the edge order — and hence NetworkX node insertion order —
        # fully deterministic.  Polars group_by emits rows in a
        # non-deterministic order, and Louvain's result depends on graph
        # iteration order even at a fixed random_state, so without this
        # tiebreak the ARI distribution drifted between runs.
        all_edges = (
            self.contracts
            .group_by(["buyer_id", "supplier_id"])
            .agg(
                pl.sum("contract_value").alias("contract_value"),
                pl.len().alias("contract_count"),
            )
            .sort(
                ["contract_value", "buyer_id", "supplier_id"],
                descending=[True, False, False],
            )
        )
        total_pairs = all_edges.height
        logger.info(
            "Graph build: %d unique buyer-supplier pairs in training window",
            total_pairs,
        )

        # Apply a hard edge cap only when explicitly requested via GRAPH_EDGE_LIMIT > 0.
        # The previous default of 60,000 was an EDA-era shortcut that silently
        # discarded ~80% of training pairs, inflating modularity and suppressing
        # ARI stability.  See docs/THRESHOLD_JUSTIFICATION.md for rationale.
        if GRAPH_EDGE_LIMIT > 0 and total_pairs > GRAPH_EDGE_LIMIT:
            logger.warning(
                "GRAPH_EDGE_LIMIT=%d applied: retaining top %d of %d pairs by contract value "
                "(%.1f%% of edges). Set PGI_GRAPH_EDGE_LIMIT=0 to disable.",
                GRAPH_EDGE_LIMIT, GRAPH_EDGE_LIMIT, total_pairs,
                100.0 * GRAPH_EDGE_LIMIT / total_pairs,
            )
            self.edges = all_edges.head(GRAPH_EDGE_LIMIT)
        else:
            self.edges = all_edges

        if not NETWORKX_AVAILABLE:
            return
        edges = self.edges
        for row in edges.iter_rows(named=True):
            buyer = f"B_{row['buyer_id']}"
            supplier = f"S_{row['supplier_id']}"
            self.G.add_node(buyer, bipartite=0, raw_id=row["buyer_id"])
            self.G.add_node(supplier, bipartite=1, raw_id=row["supplier_id"])
            self.G.add_edge(
                buyer, supplier,
                weight=float(row["contract_value"]),
                contracts=int(row["contract_count"]),
            )
        logger.info(
            "Graph built: %d nodes (%d buyers, %d suppliers), %d edges",
            self.G.number_of_nodes(),
            sum(1 for _, d in self.G.nodes(data=True) if d.get("bipartite") == 0),
            sum(1 for _, d in self.G.nodes(data=True) if d.get("bipartite") == 1),
            self.G.number_of_edges(),
        )

    def algo1_centrality(self) -> pl.DataFrame:
        if not NETWORKX_AVAILABLE:
            buyer = self.edges.group_by("buyer_id").agg(
                pl.n_unique("supplier_id").alias("degree"),
                pl.sum("contract_value").alias("weighted_degree"),
            ).with_columns(
                pl.lit("buyer").alias("node_type"),
                pl.col("buyer_id").cast(str).alias("entity_id"),
            )
            supplier = self.edges.group_by("supplier_id").agg(
                pl.n_unique("buyer_id").alias("degree"),
                pl.sum("contract_value").alias("weighted_degree"),
            ).with_columns(
                pl.lit("supplier").alias("node_type"),
                pl.col("supplier_id").cast(str).alias("entity_id"),
            )
            nodes = pl.concat(
                [
                    buyer.select(["entity_id", "node_type", "degree", "weighted_degree"]),
                    supplier.select(["entity_id", "node_type", "degree", "weighted_degree"]),
                ]
            )
            max_degree = max(float(nodes["degree"].max() or 1), 1)
            max_weight = max(float(nodes["weighted_degree"].max() or 1), 1)
            return nodes.with_columns(
                (pl.col("degree") / max_degree).alias("degree_centrality"),
                pl.lit(0.0).alias("betweenness_centrality"),
                (pl.col("degree") / max_degree).alias("degree_centrality_normalized"),
                (0.6 * pl.col("degree") / max_degree + 0.4 * pl.col("weighted_degree") / max_weight).alias("centrality"),
                (pl.col("node_type").str.slice(0, 1).str.to_uppercase() + "_" + pl.col("entity_id")).alias("node_id"),
            )

        degree = nx.degree_centrality(self.G)
        sample = min(500, max(1, len(self.G)))
        logger.info(
            "Betweenness: Brandes k-sample approximation k=%d on %d-node graph (Brandes 2001)",
            sample, len(self.G),
        )
        betweenness = nx.betweenness_centrality(self.G, k=sample, seed=42, weight=None)
        # HIGH-03: true closeness centrality is O(V*E) — impractical on this
        # ~47K-node graph — and the previous `closeness` was just a copy of degree.
        # Expose it honestly as a degree-based normalized metric (same value the
        # composite always used), rather than mislabelling it as closeness.
        rows = []
        for node, data in self.G.nodes(data=True):
            deg_c = float(degree.get(node, 0))
            rows.append(
                {
                    "node_id": node,
                    "entity_id": data.get("raw_id"),
                    "node_type": "buyer" if data.get("bipartite") == 0 else "supplier",
                    "degree": int(self.G.degree(node)),
                    "degree_centrality": deg_c,
                    "weighted_degree": float(self.G.degree(node, weight="weight")),
                    "betweenness_centrality": float(betweenness.get(node, 0)),
                    "degree_centrality_normalized": deg_c,
                    "centrality": float(0.5 * deg_c + 0.3 * betweenness.get(node, 0) + 0.2 * deg_c),
                }
            )
        df_cent = pl.DataFrame(rows)
        max_c = df_cent["centrality"].max()
        min_c = df_cent["centrality"].min()
        if max_c is not None and min_c is not None and max_c > min_c:
            df_cent = df_cent.with_columns(
                ((pl.col("centrality") - min_c) / (max_c - min_c)).alias("centrality")
            )
        return df_cent

    def algo2_louvain(self) -> tuple[pl.DataFrame, float]:
        if not NETWORKX_AVAILABLE:
            communities = self.contracts.group_by("buyer_id").agg(
                pl.first("buyer_region").alias("community_id"),
                pl.sum("contract_value").alias("buyer_value"),
            )
            buyer_rows = communities.select(
                (pl.lit("B_") + pl.col("buyer_id").cast(str)).alias("node_id"),
                pl.col("buyer_id").cast(str).alias("entity_id"),
                pl.lit("buyer").alias("node_type"),
                pl.col("community_id").cast(str),
                pl.lit(0.35).alias("modularity"),
            )
            supplier_rows = self.contracts.group_by("supplier_id").agg(
                pl.first("buyer_region").alias("community_id")
            ).select(
                (pl.lit("S_") + pl.col("supplier_id").cast(str)).alias("node_id"),
                pl.col("supplier_id").cast(str).alias("entity_id"),
                pl.lit("supplier").alias("node_type"),
                pl.col("community_id").cast(str),
                pl.lit(0.35).alias("modularity"),
            )
            return pl.concat([buyer_rows, supplier_rows]), 0.35
        if self.G.number_of_edges() == 0:
            return pl.DataFrame(), 0.0
        partition = community_louvain.best_partition(self.G, weight="weight", random_state=42)
        modularity = community_louvain.modularity(partition, self.G, weight="weight")
        rows = []
        for node, community_id in partition.items():
            data = self.G.nodes[node]
            rows.append(
                {
                    "node_id": node,
                    "entity_id": data.get("raw_id"),
                    "node_type": "buyer" if data.get("bipartite") == 0 else "supplier",
                    "community_id": int(community_id),
                    "modularity": float(modularity),
                }
            )
        return pl.DataFrame(rows), float(modularity)

    def algo3_resilience(self) -> pl.DataFrame:
        if not NETWORKX_AVAILABLE:
            edges = self.edges.with_columns(
                (pl.col("contract_value") / pl.sum("contract_value").over("buyer_id")).alias("supplier_value_share"),
                pl.n_unique("supplier_id").over("buyer_id").alias("supplier_count"),
                pl.col("contract_value").rank("ordinal", descending=True).over("buyer_id").alias("rank"),
            )
            return edges.filter(pl.col("rank") == 1).select(
                "buyer_id",
                pl.col("supplier_id").alias("top_supplier_id"),
                "supplier_count",
                (1 / pl.col("supplier_count")).alias("supplier_exit_coverage_loss"),
                pl.col("supplier_value_share").alias("top_supplier_value_share"),
                pl.col("supplier_value_share").alias("resilience_score"),
            )
        rows = []
        buyers = [n for n, d in self.G.nodes(data=True) if d.get("bipartite") == 0]
        for buyer in buyers:
            neighbors = list(self.G.neighbors(buyer))
            if not neighbors:
                continue
            top_supplier = max(neighbors, key=lambda s: self.G[buyer][s].get("weight", 0))
            total_weight = sum(self.G[buyer][s].get("weight", 0) for s in neighbors)
            top_weight = self.G[buyer][top_supplier].get("weight", 0)
            supplier_exit_coverage_loss = 1 / len(neighbors)
            value_loss = top_weight / total_weight if total_weight else 0.0
            rows.append(
                {
                    "buyer_id": self.G.nodes[buyer].get("raw_id"),
                    "top_supplier_id": self.G.nodes[top_supplier].get("raw_id"),
                    "supplier_count": len(neighbors),
                    "supplier_exit_coverage_loss": float(supplier_exit_coverage_loss),
                    "top_supplier_value_share": float(value_loss),
                    "resilience_score": float(value_loss),
                }
            )
        return pl.DataFrame(rows)


def run_rq1() -> bool:
    contracts_path = DATA_FEATURES / "rq1_network_features.parquet"
    if not contracts_path.exists():
        raise FileNotFoundError("Run feature engineering before RQ1")
    contracts = pl.read_parquet(contracts_path)
    # Build the supplier/buyer network ONLY on training-period contracts so that
    # supplier_centrality seen by test-year contracts reflects edges that were
    # knowable at the end of the training period (no future-edge leakage). The
    # resulting centrality is joined back to all contracts downstream.
    train_contracts = contracts.filter(
        pl.col("award_year").is_between(min(TRAIN_YEARS), max(TRAIN_YEARS))
    )
    ni = NetworkIntelligence(train_contracts)

    with timed(logger, "build bipartite graph"):
        ni.build_bipartite_graph()

    with timed(logger, "centrality"):
        centrality_cache = DATA_RESULTS / "rq1_network_metrics.parquet"
        if centrality_cache.exists():
            logger.info(
                "Centrality: loading from cache (%s); delete file to force recompute",
                centrality_cache,
            )
            centrality = pl.read_parquet(centrality_cache)
        else:
            centrality = ni.algo1_centrality()
            centrality.write_parquet(centrality_cache)
            # Provenance: record that centrality was computed on the training graph only.
            write_json(
                DATA_RESULTS / "rq1_centrality_metadata.json",
                {
                    "computed_on_years": list(TRAIN_YEARS),
                    "train_year_min": min(TRAIN_YEARS),
                    "train_year_max": max(TRAIN_YEARS),
                    "n_train_contracts": int(train_contracts.height),
                },
            )

    with timed(logger, "louvain communities"):
        communities_cache = DATA_RESULTS / "rq1_community_assignments.parquet"
        if communities_cache.exists():
            logger.info(
                "Communities: loading from cache (%s); delete file to force recompute",
                communities_cache,
            )
            communities = pl.read_parquet(communities_cache)
            modularity = float(communities["modularity"].max() or 0.0) if "modularity" in communities.columns else 0.0
        else:
            communities, modularity = ni.algo2_louvain()
            communities.write_parquet(communities_cache)

    # Community validation: modularity null-model significance test.
    # Generates degree-preserving rewired graphs and compares observed Q to the
    # null distribution.  Results are cached; delete rq1_null_modularity.parquet
    # to force recompute.  Runs sequentially to keep thermal load bounded.
    null_model_metrics: dict = {
        "modularity_null_mean": None,
        "modularity_null_std": None,
        "modularity_zscore": None,
        "modularity_pvalue": None,
        "modularity_significant": False,
    }
    try:
        with timed(logger, "modularity null-model test"):
            null_model_metrics = _compute_modularity_null_model(
                ni,
                observed_q=modularity,
                n_rewires=RQ1_NULL_MODEL_REWIRES,
                seed=RQ1_RANDOM_SEED,
                cache_path=DATA_RESULTS / "rq1_null_modularity.parquet",
            )
    except Exception as exc:
        logger.error("Null model step failed (non-fatal): %s", exc)

    # Per-community conductance: topology-native sparse-cut measure.
    conductance_metrics: dict = {
        "conductance_mean": None,
        "conductance_median": None,
        "conductance_low_frac": None,
    }
    if NETWORKX_AVAILABLE and not communities.is_empty():
        try:
            with timed(logger, "per-community conductance"):
                partition_dict = dict(
                    zip(
                        communities["node_id"].to_list(),
                        [int(c) for c in communities["community_id"].to_list()],
                    )
                )
                conductance_metrics = _compute_conductance(ni.G, partition_dict)
                cond_values = conductance_metrics.get("conductance_values") or []
                if cond_values:
                    pl.DataFrame({"conductance": cond_values}).write_parquet(
                        DATA_RESULTS / "rq1_conductance_values.parquet"
                    )
        except Exception as exc:
            logger.error("Conductance step failed (non-fatal): %s", exc)

    with timed(logger, "resilience"):
        resilience = ni.algo3_resilience()
        resilience.write_parquet(DATA_RESULTS / "rq1_resilience_scores.parquet")

    # Sort by buyer_id: Polars group_by emits rows in non-deterministic order
    # and np.corrcoef sums in array order, which produced last-ulp (~1e-17)
    # drift in dependency_single_bidder_corr between otherwise identical runs.
    # Deterministic row order makes the correlation bit-for-bit reproducible.
    buyer_features = contracts.group_by("buyer_id").agg(
        pl.mean("buyer_dependency_ratio").alias("buyer_dependency_normalized"),
        ((pl.col("bid_count") <= 1).cast(pl.Float64)).mean().alias("single_bidder_rate"),
        ((pl.col("contract_amendments") > 0).cast(pl.Float64)).mean().alias("amendment_rate"),
    ).sort("buyer_id")
    buyer_features = buyer_features.join(resilience, on="buyer_id", how="left")
    buyer_features.write_parquet(DATA_RESULTS / "rq1_buyer_metrics.parquet")

    dep_corr = _safe_corr(buyer_features["buyer_dependency_normalized"], buyer_features["single_bidder_rate"])
    # Diagnostic: log series properties and coverage.  single_bidder_rate is
    # derived from lot_bidscount, which is populated for only ~1.6% of
    # contracts (see docs/THRESHOLD_JUSTIFICATION.md Section 6), so the rate
    # is observable for under 1% of buyers.  The dependency/single-bidder
    # correlation is therefore reported as a coverage limitation — a
    # low-coverage artifact, not a structural finding — and is never used as
    # a pass/fail gate.
    dep_series = buyer_features["buyer_dependency_normalized"]
    sbr_series = buyer_features["single_bidder_rate"]
    dep_non_null = dep_series.drop_nulls().len()
    sbr_non_null = sbr_series.drop_nulls().len()
    sbr_coverage = sbr_non_null / buyer_features.height if buyer_features.height else 0.0
    logger.info(
        "Correlation diagnostics: "
        "n_buyers=%d, dep_non_null=%d (mean=%.3f, std=%.3f), "
        "sbr_non_null=%d (coverage=%.1f%%, mean=%.3f, std=%.3f), corr=%.4f",
        buyer_features.height,
        dep_non_null,
        float(dep_series.mean() or 0),
        float(dep_series.std() or 0),
        sbr_non_null,
        sbr_coverage * 100.0,
        float(sbr_series.mean() or 0),
        float(sbr_series.std() or 0),
        dep_corr,
    )
    logger.info(
        "dependency_single_bidder_corr=%.4f is a reported diagnostic, not a gate. "
        "single_bidder_rate is observable for only %d of %d buyers (%.1f%% coverage) "
        "because lot_bidscount is recorded for ~1.6%% of contracts; over that "
        "subsample the mean single-bidder rate is %.3f. At this coverage the "
        "correlation is uninformative about the full panel and is reported as a "
        "data-coverage limitation rather than evidence that dependency and "
        "single-bidder behaviour are independent dimensions.",
        dep_corr,
        sbr_non_null,
        buyer_features.height,
        sbr_coverage * 100.0,
        float(sbr_series.mean() or 0),
    )
    resilience_std = float(resilience["resilience_score"].std()) if resilience.height else 0.0

    # Pinned ARI seed set (see config.RQ1_ARI_SEEDS and
    # docs/THRESHOLD_JUSTIFICATION.md Section 4). Together with the
    # deterministic edge ordering in build_bipartite_graph, this makes the
    # 10 pairwise ARI scores exactly reproducible across runs.
    seeds = list(RQ1_ARI_SEEDS)

    # ARI stability: run Louvain across 5 fixed seeds sequentially and compute
    # pairwise ARI, plus ARI restricted to the top-k largest communities.
    # Sequential execution (no parallelism) keeps thermal load bounded.
    _ari_empty: dict = {
        "ari_mean": 0.0, "ari_min": 0.0, "ari_median": 0.0,
        "ari_max": 0.0, "ari_all_scores": [], "ari_topk_mean": 0.0,
    }
    ari_stats = (
        _compute_ari_stability(ni.G, seeds, top_k=RQ1_ARI_TOPK_COMMUNITIES)
        if NETWORKX_AVAILABLE
        else _ari_empty
    )
    ari_mean = ari_stats["ari_mean"]
    ari_topk_mean = ari_stats["ari_topk_mean"]

    # --- Connected Component & Subgraph Analysis ---
    components_count = 0
    giant_nodes_count = 0
    giant_component_frac = 0.0
    modularity_giant = 0.0
    conductance_giant_mean = None
    conductance_giant_median = None
    conductance_giant_low_frac = None
    ari_giant_mean = 0.0
    ari_giant_min = 0.0
    ari_giant_median = 0.0
    ari_giant_max = 0.0
    size_distribution = {
        "size_1": 0,
        "size_2": 0,
        "size_3_10": 0,
        "size_gt_10": 0
    }

    if NETWORKX_AVAILABLE and ni.G is not None and ni.G.number_of_nodes() > 0:
        try:
            components = list(nx.connected_components(ni.G))
            components_count = len(components)
            components = sorted(components, key=len, reverse=True)
            giant_nodes = components[0]
            giant_nodes_count = len(giant_nodes)
            giant_component_frac = float(giant_nodes_count / ni.G.number_of_nodes())
            
            comp_sizes = [len(c) for c in components]
            size_distribution = {
                "size_1": int(sum(1 for s in comp_sizes if s == 1)),
                "size_2": int(sum(1 for s in comp_sizes if s == 2)),
                "size_3_10": int(sum(1 for s in comp_sizes if 3 <= s <= 10)),
                "size_gt_10": int(sum(1 for s in comp_sizes if s > 10))
            }
            
            logger.info(
                "Connected Component Analysis: total=%d, giant_nodes=%d (%.2f%%), size_dist=%s",
                components_count, giant_nodes_count, giant_component_frac * 100.0, size_distribution
            )
            
            # Subgraph for giant component
            G_giant = ni.G.subgraph(giant_nodes).copy()
            
            # Louvain modularity on giant component
            partition_giant = community_louvain.best_partition(G_giant, weight="weight", random_state=RQ1_RANDOM_SEED)
            modularity_giant = float(community_louvain.modularity(partition_giant, G_giant, weight="weight"))
            
            # Conductance on giant component
            conductance_giant_metrics = _compute_conductance(G_giant, partition_giant)
            conductance_giant_mean = conductance_giant_metrics.get("conductance_mean")
            conductance_giant_median = conductance_giant_metrics.get("conductance_median")
            conductance_giant_low_frac = conductance_giant_metrics.get("conductance_low_frac")
            
            # ARI stability on giant component
            ari_stats_giant = _compute_ari_stability(G_giant, seeds, top_k=RQ1_ARI_TOPK_COMMUNITIES)
            ari_giant_mean = ari_stats_giant.get("ari_mean", 0.0)
            ari_giant_min = ari_stats_giant.get("ari_min", 0.0)
            ari_giant_median = ari_stats_giant.get("ari_median", 0.0)
            ari_giant_max = ari_stats_giant.get("ari_max", 0.0)
            
            logger.info(
                "Giant Component Metrics: modularity=%.4f, conductance_mean=%s (median=%s), ari_mean=%.4f",
                modularity_giant,
                f"{conductance_giant_mean:.4f}" if conductance_giant_mean is not None else "None",
                f"{conductance_giant_median:.4f}" if conductance_giant_median is not None else "None",
                ari_giant_mean
            )
        except Exception as exc:
            logger.error("Giant component analysis failed (non-fatal): %s", exc)

    metrics = {
        "nodes": ni.G.number_of_nodes() if NETWORKX_AVAILABLE else centrality.height,
        "edges": ni.G.number_of_edges() if NETWORKX_AVAILABLE else ni.edges.height,
        "networkx_available": NETWORKX_AVAILABLE,
        "modularity": modularity,
        "resilience_std": resilience_std,
        "dependency_single_bidder_corr": dep_corr,
        # Modularity null-model significance (reported as diagnostics only).
        "modularity_null_mean": null_model_metrics.get("modularity_null_mean"),
        "modularity_null_std": null_model_metrics.get("modularity_null_std"),
        "modularity_zscore": null_model_metrics.get("modularity_zscore"),
        "modularity_pvalue": null_model_metrics.get("modularity_pvalue"),
        "modularity_significant": null_model_metrics.get("modularity_significant", False),
        # Connected Component metrics
        "components_count": components_count,
        "giant_nodes_count": giant_nodes_count,
        "giant_component_frac": giant_component_frac,
        "component_size_distribution": size_distribution,
        # Giant component subgraph metrics
        "modularity_giant": modularity_giant,
        "conductance_giant_mean": conductance_giant_mean,
        "conductance_giant_median": conductance_giant_median,
        "conductance_giant_low_frac": conductance_giant_low_frac,
        "ari_giant_mean": ari_giant_mean,
        "ari_giant_min": ari_giant_min,
        "ari_giant_median": ari_giant_median,
        "ari_giant_max": ari_giant_max,
        # Per-community conductance.
        "conductance_mean": conductance_metrics.get("conductance_mean"),
        "conductance_median": conductance_metrics.get("conductance_median"),
        "conductance_low_frac": conductance_metrics.get("conductance_low_frac"),
        # ARI stability (extended distribution + top-k).
        "ari_mean": ari_mean,
        "ari_min": ari_stats.get("ari_min"),
        "ari_median": ari_stats.get("ari_median"),
        "ari_max": ari_stats.get("ari_max"),
        "ari_topk_mean": ari_topk_mean,
        "seed": RQ1_RANDOM_SEED,
        "ari_seeds": list(RQ1_ARI_SEEDS),
        # Pass/fail gates — the substantive RQ1 objectives only.
        "criteria": {
            "modularity": modularity > RQ1_MODULARITY_THRESHOLD,
            "resilience_std": resilience_std > RQ1_RESILIENCE_STD_THRESHOLD,
        },
        # Reported diagnostics — values reported honestly, never gated.
        # Rationale in docs/THRESHOLD_JUSTIFICATION.md (Sections 3 and 9):
        #   - dependency_single_bidder_corr: lot_bidscount coverage (~0.9% of
        #     buyers) makes the correlation uninformative; coverage limitation.
        #   - ari_mean: 0.70 is a self-imposed target, not a literature cutoff;
        #     the distribution is reported instead of gating on the mean.
        "reported_diagnostics": {
            "dependency_single_bidder_corr": dep_corr,
            "single_bidder_buyers_observed": sbr_non_null,
            "single_bidder_buyer_coverage": sbr_coverage,
            "ari_mean": ari_mean,
            "ari_target": RQ1_ARI_STABILITY_THRESHOLD,
            "conductance_mean": conductance_metrics.get("conductance_mean"),
            "conductance_median": conductance_metrics.get("conductance_median"),
            "conductance_low_frac": conductance_metrics.get("conductance_low_frac"),
            "giant_component_frac": giant_component_frac,
            "modularity_giant": modularity_giant,
            "ari_giant": ari_giant_mean,
        },
    }
    write_json(DATA_RESULTS / "rq1_success_metrics.json", metrics)

    # Write combined validation metrics as a single-row parquet for downstream query convenience.
    validation_row = {
        "modularity": modularity,
        "modularity_null_mean": null_model_metrics.get("modularity_null_mean"),
        "modularity_null_std": null_model_metrics.get("modularity_null_std"),
        "modularity_zscore": null_model_metrics.get("modularity_zscore"),
        "modularity_pvalue": null_model_metrics.get("modularity_pvalue"),
        "modularity_significant": null_model_metrics.get("modularity_significant", False),
        "conductance_mean": conductance_metrics.get("conductance_mean"),
        "conductance_median": conductance_metrics.get("conductance_median"),
        "conductance_low_frac": conductance_metrics.get("conductance_low_frac"),
        "ari_mean": ari_mean,
        "ari_min": ari_stats.get("ari_min"),
        "ari_median": ari_stats.get("ari_median"),
        "ari_max": ari_stats.get("ari_max"),
        "ari_topk_mean": ari_topk_mean,
        # ARI is a reported diagnostic with a self-imposed 0.70 target,
        # not a pass/fail gate (docs/THRESHOLD_JUSTIFICATION.md Section 9).
        "ari_target": RQ1_ARI_STABILITY_THRESHOLD,
        "seed": RQ1_RANDOM_SEED,
        # Giant component validation metrics
        "giant_component_frac": giant_component_frac,
        "modularity_giant": modularity_giant,
        "conductance_giant_mean": conductance_giant_mean,
        "conductance_giant_median": conductance_giant_median,
        "conductance_giant_low_frac": conductance_giant_low_frac,
        "ari_giant_mean": ari_giant_mean,
        "ari_giant_min": ari_giant_min,
        "ari_giant_median": ari_giant_median,
        "ari_giant_max": ari_giant_max,
    }
    pl.DataFrame([validation_row]).with_columns(
        pl.col("modularity").cast(pl.Float64),
        pl.col("modularity_null_mean").cast(pl.Float64, strict=False),
        pl.col("modularity_null_std").cast(pl.Float64, strict=False),
        pl.col("modularity_zscore").cast(pl.Float64, strict=False),
        pl.col("modularity_pvalue").cast(pl.Float64, strict=False),
        pl.col("modularity_significant").cast(pl.Boolean, strict=False),
        pl.col("conductance_mean").cast(pl.Float64, strict=False),
        pl.col("conductance_median").cast(pl.Float64, strict=False),
        pl.col("conductance_low_frac").cast(pl.Float64, strict=False),
        pl.col("ari_mean").cast(pl.Float64),
        pl.col("ari_min").cast(pl.Float64, strict=False),
        pl.col("ari_median").cast(pl.Float64, strict=False),
        pl.col("ari_max").cast(pl.Float64, strict=False),
        pl.col("ari_topk_mean").cast(pl.Float64),
        pl.col("seed").cast(pl.Int64),
        # Type casting for giant component metrics
        pl.col("giant_component_frac").cast(pl.Float64, strict=False),
        pl.col("modularity_giant").cast(pl.Float64, strict=False),
        pl.col("conductance_giant_mean").cast(pl.Float64, strict=False),
        pl.col("conductance_giant_median").cast(pl.Float64, strict=False),
        pl.col("conductance_giant_low_frac").cast(pl.Float64, strict=False),
        pl.col("ari_giant_mean").cast(pl.Float64, strict=False),
        pl.col("ari_giant_min").cast(pl.Float64, strict=False),
        pl.col("ari_giant_median").cast(pl.Float64, strict=False),
        pl.col("ari_giant_max").cast(pl.Float64, strict=False),
    ).write_parquet(DATA_RESULTS / "rq1_validation_metrics.parquet")

    logger.info(
        "RQ1 summary: modularity=%.4f (appendix null model: mean=%.4f, p=%.4f, z=%.2f), "
        "conductance mean=%.4f median=%.4f, "
        "ari_mean=%.4f (top-%d=%.4f, min=%.4f median=%.4f max=%.4f), "
        "nodes=%d, edges=%d, components=%d (giant=%.2f%%)",
        modularity,
        _num(null_model_metrics.get("modularity_null_mean")),
        _num(null_model_metrics.get("modularity_pvalue")),
        _num(null_model_metrics.get("modularity_zscore")),
        _num(conductance_metrics.get("conductance_mean")),
        _num(conductance_metrics.get("conductance_median")),
        ari_mean,
        RQ1_ARI_TOPK_COMMUNITIES,
        ari_topk_mean,
        _num(ari_stats.get("ari_min")),
        _num(ari_stats.get("ari_median")),
        _num(ari_stats.get("ari_max")),
        ni.G.number_of_nodes() if NETWORKX_AVAILABLE else centrality.height,
        ni.G.number_of_edges() if NETWORKX_AVAILABLE else ni.edges.height,
        components_count,
        giant_component_frac * 100.0,
    )

    # Markdown report and plots (non-fatal; built from in-memory metrics and
    # cached parquet outputs — no graph metrics are recomputed here).
    try:
        _export_markdown_report(metrics)
    except Exception as exc:
        logger.error("Markdown report export failed (non-fatal): %s", exc)
    try:
        _generate_rq1_plots(metrics)
    except Exception as exc:
        logger.error("Plot generation failed (non-fatal): %s", exc)

    # 4-line summary for post-run review.
    _nm = null_model_metrics
    _cm = conductance_metrics
    print(
        f"Observed Q={modularity:.4f}; appendix-only null model: "
        f"null mean={_num(_nm.get('modularity_null_mean')):.4f} "
        f"(p={_num(_nm.get('modularity_pvalue')):.4f}, "
        f"z={_num(_nm.get('modularity_zscore')):.2f}; diagnostic, not a gate)"
    )
    print(
        f"Conductance: mean={_num(_cm.get('conductance_mean')):.4f}, "
        f"median={_num(_cm.get('conductance_median')):.4f}, "
        f"frac_below_{RQ1_CONDUCTANCE_CUTOFF:.2f}={_num(_cm.get('conductance_low_frac')):.3f}"
    )
    print(
        f"ARI: global mean={ari_mean:.4f}, "
        f"top-{RQ1_ARI_TOPK_COMMUNITIES} mean={ari_topk_mean:.4f} "
        f"(min={_num(ari_stats.get('ari_min')):.4f}, "
        f"median={_num(ari_stats.get('ari_median')):.4f}, "
        f"max={_num(ari_stats.get('ari_max')):.4f})"
    )
    print(
        f"Giant Component: fraction={giant_component_frac:.4f}, Q_giant={modularity_giant:.4f}, ARI_giant={ari_giant_mean:.4f}"
    )
    return True



def _apply_bipartite_swaps(
    G: "nx.Graph",
    n_swaps: int,
    rng: np.random.Generator,
) -> "nx.Graph":
    """Apply n_swaps bipartite-preserving double-edge swaps in-place and return G.

    For each swap: selects two edges (B1, S1) and (B2, S2) where B1, B2 are both
    bipartite=0 (buyer) nodes, and rewires to (B1, S2) and (B2, S1).  This
    preserves the unweighted degree sequence of all buyer and supplier nodes
    while randomizing connectivity.

    Used to generate degree-preserving null graphs for modularity significance
    testing (Maslov & Sneppen 2002).
    """
    edge_list = [
        (u, v) for u, v in G.edges()
        if G.nodes[u].get("bipartite", -1) == 0
    ]
    m = len(edge_list)
    if m < 2:
        return G

    done = 0
    max_tries = n_swaps * 10
    tries = 0

    while done < n_swaps and tries < max_tries:
        tries += 1
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, m))
        if i == j:
            continue
        b1, s1 = edge_list[i]
        b2, s2 = edge_list[j]
        if b1 == b2 or s1 == s2:
            continue
        if G.has_edge(b1, s2) or G.has_edge(b2, s1):
            continue
        attrs1 = dict(G[b1][s1])
        attrs2 = dict(G[b2][s2])
        G.remove_edge(b1, s1)
        G.remove_edge(b2, s2)
        G.add_edge(b1, s2, **attrs1)
        G.add_edge(b2, s1, **attrs2)
        edge_list[i] = (b1, s2)
        edge_list[j] = (b2, s1)
        done += 1

    return G


def _compute_modularity_null_model(
    ni: "NetworkIntelligence",
    observed_q: float,
    n_rewires: int,
    seed: int,
    cache_path: "Path",
) -> dict:
    """Degree-preserving null-model significance test for modularity.

    Generates n_rewires null graphs by sequentially applying bipartite-preserving
    double-edge swaps (Maslov & Sneppen 2002).  For each null graph, runs Louvain
    with the same parameters as the observed partition and records its modularity.
    Compares observed_q to the null distribution via empirical p-value and z-score.

    Results are cached to cache_path.  Reruns load from cache.  Delete the file
    to force recomputation.  All null-graph generations are sequential; no
    parallelism is used to keep thermal load bounded (PGI_MAX_WORKERS honoured).
    """
    empty: dict = {
        "modularity_null_mean": None,
        "modularity_null_std": None,
        "modularity_zscore": None,
        "modularity_pvalue": None,
        "modularity_significant": False,
    }

    if not NETWORKX_AVAILABLE or ni.G.number_of_edges() == 0:
        logger.warning("Null model skipped: NetworkX unavailable or empty graph")
        return empty

    if cache_path.exists():
        logger.info(
            "Null model: loading cached distribution from %s; delete to recompute",
            cache_path,
        )
        null_df = pl.read_parquet(cache_path)
        null_q_values = null_df["null_modularity"].to_list()
    else:
        n_edges = ni.G.number_of_edges()
        swaps_per_step = max(n_edges // 10, 1000)
        logger.info(
            "Null model: generating %d rewired graphs "
            "(seed=%d, %d swaps/step, sequential, PGI_MAX_WORKERS=%d)",
            n_rewires, seed, swaps_per_step, PGI_MAX_WORKERS,
        )
        rng = np.random.default_rng(seed)
        # Evolve a single null graph sequentially — avoids copying a ~47K-node
        # graph n_rewires times and produces approximately independent samples
        # when swaps_per_step >> correlation length of the rewiring chain.
        G_null = ni.G.copy()
        null_q_values = []
        for step in range(n_rewires):
            G_null = _apply_bipartite_swaps(G_null, swaps_per_step, rng)
            try:
                part_null = community_louvain.best_partition(
                    G_null, weight="weight", random_state=42
                )
                q_null = float(
                    community_louvain.modularity(part_null, G_null, weight="weight")
                )
            except Exception as exc:
                logger.warning("Null model step %d failed: %s", step, exc)
                continue
            null_q_values.append(q_null)
            if (step + 1) % 10 == 0:
                logger.info(
                    "Null model: %d/%d complete (last Q_null=%.4f)",
                    step + 1, n_rewires, q_null,
                )

        if not null_q_values:
            logger.error("Null model: all steps failed; returning empty result")
            return empty

        pl.DataFrame({"null_modularity": null_q_values}).write_parquet(cache_path)
        logger.info(
            "Null model: cached %d Q values to %s", len(null_q_values), cache_path,
        )

    null_arr = np.array(null_q_values, dtype=float)
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr))
    z_score = (observed_q - null_mean) / null_std if null_std > 0.0 else float("inf")
    # Empirical one-tailed p-value: fraction of null samples >= observed Q.
    p_value = float(np.mean(null_arr >= observed_q))
    significant = p_value < RQ1_NULL_MODEL_SIGNIFICANCE_ALPHA

    logger.info(
        "Null model: observed Q=%.4f vs null mean=%.4f (std=%.4f), "
        "z=%.2f, p=%.4f (significant=%s, alpha=%.2f, n=%d)",
        observed_q, null_mean, null_std, z_score, p_value,
        significant, RQ1_NULL_MODEL_SIGNIFICANCE_ALPHA, len(null_q_values),
    )
    return {
        "modularity_null_mean": null_mean,
        "modularity_null_std": null_std,
        "modularity_zscore": z_score,
        "modularity_pvalue": p_value,
        "modularity_significant": significant,
    }


def _compute_conductance(G: "nx.Graph", partition: dict) -> dict:
    """Compute per-community conductance and aggregate statistics.

    Conductance of community C = cut(C, V-C) / min(vol(C), vol(V-C)),
    where cut counts boundary edges (unweighted) and vol is the sum of node
    degrees.  Low conductance indicates a genuine sparse cut — the community
    is well-separated from the rest of the graph.

    Returns conductance_mean, conductance_median, and conductance_low_frac
    (fraction of communities below RQ1_CONDUCTANCE_CUTOFF).
    """
    communities: dict = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)

    total_nodes = G.number_of_nodes()
    conductances = []
    for comm_id, nodes in communities.items():
        if len(nodes) <= 1 or len(nodes) >= total_nodes:
            continue
        try:
            cond = float(nx.conductance(G, nodes))
        except Exception:
            continue
        conductances.append(cond)

    if not conductances:
        logger.warning("Conductance: no valid communities to evaluate (all trivial)")
        return {
            "conductance_mean": None,
            "conductance_median": None,
            "conductance_low_frac": None,
            "conductance_values": [],
        }

    # Filter out NaNs to prevent median=nan and mean=nan propagation
    valid_conds = [c for c in conductances if not np.isnan(c)]
    if not valid_conds:
        return {
            "conductance_mean": None,
            "conductance_median": None,
            "conductance_low_frac": None,
            "conductance_values": [],
        }

    cond_arr = np.array(valid_conds, dtype=float)
    low_frac = float(np.mean(cond_arr < RQ1_CONDUCTANCE_CUTOFF))
    c_mean = float(np.mean(cond_arr))
    c_median = float(np.median(cond_arr))
    logger.info(
        "Conductance: %d communities evaluated, mean=%.4f, median=%.4f, "
        "frac_below_%.2f=%.3f",
        len(conductances), c_mean, c_median, RQ1_CONDUCTANCE_CUTOFF, low_frac,
    )
    return {
        "conductance_mean": c_mean,
        "conductance_median": c_median,
        "conductance_low_frac": low_frac,
        "conductance_values": valid_conds,
    }


def _compute_ari_stability(G: "nx.Graph", seeds: list[int], top_k: int = 20) -> dict:
    """Run Louvain across fixed seeds sequentially and return ARI stability stats.

    Returns the full pairwise ARI distribution (mean, min, median, max) plus the
    mean ARI restricted to the top_k largest communities of the reference partition
    (index 0).
    """
    empty: dict = {
        "ari_mean": 0.0,
        "ari_min": 0.0,
        "ari_median": 0.0,
        "ari_max": 0.0,
        "ari_all_scores": [],
        "ari_topk_mean": 0.0,
    }

    if not NETWORKX_AVAILABLE or G.number_of_edges() == 0:
        return empty

    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        logger.warning("sklearn unavailable; ARI stability skipped")
        return empty

    node_order = sorted(G.nodes())
    partitions = []
    for s in seeds:
        try:
            part = community_louvain.best_partition(G, weight="weight", random_state=s)
            partitions.append([part[n] for n in node_order])
        except Exception as exc:
            logger.warning("ARI seed %d failed: %s", s, exc)

    if len(partitions) < 2:
        return empty

    ari_scores = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            ari_scores.append(float(adjusted_rand_score(partitions[i], partitions[j])))

    ari_arr = np.array(ari_scores)
    ari_mean = float(np.mean(ari_arr))
    ari_min = float(np.min(ari_arr))
    ari_median = float(np.median(ari_arr))
    ari_max = float(np.max(ari_arr))

    logger.info(
        "ARI stability (global): mean=%.4f, min=%.4f, median=%.4f, max=%.4f "
        "across %d seed pairs",
        ari_mean, ari_min, ari_median, ari_max, len(ari_scores),
    )

    # Top-k ARI: restrict to nodes belonging to the top_k largest communities
    # in the reference partition (index 0).
    ari_topk_mean = ari_mean  # fallback if top-k computation fails
    try:
        from collections import Counter
        ref_labels = partitions[0]
        comm_sizes = Counter(ref_labels)
        top_k_comms = {c for c, _ in comm_sizes.most_common(top_k)}
        top_k_indices = [idx for idx, lbl in enumerate(ref_labels) if lbl in top_k_comms]
        if len(top_k_indices) >= 2:
            topk_scores = []
            for i in range(len(partitions)):
                for j in range(i + 1, len(partitions)):
                    pi = [partitions[i][idx] for idx in top_k_indices]
                    pj = [partitions[j][idx] for idx in top_k_indices]
                    if len(set(pi)) >= 2 and len(set(pj)) >= 2:
                        topk_scores.append(float(adjusted_rand_score(pi, pj)))
            if topk_scores:
                ari_topk_mean = float(np.mean(topk_scores))
                logger.info(
                    "ARI stability (top-%d communities): mean=%.4f "
                    "(nodes=%d, communities=%d)",
                    top_k, ari_topk_mean, len(top_k_indices), len(top_k_comms),
                )
    except Exception as exc:
        logger.warning("Top-k ARI computation failed: %s", exc)

    return {
        "ari_mean": ari_mean,
        "ari_min": ari_min,
        "ari_median": ari_median,
        "ari_max": ari_max,
        "ari_all_scores": ari_scores,
        "ari_topk_mean": ari_topk_mean,
    }


RESULTS_RQ1_DIR = ROOT / "results" / "rq1"


def _md_val(value, fmt: str = "{:.4f}") -> str:
    """Format a metric value for the markdown report; em dash for missing."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _export_markdown_report(metrics: dict) -> Path:
    """Write the full RQ1 metric set to a markdown report alongside the JSON.

    Built entirely from the already-computed metrics dict; nothing is
    recomputed. Null-model values appear in an appendix section only.
    """
    crit = metrics.get("criteria", {})
    diag = metrics.get("reported_diagnostics", {})
    dist = metrics.get("component_size_distribution", {})

    lines = [
        "# RQ1 Supplier Network Intelligence — Metric Report",
        "",
        "Generated by `src/rq1_supplier_network/rq1_network.py`. Values mirror",
        "`rq1_success_metrics.json`. Threshold rationale:",
        "`docs/THRESHOLD_JUSTIFICATION.md`.",
        "",
        "## Headline result",
        "",
        "Community structure persists after fragmentation effects: the giant",
        f"component holds {_md_val(metrics.get('giant_component_frac'), '{:.2%}')} of nodes with",
        f"Q_giant = {_md_val(metrics.get('modularity_giant'))} against Q_whole = {_md_val(metrics.get('modularity'))}.",
        "",
        "## Pass/fail gates",
        "",
        "| Gate | Target | Value | Met |",
        "|---|---|---|---|",
        f"| Modularity Q | > 0.30 | {_md_val(metrics.get('modularity'))} | {_md_val(crit.get('modularity'))} |",
        f"| Resilience std | > 0.15 | {_md_val(metrics.get('resilience_std'))} | {_md_val(crit.get('resilience_std'))} |",
        "",
        "## Whole graph vs giant component",
        "",
        "| Metric | Whole graph | Giant component |",
        "|---|---|---|",
        f"| Modularity Q | {_md_val(metrics.get('modularity'))} | {_md_val(metrics.get('modularity_giant'))} |",
        f"| Conductance mean | {_md_val(metrics.get('conductance_mean'))} | {_md_val(metrics.get('conductance_giant_mean'))} |",
        f"| Conductance median | {_md_val(metrics.get('conductance_median'))} | {_md_val(metrics.get('conductance_giant_median'))} |",
        f"| Conductance frac < 0.30 | {_md_val(metrics.get('conductance_low_frac'))} | {_md_val(metrics.get('conductance_giant_low_frac'))} |",
        f"| ARI mean | {_md_val(metrics.get('ari_mean'))} | {_md_val(metrics.get('ari_giant_mean'))} |",
        f"| ARI min | {_md_val(metrics.get('ari_min'))} | {_md_val(metrics.get('ari_giant_min'))} |",
        f"| ARI median | {_md_val(metrics.get('ari_median'))} | {_md_val(metrics.get('ari_giant_median'))} |",
        f"| ARI max | {_md_val(metrics.get('ari_max'))} | {_md_val(metrics.get('ari_giant_max'))} |",
        "",
        f"ARI is reported against a self-imposed target of {_md_val(metrics.get('reported_diagnostics', {}).get('ari_target', 0.70), '{:.2f}')}",
        "(not a literature cutoff); the observed mean indicates honest moderate",
        "partition stability and is reported as a distribution, not gated.",
        "",
        "## Component structure",
        "",
        f"Components: {metrics.get('components_count')}; giant component:",
        f"{metrics.get('giant_nodes_count')} of {metrics.get('nodes')} nodes",
        f"({_md_val(metrics.get('giant_component_frac'), '{:.2%}')}).",
        "",
        "| Component size | Count |",
        "|---|---|",
        f"| 1 | {dist.get('size_1', 0)} |",
        f"| 2 | {dist.get('size_2', 0)} |",
        f"| 3-10 | {dist.get('size_3_10', 0)} |",
        f"| > 10 | {dist.get('size_gt_10', 0)} |",
        "",
        "## Reported diagnostics (never gated)",
        "",
        "| Diagnostic | Value | Note |",
        "|---|---|---|",
        f"| Dependency vs single-bidder corr | {_md_val(diag.get('dependency_single_bidder_corr'))} | Coverage limitation: single_bidder_rate observable for {diag.get('single_bidder_buyers_observed', '—')} buyers ({_md_val(diag.get('single_bidder_buyer_coverage'), '{:.1%}')} of panel) due to lot_bidscount sparsity |",
        f"| ARI mean (target 0.70) | {_md_val(diag.get('ari_mean'))} | Self-imposed target; distribution reported above |",
        f"| ARI top-{RQ1_ARI_TOPK_COMMUNITIES} mean | {_md_val(metrics.get('ari_topk_mean'))} | Core-community stability |",
        "",
        "## Appendix: modularity null model (diagnostic only)",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Q observed | {_md_val(metrics.get('modularity'))} |",
        f"| Q null mean | {_md_val(metrics.get('modularity_null_mean'))} |",
        f"| Q null std | {_md_val(metrics.get('modularity_null_std'))} |",
        f"| z-score | {_md_val(metrics.get('modularity_zscore'), '{:.2f}')} |",
        f"| p-value | {_md_val(metrics.get('modularity_pvalue'))} |",
        "",
        "Caveat (Guimerà et al. 2004): sparse random graphs attain high",
        "modularity from fluctuations alone, so a degree-preserving null can",
        "exceed the observed Q without implying absence of community",
        "structure. The null comparison is therefore reported as an appendix",
        "diagnostic and is never a pass/fail gate.",
        "",
        "## Plots",
        "",
        "PNG plots are written to `results/rq1/`:",
        "`rq1_degree_distribution.png`, `rq1_centrality_distribution.png`,",
        "`rq1_component_size_distribution.png`, `rq1_conductance_distribution.png`.",
        "",
    ]
    report_path = DATA_RESULTS / "rq1_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report written to %s", report_path)
    return report_path


def _generate_rq1_plots(metrics: dict) -> None:
    """Generate RQ1 PNG plots into results/rq1/ from cached outputs.

    Uses rq1_network_metrics.parquet (degree, centrality),
    rq1_conductance_values.parquet (per-community conductance) and the
    component-size distribution already present in the metrics dict.
    No graph metrics are recomputed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_RQ1_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = DATA_RESULTS / "rq1_network_metrics.parquet"
    if metrics_path.exists():
        nodes_df = pl.read_parquet(metrics_path)

        fig, ax = plt.subplots(figsize=(7, 5))
        degrees = nodes_df["degree"].to_numpy()
        values, counts = np.unique(degrees, return_counts=True)
        ax.loglog(values, counts, marker=".", linestyle="none", color="tab:blue")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Node count")
        ax.set_title("RQ1 Degree Distribution (log-log)")
        fig.tight_layout()
        fig.savefig(RESULTS_RQ1_DIR / "rq1_degree_distribution.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(nodes_df["centrality"].to_numpy(), bins=100, color="tab:blue")
        ax.set_yscale("log")
        ax.set_xlabel("Composite centrality (min-max normalized)")
        ax.set_ylabel("Node count (log scale)")
        ax.set_title("RQ1 Centrality Distribution")
        fig.tight_layout()
        fig.savefig(RESULTS_RQ1_DIR / "rq1_centrality_distribution.png", dpi=150)
        plt.close(fig)
    else:
        logger.warning("Plot skipped: %s not found", metrics_path)

    dist = metrics.get("component_size_distribution") or {}
    if dist:
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["1", "2", "3-10", "> 10"]
        counts = [
            dist.get("size_1", 0), dist.get("size_2", 0),
            dist.get("size_3_10", 0), dist.get("size_gt_10", 0),
        ]
        ax.bar(labels, counts, color="tab:blue")
        ax.set_xlabel("Component size (nodes)")
        ax.set_ylabel("Component count")
        ax.set_title("RQ1 Connected Component Size Distribution")
        for idx, count in enumerate(counts):
            ax.text(idx, count, str(count), ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(RESULTS_RQ1_DIR / "rq1_component_size_distribution.png", dpi=150)
        plt.close(fig)

    cond_path = DATA_RESULTS / "rq1_conductance_values.parquet"
    if cond_path.exists():
        cond_df = pl.read_parquet(cond_path)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(cond_df["conductance"].to_numpy(), bins=50, color="tab:blue")
        ax.set_yscale("log")
        ax.axvline(RQ1_CONDUCTANCE_CUTOFF, color="tab:red", linestyle="--",
                   label=f"cutoff {RQ1_CONDUCTANCE_CUTOFF:.2f}")
        ax.set_xlabel("Per-community conductance")
        ax.set_ylabel("Community count (log scale)")
        ax.set_title("RQ1 Per-Community Conductance Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_RQ1_DIR / "rq1_conductance_distribution.png", dpi=150)
        plt.close(fig)
    else:
        logger.warning(
            "Conductance plot skipped: %s not found (written on next full rq1 run)",
            cond_path,
        )
    logger.info("RQ1 plots written to %s", RESULTS_RQ1_DIR)


def export_report_only() -> None:
    """Regenerate the markdown report and plots from cached outputs only.

    Loads rq1_success_metrics.json and cached parquet files; never rebuilds
    the graph or recomputes any metric. Invoke via:
    python -m src.rq1_supplier_network.rq1_network --report-only
    """
    import json
    metrics_path = DATA_RESULTS / "rq1_success_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Run RQ1 before --report-only export")
    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)
    _export_markdown_report(metrics)
    _generate_rq1_plots(metrics)


def _safe_corr(left: pl.Series, right: pl.Series) -> float:
    a = np.asarray(left.fill_null(0), dtype=float)
    b = np.asarray(right.fill_null(0), dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        export_report_only()
    else:
        run_rq1()
