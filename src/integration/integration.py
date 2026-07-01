import numpy as np
import polars as pl

from src.core.config import DATA_FEATURES, DATA_RESULTS
from src.common.evaluation import risk_class, safe_corr, coverage_aware_corr
from src.common.utils import get_logger, minmax, write_json


logger = get_logger("integration")


def compute_unified_risk(rq1_metrics: pl.DataFrame, rq2_predictions: pl.DataFrame, rq3_anomalies: pl.DataFrame) -> pl.DataFrame:
    rq1 = rq1_metrics.select(["buyer_id", "buyer_dependency_normalized"]).unique("buyer_id")
    rq2 = rq2_predictions.select(["contract_id", "buyer_id", "supplier_id", "cpv_division", "award_year", "ensemble_risk_prob"])
    # contract_id (persistent_id) is NOT unique in the source -- it is tender-level
    # while rows are bid/lot-level, so a contract_id can repeat thousands of times.
    # Aggregate RQ3 to one row per contract (worst-case anomaly) before joining,
    # otherwise the join multiplies rows (observed 71.8M-row explosion). See
    # tests/test_integration_no_explosion.py.
    rq3 = (
        rq3_anomalies.group_by("contract_id")
        .agg(
            pl.col("anomaly_score").max().alias("anomaly_score"),
            pl.col("consensus_flag").max().alias("consensus_flag"),
        )
    )
    merged = rq2.join(rq1, on="buyer_id", how="left").join(rq3, on="contract_id", how="left")
    if merged.height > rq2.height:
        raise ValueError(
            f"Integration join inflated rows {rq2.height} -> {merged.height}; "
            "a join key is non-unique (check rq3 aggregation / rq1 buyer uniqueness)."
        )
    dep = minmax(merged["buyer_dependency_normalized"].fill_null(0))
    gov = merged["ensemble_risk_prob"].fill_null(0).to_numpy()
    anom = merged["anomaly_score"].fill_null(0).to_numpy()
    unified = 0.35 * dep + 0.40 * gov + 0.25 * anom
    return merged.with_columns(
        pl.Series("buyer_dependency_normalized", dep),
        pl.Series("unified_risk_score", unified),
        pl.Series("risk_class", [risk_class(x) for x in unified]),
    )


def run_integration() -> bool:
    import scipy.stats as stats
    rq1_path = DATA_RESULTS / "rq1_buyer_metrics.parquet"
    rq2_path = DATA_RESULTS / "rq2_governance_predictions.parquet"
    rq3_path = DATA_RESULTS / "rq3_anomaly_flags.parquet"
    rq1_comm_path = DATA_RESULTS / "rq1_community_assignments.parquet"

    if not all(p.exists() for p in [rq1_path, rq2_path, rq3_path]):
        raise FileNotFoundError("Run rq1, rq2, and rq3 before integration")

    unified = compute_unified_risk(pl.read_parquet(rq1_path), pl.read_parquet(rq2_path), pl.read_parquet(rq3_path))
    unified.write_parquet(DATA_RESULTS / "unified_risk_scores.parquet")

    # ── Community Risk Profiling (Step 3) ──
    anova_f, anova_p = None, None
    chi2_stat, chi2_p = None, None
    num_communities_tested = 0

    if rq1_comm_path.exists():
        communities = pl.read_parquet(rq1_comm_path)
        buyer_comm = communities.filter(pl.col("node_type") == "buyer").select([
            pl.col("entity_id").alias("buyer_id"),
            pl.col("community_id")
        ]).unique("buyer_id")

        # Map Louvain community assignment onto the contracts
        unified_comm = unified.join(buyer_comm, on="buyer_id", how="left")

        # Output community-level risk summaries
        # Group by community_id and aggregate metrics
        comm_profiles = unified_comm.group_by("community_id").agg(
            pl.len().alias("contract_count"),
            pl.mean("ensemble_risk_prob").alias("mean_governance_risk"),
            (pl.col("consensus_flag").cast(pl.Float64)).mean().alias("anomaly_rate"),
            pl.mean("unified_risk_score").alias("mean_unified_risk"),
            (pl.col("risk_class") == "HIGH").sum().alias("high_risk_count")
        ).filter(pl.col("community_id").is_not_null()).sort("mean_unified_risk", descending=True)

        comm_profiles.write_csv(DATA_RESULTS / "highest_risk_communities.csv")

        # Filter communities with at least 10 contracts for statistical testing
        testable_comm = comm_profiles.filter(pl.col("contract_count") >= 10)["community_id"].to_list()
        num_communities_tested = len(testable_comm)

        if num_communities_tested >= 2:
            # 1. ANOVA on ensemble_risk_prob across valid communities
            anova_groups = []
            for cid in testable_comm:
                grp = unified_comm.filter((pl.col("community_id") == cid) & (pl.col("ensemble_risk_prob").is_not_null()))["ensemble_risk_prob"].to_list()
                if grp:
                    anova_groups.append(grp)
            if len(anova_groups) >= 2:
                anova_f, anova_p = stats.f_oneway(*anova_groups)
                anova_f = float(anova_f)
                anova_p = float(anova_p)

            # 2. Chi-Square Test of Independence on consensus_flag across valid communities
            contingency_rows = []
            for cid in testable_comm:
                c_df = unified_comm.filter(pl.col("community_id") == cid)
                flagged = int((c_df["consensus_flag"] == True).sum())
                not_flagged = int((c_df["consensus_flag"] == False).sum())
                if flagged + not_flagged >= 5:
                    contingency_rows.append([flagged, not_flagged])

            if len(contingency_rows) >= 2:
                contingency_matrix = np.array(contingency_rows)
                if contingency_matrix.sum() > 0 and contingency_matrix.shape[1] > 1:
                    try:
                        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency_matrix)
                        chi2_stat = float(chi2_stat)
                        chi2_p = float(chi2_p)
                    except Exception as e:
                        logger.warning("Chi-Square test failed to compute: %s", e)

    # NaN-aware 3-state correlations (NaN-guard bug-class fix, 2026-06-15)
    rq1_rq2 = coverage_aware_corr(unified["buyer_dependency_normalized"], unified["ensemble_risk_prob"])
    rq2_rq3 = coverage_aware_corr(unified["ensemble_risk_prob"], unified["anomaly_score"])
    
    metrics = {
        "contracts": unified.height,
        "high_risk_contracts": int((unified["risk_class"] == "HIGH").sum()),
        "mean_unified_risk": float(unified["unified_risk_score"].mean()),
        "rq1_rq2_corr": rq1_rq2["value"],
        "rq1_rq2_corr_status": rq1_rq2,
        "rq2_rq3_corr": rq2_rq3["value"],
        "rq2_rq3_corr_status": rq2_rq3,
        "anova_f_statistic": anova_f,
        "anova_p_value": anova_p,
        "chi_square_statistic": chi2_stat,
        "chi_square_p_value": chi2_p,
        "number_of_communities_tested": num_communities_tested
    }
    write_json(DATA_RESULTS / "integration_metrics.json", metrics)
    logger.info("Integration complete: %s", metrics)
    return True


if __name__ == "__main__":
    run_integration()
