import os
import polars as pl

from src.core.config import (
    DATA_CURATED,
    DATA_FEATURES,
    DATA_RESULTS,
    PARQUET_FILE,
    ROOT,
    ROW_LIMIT,
    ensure_project_dirs,
)
from src.core.data_loader import DataLoader
from src.common.utils import get_logger, timed, write_json

logger = get_logger("feature_engineering")

def run_feature_engineering(row_limit: int = ROW_LIMIT) -> bool:
    os.chdir(ROOT)  # Ensure CWD is project root so any relative refs resolve correctly
    ensure_project_dirs()
    loader = DataLoader(PARQUET_FILE, row_limit)
    outputs = {}

    # Absolute paths for parquet references inside DuckDB SQL (CWD-independent)
    bd_path = str(DATA_FEATURES / "buyer_dependency_features.parquet")
    cr_path = str(DATA_FEATURES / "competition_risk_features.parquet")
    sb_path = str(DATA_FEATURES / "supplier_behavior_features.parquet")
    cp_path = str(DATA_CURATED / "cpv_high_coverage.parquet")

    with timed(logger, "schema and data quality audit"):
        schema = loader.execute_file("00_schema_audit.sql")
        schema.write_parquet(DATA_CURATED / "schema_audit.parquet")
        quality = loader.execute_file("01_data_quality.sql", DATA_CURATED / "data_quality.parquet")
        outputs["schema_columns"] = schema.height
        outputs["quality_checks"] = quality.height

    sql_outputs = {
        "02_buyer_ecosystem.sql": DATA_CURATED / "buyer_master.parquet",
        "03_supplier_ecosystem.sql": DATA_CURATED / "supplier_master.parquet",
        "04_procedure_analysis.sql": DATA_CURATED / "procedure_analysis.parquet",
        "05_cpv_coverage.sql": DATA_CURATED / "cpv_high_coverage.parquet",
        "06_temporal_analysis.sql": DATA_CURATED / "temporal_analysis.parquet",
        "10_feature_buyer_dependency.sql": DATA_FEATURES / "buyer_dependency_features.parquet",
        "11_feature_supplier_behavior.sql": DATA_FEATURES / "supplier_behavior_features.parquet",
        "12_feature_competition_risk.sql": DATA_FEATURES / "competition_risk_features.parquet",
        "13_feature_price_proxy.sql": DATA_FEATURES / "rq3_price_features.parquet",
        "14_feature_temporal_windows.sql": DATA_FEATURES / "temporal_window_features.parquet",
    }

    for sql_file, output_path in sql_outputs.items():
        with timed(logger, f"executing {sql_file}"):
            df = loader.execute_file(sql_file, output_path)
            outputs[output_path.name] = df.height

    with timed(logger, "contracts_clean and network features creation"):
        loader.copy_query(
            """
            SELECT *
            FROM contracts_base
            WHERE buyer_id IS NOT NULL
              AND supplier_id IS NOT NULL
              AND contract_value > 0
              AND award_year BETWEEN 2006 AND 2026
            """,
            DATA_CURATED / "contracts_clean.parquet",
        )
        loader.copy_query(
            f"""
            SELECT
                c.contract_id,
                c.buyer_id,
                c.supplier_id,
                c.buyer_name,
                c.supplier_name,
                c.buyer_region,
                c.supplier_region,
                c.cpv_division,
                c.procedure_type,
                c.bid_count,
                c.contract_amendments,
                c.award_year,
                c.award_date,
                c.contract_value,
                bd.buyer_dependency_ratio,
                bd.buyer_concentration_hhi
            FROM contracts_base c
            LEFT JOIN read_parquet('{bd_path}') bd USING (buyer_id)
            WHERE c.buyer_id IS NOT NULL AND c.supplier_id IS NOT NULL AND c.contract_value > 0
            """,
            DATA_FEATURES / "rq1_network_features.parquet",
        )
        loader.copy_query(
            f"""
            -- buyer_concentration_hhi computed from HISTORICAL contracts only:
            -- a cumulative HHI as of each award_year, so a contract never sees
            -- supplier-share information from contracts awarded in the future.
            WITH supplier_shares AS (
                SELECT
                    buyer_id,
                    supplier_id,
                    award_year,
                    SUM(contract_value) OVER (
                        PARTITION BY buyer_id, supplier_id
                        ORDER BY award_year
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cumulative_sup_value,
                    SUM(contract_value) OVER (
                        PARTITION BY buyer_id
                        ORDER BY award_year
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cumulative_buyer_total
                FROM contracts_base
                WHERE buyer_id IS NOT NULL AND supplier_id IS NOT NULL AND contract_value > 0
            ),
            shares_computed AS (
                SELECT
                    buyer_id,
                    award_year,
                    POWER(cumulative_sup_value / NULLIF(cumulative_buyer_total, 0), 2) AS share_squared
                FROM supplier_shares
            ),
            hist_hhi AS (
                SELECT buyer_id, award_year, SUM(share_squared) AS hhi_historical
                FROM shares_computed
                GROUP BY buyer_id, award_year
            )
            SELECT
                c.contract_id,
                c.buyer_id,
                c.supplier_id,
                c.cpv_division,
                c.procedure_type,
                cr.procedure_risk,
                cr.competition_risk,
                c.bid_count,
                c.contract_amendments,
                c.award_year,
                c.contract_value,
                COALESCE(bd.buyer_dependency_ratio, 0) AS buyer_dependency_ratio,
                COALESCE(hh.hhi_historical, 0) AS buyer_concentration_hhi,
                COALESCE(sb.single_bid_rate, 0) AS single_bid_rate,
                COALESCE(sb.cpv_diversity, 0) AS supplier_cpv_diversity,
                COALESCE(cp.record_count, 0) AS cpv_record_count,
                CASE WHEN cp.high_coverage THEN 0.2 ELSE 0.7 END AS cpv_risk_score,
                DENSE_RANK() OVER (ORDER BY c.procedure_type) - 1 AS procedure_type_encoded,
                DENSE_RANK() OVER (ORDER BY c.procedure_type) - 1 AS proc_enc,
                DENSE_RANK() OVER (ORDER BY c.cpv_division) - 1 AS cpv_enc,
                CASE WHEN c.bid_count IS NULL THEN NULL
                     ELSE LEAST(GREATEST(c.bid_count, 1), 50) END AS bid_count_clipped,
                LN(GREATEST(c.contract_value, 1)) AS contract_value_log,
                CASE
                    WHEN c.procedure_type IN ('NEGOTIATED_WITHOUT_PUBLICATION', 'OUTRIGHT_AWARD')
                     AND c.bid_count IS NOT NULL AND c.bid_count <= 1
                     AND COALESCE(bd.buyer_dependency_ratio, 0) > 0.7
                     AND c.contract_amendments > 2 THEN 2
                    WHEN c.procedure_type ILIKE '%NEGOTIATED%'
                      OR c.procedure_type = 'RESTRICTED'
                      OR (c.bid_count IS NOT NULL AND c.bid_count <= 1)
                      OR COALESCE(bd.buyer_dependency_ratio, 0) > 0.5
                      OR c.contract_amendments BETWEEN 1 AND 2 THEN 1
                    ELSE 0
                END AS risk_label
            FROM contracts_base c
            LEFT JOIN read_parquet('{bd_path}') bd USING (buyer_id)
            LEFT JOIN hist_hhi hh ON c.buyer_id = hh.buyer_id AND c.award_year = hh.award_year
            LEFT JOIN read_parquet('{cr_path}') cr USING (contract_id)
            LEFT JOIN read_parquet('{sb_path}') sb
              ON c.contract_id = sb.focal_contract
            LEFT JOIN read_parquet('{cp_path}') cp
              ON c.cpv_division = cp.cpv_division
            WHERE c.buyer_id IS NOT NULL
              AND c.supplier_id IS NOT NULL
              AND c.contract_value > 0
            """,
            DATA_FEATURES / "rq2_governance_features.parquet",
        )
        rq2 = pl.read_parquet(DATA_FEATURES / "rq2_governance_features.parquet")
        labels = rq2.select(["contract_id", "risk_label"]).head(min(1000, rq2.height))
        labels.write_csv(DATA_FEATURES / "rq2_labels.csv")

    write_json(DATA_RESULTS / "eda_feature_summary.json", outputs)
    logger.info("Feature engineering completed successfully")
    return True

if __name__ == "__main__":
    run_feature_engineering()
