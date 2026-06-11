from pathlib import Path
import duckdb
import polars as pl

from src.core.config import (
    PARQUET_FILE,
    ROW_LIMIT,
    SQL_DIR,
    STRATA_PER_YEAR_CAP,
    STRATA_YEAR_MAX,
    STRATA_YEAR_MIN,
    STRATIFY_BY_YEAR,
)

_YEAR_EXPR = "CAST(COALESCE(tender_year, YEAR(tender_awarddecisiondate), 0) AS INTEGER)"

class DataLoader:
    def __init__(
        self,
        parquet_path: Path = PARQUET_FILE,
        row_limit: int = ROW_LIMIT,
        stratify_by_year: bool = STRATIFY_BY_YEAR,
    ):
        self.parquet_path = Path(parquet_path)
        self.row_limit = row_limit
        self.stratify_by_year = stratify_by_year
        self.con = duckdb.connect()
        self._register_views()

    def _register_views(self) -> None:
        # Escape single quotes in the parquet path before interpolating into
        # DuckDB read_parquet(...) to prevent SQL injection (MED-02).
        safe_path = str(self.parquet_path).replace("'", "''")
        if self.stratify_by_year:
            # Stratify by year to establish representative temporal training panel.
            self.con.execute(
                f"""
                CREATE OR REPLACE VIEW raw_contracts AS
                WITH ranked AS (
                    SELECT *,
                           {_YEAR_EXPR} AS _strata_year,
                           ROW_NUMBER() OVER (
                               PARTITION BY {_YEAR_EXPR}
                               ORDER BY hash(persistent_id)
                           ) AS _strata_rn
                    FROM read_parquet('{safe_path}')
                )
                SELECT * EXCLUDE (_strata_year, _strata_rn)
                FROM ranked
                WHERE _strata_year BETWEEN {STRATA_YEAR_MIN} AND {STRATA_YEAR_MAX}
                  AND _strata_rn <= {STRATA_PER_YEAR_CAP}
                """
            )
        else:
            limit_sql = f"LIMIT {self.row_limit}" if self.row_limit and self.row_limit > 0 else ""
            self.con.execute(
                f"""
                CREATE OR REPLACE VIEW raw_contracts AS
                SELECT *
                FROM read_parquet('{safe_path}')
                {limit_sql}
                """
            )
        self.con.execute(
            """
            CREATE OR REPLACE VIEW contracts_base AS
            SELECT
                persistent_id AS contract_id,
                persistent_id,
                tender_id,
                buyer_id,
                bidder_id AS supplier_id,
                COALESCE(buyer_name, buyer_id, 'UNKNOWN') AS buyer_name,
                COALESCE(bidder_name, bidder_id, 'UNKNOWN') AS supplier_name,
                COALESCE(buyer_nuts_2, buyer_nuts_1, buyer_country, 'UNKNOWN') AS buyer_region,
                COALESCE(bidder_nuts_2, bidder_nuts_1, bidder_country, 'UNKNOWN') AS supplier_region,
                COALESCE(NULLIF(REGEXP_EXTRACT(tender_cpvs, '([0-9]{2})', 1), ''), 'UNKNOWN') AS cpv_division,
                COALESCE(tender_proceduretype, 'UNKNOWN') AS procedure_type,
                -- lot_bidscount is the count of competing bids per lot (the correct competition
                -- measure). tender_recordedbidscount records the number of formal tender-package
                -- submissions, which is structurally 1 for virtually every contract in the Italian
                -- ANAC dataset and does not reflect actual bidder competition.  Using it as the
                -- primary bid-count field inflates single_bidder_rate to ~91%.  We use
                -- lot_bidscount only; rows where it is absent receive NULL (bid count unknown)
                -- and are treated as missing — not as single-bidder — throughout the pipeline.
                CAST(lot_bidscount AS DOUBLE) AS bid_count,
                CAST(COALESCE(tender_corrections_count, 0) AS DOUBLE) AS contract_amendments,
                CAST(COALESCE(tender_year, YEAR(tender_awarddecisiondate), 0) AS INTEGER) AS award_year,
                tender_awarddecisiondate AS award_date,
                CAST(COALESCE(tender_estimatedpriceUsd, tender_digiwhist_price, 0) AS DOUBLE) AS estimated_price,
                -- contract_value and final_price are intentionally the same awarded
                -- final price: contract_value is the canonical name used by network/HHI
                -- SQL, final_price the alias used by pricing/EDA SQL. estimated_price
                -- (above) is the separate baseline used for RQ3 anomaly residuals.
                CAST(COALESCE(tender_finalpriceUsd, bid_priceUsd, bid_digiwhist_price, tender_digiwhist_price, 0) AS DOUBLE) AS contract_value,
                CAST(COALESCE(tender_finalpriceUsd, bid_priceUsd, bid_digiwhist_price, tender_digiwhist_price, 0) AS DOUBLE) AS final_price
            FROM raw_contracts
            """
        )

    def execute_file(self, sql_file: str, output_path: Path | None = None) -> pl.DataFrame:
        query = (SQL_DIR / sql_file).read_text()
        df = self.con.execute(query).pl()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(output_path)
        return df

    def copy_query(self, query: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    def pl(self, query: str) -> pl.DataFrame:
        return self.con.execute(query).pl()
