from src.core.config import PARQUET_FILE


def test_row_count(con):
    result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET_FILE}')").fetchone()[0]
    assert result > 0


def test_temporal_coverage(con):
    result = con.execute(
        f"""
        SELECT MIN(COALESCE(tender_year, YEAR(tender_awarddecisiondate))),
               MAX(COALESCE(tender_year, YEAR(tender_awarddecisiondate)))
        FROM read_parquet('{PARQUET_FILE}')
        """
    ).fetchone()
    assert result[0] <= 2008 and result[1] >= 2020, f"Temporal range too narrow: {result}"


def test_no_negative_prices(con):
    result = con.execute(
        f"""
        SELECT COUNT(*) FROM read_parquet('{PARQUET_FILE}')
        WHERE COALESCE(tender_finalpriceUsd, bid_priceUsd, bid_digiwhist_price, tender_digiwhist_price, 0) < 0
        """
    ).fetchone()[0]
    assert result == 0, f"{result} negative prices found"
