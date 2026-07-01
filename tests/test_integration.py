import polars as pl
from src.core.config import DATA_RESULTS
from src.integration.integration import compute_unified_risk

def test_unified_risk_scores_exist():
    path = DATA_RESULTS / "unified_risk_scores.parquet"
    assert path.exists()
    df = pl.read_parquet(path)
    assert df.height > 0
    assert "unified_risk_score" in df.columns


def test_unified_score_bounds():
    df = pl.read_parquet(DATA_RESULTS / "unified_risk_scores.parquet")
    assert df["unified_risk_score"].min() >= 0
    assert df["unified_risk_score"].max() <= 1


def test_join_does_not_inflate_rows():
    rq2 = pl.DataFrame(
        {
            "contract_id": ["c1", "c2", "c3"],
            "buyer_id": ["b1", "b1", "b2"],
            "supplier_id": ["s1", "s2", "s3"],
            "cpv_division": ["45", "45", "30"],
            "award_year": [2018, 2019, 2020],
            "ensemble_risk_prob": [0.9, 0.1, 0.5],
        }
    )
    rq1 = pl.DataFrame({"buyer_id": ["b1", "b2"], "buyer_dependency_normalized": [0.8, 0.2]})
    # c1 appears 3x in RQ3 (the real-world non-uniqueness that caused the explosion).
    rq3 = pl.DataFrame(
        {
            "contract_id": ["c1", "c1", "c1", "c2"],
            "anomaly_score": [0.2, 0.9, 0.5, 0.1],
            "consensus_flag": [0, 1, 0, 0],
        }
    )
    out = compute_unified_risk(rq1, rq2, rq3)
    assert out.height == rq2.height, f"join inflated rows: {rq2.height} -> {out.height}"
    # Worst-case (max) anomaly retained for the duplicated contract.
    c1 = out.filter(pl.col("contract_id") == "c1")
    assert abs(c1["anomaly_score"][0] - 0.9) < 1e-9
    assert c1["consensus_flag"][0] == 1
