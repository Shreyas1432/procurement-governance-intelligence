import pandas as pd
import pytest

from src.core.config import DATA_FEATURES, DATA_CURATED, LABEL_DEFINING_COLS


def test_feature_artifacts_exist():
    required = [
        DATA_CURATED / "contracts_clean.parquet",
        DATA_FEATURES / "rq1_network_features.parquet",
        DATA_FEATURES / "rq2_governance_features.parquet",
        DATA_FEATURES / "rq3_price_features.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, missing


def test_rq2_features_schema():
    path = DATA_FEATURES / "rq2_governance_features.parquet"
    if not path.exists():
        pytest.skip("Run make features first")
    df = pd.read_parquet(path)
    required_cols = [
        "contract_id", "buyer_id", "supplier_id",
        "award_year", "bid_count", "procedure_type_encoded",
        "buyer_concentration_hhi", "risk_label",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    # Ensure label-defining cols are excluded (risk_label is the target, allowed)
    leaking = [c for c in LABEL_DEFINING_COLS if c in df.columns and c != "risk_label"]
    assert len(leaking) == 0, f"Label-defining cols in features: {leaking}"
    assert len(df) > 10_000, f"Expected >10K rows, got {len(df)}"
