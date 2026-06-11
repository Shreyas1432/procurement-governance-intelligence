import polars as pl
import pytest

from src.core.config import TRAIN_YEARS, TEST_YEARS


def test_no_future_supplier_features(feature_df):
    if feature_df.is_empty():
        pytest.skip("Feature DataFrame is empty — run 'make features' first")
    required_cols = ["supplier_feature_latest_date", "focal_award_date"]
    missing = [c for c in required_cols if c not in feature_df.columns]
    if missing:
        pytest.skip(f"Required columns missing from fixture: {missing}")
    violations = feature_df.filter(pl.col("supplier_feature_latest_date") >= pl.col("focal_award_date"))
    assert len(violations) == 0, f"{len(violations)} leakage violations found"


def test_train_test_temporal_split(train_df, test_df):
    if train_df.is_empty() or test_df.is_empty():
        pytest.skip("Train or test split is empty — run 'make features' first")
    # Boundaries derived from config, not hardcoded years
    assert train_df["award_year"].max() <= max(TRAIN_YEARS)
    assert test_df["award_year"].min() >= min(TEST_YEARS)
