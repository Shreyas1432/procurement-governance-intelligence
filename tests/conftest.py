import json
from pathlib import Path

import duckdb
import polars as pl
import pytest

from src.core.config import DATA_FEATURES, DATA_RESULTS, PARQUET_FILE, TRAIN_YEARS, TEST_YEARS


@pytest.fixture
def con():
    return duckdb.connect()


def read_json(path: Path):
    with open(path) as handle:
        return json.load(handle)


@pytest.fixture
def rq1_results():
    return read_json(DATA_RESULTS / "rq1_success_metrics.json")


@pytest.fixture
def rq2_results():
    return read_json(DATA_RESULTS / "rq2_success_metrics.json")


@pytest.fixture
def rq3_results():
    return read_json(DATA_RESULTS / "rq3_success_metrics.json")


@pytest.fixture
def feature_df():
    return pl.read_parquet(DATA_FEATURES / "supplier_behavior_features.parquet")


@pytest.fixture
def train_df():
    df = pl.read_parquet(DATA_FEATURES / "rq2_governance_features.parquet")
    return df.filter(pl.col("award_year").is_in(list(TRAIN_YEARS)))


@pytest.fixture
def test_df():
    df = pl.read_parquet(DATA_FEATURES / "rq2_governance_features.parquet")
    return df.filter(pl.col("award_year").is_in(list(TEST_YEARS)))
