"""Guards against RQ2 target leakage.

`risk_label` is computed deterministically from procedure_type/procedure_risk,
bid_count, buyer_dependency_ratio and contract_amendments (see
src/feature_engineering.py and sql/12_feature_competition_risk.sql). Feeding any
of those columns to the classifier makes the target a function of its own inputs
and yields a degenerate F1~0.999 / AUC~1.000. These tests fail loudly if that
ever regresses.
"""
import polars as pl

from src.core.config import DATA_FEATURES, RQ2_LABEL_INPUTS, RQ2_MODEL_FEATURES
from src.rq2_governance_risk.rq2_governance import GovernanceRiskClassifier


def test_model_features_exclude_label_inputs():
    leaked = set(RQ2_MODEL_FEATURES) & set(RQ2_LABEL_INPUTS)
    assert not leaked, f"Label-defining columns used as predictors (leakage): {sorted(leaked)}"


def test_classifier_uses_leakage_free_features():
    leaked = set(GovernanceRiskClassifier.FEATURES) & set(RQ2_LABEL_INPUTS)
    assert not leaked, f"Classifier trains on label inputs (leakage): {sorted(leaked)}"


def test_model_features_present_in_dataset():
    path = DATA_FEATURES / "rq2_governance_features.parquet"
    if not path.exists():
        return
    cols = set(pl.read_parquet(path).columns)
    # supplier_centrality is a network metric joined from RQ1 output at runtime
    # (see src.rq2_governance._attach_supplier_centrality), not produced by the
    # feature SQL, so it is legitimately absent from this parquet.
    runtime_joined = {"supplier_centrality"}
    missing = [f for f in RQ2_MODEL_FEATURES if f not in cols and f not in runtime_joined]
    assert not missing, f"Model features absent from feature parquet: {missing}"
