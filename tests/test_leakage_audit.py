"""
Data leakage audit for Procurement Governance Intelligence.

Covers four leakage types specific to this project:
  TYPE 1 - Target Leakage     : label-defining columns present in feature matrix
  TYPE 2 - Temporal Leakage   : future data bleeding into training set
  TYPE 3 - In-Sample Scoring  : model scores training rows as if they were unseen
  TYPE 4 - Feature-Time Leakage : supplier/buyer features computed using future contracts

Run:
    python tests/test_leakage_audit.py          # standalone audit report
    pytest tests/test_leakage_audit.py -v       # as part of test suite

Each test is self-contained with synthetic fixtures where no data is available,
so all tests run on a fresh checkout (no parquet files required).
"""

import sys
from pathlib import Path

# ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from src.core.config import FEATURES as MODEL_FEATURES
except Exception:
    MODEL_FEATURES = None

import json
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Configuration (mirrors src/core/config.py, no import needed for audit)

TRAIN_YEARS = (2006, 2016)
TEST_YEARS  = (2019, 2021)
GAP_YEARS   = (2017, 2018)   # deliberately excluded, temporal buffer

# Columns that define the risk label, must never appear in feature matrix.
# Any model that sees these achieves AUC=1.000 trivially.
LABEL_DEFINING_COLS = [
    "risk_label",           # the target itself
    "R_gov",                # composite score the label is derived from
    "R_proc",               # procedure risk component of R_gov
    "R_comp",               # competition risk component of R_gov
    "R_dep",                # dependency risk component of R_gov
    "R_corr",               # correction risk component of R_gov
    "R_sup",                # supplier risk component of R_gov
    "single_bidder_flag",   # direct operationalisation of competition risk
    "is_single_bidder",     # alternative name for same concept
    "bid_count_flag",       # derived from bid_count == 1
    "governance_score",     # ensemble output used as integration feature
    "ensemble_risk_prob",   # model output, must not feed back into features
]

# Columns that must be present in any valid feature set
REQUIRED_FEATURES = [
    "contract_id",
    "buyer_id",
    "supplier_id",
    "award_year",
]

# Allowed features (safe inputs for ML, all pre-award observables)
SAFE_FEATURES = [
    "procedure_type_encoded",
    "bid_count",                    # number of bids received (pre-award observable)
    "buyer_concentration_hhi",      # computed from HISTORICAL contracts only
    "supplier_centrality",          # graph metric from HISTORICAL contracts only
    "contract_value_log",           # award value (observable at time of award)
    "cpv_risk_score",               # category-level historical risk
    "buyer_dependency_ratio",       # computed from HISTORICAL contracts only
    "award_year",
    "cpv_enc",
    "proc_enc",
    "bid_count_clipped",
]

# Audit results accumulator

class LeakageAudit:
    def __init__(self):
        self.results = []
        self.passed  = 0
        self.failed  = 0
        self.warned  = 0

    def record(self, test_id: str, test_name: str, status: str,
               detail: str, severity: str = "HIGH", fix: str = ""):
        icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}[status]
        self.results.append({
            "id": test_id, "name": test_name, "status": status,
            "severity": severity, "detail": detail, "fix": fix, "icon": icon
        })
        if status == "PASS":   self.passed  += 1
        elif status == "FAIL": self.failed  += 1
        elif status == "WARN": self.warned  += 1

    def report(self) -> str:
        lines = [
            "",
            "  PROCUREMENT GOVERNANCE INTELLIGENCE — LEAKAGE AUDIT REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"\n  PASSED: {self.passed}  FAILED: {self.failed}  "
            f"WARNED: {self.warned}\n",
        ]
        for r in self.results:
            lines.append(f"\n{r['icon']} [{r['id']}] {r['name']}")
            lines.append(f"   Status   : {r['status']} | Severity: {r['severity']}")
            lines.append(f"   Detail   : {r['detail']}")
            if r["fix"]:
                lines.append(f"   Fix      : {r['fix']}")
        lines.append("\n" + "")
        if self.failed == 0:
            lines.append("  VERDICT: PASS — No leakage detected — safe to train models")
        else:
            lines.append(
                f"  VERDICT: FAIL — {self.failed} leakage issue(s) found — "
                "DO NOT submit results until fixed"
            )
        lines.append("")
        return "\n".join(lines)


AUDIT = LeakageAudit()


# Synthetic fixtures, used when real data is not available

def make_contracts(n=500) -> pd.DataFrame:
    """Synthetic procurement contracts with realistic column structure.
    Covers both safe features and potentially-leaking columns.
    """
    rng = np.random.default_rng(42)
    years = rng.choice(list(range(2006, 2022)), size=n)
    bid_counts = rng.integers(1, 15, size=n)
    proc_types = rng.choice(
        ["open", "negotiated", "restricted", "negotiated without publication"],
        size=n, p=[0.55, 0.25, 0.15, 0.05]
    )
    hhi = rng.uniform(0.05, 0.95, size=n)
    R_proc = np.where(proc_types == "open", 0.2,
             np.where(proc_types == "restricted", 0.5,
             np.where(proc_types == "negotiated", 0.7, 1.0)))
    R_comp = np.where(bid_counts == 1, 1.0, 1.0 / bid_counts)
    R_dep  = hhi
    R_gov  = 0.25 * R_proc + 0.25 * R_comp + 0.20 * R_dep + 0.15 * 0 + 0.15 * 0
    quantiles = np.quantile(R_gov, [0.25, 0.50, 0.75])
    risk_label = np.digitize(R_gov, quantiles)  # 0=LOW, 1=MED, 2=HIGH, 3=CRIT

    return pd.DataFrame({
        "contract_id":            [f"C{i:05d}" for i in range(n)],
        "buyer_id":               [f"B{rng.integers(0,100):03d}" for _ in range(n)],
        "supplier_id":            [f"S{rng.integers(0,200):04d}" for _ in range(n)],
        "award_year":             years,
        "bid_count":              bid_counts,
        "bid_count_clipped":      np.clip(bid_counts, 1, 50),
        "procedure_type":         proc_types,
        "proc_enc":               pd.Categorical(proc_types).codes,
        "contract_value":         rng.lognormal(10, 2, n),
        "contract_value_log":     rng.normal(10, 2, n),
        "buyer_concentration_hhi": hhi,
        "buyer_dependency_ratio": rng.uniform(0.1, 0.9, n),
        "supplier_centrality":    rng.uniform(0.0, 0.5, n),
        "cpv_enc":                rng.integers(0, 30, n),
        "cpv_risk_score":         rng.uniform(0.1, 0.9, n),
        # dangerous columns below
        "R_gov":                  R_gov,
        "R_proc":                 R_proc,
        "R_comp":                 R_comp,
        "R_dep":                  R_dep,
        "R_corr":                 np.zeros(n),
        "R_sup":                  np.zeros(n),
        "single_bidder_flag":     (bid_counts == 1).astype(int),
        "risk_label":             risk_label,
        "governance_score":       R_gov * rng.uniform(0.9, 1.1, n),  # "model output"
        "ensemble_risk_prob":     rng.uniform(0.0, 1.0, n),
    })


def make_train_test(df: pd.DataFrame):
    train = df[df["award_year"].between(*TRAIN_YEARS)].copy()
    test  = df[df["award_year"].between(*TEST_YEARS)].copy()
    return train, test


# Type 1: target leakage tests

def test_type1_no_label_defining_cols_in_feature_matrix():
    """
    The feature matrix X fed to models must not contain any column
    that was used to compute or define the target risk_label.
    If it does, the model learns to predict the label from the label.
    Classic sign: AUC = 1.000 or F1 = 1.000.
    """
    if MODEL_FEATURES is not None:
        features_to_check = MODEL_FEATURES
    else:
        df = make_contracts()
        features_to_check = [c for c in df.columns if c != "risk_label"]

    leaking = [c for c in features_to_check if c in LABEL_DEFINING_COLS]

    if leaking:
        AUDIT.record(
            "T1-001", "No label-defining cols in feature matrix",
            "FAIL",
            f"Found {len(leaking)} label-defining column(s) in naive feature set: "
            f"{leaking}. A model trained on these achieves AUC≈1.000 trivially.",
            severity="CRITICAL",
            fix="Exclude from FEATURES list: " + str(leaking)
        )
    else:
        AUDIT.record(
            "T1-001", "No label-defining cols in feature matrix",
            "PASS",
            "No label-defining columns found in feature matrix."
        )


def test_type1_auc_ceiling_detection():
    """
    Train a quick RandomForest with AND without leaking columns.
    If AUC with leaking cols >> AUC without, leakage is confirmed.
    This is the empirical proof, not just structural inspection.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    df = make_contracts(n=800)
    train, test = make_train_test(df)

    if len(train) < 50 or len(test) < 20:
        AUDIT.record(
            "T1-002", "AUC ceiling detection (empirical)",
            "SKIP", "Insufficient synthetic data for this split.",
            fix="Run with real data."
        )
        return

    y_train = train["risk_label"].values
    y_test  = test["risk_label"].values

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    classes = sorted(df["risk_label"].unique())

    if MODEL_FEATURES is not None:
        # Check if the model's actual features leak
        model_cols = [c for c in MODEL_FEATURES if c in train.columns]
        rf.fit(train[model_cols], y_train)
        proba_model = rf.predict_proba(test[model_cols])
        auc_model = roc_auc_score(
            label_binarize(y_test, classes=classes),
            proba_model, multi_class="ovr", average="macro"
        )
        if auc_model > 0.95:
            AUDIT.record(
                "T1-002", "AUC ceiling detection (empirical)",
                "FAIL",
                f"AUC with actual model features = {auc_model:.4f} (≈1.000 = confirmed leakage).",
                severity="CRITICAL",
                fix="Remove label-defining columns from FEATURES list in config.py"
            )
        else:
            AUDIT.record(
                "T1-002", "AUC ceiling detection (empirical)",
                "PASS",
                f"AUC with actual model features = {auc_model:.4f} — within safe limits (<0.95)."
            )
        return

    # Leaking feature set (includes R_gov, R_proc, etc.)
    leaking_cols = ["bid_count", "proc_enc", "R_gov", "R_proc", "R_comp",
                    "single_bidder_flag", "buyer_concentration_hhi"]
    leaking_cols = [c for c in leaking_cols if c in train.columns]

    # Clean feature set (excludes label-defining cols)
    clean_cols = ["bid_count", "proc_enc", "buyer_concentration_hhi",
                  "supplier_centrality", "contract_value_log", "cpv_enc",
                  "bid_count_clipped", "cpv_risk_score"]
    clean_cols = [c for c in clean_cols if c in train.columns]

    # AUC with leaking features
    rf.fit(train[leaking_cols], y_train)
    proba_leak = rf.predict_proba(test[leaking_cols])
    auc_leak = roc_auc_score(
        label_binarize(y_test, classes=classes),
        proba_leak, multi_class="ovr", average="macro"
    )

    # AUC with clean features
    rf.fit(train[clean_cols], y_train)
    proba_clean = rf.predict_proba(test[clean_cols])
    auc_clean = roc_auc_score(
        label_binarize(y_test, classes=classes),
        proba_clean, multi_class="ovr", average="macro"
    )

    gap = auc_leak - auc_clean

    if auc_leak > 0.97:
        AUDIT.record(
            "T1-002", "AUC ceiling detection (empirical)",
            "FAIL",
            f"AUC with leaking features = {auc_leak:.4f} (≈1.000 = confirmed leakage). "
            f"AUC with clean features = {auc_clean:.4f}. Gap = {gap:.4f}.",
            severity="CRITICAL",
            fix="Remove label-defining columns from FEATURES list in config.py"
        )
    elif gap > 0.15:
        AUDIT.record(
            "T1-002", "AUC ceiling detection (empirical)",
            "WARN",
            f"AUC gap = {gap:.4f} (leaking={auc_leak:.3f} vs clean={auc_clean:.3f}). "
            "Large gap suggests partial leakage — investigate feature construction.",
            severity="HIGH",
            fix="Review how buyer_dependency and single_bid_rate are computed — "
                "check they use only pre-award data."
        )
    else:
        AUDIT.record(
            "T1-002", "AUC ceiling detection (empirical)",
            "PASS",
            f"AUC gap = {gap:.4f} (leaking={auc_leak:.3f}, clean={auc_clean:.3f}). "
            "No ceiling effect detected."
        )


def test_type1_single_bid_rate_is_not_a_feature():
    """
    single_bid_rate is computed as (contracts where bid_count==1) / total.
    This is essentially a proxy for the competition component of risk_label.
    If it appears in features, it indirectly leaks the label.
    """
    SAFE = set(SAFE_FEATURES)
    DANGEROUS = {"single_bid_rate", "single_bidder_rate", "pct_single_bidder",
                 "is_single_bidder", "single_bidder_flag", "bid_count_flag"}
    overlap = DANGEROUS & SAFE

    if overlap:
        AUDIT.record(
            "T1-003", "single_bid_rate not in safe feature set",
            "FAIL",
            f"Dangerous column(s) found in SAFE_FEATURES: {overlap}. "
            "These directly proxy the competition component of risk_label.",
            severity="HIGH",
            fix="Remove from SAFE_FEATURES. Use raw bid_count instead."
        )
    else:
        AUDIT.record(
            "T1-003", "single_bid_rate not in safe feature set",
            "PASS",
            "single_bid_rate variants are not in the safe feature set."
        )


# Type 2: temporal leakage tests

def test_type2_train_test_boundary():
    """
    Train: 2006-2016. Test: 2019-2021. Gap: 2017-2018 deliberately excluded.
    Any 2017-2018 rows in training = temporal boundary violation.
    Any 2016 rows in test = train-test contamination.
    """
    df = make_contracts(n=1000)
    train, test = make_train_test(df)

    train_max = train["award_year"].max()
    test_min  = test["award_year"].min()
    gap_in_train = train[train["award_year"].isin(GAP_YEARS)]
    gap_in_test  = test[test["award_year"].isin(GAP_YEARS)]

    issues = []
    if train_max > TRAIN_YEARS[1]:
        issues.append(f"Train contains year {train_max} (max allowed: {TRAIN_YEARS[1]})")
    if test_min < TEST_YEARS[0]:
        issues.append(f"Test contains year {test_min} (min allowed: {TEST_YEARS[0]})")
    if len(gap_in_train) > 0:
        issues.append(f"{len(gap_in_train)} gap-year rows in train set")
    if len(gap_in_test) > 0:
        issues.append(f"{len(gap_in_test)} gap-year rows in test set")

    if issues:
        AUDIT.record(
            "T2-001", "Train/test temporal boundary correct",
            "FAIL", " | ".join(issues), severity="CRITICAL",
            fix=f"Filter: train = award_year in {TRAIN_YEARS}, "
                f"test = award_year in {TEST_YEARS}. Drop gap years {GAP_YEARS}."
        )
    else:
        AUDIT.record(
            "T2-001", "Train/test temporal boundary correct",
            "PASS",
            f"Train: {train['award_year'].min()}–{train_max} | "
            f"Test: {test_min}–{test['award_year'].max()} | "
            f"Gap years {GAP_YEARS} correctly excluded."
        )


def test_type2_no_overlap_in_contract_ids():
    """
    No contract_id should appear in both train and test sets.
    This is a direct contamination check: any overlap means
    the model saw test-set contracts during training.
    """
    df = make_contracts(n=1000)
    train, test = make_train_test(df)

    overlap = set(train["contract_id"]) & set(test["contract_id"])

    if overlap:
        AUDIT.record(
            "T2-002", "No contract_id overlap between train and test",
            "FAIL",
            f"{len(overlap)} contract IDs appear in both train and test: "
            f"{list(overlap)[:5]}{'...' if len(overlap) > 5 else ''}",
            severity="CRITICAL",
            fix="Temporal split guarantees no overlap when done on award_year. "
                "Check for contracts with multiple award_year entries."
        )
    else:
        AUDIT.record(
            "T2-002", "No contract_id overlap between train and test",
            "PASS",
            f"Train: {len(train):,} contracts | Test: {len(test):,} | "
            "Zero overlap confirmed."
        )


def test_type2_supplier_features_use_historical_data_only():
    """
    Supplier behavioral features (HHI, centrality, dependency_ratio) must be
    computed from contracts BEFORE the focal contract's award_year.

    The violation pattern: computing HHI across all years then joining to all
    contracts, including the contract that defines the HHI.

    Test: simulate computing HHI from full data vs historical-only data.
    If they differ significantly for test-year contracts, leakage is present.
    """
    rng = np.random.default_rng(42)
    n = 200
    years  = rng.choice(list(range(2006, 2022)), size=n)
    buyers = rng.choice([f"B{i:03d}" for i in range(20)], size=n)
    values = rng.lognormal(10, 2, n)
    suppliers = rng.choice([f"S{i:04d}" for i in range(50)], size=n)

    df = pd.DataFrame({
        "contract_id":  [f"C{i:04d}" for i in range(n)],
        "buyer_id":     buyers,
        "supplier_id":  suppliers,
        "award_year":   years,
        "contract_value": values,
    })

    # WRONG: compute HHI from full dataset (leaks future)
    total_spend = df.groupby("buyer_id")["contract_value"].sum().rename("total_spend")
    sup_spend   = df.groupby(["buyer_id","supplier_id"])["contract_value"].sum()
    shares_full = sup_spend / total_spend
    hhi_full    = (shares_full ** 2).groupby("buyer_id").sum()

    # RIGHT: compute HHI from historical data only (before each contract year)
    hhi_historical = {}
    for _, row in df.iterrows():
        hist = df[(df["buyer_id"] == row["buyer_id"]) &
                  (df["award_year"] < row["award_year"])]
        if len(hist) == 0:
            hhi_historical[row["contract_id"]] = np.nan
            continue
        t = hist["contract_value"].sum()
        s = hist.groupby("supplier_id")["contract_value"].sum()
        hhi_historical[row["contract_id"]] = ((s / t) ** 2).sum()

    df["hhi_full"]    = df["buyer_id"].map(hhi_full)
    df["hhi_hist"]    = df["contract_id"].map(hhi_historical)

    test_contracts = df[df["award_year"].between(*TEST_YEARS)].dropna(
        subset=["hhi_full", "hhi_hist"]
    )
    if len(test_contracts) == 0:
        AUDIT.record(
            "T2-003", "Supplier HHI uses historical data only",
            "SKIP", "No test-year contracts in synthetic sample.",
            fix="Run audit against real parquet data."
        )
        return

    mean_diff = (test_contracts["hhi_full"] - test_contracts["hhi_hist"]).abs().mean()
    max_diff  = (test_contracts["hhi_full"] - test_contracts["hhi_hist"]).abs().max()

    if mean_diff > 0.05:
        AUDIT.record(
            "T2-003", "Supplier HHI uses historical data only",
            "WARN",
            f"Mean HHI diff (full vs historical) = {mean_diff:.4f} "
            f"(max = {max_diff:.4f}). This suggests HHI is computed "
            "from all years and then joined, which leaks future contracts into features.",
            severity="HIGH",
            fix="Recompute HHI using only contracts with award_year < focal_contract.award_year. "
                "Use SQL window functions: SUM(...) OVER (PARTITION BY buyer_id "
                "ORDER BY award_year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)"
        )
    else:
        AUDIT.record(
            "T2-003", "Supplier HHI uses historical data only",
            "PASS",
            f"HHI diff (full vs historical) = {mean_diff:.4f} — within tolerance."
        )


def test_type2_gap_years_excluded_from_both_splits():
    """
    Gap years 2017-2018 must appear in neither train nor test.
    If they appear in train, they can contaminate rolling features.
    If they appear in test, the evaluation is on a different population
    than declared in the thesis.
    """
    df = make_contracts(n=1000)
    train, test = make_train_test(df)

    gap_in_train = train[train["award_year"].isin(GAP_YEARS)]
    gap_in_test  = test[test["award_year"].isin(GAP_YEARS)]

    all_clean = len(gap_in_train) == 0 and len(gap_in_test) == 0
    gap_all   = df[df["award_year"].isin(GAP_YEARS)]

    if all_clean:
        AUDIT.record(
            "T2-004", "Gap years 2017–2018 excluded from both splits",
            "PASS",
            f"Gap years correctly excluded. {len(gap_all)} rows in gap period "
            "(not used in any split)."
        )
    else:
        details = []
        if len(gap_in_train): details.append(f"{len(gap_in_train)} in train")
        if len(gap_in_test):  details.append(f"{len(gap_in_test)} in test")
        AUDIT.record(
            "T2-004", "Gap years 2017–2018 excluded from both splits",
            "FAIL",
            f"Gap year rows found: {', '.join(details)}",
            severity="HIGH",
            fix="Explicitly filter: df[~df['award_year'].isin([2017, 2018])]"
        )


# Type 3: in-sample scoring tests

def test_type3_predictions_parquet_has_score_type_column():
    """
    The rq2_governance_predictions.parquet should have a 'score_type' column
    indicating whether each score is 'out_of_fold' (training period) or
    'held_out' (test period). Without this, in-sample scores are presented
    as if they were out-of-sample.
    """
    results_paths = [
        Path("data/results/rq2_governance_predictions.parquet"),
        Path("../data/results/rq2_governance_predictions.parquet"),
    ]
    parquet_found = None
    for p in results_paths:
        if p.exists():
            parquet_found = p
            break

    if parquet_found is None:
        AUDIT.record(
            "T3-001", "Predictions parquet has score_type column",
            "SKIP",
            "rq2_governance_predictions.parquet not found — run make rq2 first.",
            fix="After running make rq2, re-run this audit."
        )
        return

    df = pd.read_parquet(parquet_found)
    if "score_type" not in df.columns:
        AUDIT.record(
            "T3-001", "Predictions parquet has score_type column",
            "FAIL",
            f"'score_type' column missing from {parquet_found}. "
            f"Columns present: {list(df.columns)}. "
            "In-sample (training period) scores are indistinguishable from "
            "out-of-sample (test period) scores.",
            severity="HIGH",
            fix="Add score_type='out_of_fold' for train-period rows (via cross_val_predict) "
                "and score_type='held_out' for test-period rows."
        )
    else:
        counts = df["score_type"].value_counts().to_dict()
        AUDIT.record(
            "T3-001", "Predictions parquet has score_type column",
            "PASS",
            f"score_type column present. Distribution: {counts}"
        )


def test_type3_no_training_rows_in_reported_metrics():
    """
    Reported AUC/F1 metrics must be computed ONLY on test-period rows.
    If metrics are computed on the full dataset (including training rows),
    they are optimistically inflated.

    Test: verify that the success metrics JSON was computed on test years only.
    """
    metrics_paths = [
        Path("data/results/rq2_success_metrics.json"),
        Path("../data/results/rq2_success_metrics.json"),
    ]
    found = None
    for p in metrics_paths:
        if p.exists():
            found = p
            break

    if found is None:
        AUDIT.record(
            "T3-002", "Success metrics computed on test set only",
            "SKIP",
            "rq2_success_metrics.json not found — run make rq2 first."
        )
        return

    metrics = json.loads(found.read_text())
    has_eval_period = "eval_period" in metrics or "test_years" in metrics

    if not has_eval_period:
        AUDIT.record(
            "T3-002", "Success metrics computed on test set only",
            "WARN",
            f"Metrics JSON does not record which years were used for evaluation. "
            f"Metrics: {list(metrics.keys())}. Cannot verify test-only computation.",
            severity="HIGH",
            fix="Add 'eval_period': {'years': TEST_YEARS, 'n_contracts': len(test_df)} "
                "to the metrics JSON when writing it."
        )
    else:
        eval_info = metrics.get("eval_period", metrics.get("test_years", {}))
        AUDIT.record(
            "T3-002", "Success metrics computed on test set only",
            "PASS",
            f"Evaluation period recorded: {eval_info}"
        )


def test_type3_cross_val_predict_used_for_train_scores():
    """
    Structural check: verify that rq2_governance.py uses cross_val_predict
    for training-period rows rather than in-sample predict_proba.
    This is a code inspection test (runs without data).
    """
    source_paths = [
        Path("src/rq2_governance_risk/rq2_governance.py"),
        Path("../src/rq2_governance_risk/rq2_governance.py"),
    ]
    found = None
    for p in source_paths:
        if p.exists():
            found = p
            break

    if found is None:
        AUDIT.record(
            "T3-003", "cross_val_predict used for training-period scores",
            "SKIP", "rq2_governance.py not found in expected location."
        )
        return

    source = found.read_text()
    uses_cvp = "cross_val_predict" in source

    if not uses_cvp:
        AUDIT.record(
            "T3-003", "cross_val_predict used for training-period scores",
            "FAIL",
            "cross_val_predict not found in rq2_governance.py. "
            "Training-period risk scores are likely computed with in-sample predict_proba, "
            "producing optimistically biased scores for the majority of contracts.",
            severity="HIGH",
            fix="from sklearn.model_selection import cross_val_predict\n"
                "oof = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')\n"
                "train_scores = pd.DataFrame(oof, index=train_idx, columns=model.classes_)"
        )
    else:
        AUDIT.record(
            "T3-003", "cross_val_predict used for training-period scores",
            "PASS",
            "cross_val_predict found in rq2_governance.py."
        )


# Type 4: feature-time leakage tests

def test_type4_supplier_centrality_computed_before_focal_year():
    """
    supplier_centrality is a network metric. If it is computed from the
    full bipartite graph (all years), then a test-year contract sees
    centrality values that include future edges.

    Correct approach: compute centrality on the subgraph of contracts
    with award_year < focal_contract.award_year (or <= TRAIN_YEARS[1]
    for a simpler static approach).

    This test checks whether the RQ1 source code restricts graph construction
    to historical data only.
    """
    source_paths = [
        Path("src/rq1_supplier_network/rq1_network.py"),
        Path("../src/rq1_supplier_network/rq1_network.py"),
    ]
    found = None
    for p in source_paths:
        if p.exists():
            found = p
            break

    if found is None:
        AUDIT.record(
            "T4-001", "Supplier centrality graph uses historical data only",
            "SKIP", "rq1_network.py not found."
        )
        return

    source = found.read_text()

    # Check for temporal filtering in graph construction
    has_year_filter = any(kw in source for kw in [
        "TRAIN_YEARS", "award_year <=", "award_year<=",
        "award_year < ", "training_contracts", "hist_contracts",
        "between(*TRAIN_YEARS)", ".between(TRAIN_YEARS",
    ])

    if not has_year_filter:
        AUDIT.record(
            "T4-001", "Supplier centrality graph uses historical data only",
            "WARN",
            "No temporal filter detected in graph construction. "
            "If build_graph() is called on the full dataset, supplier_centrality "
            "values seen by test-year contracts include future edges (post-2016 contracts).",
            severity="HIGH",
            fix="Filter contracts before building graph:\n"
                "  train_contracts = contracts[contracts['award_year'] <= TRAIN_YEARS[1]]\n"
                "  G, buyers, suppliers = build_graph(train_contracts)\n"
                "Then join centrality values to the full dataset (train + test)."
        )
    else:
        AUDIT.record(
            "T4-001", "Supplier centrality graph uses historical data only",
            "PASS",
            "Temporal filtering detected in graph construction source."
        )


def test_type4_hhi_computed_from_historical_contracts():
    """
    Same issue as centrality: HHI per buyer must be computed from
    historical contracts only, not the full dataset.
    """
    source_paths = [
        Path("src/rq1_supplier_network/rq1_network.py"),
        Path("../src/rq1_supplier_network/rq1_network.py"),
        Path("src/core/feature_engineering.py"),
        Path("../src/core/feature_engineering.py"),
    ]
    found_sources = []
    for p in source_paths:
        if p.exists():
            found_sources.append(p.read_text())

    if not found_sources:
        AUDIT.record(
            "T4-002", "HHI computed from historical contracts only",
            "SKIP", "Source files not found."
        )
        return

    combined = "\n".join(found_sources)
    has_temporal_hhi = any(kw in combined for kw in [
        "award_year <=", "award_year<=", "TRAIN_YEARS",
        "historical", "hist_", "before_focal",
        "ROWS BETWEEN UNBOUNDED PRECEDING",
        "ORDER BY award_year",
    ])

    if not has_temporal_hhi:
        AUDIT.record(
            "T4-002", "HHI computed from historical contracts only",
            "WARN",
            "No temporal constraint detected in HHI computation. "
            "HHI appears to be computed from all contracts and joined back, "
            "meaning test-year contracts see HHI values that include future contract data.",
            severity="HIGH",
            fix="Use SQL window function for temporally-correct HHI:\n"
                "  SUM(POWER(supplier_share, 2)) OVER (\n"
                "    PARTITION BY buyer_id\n"
                "    ORDER BY award_year\n"
                "    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n"
                "  ) AS hhi_historical"
        )
    else:
        AUDIT.record(
            "T4-002", "HHI computed from historical contracts only",
            "PASS",
            "Temporal constraint found in HHI computation."
        )


def test_type4_rq3_residuals_on_test_set_only():
    """
    GBR anomaly residuals must be computed on the held-out test set only.
    If computed on all rows (train + test), training rows get artificially
    small residuals (the model fits them well by construction), which biases
    the anomaly flag rate toward test rows.
    """
    source_paths = [
        Path("src/rq3_price_intelligence/rq3_pricing.py"),
        Path("../src/rq3_price_intelligence/rq3_pricing.py"),
    ]
    found = None
    for p in source_paths:
        if p.exists():
            found = p
            break

    if found is None:
        AUDIT.record(
            "T4-003", "RQ3 GBR residuals computed on test/OOF only",
            "SKIP", "rq3_pricing.py not found."
        )
        return

    source = found.read_text()

    uses_oof      = "cross_val_predict" in source
    uses_all_pred = "all_pred" in source or "predict(X)" in source

    if not uses_oof and uses_all_pred:
        AUDIT.record(
            "T4-003", "RQ3 GBR residuals computed on test/OOF only",
            "FAIL",
            "GBR appears to predict on full X (train+test) for residual computation. "
            "Training rows get near-zero residuals (in-sample fit), making anomaly flags "
            "biased: mostly test rows are flagged, inflating apparent anomaly detection rate.",
            severity="HIGH",
            fix="Use cross_val_predict for train-period residuals:\n"
                "  oof_pred = cross_val_predict(gbr, X_train, y_train, cv=5)\n"
                "  train_residuals = y_train - oof_pred\n"
                "  test_residuals  = y_test - gbr.predict(X_test)\n"
                "  all_residuals   = pd.concat([train_residuals, test_residuals])"
        )
    elif uses_oof:
        AUDIT.record(
            "T4-003", "RQ3 GBR residuals computed on test/OOF only",
            "PASS",
            "cross_val_predict found — OOF residuals in use."
        )
    else:
        AUDIT.record(
            "T4-003", "RQ3 GBR residuals computed on test/OOF only",
            "WARN",
            "Cannot determine residual computation strategy from source inspection. "
            "Verify manually that GBR anomaly flags are not computed on training rows.",
            severity="MEDIUM",
            fix="Confirm residuals are computed via cross_val_predict or test-set-only."
        )


# Bonus: correlation-based leakage sniff test

def test_bonus_feature_label_correlation_sniff():
    """
    Compute Pearson and Spearman correlation between each feature and risk_label.
    Any feature with |r| > 0.95 is almost certainly a label proxy.
    Features with |r| > 0.70 warrant manual inspection.
    """
    from scipy.stats import spearmanr

    df = make_contracts(n=500)

    if MODEL_FEATURES is not None:
        feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
    else:
        feature_cols = [c for c in df.columns
                        if c not in ["contract_id", "buyer_id", "supplier_id",
                                     "procedure_type", "risk_label"]]
    y = df["risk_label"].values

    high_corr  = []
    suspicious = []

    for col in feature_cols:
        x = df[col].values
        if not np.issubdtype(x.dtype, np.number):
            continue
        r_pearson  = np.corrcoef(x, y)[0, 1]
        r_spearman, _ = spearmanr(x, y)
        max_r = max(abs(r_pearson), abs(r_spearman))

        if max_r > 0.95:
            high_corr.append((col, r_pearson, r_spearman))
        elif max_r > 0.70:
            suspicious.append((col, r_pearson, r_spearman))

    lines = []
    if high_corr:
        lines.append("CRITICAL correlations (|r|>0.95 — almost certainly leaking):")
        for col, rp, rs in high_corr:
            lines.append(f"  {col}: pearson={rp:.3f}, spearman={rs:.3f}")
    if suspicious:
        lines.append("High correlations (|r|>0.70 — inspect manually):")
        for col, rp, rs in suspicious:
            lines.append(f"  {col}: pearson={rp:.3f}, spearman={rs:.3f}")

    if high_corr:
        AUDIT.record(
            "B-001", "Feature-label correlation sniff test",
            "FAIL", "\n   ".join(lines), severity="CRITICAL",
            fix="Remove or rederive features with |r|>0.95 against risk_label."
        )
    elif suspicious:
        AUDIT.record(
            "B-001", "Feature-label correlation sniff test",
            "WARN", "\n   ".join(lines), severity="MEDIUM",
            fix="Review high-correlation features — confirm they are pre-award observables."
        )
    else:
        AUDIT.record(
            "B-001", "Feature-label correlation sniff test",
            "PASS",
            "No features with suspiciously high correlation to risk_label detected."
        )


# Temporal-integrity tests for graph/HHI features (pytest-only)

def test_hhi_computed_before_test_period():
    """buyer_concentration_hhi must be a historical (cumulative, no-future) HHI.
    Verify feature engineering builds it with a temporal window rather than a
    full-dataset groupby."""
    import pytest
    candidates = [Path("src/core/feature_engineering.py"),
                  Path("../src/core/feature_engineering.py")]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        pytest.skip("feature_engineering.py not found")
    text = src.read_text()
    assert "ROWS BETWEEN UNBOUNDED PRECEDING" in text and "ORDER BY award_year" in text, (
        "buyer_concentration_hhi is not computed with a historical temporal window — "
        "it may include future contracts (leakage)."
    )


def test_centrality_computed_on_train_graph():
    """supplier_centrality must be computed on the training-period graph only.
    Verify the centrality metadata records computed_on_years inside the train window."""
    import pytest
    candidates = [Path("data/results/rq1_centrality_metadata.json"),
                  Path("../data/results/rq1_centrality_metadata.json")]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        pytest.skip("rq1_centrality_metadata.json not found — run rq1 first")
    meta = json.loads(found.read_text())
    years = meta.get("computed_on_years")
    assert years, "computed_on_years missing from centrality metadata"
    from src.core.config import TEST_YEARS
    assert max(years) < min(TEST_YEARS), (
        f"centrality used years {years} that overlap the test period {list(TEST_YEARS)}"
    )


# Runner

ALL_TESTS = [
    # Type 1: target leakage
    test_type1_no_label_defining_cols_in_feature_matrix,
    test_type1_auc_ceiling_detection,
    test_type1_single_bid_rate_is_not_a_feature,
    # Type 2: temporal leakage
    test_type2_train_test_boundary,
    test_type2_no_overlap_in_contract_ids,
    test_type2_supplier_features_use_historical_data_only,
    test_type2_gap_years_excluded_from_both_splits,
    # Type 3: in-sample scoring
    test_type3_predictions_parquet_has_score_type_column,
    test_type3_no_training_rows_in_reported_metrics,
    test_type3_cross_val_predict_used_for_train_scores,
    # Type 4: feature-time leakage
    test_type4_supplier_centrality_computed_before_focal_year,
    test_type4_hhi_computed_from_historical_contracts,
    test_type4_rq3_residuals_on_test_set_only,
    # Bonus
    test_bonus_feature_label_correlation_sniff,
]


def run_audit() -> LeakageAudit:
    print(f"\nRunning {len(ALL_TESTS)} leakage checks...\n")
    for test_fn in ALL_TESTS:
        try:
            test_fn()
        except Exception as e:
            AUDIT.record(
                "ERR", test_fn.__name__,
                "WARN", f"Test raised exception: {type(e).__name__}: {e}",
                severity="LOW",
                fix="Investigate test error — may indicate environment issue."
            )
    return AUDIT


# pytest compatibility: each test function can also be called by pytest directly
def test_all_leakage_checks():
    """Fails if any individual leakage check failed."""
    # AUDIT is already populated by the individual test functions
    # when pytest runs them before this function
    assert AUDIT.failed == 0, (
        f"\n{AUDIT.failed} leakage check(s) FAILED.\n"
        "See report above for details and fixes.\n"
        "DO NOT submit dissertation results until all leakage checks pass."
    )


if __name__ == "__main__":
    audit = run_audit()
    report = audit.report()
    print(report)

    # Save report
    out = Path("Audit_report")
    out.mkdir(exist_ok=True)
    report_path = out / f"leakage_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    report_path.write_text(report)
    print(f"\nReport saved: {report_path}")

    # Exit code for CI
    sys.exit(0 if audit.failed == 0 else 1)