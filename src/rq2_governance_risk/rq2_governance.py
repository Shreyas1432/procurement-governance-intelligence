import json
import pickle
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.core.config import (
    DATA_FEATURES,
    DATA_MODELS,
    DATA_RESULTS,
    LR_PARAMS,
    RF_PARAMS,
    RQ2_MODEL_FEATURES,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    XGB_PARAMS,
)
from src.common.utils import ci95, get_logger, merge_json, timed, write_json
from src.common.evaluation import coverage_aware_corr

logger = get_logger("rq2_governance")

RQ2_LOAD_COLUMNS = [
    "contract_id",
    "buyer_id",
    "supplier_id",
    "cpv_division",
    "procedure_type",
    "bid_count",
    "buyer_dependency_ratio",
    "contract_amendments",
    "award_year",
    "contract_value",
    "contract_value_log",
    "risk_label",
]

class GovernanceRiskClassifier:
    FEATURES = RQ2_MODEL_FEATURES

    def __init__(self):
        self.rf = RandomForestClassifier(**RF_PARAMS)
        self.xgb = _xgb_or_fallback()
        self.lr = LogisticRegression(**LR_PARAMS)
        self.scaler = StandardScaler()

    def train_all_models(self, X_train, y_train, X_test, y_test):
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Train RF (Algorithm 1)
        cv_f1 = cross_val_score(self.rf, X_train, y_train, cv=cv, scoring="f1_weighted")
        self.rf.fit(X_train, y_train)
        rf_pred = self.rf.predict(X_test)
        rf_prob = _proba(self.rf, X_test)
        results["rf_f1"] = float(f1_score(y_test, rf_pred, average="weighted"))
        results["rf_cv_f1"] = float(cv_f1.mean())
        results["rf_auc"] = _auc(y_test, rf_prob)

        # Inject scale_pos_weight from TRAIN class ratio to handle imbalance for XGBoost (Algorithm 2)
        if hasattr(self.xgb, "set_params") and type(self.xgb).__name__ == "XGBClassifier":
            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            if n_pos > 0:
                self.xgb.set_params(scale_pos_weight=n_neg / n_pos, max_depth=3)
        self.xgb.fit(X_train, y_train)
        xgb_prob = _proba(self.xgb, X_test)
        results["xgb_auc"] = _auc(y_test, xgb_prob)

        # Train LR (Algorithm 3)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        self.lr.fit(X_train_s, y_train)
        lr_prob = _proba(self.lr, X_test_s)
        results["lr_auc"] = _auc(y_test, lr_prob)

        # Persist the in-run 2020 TEST positive-class probability arrays
        self._test_probs = {
            "xgb": np.asarray(_positive_class(xgb_prob), dtype=float),
            "rf": np.asarray(_positive_class(rf_prob), dtype=float),
            "lr": np.asarray(_positive_class(lr_prob), dtype=float),
        }
        self._test_y = np.asarray(y_test, dtype=int)

        # Weighted-ensemble AUC on the held-out test set
        try:
            ensemble_prob = 0.5 * xgb_prob + 0.3 * rf_prob + 0.2 * lr_prob
            results["ensemble_auc"] = _auc(y_test, ensemble_prob)
        except Exception:
            results["ensemble_auc"] = 0.0

        results["rf_top3_features"] = _top_features(self.rf, self.FEATURES)
        results["xgb_top3_features"] = _top_features(self.xgb, self.FEATURES)

        logger.info("XGB  (headline): ROC-AUC %.4f", results["xgb_auc"])
        logger.info("RF   (baseline): ROC-AUC %.4f | weighted-F1 %.4f",
                    results["rf_auc"], results["rf_f1"])
        logger.info("LR   (baseline): ROC-AUC %.4f", results["lr_auc"])
        return results

    def ensemble_predict(self, X):
        xgb_prob = _positive_class(_proba(self.xgb, X))
        rf_prob = _positive_class(_proba(self.rf, X))
        lr_prob = _positive_class(_proba(self.lr, self.scaler.transform(X)))
        return 0.5 * xgb_prob + 0.3 * rf_prob + 0.2 * lr_prob

def run_rq2() -> bool:
    path = DATA_FEATURES / "rq2_governance_features.parquet"
    if not path.exists():
        raise FileNotFoundError("Run feature engineering before RQ2")
    # NOTE: supplier_centrality (RQ1 hand-off) was dropped from the model on
    # 2026-06-12 as a documented null (THRESHOLD_JUSTIFICATION.md Section 10), so
    # the former _attach_supplier_centrality join is no longer performed here.
    df = (
        pl.scan_parquet(path)
        .select(RQ2_LOAD_COLUMNS)
        .filter(pl.col("award_year") <= max(TEST_YEARS))
        .collect(engine="streaming")
    )
    df = df.with_columns(pl.col(RQ2_MODEL_FEATURES).fill_null(0))
    pdf = df.to_pandas()
    X = pdf[RQ2_MODEL_FEATURES].astype(float)
    y = pdf["risk_label"].astype(int)
    
    # Stratified temporal splitting
    train_mask = pdf["award_year"].between(min(TRAIN_YEARS), max(TRAIN_YEARS))
    test_mask = pdf["award_year"].between(min(TEST_YEARS), max(TEST_YEARS))
    
    # Fallback to random if temporal window contains insufficient test records
    if train_mask.sum() < 100 or test_mask.sum() < 100 or y[train_mask].nunique() < 2 or y[test_mask].nunique() < 2:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(pdf.index, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None)
        train_mask = pdf.index.isin(train_idx)
        test_mask = pdf.index.isin(test_idx)
        strategy = "stratified_random"
    else:
        strategy = "configured_temporal"

    split_meta = {
        "split_strategy": strategy,
        "train_n": int(train_mask.sum()),
        "test_n": int(test_mask.sum()),
        "train_years": [int(pdf.loc[train_mask, "award_year"].min()), int(pdf.loc[train_mask, "award_year"].max())],
        "test_years": [int(pdf.loc[test_mask, "award_year"].min()), int(pdf.loc[test_mask, "award_year"].max())],
    }

    clf = GovernanceRiskClassifier()
    with timed(logger, "train RF/XGB/LR models"):
        metrics = clf.train_all_models(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
    metrics.update(split_meta)

    # --- Honest headline metrics on the 2020 TEST split ---------------------
    # The operating threshold is chosen by maximising F1 on the VAL 2019 split
    # ONLY, then frozen and applied to the 2020 TEST split (the test labels never
    # influence threshold selection). XGBoost is the headline model; it is already
    # fitted on TRAIN (2014-2018) inside train_all_models. Reports ROC-AUC, PR-AUC,
    # F1, precision, recall and the confusion matrix at the frozen threshold.
    val_mask = pdf["award_year"].between(min(VAL_YEARS), max(VAL_YEARS))
    if strategy == "configured_temporal" and val_mask.sum() > 0 and y[val_mask].nunique() > 1:
        val_prob = _positive_class(_proba(clf.xgb, X[np.asarray(val_mask)]))
        test_prob = _positive_class(_proba(clf.xgb, X[np.asarray(test_mask)]))
        thr = _select_threshold_max_f1(y[val_mask].to_numpy(), val_prob)
        headline = _headline_metrics(y[test_mask].to_numpy(), test_prob, thr)
        headline["operating_threshold"] = float(thr)
        headline["threshold_selection_rule"] = (
            "argmax F1 on VAL 2019, frozen and applied to TEST 2020 (test labels "
            "never used in threshold selection)"
        )
        # Also report at a fixed default threshold of 0.5. The 2019->2020 base-rate
        # drift (10% -> 37% positive) makes the VAL-frozen operating point fragile, so
        # both F1s are recorded; ROC-AUC (threshold-free) remains the headline.
        head_05 = _headline_metrics(y[test_mask].to_numpy(), test_prob, 0.5)
        headline["f1_at_0p5"] = head_05["f1"]
        headline["precision_at_0p5"] = head_05["precision"]
        headline["recall_at_0p5"] = head_05["recall"]
        headline["confusion_matrix_at_0p5"] = head_05["confusion_matrix"]
        headline["headline_model"] = "xgboost"
        headline["val_n"] = int(val_mask.sum())
        headline["val_years"] = [int(min(VAL_YEARS)), int(max(VAL_YEARS))]
        headline["class_balance"] = {"0": int((y == 0).sum()), "1": int((y == 1).sum())}
        headline["positive_rate"] = float((y == 1).mean())
        metrics["test_headline"] = headline
        logger.info(
            "RQ2 headline (2020 TEST): ROC-AUC=%.4f PR-AUC=%.4f | F1=%.4f @VAL-frozen thr %.4f "
            "(P=%.4f R=%.4f) | F1=%.4f @0.5 default",
            headline["roc_auc"], headline["pr_auc"], headline["f1"], thr,
            headline["precision"], headline["recall"], headline["f1_at_0p5"],
        )
    else:
        logger.info("VAL 2019 unavailable or single-class — headline threshold metrics skipped")

    # --- Honest risk scoring (no in-sample leakage) -------------------------
    # Training-period contracts are scored with out-of-fold probabilities so no
    # contract is scored by a model that was fitted on it. Test-period contracts
    # use held-out probabilities from models fitted on the training period only.
    # The 0.5 / 0.3 / 0.2 ensemble weights are preserved exactly.
    train_idx_mask = np.asarray(train_mask)
    test_idx_mask = np.asarray(test_mask)
    X_train, y_train = X[train_idx_mask], y[train_idx_mask]

    # --- Cross-validation diagnostics (corrected 2026-06-10) ----------------
    # The previous non-shuffled StratifiedKFold(5) ran over the year-ordered
    # training block, which made fold 1 (earliest contracts) trivially separable
    # and produced a fold-1=1.000 artifact reported as 0.799 +/- 0.104. That
    # number misrepresented stability and is no longer reported. Two honest
    # estimates replace it; the 2020 hold-out test AUC remains the headline:
    #   1. SHUFFLED StratifiedKFold(5) -- internal stability, explicitly NON-temporal.
    #   2. Forward-chaining temporal CV (train years <= t, validate t+1) -- the
    #      temporally-honest estimate of year-over-year generalisation.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_f1_ci = ci95(cross_val_score(clf.rf, X_train, y_train, cv=skf, scoring="f1_weighted"))
    try:
        xgb_auc_shuffled = ci95(cross_val_score(clf.xgb, X_train, y_train, cv=skf, scoring="roc_auc"))
    except Exception:
        xgb_auc_shuffled = ci95([])
    train_years_arr = pdf.loc[train_mask, "award_year"].to_numpy()
    xgb_auc_temporal = ci95(_forward_chaining_auc(clf.xgb, X_train, y_train, train_years_arr))
    logger.info(
        "XGB AUC: shuffled 5-fold CV (non-temporal) %.3f +/- %.3f | "
        "forward-chaining temporal CV %.3f +/- %.3f | "
        "headline 2020 hold-out test AUC %.3f",
        xgb_auc_shuffled["mean"], xgb_auc_shuffled["std"],
        xgb_auc_temporal["mean"], xgb_auc_temporal["std"],
        metrics.get("xgb_auc", 0.0),
    )
    merge_json(DATA_RESULTS / "model_performance_ci.json", {
        "rq2_rf_f1": rf_f1_ci,
        # rq2_xgb_auc now holds the shuffled-CV stability number (the dashboard
        # reads this key); the non-shuffled artifact is no longer written.
        "rq2_xgb_auc": xgb_auc_shuffled,
        "rq2_xgb_auc_cv_shuffled": xgb_auc_shuffled,
        "rq2_xgb_auc_cv_temporal": xgb_auc_temporal,
    })
    metrics["xgb_auc_cv_shuffled"] = xgb_auc_shuffled
    metrics["xgb_auc_cv_temporal"] = xgb_auc_temporal

    oof_xgb = _positive_class(cross_val_predict(clf.xgb, X_train, y_train, cv=5, method="predict_proba"))
    oof_rf = _positive_class(cross_val_predict(clf.rf, X_train, y_train, cv=5, method="predict_proba"))
    oof_lr = _positive_class(cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(**LR_PARAMS)), X_train, y_train, cv=5, method="predict_proba"))
    train_scores = 0.5 * oof_xgb + 0.3 * oof_rf + 0.2 * oof_lr
    test_scores = clf.ensemble_predict(X[test_idx_mask])

    base_cols = ["contract_id", "buyer_id", "supplier_id", "cpv_division", "procedure_type", "award_year", "contract_amendments", "bid_count"]
    base_pdf = df.select(base_cols).to_pandas()
    train_out = base_pdf[train_idx_mask].copy()
    train_out["ensemble_risk_prob"] = train_scores
    train_out["score_type"] = "out_of_fold"
    test_out = base_pdf[test_idx_mask].copy()
    test_out["ensemble_risk_prob"] = test_scores
    test_out["score_type"] = "held_out"
    all_scores = pd.concat([train_out, test_out], ignore_index=True)
    all_scores["risk_class"] = np.where(
        all_scores["ensemble_risk_prob"] >= 0.75, "HIGH",
        np.where(all_scores["ensemble_risk_prob"] >= 0.45, "MEDIUM", "LOW"),
    )
    out = pl.from_pandas(all_scores)
    out.write_parquet(DATA_RESULTS / "rq2_governance_predictions.parquet")
    
    DATA_MODELS.mkdir(parents=True, exist_ok=True)
    for name, model in {"rq2_rf_model.pkl": clf.rf, "rq2_xgb_model.pkl": clf.xgb, "rq2_lr_model.pkl": clf.lr}.items():
        with open(DATA_MODELS / name, "wb") as handle:
            pickle.dump(model, handle)
            
    high_series = out.filter(pl.col("risk_class") == "HIGH")["contract_amendments"]
    low_series = out.filter(pl.col("risk_class") == "LOW")["contract_amendments"]
    high = (high_series.gt(0).mean() if high_series.len() else 0.0) or 0.0
    low = (low_series.gt(0).mean() if low_series.len() else 0.0) or 0.0
    
    metrics["n_high_risk"] = int(high_series.len())
    metrics["n_low_risk"] = int(low_series.len())
    metrics["high_risk_amendment_rate"] = float(high)
    metrics["low_risk_amendment_rate"] = float(low)
    metrics["enrichment_ratio"] = float(high / low) if low else 0.0
    
    # Buyer-level aggregates of governance risk vs competition proxies.
    # MED-01: the negotiated/restricted share is now named honestly
    # (negotiated_restricted_rate); single_bid_rate is the TRUE single-bidder
    # rate (bid_count <= 1), and the R_gov correlation uses that true rate.
    # Sort by buyer_id before any cross-buyer reduction: an unsorted group_by emits
    # rows in non-deterministic order, so the float summation in .mean() drifts in the
    # last ULP between runs. Sorting makes negotiated_restricted_rate (and the r_gov
    # correlation) byte-identical across runs (RQ1 Section 4 precedent).
    buyer_agg = out.group_by("buyer_id").agg(
        pl.mean("ensemble_risk_prob").alias("r_gov"),
        ((pl.col("procedure_type").str.contains("(?i)negotiated") | (pl.col("procedure_type") == "RESTRICTED")).cast(pl.Float64)).mean().alias("negotiated_restricted_rate"),
        ((pl.col("bid_count") <= 1).cast(pl.Float64)).mean().alias("single_bid_rate"),
    ).sort("buyer_id")
    # NaN-aware, three-state correlation (NaN-guard bug fix, 2026-06-15). The old
    # guard fill-null/np.std path returned a fabricated 0.0 when single_bid_rate was
    # null for buyers with no observed bid_count (degenerate low coverage) -- which
    # looked identical to a measured-null correlation. coverage_aware_corr drops
    # null pairs and reports an explicit state; the headline value is None when the
    # observed buyers are an unrepresentative minority (see *_status block in
    # _emit_rq2_diagnostics). Reported diagnostic, never a gate.
    rgov_corr_result = coverage_aware_corr(
        buyer_agg["r_gov"], buyer_agg["single_bid_rate"]
    )
    metrics["rgov_single_bidder_corr"] = rgov_corr_result["value"]
    metrics["single_bid_rate"] = float((out["bid_count"] <= 1).cast(pl.Float64).mean())
    metrics["negotiated_restricted_rate"] = float(buyer_agg["negotiated_restricted_rate"].mean())

    # SHAP analysis on the clean XGB model (RQ1 has no SHAP — deterministic by design).
    # Skipped when the model is single-feature: a one-feature SHAP decomposition is
    # degenerate (every prediction's attribution is the lone feature). The value-only
    # interpretability artifact is a partial-dependence/calibration curve instead
    # (data/results/rq2_value_only_pdp.png).
    if type(clf.xgb).__name__ == "XGBClassifier" and len(clf.FEATURES) >= 2:
        with timed(logger, "SHAP analysis (XGB, clean features)"):
            shap_meta = run_shap(
                clf.xgb,
                X.loc[train_mask],   # full training-period rows (sample drawn inside)
                y.loc[train_mask],
                features=clf.FEATURES,
                sample_n=4000,
                seed=42,
            )
        metrics.update(shap_meta)
    elif type(clf.xgb).__name__ == "XGBClassifier":
        logger.info("Single-feature model (%s) — model SHAP skipped (degenerate); "
                    "see rq2_value_only_pdp.png for partial dependence", clf.FEATURES)
    else:
        logger.info("XGB not available — SHAP skipped (GBR fallback in use)")

    # --- Reporting-integrity DIAGNOSTICS (value-only lock preserved) ---------
    # All blocks below are reporting/diagnostic only. They add NO feature, change
    # NO model/threshold, and leave the headline ROC-AUC untouched; every figure is
    # computed from arrays already produced in this run (or read from in-run JSON
    # artifacts). Determinism is preserved (closed-form, sorted, no resampling).
    _emit_rq2_diagnostics(metrics, df, pdf, y, train_mask, test_mask, out, clf)

    write_json(DATA_RESULTS / "rq2_success_metrics.json", metrics)
    return True


def _emit_rq2_diagnostics(metrics, df, pdf, y, train_mask, test_mask, out, clf):
    """Assemble the RQ2 reporting-integrity diagnostic blocks (T1-T8) and write the
    candidate-disposition artifacts (T4) and the degenerate-correlation null (T2).
    Mutates ``metrics`` in place; writes no model or threshold."""
    # ---- T1: population reconciliation -------------------------------------
    val_mask = pdf["award_year"].between(min(VAL_YEARS), max(VAL_YEARS))
    n_train, n_val, n_test = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
    train_pos = int(y[train_mask].sum())
    val_pos = int(y[val_mask].sum())
    test_pos = int(y[test_mask].sum())
    panel_total = n_train + n_val + n_test
    panel_pos = train_pos + val_pos + test_pos
    scored_total = int(out.height)
    n_oof = int(out.filter(pl.col("score_type") == "out_of_fold").height)
    n_held = int(out.filter(pl.col("score_type") == "held_out").height)
    high_n = int(metrics["n_high_risk"])
    low_n = int(metrics["n_low_risk"])
    medium_n = scored_total - high_n - low_n
    high_plus_low = high_n + low_n

    metrics["populations"] = {
        "classifier_population": {
            "definition": "Full year-stratified 2014-2020 panel; rows with a risk_label "
                          "and contract_value_log, temporally split into TRAIN/VAL/TEST. "
                          "This is the population the classifier is fit and evaluated on.",
            "filter": "award_year in [2014..2020]; TRAIN 2014-2018, VAL 2019, TEST 2020",
            "n_total": panel_total,
            "train": {"years": [2014, 2018], "n": n_train, "label_positives": train_pos},
            "val": {"years": [2019, 2019], "n": n_val, "label_positives": val_pos},
            "test": {"years": [2020, 2020], "n": n_test, "label_positives": test_pos},
            "label_positives_total": panel_pos,
            "positive_rate": (panel_pos / panel_total) if panel_total else 0.0,
            "note": "label_positives_total reconciles to the binary class-1 count "
                    "(class_balance['1']).",
        },
        "rgov_scored_population": {
            "definition": "Contracts that receive an HONEST ensemble risk score (R_gov): "
                          "TRAIN rows scored out-of-fold + TEST rows scored held-out. "
                          "VAL 2019 is EXCLUDED because it is used only to select the "
                          "operating threshold and is never scored.",
            "filter": "award_year in TRAIN_YEARS (out-of-fold) UNION award_year in "
                      "TEST_YEARS (held-out); VAL 2019 excluded",
            "n_total": scored_total,
            "n_train_out_of_fold": n_oof,
            "n_test_held_out": n_held,
            "why_differs_from_classifier_population": (
                f"Excludes the {n_val:,} VAL-2019 rows (threshold-selection only, never "
                f"scored): {panel_total:,} panel - {n_val:,} VAL = {scored_total:,} scored."
            ),
            "risk_buckets": {
                "bucket_rule": "ensemble_risk_prob >= 0.75 -> HIGH; >= 0.45 -> MEDIUM; else LOW",
                "is_rgov_score_bucket_not_classifier_label": True,
                "HIGH": high_n,
                "MEDIUM": medium_n,
                "LOW": low_n,
                "high_plus_low": high_plus_low,
                "reconciliation_note": (
                    f"n_high_risk + n_low_risk = {high_plus_low:,} EXCLUDES the "
                    f"{medium_n:,} MEDIUM rows and the {n_val:,} VAL rows; it is NOT the "
                    f"classifier population ({panel_total:,}) and not the scored "
                    f"population ({scored_total:,})."
                ),
            },
        },
        "enrichment_population": {
            "definition": "high_risk_amendment_rate, low_risk_amendment_rate and "
                          "enrichment_ratio are computed over the HIGH and LOW R_gov "
                          "SCORE buckets of the rgov_scored_population (NOT the "
                          "classifier label).",
            "high_low_are_rgov_score_buckets": True,
            "enrichment_denominator": {"high_risk_n": high_n, "low_risk_n": low_n},
        },
    }
    # Companion top-level naming for the enrichment fields (T1).
    metrics["enrichment_population"] = (
        "rgov_scored_population HIGH and LOW R_gov score buckets (NOT the classifier label)"
    )
    metrics["enrichment_denominator"] = {"high_risk_n": high_n, "low_risk_n": low_n}

    # ---- T3: single-bidder definitions (denominator reconciliation) ---------
    scored_obs = int(out.filter(pl.col("bid_count").is_not_null()).height)
    panel_obs = int(df.filter(pl.col("bid_count").is_not_null()).height)
    panel_rate = float((df["bid_count"] <= 1).cast(pl.Float64).mean() or 0.0)
    metrics["single_bidder_definitions"] = {
        "note": "All rates are 'single-bidder among OBSERVED lots' (bid_count <= 1 where "
                "bid_count is non-null); they differ only by POPULATION and denominator, "
                "not definition. bid_count (lot_bidscount) coverage is sparse "
                "(~1.6% raw / ~3.6% scored panel), so the choice of denominator matters.",
        "rq2_scored_observed": {
            "rate": float(metrics["single_bid_rate"]),
            "definition": "single-bidder (bid_count <= 1) among rgov_scored_population "
                          "contracts with non-null bid_count -- the value currently "
                          "reported as single_bid_rate.",
            "population": "rgov_scored_population (TRAIN out-of-fold + TEST held-out; "
                          "VAL excluded)",
            "denominator": "scored contracts with non-null bid_count",
            "n_observed": scored_obs,
            "n_population": scored_total,
            "coverage_fraction": (scored_obs / scored_total) if scored_total else 0.0,
            "is_current_single_bid_rate": True,
        },
        "panel_observed": {
            "rate": panel_rate,
            "definition": "single-bidder (bid_count <= 1) among full 2014-2020 panel "
                          "contracts with non-null bid_count.",
            "population": "classifier_population (TRAIN+VAL+TEST)",
            "denominator": "panel contracts with non-null bid_count",
            "n_observed": panel_obs,
            "n_population": int(df.height),
            "coverage_fraction": (panel_obs / int(df.height)) if df.height else 0.0,
        },
        "raw_dataset_observed_eda": {
            "rate": 0.298,
            "rate_train_window_2014_2018": 0.313,
            "definition": "single-bidder rate among OBSERVED lots (lot_bidscount "
                          "non-null) over the FULL raw ANAC dataset -- the 29.8% figure "
                          "cited in the thesis.",
            "population": "full raw IT_DIB_2023 dataset (~12.1M lot-level rows)",
            "denominator": "raw lots with non-null lot_bidscount (~192K, ~1.6%)",
            "source": "docs/THRESHOLD_JUSTIFICATION.md Section 6 (bid-count field "
                      "correction); an EDA-stage figure, NOT recomputed in the RQ2 run.",
            "is_thesis_cited_29_8pct": True,
        },
        "recommendation": "Cite ONE definition consistently. The thesis 29.8% is the "
                          "full-raw-dataset observed-lot rate; the RQ2 module reports the "
                          "scored-panel observed-lot rate. They agree in DEFINITION and "
                          "differ only in POPULATION/denominator.",
    }

    # ---- T2: rgov_single_bidder_corr degenerate-coverage diagnostic ---------
    buyer_agg = out.group_by("buyer_id").agg(
        pl.mean("ensemble_risk_prob").alias("r_gov"),
        ((pl.col("bid_count") <= 1).cast(pl.Float64)).mean().alias("single_bid_rate"),
    ).sort("buyer_id")
    cres = coverage_aware_corr(buyer_agg["r_gov"], buyer_agg["single_bid_rate"])
    b_obs = buyer_agg["single_bid_rate"].drop_nulls().to_numpy()
    var_obs = float(np.var(b_obs)) if b_obs.size else 0.0
    corr_status = {
        "value": cres["value"],
        "status": cres["status"],
        "n_buyers_total": cres["n_total"],
        "n_buyers_nonnull_single_bid_rate": cres["n_paired"],
        "buyer_coverage_fraction": cres["coverage_fraction"],
        "contract_coverage_fraction": (scored_obs / scored_total) if scored_total else 0.0,
        "variance_single_bid_rate_observed": var_obs,
        "denominator": "buyer-level rows of rgov_scored_population; single_bid_rate is "
                       "defined only for buyers with at least one observed bid_count.",
        "exact_zero_mechanism": "FIXED 2026-06-15. Previously single_bid_rate was null "
                                "for buyers with no observed bid_count, the full-array "
                                "np.std was NaN, the 'std > 0' guard was False, pearsonr "
                                "was skipped and the function returned a fabricated 0.0. "
                                "The metric now drops null pairs and reports an explicit "
                                "3-state result; at this coverage the state is "
                                "'degenerate_low_coverage' and the headline value is null.",
        "observed_value": cres["observed_value"],
        "interpretation": "Uninformative at this coverage (mirrors the RQ1 "
                          "dependency_single_bidder_corr limitation, Section 9). The "
                          "observed-only r is reported in 'observed_value' but is computed "
                          "on an unrepresentative minority of buyers. Reported, not gated; "
                          "relocated to rq2_feature_exploration_nulls.json.",
    }
    metrics["rgov_single_bidder_corr_status"] = corr_status
    merge_json(DATA_RESULTS / "rq2_feature_exploration_nulls.json",
               {"rgov_single_bidder_corr": corr_status})

    # ---- T6: DeLong AUC-difference significance (2020 test, no refit) --------
    tp = getattr(clf, "_test_probs", None)
    ty = getattr(clf, "_test_y", None)
    if tp is not None and ty is not None and len(np.unique(ty)) > 1:
        xgb_lr = _delong_pair(ty, tp["xgb"], tp["lr"])
        xgb_rf = _delong_pair(ty, tp["xgb"], tp["rf"])
        metrics["auc_significance"] = {
            "method": "DeLong paired AUC-difference test (fast closed-form, Sun & Xu 2014 "
                      "/ DeLong et al. 1988); no bootstrap, deterministic.",
            "test_set": "2020 hold-out TEST",
            "n": int(len(ty)),
            "n_positive": int(np.sum(ty)),
            "xgb_minus_lr": {"comparison": "XGB - LR", **xgb_lr},
            "xgb_minus_rf": {"comparison": "XGB - RF", **xgb_rf},
            "note": "Substantiates or softens the 'XGB edges LR/RF' claim on the 2020 "
                    "point estimate; reported as computed.",
        }

    # ---- T7: PR-AUC base-rate context --------------------------------------
    metrics["pr_auc_baseline"] = (test_pos / n_test) if n_test else 0.0
    metrics["pr_auc_baseline_note"] = (
        "pr_auc_baseline is the 2020-test positive rate -- the PR-AUC of a random "
        "classifier. PR-AUC is base-rate-dependent and would be lower at the ~16% panel "
        "positive rate."
    )

    # ---- T5: rf_cv_f1 annotation (metadata only) ---------------------------
    metrics["rf_cv_f1_note"] = (
        "shuffled-CV F1; base-rate-inflated and NOT comparable to temporal-test F1 "
        "(0.494) due to 2019->2020 COVID base-rate drift."
    )

    # ---- T8: metric framing labels (no new computation) --------------------
    if isinstance(metrics.get("test_headline"), dict):
        metrics["test_headline"]["framing_label"] = "2020 point estimate"
    if isinstance(metrics.get("xgb_auc_cv_temporal"), dict):
        metrics["xgb_auc_cv_temporal"]["framing_label"] = (
            "generalisation estimate (forward-chaining temporal CV)"
        )

    # ---- T4: candidate-disposition table -----------------------------------
    _write_feature_disposition(metrics)


# Canonical feature-count statement (B1). Used VERBATIM in rq2_feature_disposition.md,
# rq2_report.md and THRESHOLD_JUSTIFICATION.md Section 10 so all three agree.
CANONICAL_DISPOSITION_STATEMENT = (
    "8 features were evaluated. cpv_enc and cpv_risk_score are two encodings of one "
    "procedure-type signal; award_year is the temporal-split key (constant at test). "
    "Excluding these leaves 5 substantive non-value candidates, of which 0 survived and "
    "contract_value_log was retained -- i.e. value-only. 4 are documented nulls "
    "(supplier_centrality, buyer_concentration_hhi, buyer_region, tender_lotscount); the "
    "5th removal is the cpv pair as a procedure proxy."
)

# Gate-statistic generalisation (B2): the leakage/proxy gate is "association with a label
# input by the APPROPRIATE statistic" -- Pearson |corr| for numeric features, eta
# (correlation ratio) for a categorical feature against a numeric label input.
GATE_STATISTIC_NOTE = (
    "The proxy gate generalises to 'association with a label input by the appropriate "
    "statistic': Pearson |corr| for numeric candidates (e.g. buyer_concentration_hhi 0.77), "
    "and eta (correlation ratio, categorical->numeric) for categorical candidates "
    "(buyer_region 0.966). eta is NOT the Pearson |corr| gate; the two are distinct "
    "statistics applied to distinct feature types."
)


def _write_feature_disposition(metrics):
    """Emit rq2_feature_disposition.json + .md: every initial candidate feature with
    its disposition_type, decisive_metric (value) and the gate it tripped, plus the
    reconciling N_initial -> N_kept counts authoritative for report Section 6.4.
    Decisive figures are read from the in-run null artifacts where available."""
    def _load(name):
        path = DATA_RESULTS / name
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    hhi = _load("rq2_hhi_proxy_null.json")
    expl = _load("rq2_feature_exploration_nulls.json")
    cand = expl.get("candidates", {})
    lots = cand.get("tender_lotscount", {}).get("proxy_audit", {})
    region = cand.get("buyer_region", {}).get("proxy_audit", {})

    PROXY_GATE = 0.55   # leakage WARN gate (test_full_leakage_audit_three_rqs.py)
    headline_auc = float(metrics.get("xgb_auc", 0.0))
    hhi_corr = float(hhi.get("corr_with_buyer_dependency_ratio", 0.0))
    lots_corr = float(lots.get("corr_value_log", 0.0))
    lots_lift = float(cand.get("tender_lotscount", {}).get("marginal_test_auc_lift", 0.0))
    region_eta = float(region.get("eta_contract_amendments", 0.0))
    region_lift = float(cand.get("buyer_region", {}).get("marginal_test_auc_lift", 0.0))

    candidates = [
        {
            "feature": "contract_value_log", "stage": "modelling_candidate",
            "disposition_type": "kept",
            "decisive_metric": "2020 TEST ROC-AUC (headline)",
            "decisive_value": headline_auc, "gate_tripped": "none",
            "note": "The entire value-only model.",
        },
        {
            "feature": "tender_lotscount", "stage": "clean_feature_exploration",
            "disposition_type": "redundancy_drop",
            "decisive_metric": "|corr| with contract_value_log (Pearson)",
            "decisive_value": lots_corr,
            "gate_statistic": "Pearson |corr|",
            "gate_tripped": (f"NO -- {lots_corr:.2f} is BELOW the {PROXY_GATE} leakage "
                             f"gate; dropped for redundancy / negligible marginal lift "
                             f"(+{lots_lift:.4f}), NOT a leakage rejection."),
            "marginal_test_auc_lift": lots_lift,
            "marginal_lift_note": (
                "FLAG (B3): tender_lotscount and buyer_region BOTH store marginal lift "
                f"+{lots_lift:.5f}; tender_lotscount has no per-fold evidence and "
                "buyer_region's own per-fold CV lifts average ~0.00575 (!= 0.0013), so "
                "the twin value is UNVERIFIED / likely templated. A precise independent "
                "recompute needs an ablation refit (disallowed this pass: no separate "
                "refits). Substantive conclusion is unchanged: the lift is negligible (~0)."
            ),
            "source": "rq2_feature_exploration_nulls.json",
        },
        {
            "feature": "buyer_concentration_hhi", "stage": "modelling_candidate",
            "disposition_type": "leakage_reject",
            "decisive_metric": "|corr| with buyer_dependency_ratio (a label-rule input)",
            "decisive_value": hhi_corr,
            "gate_tripped": (f"YES -- {hhi_corr:.2f} > {PROXY_GATE} proxy gate; "
                             f"reconstructs the label's dependency trigger."),
            "source": "rq2_hhi_proxy_null.json",
        },
        {
            "feature": "buyer_region", "stage": "clean_feature_exploration",
            "disposition_type": "leakage_reject",
            "decisive_metric": "eta (correlation ratio, categorical->numeric) with "
                               "contract_amendments (a label-rule input)",
            "decisive_value": region_eta,
            "gate_statistic": "eta (correlation ratio) -- NOT Pearson |corr|",
            "gate_tripped": (f"YES (by eta, not Pearson) -- eta {region_eta:.3f} >> "
                             f"{PROXY_GATE}."),
            "mechanism": (
                "Second-order proxy: categorical buyer_region explains 96.6% of "
                "contract_amendments variance (eta=0.966); contract_amendments is itself a "
                "label-rule input, so buyer_region is a proxy-of-a-proxy. Rejected by the "
                "proxy principle, evidenced by eta (categorical association), distinct from "
                "the Pearson-|corr| gate that rejected buyer_concentration_hhi."
            ),
            "marginal_test_auc_lift": region_lift,
            "marginal_lift_note": (
                f"FLAG (B3): stored +{region_lift:.5f} equals tender_lotscount's at 4 dp but "
                "buyer_region's own per_fold_lifts [0.006, 0.013, 0.001, 0.003] average "
                "~0.00575 != 0.0013 -- the headline test-lift is UNVERIFIED / likely "
                "templated. Precise recompute needs an ablation refit (disallowed: no "
                "separate refits). Conclusion unchanged: negligible."
            ),
            "source": "rq2_feature_exploration_nulls.json",
        },
        {
            "feature": "supplier_centrality", "stage": "modelling_candidate",
            "disposition_type": "null_integration",
            "decisive_metric": "standalone 2020 TEST ROC-AUC (RQ1->RQ2 hand-off)",
            "decisive_value": 0.49,
            "gate_tripped": "none -- ~chance alone (~0.49), ~0 marginal; reported null.",
            "source": "documented prior, THRESHOLD_JUSTIFICATION.md Section 10.4",
        },
        {
            "feature": "cpv_enc", "stage": "modelling_candidate",
            "disposition_type": "leakage_reject",
            "decisive_metric": "|corr| ~0.60 with the label (procedure_type proxy); "
                               "arbitrary high-cardinality ordinal encoding",
            "decisive_value": 0.60,
            "gate_tripped": (f"YES -- ~0.60 > {PROXY_GATE} proxy WARN gate; ~0 marginal "
                             f"AUC over contract_value_log."),
            "source": "documented prior, THRESHOLD_JUSTIFICATION.md Section 10.1",
        },
        {
            "feature": "cpv_risk_score", "stage": "modelling_candidate",
            "disposition_type": "leakage_reject",
            "decisive_metric": "|corr| ~0.60 with the label; deterministic function of cpv_enc",
            "decisive_value": 0.60,
            "gate_tripped": (f"YES -- ~0.60 > {PROXY_GATE} proxy WARN gate; redundant "
                             f"procedure_type proxy, ~0 marginal AUC."),
            "source": "documented prior, THRESHOLD_JUSTIFICATION.md Section 10.1",
        },
        {
            "feature": "award_year", "stage": "modelling_candidate",
            "disposition_type": "constant_at_test",
            "decisive_metric": "distinct award_year values within the 2020 TEST split",
            "decisive_value": 1,
            "gate_tripped": "constant (=2020) at test -> zero discriminative variance; "
                            "retained only as the temporal-split key.",
            "source": "TEST split is TEST_YEARS=[2020]",
        },
    ]

    # B1: one consistent definition of "candidate". The literal table has 8 rows;
    # bucket them transparently instead of forcing an unexplained "six".
    n_evaluated = len(candidates)            # 8
    n_kept = sum(1 for c in candidates if c["disposition_type"] == "kept")  # 1
    cpv_pair = ["cpv_enc", "cpv_risk_score"]
    split_key = ["award_year"]
    documented_nulls = ["supplier_centrality", "buyer_concentration_hhi",
                        "buyer_region", "tender_lotscount"]
    # 5 substantive non-value candidate SIGNALS (cpv pair counts once): the 4 nulls
    # plus the cpv procedure-proxy signal.
    substantive_non_value_signals = ["cpv (procedure proxy)"] + documented_nulls
    consistent = (
        n_evaluated == 8 and n_kept == 1
        and len(documented_nulls) == 4
        and len(substantive_non_value_signals) == 5
    )

    reconciliation = {
        "N_evaluated": n_evaluated,
        "canonical_statement": CANONICAL_DISPOSITION_STATEMENT,
        "buckets": {
            "kept_value_feature": ["contract_value_log"],
            "cpv_pair_two_encodings_one_procedure_signal": cpv_pair,
            "temporal_split_key_constant_at_test": split_key,
            "documented_nulls": documented_nulls,
        },
        "counts": {
            "N_evaluated": n_evaluated,
            "N_kept": n_kept,
            "N_documented_nulls": len(documented_nulls),
            "cpv_encodings_of_one_signal": len(cpv_pair),
            "temporal_split_keys": len(split_key),
            "substantive_non_value_candidate_signals": len(substantive_non_value_signals),
            "substantive_non_value_signals_survived": 0,
        },
        "gate_statistic_generalisation": GATE_STATISTIC_NOTE,
        "supersedes": "the earlier unexplained 'N_initial=6 -> N_kept=1' framing; the "
                      "literal table has 8 rows, bucketed transparently above.",
        "authoritative_for": "rq2_report.md Section 6.4 feature drop-ladder",
        "consistent": consistent,
        "flag": ("OK: 8 evaluated reconcile (1 kept value feature; cpv pair = 1 signal; "
                 "award_year = split key; 4 documented nulls; 0 substantive non-value "
                 "candidates survived)."
                 if consistent else
                 f"FLAG: bucketing INCONSISTENT -- N_evaluated={n_evaluated}, N_kept={n_kept}, "
                 f"nulls={len(documented_nulls)}."),
    }

    disposition = {
        "summary": "Disposition of every RQ2 candidate feature evaluated. disposition_type "
                   "in {kept, leakage_reject, redundancy_drop, constant_at_test, "
                   "null_integration}. tender_lotscount is a redundancy_drop (r below the "
                   "0.55 leakage gate), NOT a leakage rejection; buyer_region is rejected by "
                   "eta (correlation ratio), a different statistic from the Pearson |corr| "
                   "gate.",
        "leakage_gate": PROXY_GATE,
        "candidates": candidates,
        "reconciliation": reconciliation,
    }
    write_json(DATA_RESULTS / "rq2_feature_disposition.json", disposition)

    # Markdown twin
    lines = [
        "# RQ2 Feature Candidate Disposition",
        "",
        "Disposition of every candidate feature evaluated for the RQ2 governance-risk "
        "model, with the decisive metric and the gate it tripped. `disposition_type` in "
        "{kept, leakage_reject, redundancy_drop, constant_at_test, null_integration}.",
        "",
        f"Leakage proxy gate: association with a label input by the appropriate statistic "
        f"(Pearson |corr| >= {PROXY_GATE} for numeric; eta for categorical). "
        "`tender_lotscount` sits BELOW this gate (r=%.2f) -- a redundancy drop, not a "
        "leakage rejection. `buyer_region` is rejected by eta=%.3f (categorical "
        "association), a DISTINCT statistic from Pearson |corr|." % (lots_corr, region_eta),
        "",
        "| feature | disposition_type | decisive_metric | value | gate_tripped |",
        "|---|---|---|---|---|",
    ]
    for c in candidates:
        v = c["decisive_value"]
        vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(
            f"| `{c['feature']}` | {c['disposition_type']} | {c['decisive_metric']} "
            f"| {vstr} | {c['gate_tripped']} |"
        )
    lines += [
        "",
        "## Candidate-count reconciliation (authoritative for report Section 6.4)",
        "",
        CANONICAL_DISPOSITION_STATEMENT,
        "",
        f"- **N_evaluated: {n_evaluated}** (the literal table above)",
        f"- kept value feature: **{n_kept}** (`contract_value_log`)",
        f"- cpv pair (two encodings of one procedure signal): {', '.join('`%s`' % f for f in cpv_pair)}",
        f"- temporal-split key (constant at test): `{split_key[0]}`",
        f"- documented nulls (**4**): {', '.join('`%s`' % f for f in documented_nulls)}",
        "- substantive non-value candidate signals: **5** (the 4 nulls + the cpv "
        "procedure proxy), of which **0 survived**",
        "",
        "### Gate statistic (B2)",
        "",
        GATE_STATISTIC_NOTE,
        "",
        "### Marginal-lift caveat (B3)",
        "",
        "`tender_lotscount` and `buyer_region` both store a +0.0013 marginal test-AUC "
        "lift. This is FLAGGED as unverified/likely templated: `tender_lotscount` has no "
        "per-fold evidence and `buyer_region`'s own per-fold CV lifts "
        "[0.006, 0.013, 0.001, 0.003] average ~0.00575 (not 0.0013). A precise independent "
        "recompute requires an ablation refit, disallowed this pass (no separate refits). "
        "The substantive conclusion is unaffected: both lifts are negligible (~0).",
        "",
        f"**{reconciliation['flag']}**",
        "",
    ]
    (DATA_RESULTS / "rq2_feature_disposition.md").write_text("\n".join(lines))
    return True

def _xgb_or_fallback():
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(**XGB_PARAMS, eval_metric="mlogloss")
    except Exception as exc:
        logger.warning("XGBoost unavailable; using GradientBoostingClassifier fallback: %s", exc)
        return GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)

def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    pred = model.predict(X)
    classes = np.unique(pred)
    # Map predictions to contiguous 0-based indices so non-0-based or
    # non-contiguous class labels don't cause an IndexError (HIGH-10).
    label_to_idx = {c: i for i, c in enumerate(classes)}
    safe_idx = np.array([label_to_idx.get(p, 0) for p in pred])
    return np.eye(len(classes))[safe_idx]

def _positive_class(proba):
    if proba.ndim == 1:
        return proba
    return proba[:, -1]

def _forward_chaining_auc(estimator, X, y, years):
    """Forward-chaining temporal CV AUC. For each consecutive training year t,
    fit a fresh clone of ``estimator`` on all rows with award_year <= t and
    validate on award_year == t+1, returning the per-fold positive-class ROC-AUC.
    Folds whose validation slice is empty or single-class are skipped. This is the
    temporally-honest counterpart to the (non-temporal) shuffled K-fold CV."""
    from sklearn.base import clone
    Xv = pd.DataFrame(X).reset_index(drop=True)
    yv = pd.Series(np.asarray(y)).reset_index(drop=True)
    years = np.asarray(years)
    uniq = sorted({int(v) for v in years})
    aucs = []
    for t in uniq[:-1]:
        tr = years <= t
        va = years == (t + 1)
        if va.sum() == 0 or np.unique(yv[tr]).size < 2 or np.unique(yv[va]).size < 2:
            continue
        model = clone(estimator)
        model.fit(Xv[tr], yv[tr])
        proba = _positive_class(model.predict_proba(Xv[va]))
        aucs.append(float(roc_auc_score(yv[va], proba)))
    return aucs

def _auc(y_true, proba):
    try:
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except Exception:
        return 0.0

# --- DeLong AUC-difference test (closed form, deterministic, no resampling) ----
# Implements the fast DeLong covariance estimator (Sun & Xu 2014; DeLong et al.
# (1988) so two ROC-AUCs measured on the SAME 2020 test set can be compared with a
# proper paired variance. Closed-form only no bootstrap  so reruns are
# byte-identical. Diagnostic for T6; it does not alter any model or threshold.
def _delong_midrank(x):
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    mid = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        mid[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = mid
    return out


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, N) array, columns ordered positives-first; m = #positives.
    Returns (aucs[k], covariance[k,k]) using the fast DeLong estimator."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _delong_midrank(pos[r, :])
        ty[r, :] = _delong_midrank(neg[r, :])
        tz[r, :] = _delong_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def _delong_pair(y_true, prob_a, prob_b):
    """Paired DeLong test for AUC(a) - AUC(b) on a shared sample. Returns AUCs,
    difference, standard error, 95% CI and a two-sided normal-approx p-value."""
    from scipy.stats import norm
    y_true = np.asarray(y_true, dtype=int)
    order = (-y_true).argsort(kind="mergesort")   # positives (label 1) first, stable
    m = int(y_true.sum())
    preds = np.vstack((np.asarray(prob_a, dtype=float),
                       np.asarray(prob_b, dtype=float)))[:, order]
    aucs, cov = _fast_delong(preds, m)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    diff = auc_a - auc_b
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0:
        se, z, p = 0.0, 0.0, 1.0
    else:
        se = float(np.sqrt(var))
        z = diff / se
        p = float(2.0 * norm.sf(abs(z)))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_difference": diff,
        "std_error": se,
        "ci95": [diff - 1.96 * se, diff + 1.96 * se],
        "z": float(z),
        "p_value": p,
    }


def _select_threshold_max_f1(y_true, prob):
    """Return the probability threshold that maximises binary F1 on (y_true, prob).
    Used on the VAL split only; the chosen value is frozen onto TEST so that test
    labels never influence threshold selection."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1))])

def _headline_metrics(y_true, prob, thr):
    """ROC-AUC, PR-AUC and threshold-dependent F1/precision/recall + confusion
    matrix for the positive (HIGH-risk) class at the frozen operating threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

def _top_features(model, features):
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "term_importances_"):
        scores = model.term_importances_
    elif hasattr(model, "term_importances"):
        scores = model.term_importances()
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_).mean(axis=0)
    else:
        return []
    idx = np.argsort(scores)[::-1][:3]
    return [features[i] for i in idx]

def run_shap(clf_xgb, X_train: pd.DataFrame, y_full: pd.Series,
             features: list[str], sample_n: int = 4000, seed: int = 42) -> dict:
    """
    Compute SHAP values for the trained XGB model and export plots + HTML to DATA_RESULTS.

    Uses XGBoost's native pred_contribs (mathematically identical to TreeExplainer SHAP
    values) to avoid a version-mismatch issue between shap and xgboost 3.x.  Produces:
      - rq2_shap_summary.png   — beeswarm (SHAP value vs feature value per row)
      - rq2_shap_bar.png       — mean |SHAP| bar chart
      - rq2_shap_global.html   — global ranking table
      - rq2_shap_local.png     — waterfall-style chart for one HIGH-risk contract
      - rq2_shap_local.html    — local explanation table

    Returns a dict with keys: shap_top3_features, shap_sample_n, shap_seed,
    supplier_centrality_shap_rank.
    """
    try:
        import xgboost as xgb_lib
    except ImportError:
        logger.warning("XGBoost not available — skipping SHAP analysis")
        return {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping SHAP plots")
        return {}

    # Build a stratified sample so both classes are represented
    rng  = np.random.default_rng(seed)
    idx0 = y_full.index[y_full == 0].tolist()
    idx1 = y_full.index[y_full == 1].tolist()
    n1   = min(sample_n // 2, len(idx1))
    n0   = sample_n - n1
    if n0 == 0 or n1 == 0:
        logger.warning("SHAP: not enough samples in one class — skipping")
        return {}
    s_idx  = list(rng.choice(idx0, n0, replace=False)) + list(rng.choice(idx1, n1, replace=False))
    X_samp = X_train.loc[s_idx, features]
    y_samp = y_full.loc[s_idx]
    actual_n = len(X_samp)
    logger.info("SHAP: sample_n=%d (seed=%d), class dist: %s", actual_n, seed,
                dict(y_samp.value_counts().sort_index()))

    # Compute SHAP via XGBoost native pred_contribs (equivalent to TreeExplainer)
    booster = clf_xgb.get_booster()
    dmat    = xgb_lib.DMatrix(X_samp, feature_names=features)
    contrib = booster.predict(dmat, pred_contribs=True)   # shape: (n, n_features+1) binary
    if contrib.ndim == 3:
        sv2d = contrib[:, :-1, 1]    # positive class, drop bias column
    else:
        sv2d = contrib[:, :-1]       # drop bias column

    mean_abs = np.abs(sv2d).mean(axis=0)
    ranking  = sorted(zip(features, mean_abs), key=lambda x: x[1], reverse=True)
    top3     = [f for f, _ in ranking[:3]]
    logger.info("SHAP top-3 features: %s", top3)

    # --- Global: beeswarm dot plot ---
    try:
        DATA_RESULTS.mkdir(parents=True, exist_ok=True)
        rng_vis = np.random.default_rng(0)
        cmap = plt.cm.RdBu_r   # type: ignore[attr-defined]
        fig, ax = plt.subplots(figsize=(10, 5))
        feats_s = [f for f, _ in ranking]
        for i, feat in enumerate(feats_s):
            fi      = features.index(feat)
            sv_col  = sv2d[:, fi]
            fv_col  = X_samp[feat].values
            fv_norm = (fv_col - fv_col.min()) / (np.ptp(fv_col) + 1e-9)
            jitter  = rng_vis.uniform(-0.3, 0.3, size=len(sv_col))
            ax.scatter(sv_col, np.full(len(sv_col), len(feats_s) - 1 - i) + jitter,
                       c=fv_norm, cmap=cmap, alpha=0.25, s=6, linewidths=0)
        ax.set_yticks(range(len(feats_s)))
        ax.set_yticklabels(feats_s[::-1])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value  (positive = pushes toward risk_label=1)")
        ax.set_title(f"XGB — SHAP Beeswarm  (n={actual_n}, seed={seed})")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))   # type: ignore[attr-defined]
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Feature value (scaled)", shrink=0.7)
        plt.tight_layout()
        plt.savefig(DATA_RESULTS / "rq2_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Global bar
        vals_s = [float(v) for _, v in ranking]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(feats_s[::-1], vals_s[::-1], color="#2c6fad")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"XGB — Global Feature Importance (SHAP, n={actual_n}, seed={seed})")
        for j, (f, v) in enumerate(zip(feats_s[::-1], vals_s[::-1])):
            ax.text(v + max(vals_s) * 0.01, j, f"{v:.4f}", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(DATA_RESULTS / "rq2_shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Local: waterfall-style for highest-risk contract
        proba_samp = clf_xgb.predict_proba(X_samp)[:, 1]
        hi_idx     = int(np.argmax(proba_samp))
        hi_prob    = float(proba_samp[hi_idx])
        local_sv   = sv2d[hi_idx]
        feat_vals  = X_samp.iloc[hi_idx]
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#d73027" if v > 0 else "#4575b4" for v in local_sv]
        bars = ax.barh(features, local_sv, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"XGB Local Explanation — HIGH-risk contract  (pred prob={hi_prob:.3f})")
        ax.set_xlabel("SHAP value  (positive = pushes toward risk_label=1)")
        for bar, sv_v, feat in zip(bars, local_sv, features):
            fv = feat_vals[feat]
            ax.text(sv_v + (max(np.abs(local_sv)) * 0.02 if sv_v >= 0
                            else -max(np.abs(local_sv)) * 0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"={fv:.2f}", va="center",
                    ha="left" if sv_v >= 0 else "right", fontsize=8)
        plt.tight_layout()
        plt.savefig(DATA_RESULTS / "rq2_shap_local.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Local: waterfall-style for a LOW-risk contract (lowest predicted prob)
        lo_idx   = int(np.argmin(proba_samp))
        lo_prob  = float(proba_samp[lo_idx])
        lo_sv    = sv2d[lo_idx]
        lo_vals  = X_samp.iloc[lo_idx]
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#d73027" if v > 0 else "#4575b4" for v in lo_sv]
        bars = ax.barh(features, lo_sv, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"XGB Local Explanation — LOW-risk contract  (pred prob={lo_prob:.3f})")
        ax.set_xlabel("SHAP value  (positive = pushes toward risk_label=1)")
        for bar, sv_v, feat in zip(bars, lo_sv, features):
            fv = lo_vals[feat]
            ax.text(sv_v + (max(np.abs(lo_sv)) * 0.02 if sv_v >= 0
                            else -max(np.abs(lo_sv)) * 0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"={fv:.2f}", va="center",
                    ha="left" if sv_v >= 0 else "right", fontsize=8)
        plt.tight_layout()
        plt.savefig(DATA_RESULTS / "rq2_shap_local_low.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Dependence plots: SHAP value vs feature value, one per model feature.
        # Coloured by the OTHER feature to surface any interaction.
        for fi_, feat in enumerate(features):
            other = features[1 - fi_] if len(features) == 2 else None
            fig, ax = plt.subplots(figsize=(7, 4))
            fv_col = X_samp[feat].values
            sv_col = sv2d[:, features.index(feat)]
            if other is not None:
                c = X_samp[other].values
                sc = ax.scatter(fv_col, sv_col, c=c, cmap="viridis", s=8, alpha=0.4, linewidths=0)
                plt.colorbar(sc, ax=ax, label=f"{other} (colour)", shrink=0.8)
            else:
                ax.scatter(fv_col, sv_col, color="#2c6fad", s=8, alpha=0.4, linewidths=0)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xlabel(f"{feat} value")
            ax.set_ylabel(f"SHAP value for {feat}")
            ax.set_title(f"XGB — SHAP Dependence: {feat}  (n={actual_n}, seed={seed})")
            plt.tight_layout()
            plt.savefig(DATA_RESULTS / f"rq2_shap_dependence_{feat}.png", dpi=150, bbox_inches="tight")
            plt.close("all")

        logger.info("SHAP plots saved to %s", DATA_RESULTS)
    except Exception as plot_exc:
        logger.warning("SHAP plot export failed: %s", plot_exc)

    # HTML exports
    try:
        xgb_fi = dict(zip(features, clf_xgb.feature_importances_))
        shap_table = pd.DataFrame({
            "feature":       feats_s,
            "mean_abs_shap": [float(v) for _, v in ranking],
            "native_fi":     [float(xgb_fi.get(f, 0)) for f in feats_s],
        })
        (DATA_RESULTS / "rq2_shap_global.html").write_text(
            "<html><head><title>SHAP Global</title>"
            "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:6px 12px}</style></head><body>"
            f"<h2>XGB SHAP — Global Feature Ranking (n={actual_n}, seed={seed})</h2>"
            "<p>Computed via XGBoost native pred_contribs. "
            "Column <em>native_fi</em> is the model's gain-based feature importance for comparison.</p>"
            f"{shap_table.to_html(index=False, float_format='%.5f')}"
            "</body></html>"
        )
        proba_samp = clf_xgb.predict_proba(X_samp)[:, 1]
        hi_idx = int(np.argmax(proba_samp))
        hi_prob = float(proba_samp[hi_idx])
        feat_vals = X_samp.iloc[hi_idx]
        local_sv  = sv2d[hi_idx]
        local_df = pd.DataFrame({
            "feature":       features,
            "feature_value": [float(feat_vals[f]) for f in features],
            "shap_value":    list(local_sv),
        }).sort_values("shap_value", ascending=False)
        (DATA_RESULTS / "rq2_shap_local.html").write_text(
            "<html><head><title>SHAP Local</title>"
            "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:6px 12px}</style></head><body>"
            f"<h2>XGB SHAP — Local Explanation (pred risk_prob={hi_prob:.4f})</h2>"
            "<p>Red (positive SHAP) = pushes toward HIGH risk. "
            "Blue (negative) = pushes toward LOW risk.</p>"
            f"{local_df.to_html(index=False, float_format='%.5f')}"
            "</body></html>"
        )
        logger.info("SHAP HTML exports saved.")
    except Exception as html_exc:
        logger.warning("SHAP HTML export failed: %s", html_exc)

    sc_rank = next((i + 1 for i, (f, _) in enumerate(ranking) if f == "supplier_centrality"), -1)
    if sc_rank > 0:
        logger.info("supplier_centrality SHAP rank: #%d -- network supplier-centrality does "
                    "not meaningfully predict governance risk (reported null, not a validated "
                    "RQ1->RQ2 hand-off; see Probe A integration check)", sc_rank)

    # Cross-check: native FI vs SHAP ranking agreement
    if hasattr(clf_xgb, "feature_importances_"):
        xgb_fi = dict(zip(features, clf_xgb.feature_importances_))
        native_top3 = {f for f, _ in sorted(xgb_fi.items(), key=lambda x: x[1], reverse=True)[:3]}
        shap_top3   = set(top3)
        logger.info("SHAP/native top-3 overlap: %d/3 (native=%s, shap=%s)",
                    len(native_top3 & shap_top3), sorted(native_top3), top3)

    return {
        "shap_top3_features":              top3,
        "shap_sample_n":                   actual_n,
        "shap_seed":                       seed,
        "supplier_centrality_shap_rank":   sc_rank,
        "shap_method":                     "xgb_pred_contribs",
    }


if __name__ == "__main__":
    run_rq2()
