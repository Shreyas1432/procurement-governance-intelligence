"""
RQ2 - Governance Risk. Read-only over locked RQ2 parquet/JSON/PNG artifacts.

The leakage audit, the model, and the temporal-drift honesty check. Per the
hard constraints: the drop from shuffled-CV ROC-AUC to out-of-time ROC-AUC
(0.612) is framed as a finding that reveals drift in the underlying process,
not as something the pipeline prevents. Named per-contract risk data sits
behind the admin-only, logged reveal pattern.

Data note: rq2_governance_predictions.parquet contains exact duplicate rows
baked into the locked artifact (verified upstream data quality issue, not a
dashboard bug). The per-contract table below de-duplicates for display and
discloses the gap against the locked headline count rather than hiding it.
See _data_hygiene.py for details.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import require_auth, render_logout_button, anonymize_id, log_action, PUBLIC_BUILD, DASHBOARD_DATA_RESULTS, DASHBOARD_DATA_FEATURES
from dashboard._theme import inject_base_css, INK, INK_MUTED, FONT_SERIF, RUST, RUST_DARK, RUST_TINT
from dashboard._components import (
    small_stat_card, methodology_callout, neutral_methodology_note,
    chart_card, why_this_matters, status_pill, risk_label_definition_banner,
)
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher
from dashboard._data_hygiene import dedupe_for_display, dedup_disclosure_note, safe_id

inject_base_css()
require_auth()
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / RQ2 Governance Risk")

role = st.session_state.get("role")


@st.cache_data(show_spinner="Loading governance risk artifacts...")
def load_rq2():
    base = DASHBOARD_DATA_RESULTS
    return {
        "success": json.loads((base / "rq2_success_metrics.json").read_text()),
        "disposition": json.loads((base / "rq2_feature_disposition.json").read_text()),
        "ci": json.loads((base / "model_performance_ci.json").read_text()),
        "predictions": pl.read_parquet(base / "rq2_governance_predictions.parquet"),
    }


@st.cache_data(show_spinner="Loading contract-value curve position...")
def load_value_curve():
    """Read-only, descriptive-only lookup of each contract's contract_value_log
    (the model's one feature) and its percentile rank, joined from the locked
    feature file onto the locked predictions file. No .fit()/.predict(), no
    recomputation of the model itself: this only reads an existing column and
    computes a rank/decile over it, the same kind of descriptive statistic as
    a dataframe .describe() or .rank(). Both source tables are de-duplicated
    on contract_id first (each has the same upstream full-row-duplication
    issue documented in _data_hygiene.py); joining before de-duplicating would
    fan out combinatorially on the repeated keys.

    Returns (value_lookup: dict[contract_id, float], deciles: list[float],
    decile_mean_risk: list[float]) so the per-contract explanation can state
    both a percentile and whether that percentile sits in a locked, observed
    high-flag-rate region of the curve, without ever re-fitting anything.
    """
    base = DASHBOARD_DATA_RESULTS
    features_path = DASHBOARD_DATA_FEATURES / "rq2_governance_features.parquet"
    feat = pl.read_parquet(features_path).select(["contract_id", "contract_value_log"])
    feat_u = feat.unique(subset=["contract_id"], keep="first")

    pred = pl.read_parquet(base / "rq2_governance_predictions.parquet").select(
        ["contract_id", "ensemble_risk_prob"]
    )
    pred_u = pred.unique(subset=["contract_id"], keep="first")

    joined = pred_u.join(feat_u, on="contract_id", how="inner")
    vals = joined["contract_value_log"].to_numpy()
    risk = joined["ensemble_risk_prob"].to_numpy()

    import numpy as np
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(vals) + 1)
    percentiles = ranks / len(vals) * 100.0
    value_lookup = dict(zip(joined["contract_id"].to_list(), joined["contract_value_log"].to_list()))
    percentile_lookup = dict(zip(joined["contract_id"].to_list(), percentiles.tolist()))

    n_buckets = 10
    edges = np.quantile(vals, np.linspace(0, 1, n_buckets + 1))
    decile_mean_risk = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (vals >= lo) & (vals <= hi) if i == n_buckets - 1 else (vals >= lo) & (vals < hi)
        decile_mean_risk.append(float(risk[mask].mean()) if mask.any() else None)

    return value_lookup, percentile_lookup, edges.tolist(), decile_mean_risk, joined.height


d = load_rq2()
success = d["success"]
disposition = d["disposition"]
ci = d["ci"]
predictions_raw = d["predictions"]
headline = success["test_headline"]
classifier_pop = success["populations"]["classifier_population"]

# De-duplicate once for every use of the per-contract table on this page.
predictions, n_before, n_after = dedupe_for_display(predictions_raw, key_cols=None)

value_lookup, percentile_lookup, value_deciles, decile_mean_risk, n_value_matched = load_value_curve()
_valid_deciles = [m for m in decile_mean_risk if m is not None]
_overall_mean_risk = sum(_valid_deciles) / len(_valid_deciles) if _valid_deciles else 0.0
_high_flag_decile_idx = max(range(len(decile_mean_risk)), key=lambda i: (decile_mean_risk[i] if decile_mean_risk[i] is not None else -1))

# hero
st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:32px;color:{INK};margin:0 0 8px;\">"
    f"RQ2: Governance Risk</h1>",
    unsafe_allow_html=True,
)
risk_label_definition_banner()

# leakage audit hero
recon = disposition["reconciliation"]
n_evaluated_documented = recon["counts"]["N_evaluated"]  # 8 individually documented
n_rolling_temporal = 5  # grouped, never individually named (see docs/cross_integration_diffs.md)
n_total_candidates = n_evaluated_documented + n_rolling_temporal  # 13

methodology_callout(
    "The leakage audit",
    "Thirteen candidate features were checked against the risk label",
    f"Twelve were ruled out for temporal, leakage, integration, redundancy, "
    f"or constancy reasons. One survived: <code>contract_value_log</code>. "
    f"Eight candidates are documented individually below. The remaining five "
    f"are a grouped batch of pre-model rolling-temporal signals that added "
    f"no independent lift over the value-only model.",
)

b1, b2, b3 = st.columns(3)
with b1:
    st.markdown(status_pill("1 retained", "positive"), unsafe_allow_html=True)
with b2:
    st.markdown(status_pill("12 reasoned nulls", "neutral"), unsafe_allow_html=True)
with b3:
    st.markdown(status_pill("4 direct leakage", "caution"), unsafe_allow_html=True)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

audit_rows = []
for cand in disposition["candidates"]:
    audit_rows.append({
        "Feature": cand["feature"],
        "Disposition": cand["disposition_type"],
        "Decisive metric": cand["decisive_metric"],
        "Value": cand["decisive_value"],
        "Reasoning": cand.get("note") or cand.get("mechanism") or cand.get("gate_tripped", ""),
    })
audit_rows.append({
    "Feature": "(5 rolling-temporal signals, grouped)",
    "Disposition": "not_retained",
    "Decisive metric": "marginal lift over contract_value_log",
    "Value": None,
    "Reasoning": "Pre-model velocity and rate features, excluded for no independent lift. Not individually named in the disposition ledger (see docs/cross_integration_diffs.md).",
})
# Explicit height (Task 5): all 13 rows reachable by scrolling, without a
# cell click first enabling scroll. 9 rows fit before scroll kicks in, at
# roughly 35px per row plus header; the table itself scrolls once content
# exceeds this height.
st.dataframe(
    audit_rows, width="stretch", height=360, hide_index=True,
    key="rq2_leakage_audit_table",
    column_config={
        "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
    },
)
why_this_matters(
    f"{n_total_candidates} candidates checked (8 documented individually plus 5 grouped rolling-temporal signals). "
    f"1 retained, 12 reasoned nulls. Scroll the table above to see all {len(audit_rows)} rows."
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# model performance
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>Model performance</div>",
    unsafe_allow_html=True,
)
m1, m2, m3, m4 = st.columns(4)
with m1:
    small_stat_card(f"{headline['roc_auc']:.4f}", "ROC-AUC", "XGBoost, 2020 held-out test")
with m2:
    small_stat_card(f"{headline['pr_auc']:.4f}", "PR-AUC", "average_precision_score")
with m3:
    small_stat_card(f"{headline['operating_threshold']:.3f}", "Operating threshold", "argmax F1 on VAL 2019, frozen")
with m4:
    small_stat_card(f"{headline['f1']:.4f}", "F1 at threshold", f"Precision {headline['precision']:.3f}, recall {headline['recall']:.3f}")

st.caption(
    f"Baselines on the same task: Logistic Regression ROC-AUC {success['lr_auc']:.4f}. "
    f"Random Forest ROC-AUC {success['rf_auc']:.4f}."
)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

cm1, cm2 = st.columns([1, 1])
with cm1:
    chart_card("Confusion matrix", f"At the frozen operating threshold ({headline['operating_threshold']:.3f}).")
    cm = headline["confusion_matrix"]
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=["Predicted low risk", "Predicted high risk"], y=["Actual low risk", "Actual high risk"],
        text=z, texttemplate="%{text:,}", colorscale=[[0, "#F3F1EC"], [1, RUST]],
        showscale=False,
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch", key="rq2_confusion_matrix")
with cm2:
    chart_card("Risk score by contract value", "Binned mean predicted risk against contract value, the model's only feature.")
    img_path = DASHBOARD_DATA_RESULTS / "rq2_value_only_pdp.png"
    if img_path.exists():
        st.image(str(img_path), width="stretch")
        why_this_matters("Locked partial dependence plot from the pipeline run, not recomputed here. The relationship is not a straight line: risk does not simply rise or fall with value.")
    else:
        st.info("Partial dependence image not found in locked artifacts.")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# temporal validation
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>Temporal validation</div>",
    unsafe_allow_html=True,
)
xgb_shuffled = ci.get("rq2_xgb_auc_cv_shuffled", ci.get("rq2_xgb_auc", {}))
xgb_temporal = ci.get("rq2_xgb_auc_cv_temporal", {})

tc1, tc2 = st.columns(2)
with tc1:
    small_stat_card(
        f"{xgb_shuffled.get('mean', 0):.3f} (plus or minus {xgb_shuffled.get('std', 0):.3f})",
        "Shuffled-split CV ROC-AUC", "Standard cross-validation, non-temporal",
    )
with tc2:
    small_stat_card(
        f"{xgb_temporal.get('mean', 0):.3f} (plus or minus {xgb_temporal.get('std', 0):.3f})",
        "Out-of-time CV ROC-AUC", "Forward-chaining temporal split",
    )

st.markdown(
    f"""
    <div style="background:{RUST_TINT};border-left:4px solid {RUST};border-radius:8px;
                padding:20px 24px;margin-top:12px;">
      <div style="font-size:14.5px;line-height:1.65;color:{INK};">
        Cross-validated on a strict out-of-time split, mean ROC-AUC drops to
        <strong>{xgb_temporal.get('mean', 0):.3f} (plus or minus {xgb_temporal.get('std', 0):.3f})</strong>,
        well below the shuffled-split figure. That gap is the finding. It
        reveals drift in the underlying process. The pipeline reports this
        honestly instead of only the more flattering shuffled number. It does
        not claim to prevent the drift.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# per-contract table
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:4px;'>Per-contract risk</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div style="background:{RUST_TINT};border-left:4px solid {RUST};border-radius:8px;
                padding:16px 20px;margin-bottom:14px;">
      <div style="font-size:14px;line-height:1.6;color:{INK};">
        <strong>This is a deliberately value-only model.</strong> After the leakage audit above,
        <code>contract_value_log</code> is the model's only feature. Every score below is
        driven by contract value alone, not by procedure type, bid count, or amendments.
        The "why flagged" column states where each contract's value sits on the observed
        value-to-risk curve, since that is the only visible basis this model has for its score.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"{dedup_disclosure_note(n_before, n_after, 'high-risk contract count', success['n_high_risk'])}"
)
st.caption(
    f"This is the rgov_scored_population (out-of-fold train plus held-out test rows; the "
    f"validation split is excluded because it is used only to pick the operating threshold). "
    f"It is a different population from the classifier_population "
    f"({classifier_pop['n_total']:,} rows) used to fit and evaluate the model above, and the "
    f"two counts should not be added together."
)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    risk_filter = st.selectbox("Filter by risk class", ["All", "HIGH", "MEDIUM", "LOW"], key="rq2_risk_filter")
with fc2:
    buyer_search = st.text_input("Filter by buyer ID (admin only, contains)", key="rq2_buyer_filter", disabled=(role != "ADMIN"))
with fc3:
    supplier_search = st.text_input("Filter by supplier ID (admin only, contains)", key="rq2_supplier_filter", disabled=(role != "ADMIN"))

filtered = predictions if risk_filter == "All" else predictions.filter(pl.col("risk_class") == risk_filter)
if role == "ADMIN" and buyer_search:
    filtered = filtered.filter(pl.col("buyer_id").cast(pl.Utf8).str.contains(buyer_search, literal=True))
if role == "ADMIN" and supplier_search:
    filtered = filtered.filter(pl.col("supplier_id").cast(pl.Utf8).str.contains(supplier_search, literal=True))

top = filtered.sort("ensemble_risk_prob", descending=True).head(300).to_pandas()


def _value_on_curve_sentence(contract_id) -> str:
    """Honest, locked-data basis for the score: where this contract's
    contract_value_log (the model's ONLY feature) sits on the observed
    value -> mean-predicted-risk curve. Deciles and per-decile mean risk are
    computed once, read-only, in load_value_curve() above (a descriptive
    statistic over an existing column, not a re-fit). Never invents a
    percentile for a contract missing from the joined value lookup (a small
    minority, since the feature and prediction files use slightly different
    row sets after de-duplication on contract_id)."""
    pct = percentile_lookup.get(contract_id)
    if pct is None:
        return "This contract's position on the value curve could not be located in the locked feature file (no matching contract_id after de-duplication)."
    decile_idx = min(int(pct // 10), len(decile_mean_risk) - 1)
    this_decile_mean = decile_mean_risk[decile_idx]
    region = "upper" if decile_idx >= 8 else ("lower" if decile_idx <= 1 else "middle")
    is_high_flag_region = decile_idx == _high_flag_decile_idx
    sentence = (
        f"This contract's log-value is in the {pct:.0f}th percentile of the "
        f"{n_value_matched:,}-contract value distribution, in the {region} region of the "
        f"value -> risk curve."
    )
    if this_decile_mean is not None:
        if is_high_flag_region:
            sentence += (
                f" This is the decile with the highest observed mean flag rate "
                f"({this_decile_mean*100:.1f}%, versus {_overall_mean_risk*100:.1f}% "
                f"averaged across all deciles)."
            )
        elif this_decile_mean > _overall_mean_risk:
            sentence += f" Mean flag rate in this decile is {this_decile_mean*100:.1f}%, above the {_overall_mean_risk*100:.1f}% cross-decile average."
        else:
            sentence += f" Mean flag rate in this decile is {this_decile_mean*100:.1f}%, at or below the {_overall_mean_risk*100:.1f}% cross-decile average."
    return sentence


def _why_flagged(row) -> str:
    """Row-level explanation with two parts, both grounded in locked
    artifacts and neither one fabricated:

    1. Where this contract's value sits on the observed value -> risk curve
       (the model's ONLY feature is contract_value_log, so this is the
       visible basis for every score on this page, always present).
    2. Rule-label contributing factors, where they happen to exist for this
       row (procedure type, low bid count, amendments). These are NOT the
       model's features; they are context from the deterministic risk_label
       rule definition, shown only when actually present on the row, and
       explicitly stated as absent otherwise rather than papering over the
       value-only model with invented reasons."""
    curve_sentence = _value_on_curve_sentence(row.get("contract_id"))

    bits = []
    proc = row.get("procedure_type")
    if proc and str(proc).upper() not in ("UNKNOWN", "OPEN", ""):
        bits.append(f"procedure type is {proc}")
    bid_count = row.get("bid_count")
    if bid_count is not None and bid_count <= 1:
        bits.append("one bidder or fewer on record")
    amendments = row.get("contract_amendments")
    if amendments is not None and amendments > 0:
        bits.append(f"{int(amendments)} contract amendment(s) on record")

    if bits:
        factor_sentence = "Rule-label contributing factors also present on this row: " + ", ".join(bits) + "."
    else:
        factor_sentence = (
            "No rule-label contributing factors (procedure type, low bid count, amendments) "
            "are present on this row; the value-on-curve position above is the only visible basis for the score."
        )
    return curve_sentence + " " + factor_sentence


top["why_flagged"] = top.apply(_why_flagged, axis=1)
top["buyer_id_display"] = top["buyer_id"].apply(safe_id)
top["supplier_id_display"] = top["supplier_id"].apply(safe_id)

if role == "ADMIN" and not PUBLIC_BUILD:
    st.caption("Reveal is admin only. Every reveal is logged.")
    show_ids = st.checkbox("Reveal named buyer and supplier IDs for this table (logs one entry)", key="rq2_reveal_table")
    if show_ids:
        log_action("ADMIN", "Reveal named entity", "RQ2 per-contract table (bulk)", "success")
        top["buyer_id_display"] = top["buyer_id"].apply(safe_id)
        top["supplier_id_display"] = top["supplier_id"].apply(safe_id)
    else:
        top["buyer_id_display"] = "(reveal to view)"
        top["supplier_id_display"] = "(reveal to view)"
else:
    top["buyer_id_display"] = top["buyer_id"].apply(lambda x: anonymize_id(x, "B"))
    top["supplier_id_display"] = top["supplier_id"].apply(lambda x: anonymize_id(x, "S"))

st.dataframe(
    top[["contract_id", "buyer_id_display", "supplier_id_display", "cpv_division", "procedure_type",
         "award_year", "ensemble_risk_prob", "risk_class", "why_flagged"]],
    width="stretch", height=420, hide_index=True, key="rq2_per_contract_table",
    column_config={
        "buyer_id_display": st.column_config.TextColumn("Buyer ID", width="medium"),
        "supplier_id_display": st.column_config.TextColumn("Supplier ID", width="medium"),
        "why_flagged": st.column_config.TextColumn("Why flagged", width="large"),
    },
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
neutral_methodology_note(
    "Method choice: constrained XGBoost, not deep learning",
    "A constrained XGBoost (max depth 3) was chosen over neural networks for "
    "auditability. The single-feature model is fully interpretable, "
    "deterministic with a fixed seed, and its partial dependence behavior "
    "can be inspected directly rather than approximated after the fact.",
)
