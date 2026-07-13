"""
RQ3 - Price Variance. Read-only over locked RQ3 parquet/JSON artifacts.

Ensemble consensus (IsolationForest, LOF, EllipticEnvelope; a flag requires
agreement from at least 2 of 3) across CPV categories. Consensus rate is the
primary gate; CV R-squared and RMSE are diagnostic only. CPV drill-down uses
a full-text reason column that is never truncated.

Data note: rq3_anomaly_flags.parquet contains exact duplicate rows baked
into the locked artifact (verified upstream data quality issue, not a
dashboard bug). The anomaly table below de-duplicates for display and
discloses the gap against the locked consensus count rather than hiding it.
See _data_hygiene.py for details.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import require_auth, render_logout_button, anonymize_id, log_action, PUBLIC_BUILD
from dashboard._theme import inject_base_css, INK, INK_MUTED, FONT_SERIF, PRIMARY, AMBER, SAGE
from dashboard._components import small_stat_card, neutral_methodology_note, chart_card, why_this_matters
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher
from dashboard._data_hygiene import dedupe_for_display, dedup_disclosure_note, safe_id

inject_base_css()
require_auth()
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / RQ3 Price Variance")

role = st.session_state.get("role")


def _reason_text(row: dict, sd: float) -> str:
    """Plain-language, row-consistent audit trail. iso_flag / lof_flag /
    ee_flag are the 3-detector ensemble that decides consensus_flag.
    gb_flag / gb_residual are a separate diagnostic channel (a price
    prediction residual), not part of the consensus vote. The wording only
    claims a price deviation when the row's own prices actually differ, so
    a row where final_price equals estimated_price never gets a price-
    deviation sentence, even if other detectors fired on non-price
    features (procedure type, bid count, CPV mix)."""
    final_price = row.get("final_price")
    estimated_price = row.get("estimated_price")
    same_price = (
        final_price is not None and estimated_price is not None
        and abs(float(final_price) - float(estimated_price)) < 0.01
    )

    reasons = []
    if row.get("iso_flag"):
        reasons.append("IsolationForest: this contract's combination of procedure type, bid count, contract value, and CPV category is atypical compared to similar contracts.")
    if row.get("lof_flag"):
        if same_price:
            reasons.append("Local Outlier Factor: this contract sits in an unusual position relative to its closest neighbors in feature space, based on non-price features since the final price matches the estimate here.")
        else:
            reasons.append("Local Outlier Factor: this contract is priced unusually relative to its closest neighbors in feature space.")
    if row.get("ee_flag"):
        reasons.append("Elliptic Envelope: this contract falls outside the robust-covariance ellipse fit to typical contracts in this segment.")
    if row.get("gb_flag"):
        z = (row.get("gb_residual", 0.0) or 0.0) / (sd or 1.0)
        if same_price:
            reasons.append(f"Diagnostic note: the Gradient Boosting price model flags a residual of {z:+.1f} standard deviations for this row, though the final price equals the estimate, so this reflects the model's internal calculation rather than an observed price gap.")
        else:
            reasons.append(f"Diagnostic: Gradient Boosting price residual is {z:+.1f} standard deviations from what this CPV and procedure type would predict. This diagnostic is not part of the consensus vote.")

    n_consensus_methods = int(bool(row.get("iso_flag"))) + int(bool(row.get("lof_flag"))) + int(bool(row.get("ee_flag")))
    prefix = f"Flagged by {n_consensus_methods} of 3 consensus detectors. "
    if same_price:
        prefix += "Final price matches the estimate for this contract; any flag here is based on non-price features. "
    if not reasons:
        return prefix + "No individual detector reason recorded for this row."
    return prefix + " ".join(reasons)


@st.cache_data(show_spinner="Loading price variance artifacts...")
def load_rq3():
    base = ROOT / "data" / "results"
    anomalies = pl.read_parquet(base / "rq3_anomaly_flags.parquet")
    success = json.loads((base / "rq3_success_metrics.json").read_text())
    ci = json.loads((base / "model_performance_ci.json").read_text())
    return anomalies, success, ci


anomalies_raw, success, ci = load_rq3()
anomalies, n_before, n_after = dedupe_for_display(anomalies_raw)

# hero
st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:32px;color:{INK};margin:0 0 8px;\">"
    f"RQ3: Price Variance</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-size:14.5px;color:{INK_MUTED};margin-bottom:20px;max-width:760px;'>"
    f"Three independent, white-box anomaly detectors each vote on whether a contract's "
    f"price and configuration look unusual. A flag needs agreement from at least 2 of 3. "
    f"The consensus rate is the primary gate; R-squared and RMSE below are diagnostic only."
    f"</div>",
    unsafe_allow_html=True,
)

# ensemble explainer
neutral_methodology_note(
    "Ensemble consensus",
    "IsolationForest, Local Outlier Factor, and Elliptic Envelope (robust covariance) "
    "each flag price outliers using a different statistical idea of what counts as "
    "unusual. A flag needs agreement from at least 2 of 3. This favors precision "
    "(fewer false positives) over recall, which is the right trade-off when the "
    "output feeds a limited audit capacity.",
)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    small_stat_card(
        f"{success['consensus_anomaly_rate']*100:.2f}%", "Consensus rate (primary gate)",
        f"{success['consensus_anomaly_count']:,} of {success['total_contracts']:,} contract rows in the locked artifact",
    )
with m2:
    gb_r2 = ci.get("rq3_gbr_r2", {})
    small_stat_card(
        f"{success['gb_r2']:.4f}", "CV R-squared (diagnostic only)",
        f"95% CI [{gb_r2.get('ci95', [0,0])[0]:.3f}, {gb_r2.get('ci95', [0,0])[1]:.3f}]" if gb_r2 else "",
    )
with m3:
    small_stat_card(
        f"{success['gb_rmse']:.3f}", "RMSE (diagnostic only)",
        f"versus baseline {success['baseline_rmse']:.3f} ({success['rmse_reduction']*100:.1f}% lower)",
    )

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# agreement visual
chart_card("Detector agreement", "How many of the 3 consensus detectors flagged each row, after removing duplicate rows.")
flag_counts = (
    anomalies.select(
        (pl.col("iso_flag").cast(pl.Int8) + pl.col("lof_flag").cast(pl.Int8) + pl.col("ee_flag").cast(pl.Int8)).alias("n_flags")
    )["n_flags"].value_counts().sort("n_flags")
)
pdf = flag_counts.to_pandas()
pdf["n_flags"] = pdf["n_flags"].astype(str) + " of 3"
fig = px.bar(
    pdf, x="n_flags", y="count", color="n_flags",
    color_discrete_sequence=["#D8D5CC", AMBER, AMBER, SAGE],
    labels={"n_flags": "Detectors agreeing", "count": "Rows"},
)
fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white", height=320)
st.plotly_chart(fig, width="stretch", key="rq3_agreement_bar")
why_this_matters("2 or more of 3 is the audit-relevant threshold. A single-detector flag is lower confidence and is not treated as a consensus anomaly.")

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# CPV drill-down
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>CPV category drill-down</div>",
    unsafe_allow_html=True,
)
all_cpv = sorted([c for c in anomalies["cpv_division"].drop_nulls().unique().to_list() if c not in ("UNKNOWN", "")])
cpv_pick = st.multiselect("CPV division", all_cpv, default=all_cpv[:5] if len(all_cpv) >= 5 else all_cpv, key="rq3_cpv_pick")
consensus_only = st.checkbox("Consensus anomalies only", value=False, key="rq3_consensus_toggle")

df = anomalies.filter(pl.col("cpv_division").is_in(cpv_pick)) if cpv_pick else anomalies
if consensus_only:
    df = df.filter(pl.col("consensus_flag"))

sc1, sc2 = st.columns([2, 1])
with sc1:
    chart_card("Price versus estimate", "Final price against estimated price, colored by consensus flag.")
    if df.height > 0:
        pdf2 = df.to_pandas()
        fig2 = px.scatter(
            pdf2, x="estimated_price", y="final_price", color="consensus_flag",
            color_discrete_map={True: "#C85A54", False: "#B7B3A9"},
            opacity=0.5, labels={"estimated_price": "Estimated price", "final_price": "Final price", "consensus_flag": "Consensus"},
        )
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380)
        st.plotly_chart(fig2, width="stretch", key="rq3_scatter")
    else:
        st.info("No rows match the current filters.")
with sc2:
    chart_card("Anomaly rate by CPV", "")
    if df.height > 0:
        cpv_rate = (
            df.group_by("cpv_division")
            .agg(pl.len().alias("total"), pl.sum("consensus_flag").alias("consensus"))
            .with_columns((pl.col("consensus") / pl.col("total")).alias("rate"))
            .sort("consensus", descending=True)
            .head(10)
            .to_pandas()
        )
        fig3 = px.bar(cpv_rate, x="rate", y="cpv_division", orientation="h",
                      labels={"rate": "Consensus rate", "cpv_division": "CPV"},
                      color="rate", color_continuous_scale=[[0, "#EEF2EA"], [1, "#C85A54"]])
        fig3.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white", height=380)
        st.plotly_chart(fig3, width="stretch", key="rq3_cpv_rate_bar")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# anomaly table
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:4px;'>Anomaly audit trail</div>",
    unsafe_allow_html=True,
)
st.caption(dedup_disclosure_note(n_before, n_after, "consensus anomaly count", success["consensus_anomaly_count"]))
st.caption("Full-text reason column. Widen the column or use the cell's expand control to read the full explanation.")

if df.height > 0:
    sd = float(df["gb_residual"].drop_nulls().std() or 1.0) if "gb_residual" in df.columns else 1.0
    table_pdf = df.sort("anomaly_score", descending=True).head(300).to_pandas()
    table_pdf["reason_text"] = table_pdf.apply(lambda r: _reason_text(r.to_dict(), sd), axis=1)
    table_pdf["n_detectors_agreeing"] = (
        table_pdf["iso_flag"].astype(bool).astype(int)
        + table_pdf["lof_flag"].astype(bool).astype(int)
        + table_pdf["ee_flag"].astype(bool).astype(int)
    )
    table_pdf["buyer_id"] = table_pdf["buyer_id"].apply(safe_id)
    table_pdf["supplier_id"] = table_pdf["supplier_id"].apply(safe_id)

    if role == "ADMIN" and not PUBLIC_BUILD:
        reveal_ids = st.checkbox("Reveal named buyer and supplier IDs (logs one entry)", key="rq3_reveal_table")
        if reveal_ids:
            log_action("ADMIN", "Reveal named entity", "RQ3 anomaly table (bulk)", "success")
        else:
            table_pdf["buyer_id"] = "(reveal to view)"
            table_pdf["supplier_id"] = "(reveal to view)"
    else:
        table_pdf["buyer_id"] = table_pdf["buyer_id"].apply(lambda x: anonymize_id(x, "B"))
        table_pdf["supplier_id"] = table_pdf["supplier_id"].apply(lambda x: anonymize_id(x, "S"))

    st.dataframe(
        table_pdf[["contract_id", "buyer_id", "supplier_id", "cpv_division", "award_year",
                    "final_price", "estimated_price", "n_detectors_agreeing", "consensus_flag", "reason_text"]],
        width="stretch", height=420, hide_index=True, key="rq3_anomaly_table",
        column_config={
            "reason_text": st.column_config.TextColumn("Reason (why flagged)", width="large"),
            "final_price": st.column_config.NumberColumn("Final price", format="%.0f"),
            "estimated_price": st.column_config.NumberColumn("Estimated price", format="%.0f"),
            "n_detectors_agreeing": st.column_config.NumberColumn("Detectors agreeing (of 3)", width="small"),
            "consensus_flag": st.column_config.CheckboxColumn("Consensus (2+ of 3)"),
        },
    )
else:
    st.info("No rows match the current filters.")

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
neutral_methodology_note(
    "Why three detectors instead of one",
    "IsolationForest (tree-based isolation), Local Outlier Factor (local density), "
    "and Elliptic Envelope (robust covariance) each rest on a different geometric "
    "assumption about what counts as unusual. Requiring agreement from 2 of 3 is "
    "more conservative than any single method, which reduces false positives that "
    "would otherwise use up audit capacity on one model's idiosyncrasies.",
)
