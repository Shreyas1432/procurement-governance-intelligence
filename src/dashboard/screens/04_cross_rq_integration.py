"""
Cross-RQ Integration. Read-only over locked integration_metrics.json /
cross_effect_sizes.json / unified_risk_scores.parquet.

r = -0.011 (buyer_dependency_normalized vs ensemble_risk_prob), n=594,476.
"Network dependency and procedural risk are independent axes." Community x
risk ANOVA/Chi-sq. RQ2<->RQ3 correlation is degenerate_low_coverage: never
render as 0.0 or as the observed 0.739 without the coverage caveat. The
594,476 integration universe must never be merged with the 714,681 RQ2
risk-label population (that number belongs to the RQ2 screen only).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import polars as pl
import plotly.express as px
import streamlit as st
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import require_auth, render_logout_button
from dashboard._theme import inject_base_css, INK, INK_MUTED, FONT_SERIF, RUST, RUST_TINT, RUST_DARK
from dashboard._components import small_stat_card, neutral_methodology_note, chart_card, why_this_matters, status_pill
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher

inject_base_css()
require_auth()
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / Cross-RQ Integration")


@st.cache_data(show_spinner="Loading integration artifacts...")
def load_integration():
    base = ROOT / "data" / "results"
    integ = json.loads((base / "integration_metrics.json").read_text())
    effects = json.loads((base / "cross_effect_sizes.json").read_text())
    unified = pl.read_parquet(base / "unified_risk_scores.parquet")
    community = pl.read_parquet(base / "rq1_community_assignments.parquet")
    return integ, effects, unified, community


integ, effects, unified, community = load_integration()

# hero
st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:32px;color:{INK};margin:0 0 8px;\">"
    f"Cross-RQ Integration</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-size:14.5px;color:{INK_MUTED};margin-bottom:20px;max-width:760px;'>"
    f"Where network position, governance risk, and price anomalies do, and don't, line up. "
    f"Integration universe: <strong>{integ['contracts']:,}</strong> contracts (the honest "
    f"out-of-fold + held-out scored population, distinct from RQ2's 714,681-contract "
    f"classifier population, which is never merged with this figure).</div>",
    unsafe_allow_html=True,
)

# RQ1 x RQ2 scatter
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>"
    f"Network dependency vs. governance risk</div>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 2])
with c1:
    corr_status = integ["rq1_rq2_corr_status"]
    small_stat_card(f"r = {corr_status['value']:.3f}", "Pearson correlation", f"n = {corr_status['n_paired']:,} (100% coverage, status: computed)")
    st.caption("Network dependency and procedural risk are independent axes.")
    spearman_sample = unified.select(["buyer_dependency_normalized", "ensemble_risk_prob"]).drop_nulls()
    rho, _ = spearmanr(
        spearman_sample["buyer_dependency_normalized"].to_numpy(),
        spearman_sample["ensemble_risk_prob"].to_numpy(),
    )
    st.caption(f"Spearman rho = {rho:.3f} (descriptive only, full population, monotonic-relationship check).")
with c2:
    chart_card("Buyer dependency vs. ensemble risk (sampled)", "A random sample for plotting only. The correlation itself is computed on the full population.")
    sample = unified.select(["buyer_dependency_normalized", "ensemble_risk_prob"]).sample(n=min(5000, unified.height), seed=42).to_pandas()
    fig = px.scatter(
        sample, x="buyer_dependency_normalized", y="ensemble_risk_prob", opacity=0.35,
        labels={"buyer_dependency_normalized": "Buyer dependency (RQ1)", "ensemble_risk_prob": "Ensemble risk (RQ2)"},
        color_discrete_sequence=["#3B5998"],
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=340)
    st.plotly_chart(fig, width="stretch", key="cross_rq1_rq2_scatter")
why_this_matters(
    f"r = {corr_status['value']:.4f}, n = {corr_status['n_paired']:,}, close to zero. "
    f"Network dependency and procedural risk move independently: a buyer's reliance "
    f"on one supplier tells you almost nothing about its governance risk score, and "
    f"the reverse. The two signals are complementary rather than redundant, which is "
    f"why RQ1 and RQ2 are useful as separate lenses rather than one predicting the other."
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# community x risk
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>"
    f"Community x risk</div>",
    unsafe_allow_html=True,
)
e1, e2, e3, e4 = st.columns(4)
with e1:
    small_stat_card(f"{effects['anova_f_statistic']:,.0f}", "ANOVA F", f"across {effects['n_communities_tested']} communities")
with e2:
    small_stat_card(f"{effects['anova_eta_squared']:.4f}", "eta-squared", "effect size, ensemble_risk_prob")
with e3:
    small_stat_card(f"{effects['chi2_statistic']:,.0f}", "Chi-square", "anomaly_flag x community")
with e4:
    small_stat_card(f"{effects['chi2_cramer_v']:.4f}", "Cramer's V", "effect size, categorical association")
st.caption(
    f"Chi-square = {effects['chi2_statistic']:,.0f} (p < 0.001) shows anomaly flags are "
    f"not evenly spread across communities. Cramer's V = {effects['chi2_cramer_v']:.4f} "
    f"gives the size of that association on a 0 to 1 scale, where 0 is no association "
    f"and 1 is perfect association: this value sits on the smaller end, so community "
    f"membership is a statistically real but modest predictor of anomaly incidence."
)

chart_card("Mean governance risk by Louvain community (top 20 by size)", "")
buyer_communities = (
    community.filter(pl.col("node_type") == "buyer")
    .select(["entity_id", "community_id"])
    .rename({"entity_id": "buyer_id"})
)
community_risk = (
    unified.join(buyer_communities, on="buyer_id", how="inner")
    .group_by("community_id")
    .agg(pl.len().alias("n"), pl.mean("ensemble_risk_prob").alias("mean_risk"))
    .filter(pl.col("n") >= 10)
    .sort("n", descending=True)
    .head(20)
    .to_pandas()
)
if not community_risk.empty:
    fig = px.bar(
        community_risk, x="community_id", y="mean_risk", color="mean_risk",
        color_continuous_scale=[[0, "#EEF2EA"], [1, "#C85A54"]],
        labels={"community_id": "Community", "mean_risk": "Mean ensemble risk"},
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=340, xaxis_type="category")
    st.plotly_chart(fig, width="stretch", key="cross_community_risk_bar")
why_this_matters(f"F={effects['anova_f_statistic']:,.0f}, eta^2={effects['anova_eta_squared']:.4f}: large, statistically significant variation in risk across communities (79 tested).")

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# coverage-aware correlation
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>"
    f"RQ2 <-> RQ3: a coverage-aware null</div>",
    unsafe_allow_html=True,
)
rq2rq3 = integ["rq2_rq3_corr_status"]
st.markdown(
    f"""
    <div style="background:#F3F1EC;border-left:4px solid #9B9791;border-radius:8px;
                padding:18px 22px;">
      <div style="font-size:14px;line-height:1.6;color:{INK_MUTED};">
        The correlation between ensemble governance risk (RQ2) and price-anomaly
        score (RQ3) is reported as <code>{rq2rq3['status']}</code>, not a number.
        Only <strong>{rq2rq3['n_paired']:,}</strong> of {rq2rq3['n_total']:,} contracts
        ({rq2rq3['coverage_fraction']*100:.1f}% coverage) have both scores populated,
        too sparse a base to report a headline correlation. The observed-only value
        (<strong>r = {rq2rq3['observed_value']:.3f}</strong>) is preserved here for
        transparency but is <strong>not</strong> the headline figure and must never
        be rendered as if it were a full-coverage 0.0 or a reliable estimate.
        This is exactly the three-state reporting system
        (<code>computed</code> / <code>degenerate_low_coverage</code> / not-computed)
        used throughout this dashboard's correlation reporting.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# conduct, not corruption
st.markdown(
    f"""
    <div style="background:{RUST_TINT};border-left:4px solid {RUST};border-radius:8px;
                padding:22px 26px;">
      <div style="font-size:12px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;
                  color:{RUST_DARK};margin-bottom:8px;">Conduct, not corruption</div>
      <div style="font-size:14.5px;line-height:1.7;color:{INK};">
        The amendment arm of the risk label is <strong>97.6% zero</strong>:
        amendments are rare across the dataset as a whole. Region <strong>ITH5</strong>
        holds <strong>59.1%</strong> of the observed amendment-rate signal, but on a
        base of only <strong>n = 88</strong> amendments: a small-sample recording
        concentration, not evidence of regional corruption. Sub-national NUTS codes
        are populated for only <strong>~549 of ~600,000</strong> contracts, so any
        regional breakdown here draws on a sparse, partial-coverage sample and
        should be read with that limit firmly in mind.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
neutral_methodology_note(
    "ANOVA + Chi-square across Louvain communities",
    "These tests integrate RQ1's community structure with RQ2's risk label and "
    "RQ3's anomaly flags, checking whether market-structure position "
    "(community membership) is statistically associated with governance risk "
    "and price-anomaly incidence, distinct from the pairwise correlations above.",
)
