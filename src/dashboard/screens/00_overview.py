"""
Overview: the design's "Welcome back" authenticated home screen. Dispatched
by app.py's st.navigation() router; do not run this file directly with
`streamlit run` (use app.py as the entry point).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import (
    require_auth, render_change_password_form, render_logout_button,
    ADMIN, VIEWER, ROLE_LONG_LABEL, DASHBOARD_DATA_RESULTS,
)
from dashboard._theme import inject_base_css, INK, INK_MUTED, INK_QUIET, FONT_SERIF, PRIMARY
from dashboard._components import stat_card, caveat_banner, methodology_callout
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher

inject_base_css()

require_auth()
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / Overview")

role = st.session_state.get("role")
role_long = ROLE_LONG_LABEL.get(role, role)

# hero
st.markdown(
    f"<div style='font-size:12px;font-weight:700;letter-spacing:0.05em;"
    f"text-transform:uppercase;color:{PRIMARY};margin-bottom:8px;'>Research Publication</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:36px;color:{INK};margin:0 0 8px;\">"
    f"Welcome back</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-size:15.5px;color:{INK_MUTED};margin-bottom:28px;max-width:720px;'>"
    f"You're viewing PGI as <strong>{role_long}</strong>. "
    + (
        "Named buyer, supplier, and contract identifiers are anonymized throughout your session: nothing is withheld, only pseudonymized."
        if role == VIEWER
        else "Named entities are visible to you; every reveal is appended to the audit log."
    )
    + "</div>",
    unsafe_allow_html=True,
)

caveat_banner()

# stats
meta_path = DASHBOARD_DATA_RESULTS / "pipeline_metadata.json"
rq2_path = DASHBOARD_DATA_RESULTS / "rq2_success_metrics.json"
rq3_path = DASHBOARD_DATA_RESULTS / "rq3_success_metrics.json"

try:
    rq2_meta = json.loads(rq2_path.read_text())
    classifier_pop = rq2_meta["populations"]["classifier_population"]
    n_labeled = classifier_pop["n_total"]
    positive_rate = classifier_pop["positive_rate"]
except Exception:
    n_labeled, positive_rate = None, None

try:
    rq3_meta = json.loads(rq3_path.read_text())
    consensus_rate = rq3_meta["consensus_anomaly_rate"]
    consensus_n = rq3_meta["consensus_anomaly_count"]
except Exception:
    consensus_rate, consensus_n = None, None

c1, c2, c3, c4 = st.columns(4)
with c1:
    stat_card(
        "Full dataset · 2006-2021",
        "~12.1M",
        "Procurement records analyzed across the study window.",
    )
with c2:
    stat_card(
        "RQ1 · network population",
        "161,695",
        "Supplier & buyer nodes in the market-structure graph.",
    )
with c3:
    stat_card(
        "RQ2 · labeled contracts",
        f"{n_labeled:,}" if n_labeled else "N/A",
        f"{positive_rate*100:.2f}% flagged as high procedural-risk." if positive_rate else "",
    )
with c4:
    stat_card(
        "RQ3 · consensus anomalies",
        f"{consensus_rate*100:.2f}%" if consensus_rate else "N/A",
        f"{consensus_n:,} contracts flagged by ≥ 2 of 3 detectors." if consensus_n else "",
    )

st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

# nav
st.markdown(
    f"<h2 style=\"font-family:{FONT_SERIF};font-size:22px;color:{INK};margin:0 0 16px;\">"
    f"Research questions</h2>",
    unsafe_allow_html=True,
)

nav_cards = [
    ("RQ1: Supplier Network", "Community structure, centrality, and structural what-if exposure."),
    ("RQ2: Governance Risk", "The leakage audit, the model, and the temporal-drift honesty check."),
    ("RQ3: Price Variance", "Ensemble anomaly consensus across three white-box detectors."),
    ("Cross-RQ Integration", "Where network position, risk, and price anomalies do (and don't) line up."),
]
cols = st.columns(4)
for col, (title, desc) in zip(cols, nav_cards):
    with col:
        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:1px solid #E4E1D9;border-radius:10px;
                        padding:18px;height:150px;">
              <div style="font-family:{FONT_SERIF};font-size:15px;font-weight:700;
                          color:{INK};margin-bottom:8px;">{title}</div>
              <div style="font-size:13px;line-height:1.5;color:{INK_MUTED};">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
st.caption("Use the page navigation in the sidebar to open any screen directly.")

if role == ADMIN:
    render_change_password_form()

# methodology
with st.expander("Methodology / provenance", expanded=False):
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        st.markdown(f"**Pipeline run:** {meta.get('run_date', 'N/A')} · deterministic seed 42")
        st.markdown("**Source records:** ~12.1M · 2006-2021")
        st.markdown("**Temporal split:** rolling train / out-of-time test")
        st.markdown("**Golden master:** locked at feature-set finalization")
    else:
        st.markdown("Pipeline metadata not found.")
    st.caption(
        "This dashboard reads locked parquet/JSON artifacts only. It never calls "
        ".fit()/.predict() and never imports sklearn/xgboost at runtime. All model "
        "outputs shown are frozen at pipeline-run time."
    )

methodology_callout(
    "Methodology",
    "Thirteen candidate features, one survivor",
    "The RQ2 model's single feature (contract_value_log) is the result of a "
    "leakage audit that screened thirteen candidates and reasoned out twelve "
    "for temporal, leakage, integration, redundancy, or constancy issues. "
    "See the RQ2 screen for the full audit trail.",
)
