"""
Methodology & Reproducibility. Public-safe, aggregate-only screen.
Three contributions, design constraints, ethics boundaries, reproducibility
checklist, tech-stack note.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import require_auth, render_logout_button
from dashboard._theme import inject_base_css, INK, INK_MUTED, FONT_SERIF
from dashboard._components import methodology_callout, neutral_methodology_note
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher

inject_base_css()
require_auth()
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / Methodology & Reproducibility")

st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:32px;color:{INK};margin:0 0 8px;\">"
    f"Methodology &amp; Reproducibility</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-size:14.5px;color:{INK_MUTED};margin-bottom:24px;max-width:760px;'>"
    f"A fully white-box, interpretable architecture. No neural networks or "
    f"learned embeddings anywhere. Every algorithm is chosen for auditability "
    f"and explainability over raw predictive power.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='font-size:18px;font-weight:700;color:{INK};margin-bottom:12px;'>Three contributions</div>",
    unsafe_allow_html=True,
)
methodology_callout(
    "Contribution 1",
    "A data-leakage audit as a first-class artifact",
    "Thirteen candidate features were screened for association with the risk "
    "label's own construction rules as well as the label itself, catching "
    "second-order proxies (e.g. buyer_region explaining 96.6% of the variance "
    "in contract_amendments, itself a label input) that a naive correlation "
    "check against the label would miss.",
)
methodology_callout(
    "Contribution 2",
    "Coverage-aware correlation reporting",
    "Every cross-RQ correlation is reported in one of three states: computed "
    "(sufficient paired coverage), degenerate_low_coverage (too sparse to "
    "trust: reported as null with the observed value preserved for "
    "transparency, never silently rendered as 0.0), or not-computed. This "
    "prevents a sparse pairing from masquerading as a reliable null result.",
)
methodology_callout(
    "Contribution 3",
    "Determinism and fixed seeds throughout",
    "Every stochastic step (Louvain community detection, train/test splits, "
    "cross-validation folds, the anomaly-ensemble contamination search) uses "
    "a fixed seed (42, with named alternates for stability checks). A locked "
    "golden-master snapshot of every headline number lets any regression be "
    "caught immediately.",
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='font-size:18px;font-weight:700;color:{INK};margin-bottom:12px;'>Design constraints</div>",
    unsafe_allow_html=True,
)
neutral_methodology_note(
    "Constraints",
    "No neural networks or learned graph embeddings (Louvain, PageRank, "
    "betweenness, Random Forest, constrained XGBoost, and Logistic Regression "
    "cover the full 3x3 algorithm matrix). No re-fitting inside this dashboard: "
    "every number here is read from a locked parquet/JSON artifact, never "
    "recomputed by calling .fit() or .predict(). Partial dependence plots, "
    "not SHAP, for feature-effect explanation: simpler and sufficient for a "
    "one-feature model.",
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='font-size:18px;font-weight:700;color:{INK};margin-bottom:12px;'>Ethics boundaries</div>",
    unsafe_allow_html=True,
)
neutral_methodology_note(
    "Boundaries",
    "Named procurement entities (buyers, suppliers, contract IDs) are role-gated "
    "and logged: viewer sessions see stable anonymized pseudonyms throughout; "
    "admin sessions can reveal named entities only through an explicit action "
    "that is appended to the audit log. The network 'what-if' exploration is "
    "structural only: degree drops, orphaned counterparties, local "
    "fragmentation counts, never a financial loss or revenue projection. "
    "The governance-risk label describes a procedural-recording irregularity, "
    "not a corruption determination.",
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='font-size:18px;font-weight:700;color:{INK};margin-bottom:12px;'>Reproducibility checklist</div>",
    unsafe_allow_html=True,
)
checklist = [
    "Deterministic seeding (seed=42, with named alternates for stability checks) across every stochastic step",
    "Locked golden-master snapshot of every headline figure, checked before any dashboard change ships",
    "Coverage-aware correlation reporting (computed / degenerate_low_coverage / not-computed)",
    "Target-leakage guard: forbidden feature lists checked against label-defining and label-rule-input columns",
    "This dashboard reads locked parquet/JSON/PNG artifacts only, no .fit()/.predict() calls at runtime",
]
for item in checklist:
    st.markdown(f"- {item}")

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
neutral_methodology_note(
    "Technology stack",
    "Python analysis pipeline (DuckDB for SQL feature engineering, Polars for "
    "dataframe operations), XGBoost / scikit-learn models (constrained to "
    "shallow, interpretable configurations), deterministic seeding throughout, "
    "and a golden-master snapshot locking every headline number this dashboard "
    "displays.",
)
