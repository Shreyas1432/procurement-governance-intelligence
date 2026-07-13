"""
About + Data Limitations. Public-safe, aggregate only. No named entities on
this screen regardless of role.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import require_auth, render_logout_button
from dashboard._theme import inject_base_css, INK, INK_MUTED, FONT_SERIF
from dashboard._components import neutral_methodology_note, caveat_banner
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher

inject_base_css()
require_auth()
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / About and Data Limitations")

st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:32px;color:{INK};margin:0 0 8px;\">"
    f"About PGI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-size:14.5px;color:{INK_MUTED};margin-bottom:20px;max-width:760px;'>"
    f"Procurement Governance Intelligence is a research publication analyzing "
    f"public procurement records for structural, risk, and price-variance "
    f"signals, using a fully white-box, auditable analysis pipeline. It "
    f"engages with two regulatory anchors: <strong>GDPR Article 22</strong> "
    f"(the right not to be subject to a decision based solely on automated "
    f"processing) and <strong>EU Directive 2014/24/EU</strong> (public "
    f"procurement transparency requirements).</div>",
    unsafe_allow_html=True,
)

r1, r2, r3 = st.columns(3)
with r1:
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E4E1D9;border-radius:10px;padding:18px;height:170px;">
          <div style="font-family:{FONT_SERIF};font-size:15px;font-weight:700;color:{INK};margin-bottom:8px;">RQ1: Supplier Network</div>
          <div style="font-size:13px;line-height:1.5;color:{INK_MUTED};">Community structure and centrality in the buyer-supplier procurement graph, using Louvain communities, PageRank + betweenness centrality, and modularity null-model z-scores.</div>
        </div>
        """, unsafe_allow_html=True,
    )
with r2:
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E4E1D9;border-radius:10px;padding:18px;height:170px;">
          <div style="font-family:{FONT_SERIF};font-size:15px;font-weight:700;color:{INK};margin-bottom:8px;">RQ2: Governance Risk</div>
          <div style="font-size:13px;line-height:1.5;color:{INK_MUTED};">A procedural-recording-irregularity risk label, modeled with Random Forest, constrained XGBoost, and Logistic Regression after a thirteen-candidate leakage audit.</div>
        </div>
        """, unsafe_allow_html=True,
    )
with r3:
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E4E1D9;border-radius:10px;padding:18px;height:170px;">
          <div style="font-family:{FONT_SERIF};font-size:15px;font-weight:700;color:{INK};margin-bottom:8px;">RQ3: Price Variance</div>
          <div style="font-size:13px;line-height:1.5;color:{INK_MUTED};">Consensus price-anomaly detection using Isolation Forest, Local Outlier Factor, and Elliptic Envelope, requiring agreement from at least 2 of 3 detectors.</div>
        </div>
        """, unsafe_allow_html=True,
    )

st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<h2 style=\"font-family:{FONT_SERIF};font-size:24px;color:{INK};margin:0 0 14px;\">"
    f"Data limitations</h2>",
    unsafe_allow_html=True,
)

neutral_methodology_note(
    "Envelope counts vs. competing bids",
    "There is ambiguity in whether some fields record the number of bidding "
    "envelopes received or the number of genuinely competing bids. This "
    "distinction matters for single-bidder-rate interpretation and is flagged "
    "rather than silently resolved.",
)
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
neutral_methodology_note(
    "Bid-count field is largely absent",
    "The bid-count field (lot_bidscount) is null for approximately 98.4% of "
    "records raw-wide, and only ~3.6% populated within the scored panel used "
    "for RQ2. Any statistic built on it describes a small observed subset, "
    "not the full population.",
)
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
neutral_methodology_note(
    "Single-bidder rate is a partial-coverage estimate",
    "The oft-cited ~29.8% single-bidder rate is computed among OBSERVED lots "
    "only (bid_count non-null) over the full raw dataset (about 1.6% "
    "coverage raw-wide, 3.6% within the RQ2 scored panel). It is not a rate "
    "over the full population and should not be read as one.",
)
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
neutral_methodology_note(
    "Sub-national geography is sparse",
    "Sub-national NUTS region codes are populated for only approximately 549 "
    "of ~600,000 contracts. Any regional breakdown (including the ITH5 "
    "amendment-concentration finding) draws on this sparse, partial-coverage "
    "sample and should be read with that limit in mind.",
)

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
caveat_banner()

st.markdown(
    f"<div style='text-align:center;font-size:12.5px;color:{INK_MUTED};margin-top:24px;'>"
    f"Not a corruption-detection tool. Risk labels describe procedural-recording "
    f"irregularities only.</div>",
    unsafe_allow_html=True,
)
