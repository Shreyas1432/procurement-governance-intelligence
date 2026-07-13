"""
PGI Dashboard - Procurement Governance Intelligence

Read-only viewer over LOCKED RQ1/RQ2/RQ3 artifacts. NEVER re-fits, NEVER trains.

This is the app's entrypoint and st.navigation() ROUTER. It authenticates the
user, then defines and dispatches the design's exact 9 screens (10 counting
Sign In as a state rather than a page): Overview, RQ1 Supplier Network, RQ2
Governance Risk, RQ3 Price Variance, Cross-RQ Integration, Methodology &
Reproducibility, Audit Log (admin-only, omitted from viewer nav), About and
Data Limitations. Screen modules live in `screens/*.py`. The legacy
`pages/` auto-discovery directory has been removed; as soon as this file
calls st.navigation(), Streamlit ignores pages/ across all sessions anyway,
but removing it avoids any ambiguity on first load.

Design contract (do not violate):
  * Reads locked parquet/JSON/PNG only. No model.predict(), no .fit(), no sklearn imports.
  * "What-if" is a DETERMINISTIC structural-exposure recompute on the locked graph
    (orphaned counterparties, local component fragmentation, degree/dependency shift).
    It is NOT a euro-loss prediction and must never be presented as one.
  * "Governance risk" = procedural-irregularity proxy (recording artifact concentrated
    in NUTS ITH5, n=88), NOT corruption. The caveat banner is mandatory and on by default.
  * Two populations must never be merged: the classifier_population (714,681 -
    TRAIN+VAL+TEST labeled panel, RQ2 screen) and the rgov_scored_population /
    cross-RQ integration universe (594,476 - out-of-fold TRAIN + held-out TEST
    only, VAL excluded; Cross-RQ Integration screen).
    See data/results/rq2_success_metrics.json:"populations" for the reconciliation.

Deploy mode: account-gated (local JSON user store via `_auth.py`). VIEWER
("examiner") sees every screen (Audit Log excepted) with named entities
anonymized; ADMIN ("author") sees named entities behind an explicit, logged
"Reveal" action. This is an INTERIM control, see the honesty note on the
sign-in screen and Task 7 in the project report. OIDC (`st.login`) is a
deploy-time change and out of scope for this pass.

Run:  streamlit run app.py   (or: python -m streamlit run src/dashboard/app.py)
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]   # src/dashboard/app.py -> repo root
sys.path.insert(0, str(ROOT / "src"))        # so `dashboard.*` package imports resolve

from dashboard._auth import require_auth, ADMIN, PUBLIC_BUILD
from dashboard._theme import inject_base_css

st.set_page_config(page_title="PGI Dashboard", layout="wide")
inject_base_css()

# auth
# Renders the Sign In screen and stops here if nobody is authenticated yet.
# Once authenticated (either role), we fall through to the navigation router.
require_auth()

role = st.session_state.get("role")

SCREENS_DIR = Path(__file__).resolve().parent / "screens"

overview = st.Page(str(SCREENS_DIR / "00_overview.py"), title="Overview", default=True)
rq1 = st.Page(str(SCREENS_DIR / "01_rq1_supplier_network.py"), title="RQ1 · Supplier Network")
network_resilience = st.Page(str(SCREENS_DIR / "01b_network_resilience.py"), title="Network Resilience")
rq2 = st.Page(str(SCREENS_DIR / "02_rq2_governance_risk.py"), title="RQ2 · Governance Risk")
rq3 = st.Page(str(SCREENS_DIR / "03_rq3_price_variance.py"), title="RQ3 · Price Variance")
integration = st.Page(str(SCREENS_DIR / "04_cross_rq_integration.py"), title="Cross-RQ Integration")
methodology = st.Page(str(SCREENS_DIR / "05_methodology.py"), title="Methodology")
about = st.Page(str(SCREENS_DIR / "07_about_data_limitations.py"), title="About & Data Limitations")

nav = {
    "PGI": [overview, rq1, network_resilience, rq2, rq3, integration, methodology, about],
}

# Audit Log is admin-only. Omit it from the navigation menu entirely for
# viewer sessions (the underlying screen also self-enforces via
# require_auth(role=ADMIN), so this is a UX nicety, not the security boundary).
# The "and not PUBLIC_BUILD" is defense in depth: role can never actually be
# ADMIN in a public build (_auth.authenticate() refuses it), this just makes
# that guarantee explicit at the one place a mistake here would matter most.
if role == ADMIN and not PUBLIC_BUILD:
    audit_log = st.Page(str(SCREENS_DIR / "06_audit_log.py"), title="Audit Log (admin)")
    nav["Admin"] = [audit_log]

pg = st.navigation(nav)
pg.run()
