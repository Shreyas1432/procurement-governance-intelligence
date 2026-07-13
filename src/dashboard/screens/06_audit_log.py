"""
Audit Log. Admin-only screen (require_auth(role=ADMIN), the one screen with
no viewer-safe rendering, since its entire content is who-did-what). Reads
logs/access.log via _auth.read_audit_log(). Exporting the log is itself
logged; viewing/filtering it is not.
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import require_auth, render_logout_button, ADMIN, read_audit_log, log_action
from dashboard._theme import inject_base_css, INK, INK_MUTED, FONT_SERIF
from dashboard._components import small_stat_card
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher

inject_base_css()
require_auth(role=ADMIN)  # the one screen with no viewer-safe rendering
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / Audit Log (admin only)")

st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:32px;color:{INK};margin:0 0 8px;\">"
    f"Audit Log</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-size:14.5px;color:{INK_MUTED};margin-bottom:20px;max-width:760px;'>"
    f"Every sign-in (success and failure), named-entity reveal, and export is "
    f"recorded here. This export action is itself logged.</div>",
    unsafe_allow_html=True,
)

rows = read_audit_log()

c1, c2, c3 = st.columns(3)
with c1:
    small_stat_card(f"{len(rows):,}", "Total events", "")
with c2:
    n_reveals = sum(1 for r in rows if "reveal" in r["action"].lower())
    small_stat_card(f"{n_reveals:,}", "Reveal actions", "Admin-only, always logged")
with c3:
    n_failed = sum(1 for r in rows if r["outcome"] == "failure")
    small_stat_card(f"{n_failed:,}", "Failed sign-ins", "")

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

roles_present = sorted({r["role"] for r in rows}) or ["-"]
actions_present = sorted({r["action"] for r in rows}) or ["-"]

fc1, fc2, fc3 = st.columns(3)
with fc1:
    role_filter = st.multiselect("Role", roles_present, default=roles_present, key="audit_role_filter")
with fc2:
    action_filter = st.multiselect("Action", actions_present, default=actions_present, key="audit_action_filter")
with fc3:
    search = st.text_input("Search resource", "", key="audit_search")

filtered = [
    r for r in rows
    if r["role"] in role_filter and r["action"] in action_filter
    and (search.lower() in r["resource"].lower() if search else True)
]

st.dataframe(
    filtered, width="stretch", height=420, hide_index=True, key="audit_log_table",
    column_config={
        "ts": st.column_config.TextColumn("Timestamp", width="medium"),
        "role": st.column_config.TextColumn("Role", width="small"),
        "action": st.column_config.TextColumn("Action", width="medium"),
        "resource": st.column_config.TextColumn("Resource", width="large"),
        "outcome": st.column_config.TextColumn("Outcome", width="small"),
    },
)

st.caption(f"Showing {len(filtered):,} of {len(rows):,} events.")

if st.button("Export filtered events (logs this export)"):
    log_action("ADMIN", "Export audit log", f"{len(filtered)} events", "success")
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["ts", "role", "action", "resource", "outcome"])
    writer.writeheader()
    writer.writerows(filtered)
    st.download_button(
        "Download CSV", buf.getvalue(), file_name="pgi_audit_log_export.csv",
        mime="text/csv", key="audit_download_csv",
    )
    st.success("Export logged.")
