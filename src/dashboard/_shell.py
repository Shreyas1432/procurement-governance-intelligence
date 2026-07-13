"""
Shared page chrome for the PGI dashboard: sidebar wordmark, per-screen
breadcrumb + role bar, and the st.navigation() page router.

This is the visual-fidelity layer requested on top of the existing
_theme.py/_components.py/_auth.py. It does not change auth or data
semantics, only how every screen introduces itself.
"""
from __future__ import annotations

import streamlit as st

from dashboard._theme import INK, INK_MUTED, INK_QUIET, BORDER, SURFACE, PRIMARY, PRIMARY_TINT, RUST_DARK, RUST_TINT, FONT_SERIF
from dashboard._auth import ADMIN, ROLE_LONG_LABEL


def render_sidebar_wordmark() -> None:
    """PGI wordmark at the top of the sidebar, above st.navigation's page list."""
    st.sidebar.markdown(
        f"""
        <div style="padding:4px 0 18px;border-bottom:1px solid {BORDER};margin-bottom:8px;">
          <div style="font-family:{FONT_SERIF};font-size:19px;font-weight:700;color:{INK};">PGI</div>
          <div style="font-size:11px;color:{INK_QUIET};letter-spacing:0.03em;">
            Procurement Governance Intelligence
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topbar(breadcrumb: str) -> None:
    """Per-screen top bar: left breadcrumb ('PGI / RQ1 Supplier Network'),
    right role badge ('ADMIN · NAMED' / 'VIEWER · ANONYMIZED'). Matches the
    design's persistent role-context affordance on every authenticated screen.
    Call require_auth() before this."""
    role = st.session_state.get("role")
    if role == ADMIN:
        badge_bg, badge_color, badge_text = RUST_TINT, RUST_DARK, "ADMIN · NAMED"
    else:
        badge_bg, badge_color, badge_text = PRIMARY_TINT, PRIMARY, "VIEWER · ANONYMIZED"

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"<div style='font-size:13px;color:{INK_QUIET};margin-bottom:2px;'>{breadcrumb}</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"<div style='text-align:right;'>"
            f"<span style='font-size:11.5px;font-weight:700;letter-spacing:0.03em;"
            f"color:{badge_color};background:{badge_bg};padding:5px 11px;"
            f"border-radius:20px;white-space:nowrap;'>{badge_text}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown(f"<div style='border-bottom:1px solid {BORDER};margin:6px 0 20px;'></div>", unsafe_allow_html=True)


def render_role_switcher() -> None:
    """Sidebar hint that switching role means signing out first. The actual
    sign-out control lives in _auth.render_logout_button() (called once per
    page); this just adds the "why" copy so the affordance reads as an
    intentional role switch, not just a logout, matching the design."""
    role = st.session_state.get("role")
    other = "author (admin)" if role != ADMIN else "examiner (viewer)"
    st.sidebar.caption(f"To continue as {other}, use Log out below and sign in again.")


def entity_picker(
    label: str,
    candidate_ids: list,
    context_for: "callable",
    prefix: str = "",
    key: str = "entity_picker",
) -> object | None:
    """Shared, correct entity selector: builds every option label ONCE with
    a plain string helper (never a widget-rendering call, so it is safe
    inside a loop or format_func), offers a search box, and shows the
    selected entity's baseline context as a caption before any action is
    taken. This is the fix for the "selector shows blank/... options"
    defect: the previous version called a reveal-button-rendering function
    once per option, which both broke label text and duplicated buttons.

    Args:
      candidate_ids: list of real entity IDs (admin sees these directly;
        viewer sees a pseudonym computed from the same values).
      context_for: callable(real_id) -> str returning the structural
        context suffix, e.g. "degree 42, community 3, 12 contracts".
      key: must be unique per call site on the page.

    Returns the selected real entity ID (or None if the filtered list is
    empty), for both roles -- the CALLER decides what to reveal, this
    function only decides what LABEL to show while picking.
    """
    from dashboard._auth import entity_label_for_list, ADMIN, PUBLIC_BUILD

    role = st.session_state.get("role")
    admin_search = role == ADMIN and not PUBLIC_BUILD
    labels = [entity_label_for_list(cid, prefix=prefix, context=context_for(cid)) for cid in candidate_ids]

    search = st.text_input(
        f"Search {label.lower()}" + (" by ID" if admin_search else " (viewer sessions search by structural context only)"),
        key=f"{key}_search",
    )
    if search:
        if admin_search:
            # ADMIN-only, FULL-build-only: matches against real IDs directly.
            keep = [i for i, cid in enumerate(candidate_ids) if search.lower() in str(cid).lower()]
        else:
            # PUBLIC build always takes this branch even if role somehow held
            # ADMIN: matching against the already-anonymized labels means a
            # real ID is never used as a search key, closing an indirect
            # existence-oracle leak (matches would reveal real-ID substrings
            # even without ever displaying them).
            keep = [i for i, lab in enumerate(labels) if search.lower() in lab.lower()]
    else:
        keep = list(range(len(candidate_ids)))

    if not keep:
        st.warning("No matches for that search.")
        return None

    pick_idx = st.selectbox(label, keep, format_func=lambda i: labels[i], key=f"{key}_select")
    st.caption(f"Baseline: {labels[pick_idx]}")
    return candidate_ids[pick_idx]


def node_info_panel(
    entity_id,
    node_type: str,
    degree: int,
    community_id,
    composite_centrality: float,
    resource_label: str,
) -> None:
    """Compact info panel for a selected/clicked node: type, degree,
    community, composite centrality, and (admin only) the named ID behind a
    logged reveal. Renders an empty state if entity_id is None."""
    from dashboard._auth import reveal_or_anonymize

    if entity_id is None:
        st.info("No node selected. Choose an entity above to see its details here.")
        return

    role = st.session_state.get("role")
    prefix = "B" if str(node_type).lower().startswith("b") else "S"
    display_id = reveal_or_anonymize(entity_id, resource_label, anon_prefix=prefix, key=f"node_info_{resource_label}")

    st.markdown(
        f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-radius:10px;padding:16px 18px;">
          <div style="font-size:12px;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;
                      color:{INK_QUIET};margin-bottom:8px;">Node detail</div>
          <div style="font-family:monospace;font-size:14px;font-weight:600;color:{INK};margin-bottom:10px;">
            {display_id if display_id else "(reveal to view)"}
          </div>
          <div style="font-size:13px;color:{INK_MUTED};line-height:1.7;">
            Type: {node_type}<br>
            Degree: {degree}<br>
            Community: {community_id}<br>
            Composite centrality: {composite_centrality:.4f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
