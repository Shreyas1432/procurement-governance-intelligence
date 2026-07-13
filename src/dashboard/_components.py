"""
Shared component helpers for the PGI dashboard, matching docs/design/PGI.dc.html.

Every function renders HTML matching the design's exact markup structure
(padding, radius, border, color) so every screen looks like one product
rather than five independently-styled pages. Read-only presentation layer,
no computation happens here.
"""
from __future__ import annotations

import streamlit as st

from dashboard._theme import (
    INK, INK_MUTED, INK_FAINT, INK_QUIET, BORDER, BORDER_LIGHT, SURFACE,
    CANVAS_ALT, PANEL, PRIMARY, PRIMARY_TINT, SAGE, SAGE_DARK, SAGE_TINT,
    RUST, RUST_DARK, RUST_TINT, FONT_SERIF, RISK_CAVEAT,
)


def stat_card(scope: str, value: str, caption: str) -> None:
    """Overview/Home-style stat card: uppercase scope label, big serif value,
    muted caption. Matches the design's homeStats / rq1Stats card pattern."""
    st.markdown(
        f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-radius:12px;padding:22px 20px;">
          <div style="font-size:11px;font-weight:600;letter-spacing:0.05em;
                      text-transform:uppercase;color:{INK_QUIET};margin-bottom:10px;">
            {scope}
          </div>
          <div style="font-family:{FONT_SERIF};font-size:26px;font-weight:700;
                      color:{INK};margin-bottom:6px;">
            {value}
          </div>
          <div style="font-size:13.5px;line-height:1.5;color:{INK_FAINT};">
            {caption}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def small_stat_card(value: str, label: str, caption: str = "") -> None:
    """Compact stat card used on RQ1/RQ2/RQ3 metric rows (smaller than the
    Overview stat_card, no uppercase scope eyebrow)."""
    cap_html = (
        f"<div style='font-size:12px;line-height:1.45;color:{INK_QUIET};'>{caption}</div>"
        if caption else ""
    )
    st.markdown(
        f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-radius:10px;padding:16px 18px;">
          <div style="font-family:{FONT_SERIF};font-size:21px;font-weight:700;
                      color:{INK};margin-bottom:4px;">{value}</div>
          <div style="font-size:12.5px;font-weight:600;color:{INK_MUTED};
                      margin-bottom:4px;">{label}</div>
          {cap_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def methodology_callout(eyebrow: str, title: str, body: str) -> None:
    """Sage-accented callout for methodology notes (why this method, not
    that one). Matches the design's composite-centrality / Louvain callouts."""
    st.markdown(
        f"""
        <div style="background:{SAGE_TINT};border-left:4px solid {SAGE};
                    border-radius:8px;padding:20px 22px;margin-bottom:24px;">
          <div style="font-size:12px;font-weight:700;letter-spacing:0.05em;
                      text-transform:uppercase;color:{SAGE_DARK};margin-bottom:8px;">
            {eyebrow}
          </div>
          <div style="font-family:{FONT_SERIF};font-size:15px;font-weight:700;
                      color:{INK};margin-bottom:6px;">{title}</div>
          <div style="font-size:14px;line-height:1.6;color:{INK_MUTED};">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def neutral_methodology_note(eyebrow: str, body: str) -> None:
    """Grey/neutral methodology footnote (design's '.methodology-note',
    e.g. RQ1's Louvain-vs-alternatives paragraph)."""
    st.markdown(
        f"""
        <div style="background:{PANEL};border-left:4px solid {INK_QUIET};
                    border-radius:8px;padding:18px 22px;">
          <div style="font-size:12px;font-weight:700;letter-spacing:0.05em;
                      text-transform:uppercase;color:{INK_FAINT};margin-bottom:8px;">
            {eyebrow}
          </div>
          <div style="font-size:14px;line-height:1.6;color:{INK_MUTED};">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def caveat_banner(text: str = RISK_CAVEAT, title: str = "Governance caveat") -> None:
    """Rust-accented caveat banner, the mandatory governance-risk reminder
    shown on every risk-bearing screen. Plain text, no emoji."""
    st.markdown(
        f"""
        <div style="background:{RUST_TINT};border-left:4px solid {RUST};
                    border-radius:8px;padding:20px 24px;display:flex;gap:14px;
                    align-items:flex-start;margin-bottom:24px;">
          <div style="width:20px;height:20px;border-radius:50%;border:2px solid {RUST_DARK};
                      flex-shrink:0;margin-top:2px;display:flex;align-items:center;
                      justify-content:center;font-size:13px;font-weight:700;color:{RUST_DARK};">!</div>
          <div>
            <div style="font-size:14.5px;font-weight:700;color:{INK};margin-bottom:4px;">{title}</div>
            <div style="font-size:14.5px;line-height:1.6;color:{INK_MUTED};">{text}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label_definition_banner() -> None:
    """RQ2's compact rust banner: 'Risk label definition: ... not a corruption
    determination.'"""
    st.markdown(
        f"""
        <div style="background:{RUST_TINT};border-left:4px solid {RUST};
                    border-radius:8px;padding:14px 18px;font-size:13.5px;
                    line-height:1.55;color:{INK_MUTED};margin-bottom:24px;max-width:680px;">
          <strong style="color:{INK};">Risk label definition:</strong> a
          procedural-recording irregularity: a paperwork and process signal.
          It is <strong>not</strong> a corruption determination.
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_card(title: str, caption: str = "") -> None:
    """Header row for a chart card (title + 'why this matters' caption).
    Caller renders the actual Plotly chart immediately after calling this
    inside the same st.container()/columns context."""
    st.markdown(
        f"<div style='font-size:13px;font-weight:700;color:{INK};margin-bottom:10px;'>{title}</div>",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def why_this_matters(text: str) -> None:
    """Small muted caption under a chart, the design's convention of a
    one-line interpretive note under every figure."""
    st.markdown(
        f"<div style='font-size:11.5px;color:{INK_QUIET};margin-top:6px;'>{text}</div>",
        unsafe_allow_html=True,
    )


def role_badge(role: str) -> str:
    """Small pill badge for the current role, shown in the top bar.
    Returns HTML (caller wraps in st.markdown)."""
    if role == "ADMIN":
        bg, color, text = RUST_TINT, RUST_DARK, "AUTHOR (ADMIN)"
    else:
        bg, color, text = PRIMARY_TINT, PRIMARY, "EXAMINER (VIEWER)"
    return (
        f"<span style='font-size:12px;font-weight:700;letter-spacing:0.03em;"
        f"color:{color};background:{bg};padding:6px 12px;border-radius:20px;"
        f"white-space:nowrap;'>{text}</span>"
    )


def status_pill(text: str, tone: str = "neutral") -> str:
    """Small rounded status pill (e.g. RETAINED / leakage-reject / computed /
    degenerate_low_coverage). tone in {positive, caution, neutral}."""
    tones = {
        "positive": (SAGE_TINT, SAGE_DARK),
        "caution": (RUST_TINT, RUST_DARK),
        "neutral": (PANEL, INK_FAINT),
    }
    bg, color = tones.get(tone, tones["neutral"])
    return (
        f"<span style='font-size:11.5px;font-weight:700;padding:4px 10px;"
        f"border-radius:20px;background:{bg};color:{color};white-space:nowrap;'>{text}</span>"
    )


def anonymized_note(is_viewer: bool) -> str:
    """Trailing clause appended to the risk-caveat banner for viewer-role
    sessions, matching the design's {{ anonymizedNote }} binding."""
    if is_viewer:
        return "Named entities are anonymized throughout your session."
    return ""


WHAT_IF_FORCE_GRAPH_NODE_CAP = 150


def what_if_force_graph(nodes: list[dict], links: list[dict], removed_ids: list[str], height: int = 460) -> None:
    """Interactive, capped before/after force-directed graph for a structural
    node-removal what-if. Read-only visual companion to the deterministic
    summary cards and table on the caller's page -- this component never
    computes anything itself; it only renders a JSON payload the caller
    already built from the sampled resilience graph, and lets the user
    toggle between "before removal" and "after removal" to SEE which edges
    are severed, rather than only reading a count.

    Hard cap: `nodes` must already be <= WHAT_IF_FORCE_GRAPH_NODE_CAP (150)
    entries before calling this -- enforced again here defensively so this
    component can never be made to render the full sampled graph by a
    future caller mistake.

    Each node dict: {id, label, kind ("buyer"|"supplier"), degree,
    community, exposed_value, is_articulation_point}. `label` is whatever
    the caller decided to show (real ID for an admin who has revealed it,
    anonymized pseudonym for a viewer or an admin who has not revealed it)
    -- this component does no anonymization/reveal logic of its own.
    `exposed_value` drives node size (log-scaled, since award value spans
    orders of magnitude) and must be a number the caller already computed
    (this component performs no computation of its own). Each link dict:
    {source, target} referencing node ids. `removed_ids` are the node(s) in
    the current removal set; they and every link touching them are
    rendered in a visually distinct "removed" state in the after view.

    Uses the force-graph library (CDN, MIT licensed) via
    streamlit.components.v1.html. Toggling before/after does not restart
    the physics simulation from scratch; it fades the removed nodes and
    their severed links out/in and lets the existing layout settle, which
    reads as a simple animated transition rather than a jarring re-layout.
    """
    import json as _json
    import streamlit.components.v1 as _components

    if len(nodes) > WHAT_IF_FORCE_GRAPH_NODE_CAP:
        nodes = nodes[:WHAT_IF_FORCE_GRAPH_NODE_CAP]
        kept_ids = {n["id"] for n in nodes}
        links = [l for l in links if l["source"] in kept_ids and l["target"] in kept_ids]

    payload = _json.dumps({"nodes": nodes, "links": links, "removedIds": list(removed_ids)})

    html = f"""
    <div id="wif-container" style="width:100%;">
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;font-family:Inter,sans-serif;">
        <button id="wif-btn-before" class="wif-toggle-btn wif-active"
                style="font-size:12.5px;font-weight:600;padding:6px 14px;border-radius:20px;
                       border:1px solid {BORDER};background:{PRIMARY_TINT};color:{PRIMARY};cursor:pointer;">
          Before removal
        </button>
        <button id="wif-btn-after" class="wif-toggle-btn"
                style="font-size:12.5px;font-weight:600;padding:6px 14px;border-radius:20px;
                       border:1px solid {BORDER};background:{SURFACE};color:{INK_MUTED};cursor:pointer;">
          After removal
        </button>
        <span id="wif-tooltip" style="font-size:12px;color:{INK_MUTED};margin-left:8px;"></span>
      </div>
      <div id="wif-graph" style="width:100%;height:{height}px;border:1px solid {BORDER};border-radius:10px;background:#FFFFFF;"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/force-graph@1.43.5/dist/force-graph.min.js"></script>
    <script>
      (function() {{
        const data = {payload};
        const removedSet = new Set(data.removedIds);
        const colorFor = (n) => {{
          if (removedSet.has(n.id)) return "{RUST}";
          if (n.kind === "buyer") return "{PRIMARY}";
          return "#7A9B6F";
        }};
        const graphData = {{
          nodes: data.nodes.map(n => Object.assign({{}}, n)),
          links: data.links.map(l => Object.assign({{}}, l)),
        }};

        const logValues = graphData.nodes.map(n => Math.log10(Math.max(n.exposed_value || 1, 1)));
        const minLog = logValues.length ? Math.min(...logValues) : 0;
        const maxLog = logValues.length ? Math.max(...logValues) : 1;
        const valFor = (n) => {{
          const v = Math.log10(Math.max(n.exposed_value || 1, 1));
          const norm = maxLog > minLog ? (v - minLog) / (maxLog - minLog) : 0.5;
          const boost = removedSet.has(n.id) ? 6 : 0;
          return 3 + norm * 12 + boost;
        }};

        const elem = document.getElementById('wif-graph');
        const tooltip = document.getElementById('wif-tooltip');
        let showingAfter = false;

        const Graph = ForceGraph()(elem)
          .graphData(graphData)
          .nodeId('id')
          .nodeLabel(n => {{
            const lines = [
              n.label,
              'Type: ' + (removedSet.has(n.id) ? n.kind + ' (removed)' : n.kind),
              'Degree: ' + n.degree,
              'Community: ' + n.community,
              'Exposed value: EUR ' + Math.round(n.exposed_value || 0).toLocaleString(),
              n.is_articulation_point ? 'Structural cut vertex: yes' : null,
            ].filter(Boolean);
            return lines.join('<br/>');
          }})
          .nodeColor(n => {{
            if (showingAfter && removedSet.has(n.id)) return '#D8D5CC';
            return colorFor(n);
          }})
          .nodeVal(valFor)
          .linkColor(l => {{
            const touchesRemoved = removedSet.has(l.source.id || l.source) || removedSet.has(l.target.id || l.target);
            if (showingAfter && touchesRemoved) return 'rgba(200,90,84,0.35)';
            return '#D8D5CC';
          }})
          .linkWidth(l => {{
            const touchesRemoved = removedSet.has(l.source.id || l.source) || removedSet.has(l.target.id || l.target);
            return (showingAfter && touchesRemoved) ? 2.5 : 1;
          }})
          .linkLineDash(l => {{
            const touchesRemoved = removedSet.has(l.source.id || l.source) || removedSet.has(l.target.id || l.target);
            return (showingAfter && touchesRemoved) ? [4, 3] : null;
          }})
          .onNodeClick(n => {{
            tooltip.innerText = n.label + ' -- degree ' + n.degree + ', community ' + n.community +
              ', exposed value EUR ' + Math.round(n.exposed_value || 0).toLocaleString();
          }})
          .onNodeHover(n => {{
            elem.style.cursor = n ? 'pointer' : 'default';
          }})
          .cooldownTicks(80);

        function renderState(after) {{
          showingAfter = after;
          Graph.nodeColor(Graph.nodeColor());
          Graph.linkColor(Graph.linkColor());
          Graph.linkWidth(Graph.linkWidth());
          Graph.linkLineDash(Graph.linkLineDash());
          if (after) {{
            tooltip.innerText = 'Showing after removal: the removed node(s) and their severed ties are dimmed and dashed.';
          }} else {{
            tooltip.innerText = 'Showing before removal: the full capped subgraph as it stands today.';
          }}
        }}

        const btnBefore = document.getElementById('wif-btn-before');
        const btnAfter = document.getElementById('wif-btn-after');
        function setActive(after) {{
          renderState(after);
          if (after) {{
            btnAfter.style.background = '{RUST_TINT}';
            btnAfter.style.color = '{RUST_DARK}';
            btnBefore.style.background = '{SURFACE}';
            btnBefore.style.color = '{INK_MUTED}';
          }} else {{
            btnBefore.style.background = '{PRIMARY_TINT}';
            btnBefore.style.color = '{PRIMARY}';
            btnAfter.style.background = '{SURFACE}';
            btnAfter.style.color = '{INK_MUTED}';
          }}
        }}
        btnBefore.addEventListener('click', () => setActive(false));
        btnAfter.addEventListener('click', () => setActive(true));
        setActive(false);
      }})();
    </script>
    """
    _components.html(html, height=height + 50, scrolling=False)
