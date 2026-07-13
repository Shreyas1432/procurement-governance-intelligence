"""
Network Resilience: what-if. Read-only over the Phase A resilience artifacts
(data/results/resilience_exposure_table.parquet, resilience_metrics.json)
and the src.rq4_resilience.resilience module's load_sample()/
simulate_removal(). This screen never re-fits, recomputes the exposure
table, or writes anything; it only reads Phase A's already-computed numbers
and lets the user pick a removal scenario to see them rendered.

The dataset supports structural exposure only: exposed_contract_value is
purely the award value currently routed through a node, and this screen
makes no monetary, competitive-standing, or corruption estimate beyond
that fact. See the caveat banner and the data-limitations expander at the
bottom of this page.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import require_auth, render_logout_button, anonymize_id, log_action, PUBLIC_BUILD
from dashboard._theme import inject_base_css, INK, INK_MUTED, FONT_SERIF, PRIMARY
from dashboard._components import small_stat_card, neutral_methodology_note, why_this_matters, caveat_banner, what_if_force_graph, WHAT_IF_FORCE_GRAPH_NODE_CAP
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher, entity_picker
from dashboard.components.graph_loader import neighbourhood_subgraph
from src.rq4_resilience.resilience import load_sample, simulate_removal

inject_base_css()
require_auth()
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / RQ1 / Network Resilience")

role = st.session_state.get("role")

# The mandated caveat string from Phase A (src.rq4_resilience.resilience.CAVEAT),
# reproduced literally here so this screen shows the exact same wording; every
# simulate_removal() result also carries it in result["caveat"].
EXPOSED_VALUE_CAVEAT = (
    "exposed value is the award value currently routed through a node, "
    "not an estimate of financial loss."
)

# UI interaction cap, distinct from the render cap: keeps a "whole community"
# removal scenario interactive (some Louvain communities on the sample have
# thousands of members; simulating removal of all of them would be slow and
# the render is capped at 150 regardless).
UI_COMMUNITY_REMOVAL_CAP = 30


@st.cache_resource(show_spinner="Loading resilience sample graph...")
def get_sample_graph():
    return load_sample()


@st.cache_data(show_spinner="Loading resilience artifacts...")
def load_artifacts():
    base = ROOT / "data" / "results"
    metrics = json.loads((base / "resilience_metrics.json").read_text())
    table = pl.read_parquet(base / "resilience_exposure_table.parquet")
    node_info = {row["node_id"]: row for row in table.iter_rows(named=True)}
    community_partition = {nid: info["community_id"] for nid, info in node_info.items()}
    return metrics, table, node_info, community_partition


graph, sample_coverage = get_sample_graph()
metrics, exposure_table, node_info, community_partition = load_artifacts()


def _display_id(node_id: str) -> str:
    """Role-respecting label for a node id, matching the dashboard's shared
    reveal/anonymize convention: ADMIN sees the named id, VIEWER sees a
    stable pseudonym. Does its own reveal-free anonymization (no button)
    since this is used inside lists/graph labels, not a single detail panel."""
    raw = node_id[2:]
    prefix = "B" if node_id.startswith("B_") else "S"
    if role == "ADMIN" and not PUBLIC_BUILD:
        return raw
    return anonymize_id(raw, prefix=prefix)


# hero
st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:32px;color:{INK};margin:0 0 8px;\">"
    f"Network Resilience</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-size:14.5px;color:{INK_MUTED};margin-bottom:16px;max-width:760px;'>"
    f"What happens to the network's structure if one buyer or supplier, or a small "
    f"group of them, is removed. Every number below comes from the resilience module's "
    f"deterministic recompute on a fixed 35% value-stratified sample of the training "
    f"graph, never a financial forecast or a prediction of wrongdoing.</div>",
    unsafe_allow_html=True,
)
caveat_banner(
    text=f"This screen shows structural exposure only: {EXPOSED_VALUE_CAVEAT} It never "
         f"estimates a competitive-standing figure, and a high structural exposure score "
         f"is not evidence of wrongdoing by the removed entity.",
    title="Structural exposure only",
)

# summary strip
cov = sample_coverage["coverage"]
total_sample_nodes = cov["n_buyers"] + cov["n_suppliers"]
n_ap = metrics["global_summary"]["n_articulation_points"]
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>Sample coverage</div>",
    unsafe_allow_html=True,
)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    small_stat_card(f"{total_sample_nodes:,}", "Sampled nodes", f"{cov['n_buyers']:,} buyers, {cov['n_suppliers']:,} suppliers")
with c2:
    small_stat_card(f"{cov['n_edges']:,}", "Sampled edges", f"{cov['edge_frac_retained']:.1%} of training edges")
with c3:
    small_stat_card(f"{cov['value_frac_retained']:.1%}", "Value retained", "Share of total contract value in the sample")
with c4:
    small_stat_card(f"{metrics['global_summary']['n_communities']:,}", "Communities", "Louvain, computed once on the sample")
with c5:
    small_stat_card(f"{n_ap:,}", "Structural cut vertices", f"{n_ap / total_sample_nodes:.1%} of sampled nodes")
why_this_matters(
    f"A cut vertex (articulation point) is a node whose removal splits the sample into "
    f"more pieces. {n_ap:,} of {total_sample_nodes:,} sampled nodes ({n_ap / total_sample_nodes:.1%}) "
    f"qualify, which is a structural-fragility observation about this hub-and-spoke "
    f"procurement network, not a claim about any individual entity's conduct."
)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# controls
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>Simulate a removal</div>",
    unsafe_allow_html=True,
)
mode = st.radio("Remove a", ["Buyer", "Supplier"], horizontal=True, key="nr_mode")
node_type_val = mode.lower()
prefix_letter = "B" if mode == "Buyer" else "S"

candidates_df = (
    exposure_table.filter(pl.col("node_type") == node_type_val)
    .sort("exposed_contract_value", descending=True)
    .head(100)
)
cand_rows = {row["node_id"]: row for row in candidates_df.iter_rows(named=True)}
cand_raw_ids = [nid[2:] for nid in cand_rows.keys()]

target_raw_id = entity_picker(
    f"Select a {node_type_val} to simulate removing (top 100 by exposed value)",
    cand_raw_ids,
    context_for=lambda rid: (
        f"degree {cand_rows[f'{prefix_letter}_{rid}']['degree']}, "
        f"community {cand_rows[f'{prefix_letter}_{rid}']['community_id']}, "
        f"exposed value EUR {cand_rows[f'{prefix_letter}_{rid}']['exposed_contract_value']:,.0f}"
    ),
    prefix=prefix_letter,
    key="nr_target",
)

scenario = st.radio(
    "Scenario", ["Single node", "Whole Louvain community (capped)"],
    horizontal=True, key="nr_scenario",
)

if "nr_result" not in st.session_state:
    st.session_state["nr_result"] = None

if target_raw_id is not None:
    node_key = f"{prefix_letter}_{target_raw_id}"
    removal_set = [node_key]
    community_note = None
    if scenario == "Whole Louvain community (capped)":
        community_id = node_info.get(node_key, {}).get("community_id")
        members = sorted(
            (nid for nid, info in node_info.items() if info["community_id"] == community_id),
            key=lambda n: (-node_info[n]["exposed_contract_value"], n),
        )
        if len(members) > UI_COMMUNITY_REMOVAL_CAP:
            community_note = (
                f"Community {community_id} has {len(members):,} members; simulating removal "
                f"of the top {UI_COMMUNITY_REMOVAL_CAP} by exposed value for interactivity. "
                f"The rendered graph below is separately capped at {WHAT_IF_FORCE_GRAPH_NODE_CAP}."
            )
        removal_set = members[:UI_COMMUNITY_REMOVAL_CAP]

    if community_note:
        st.caption(community_note)

    if st.button("Simulate removal", key="nr_simulate", type="primary"):
        result = simulate_removal(graph, removal_set, community_partition=community_partition)
        st.session_state["nr_result"] = {
            "removal_set": removal_set,
            "result": result,
            "mode": mode,
            "scenario": scenario,
        }
else:
    st.info("Select a buyer or supplier above, then click Simulate removal.")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

if st.session_state["nr_result"] is None:
    st.info("No removal simulated yet. Pick an entity and scenario above, then click Simulate removal.")
else:
    saved = st.session_state["nr_result"]
    removal_set = saved["removal_set"]
    res = saved["result"]
    other_label = "Supplier" if saved["mode"] == "Buyer" else "Buyer"

    # graph render (Part 3)
    st.markdown(
        f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>"
        f"Interactive what-if: before and after removal</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Toggle between the capped subgraph as it stands today and the same subgraph with "
        "the removed node(s) dimmed and their ties dashed. Node size encodes exposed contract "
        f"value (log-scaled). Capped at {WHAT_IF_FORCE_GRAPH_NODE_CAP} nodes: the selected "
        "node(s) plus their highest-exposure neighbours."
    )

    subgraph = neighbourhood_subgraph(graph, removal_set, node_info, cap=WHAT_IF_FORCE_GRAPH_NODE_CAP)
    graph_nodes = [
        {
            "id": n["id"],
            "label": _display_id(n["id"]),
            "kind": n["node_type"],
            "degree": n["degree"],
            "community": n["community_id"],
            "exposed_value": n["exposed_contract_value"],
            "is_articulation_point": n["is_articulation_point"],
        }
        for n in subgraph["nodes"]
    ]
    graph_links = subgraph["links"]
    assert len(graph_nodes) <= WHAT_IF_FORCE_GRAPH_NODE_CAP, "what-if graph must never exceed the node cap"
    what_if_force_graph(graph_nodes, graph_links, removed_ids=removal_set)
    why_this_matters(
        f"{len(graph_nodes)} of {WHAT_IF_FORCE_GRAPH_NODE_CAP}-cap nodes shown: the "
        f"{len(removal_set)} removed node(s) plus their highest-exposure neighbours in this "
        f"{total_sample_nodes:,}-node sample. The panel below uses the full simulate_removal "
        f"result, not this capped render."
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:14px;font-weight:700;color:{INK};margin-bottom:8px;'>"
        f"Static fallback (2D, no external script)</div>",
        unsafe_allow_html=True,
    )
    kept_ids = {n["id"] for n in subgraph["nodes"]}
    G_static = nx.Graph()
    for n in subgraph["nodes"]:
        G_static.add_node(n["id"])
    for l in subgraph["links"]:
        G_static.add_edge(l["source"], l["target"])
    pos = nx.spring_layout(G_static, seed=42, k=0.6) if G_static.number_of_nodes() else {}
    edge_x, edge_y = [], []
    for u, v in G_static.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    removed_set = set(removal_set)
    node_ids_static = list(G_static.nodes())
    node_x = [pos[n][0] for n in node_ids_static]
    node_y = [pos[n][1] for n in node_ids_static]
    node_color = [
        "#C85A54" if n in removed_set else (PRIMARY if n.startswith("B_") else "#7A9B6F")
        for n in node_ids_static
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.6, color="#D8D5CC"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=8, color=node_color),
        hovertext=[_display_id(n) for n in node_ids_static], hoverinfo="text",
    ))
    fig.update_layout(
        showlegend=False, height=340, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch", key="nr_static_fallback")
    why_this_matters("Same capped node/link set as the graph above, rendered without any external script, for a browser that cannot load the CDN component.")

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # before/after panel (Part 4)
    st.markdown(
        f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>Structural exposure: before and after</div>",
        unsafe_allow_html=True,
    )
    before_col, after_col = st.columns(2)
    with before_col:
        st.markdown(f"<div style='font-size:13px;font-weight:700;color:{INK_MUTED};margin-bottom:6px;'>BEFORE</div>", unsafe_allow_html=True)
        small_stat_card(f"EUR {res['exposed_contract_value']:,.0f}", "Exposed contract value", EXPOSED_VALUE_CAVEAT)
        small_stat_card(f"{res['sole_source_before']:,}", "Sole-source buyers", "Buyers with exactly one supplier, whole sample")
        small_stat_card(f"{res['largest_component_share_before']:.2%}", "Largest component share", "Share of sample nodes in the giant component")
    with after_col:
        st.markdown(f"<div style='font-size:13px;font-weight:700;color:{INK_MUTED};margin-bottom:6px;'>AFTER</div>", unsafe_allow_html=True)
        small_stat_card(f"{len(res['orphaned_counterparties']):,}", "Orphaned counterparties", "Would lose their only remaining tie")
        small_stat_card(f"{res['sole_source_after']:,}", "Sole-source buyers", "Buyers with exactly one supplier, whole sample")
        small_stat_card(f"{res['largest_component_share_after']:.2%}", "Largest component share", "Share of remaining nodes in the giant component")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    ap_flags = res["is_articulation_point"]
    if len(removal_set) == 1:
        ap_val = "Yes" if ap_flags.get(removal_set[0], False) else "No"
        small_stat_card(ap_val, "Structural cut vertex", "Removing this node alone would split the sample into more pieces")
    else:
        n_ap_in_set = sum(1 for v in ap_flags.values() if v)
        small_stat_card(f"{n_ap_in_set:,} of {len(removal_set):,}", "Structural cut vertices in removal set", "Counted individually against the original graph")

    stranded = res["stranded_buyers"]
    small_stat_card(f"{len(stranded):,}", "Stranded buyers (one-step)", "Buyers left with zero suppliers after this single-step removal; no further cascade is modelled")

    dep = res["dependency_shift"]
    if dep:
        largest_shift = max(abs(v["after"] - v["before"]) for v in dep.values())
        small_stat_card(f"{len(dep):,} buyers affected", "Dependency shift", f"Largest single-buyer shift in top-supplier value share: {largest_shift:.2f}")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    communities_touched = res["communities_touched"]
    if communities_touched:
        st.markdown(
            f"<div style='font-size:14px;font-weight:700;color:{INK};margin-bottom:6px;'>Communities touched</div>",
            unsafe_allow_html=True,
        )
        ct_rows = [
            {"Community": cid, "Internal value removed": f"{v['internal_value_removed_share']:.1%}"}
            for cid, v in communities_touched.items()
        ]
        st.dataframe(ct_rows, width="stretch", hide_index=True, key="nr_communities_touched")
        why_this_matters("Share of a community's own internal edge value that runs through the removed node(s), not a share of the whole sample.")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    admin_show_all = False
    if role == "ADMIN" and not PUBLIC_BUILD and (res["orphaned_counterparties"] or res["stranded_buyers"] or res["substitutability"]):
        admin_show_all = st.checkbox("Reveal named IDs in the lists below (logs one entry)", key="nr_reveal_lists")
        if admin_show_all:
            log_action("ADMIN", "Reveal named entity", f"Network Resilience what-if for {removal_set}", "success")

    def _label_for_list(nid: str) -> str:
        if role == "ADMIN" and not PUBLIC_BUILD:
            return nid[2:] if admin_show_all else "(reveal to view)"
        return _display_id(nid)

    if res["orphaned_counterparties"]:
        st.markdown(
            f"<div style='font-size:14px;font-weight:700;color:{INK};margin-bottom:6px;'>Orphaned counterparties</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            [{"Entity": _label_for_list(nid)} for nid in res["orphaned_counterparties"]],
            width="stretch", hide_index=True, key="nr_orphaned_table",
        )

    substitutability = res["substitutability"]
    if substitutability:
        st.markdown(
            f"<div style='font-size:14px;font-weight:700;color:{INK};margin-bottom:6px;'>Substitutability (existence proxy)</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Count of alternative suppliers active in the same CPV category as the affected "
            "buyer. This is an existence proxy only: it says nothing about supplier capacity, "
            "competitive pricing, or quality, none of which are observable in this data."
        )
        st.dataframe(
            [
                {f"{other_label}": _label_for_list(buyer), "Alternatives (existence only)": count}
                for buyer, count in substitutability.items()
            ],
            width="stretch", hide_index=True, key="nr_substitutability_table",
        )

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

with st.expander("What this data cannot model", expanded=False):
    st.markdown(
        "This screen answers one narrow structural question: how many other entities in the "
        "sampled graph depend on a node as their only or primary counterparty, and how does "
        "removing it reshape the local topology. It cannot and does not estimate:\n\n"
        "- What it would take, in time or resources, to replace a supplier or find a new buyer relationship\n"
        "- Whether a supplier is financially sound or likely to fail\n"
        "- Whether an alternative supplier would actually work in practice: capacity, competitive "
        "pricing, and quality are not observable in this data, so the substitutability figure above "
        "is only whether another supplier is active in the same CPV category\n"
        "- A euro figure representing a monetary consequence of removal, or a competitive-standing "
        "figure; exposed value is a factual sum of award value only\n"
        "- Whether removal is imminent or likely, or whether the removed entity has done anything wrong"
    )

neutral_methodology_note(
    "What this screen does and does not measure",
    "Removal here means the node(s) and their edges are deleted from a fixed, 35% "
    "value-stratified sample of the training-period graph (seed 42, deterministic), and the "
    "resulting topology is recomputed by src.rq4_resilience.resilience.simulate_removal(). "
    "It does not simulate a market response, does not re-fit any model, and does not imply "
    "that removal is imminent or that the selected entity is under suspicion.",
)
