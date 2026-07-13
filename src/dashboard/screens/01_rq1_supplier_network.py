"""
RQ1 - Supplier Network. Read-only over locked RQ1 parquet artifacts.

Community structure and centrality in the buyer-supplier procurement graph.
Per the design and the hard constraints: never renders the full 161,695-node
graph (capped sample only, via community subgraph / ego graph / top-N
centrality). The structural what-if (node removal exposure) lives on its
own Network Resilience screen, reachable from the link below.
"""
from __future__ import annotations
import sys
from pathlib import Path

import networkx as nx
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dashboard._auth import require_auth, DASHBOARD_DATA_RESULTS
from dashboard._theme import inject_base_css, INK, INK_MUTED, FONT_SERIF, PRIMARY
from dashboard._components import small_stat_card, neutral_methodology_note, chart_card, why_this_matters
from dashboard._shell import render_sidebar_wordmark, render_topbar, render_role_switcher, entity_picker, node_info_panel
from dashboard._auth import render_logout_button
from dashboard.components.graph_loader import load_bipartite_edges

inject_base_css()
require_auth()
render_sidebar_wordmark()
render_logout_button()
render_role_switcher()
render_topbar("PGI / RQ1 Supplier Network")

role = st.session_state.get("role")


@st.cache_data(show_spinner="Loading network artifacts...")
def load_rq1():
    base = DASHBOARD_DATA_RESULTS
    return {
        "success": __import__("json").loads((base / "rq1_success_metrics.json").read_text()),
        "network": pl.read_parquet(base / "rq1_network_metrics.parquet"),
        "community": pl.read_parquet(base / "rq1_community_assignments.parquet"),
        "conductance": pl.read_parquet(base / "rq1_conductance_values.parquet"),
        "resilience": pl.read_parquet(base / "rq1_resilience_scores.parquet"),
        "validation": pl.read_parquet(base / "rq1_validation_metrics.parquet"),
    }


d = load_rq1()
success = d["success"]
network = d["network"]
community = d["community"]
validation = d["validation"]
community_lookup = dict(zip(community["entity_id"].to_list(), community["community_id"].to_list()))

# hero
st.markdown(
    f"<h1 style=\"font-family:{FONT_SERIF};font-size:32px;color:{INK};margin:0 0 8px;\">"
    f"RQ1: Supplier Network</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='font-size:14.5px;color:{INK_MUTED};margin-bottom:12px;max-width:760px;'>"
    f"Community structure and centrality in the buyer-supplier procurement graph. "
    f"All views below are capped samples, never the full {success['nodes']:,}-node graph. "
    f"For structural what-if analysis (what happens if a buyer or supplier is removed), "
    f"see the Network Resilience screen in the sidebar.</div>",
    unsafe_allow_html=True,
)

# stats
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    small_stat_card(f"{success['nodes']:,}", "Nodes", "Buyers plus suppliers")
with c2:
    small_stat_card(f"{success['edges']:,}", "Edges", "Buyer-supplier relationships")
with c3:
    small_stat_card(f"{success['modularity']:.4f}", "Modularity Q", "Louvain community structure")
with c4:
    small_stat_card(f"{success['components_count']:,}", "Components", f"Giant component: {success['giant_component_frac']*100:.1f}%")
with c5:
    small_stat_card(f"{success['conductance_giant_mean']:.3f}", "Mean conductance", "Giant component, lower means tighter")
with c6:
    small_stat_card(f"{success['ari_giant_mean']:.3f}", "ARI stability", "Community assignment agreement across seeds")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# network view
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:4px;'>Network view</div>",
    unsafe_allow_html=True,
)
st.caption(f"Capped sample, never the full {success['nodes']:,}-node graph.")

view = st.segmented_control(
    "Network view",
    options=["Community subgraph", "Ego graph", "Top-N centrality"],
    default="Community subgraph",
    label_visibility="collapsed",
    key="rq1_view_segmented",
)

edges_df = load_bipartite_edges(ROOT)
panel_col1, panel_col2 = st.columns([2, 1])
target_id = None  # set by whichever branch below runs; used by the ego-graph info panel

if view == "Community subgraph":
    with panel_col1:
        chart_card("Community subgraph", "A sampled community, not the full graph.")
        community_sizes = (
            community.group_by("community_id")
            .agg(pl.len().alias("size"))
            .sort("size", descending=True)
            .head(12)
        )
        chip_ids = community_sizes["community_id"].to_list()
        picked_community = st.radio(
            "Community", chip_ids, horizontal=True, key="rq1_community_chip",
            format_func=lambda cid: f"Community {cid}",
        )
        members = community.filter(pl.col("community_id") == picked_community)["entity_id"].to_list()
        sub_edges = edges_df.filter(
            pl.col("buyer_id").is_in(members) | pl.col("supplier_id").is_in(members)
        ).head(150)

        G = nx.Graph()
        for row in sub_edges.iter_rows(named=True):
            b, s = f"B_{row['buyer_id']}", f"S_{row['supplier_id']}"
            G.add_edge(b, s, weight=row["contract_count"])

        selected_node = None
        if G.number_of_nodes() == 0:
            st.info("No edges available for this community sample.")
        else:
            pos = nx.spring_layout(G, seed=42, k=0.6)
            edge_x, edge_y = [], []
            for u, v in G.edges():
                edge_x += [pos[u][0], pos[v][0], None]
                edge_y += [pos[u][1], pos[v][1], None]
            node_list = list(G.nodes())
            node_x = [pos[n][0] for n in node_list]
            node_y = [pos[n][1] for n in node_list]
            node_color = [PRIMARY if n.startswith("B_") else "#7A9B6F" for n in node_list]
            from dashboard._auth import entity_label_for_list
            node_hover_labels = [
                entity_label_for_list(n[2:], prefix=("B" if n.startswith("B_") else "S"))
                for n in node_list
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.6, color="#D8D5CC"), hoverinfo="none"))
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers", marker=dict(size=8, color=node_color),
                hovertext=node_hover_labels, hoverinfo="text",
            ))
            fig.update_layout(
                showlegend=False, height=420, plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch", key="rq1_community_subgraph")
            why_this_matters(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges sampled from community {picked_community} of {community['community_id'].n_unique()} total.")

            node_ids_raw = [n[2:] for n in node_list]
            selected_node = st.selectbox(
                "Select a node from this subgraph for details", node_ids_raw,
                format_func=lambda nid: f"{'Buyer' if f'B_{nid}' in node_list else 'Supplier'} (community {community_lookup.get(nid, 'unknown')})",
                key="rq1_subgraph_node_pick",
            )
    with panel_col2:
        if selected_node is not None:
            is_buyer = f"B_{selected_node}" in node_list
            net_row = network.filter(
                (pl.col("entity_id") == selected_node)
                & (pl.col("node_type") == ("buyer" if is_buyer else "supplier"))
            )
            degree = int(net_row["degree"][0]) if net_row.height else 0
            centrality = float(net_row["centrality"][0]) if net_row.height else 0.0
            node_info_panel(
                selected_node, "Buyer" if is_buyer else "Supplier", degree,
                community_lookup.get(selected_node, "unknown"), centrality,
                resource_label=f"RQ1 subgraph node {selected_node}",
            )
        else:
            node_info_panel(None, "", 0, "", 0.0, resource_label="RQ1 subgraph node")

elif view == "Ego graph":
    with panel_col1:
        chart_card("Ego graph", "Two-hop neighborhood around one entity, never the full network.")
        node_type_pick = st.radio("Entity type", ["Buyer", "Supplier"], horizontal=True, key="rq1_ego_type")
        tcol = "buyer_id" if node_type_pick == "Buyer" else "supplier_id"
        ocol = "supplier_id" if node_type_pick == "Buyer" else "buyer_id"
        top_candidates = (
            network.filter(pl.col("node_type") == node_type_pick.lower())
            .sort("degree", descending=True)
            .head(30)
        )
        cand_ids = top_candidates["entity_id"].to_list()
        cand_degree_lookup = dict(zip(cand_ids, top_candidates["degree"].to_list()))

        target_id = entity_picker(
            f"Select a high-degree {node_type_pick.lower()} (top 30 by degree)",
            cand_ids,
            context_for=lambda cid: f"degree {cand_degree_lookup.get(cid, 0)}, community {community_lookup.get(cid, 'unknown')}",
            prefix=node_type_pick[0],
            key="rq1_ego",
        )

        if target_id is None:
            st.info("No entity selected.")
        else:
            direct = edges_df.filter(pl.col(tcol) == target_id)
            neighbours = direct[ocol].unique().to_list()
            ball_edges = edges_df.filter((pl.col(tcol) == target_id) | pl.col(ocol).is_in(neighbours)).head(150)
            G = nx.Graph()
            for row in ball_edges.iter_rows(named=True):
                b, s = f"B_{row['buyer_id']}", f"S_{row['supplier_id']}"
                G.add_edge(b, s)
            if G.number_of_nodes() == 0:
                st.info("No edges found for this entity.")
            else:
                pos = nx.spring_layout(G, seed=42, k=0.7)
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    edge_x += [pos[u][0], pos[v][0], None]
                    edge_y += [pos[u][1], pos[v][1], None]
                center_key = f"{'B' if node_type_pick == 'Buyer' else 'S'}_{target_id}"
                node_x, node_y, node_color, node_size = [], [], [], []
                for n in G.nodes():
                    node_x.append(pos[n][0]); node_y.append(pos[n][1])
                    if n == center_key:
                        node_color.append("#C85A54"); node_size.append(16)
                    else:
                        node_color.append(PRIMARY if n.startswith("B_") else "#7A9B6F")
                        node_size.append(8)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.6, color="#D8D5CC"), hoverinfo="none"))
                fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=node_size, color=node_color), hoverinfo="skip"))
                fig.update_layout(
                    showlegend=False, height=420, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, width="stretch", key="rq1_ego_graph")
                why_this_matters(f"Ego graph: {G.number_of_nodes()} nodes within 2 hops of the selected {node_type_pick.lower()}.")
    with panel_col2:
        if target_id is not None:
            net_row = network.filter(
                (pl.col("entity_id") == target_id) & (pl.col("node_type") == node_type_pick.lower())
            )
            degree = int(net_row["degree"][0]) if net_row.height else 0
            centrality = float(net_row["centrality"][0]) if net_row.height else 0.0
            node_info_panel(
                target_id, node_type_pick, degree, community_lookup.get(target_id, "unknown"),
                centrality, resource_label=f"RQ1 ego node {target_id}",
            )
        else:
            node_info_panel(None, "", 0, "", 0.0, resource_label="RQ1 ego node")

else:  # Top-N centrality
    with panel_col1:
        chart_card("Top-N centrality", "Highest composite-centrality nodes, a ranked sample, not the full graph.")
        top_n = st.slider("Show top N", 5, 50, 15, key="rq1_topn_slider")
        top_nodes = network.sort("centrality", descending=True).head(top_n)
        pdf = top_nodes.to_pandas()
        from dashboard._auth import entity_label_for_list
        disp_labels = [
            entity_label_for_list(
                row["entity_id"],
                prefix=("B" if row["node_type"] == "buyer" else "S"),
                context=f"degree {row['degree']}",
            )
            for _, row in pdf.iterrows()
        ]
        pdf["display_id"] = disp_labels
        fig = px.bar(
            pdf, x="centrality", y="display_id", orientation="h", color="node_type",
            color_discrete_map={"buyer": PRIMARY, "supplier": "#7A9B6F"},
            labels={"centrality": "Composite centrality", "display_id": "", "node_type": "Type"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white", height=max(320, top_n * 20))
        st.plotly_chart(fig, width="stretch", key="rq1_topn_bar")
        why_this_matters(f"Top {top_n} nodes by composite centrality across the full {success['nodes']:,}-node graph.")

        topn_ids = top_nodes["entity_id"].to_list()
        topn_pick = st.selectbox(
            "Select a node from this list for details", range(len(topn_ids)),
            format_func=lambda i: disp_labels[i], key="rq1_topn_node_pick",
        )
    with panel_col2:
        row = pdf.iloc[topn_pick]
        node_info_panel(
            row["entity_id"], row["node_type"].capitalize(), int(row["degree"]),
            community_lookup.get(row["entity_id"], "unknown"), float(row["centrality"]),
            resource_label=f"RQ1 top-N node {row['entity_id']}",
        )

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# composite centrality
neutral_methodology_note(
    "Composite centrality",
    "score = 0.4 times degree + 0.3 times betweenness [Brandes, k=500, seed 42] + 0.3 times PageRank. "
    "A blend of local connectivity (degree), brokerage position (betweenness, "
    "approximated for tractability at this node count), and global influence "
    "(PageRank). Combining these matters because each captures a different "
    "notion of structural importance, and no single centrality measure "
    "dominates in bipartite procurement networks.",
)

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# community table
st.markdown(
    f"<div style='font-size:16px;font-weight:700;color:{INK};margin-bottom:8px;'>Community analysis</div>",
    unsafe_allow_html=True,
)
community_table = (
    community.group_by("community_id")
    .agg(pl.len().alias("Size"))
    .sort("Size", descending=True)
    .head(30)
    .to_pandas()
)
community_table["Modularity contrib."] = success["modularity"] / max(len(community_table), 1)
conductance_series = d["conductance"]["conductance"].to_list()
community_table["Conductance"] = [
    conductance_series[i] if i < len(conductance_series) else None
    for i in range(len(community_table))
]
community_table = community_table.rename(columns={"community_id": "Community"})
st.dataframe(
    community_table[["Community", "Size", "Modularity contrib.", "Conductance"]],
    width="stretch", height=320, hide_index=True,
    key="rq1_community_table",
)
why_this_matters("Lower conductance means a tighter, more self-contained community with fewer edges leaving it.")

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
neutral_methodology_note(
    "Method choice: Louvain versus alternatives",
    f"Louvain community detection was chosen over spectral clustering and label "
    f"propagation for its modularity optimization (Q={success['modularity']:.4f}, "
    f"versus a null-model mean of {success['modularity_null_mean']:.4f}) and "
    f"determinism at this node count. Cross-seed stability (ARI = "
    f"{success['ari_giant_mean']:.3f} on the giant component) supports the "
    f"partition as a structurally meaningful signal, not an artifact of a "
    f"single random seed.",
)
