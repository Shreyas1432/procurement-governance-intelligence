"""
Shared Plotly chart builders for the PGI dashboard.
All functions return a Plotly figure and accept Polars DataFrames.
"""
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pandas.core.groupby.generic import DataFrameGroupBy

# Patch pandas 3.0+ groupby compatibility with Plotly Express
_orig_get_group = DataFrameGroupBy.get_group
def _patched_get_group(self, name):
    try:
        return _orig_get_group(self, name)
    except KeyError:
        if not isinstance(name, tuple):
            try:
                return _orig_get_group(self, (name,))
            except KeyError:
                pass
        raise
DataFrameGroupBy.get_group = _patched_get_group

RISK_COLORS = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
TEAL = "#0f766e"
TEAL_LIGHT = "#ccfbf1"



# ── Risk Distribution ────────────────────────────────────────────────────────

def risk_histogram(df: pl.DataFrame) -> go.Figure:
    pdf = df.to_pandas()
    fig = px.histogram(
        pdf, x="unified_risk_score", color="risk_class",
        nbins=40,
        color_discrete_map=RISK_COLORS,
        title="Unified Risk Score Distribution",
        labels={"unified_risk_score": "Risk Score (0-1)", "count": "Contracts"},
        category_orders={"risk_class": ["HIGH", "MEDIUM", "LOW"]},
    )
    fig.update_layout(bargap=0.05, legend_title="Risk Class",
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def risk_heatmap(df: pl.DataFrame) -> go.Figure:
    if df.is_empty():
        return go.Figure()
    pivot = (
        df.group_by(["cpv_division", "award_year"])
        .agg(pl.mean("unified_risk_score").alias("risk"))
        .filter(pl.col("cpv_division") != "UNKNOWN")
        .sort(["award_year", "cpv_division"])
        .to_pandas()
    )
    if pivot.empty:
        return go.Figure()
    fig = px.density_heatmap(
        pivot, x="award_year", y="cpv_division", z="risk",
        histfunc="avg",
        color_continuous_scale="RdYlGn_r",
        title="Governance Risk Heat Map (CPV Division × Year)",
        labels={"award_year": "Award Year", "cpv_division": "CPV Division", "risk": "Avg Risk"},
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def risk_trend(df: pl.DataFrame) -> go.Figure:
    if df.is_empty():
        return go.Figure()
    trend = (
        df.group_by("award_year")
        .agg(pl.mean("unified_risk_score").alias("avg_risk"),
             pl.len().alias("contracts"))
        .sort("award_year")
        .to_pandas()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["award_year"], y=trend["avg_risk"],
        mode="lines+markers", name="Avg Risk",
        line=dict(color=TEAL, width=2),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title="Mean Risk Score Trend by Year",
        xaxis_title="Award Year", yaxis_title="Mean Risk Score",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def risk_class_pie(df: pl.DataFrame) -> go.Figure:
    if df.is_empty():
        return go.Figure()
    counts = (
        df.group_by("risk_class")
        .agg(pl.len().alias("count"))
        .to_pandas()
    )
    fig = px.pie(
        counts, values="count", names="risk_class",
        color="risk_class",
        color_discrete_map=RISK_COLORS,
        title="Risk Class Breakdown",
        hole=0.45,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(showlegend=False, paper_bgcolor="white")
    return fig


# ── Supplier Charts ──────────────────────────────────────────────────────────

def supplier_risk_timeline(df: pl.DataFrame) -> go.Figure:
    if df.is_empty():
        return go.Figure()
    trend = (
        df.group_by("award_year")
        .agg(pl.mean("unified_risk_score").alias("avg_risk"),
             pl.len().alias("contracts"))
        .sort("award_year")
        .to_pandas()
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trend["award_year"], y=trend["contracts"],
        name="Contracts", yaxis="y2",
        marker_color="#e2e8f0", opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=trend["award_year"], y=trend["avg_risk"],
        mode="lines+markers", name="Avg Risk",
        line=dict(color=TEAL, width=2),
        marker=dict(size=7, color=TEAL)
    ))
    fig.update_layout(
        title="Supplier Risk & Activity Timeline",
        xaxis_title="Year",
        yaxis=dict(title="Mean Risk Score", range=[0, 1]),
        yaxis2=dict(title="Contract Count", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def buyer_relationship_bar(df: pl.DataFrame) -> go.Figure:
    """Top buyers by risk contribution for a supplier.

    Not called from any screen today. Renders raw buyer_id with no
    anonymize_id()/entity_label_for_list() pass -- do not wire this into a
    screen without adding one, or it will leak real IDs under PUBLIC_BUILD.
    """
    if df.is_empty():
        return go.Figure()
    agg = (
        df.group_by("buyer_id")
        .agg(pl.len().alias("contracts"),
             pl.mean("unified_risk_score").alias("avg_risk"))
        .sort("contracts", descending=True)
        .head(20)
        .to_pandas()
    )
    fig = px.bar(
        agg, y="buyer_id", x="contracts",
        color="avg_risk", color_continuous_scale="RdYlGn_r",
        orientation="h",
        title="Top Buyer Relationships",
        labels={"buyer_id": "Buyer ID", "contracts": "Contract Count", "avg_risk": "Avg Risk"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def centrality_gauge(centrality: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=centrality * 100,
        title={"text": "Network Centrality (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": TEAL},
            "steps": [
                {"range": [0, 30], "color": TEAL_LIGHT},
                {"range": [30, 70], "color": "#99f6e4"},
                {"range": [70, 100], "color": "#2dd4bf"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.75, "value": 30},
        },
    ))
    fig.update_layout(height=250, paper_bgcolor="white")
    return fig


# ── Buyer Charts ─────────────────────────────────────────────────────────────

def supplier_relationship_bar(df: pl.DataFrame) -> go.Figure:
    """Top suppliers for a given buyer.

    Not called from any screen today. Renders raw supplier_id with no
    anonymize_id()/entity_label_for_list() pass -- do not wire this into a
    screen without adding one, or it will leak real IDs under PUBLIC_BUILD.
    """
    if df.is_empty():
        return go.Figure()
    agg = (
        df.group_by("supplier_id")
        .agg(pl.len().alias("contracts"),
             pl.mean("unified_risk_score").alias("avg_risk"))
        .sort("contracts", descending=True)
        .head(20)
        .to_pandas()
    )
    fig = px.bar(
        agg, y="supplier_id", x="contracts",
        color="avg_risk", color_continuous_scale="RdYlGn_r",
        orientation="h",
        title="Top Supplier Relationships (by Contract Count)",
        labels={"supplier_id": "Supplier ID", "contracts": "Contracts", "avg_risk": "Avg Risk"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def dependency_gauge(dep: float) -> go.Figure:
    """Buyer dependency ratio gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dep * 100,
        number={"suffix": "%"},
        delta={"reference": 50, "valueformat": ".1f"},
        title={"text": "Supplier Dependency<br><span style='font-size:0.8em'>Top-5 Concentration</span>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#f59e0b"},
            "steps": [
                {"range": [0, 40], "color": "#dcfce7"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.75, "value": 70},
        },
    ))
    fig.update_layout(height=250, paper_bgcolor="white")
    return fig


def buyer_risk_timeline(df: pl.DataFrame) -> go.Figure:
    return supplier_risk_timeline(df)  # same logic


def amendment_rate_bar(df: pl.DataFrame) -> go.Figure:
    """Procedure type × amendment rate breakdown."""
    if "procedure_type" not in df.columns or df.is_empty():
        return go.Figure()
    agg = (
        df.group_by("procedure_type")
        .agg(
            pl.len().alias("total"),
            pl.mean("contract_amendments").alias("avg_amendments"),
        )
        .filter(pl.col("procedure_type") != "UNKNOWN")
        .sort("avg_amendments", descending=True)
        .to_pandas()
    )
    if agg.empty:
        return go.Figure()
    fig = px.bar(
        agg, x="procedure_type", y="avg_amendments",
        color="avg_amendments", color_continuous_scale="RdYlGn_r",
        title="Amendment Rate by Procedure Type",
        labels={"procedure_type": "Procedure", "avg_amendments": "Avg Amendments"},
    )
    fig.update_layout(xaxis_tickangle=-30,
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


# ── Anomaly Charts ───────────────────────────────────────────────────────────

def anomaly_scatter(df: pl.DataFrame) -> go.Figure:
    """Price anomaly map, estimated vs final price.

    Not called from any screen today. hover_data includes raw contract_id/
    buyer_id/supplier_id with no anonymize_id()/entity_label_for_list() pass
    -- do not wire this into a screen without adding one, or it will leak
    real IDs under PUBLIC_BUILD.
    """
    if df.is_empty():
        return go.Figure()
    pdf = df.filter(
        (pl.col("final_price") > 0) & (pl.col("estimated_price") > 0)
    ).to_pandas()
    if pdf.empty:
        return go.Figure()
    pdf["Anomaly"] = pdf["consensus_flag"].map({True: "Consensus Anomaly", False: "Normal"})
    fig = px.scatter(
        pdf, x="estimated_price", y="final_price",
        color="Anomaly",
        color_discrete_map={"Consensus Anomaly": "#ef4444", "Normal": "#94a3b8"},
        opacity=0.6,
        hover_data=["contract_id", "buyer_id", "supplier_id", "cpv_division", "anomaly_score"],
        log_x=True, log_y=True,
        title="Price Anomaly Map: Estimated vs Final Price",
        labels={"estimated_price": "Estimated Price (€, log)", "final_price": "Final Price (€, log)"},
    )
    # Diagonal reference line
    max_val = max(pdf["estimated_price"].max(), pdf["final_price"].max())
    min_val = max(pdf[["estimated_price", "final_price"]].min().min(), 1)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Fair Price",
        line=dict(color="#64748b", dash="dash", width=1),
    ))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def residual_histogram(df: pl.DataFrame) -> go.Figure:
    if df.is_empty() or "gb_residual" not in df.columns:
        return go.Figure()
    pdf = df.to_pandas()
    pdf["Anomaly"] = pdf["consensus_flag"].map({True: "Anomaly", False: "Normal"})
    fig = px.histogram(
        pdf, x="gb_residual", color="Anomaly", nbins=60,
        color_discrete_map={"Anomaly": "#ef4444", "Normal": "#94a3b8"},
        barmode="overlay",
        title="Price Residual Distribution (GB Model)",
        labels={"gb_residual": "Log-Price Residual", "count": "Contracts"},
        opacity=0.75,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#0f766e", annotation_text="Expected")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


def anomaly_method_bar(df: pl.DataFrame) -> go.Figure:
    if df.is_empty():
        return go.Figure()
    methods = {
        "GB Regressor": int(df["gb_flag"].sum()) if "gb_flag" in df.columns else 0,
        "Local Outlier Factor": int(df["lof_flag"].sum()) if "lof_flag" in df.columns else 0,
        "Isolation Forest": int(df["iso_flag"].sum()) if "iso_flag" in df.columns else 0,
        "Consensus (all 3)": int(df["consensus_flag"].sum()) if "consensus_flag" in df.columns else 0,
    }
    fig = go.Figure(go.Bar(
        x=list(methods.keys()), y=list(methods.values()),
        marker_color=["#94a3b8", "#94a3b8", "#94a3b8", "#ef4444"],
    ))
    fig.update_layout(
        title="Anomalies Detected by Method",
        xaxis_title="Detection Method", yaxis_title="Anomaly Count",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def anomaly_score_dist(df: pl.DataFrame) -> go.Figure:
    if df.is_empty():
        return go.Figure()
    pdf = df.to_pandas()
    fig = px.histogram(
        pdf, x="anomaly_score", nbins=20,
        color_discrete_sequence=[TEAL],
        title="Anomaly Score Distribution",
        labels={"anomaly_score": "Anomaly Score (0-1)", "count": "Contracts"},
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    return fig


# ── Network Charts ───────────────────────────────────────────────────────────

def centrality_distribution(df: pl.DataFrame) -> go.Figure:
    """Histogram of centrality scores for all network nodes."""
    if df.is_empty():
        return go.Figure()
    suppliers = df.filter(pl.col("node_type") == "supplier").to_pandas()
    buyers = df.filter(pl.col("node_type") == "buyer").to_pandas()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=suppliers["centrality"], name="Suppliers",
                               marker_color="#0f766e", opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=buyers["centrality"], name="Buyers",
                               marker_color="#f59e0b", opacity=0.7, nbinsx=50))
    fig.update_layout(
        barmode="overlay",
        title="Network Centrality Distribution",
        xaxis_title="Centrality Score", yaxis_title="Nodes",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def relationship_bar(df: pl.DataFrame, entity_col: str, value_col: str, title: str) -> go.Figure:
    if df.is_empty():
        return go.Figure()
    pdf = df.sort(value_col, descending=True).head(20).to_pandas()
    return px.bar(pdf, x=value_col, y=entity_col, orientation="h", title=title,
                  color_discrete_sequence=[TEAL])
