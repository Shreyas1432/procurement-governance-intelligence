import json
from pathlib import Path

import polars as pl
import streamlit as st

# Risk classes actually produced by the pipeline (LOW/MEDIUM/HIGH). The original
# spec mentioned CRITICAL/STABLE, which this pipeline does not emit.
RISK_CLASS_OPTIONS = ["All", "HIGH", "MEDIUM", "LOW"]


def global_risk_selectbox():
    """Render the global Risk Class selector in the sidebar (CHANGE 1). Returns the
    selected value. Uses a shared session key so the choice persists across pages."""
    return st.sidebar.selectbox(
        "Risk Class", options=RISK_CLASS_OPTIONS, key="global_risk_class",
        help="Global filter applied to every tab (where a risk_class column exists).",
    )


def apply_global_risk_filter(df: pl.DataFrame) -> pl.DataFrame:
    """Filter ``df`` by the global Risk Class selection. No-op when 'All' is
    selected or when the dataframe has no risk_class column (e.g. anomaly/network
    tables), so it is always safe to call."""
    sel = st.session_state.get("global_risk_class", "All")
    if sel and sel != "All" and isinstance(df, pl.DataFrame) and "risk_class" in df.columns:
        return df.filter(pl.col("risk_class") == sel)
    return df


def global_risk_banner(df: pl.DataFrame) -> None:
    """Show the active-filter banner at the top of a tab (CHANGE 1)."""
    sel = st.session_state.get("global_risk_class", "All")
    if sel and sel != "All":
        n = df.height if isinstance(df, pl.DataFrame) and "risk_class" in df.columns else 0
        st.warning(f"Showing: {sel} risk entities only ({n:,} records)")


def render_provenance_sidebar(root: Path) -> None:
    """DEPRECATED: kept only so any un-migrated caller doesn't crash. Data
    provenance now belongs in a collapsed main-body expander (see
    `render_provenance_expander`), not front-and-centre in the sidebar."""
    render_provenance_expander(root)


def render_provenance_expander(root: Path, expanded: bool = False) -> None:
    """Data-provenance in a collapsed 'Methodology / provenance' expander in the
    main body, not shown unconditionally to every end user in the sidebar."""
    with st.expander("Methodology / provenance", expanded=expanded):
        meta_path = Path(root) / "data" / "results" / "pipeline_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
            tr = meta.get("total_records")
            tr_str = f"{tr:,}" if isinstance(tr, (int, float)) else "N/A"
            st.markdown(f"Pipeline run: {meta.get('run_date', 'N/A')}")
            st.markdown(f"Source records: {tr_str}")
            st.markdown(
                f"Temporal split: Train <= {meta.get('train_cutoff', 2018)}, "
                f"Test >= {meta.get('test_start', 2019)}"
            )
        else:
            st.markdown("Pipeline metadata not found.")
        st.caption(
            "This dashboard reads locked parquet/JSON artifacts only. It never "
            "calls .fit()/.predict() and never imports sklearn/xgboost."
        )


RISK_CAVEAT = (
    "**Governance-risk label = procedural-recording irregularity, not corruption.** "
    "The label's amendment arm is regionally concentrated (NUTS ITH5, 59.1% on n=88). "
    "Do not read a named entity as 'corrupt' or 'guilty' from this dashboard."
)


def render_risk_caveat() -> None:
    """Mandatory caveat banner for every risk surface."""
    st.warning(RISK_CAVEAT)


def cascading_entity_options(
    unified: pl.DataFrame, id_col: str, risk_class: str
) -> list:
    """Entities present in `unified`, filtered to those with at least one
    contract in the selected global Risk Class (cascading filter). With
    'All' selected, every entity is returned unfiltered."""
    df = unified
    if risk_class and risk_class != "All" and "risk_class" in df.columns:
        df = df.filter(pl.col("risk_class") == risk_class)
    return df[id_col].drop_nulls().unique().sort().to_list()


def why_flagged_panel(entity_rows: pl.DataFrame, entity_label: str) -> None:
    """Concise 'why flagged' panel for a selected entity, derived from locked
    RQ2 rule-label inputs. Factual and short, no speculation, no financial
    claims. entity_rows must contain the governance-prediction columns for
    the contracts belonging to this entity."""
    if entity_rows is None or entity_rows.height == 0:
        st.info(f"No governance-risk records found for {entity_label}.")
        return

    cols = entity_rows.columns
    st.markdown(f"**Why flagged: {entity_label}**")

    lines = []
    if "procedure_type" in cols:
        top_proc = (
            entity_rows.group_by("procedure_type")
            .agg(pl.len().alias("n"))
            .filter(pl.col("procedure_type") != "UNKNOWN")
            .sort("n", descending=True)
        )
        if top_proc.height > 0:
            p = top_proc.row(0, named=True)
            lines.append(f"- Most common procedure type: **{p['procedure_type']}** ({p['n']:,} contracts)")
    if "bid_count" in cols:
        low_bid = entity_rows.filter(pl.col("bid_count") <= 1).height
        if low_bid > 0:
            lines.append(f"- **{low_bid:,}** contract(s) recorded with a single bidder (bid_count <= 1)")
    if "buyer_dependency_normalized" in cols:
        dep = entity_rows["buyer_dependency_normalized"].mean()
        if dep is not None:
            lines.append(f"- Mean buyer-dependency ratio: **{dep:.1%}**")
    if "contract_amendments" in cols:
        amend_rate = (entity_rows["contract_amendments"] > 0).mean()
        if amend_rate is not None:
            lines.append(f"- Amendment rate: **{amend_rate:.1%}** of contracts")
    if "ensemble_risk_prob" in cols:
        mean_prob = entity_rows["ensemble_risk_prob"].mean()
        if mean_prob is not None:
            lines.append(
                f"- Mean governance-risk probability: **{mean_prob:.3f}** "
                f"(operating threshold: 0.356, PR-AUC 0.8336)"
            )

    if lines:
        st.markdown("\n".join(lines))
    else:
        st.info("Rule-label inputs not available in the loaded data for this entity.")
    st.caption(
        "This reflects the RQ2 rule-label inputs (procedure type, bid count, "
        "buyer dependency, amendments) and the value-model's PR/threshold context, "
        "not a corruption determination."
    )


def select_or_all(label: str, values, key: str):
    options = ["ALL"] + [str(v) for v in values if v is not None]
    return st.selectbox(label, options, key=key)


def apply_common_filters(df: pl.DataFrame, buyer="ALL", supplier="ALL", cpv="ALL", year_range=None) -> pl.DataFrame:
    out = df
    if buyer != "ALL" and "buyer_id" in out.columns:
        out = out.filter(pl.col("buyer_id").cast(str) == buyer)
    if supplier != "ALL" and "supplier_id" in out.columns:
        out = out.filter(pl.col("supplier_id").cast(str) == supplier)
    if cpv != "ALL" and "cpv_division" in out.columns:
        out = out.filter(pl.col("cpv_division").cast(str) == cpv)
    if year_range and "award_year" in out.columns:
        out = out.filter(pl.col("award_year").is_between(year_range[0], year_range[1]))
    return out
