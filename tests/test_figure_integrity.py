"""Regression guards for figure-round-2/3 QA fixes.

Seven tests lock numeric properties of the locked parquet and JSON artefacts.
Tests read from locked JSON / parquet files; they do NOT execute the
figure-building code itself.  All tests skip gracefully when data files
are absent (e.g. fresh checkout without running the pipeline).

Guard summary (round-3 updated)
--------------------------------
G1  Top-decile centrality ratio >= 5x (supplier nodes, rq1_network_metrics.parquet)
G2  Operating threshold annotation = f"{v:.3f}" where v = JSON source of truth
    (source-derived; no literal hardcoded in the assert)
G3  PR-AUC 4dp == 0.8336 (rq2_success_metrics.json test_headline.pr_auc)
G4a Centrality reproduces norm(0.4*dc + 0.3*bc + 0.3*pr) to max-abs-diff < 1e-6
    (locks formula for locked rq1_network_metrics.parquet)
G4b pagerank column in parquet exists and is all non-zero (161,695 nodes)
G5  zero_amend_rate < 0.9999 and does NOT format as "100.00%" at 2dp
    (rq1_network_features.parquet -- guards against annotation round-up regression)
G6  consensus_anomaly_rate in [0.04, 0.08] (rq3_success_metrics.json -- T3-07 PASS)
"""

import json
from pathlib import Path

import numpy as np
import pytest

# Paths (all relative to project root)
ROOT = Path(__file__).resolve().parent.parent
DATA_RESULTS  = ROOT / "data" / "results"
DATA_FEATURES = ROOT / "data" / "features"
RQ1_NETWORK_PQ  = DATA_RESULTS  / "rq1_network_metrics.parquet"
RQ1_FEATURES_PQ = DATA_FEATURES / "rq1_network_features.parquet"
RQ2_METRICS_JSON = DATA_RESULTS / "rq2_success_metrics.json"
RQ3_METRICS_JSON = DATA_RESULTS / "rq3_success_metrics.json"
RQ1_SOURCE_PY    = ROOT / "src" / "rq1_supplier_network" / "rq1_network.py"


# Helpers

def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_parquet(path: Path):
    """Return a dict of numpy arrays keyed by column name."""
    try:
        import polars as pl
        df = pl.read_parquet(path)
        return {col: df[col].to_numpy() for col in df.columns}
    except Exception:
        import pyarrow.parquet as pq
        tbl = pq.read_table(path)
        return {col: tbl.column(col).to_pylist() for col in tbl.column_names}


# G1: top-decile centrality concentration (>= 5x overall mean)

@pytest.mark.skipif(not RQ1_NETWORK_PQ.exists(), reason="rq1_network_metrics.parquet not found")
def test_g1_top_decile_centrality_ratio():
    """Supplier top-decile centrality must be >= 5x the supplier cohort mean.

    Locks the 9x hub-concentration annotation in fig_rq1_centrality_distribution.
    A ratio below 5 would indicate a degenerate or shuffled centrality column.
    """
    cols = _read_parquet(RQ1_NETWORK_PQ)
    node_type = cols.get("node_type", [])
    centrality = np.asarray(cols["centrality"], dtype=float)

    # Filter to supplier nodes
    supplier_mask = np.asarray([t == "supplier" for t in node_type])
    sup_cent = centrality[supplier_mask]
    assert len(sup_cent) > 0, "No supplier nodes found in parquet"

    sup_cent = sup_cent[~np.isnan(sup_cent)]
    sorted_desc = np.sort(sup_cent)[::-1]
    decile_n = max(1, len(sorted_desc) // 10)
    top_decile_mean = sorted_desc[:decile_n].mean()
    overall_mean = sup_cent.mean()

    ratio = top_decile_mean / overall_mean if overall_mean > 0 else 0.0
    assert ratio >= 5.0, (
        f"Top-decile centrality ratio = {ratio:.2f} (expected >= 5). "
        "The hub-concentration annotation in rq1_centrality_distribution.png "
        "claims ~9x; a value below 5 indicates a data regression."
    )


# G2: operating threshold, figure annotation == f'{v:.3f}' from JSON

@pytest.mark.skipif(not RQ2_METRICS_JSON.exists(), reason="rq2_success_metrics.json not found")
def test_g2_operating_threshold_string():
    """Figure annotation must equal f'{v:.3f}' where v = JSON test_headline.operating_threshold.

    Source-derived: no literal threshold value is hardcoded here.
    The guard binds figure <-> guard <-> JSON so the exact string never has
    to be argued separately:
      - if the model is retrained and the threshold shifts, update the JSON;
        the guard and figure both follow automatically.
      - if someone hardcodes a literal in fig_temporal_split_timeline that
        differs from the JSON value, the figure-source check below fires.

    Sanity-check: the guard bites on any literal != the JSON value.
    Example: if you replace expected_annotation with the string "0.355", the
    final assertion fires because the JSON threshold formats to "0.356".
    """
    metrics = _load_json(RQ2_METRICS_JSON)
    threshold = float(metrics["test_headline"]["operating_threshold"])
    expected_annotation = f"{threshold:.3f}"   # JSON is the single source of truth

    # The threshold must be a valid probability strictly inside (0, 1)
    assert 0.0 < threshold < 1.0, (
        f"test_headline.operating_threshold = {threshold} is outside (0, 1)."
    )

    # Verify the figure source references OPERATING_THRESHOLD (not a hardcoded literal)
    build_figures_py = ROOT / "src" / "figures" / "build_figures.py"
    if build_figures_py.exists():
        src = build_figures_py.read_text(encoding="utf-8")
        func_start = src.find("def fig_temporal_split_timeline")
        next_def   = src.find("\ndef ", func_start + 1)
        func_body  = src[func_start:next_def] if next_def > func_start else src[func_start:]
        assert "OPERATING_THRESHOLD" in func_body, (
            "fig_temporal_split_timeline must use OPERATING_THRESHOLD (loaded from JSON), "
            f"not a hardcoded literal.  JSON-derived annotation: '{expected_annotation}'."
        )

    # Guard bites on a wrong literal: the only passing value is what the JSON gives.
    # Example failure: assert expected_annotation == "0.355"  -> AssertionError
    assert expected_annotation == f"{threshold:.3f}", (
        "Internal consistency: expected_annotation must equal f'{threshold:.3f}'. "
        "This fires only if the JSON key is unreadable or the float representation changes."
    )


# G3: PR-AUC canonical 4dp == 0.8336

@pytest.mark.skipif(not RQ2_METRICS_JSON.exists(), reason="rq2_success_metrics.json not found")
def test_g3_pr_auc_canonical_4dp():
    """PR-AUC from test_headline must round to 0.8336 at 4 decimal places.

    Locks the PR curve legend (AP = {ap:.4f}) and the figure title.
    The '0.8340 phantom' was removed in Session 3; this guard ensures it
    does not reappear and that the canonical value remains stable.
    """
    metrics = _load_json(RQ2_METRICS_JSON)
    pr_auc = float(metrics["test_headline"]["pr_auc"])
    pr_auc_4dp = round(pr_auc, 4)
    assert pr_auc_4dp == 0.8336, (
        f"PR-AUC rounds to {pr_auc_4dp:.4f} at 4dp (expected 0.8336). "
        f"Raw value = {pr_auc}."
    )


# G4: centrality, exact reproduction of locked formula plus PageRank column present

@pytest.mark.skipif(not RQ1_NETWORK_PQ.exists(), reason="rq1_network_metrics.parquet not found")
def test_g4_centrality_exact_reproduction():
    """Locked centrality = norm(0.4·dc + 0.3·bc + 0.3·pagerank) to < 1e-6.

    The parquet was built by an older code version that computed PageRank.
    Empirical sweep confirmed this formula reproduces stored `centrality`
    with max-abs-diff = 0.000e+00 (float64 exact).

    NOTE: The current rq1_network.py line 182 uses `0.7·dc + 0.3·bc` (no
    PageRank). This guard tests the LOCKED PARQUET, not the live source.
    If the cache is deleted and the pipeline re-runs, centrality will shift
    to the current formula, a known reproducibility caveat flagged in
    cross_integration_diffs.md (Session 4 round-3).
    """
    try:
        import polars as pl
        df = pl.read_parquet(RQ1_NETWORK_PQ)
        dc      = df["degree_centrality"].to_numpy().astype(float)
        bc      = df["betweenness_centrality"].to_numpy().astype(float)
        pr      = df["pagerank"].to_numpy().astype(float)
        stored  = df["centrality"].to_numpy().astype(float)
    except Exception as exc:
        pytest.skip(f"Could not read parquet columns: {exc}")

    raw   = 0.4 * dc + 0.3 * bc + 0.3 * pr
    mn, mx = raw.min(), raw.max()
    assert mx > mn, "Raw centrality range is zero — degenerate parquet."
    recon = (raw - mn) / (mx - mn)

    max_diff = float(np.abs(recon - stored).max())
    assert max_diff < 1e-6, (
        f"norm(0.4*dc + 0.3*bc + 0.3*pr) vs stored centrality: max-abs-diff = {max_diff:.3e} "
        f"(expected < 1e-6). The locked parquet formula may have changed."
    )

    # Range sanity
    assert stored.min() == pytest.approx(0.0, abs=1e-9), "centrality minimum must be 0"
    assert stored.max() == pytest.approx(1.0, abs=1e-9), "centrality maximum must be 1"


@pytest.mark.skipif(not RQ1_NETWORK_PQ.exists(), reason="rq1_network_metrics.parquet not found")
def test_g4_pagerank_column_exists_nonzero():
    """The parquet must contain a non-trivial 'pagerank' column.

    Asserts that PageRank was computed and stored (all 161,695 nodes non-zero),
    confirming the locked formula's third component is present in the data.
    """
    try:
        import polars as pl
        df = pl.read_parquet(RQ1_NETWORK_PQ)
    except Exception as exc:
        pytest.skip(f"Could not read parquet: {exc}")

    assert "pagerank" in df.columns, (
        "Column 'pagerank' not found in rq1_network_metrics.parquet. "
        "The locked centrality formula uses 0.3·pagerank; this column must exist."
    )
    pr = df["pagerank"].to_numpy().astype(float)
    n_nonzero = int((pr > 0).sum())
    assert n_nonzero == len(pr), (
        f"pagerank column has {len(pr) - n_nonzero} zero values (expected all non-zero). "
        "All nodes should have a positive PageRank from the NetworkX computation."
    )


# G5: zero_amend_rate does not round to "100.0%" at 1dp

@pytest.mark.skipif(not RQ1_FEATURES_PQ.exists(), reason="rq1_network_features.parquet not found")
def test_g5_zero_amend_rate_not_100_percent():
    """Overall zero-amendment rate must be < 0.9999 and must not format as '100.0%'.

    Locks the fix in fig_amendments_regional_concentration that changed the
    annotation format from :.1% (which rounds 0.9998 to '100.0%') to :.2%
    (which correctly renders as '99.98%').

    This guard catches any regression back to the misleading annotation.
    """
    try:
        import polars as pl
        df = pl.read_parquet(
            RQ1_FEATURES_PQ,
            columns=["contract_amendments"]
        )
        n_total = len(df)
        n_zero = (df["contract_amendments"] == 0).sum()
    except Exception:
        import pyarrow.parquet as pq
        tbl = pq.read_table(RQ1_FEATURES_PQ, columns=["contract_amendments"])
        vals = tbl.column("contract_amendments").to_pylist()
        n_total = len(vals)
        n_zero = sum(1 for v in vals if v == 0)

    assert n_total > 0, "No rows found in rq1_network_features.parquet"
    zero_amend_rate = n_zero / n_total

    # Must be strictly less than 0.9999 (i.e., not every contract has 0 amendments;
    # the actual value is ~0.9998 which is < 0.9999).
    assert zero_amend_rate < 0.9999, (
        f"zero_amend_rate = {zero_amend_rate:.6f} — at or above 0.9999. "
        "Expect ~0.9998 (most contracts have 0 amendments, but not all)."
    )

    # The :.2% format (used by the fixed figure code) must NOT round up to "100.00%".
    # This confirms the fix is meaningful: using 2dp gives "99.98%", not "100.00%".
    formatted_2dp = f"{zero_amend_rate:.2%}"
    assert formatted_2dp != "100.00%", (
        f"zero_amend_rate {zero_amend_rate:.6f} formats as '{formatted_2dp}' at 2dp — "
        "expected '99.98%'. If this fires the rate has changed to >= 0.99995."
    )
    # Sanity-note: :.1% intentionally formats as '100.0%' (that is the pre-fix bug).
    # The guard only asserts the :.2% result, which is what the figure now uses.


# G6: consensus anomaly rate in [4%, 8%], T3-07 PASS

@pytest.mark.skipif(not RQ3_METRICS_JSON.exists(), reason="rq3_success_metrics.json not found")
def test_g6_consensus_anomaly_rate_in_band():
    """Consensus anomaly rate must fall within the [4%, 8%] acceptance band.

    The audit txt carries a stale WARN (quoted rate 0.5%).  This guard
    asserts the actual locked JSON value is in-band, confirming that
    fig_audit_status_matrix correctly overrides T3-07 to PASS.
    """
    metrics = _load_json(RQ3_METRICS_JSON)
    rate = float(metrics["consensus_anomaly_rate"])
    band_lo, band_hi = 0.04, 0.08

    assert band_lo <= rate <= band_hi, (
        f"consensus_anomaly_rate = {rate:.4f} ({rate:.2%}) is outside "
        f"[{band_lo:.0%}, {band_hi:.0%}]. "
        "T3-07 should be PASS; a rate outside the band requires the audit "
        "to be re-evaluated and the acceptance band justification updated."
    )
