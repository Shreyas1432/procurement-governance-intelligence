"""
Shared display-layer data hygiene for the PGI dashboard.

Read-only helpers only: no computation, no re-fit, no changes to locked
artifacts. These functions exist because several locked parquet files
(rq2_governance_predictions.parquet, rq3_anomaly_flags.parquet,
unified_risk_scores.parquet) contain exact full-row duplicates baked in
upstream (verified: about 14% of rows in each are byte-identical repeats of
another row, in blocks of a few hundred to several thousand). This is a
known upstream data-quality issue in the locked artifacts, not a dashboard
bug and not something this dashboard is permitted to fix at the source
(compute files are frozen). The functions here only affect what is
displayed, never the underlying files.

Important: the locked success-metrics JSON files (rq2_success_metrics.json,
rq3_success_metrics.json) compute their headline counts (46,586 HIGH-risk
contracts; 1,046 consensus anomalies) by counting the UNDEDUPLICATED rows.
De-duplicating a table for display will make its row count differ from
those locked headline figures. This is expected and is disclosed to the
user via `dedup_disclosure_note()` rather than hidden -- do not "fix" the
discrepancy by silently changing the stat cards elsewhere that cite the
locked headline numbers.
"""
from __future__ import annotations

import polars as pl


def dedupe_for_display(df: pl.DataFrame, key_cols: list[str] | None = None) -> tuple[pl.DataFrame, int, int]:
    """Drop exact full-row duplicates (not just duplicate keys) so a single
    real event is never rendered as if it were several identical, distinct
    rows. Returns (deduped_df, n_before, n_after).

    Uses a full-row `unique()` rather than a subset-key drop so genuinely
    distinct rows sharing the same key (e.g. two different lots under one
    contract+supplier pair with different values) are preserved -- only
    byte-identical repeats are collapsed.
    """
    n_before = df.height
    deduped = df.unique(subset=None, keep="first", maintain_order=True)
    n_after = deduped.height
    return deduped, n_before, n_after


def dedup_disclosure_note(n_before: int, n_after: int, locked_count_label: str, locked_count: int) -> str:
    """User-facing sentence disclosing the de-dup and the resulting gap
    against a locked headline count, instead of silently hiding either
    number. Plain sentence, no em-dash, no emoji."""
    n_removed = n_before - n_after
    if n_removed <= 0:
        return f"Showing {n_after:,} rows. No duplicate rows found."
    return (
        f"Showing {n_after:,} distinct rows after removing {n_removed:,} exact duplicate rows "
        f"present in the locked source file. The locked {locked_count_label} figure "
        f"({locked_count:,}) was computed on the undeduplicated file and reflects the same "
        f"upstream duplication, not a different population. This is a known data quality "
        f"issue in the locked artifact, noted here rather than corrected, since the "
        f"underlying compute files cannot be changed in this pass."
    )


def safe_id(value, placeholder: str = "unknown") -> str:
    """Return a display-safe string for a possibly-missing entity ID. Never
    renders a raw None or the literal string 'None'."""
    if value is None:
        return placeholder
    text = str(value).strip()
    if not text or text.lower() == "none":
        return placeholder
    return text
