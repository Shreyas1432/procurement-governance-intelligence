"""Generate data/public/{features,results} -- pseudonymized-at-rest mirrors
of data/{features,results} for the Streamlit Cloud public build.

Run WITHOUT PGI_PUBLIC_BUILD=1 set, with the real production anon_salt in a
local, gitignored .streamlit/secrets.toml (matching whatever is configured
in the Streamlit Cloud secrets manager). Running this with
PGI_PUBLIC_BUILD=1 set would make dashboard._auth.anonymize_id() a no-op
passthrough and write RAW, unhashed IDs into data/public/ -- the opposite
of what this script exists to do.

Deterministic and re-runnable: same source files + same salt always produce
byte-identical output, since anonymize_id() is a pure function of
(real_id, prefix, salt).

Usage:
    .venv/bin/python scripts/build_public_data.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

if os.environ.get("PGI_PUBLIC_BUILD") == "1":
    raise RuntimeError(
        "PGI_PUBLIC_BUILD=1 is set. Unset it before running this script -- "
        "with it set, anonymize_id() becomes a passthrough and this script "
        "would write raw, unhashed IDs into data/public/."
    )

from dashboard._auth import anonymize_id  # noqa: E402  (path setup must run first)

DATA_FEATURES = ROOT / "data" / "features"
DATA_RESULTS = ROOT / "data" / "results"
PUBLIC_FEATURES = ROOT / "data" / "public" / "features"
PUBLIC_RESULTS = ROOT / "data" / "public" / "results"


def _node_id_col(id_col: str, type_col: str) -> pl.Expr:
    """Pseudonymize a 'B_<raw>' / 'S_<raw>' node-key column, keeping the
    prefix + underscore structure the app parses via node_id[2:] /
    node_id.startswith("B_"). The letter comes from the row's own type
    column (not string-parsed from the key) so it can never disagree.
    """
    return pl.struct([id_col, type_col]).map_elements(
        lambda row: (
            ("B_" if row[type_col] == "buyer" else "S_")
            + anonymize_id(row[id_col][2:], prefix=("B" if row[type_col] == "buyer" else "S"))
        ),
        return_dtype=pl.Utf8,
    ).alias(id_col)


def _entity_id_col(id_col: str, type_col: str) -> pl.Expr:
    """Pseudonymize a bare buyer/supplier id column, letter from type_col."""
    return pl.struct([id_col, type_col]).map_elements(
        lambda row: anonymize_id(row[id_col], prefix=("B" if row[type_col] == "buyer" else "S")),
        return_dtype=pl.Utf8,
    ).alias(id_col)


def _fixed_prefix_col(col: str, prefix: str) -> pl.Expr:
    return pl.col(col).map_elements(lambda v: anonymize_id(v, prefix=prefix), return_dtype=pl.Utf8).alias(col)


# Each parquet spec: (relative path under features/ or results/, transforms).
# A transform is either a fixed prefix string ("B"/"S"/"C") applied via
# anonymize_id, or "node_id" / "entity_id" for the type-column-driven forms.
FEATURES_PARQUET = {
    "rq1_network_features.parquet": {
        "buyer_id": "B", "supplier_id": "S", "contract_id": "C",
        "_drop": ["buyer_name", "supplier_name"],
    },
    "rq2_governance_features.parquet": {
        "buyer_id": "B", "supplier_id": "S", "contract_id": "C",
    },
}

RESULTS_PARQUET = {
    "rq1_network_metrics.parquet": {"node_id": "node_id", "entity_id": "entity_id"},
    "rq1_community_assignments.parquet": {"node_id": "node_id", "entity_id": "entity_id"},
    "rq1_conductance_values.parquet": {},
    "rq1_resilience_scores.parquet": {"buyer_id": "B", "top_supplier_id": "S"},
    "rq1_validation_metrics.parquet": {},
    "rq2_governance_predictions.parquet": {"contract_id": "C", "buyer_id": "B", "supplier_id": "S"},
    "rq3_anomaly_flags.parquet": {"contract_id": "C", "buyer_id": "B", "supplier_id": "S"},
    "unified_risk_scores.parquet": {"contract_id": "C", "buyer_id": "B", "supplier_id": "S"},
    "resilience_exposure_table.parquet": {"node_id": "node_id"},
}

# Copied through byte-identical: verified in Step 0 to embed no entity IDs.
RESULTS_PASSTHROUGH = [
    "rq1_success_metrics.json",
    "rq2_success_metrics.json",
    "rq2_feature_disposition.json",
    "rq3_success_metrics.json",
    "model_performance_ci.json",
    "integration_metrics.json",
    "cross_effect_sizes.json",
    "pipeline_metadata.json",
    "resilience_metrics.json",
    "rq2_value_only_pdp.png",
]


def _pseudonymize(df: pl.DataFrame, spec: dict) -> pl.DataFrame:
    drop_cols = spec.get("_drop", [])
    exprs = []
    for col, kind in spec.items():
        if col == "_drop":
            continue
        if kind == "node_id":
            exprs.append(_node_id_col(col, "node_type"))
        elif kind == "entity_id":
            exprs.append(_entity_id_col(col, "node_type"))
        else:
            exprs.append(_fixed_prefix_col(col, kind))
    out = df.with_columns(exprs) if exprs else df
    if drop_cols:
        out = out.drop([c for c in drop_cols if c in out.columns])
    return out


def build() -> None:
    PUBLIC_FEATURES.mkdir(parents=True, exist_ok=True)
    PUBLIC_RESULTS.mkdir(parents=True, exist_ok=True)

    for name, spec in FEATURES_PARQUET.items():
        src = DATA_FEATURES / name
        df = pl.read_parquet(src)
        _pseudonymize(df, spec).write_parquet(PUBLIC_FEATURES / name)
        print(f"features/{name}: {df.height} rows, columns pseudonymized: {[c for c in spec if c != '_drop']}, dropped: {spec.get('_drop', [])}")

    for name, spec in RESULTS_PARQUET.items():
        src = DATA_RESULTS / name
        df = pl.read_parquet(src)
        _pseudonymize(df, spec).write_parquet(PUBLIC_RESULTS / name)
        print(f"results/{name}: {df.height} rows, columns pseudonymized: {list(spec.keys())}")

    for name in RESULTS_PASSTHROUGH:
        src = DATA_RESULTS / name
        shutil.copyfile(src, PUBLIC_RESULTS / name)
        print(f"results/{name}: copied unchanged (verified no embedded IDs)")


if __name__ == "__main__":
    build()
