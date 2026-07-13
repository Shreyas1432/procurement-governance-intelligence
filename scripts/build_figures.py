"""
build_figures.py: deterministic, idempotent figures pipeline for PGI thesis.

Populates docs/figures/ from committed result artefacts so that every \\pgifig{}
reference in Thesis_Report.tex resolves to a real PNG instead of a placeholder box.

Two operations only:
  1. COPY: sync existing committed PNGs from results/ and data/results/ to docs/figures/
             under the exact filename the \\pgifig{} macro expects.
  2. GENERATE: create missing data-driven figures by reading committed parquets/JSONs.
             No model re-fit, no metric recomputation, no locked number changed.

Figures that have no committed data artefact (conceptual diagrams) are flagged for the
author; this script never fabricates data.

Run:
    .venv/bin/python scripts/build_figures.py
    make figures
"""

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "docs" / "figures"

# Mapping: thesis \\pgifig filename -> committed source PNG (copy-only, never regenerated)
COPY_MAP = {
    "rq1_degree_distribution.png":
        ROOT / "results" / "rq1" / "rq1_degree_distribution.png",
    "rq1_centrality_distribution.png":
        ROOT / "results" / "rq1" / "rq1_centrality_distribution.png",
    "rq1_component_size_distribution.png":
        ROOT / "results" / "rq1" / "rq1_component_size_distribution.png",
    "rq1_conductance_distribution.png":
        ROOT / "results" / "rq1" / "rq1_conductance_distribution.png",
    "rq2_value_only_pdp.png":
        ROOT / "data" / "results" / "rq2_value_only_pdp.png",
}

# Data-driven figures that must be generated from committed artefacts
GENERATED_FIGURES = ["rq3_anomaly_scatter.png"]

# Figures with no committed data artefact: author must supply these manually
AUTHOR_SUPPLIED = ["architecture_high_res.png"]

EXIT_CODE = 0


def ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def sync_existing() -> None:
    """Copy committed result PNGs into docs/figures/ (overwrites, idempotent)."""
    for dest_name, src_path in COPY_MAP.items():
        dest = FIGURES_DIR / dest_name
        if not src_path.exists():
            print(f"  WARNING: source not found: {src_path.relative_to(ROOT)}")
            global EXIT_CODE
            EXIT_CODE = 1
            continue
        shutil.copy2(src_path, dest)
        print(f"  copied  {src_path.relative_to(ROOT)}  ->  docs/figures/{dest_name}")


def generate_rq3_anomaly_scatter() -> None:
    """
    Generate rq3_anomaly_scatter.png.

    Source artefact: data/results/rq3_anomaly_flags.parquet
    Columns used:   predicted_log_price, gb_residual, consensus_flag
    Derivation:     log_price_actual = predicted_log_price + gb_residual
                    (algebraic identity from how gb_residual is defined;
                     no model re-fit required)

    The plot shows the GBR price residual (actual minus predicted log-price) vs the
    predicted log price.  Contracts flagged by the consensus ensemble (>=2 of 3
    detectors) are highlighted in red; normal contracts in grey.
    """
    try:
        import polars as pl
    except ImportError:
        print("  SKIP rq3_anomaly_scatter.png: polars not installed "
              "(run: pip install polars --break-system-packages)")
        return

    flags_path = ROOT / "data" / "results" / "rq3_anomaly_flags.parquet"
    if not flags_path.exists():
        print(f"  SKIP rq3_anomaly_scatter.png: artefact missing: {flags_path.relative_to(ROOT)}")
        print("  Run 'make rq3' first, then re-run 'make figures'.")
        global EXIT_CODE
        EXIT_CODE = 1
        return

    df = pl.read_parquet(flags_path)
    pred = df["predicted_log_price"].to_numpy()
    resid = df["gb_residual"].to_numpy()
    flag = df["consensus_flag"].to_numpy()

    normal_mask = ~flag
    anom_mask = flag
    n_total = len(flag)
    n_anom = int(anom_mask.sum())
    n_normal = int(normal_mask.sum())
    rate_pct = n_anom / n_total * 100

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Normal contracts: small, translucent grey, drawn first (underneath)
    ax.scatter(
        pred[normal_mask], resid[normal_mask],
        s=6, alpha=0.20, color="#888888", linewidths=0,
        label=f"Normal ({n_normal:,})",
        rasterized=True,
    )
    # Anomalous contracts: larger, opaque red, drawn on top
    ax.scatter(
        pred[anom_mask], resid[anom_mask],
        s=18, alpha=0.80, color="#d62728", linewidths=0,
        label=f"Consensus anomaly ({n_anom:,}, {rate_pct:.2f}%)",
        rasterized=True,
    )

    # Zero-residual reference line (model expectation)
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.45)

    ax.set_xlabel("Predicted Log Price (ln EUR)", fontsize=11)
    ax.set_ylabel("Price Residual: Actual - Predicted (ln EUR)", fontsize=11)
    ax.set_title(
        "RQ3: Within-Category Price Anomalies, Consensus Ensemble "
        "(IsolationForest + LOF + EllipticEnvelope, ≥2 of 3 detectors)",
        fontsize=10,
    )
    ax.legend(fontsize=10, markerscale=2.5, framealpha=0.8)

    plt.tight_layout()

    dest = FIGURES_DIR / "rq3_anomaly_scatter.png"
    fig.savefig(dest, dpi=150, metadata={"Date": None})
    plt.close(fig)
    print(
        f"  generated docs/figures/rq3_anomaly_scatter.png "
        f"({n_anom:,} anomalies / {n_total:,} total = {rate_pct:.2f}%)"
    )


def warn_author_supplied() -> None:
    """Report status of figures that have no data artefact and must be author-supplied."""
    for name in AUTHOR_SUPPLIED:
        dest = FIGURES_DIR / name
        if dest.exists():
            print(f"  present (author-supplied): docs/figures/{name}")
        else:
            print(
                f"  AUTHOR-ACTION REQUIRED: docs/figures/{name} has no committed data "
                f"artefact. This is a conceptual architecture diagram. Place it manually "
                f"in docs/figures/ before compiling the thesis."
            )


def main() -> None:
    print("Building thesis figures pipeline")
    ensure_figures_dir()

    print("Step 1: syncing committed result PNGs")
    sync_existing()

    print("Step 2: generating missing data-driven figures")
    generate_rq3_anomaly_scatter()

    print("Step 3: checking author-supplied figures")
    warn_author_supplied()

    resolved = sorted(p.name for p in FIGURES_DIR.iterdir() if p.suffix == ".png")
    print(f"docs/figures/ now contains {len(resolved)} PNG(s): {', '.join(resolved)}")

    all_thesis_refs = list(COPY_MAP.keys()) + GENERATED_FIGURES + AUTHOR_SUPPLIED
    missing = [n for n in all_thesis_refs if not (FIGURES_DIR / n).exists()]
    if missing:
        print(f"Still missing from docs/figures/ (need author action): {missing}")
    else:
        print("All thesis \\pgifig references resolved.")

    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()
