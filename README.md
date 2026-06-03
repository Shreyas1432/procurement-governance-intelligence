# Procurement Governance Intelligence

Procurement Governance Intelligence is an MSc dissertation project for analysing Italian public procurement records through three connected research questions:

- **RQ1: Supplier Network Intelligence** - how buyer-supplier structures reveal concentration, dependence, communities, and resilience.
- **RQ2: Governance Risk Intelligence** - how competition, procedure, concentration, and network signals support risk classification without target leakage.
- **RQ3: Category-Constrained Price Intelligence** - how CPV-aware price models identify abnormal pricing patterns within comparable procurement categories.

The project is intentionally SQL-first for feature construction and Python-first for graph analytics, modeling, validation, reporting, and dashboard delivery.

## Motivation

Public procurement risk is rarely visible through a single contract field. A supplier may look normal on price but occupy a dominant network position; a buyer may appear compliant while repeatedly using low-competition procedures; a price anomaly may be explainable by category mix unless CPV constraints are enforced. This repository builds a reproducible analytics platform that connects those signals into a governance intelligence workflow.

## Repository Structure

```text
data/
  raw/                 raw parquet input location
  processed/           SQL-derived cleaned and descriptive artifacts
  features/            RQ feature tables
  models/              trained model artifacts
  results/             RQ and integration outputs
results/eda/           research EDA reports
src/
  config.py            paths, thresholds, feature policy
  data_loader.py       DuckDB registration and SQL execution
  feature_engineering.py
  rq1_network.py       supplier network intelligence
  rq2_governance.py    governance risk modeling
  rq3_pricing.py       price anomaly modeling
  integration.py       unified cross-RQ risk scoring
  eda/                 research-oriented EDA reports
sql/                   reproducible SQL feature definitions
dashboard/             Streamlit review interface
docs/                  audit, methodology, supervisor, and refactor notes
tests/                 deterministic quality and regression tests
archive/               deprecated experiments and generated clutter
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The raw input is expected at:

```text
data/raw/IT_DIB_2023.parquet
```

You can override it with:

```bash
export PGI_PARQUET_PATH=/path/to/IT_DIB_2023.parquet
```

## Execution

```bash
make eda          # SQL EDA, feature tables, and research EDA reports
make rq1          # supplier network intelligence
make rq2          # governance risk classifiers
make rq3          # price anomaly models
make integrate    # unified cross-RQ scoring
make test         # deterministic regression tests
make dashboard    # Streamlit dashboard
```

For a full local run:

```bash
make all
```

Useful runtime controls:

```bash
export PGI_ROW_LIMIT=0              # full scan when stratification is disabled
export PGI_GRAPH_EDGE_LIMIT=60000   # constrain graph computation locally
```

## Outputs

- `data/processed/*.parquet` - cleaned base and descriptive procurement artifacts.
- `data/features/*.parquet` - RQ feature tables.
- `data/results/*.json` and `data/results/*.parquet` - model metrics, predictions, anomaly flags, network metrics, and unified risk scores.
- `results/eda/*.md` - research EDA reports aligned to RQ1, RQ2, RQ3, and cross-RQ interpretation.
- `results/eda/feature_readiness_matrix.csv` - governance feature readiness matrix.

## Research Contributions

1. A reproducible SQL feature pipeline for procurement governance analytics.
2. A supplier network intelligence layer for centrality, community, concentration, and resilience analysis.
3. A leakage-controlled governance risk modeling workflow.
4. A CPV-constrained price anomaly workflow that avoids comparing structurally different categories.
5. A cross-RQ integration layer connecting network dependence, governance probability, and price anomaly evidence.

## Documentation

- [Repository audit](docs/repository_audit.md)
- [Refactor summary](docs/refactor_summary.md)
- [Supervisor update](docs/supervisor_updates/current_project_state.md)
- [Methodology notes](docs/methodology/)
- [Test strategy](tests/README.md)
