# Architecture

Raw procurement Parquet is queried through DuckDB SQL, transformed into processed and feature Parquet tables, consumed by RQ-specific modules, integrated into unified risk scores, and served through a Streamlit dashboard.

Stages:

1. Raw data: `data/raw/IT_DIB_2023.parquet`
2. SQL EDA and features: `sql/*.sql` through `src.data_loader.DataLoader`
3. Processed data: `data/processed/*.parquet`
4. Feature tables: `data/features/*.parquet`
5. Models and results: `data/models`, `data/results`
6. Dashboard: `dashboard/app.py`
