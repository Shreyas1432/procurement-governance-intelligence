# Methodology

The implementation follows a three-research-question structure:

- RQ1: buyer-supplier network intelligence using centrality, Louvain communities, and resilience simulation.
- RQ2: governance risk classification using Random Forest, XGBoost or fallback boosting, and Logistic Regression.
- RQ3: price anomaly detection using Gradient Boosting residuals, Local Outlier Factor, and Isolation Forest.

All feature engineering is SQL-first with DuckDB over Parquet. Python modules consume pre-computed Parquet feature tables.
