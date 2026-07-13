# Why Feature Engineering

The project uses SQL-first feature engineering so transformations are inspectable, reproducible, and close to the data. This is important for supervisor review and dissertation defensibility.

Features are grouped around the three research questions:

- RQ1 uses buyer-supplier edges, values, categories, and temporal coverage.
- RQ2 uses leakage-controlled governance predictors such as concentration, supplier behavior, centrality, and category risk.
- RQ3 uses CPV-aware price, estimate, procedure, competition, and year variables.

The feature layer is intentionally conservative. It favors transparent procurement indicators over opaque derived variables.
