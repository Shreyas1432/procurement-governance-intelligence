# Test Strategy

The test suite protects reproducibility and research validity rather than chasing broad line coverage.

## Kept Tests

- SQL validation tests execute every SQL-first EDA query against registered DuckDB views.
- SQL-first EDA artifact tests confirm every chart has PNG, HTML, Markdown, metadata JSON, and intermediate parquet.
- Data quality checks confirm the raw parquet is readable, temporally broad, and free of negative prices.
- Feature artifact tests confirm the SQL pipeline produced the required RQ feature tables.
- Leakage tests guard against future-dated supplier features and label-defining predictors entering RQ2 models.

## Archived Tests

`tests/test_eda_exports.py` was archived under `archive/refactor_20260602/brittle_tests/` because it asserted local PNG/HTML presentation exports rather than deterministic source behavior.

RQ threshold and integration behavior tests were archived under `archive/refactor_20260602/archived_tests/` to keep the active suite focused on SQL validation, data quality, and feature generation as required by the refactor brief.

Brittle tests should be avoided when they depend on presentation image size, transient caches, local-only generated files, or exact model scores that can shift across dependency versions.

## Running Tests

```bash
make test
```

Most tests expect generated artifacts from `make eda`, `make rq1`, `make rq2`, `make rq3`, and `make integrate`.
