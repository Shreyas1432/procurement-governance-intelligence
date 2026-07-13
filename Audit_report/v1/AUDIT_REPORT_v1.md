# Project_Main Audit - Version 1 (Baseline)

- **Target:** `procurement-governance-intelligence` (Project_Main)
- **Date:** 2026-06-02
- **Scope:** `src/`, root `phase*.py`, `run_full_pipeline.py`, `tests/`, `requirements.txt`, `pyproject.toml`, `Makefile`, `.env.example` (~4,117 LOC production)
- **Excluded:** `archive/`, `graphify-out/`, `data/` (gitignored / external symlink), `notebooks/`

## Tally: 21 findings - 0 CRITICAL, 6 MAJOR, 14 MINOR, 1 INFO

| ID | Sev | Category | Title |
|----|-----|----------|-------|
| PM-V1-001 | MAJOR | TEST_INTEGRITY | Success tests assert on committed JSON, not recomputed behavior |
| PM-V1-002 | MAJOR | TEST_INTEGRITY | `test_ensemble_beats_solo` vacuous/misnamed (dead `best_solo`) |
| PM-V1-003 | MAJOR | TEST_INTEGRITY | Leakage guards pass vacuously when data absent (silent `return`) |
| PM-V1-004 | MAJOR | EDGE_CASE | RQ1 Louvain fallback hardcodes `modularity=0.35`, passes threshold |
| PM-V1-005 | MAJOR | DEPENDENCY | Missing prod deps: `matplotlib`, `python-pptx` |
| PM-V1-006 | MAJOR | SECURITY | `config.py` `exec()`s reconstructed stdlib source, swallowed |
| PM-V1-007 | MINOR | EDGE_CASE | Binary `scale_pos_weight` on 3-class target |
| PM-V1-008 | MINOR | EDGE_CASE | `_proba` fallback indexes one-hot by raw label value |
| PM-V1-009 | MINOR | EDGE_CASE | Dashboard divides by row count without empty guard |
| PM-V1-010 | MINOR | EDGE_CASE | RQ2 publishes in-sample risk scores for train rows |
| PM-V1-011 | MINOR | EDGE_CASE | RQ3 anomaly residuals computed in-sample |
| PM-V1-012 | MINOR | SECURITY | SQL built via f-string path interpolation (DuckDB) |
| PM-V1-013 | MINOR | SECURITY | Data-derived ID rendered with `unsafe_allow_html` |
| PM-V1-014 | MINOR | TEST_INTEGRITY | EDA/feature tests assert only file existence |
| PM-V1-015 | MINOR | TEST_INTEGRITY | `test_shap_top_features` couples to ranking; no SHAP exists |
| PM-V1-016 | MINOR | TEST_INTEGRITY | conftest split (2016/2019) ≠ production split (2018/2019) |
| PM-V1-017 | MINOR | TEST_INTEGRITY | RQ1 thresholds near-trivial / sample-tuned |
| PM-V1-018 | MINOR | EDGE_CASE | Unvalidated `int(os.environ[...])` at import |
| PM-V1-019 | MINOR | DEPENDENCY | `shap` declared but never imported |
| PM-V1-020 | MINOR | DEPENDENCY | Notebook deps unused; runtime deps unpinned |
| PM-V1-021 | INFO | DEPENDENCY | `configs/` absent; data via external symlink (MISSING_CONTEXT) |

## Headline themes

1. **Test integrity is the dominant risk.** The success-criteria suite reads committed `*_success_metrics.json` rather than recomputing; leakage guards skip silently without data; one comparative test is vacuous. The green suite over-states verification rigor (PM-V1-001/002/003).
2. **Most consequential logic defect:** RQ1's hardcoded `modularity=0.35` in the NetworkX-fallback path (reachable on Python 3.14) trivially satisfies a thesis threshold (PM-V1-004).
3. **Security is otherwise sound:** no hardcoded secrets, no `pickle.load`/`eval`/`yaml.load` (no deserialization RCE), no `shell=True`. The lone structural risk is the `exec()` stdlib monkeypatch (PM-V1-006).
4. **Reproducibility gaps:** two missing prod deps and unpinned versions (PM-V1-005/020).

## Priority fix order (highest leverage first)
1. PM-V1-001/003: make success + leakage tests recompute behavior and SKIP (not PASS) without data.
2. PM-V1-004: remove the hardcoded modularity fallback.
3. PM-V1-005: add `matplotlib`, `python-pptx` to requirements.
4. PM-V1-006: drop the stdlib `exec()` patch; pin interpreter.
5. Remaining MINORs as cleanup.
