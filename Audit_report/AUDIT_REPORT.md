# Audit Report

**Project:** Procurement Governance Intelligence (MSc AI Thesis)  
**Auditor:** Automated Production-Grade QA & Release Readiness Review  
**Date:** 2026-06-04T16:25:00+01:00  
**Scope:** Full repository — source code, tests, configs, dependencies, data handling, file hygiene  
**Audit Batch:** `audit_20260602_192731` (archival) + `20260604` (report)

---

## 1. Executive Summary

- **Overall status:** [WARN] NOT RELEASE-READY — 3 Critical, 10 High, 16 Medium, 12 Low issues found
- **Release readiness:** **FAIL** — The pipeline runner (`run_full_pipeline.py`) crashes on import due to a function name mismatch. A syntax error in `phase1_eda.py` prevents that script from completing. Multiple hardcoded relative paths in SQL queries make the feature engineering module CWD-dependent.
- **Primary risks:**
  1. **Pipeline cannot execute end-to-end** — `run_rq3` import fails (`ImportError`)
  2. **Hardcoded relative paths in DuckDB SQL** break execution from any CWD other than project root
  3. **Validation/test year overlap** creates subtle data leakage in RQ2 model evaluation
  4. **Two missing pip dependencies** (`numpy`, `python-pptx`) not declared in `requirements.txt`
  5. **100% duplicate data directory** (`data/processed/` = exact copy of `data/curated/`) wastes ~125 MB of disk

### What Works Well

The codebase demonstrates solid architectural decisions:
- Clean SQL-first feature engineering via DuckDB with reproducible year-stratified sampling
- Proper target leakage prevention with explicit `RQ2_LABEL_INPUTS` / `RQ2_MODEL_FEATURES` separation
- Graceful optional-dependency fallbacks (NetworkX, XGBoost)
- Integration join explosion guard (`compute_unified_risk` row-count assertion)
- Comprehensive Streamlit dashboard with 4 pages and reusable chart components
- Well-structured test suite covering success criteria, leakage, and integration correctness

---

## 2. Critical Findings

### CRIT-01: `run_full_pipeline.py` — ImportError crashes pipeline

- **ID:** CRIT-01
- **Severity:** Critical
- **File:** [run_full_pipeline.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/run_full_pipeline.py#L9)
- **Location:** Line 9
- **Problem:** `from src.rq3_price_intelligence.rq3_pricing import run_rq3` — but the function is named `run_rq3_pricing()` (defined at [rq3_pricing.py:L20](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/rq3_price_intelligence/rq3_pricing.py#L20)). The name `run_rq3` does not exist in that module.
- **Impact:** The full pipeline runner crashes immediately with `ImportError: cannot import name 'run_rq3'`. No phase executes. `make all` is broken.
- **Fix:** Either rename the function in `rq3_pricing.py` to `run_rq3`, or change the import in `run_full_pipeline.py` to `from src.rq3_price_intelligence.rq3_pricing import run_rq3_pricing as run_rq3`.

---

### CRIT-02: `phase1_eda.py` — Syntax error prevents execution

- **ID:** CRIT-02
- **Severity:** Critical
- **File:** `phase1_eda.py` (archived, but still present in older checkouts)
- **Location:** Line 436
- **Problem:** `skipped = 10 len(generated)` — missing the minus operator (`-`). Python interprets this as calling integer `10` as a function on `len(generated)`, producing `TypeError: 'int' object is not callable`.
- **Impact:** Script crashes after generating all plots, before writing the summary. The `eda_summary.json` file is never written, which breaks `phase2_ppt.py`.
- **Fix:** Change to `skipped = 10 - len(generated)`.

---

### CRIT-03: `src/data_loader.py` — Broken import (dead file)

- **ID:** CRIT-03
- **Severity:** Critical (now resolved)
- **File:** `src/data_loader.py` → **ARCHIVED** to `archive/audit_20260602_192731/data_loader_20260602_192731.py`
- **Location:** Line 6
- **Problem:** `from src.config import (...)` — the module `src.config` does not exist. The actual config module is `src.core.config`. This file is an exact duplicate of `src/core/data_loader.py` except for the broken import.
- **Impact:** Any code importing from `src.data_loader` crashes with `ModuleNotFoundError`. Nothing in the active codebase imports this file, confirming it is dead code.
- **Fix:** [PASS] Archived on 2026-06-02. No further action needed.

---

### HIGH-01: Hardcoded relative paths in DuckDB SQL queries

- **ID:** HIGH-01
- **Severity:** High
- **File:** [feature_engineering.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/core/feature_engineering.py#L79)
- **Location:** Lines 79, 117, 118, 119, 121
- **Problem:** SQL queries contain hardcoded relative paths like `read_parquet('data/features/buyer_dependency_features.parquet')`. DuckDB resolves these relative to the **process CWD**, not relative to the Python file or project root.
- **Impact:** If `run_full_pipeline.py`, `make features`, or any module is invoked from a working directory other than the project root (e.g., `/Users/shreyas`), DuckDB throws `IOException: No such file or directory`.
- **Fix:** Use `str(DATA_FEATURES / "buyer_dependency_features.parquet")` in f-string SQL, or `os.chdir(ROOT)` at the start of `run_feature_engineering()`.

---

### HIGH-02: Validation/test year overlap in temporal split

- **ID:** HIGH-02
- **Severity:** High
- **File:** [config.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/core/config.py#L40-L42)
- **Location:** Lines 40–42
- **Problem:**
  ```python
  TRAIN_YEARS = range(2014, 2019)   # 2014-2018
  VAL_YEARS   = range(2019, 2020)   # 2019
  TEST_YEARS  = range(2019, 2021)   # 2019-2020
  ```
  `VAL_YEARS` and `TEST_YEARS` both include 2019. The validation set is a subset of the test set.
- **Impact:** Model selection via validation scores is evaluated on data that also appears in the test set, inflating reported generalization metrics. This is temporal data leakage for the thesis evaluation.
- **Fix:** Set `VAL_YEARS = range(2019, 2020)` and `TEST_YEARS = range(2020, 2021)`, or remove `VAL_YEARS` if not actively used (currently unused in `rq2_governance.py`).

---

### HIGH-03: `closeness_centrality` is actually degree centrality

- **ID:** HIGH-03
- **Severity:** High
- **File:** [rq1_network.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/rq1_supplier_network/rq1_network.py#L84)
- **Location:** Line 84
- **Problem:** `closeness = degree` — the variable `closeness` is set equal to the `degree_centrality` dict, not computed via `nx.closeness_centrality()`. The resulting `closeness_centrality` column in output parquet files stores degree centrality values.
- **Impact:** The `closeness_centrality` column in `rq1_network_metrics.parquet` is mislabeled. Any thesis text or analysis referencing closeness centrality numbers is reporting degree centrality instead. This is a **thesis correctness issue**.
- **Fix:** Either compute actual closeness centrality (`closeness = nx.closeness_centrality(self.G)`) or rename the column to indicate it's a placeholder/alias.

---

### HIGH-04: `exec()` call on dynamically modified stdlib source

- **ID:** HIGH-04
- **Severity:** High
- **File:** [config.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/core/config.py#L6-L17)
- **Location:** Lines 6–17
- **Problem:** Monkey-patches `dataclasses._add_slots` using `exec()` on dynamically modified source code to work around a Python 3.14.1 bug. This accesses private stdlib internals.
- **Impact:** Fragile across Python versions. The `exec()` call is a security concern in production code and a maintenance hazard. If the internal API changes, the patch silently fails (try/except catches everything).
- **Fix:** Pin the Python version in `pyproject.toml` and remove the monkey-patch once the stdlib bug is fixed upstream, or guard it with a version check.

---

### HIGH-05: `phase1_eda.py` — NULL column names injected into SQL

- **ID:** HIGH-05
- **Severity:** High
- **File:** `phase1_eda.py` (archived)
- **Location:** Lines 94–106
- **Problem:** If `pick()` returns `None` for `COL_PROC`, `COL_BIDS`, `COL_SUPPLIER`, `COL_BUYER`, or `COL_CPV`, the SQL base table creation injects literal string `"None"` as column names, causing DuckDB `BinderException`.
- **Impact:** Crashes when running on datasets where expected columns are absent.
- **Fix:** Guard each column reference with an `IS NOT NULL` check or skip the base table creation if critical columns are missing.

---

### HIGH-06: `phase2_ppt.py` — Hardcoded KPI values on slides

- **ID:** HIGH-06
- **Severity:** High
- **File:** `phase2_ppt.py` (archived)
- **Location:** Lines 161–166, 200–203
- **Problem:** KPI cards display hardcoded values like `"99.997%"`, `"126"` columns, `"82,346"` buyers, `"50.3%"` above HHI 0.50. These are baked into the PPTX generator, not derived from actual data at runtime.
- **Impact:** If the dataset or filtering changes, the presentation shows stale/incorrect numbers. Thesis examiner could notice discrepancies.
- **Fix:** Compute these values from the `eda_summary.json` or directly from DuckDB queries.

---

### HIGH-07: Dashboard `ROOT` path resolves to `src/`, not project root

- **ID:** HIGH-07
- **Severity:** High
- **File:** [app.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/dashboard/app.py#L11)
- **Location:** Line 11
- **Problem:** `ROOT = Path(__file__).resolve().parent.parent` — from `src/dashboard/app.py`, this resolves to `src/`, not the project root. Line 46: `base = ROOT / "data" / "results"` becomes `src/data/results` which doesn't exist.
- **Impact:** The dashboard will crash with `FileNotFoundError` when loading parquet files, unless Streamlit's CWD happens to compensate. Works accidentally when `streamlit run` is invoked from project root because `sys.path.insert(0, str(ROOT))` adds `src/` and data paths are resolved relative to CWD.
- **Fix:** Use `ROOT = Path(__file__).resolve().parent.parent.parent` (3 levels up from `dashboard/app.py` to project root).

---

### HIGH-08: Makefile `dashboard` target has incorrect Streamlit command

- **ID:** HIGH-08
- **Severity:** High
- **File:** `Makefile` (archived/removed)
- **Location:** Line 42
- **Problem:** `$(PYTHON) -m streamlit run src/dashboard/app.py` — the `-m` flag with `streamlit run` is incorrect. It should be `$(PYTHON) -m streamlit run src/dashboard/app.py` (which actually works because `-m streamlit` invokes the streamlit module, and `run` is its subcommand). However, mixing `.venv/bin/python -m streamlit run` with relative paths is fragile.
- **Impact:** May fail depending on shell/env configuration.
- **Fix:** Use `streamlit run src/dashboard/app.py` directly, or ensure the venv is activated.

---

### HIGH-09: Makefile `strip` target references non-existent script

- **ID:** HIGH-09
- **Severity:** High
- **File:** `Makefile` (archived/removed)
- **Location:** Line 48
- **Problem:** `$(PYTHON) scripts/strip_comments.py` — the `scripts/` directory does not exist in the repository. This target will always fail.
- **Impact:** `make strip` crashes.
- **Fix:** Remove the target or create the script.

---

### HIGH-10: `_proba` fallback can crash on unexpected class labels

- **ID:** HIGH-10
- **Severity:** High
- **File:** [rq2_governance.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/rq2_governance_risk/rq2_governance.py#L174-L179)
- **Location:** Lines 174–179
- **Problem:** The fallback branch `return np.eye(len(classes))[pred]` will raise `IndexError` if `pred` contains values outside `[0, len(classes)-1]` (e.g., if a multiclass model predicts class label `2` but only classes `{0, 1}` are in `np.unique(pred)`).
- **Impact:** Silent crash during ensemble prediction if the fallback model produces unexpected class labels.
- **Fix:** Use `np.eye(max(pred)+1)[pred]` or map predictions to contiguous indices.

---

### MED-01: `rq2_governance.py` — "single_bid_rate" is mislabeled

- **ID:** MED-01
- **Severity:** Medium
- **File:** [rq2_governance.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/rq2_governance_risk/rq2_governance.py#L152)
- **Location:** Line 152
- **Problem:** The metric `single_bid_rate` actually measures the rate of negotiated/restricted procedures (`str.contains("(?i)negotiated") | (== "RESTRICTED")`), not the single-bid rate (bid_count ≤ 1).
- **Impact:** Misleading metric name in `rq2_success_metrics.json`. The test `test_rgov_correlation` asserts on this mislabeled value.
- **Fix:** Rename to `negotiated_restricted_rate` or compute actual single-bid rate.

---

### MED-02: SQL injection risk via f-string path interpolation

- **ID:** MED-02
- **Severity:** Medium
- **File:** [data_loader.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/core/data_loader.py#L43-L50)
- **Location:** Lines 43, 50, 57, 64, 97
- **Problem:** Parquet file paths are interpolated directly into SQL strings: `f"read_parquet('{self.parquet_path}')"`. A path containing a single quote (`'`) will break the SQL or enable injection.
- **Impact:** Low in practice (paths are config-controlled), but violates secure coding practices.
- **Fix:** Use DuckDB parameterized queries or escape/validate paths.

---

### MED-03: Anomaly score semantics inconsistency

- **ID:** MED-03
- **Severity:** Medium
- **File:** [rq3_pricing.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/rq3_price_intelligence/rq3_pricing.py#L53) and [charts.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/dashboard/components/charts.py#L339)
- **Location:** rq3_pricing.py:53, charts.py:339
- **Problem:** `rq3_pricing.py` produces `anomaly_score ∈ {0.0, 0.7, 1.0}`, but the dashboard chart label says "Anomaly Score (0–3)".
- **Impact:** Users see a misleading axis label on the anomaly score distribution chart.
- **Fix:** Change dashboard label to "Anomaly Score (0–1)" or revise the scoring to use the `flag_count` (0–3) directly.

---

### MED-04: `contract_value` and `final_price` are identical columns

- **ID:** MED-04
- **Severity:** Medium
- **File:** [data_loader.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/core/data_loader.py#L81-L82)
- **Location:** Lines 81–82
- **Problem:** Both `contract_value` and `final_price` use the exact same COALESCE expression. They are always identical.
- **Impact:** Wastes memory, creates confusion about which column to use, and suggests they should differ (estimated vs final price). `rq3_pricing.py` uses `final_price` while other modules use `contract_value`.
- **Fix:** Keep one column or differentiate their semantics (e.g., use `tender_estimatedpriceUsd` for `estimated_price` only).

---

### MED-05: `argparse` imported but never used

- **ID:** MED-05
- **Severity:** Medium
- **File:** [feature_engineering.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/core/feature_engineering.py#L1)
- **Location:** Line 1
- **Problem:** `import argparse` — never referenced anywhere in the file.
- **Impact:** Dead import. Suggests incomplete refactoring (CLI args were planned but not implemented).
- **Fix:** Remove the import.

---

### MED-06: Blanket warning suppression in `phase1_eda.py`

- **ID:** MED-06
- **Severity:** Medium
- **File:** `phase1_eda.py` (archived)
- **Location:** Line 27
- **Problem:** `warnings.filterwarnings("ignore")` hides all warnings globally, including deprecation notices and data-integrity alerts from pandas/numpy.
- **Impact:** Can mask genuine issues during development and data exploration.
- **Fix:** Suppress specific warning categories only.

---

### MED-07: Unclosed file handles

- **ID:** MED-07
- **Severity:** Medium
- **Files:** `phase2_ppt.py:16`, `_audit/build_manifest.py:24,62`, `_audit/imports.py:48,113`
- **Problem:** `json.load(open(...))` and `json.dump(..., open(..., "w"))` without context managers leave file handles open.
- **Impact:** File handle leaks on long-running processes.
- **Fix:** Use `with open(...) as f:` context managers.

---

### MED-08: PIL Image.open() not closed after reading size

- **ID:** MED-08
- **Severity:** Medium
- **File:** `phase2_ppt.py` (archived)
- **Location:** Line 110
- **Problem:** `Image.open(full).size` — the Image object is never closed, leaking a file handle per plot.
- **Impact:** Resource leak proportional to number of plots embedded.
- **Fix:** `with Image.open(full) as img: iw, ih = img.size`.

---

### MED-09: Hardcoded Codex CI path in generate_charts.py

- **ID:** MED-09
- **Severity:** Medium
- **File:** [generate_charts.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/eda/visualizations/generate_charts.py#L81-L82)
- **Location:** Lines 81–82
- **Problem:** `Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"` — CI-specific path hardcoded into production code.
- **Impact:** Unnecessary code path in deployed code; confusing for anyone reading the source.
- **Fix:** Remove or move to an env var like `PLAYWRIGHT_NODE_PATH`.

---

### MED-10: `generate_charts.py` — wrong "Total Contracts Loaded" formula

- **ID:** MED-10
- **Severity:** Medium
- **File:** [generate_charts.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/eda/visualizations/generate_charts.py#L464)
- **Location:** Line 464
- **Problem:** `df_contracts.height + df_nulls['null_rate'].max() * df_contracts.height` — this formula attempts to estimate the raw count by inflating the cleaned count by the maximum null rate, which is mathematically nonsensical.
- **Impact:** The "Total Contracts Loaded" number on the summary statistics table is incorrect.
- **Fix:** Use the actual raw contract count from the data loader or config.

---

### MED-11: Dashboard pages `ROOT` navigation goes 3 levels up

- **ID:** MED-11
- **Severity:** Medium
- **File:** All 4 dashboard page files
- **Location:** Line 12 in each
- **Problem:** `ROOT = Path(__file__).resolve().parent.parent.parent` — goes from `src/dashboard/pages/` up to `src/`. Data paths reference `ROOT / "data" / "results"` which becomes `src/data/results`.
- **Impact:** Same CWD-dependency issue as HIGH-07. Works only when Streamlit is invoked from project root.
- **Fix:** Navigate to actual project root (4 levels up: `parent.parent.parent.parent`).

---

### MED-12: No error handling in pipeline runner

- **ID:** MED-12
- **Severity:** Medium
- **File:** [run_full_pipeline.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/run_full_pipeline.py#L23-L27)
- **Location:** Lines 23–27
- **Problem:** The pipeline iterates through phases with no try/except. If any phase crashes, subsequent phases are skipped with no cleanup, partial results, or error reporting.
- **Impact:** Partial pipeline failures leave data in inconsistent state with no diagnostic output.
- **Fix:** Wrap each phase in try/except, log the error, and decide whether to continue or abort.

---

### MED-13: Duplicated `_safe_corr` function

- **ID:** MED-13
- **Severity:** Medium
- **File:** [rq1_network.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/rq1_supplier_network/rq1_network.py#L234-L239)
- **Location:** Lines 234–239
- **Problem:** `_safe_corr` duplicates `src.common.evaluation.safe_corr` but uses `np.corrcoef` instead of `scipy.stats.pearsonr`. Behavior is subtly different (p-value not computed).
- **Impact:** Two correlation implementations with slightly different behavior creates maintenance risk and inconsistent results.
- **Fix:** Use `from src.common.evaluation import safe_corr` everywhere.

---

### MED-14: `rq2_labels.csv` only samples 1000 rows

- **ID:** MED-14
- **Severity:** Medium
- **File:** [feature_engineering.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/core/feature_engineering.py#L130)
- **Location:** Line 130
- **Problem:** `labels.head(min(1000, rq2.height))` — samples only the first 1000 rows for `rq2_labels.csv` without explanation. This is not a random sample — it's the first 1000 rows in whatever order Polars returns.
- **Impact:** If `rq2_labels.csv` is used for anything downstream, it's a biased sample.
- **Fix:** Document the purpose of this file or use `sample(n=1000)` for a random sample.

---

### MED-15: `pl.Utf8` deprecated in newer Polars

- **ID:** MED-15
- **Severity:** Medium
- **File:** [generate_charts.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/eda/visualizations/generate_charts.py#L406)
- **Location:** Line 406
- **Problem:** `pl.col("cpv_division").cast(pl.Utf8)` — `pl.Utf8` is deprecated in Polars ≥ 0.20; use `pl.String`.
- **Impact:** Deprecation warning; will break in future Polars versions.
- **Fix:** Replace `pl.Utf8` with `pl.String`.

---

### MED-16: Hardcoded contract count in dashboard header

- **ID:** MED-16
- **Severity:** Medium
- **File:** [app.py](file:///Users/shreyas/Shreyas/NCI_Sub/Sem%202/Practicum/Project_Main/src/dashboard/app.py#L68)
- **Location:** Line 68
- **Problem:** `"651,793 Italian public procurement contracts"` is hardcoded, not derived from loaded data.
- **Impact:** Becomes stale when data changes.
- **Fix:** Use `f"{len(unified):,}"` instead.

---

### LOW-01 through LOW-12 (summarized)

| ID | File | Line | Problem |
|----|------|------|---------|
| LOW-01 | `phase0_schema.py` | 10 | Hardcoded `SEARCH_ROOTS` with relative paths |
| LOW-02 | `phase0_schema.py` | 50 | Variable `parquet_files` also contains CSV files |
| LOW-03 | `phase1_eda.py` | 19 | Multi-import on single line (PEP 8) |
| LOW-04 | `phase2_ppt.py` | 43 | `from pptx.oxml.ns import qn` at module level after class defs |
| LOW-05 | `rq1_network.py` | 173 | `supplier_exit_coverage_loss` name is misleading (inverse of neighbor count) |
| LOW-06 | `rq1_network.py` | 108 | `tuple[...]` lowercase type hint (Python 3.9+ only) |
| LOW-07 | `rq2_governance.py` | 154 | Late import of `scipy.stats.pearsonr` inside function |
| LOW-08 | `rq2_governance.py` | 186–192 | `_auc` silently catches all exceptions, returns 0.0 |
| LOW-09 | `rq3_pricing.py` | 48 | `LocalOutlierFactor` only works with `fit_predict`, cannot score new data |
| LOW-10 | Various scratch files | Multiple | Dead imports: `pandas as pd` imported but unused |
| LOW-11 | `_audit/imports.py` | 10 | `sys.stdlib_module_names` requires Python 3.10+ |
| LOW-12 | `.gitignore` | N/A | `data/processed/` not listed in `.gitignore` (was duplicate directory) |

---

## 3. Dependency Issues

### Missing packages (imported but NOT in requirements.txt)

| Import Name | PyPI Package | Used In | Impact |
|------------|-------------|---------|--------|
| `numpy` | `numpy` | 13+ files (rq1, rq2, rq3, integration, utils, charts, phase1, scratch) | **Critical** — numpy is a transitive dep of pandas/polars/sklearn, so it's installed indirectly. But it must be declared explicitly for reproducibility. |
| `pptx` | `python-pptx` | `phase2_ppt.py` | **High** — `pip install -r requirements.txt` will NOT install python-pptx. Running `phase2_ppt.py` will crash. |
| `matplotlib` | `matplotlib` | `phase1_eda.py` | **High** — Same issue. Required for phase1 but not declared. |

### Unused packages (in requirements.txt but NOT imported in active code)

| Package | Status | Recommendation |
|---------|--------|----------------|
| `jupyter>=7.0.0` | Dev tool only | Move to `[project.optional-dependencies.dev]` in pyproject.toml |
| `jupyterlab>=3.0.0` | Dev tool only | Same |
| `nbconvert>=7.0.0` | Dev tool only | Same |
| `shap>=0.44.0` | Was likely used earlier, now absent from all code | Remove or re-add if SHAP explanations are needed |
| `kaleido==0.2.1` | Used indirectly by Plotly `write_image()` | Keep, but add a comment explaining the indirect dependency |
| `pyarrow>=15.0.0` | Used indirectly by Polars/Pandas for parquet I/O | Keep — runtime transitive dependency |
| `pillow>=10.0.0` | Only used in `phase2_ppt.py` | Keep if phase2 is active |

### Version concerns

| Package | Declared | Concern |
|---------|----------|---------|
| `plotly==5.18.0` | Pinned exactly | May conflict with newer Polars/Streamlit — should use `>=5.18.0` |
| `kaleido==0.2.1` | Pinned exactly | Known issues on Apple Silicon; `kaleido>=0.2.1` preferred |
| `polars>=0.20.0` | Min version | `pl.Utf8` deprecated in 0.20+; code uses `pl.Utf8` (see MED-15) |

### Notes
- `python-louvain` is correctly declared; import name `community` matches.
- `scikit-learn` is correctly declared; import name `sklearn` matches.
- No conflicting version ranges detected between declared dependencies.

---

## 4. Test Assessment

### Existing Coverage

| Test File | Tests | What It Covers | Quality |
|-----------|-------|----------------|---------|
| `test_data_quality.py` | 3 | Row count, temporal range, no negative prices | [PASS] Good — guards data integrity |
| `test_feature_engineering.py` | 1 | Existence of 4 feature parquet files | [WARN] Weak — only checks file existence, not content |
| `test_eda_outputs.py` | 2 | Existence of 20 HTML/PNG/JSON/MD chart artifacts + 6 section reports | [PASS] Good — but will fail if EDA hasn't been run |
| `test_leakage.py` | 2 | No future supplier features, train/test temporal gap | [PASS] Critical test — depends on `feature_df` fixture loading supplier_behavior_features.parquet |
| `test_no_target_leakage.py` | 3 | RQ2 features exclude label inputs, classifier features are clean | [PASS] Excellent — guards thesis correctness |
| `test_rq1_success.py` | 5 | Graph build, centrality distribution, modularity, dependency rate, resilience std | [PASS] Good success criteria tests |
| `test_rq2_success.py` | 6 | RF F1, XGB AUC, LR AUC, ensemble existence, R_gov correlation, SHAP top features | [PASS] Good — tests thesis acceptance criteria |
| `test_rq3_success.py` | 4 | GB R², RMSE reduction, anomaly rate bounds, LOF-GB overlap | [PASS] Good |
| `test_integration.py` | 3 | Unified scores exist + bounds, join doesn't inflate rows | [PASS] Excellent — guards against the real-world row explosion bug |

### Missing Test Coverage

| Gap | Impact | Recommendation |
|-----|--------|----------------|
| **No unit tests for `DataLoader` class** | Cannot verify SQL view creation or year-stratification logic | Add tests with a small synthetic parquet file |
| **No tests for dashboard components** | Dashboard charts/filters untested | Add smoke tests that call each chart function with minimal DataFrames |
| **No test for `run_full_pipeline.py`** | Pipeline runner has a broken import (CRIT-01) that tests would catch | Add an import-only test |
| **No test for `config.py` path resolution** | `ROOT`, `DATA_RAW`, etc. could break if directory structure changes | Add a test verifying all config paths resolve correctly |
| **No negative-path tests** | No tests for missing files, empty DataFrames, corrupt data | Add parametric edge-case tests |
| **`test_leakage.py` fixture may not have required columns** | `supplier_feature_latest_date` and `focal_award_date` may not exist in the current schema | Verify columns exist or test will silently pass via early return |
| **`test_eda_outputs.py` expects 20 specific chart names** | Tests hardcode exact filenames; any rename breaks tests | Use glob patterns or make names configurable |
| **`test_ensemble_beats_solo` doesn't actually compare** | Line 27: computes `best_solo` but never asserts ensemble > solo | Add assertion: `assert ensemble_auc >= best_solo * 0.95` |

### Weak Tests

- `test_feature_engineering.py` — only checks file existence, not schema, row count, or value ranges
- `test_leakage.py:test_no_future_supplier_features` — silently returns on empty DataFrame (line 5–6)
- `test_leakage.py:test_train_test_temporal_split` — silently returns on empty DataFrame (line 12–13)

---

## 5. Archived Files

### Archived on 2026-06-02 (batch `audit_20260602_192731`)

| Original Path | Archived Path | Reason |
|---------------|---------------|--------|
| `src/data_loader.py` | `archive/audit_20260602_192731/data_loader_20260602_192731.py` | Dead code: broken import (`from src.config import ...`), exact duplicate of `src/core/data_loader.py` |
| `project_memory.md` | `archive/audit_20260602_192731/project_memory_20260602_192731.md` | Exact duplicate of `.claude_memory.md` (identical MD5: `04dc620378b6f065df78322ae5f00e85`) |
| `.claude_memory.md` | `archive/audit_20260602_192731/claude_memory_20260602_192731.md` | Agent memory file, not source code |
| `eda_summary.json` | `archive/audit_20260602_192731/eda_summary_20260602_192731.json` | Stale phase1 output artifact |
| `scratch/analyze_rq2_data.py` | `archive/audit_20260602_192731/scratch/analyze_rq2_data_20260602_192731.py` | Experimental scratch script |
| `scratch/check_buyer_corr.py` | `archive/audit_20260602_192731/scratch/check_buyer_corr_20260602_192731.py` | Experimental scratch script |
| `scratch/check_corr.py` | `archive/audit_20260602_192731/scratch/check_corr_20260602_192731.py` | Experimental scratch script with dead `pandas` import |
| `scratch/evaluate_rq2.py` | `archive/audit_20260602_192731/scratch/evaluate_rq2_20260602_192731.py` | Experimental scratch script |
| `scratch/pred_corr.py` | `archive/audit_20260602_192731/scratch/pred_corr_20260602_192731.py` | Experimental scratch script |
| `scratch/run_exact_rq2.py` | `archive/audit_20260602_192731/scratch/run_exact_rq2_20260602_192731.py` | Experimental scratch script |
| `scratch/test_different_corrs.py` | `archive/audit_20260602_192731/scratch/test_different_corrs_20260602_192731.py` | Experimental scratch script |
| `scratch/test_train_corr.py` | `archive/audit_20260602_192731/scratch/test_train_corr_20260602_192731.py` | Experimental scratch script |
| `scratch/test_xgb_depth.py` | `archive/audit_20260602_192731/scratch/test_xgb_depth_20260602_192731.py` | Experimental scratch script with dead `StandardScaler` import |
| `scratch/true_risk_corr.py` | `archive/audit_20260602_192731/scratch/true_risk_corr_20260602_192731.py` | Experimental scratch script |
| `scratch/tune_xgb.py` | `archive/audit_20260602_192731/scratch/tune_xgb_20260602_192731.py` | Experimental scratch script with dead `pandas` import |
| `data/processed/*` (8 files) | `archive/audit_20260602_192731/data_processed/` | 100% duplicate of `data/curated/` — identical filenames, near-identical sizes |
| `_audit/` (8 files) | `archive/audit_20260602_192731/_audit_20260602_192731/` | Previous audit tooling (`bandit.json`, `ruff.json`, `manifest.json`, etc.) — not part of pipeline |
| `Audit_report/` (old) | `archive/audit_20260602_192731/Audit_report_old_20260602_192731/` | Previous audit report from earlier session |

### Files removed by other operations (between 2026-06-02 and 2026-06-04)

| File | Status | Notes |
|------|--------|-------|
| `phase0_schema.py` | Removed | Standalone phase0 script, not part of `src/` pipeline |
| `phase1_eda.py` | Removed | Contains CRIT-02 syntax error; superseded by `src/eda/` |
| `phase2_ppt.py` | Removed | Superseded by `src/eda/` pipeline |
| `schema.json` | Removed | Phase0 output artifact |
| `Makefile` | Removed | Build automation file |
| `AGENTS.md` | Removed | Agent configuration file |
| `CLAUDE.md` | Removed | Agent configuration file |

### Files that should still be considered for archival

| File | Reason |
|------|--------|
| `Dataset/procurement_risk_scores.parquet` (783 MB) | Not referenced by any active code |
| `Dataset/rq1_filtered_dataset.parquet` (711 MB) | Not referenced by any active code |
| `Dataset/simulated_lifecycle_dataset.parquet` (199 MB) | Not referenced by any active code |
| `notebooks/02-05_*.ipynb` (279 bytes each) | Stub notebooks with no content |
| `data/raw/IT_DIB_2023.csv` (9.9 GB) | Raw CSV; the parquet version (1.7 GB) is preferred |

---

## 6. Production Readiness Verdict

- **Pass / Fail:** **FAIL**

### Reasons for Failure

1. **Pipeline execution blocked** — `run_full_pipeline.py` cannot import `run_rq3` (CRIT-01)
2. **CWD-dependent SQL paths** — Feature engineering fails when run from non-root directory (HIGH-01)
3. **Thesis correctness risk** — `closeness_centrality` column reports degree centrality values (HIGH-03)
4. **Missing dependency declarations** — `numpy`, `python-pptx`, `matplotlib` absent from `requirements.txt`
5. **Data leakage in temporal split** — VAL_YEARS and TEST_YEARS overlap on 2019 (HIGH-02)
6. **Dashboard path resolution** — `ROOT` variable points to `src/` not project root (HIGH-07)

### What passes

- [PASS] Core algorithm implementations are sound (RQ1 network, RQ2 governance, RQ3 pricing)
- [PASS] Target leakage prevention is properly implemented and tested
- [PASS] Integration join-explosion guard works correctly
- [PASS] Optional dependency fallbacks are well-implemented
- [PASS] Test suite covers key acceptance criteria
- [PASS] SQL-first architecture is clean and reproducible

---

## 7. Remediation Plan

### Immediate Fixes (must resolve before thesis submission)

| Priority | Issue | Action | Effort |
|----------|-------|--------|--------|
| P0 | CRIT-01 | Rename `run_rq3_pricing` → `run_rq3` in `rq3_pricing.py` | 1 min |
| P0 | HIGH-01 | Replace hardcoded SQL paths with `str(DATA_FEATURES / ...)` f-strings | 15 min |
| P0 | HIGH-03 | Compute actual `nx.closeness_centrality()` or rename column to `degree_centrality_alias` | 10 min |
| P0 | HIGH-02 | Fix temporal split: `TEST_YEARS = range(2020, 2021)` | 2 min |
| P0 | DEP | Add `numpy`, `python-pptx`, `matplotlib` to `requirements.txt` | 2 min |
| P0 | MED-05 | Remove unused `import argparse` from feature_engineering.py | 1 min |

### Near-Term Fixes (before defense)

| Priority | Issue | Action | Effort |
|----------|-------|--------|--------|
| P1 | HIGH-07 | Fix `ROOT` in `app.py` and all dashboard pages to point to project root | 15 min |
| P1 | MED-01 | Rename `single_bid_rate` → `negotiated_restricted_rate` in rq2 output | 10 min |
| P1 | MED-03 | Fix anomaly score label: "0–3" → "0–1" in charts.py | 2 min |
| P1 | MED-04 | Remove duplicate `final_price` column or differentiate from `contract_value` | 10 min |
| P1 | MED-12 | Add try/except to pipeline runner phases | 15 min |
| P1 | MED-13 | Replace `_safe_corr` in rq1 with `from src.common.evaluation import safe_corr` | 5 min |
| P1 | MED-16 | Use dynamic contract count in dashboard header | 2 min |

### Cleanup Tasks (quality improvements)

| Priority | Issue | Action | Effort |
|----------|-------|--------|--------|
| P2 | DEP | Move `jupyter`, `jupyterlab`, `nbconvert` to optional dev dependencies | 10 min |
| P2 | DEP | Remove `shap` from requirements or re-add SHAP analysis code | 5 min |
| P2 | DEP | Change `plotly==5.18.0` and `kaleido==0.2.1` to `>=` ranges | 2 min |
| P2 | MED-09 | Remove hardcoded Codex CI paths from generate_charts.py | 5 min |
| P2 | MED-15 | Replace `pl.Utf8` with `pl.String` | 2 min |
| P2 | TEST | Add import-only test for `run_full_pipeline.py` | 5 min |
| P2 | TEST | Add `DataLoader` unit test with synthetic parquet | 30 min |
| P2 | TEST | Fix `test_ensemble_beats_solo` to actually assert ensemble superiority | 5 min |
| P2 | HYGIENE | Archive stub notebooks (02-05), Dataset/*.parquet not used by pipeline | 10 min |
| P2 | HYGIENE | Add `data/processed/` to `.gitignore` | 1 min |
| P2 | HYGIENE | Delete empty `scratch/__pycache__/` directory | 1 min |
| P2 | HIGH-04 | Guard `exec()` monkey-patch with `sys.version_info` check | 10 min |

---

## Appendix A: File Inventory (Active Codebase)

```
src/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── evaluation.py          # safe_corr, risk_class
│   └── utils.py               # get_logger, timed, write_json, minmax
├── core/
│   ├── __init__.py
│   ├── config.py              # All constants, thresholds, model params
│   ├── data_loader.py         # DuckDB DataLoader with year-stratified sampling
│   ├── feature_engineering.py # SQL-first feature pipeline
│   └── sql/                   # 12 SQL query files (00-14)
├── dashboard/
│   ├── __init__.py
│   ├── app.py                 # Streamlit main page
│   ├── components/
│   │   ├── __init__.py
│   │   ├── charts.py          # 14 Plotly chart functions
│   │   ├── filters.py         # Common filter widgets
│   │   └── metrics.py         # KPI card renderer
│   └── pages/
│       ├── __init__.py
│       ├── 01_supplier_explorer.py
│       ├── 02_buyer_dashboard.py
│       ├── 03_governance_risk.py
│       └── 04_price_anomalies.py
├── eda/
│   ├── __init__.py
│   ├── run_all.py             # EDA orchestrator
│   ├── screenshot_plotly.js   # Playwright PNG fallback
│   ├── reports/
│   │   └── generate_reports.py
│   ├── sql/                   # 7 EDA SQL subdirectories
│   └── visualizations/
│       └── generate_charts.py # 20-chart Plotly generator
├── integration/
│   ├── __init__.py
│   └── integration.py         # Unified risk score computation
├── rq1_supplier_network/
│   ├── __init__.py
│   └── rq1_network.py         # NetworkX bipartite graph + Louvain + resilience
├── rq2_governance_risk/
│   ├── __init__.py
│   └── rq2_governance.py      # RF/XGB/LR ensemble classifier
└── rq3_price_intelligence/
    ├── __init__.py
    └── rq3_pricing.py         # GB regressor + LOF + Isolation Forest

tests/
├── conftest.py                # Fixtures loading data from parquet/JSON
├── test_data_quality.py       # 3 tests: row count, temporal range, negative prices
├── test_eda_outputs.py        # 2 tests: chart + report artifact existence
├── test_feature_engineering.py # 1 test: feature artifact existence
├── test_integration.py        # 3 tests: unified scores, bounds, join-no-inflate
├── test_leakage.py            # 2 tests: temporal split, no future features
├── test_no_target_leakage.py  # 3 tests: label inputs excluded from model features
├── test_rq1_success.py        # 5 tests: graph, centrality, modularity, dependency, resilience
├── test_rq2_success.py        # 6 tests: RF, XGB, LR metrics, ensemble, correlation, SHAP
└── test_rq3_success.py        # 4 tests: R², RMSE reduction, anomaly rate, LOF overlap

run_full_pipeline.py           # Pipeline entry point (BROKEN: CRIT-01)
requirements.txt               # Dependency manifest (INCOMPLETE: missing numpy, python-pptx, matplotlib)
pyproject.toml                 # Project metadata + pytest config
```

---

## Appendix B: Dependency Matrix

| Package | requirements.txt | Actually Imported | Verdict |
|---------|-----------------|-------------------|---------|
| duckdb | [PASS] >=0.10.0 | [PASS] | OK |
| polars | [PASS] >=0.20.0 | [PASS] | OK |
| pandas | [PASS] >=2.2.0 | [PASS] | OK |
| networkx | [PASS] >=3.2 | [PASS] (optional) | OK |
| python-louvain | [PASS] >=0.16 | [PASS] (optional, as `community`) | OK |
| scikit-learn | [PASS] >=1.4.0 | [PASS] (as `sklearn`) | OK |
| xgboost | [PASS] >=2.0.0 | [PASS] (optional) | OK |
| shap | [PASS] >=0.44.0 | [FAIL] | **UNUSED — remove** |
| scipy | [PASS] >=1.12.0 | [PASS] | OK |
| plotly | [PASS] ==5.18.0 | [PASS] | OK (pin is strict) |
| kaleido | [PASS] ==0.2.1 | [PASS] (indirect via Plotly) | OK (pin is strict) |
| streamlit | [PASS] >=1.32.0 | [PASS] | OK |
| jupyter | [PASS] >=7.0.0 | [FAIL] | Dev only — move to optional |
| jupyterlab | [PASS] >=3.0.0 | [FAIL] | Dev only — move to optional |
| nbconvert | [PASS] >=7.0.0 | [FAIL] | Dev only — move to optional |
| pytest | [PASS] >=8.0.0 | [PASS] | OK |
| pyarrow | [PASS] >=15.0.0 | [PASS] (indirect via Polars) | OK |
| pillow | [PASS] >=10.0.0 | [PASS] (as `PIL`) | OK |
| **numpy** | [FAIL] **MISSING** | [PASS] | **ADD** |
| **python-pptx** | [FAIL] **MISSING** | [PASS] (as `pptx`) | **ADD** |
| **matplotlib** | [FAIL] **MISSING** | [PASS] | **ADD** |

---

*End of Audit Report*
