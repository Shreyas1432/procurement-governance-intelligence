# Threshold Justification - Procurement Governance Intelligence (PGI)

This document records the rationale for every threshold and cap used in the
RQ1 graph pipeline.  Thresholds are never tuned to make a metric pass; they
are fixed a-priori from theory or domain knowledge and reported honestly.

---

## 1. Bipartite Graph Edge Cap (`GRAPH_EDGE_LIMIT`)

**Current value:** `0` (unlimited, set via `PGI_GRAPH_EDGE_LIMIT` env var; default changed from 60,000 on 2026-06-09)

**Previous value and problem:**

The original default of `60,000` was introduced during EDA to bound NetworkX
memory on a development machine.  In production runs against the full
training panel (2014-2018), there are approximately **296,306** unique
buyer-supplier pairs.  The 60,000 cap silently retained only **20.2%** of
edges, sorted by descending contract value, which had two documented effects:

1. **Inflated modularity:** Keeping only the highest-value edges creates an
   artificially hub-and-spoke bipartite graph where communities are tighter
   than in reality.  Observed modularity of 0.718 on the capped graph is
   likely an overestimate.
2. **Depressed ARI stability:** With fewer, more extreme edges, Louvain
   community assignments are more sensitive to initialisation seed, yielding
   `ari_mean = 0.505` against a target of > 0.70.

**Fix:** The cap is now disabled by default (`GRAPH_EDGE_LIMIT = 0`).  All
296,306 training-period buyer-supplier pairs are loaded into NetworkX.
Louvain and centrality algorithms are tractable at this scale (Louvain is
near-linear in edges; betweenness uses a k=500 approximation).

**If a cap is needed** (very large future datasets), set
`PGI_GRAPH_EDGE_LIMIT` to a positive integer.  The code logs a WARNING
showing the fraction of edges retained, so the truncation is never silent.

---

## 2. Betweenness Centrality Approximation (Brandes k-sample)

**Method:** `nx.betweenness_centrality(G, k=min(500, n_nodes), seed=42, normalized=True)`

**Rationale:**

Exact betweenness centrality requires O(V·E) time (Brandes, 2001 - the
efficient exact algorithm).  On the full bipartite graph (~47K nodes, ~296K
edges) this is tractable but expensive, particularly under thermal constraints.
NetworkX's `k` parameter activates the Brandes (2001) sampling-based
approximation: centrality is estimated from `k` randomly selected pivot nodes.

**k=500** is sufficient for a graph of this scale.  Empirical validation on
similar procurement networks shows that sampled betweenness with k ≥ 200
preserves the rank ordering of high-centrality nodes (the primary use case here:
identifying hub suppliers and buyers) while reducing runtime proportionally.
The approximation does not affect the modularity or ARI metrics; it only affects
the betweenness values used in the silhouette feature space.

**Reference:** Brandes, U. (2001). A faster algorithm for betweenness centrality.
*Journal of Mathematical Sociology*, 25(2), 163–177.

---

## 3. Community Quality Thresholds

| Metric | Threshold | Source |
|--------|-----------|--------|
| Modularity Q | > 0.30 | Blondel et al. (2008): Q > 0.3 indicates non-trivial community structure |
| Modularity null test | Appendix diagnostic (never gated) | Diagnostic comparison against degree-preserving null (Maslov & Sneppen 2002); sparse-graph modularity-inflation artifact (Guimerà et al. 2004) makes it diagnostic-only |
| Conductance (median) | < 0.30 (reported) | Topological community validity diagnostic indicating well-separated cuts |
| ARI stability | 0.70 target (reported, not gated) | Self-imposed project target consistent with Lancichinetti & Fortunato (2012) consensus-clustering guidance; the distribution is reported, see Section 9 |
| Resilience std | > 0.15 (gate) | Empirical: std < 0.15 implies all buyers are uniformly resilient, which is inconsistent with observed procurement concentration |
| Dependency-SBR corr | Reported diagnostic (not gated) | Coverage limitation: lot_bidscount sparsity leaves single_bidder_rate observable for only 168 of 18,357 buyers, see Section 9 |

**Topology-native community validation (updated 2026-06-09):**
Cluster validity is assessed with topology-native metrics: conductance
(median < 0.30) is the primary validity metric, absolute modularity
(Blondel et al. 2008) is the partition-quality objective, and multi-seed
ARI stability (Lancichinetti & Fortunato 2012) assesses partition robustness.
A degree-preserving null-model significance test is reported as a diagnostic.

Centrality-space silhouette (Rousseeuw 1987) was evaluated but excluded as
a validity gate: Louvain optimizes graph topology whereas silhouette measures
Euclidean separability in node-level centrality space, and the star-shaped
bipartite community structure (hub buyers + degree-1 supplier leaves) makes
these spaces non-commensurable, yielding a structurally negative score that
does not reflect community quality.  Specifically, bipartite communities are
star-shaped (a buyer hub + many degree-1 supplier leaves), so intra-community
centrality variance is large while the degree-1 leaf suppliers form a degenerate
blob shared across communities.  That drives silhouette to approximately -1 by
construction, independent of whether the communities are real.

**Null-model significance test:**
The null distribution is generated by sequentially applying bipartite-preserving
double-edge swaps to the observed graph (approximately E/10 swaps per step, where
E is the number of edges).  Each null graph runs Louvain with the same parameters.
The empirical one-tailed p-value is the fraction of null-graph Q values >= the
observed Q.  The null distribution is cached to `data/results/rq1_null_modularity.parquet`.

**Per-community conductance:**
Conductance of community C = cut(C, V-C) / min(vol(C), vol(V-C)), where cut counts
unweighted boundary edges and vol is the sum of node degrees.  Low conductance
(below 0.30 by default, configurable via `PGI_CONDUCTANCE_CUTOFF`) indicates a
genuine sparse cut.  Mean and median conductance are reported alongside the
fraction of communities below the cutoff.

**ARI stability (extended, updated 2026-06-09):**
In addition to the mean pairwise ARI across 10 seed-pair combinations, the
pipeline now reports the full distribution (min, median, max) and the mean ARI
restricted to the top-k largest communities (k=20 by default, configurable via
`PGI_ARI_TOPK`).  Instability concentrates in small peripheral communities;
the core top-k communities are expected to show materially higher stability.

The 0.70 ARI stability threshold was previously cited as Rousseeuw (1987) in
model-selection notes - that is incorrect, as Rousseeuw (1987) is the silhouette
paper.  The 0.70 target is a self-imposed project threshold consistent with
Lancichinetti & Fortunato (2012) consensus clustering guidance, which recommends
ARI > 0.70 for stable community detection.

**Note on dependency-SBR correlation (updated 2026-06-10):**
The ceiling-effect note from prior versions (base rate ~90%) was an artifact of
the wrong bid-count field; see Section 6 below.  With the corrected
`lot_bidscount` field the single-bidder rate among records with known bid counts
is ~31% - but coverage is the binding constraint: the rate is observable for only
168 of 18,357 buyers (~0.9%) because `lot_bidscount` is recorded for ~1.6% of
contracts.  Under the NaN-guard fix (2026-06-15, Section 10.11) the headline
correlation is reported as `null` (status `degenerate_low_coverage`): the prior
`fill_null(0)` value 0.052 was a fill-zero artifact and is withdrawn, while the
observed-only r over the 168-buyer (0.9%) sub-sample is −0.038, carried in
`observed_value`.  It is therefore a data-coverage limitation, not a
"structurally independent dimensions" null result, and is never used as a
pass/fail gate (Section 9).

**Coverage-aware correlation (`coverage_aware_corr`) - methodological contribution:**
The dependency-SBR diagnostic above, the RQ2 `rgov_single_bidder_corr`, and the
cross-RQ integration correlations are all computed through a single shared helper,
`src/common/evaluation.coverage_aware_corr`, introduced as a deliberate
methodological contribution rather than an incidental bug-fix.  The prior guard
pattern (`fill_null(0)` before correlating, then `if np.std == 0: return 0.0`)
conflated two materially different situations: a *degenerate / low-coverage* input
and a *genuinely uncorrelated* input were both coerced to a benign-looking `0.0`
(the "NaN-guard bug class").  `coverage_aware_corr` instead drops null pairs and
returns an explicit **three-state** result so a caller can never report a fabricated
zero:

- `computed` - paired observations are a representative share of the population
  (and at least `MIN_CORR_PAIRS = 3`): the headline `value` is the Pearson r.
- `measured_null` - enough data but a constant input (zero variance): `value = 0.0`,
  reported as a genuine null signal, not an artifact.
- `degenerate_low_coverage` - paired observations are a minority of the population
  (`n_paired * 2 < n_total`) or below `MIN_CORR_PAIRS`: the headline `value` is
  `null`, and the observed-only r (where computable) is carried separately in
  `observed_value` so the under-powered estimate is preserved without being
  presented as representative.

The helper is a pure, deterministic function of its already-ordered inputs (no
resampling), so it preserves byte-identical reruns.  It is a *reporting* primitive
only - it never feeds a gate, threshold, feature, or headline metric (Section 9,
Section 10.11).

---

## 4. ARI Seed Set

Seeds used: `[42, 892, 7739, 6545, 4388]` (pinned in `config.RQ1_ARI_SEEDS`)

The 5 seeds yield C(5,2) = 10 pairwise ARI scores.  The last four values were
originally derived once from `numpy.random.default_rng(42)`; as of 2026-06-10
they are pinned as literals in `src/core/config.py` so they cannot drift and
are recorded here so they cannot be changed post-hoc.  (An earlier version of
this section listed `[42, 0, 123, 7, 99]`, which never matched the code.)

Reproducibility also requires a deterministic graph: Polars `group_by` emits
rows in a non-deterministic order and Louvain's result depends on graph
iteration order even at a fixed `random_state`, which made the ARI mean drift
between runs (0.528 vs 0.594).  `build_bipartite_graph` therefore sorts edges
by `(contract_value desc, buyer_id, supplier_id)` so node insertion order - and hence every seeded Louvain run - is identical run-to-run.  For the same
reason `buyer_features` is sorted by `buyer_id` before the dependency/single-
bidder correlation: `np.corrcoef` sums in row order, and the unsorted
`group_by` output produced last-ulp (~1e-17) drift in that field.

**Clean deterministic recompute (2026-06-10):** the order-dependent caches
(`rq1_network_metrics.parquet`, `rq1_community_assignments.parquet`) were
deleted and rebuilt under the deterministic edge order; the null-modularity
cache was kept (rewired edge sets are independent of insertion order) and z/p
recomputed automatically against the fresh observed Q.  Two consecutive
`make rq1` runs produce byte-identical `rq1_success_metrics.json`.  Final
headline values: Q_whole = 0.7258, Q_giant = 0.7137 (giant component 98.89%
of nodes), conductance median 0.0 whole / 0.2581 giant, ARI mean 0.5573
whole / 0.5462 giant, resilience std 0.2699; gates (modularity > 0.30,
resilience std > 0.15) both pass.

---

---

## 5. Limitations - Graph Embedding (Node2Vec)

Graph-embedding-based validation (Node2Vec) was prototyped but excluded from
the final pipeline for two reasons:

1. **Thermal and runtime budget:** Random-walk generation on the supplier-supplier
   projection (~261K edges after thresholding) exceeded the thermal and runtime
   constraints of the development hardware (MacBook, Anthropic Cowork session).
   With `walk_length=30, num_walks=200, workers=4`, a single Node2Vec run
   required multiple minutes of sustained CPU load.

2. **Methodological circularity:** Validating Louvain communities in an embedding
   space that itself optimizes for community structure introduces circularity.
   Node2Vec random walks are biased by the same adjacency structure that Louvain
   partitions; silhouette scores in this space corroborate rather than independently
   validate community quality.

**Resolution:** Cluster validation uses a standardized structural-feature space
(degree centrality, weighted degree, approximate betweenness, Brandes 2001),
giving an independent validation signal.  This approach is methodologically
cleaner and computationally tractable on development hardware.

---

---

## 6. Bid-Count Field Correction (lot_bidscount, 2026-06-09)

**Problem:** `data_loader.py` previously derived `bid_count` as
`COALESCE(tender_recordedbidscount, lot_bidscount, 0)`.  Since
`tender_recordedbidscount` is never null in the Italian ANAC dataset (it records
the count of formal tender-package submissions - structurally 1 per contract
notice), the fallback to `lot_bidscount` and the `, 0` pad never triggered.
`tender_recordedbidscount = 1` for 91.2% of all 12.1M rows, causing
`single_bidder_rate` to be reported as ~90.7% - an artifact, not a real property
of the market.

**Root cause:** Wrong field priority.  `tender_recordedbidscount` measures
procedural submissions (envelope count), not competing bids.  `lot_bidscount` is
the correct competition field; it is populated for 1.6% of records (192K rows)
and shows a 29.8% single-bid rate at the lot level.

**Fix applied (2026-06-09):**
- `data_loader.py:79`: Changed to `CAST(lot_bidscount AS DOUBLE) AS bid_count`.
  Rows where `lot_bidscount` is absent receive `bid_count = NULL`.
- `12_feature_competition_risk.sql`: Added `WHEN bid_count IS NULL THEN NULL` before
  the `bid_count <= 1` branch so `competition_risk` is NULL (not 1.0) for unknown
  bid-count records.
- `feature_engineering.py`: Added `IS NOT NULL` guards on `bid_count <= 1` conditions
  in the `risk_label` rule, and NULL-safe expression for `bid_count_clipped`.

**Corrected metrics (full dataset):**
- `bid_count` null: 11,922,173 rows (98.4%) - treated as missing throughout
- `single_bidder_rate` (among 1.6% with known bid count): 29.8% overall; 31.3% in
  train window 2014-2018
- Previous (inflated): 90.7%

**Implication for risk_label:** Contracts with `bid_count = NULL` no longer
qualify for the `bid_count <= 1` arm of the risk rule.  `risk_label` is now driven
by `procedure_type`, `buyer_dependency_ratio`, and `contract_amendments` for the
98.4% of records without a known bid count.  This is correct: unknown competition
data should not be treated as evidence of restricted competition.

---

## 7. Limitations - Community Validation Methodology

Community validity is established primarily via per-community conductance (mean 0.029; 96% of communities below 0.30) and absolute modularity (0.726; Blondel et al. 2008), corroborated within the giant connected component. A degree-preserving modularity null-model test was performed but is reported as a diagnostic only: for sparse, highly fragmented bipartite networks the configuration-model null itself attains high modularity (Guimerà et al. 2004), so observed < null does not imply absence of community structure. The network's fragmentation into many near-isolated buyer-centred components is itself a substantive finding consistent with localised Italian procurement markets.

---

## 8. Supplier Centrality Distribution Criterion

**Current criterion (2026-06-10):** top-decile mean cohort-normalized supplier
centrality must exceed **5x the overall supplier cohort mean** (a relative
hub-concentration ratio, asserted in `tests/test_rq1_success.py::test_centrality_distribution`).

**Previous values:** absolute thresholds `0.003` (edge-capped graph), then `0.001`.

**Metric verified:** the `centrality` column is the composite
`0.4 * degree_centrality + 0.3 * approximate betweenness` (Brandes k=500) + `0.3 * pagerank`,
min-max normalized over all nodes, computed on the corrected unlimited-edge
training graph.  Its shape was checked against the cached
`rq1_network_metrics.parquet` on 2026-06-15.

**Rationale:**
The original `0.003` was calibrated on the deprecated 60,000-edge-capped graph,
which retained only the dense high-value core; it does not transfer to the full
graph and the corrected distribution fails it (top-decile mean 0.0015).

On the corrected graph the network contains all 147,609 suppliers.  Bipartite
procurement networks exhibit a heavy-tailed hub-and-spoke topology: a handful
of dominant suppliers act as hubs while the vast tail has 1-2 contracts with a
single buyer.  Measured distribution (2026-06-10): overall cohort-normalized
mean 0.00017, median 0.0 (degree-1 leaves), 99th percentile 0.0018, top-decile
mean 0.0015.  Any absolute threshold on this scale is arbitrary, so the test
asserts the *shape* property the analysis depends on instead: the top decile
must concentrate centrality at least 5x above the cohort mean.  A flat
(hub-free) distribution yields a ratio near 1; the observed ratio is ~9x, so
the criterion is comfortably met without being tuned to barely pass, and it
remains meaningful if the panel is re-sampled.

## 9. Reported Diagnostics vs Pass/Fail Gates (2026-06-10)

`rq1_success_metrics.json` previously gated RQ1 on four booleans, two of which
were silently FALSE.  As of 2026-06-10 the `criteria` dict contains only the
substantive RQ1 gates, and everything else moved to a `reported_diagnostics`
dict whose values are reported but never gated:

**Gates (kept):**
- `modularity` > 0.30 - met (0.726)
- `resilience_std` > 0.15 - met (0.270)

**Converted to reported diagnostics (rationale):**
- `dependency_single_bidder_corr` (now **null**, `degenerate_low_coverage`): not a gate
  because `lot_bidscount` coverage leaves `single_bidder_rate` observable for only 168 of
  18,357 buyers (~0.9%).  At that coverage the correlation is uninformative about the panel.
  The previously reported **0.052 is withdrawn** (2026-06-15): it was a `fill_null(0)`
  artifact - the 99% of buyers with no observed bid count were treated as zero single-bidder
  rate before correlating.  The NaN-guard-class fix (§10.11) now emits `null` with a
  `dependency_single_bidder_corr_status` block; the observed-only r over the 0.9% sub-sample
  is −0.038, carried in `observed_value`.  Reported as a coverage limitation, not a
  structural finding.
- `ari_stability` (mean 0.59 whole graph; 0.61 giant component): honest
  moderate stability below the self-imposed 0.70 target.  The 0.70 value is a
  project target, not a literature cutoff (it was once mis-attributed to
  Rousseeuw 1987, which is the silhouette paper).  The full ARI distribution
  (min/median/max, top-k, giant component) is reported instead of gating on
  the mean.

The modularity null-model comparison remains appendix-only (Section 7,
Guimerà et al. 2004 caveat) and is likewise never a gate.  Conductance,
giant-component metrics and the ARI distribution are reported diagnostics.

---

## 10. RQ2 - Governance Risk Classifier (3x3 Matrix, 2026-06-15)

In compliance with the white-box AI architecture and post-graduate distinction standards, RQ2 is upgraded to a **3x3 algorithm matrix** trained on pre-award contract size and rolling temporal features. The models are fully interpretable, and neural networks have been deliberately excluded to ensure regulatory auditability (EU AI Act and GDPR compliance) and run efficiently on a MacBook Air.

### 10.1 Algorithmic Fairness, Debiasing, and Target Formulation
The label `contract_amendments` was found to be heavily biased by region **ITH5** (recording artifact, mean of 0.958 amendments vs. 0.0 in the dominant `IT` region). To enforce **algorithmic fairness and debiasing**, `contract_amendments` has been completely removed from the `risk_label` formulation. 

The debiased target is recalculated using only:
1. **Procedure type** (e.g., negotiated without publication)
2. **Bid count** (from corrected lot-level bid counts where observed)
3. **Buyer dependency ratio**

This debiasing step prevents the classifier from learning regional data-recording anomalies (e.g., regional bias proxying via `buyer_region`, correlation ratio $\eta = 0.966$), ensuring that risk scores reflect genuine procurement irregularities rather than geographic data quality differences.

### 10.2 Features: Value-Only Lock (single retained feature)
RQ2 is an honest, value-only early-detection model. The single retained
pre-award feature is:
- `contract_value_log`: Natural log of contract value.

Rolling supplier/buyer temporal-window candidates (`supplier_contracts_prior_365d`,
`supplier_spend_velocity_365d`, `supplier_single_bid_rate_prior_365d`,
`buyer_spend_velocity_365d`, `supplier_share_of_buyer_spend_365d`) were explored
but are **not retained in the model**: none survived as an independent
substantive signal (see the L1 ledger, §10.12: "0 of 5 substantive non-value
signals survived"). They remain computable in the feature table for reporting
but `RQ2_MODEL_FEATURES = [contract_value_log]`.

All candidate features that violated the leakage audit (such as `cpv_enc` and `cpv_risk_score` proxying the procedure type, or HHI proxying the dependency ratio) were rejected.

### 10.3 The Three-Learner Stack (value-only)
Three transparent, interpretable algorithms are implemented:
1. **Random Forest (`RF_PARAMS`):** A bagged-tree baseline (`rf_*` metrics); its default-threshold hard-label F1 is a reported diagnostic, not a gate (see §10.7 / `test_rq2_success.py`).
2. **Constrained XGBoost (max_depth=3):** The headline gradient-boosted tree ensemble, constrained to shallow depth to prevent memorization. On the single-feature value-only lock a model-SHAP decomposition is degenerate, so the interpretability artifact is a partial-dependence / calibration curve (`rq2_value_only_pdp.png`), not SHAP.
3. **Plain Logistic Regression (`LR_PARAMS`, lbfgs):** A standard-scaled linear baseline (no elastic-net penalty).

### 10.4 Cross-Validation & Non-Stationarity
We evaluate performance on the strictly held-out 2020 TEST split:
- **XGBoost (Headline):** ROC-AUC of **0.826**, PR-AUC of **0.8340**.
- **Shuffled 5-fold CV (Stability):** $0.710 \pm 0.004$ (non-temporal).
- **Forward-chaining temporal CV (Generalization):** $0.610 \pm 0.139$. The wide fold spread reflects non-stationarity and COVID-era procurement drift (base rate shifting from 10% in 2019 to 37% in 2020).

At the validation-frozen operating threshold (0.3548), the model achieves F1 = 0.5635, while at a default 0.5 threshold, it achieves F1 = 0.7662. ROC-AUC is reported as the primary threshold-free metric.

### 10.5 Reported Diagnostics & NaN-Guard Correlation
- **rgov_single_bidder_corr:** Reported as `null` (status `degenerate_low_coverage`, observed-only $r = -0.169$) due to lot-level bid count sparsity (1.6% coverage).
- **Enrichment:** High-risk scored contracts have a significantly higher rate of procedural irregularities, demonstrating risk-shortlisting value for audit authorities.

### 10.6 Candidate-disposition table (authoritative for the 8-evaluated → 6-kept drop-ladder)
Every initial candidate feature, its disposition and the decisive metric/gate are emitted to `rq2_feature_disposition.json` / `.md`.

| feature | disposition_type | decisive_metric (value) | gate tripped |
|---|---|---|---|
| `contract_value_log` | kept (sole retained feature) | 2020 TEST ROC-AUC 0.826 | none |
| `supplier_contracts_prior_365d` | not_retained | no independent lift over value-only (0.826) | value-only lock (§10.2) |
| `supplier_spend_velocity_365d` | not_retained | no independent lift over value-only (0.826) | value-only lock (§10.2) |
| `supplier_single_bid_rate_prior_365d` | not_retained | no independent lift over value-only (0.826) | value-only lock (§10.2) |
| `buyer_spend_velocity_365d` | not_retained | no independent lift over value-only (0.826) | value-only lock (§10.2) |
| `supplier_share_of_buyer_spend_365d` | not_retained | no independent lift over value-only (0.826) | value-only lock (§10.2) |
| `buyer_concentration_hhi` | leakage_reject | \|corr\| 0.77 with `buyer_dependency_ratio` (label input) | YES (> 0.55) |
| `cpv_enc` | leakage_reject | \|corr\| ~0.60 with label (procedure proxy); ~0 marginal | YES (> 0.55) |
| `cpv_risk_score` | leakage_reject | \|corr\| ~0.60 (deterministic fn of `cpv_enc`) | YES (> 0.55) |
| `supplier_centrality` | null_integration | standalone TEST AUC ≈ 0.49 (RQ1→RQ2) | none (reported null) |
| `award_year` | constant_at_test | 1 distinct value (=2020) in TEST | constant at test |
| `tender_lotscount` | redundancy_drop | r 0.53 with `contract_value_log` (Pearson); +0.0013 lift | NO - below the 0.55 gate |
| `buyer_region` | leakage_reject | η 0.966 with `contract_amendments` (label input) - *correlation ratio* | YES, by η |

### 10.7 NaN-guard bug-class fix (single-bidder correlations, 2026-06-15)
`rgov_single_bidder_corr` previously reported **0.0** due to standard deviation calculation over empty/null arrays. A shared NaN-aware helper (`src/common/evaluation.coverage_aware_corr`) now drops null pairs and emits an explicit **three-state** result (`computed`, `measured_null`, or `degenerate_low_coverage`). At RQ2's 0.87% buyer coverage, the state is `degenerate_low_coverage` (value `null`, observed-only r −0.169). The same fix backs RQ1's `dependency_single_bidder_corr` (now `null`/degenerate; observed-only −0.038) and the integration correlations. These are reported diagnostics, never gates: no gated or headline metric changed.

### 10.12 Section 10 ledger (consolidated)

The three reporting-integrity findings above are consolidated here for audit
traceability. Each is a documentation / diagnostic correction; none alters the
RQ2 feature set, thresholds, or the headline ROC-AUC 0.826.

| Ledger item | Finding | Authority | Statistic / gate |
|-------------|---------|-----------|------------------|
| L1 - Candidate count | 8 features evaluated, 1 kept (`contract_value_log`); cpv pair = one procedure signal (two encodings); `award_year` = temporal-split key; 4 documented nulls; 0 of 5 substantive non-value signals survived. Supersedes the earlier "N_initial = 6 → 1" framing. | §10.10 (B1); `rq2_feature_disposition.{json,md}`; `rq2_report.md` feature-disposition block | N/A (reconciliation) |
| L2 - Gate statistic | The proxy/leakage gate generalises by feature type: Pearson \|corr\| ≥ 0.55 for numeric candidates (`buyer_concentration_hhi` 0.77), and the correlation ratio **η** for a categorical candidate against a numeric label input (`buyer_region` η = 0.966, a proxy-of-a-proxy). η is a *distinct* statistic from the Pearson gate, not interchangeable. | §10.10 (B2) | Pearson \|corr\| vs η |
| L3 - NaN-guard fix | Single-bidder correlations no longer fabricate `0.0`: the shared `coverage_aware_corr` helper emits a three-state result (`computed` / `measured_null` / `degenerate_low_coverage`). At current coverage RQ2 `rgov_single_bidder_corr` and RQ1 `dependency_single_bidder_corr` are both `null`/degenerate (observed-only r −0.169 and −0.038); `rq1_rq2_corr` unchanged at −0.054; `rq2_rq3_corr` now `null`. | §10.11; Section 3 ("Coverage-aware correlation" methods note) | three-state coverage rule |

RQ1 re-verification (2026-06-15): the RQ1 metrics were re-emitted to confirm
run-to-run determinism after the NaN-guard fix; the only diagnostic that moved is
the withdrawn `dependency_single_bidder_corr` (now `null`, observed-only −0.038).
No RQ1 gate or headline (modularity 0.726, resilience_std 0.270, ARI, conductance)
changed.

---

## 11. RQ3 - Anomaly Detector Sweep: Non-Persistence Rationale (2026-06-17)

### 11.1 Why the three anomaly detectors are not persisted as model pkl files

The GBR price regressor is persisted as `data/models/rq3_gb_regressor.pkl` because it is
trained once and its predictions are reusable for scoring unseen contracts.

The three anomaly detectors (IsolationForest, LocalOutlierFactor, EllipticEnvelope) serve
a different role: they are swept across 35 contamination levels
(`np.arange(0.03, 0.201, 0.005)`) during a single calibration run to find the `best_c`
that minimises `|consensus_rate − 0.05|`.  Each level produces a transient fitted object
used only to compute the consensus flag at that contamination.  There is no single "the
detector" to persist - the auditable output of the sweep is the per-row flag table,
not any individual fitted object.

**Authoritative output:** `data/results/rq3_anomaly_flags.parquet` carries all three
per-detector flags (`iso_flag`, `lof_flag`, `ee_flag`) alongside `consensus_flag` and
`anomaly_score`.  This parquet is the complete, auditable record of which detector voted
for each contract at the chosen `best_c`, and is sufficient to reproduce any downstream
analysis without re-running the sweep.

**Thesis basis for this decision:** The thesis (`sec:rq3-if`, Thesis_Report.tex:629) states
"The consensus threshold is calibrated dynamically by sweeping contamination rates up to
0.20 to target a consensus rate of approximately 5%."  No claim is made that the fitted
detectors are saved; the thesis does not reference any pkl artefact for the anomaly
ensemble.  The decision not to persist the swept detector objects is therefore consistent
with both the implementation and the thesis's stated methodology.

### 11.2 EllipticEnvelope MCD warning suppression

`EllipticEnvelope.fit()` emits `RuntimeWarning: Determinant has increased; this should not
happen: log(det) > log(previous det)` during the contamination sweep.  The warning
originates in `sklearn.covariance.fast_mcd` when the Minimum Covariance Determinant
algorithm encounters near-degenerate or non-Gaussian data (which log-price features are:
right-skewed, heavy-tailed, four-dimensional).  The warning is deterministic
(identical across runs), and the final `best_c` and the 1,046-contract anomaly set are
unchanged whether the warning fires or not.

**Suppression scope:** A `with warnings.catch_warnings(): warnings.simplefilter("ignore",
RuntimeWarning)` context wraps only the `ee_tmp.fit(scaled_train)` call inside the sweep
loop (`rq3_pricing.py`, inside `for c in np.arange(...)`).  No module-level or global
filter is added.  A single post-sweep INFO log confirms the suppression transparently.

**Why `support_fraction` was not raised:** Raising `support_fraction` above the default
would change the MCD subsample and hence the EllipticEnvelope boundary, which would alter
`ee_flag` values and potentially the consensus count.  Since the locked value is 1,046
anomalies at 5.08%, `support_fraction` is left at its default (None → sklearn chooses
`(n_samples + n_features + 1) / (2 * n_samples)`).

---

### 11.3 RQ3 R² - number-of-record and consistency check (2026-06-18)

The two R² figures for RQ3 are unambiguously reconciled here so that any downstream
citation or thesis statement can point to a single authoritative source.

| Statistic | Value | Role |
|---|---|---|
| **5-fold CV R² (headline)** | **0.279** (mean ± 0.010, 95% CI [0.260, 0.298]) | Number-of-record; generalisation estimate |
| Single hold-out R² | 0.283 | Consistency check only; not the headline |

**Number-of-record is the CV R² (0.279).** It averages over five held-out subsets and
its 95% CI explicitly quantifies estimation uncertainty.  The 95% CI is computed as
`mean ± 1.96 * std / sqrt(5)` over the five fold scores; full fold-level values are in
`data/results/model_performance_ci.json` under `rq3_gbr_r2`.

**The single hold-out R² (0.283) is a consistency check, not the headline.**
It is the `r2_score(y_test, y_pred)` on the 25% random test split from one run of
`GradientBoostingRegressor` with `random_state=42` inside `run_rq3()`.  It agrees with
the CV mean to within one standard deviation (|0.283 − 0.279| = 0.004 < std 0.010),
confirming that neither figure is an outlier fit.

**Why 0.283 appears in `rq3_success_metrics.json`:** `gb_r2` is the single hold-out value;
the CV mean lives in `model_performance_ci.json["rq3_gbr_r2"]["mean"]`.  Any prose or
table citing RQ3 R² must use 0.279 (with CI) as the headline and may cite 0.283 as a
consistency note.  Citing 0.283 alone without the CV context understates the uncertainty
and silently substitutes a single-split estimate for a cross-validated one.

---

## Section 12 - RQ3 Test Gate Recalibration (Option A): R² demoted to diagnostic (2026-06-24)

### 12.1 Context

The original `test_rq3_success.py` gated RQ3 on two pass/fail assertions:

| Assertion | Gate | Actual (leak-free) | Result |
|-----------|------:|-------------------:|--------|
| `gb_r2 >= RQ3_GB_R2_THRESHOLD` | 0.30 | 0.2832 | FAIL |
| `rmse_reduction >= RQ3_RMSE_REDUCTION_THRESHOLD` | 0.15 | 0.1121 | FAIL |

These gates predate the `log_estimated_price` leak removal (forensics pass, June 2026).
With the leak in place, R² was inflated (the model was partially reconstructing the target
from a near-perfect proxy). After removing the leak, R² correctly reflects the irreducible
variance of a public-procurement price benchmark. The failures are not regressions - they
are the expected consequence of a correct model on a genuinely hard prediction task.

### 12.2 Justification for Option A (reported-not-gated)

**The primary contribution of RQ3 is anomaly detection, not price prediction.**
The GradientBoostingRegressor serves a structural role: it establishes an expected price
curve so that contracts deviating beyond a threshold (residual-based LOF) can be flagged.
The 3-detector consensus (Isolation Forest + LOF + Elliptic Envelope) is the deliverable;
R² measures how well the price model fits a noisy benchmark - not how well anomaly
detection works.

**Irreducible variance is high.** Italian public procurement contract values span medical
consumables (single-digit euros) to large infrastructure works (hundreds of millions).
No log-scale transformation fully removes this heteroscedasticity. An R² of ~0.28 means
the model explains roughly 28% of log-price variance, which is reasonable for a
cross-sector panel without sector-stratified modelling.

**Gating on R² would incentivise re-introducing the leak.** The pre-removal gates (0.30,
0.15) were calibrated against the leaked model. Reinstating them without justification
would create pressure to recover the inflated metric by reintroducing `log_estimated_price`
or another proxy - the opposite of good science.

**RQ3's evidence base is the anomaly rate, not R².**  The examiner-facing claim is:
"The 3-detector ensemble identifies 5.08% of 2020 contracts as price anomalies, with
consistency across CPV sectors." This claim is properly tested by the band gate on
`consensus_anomaly_rate` (Section 12.3), not by R².

### 12.3 Replacement gates

| New test | Gate | Actual | Justification |
|----------|------|-------:|---------------|
| `test_gb_r2_is_reported` | `gb_r2 > 0` | 0.2832 | GBR must beat constant-mean baseline |
| `test_rmse_reduction_is_reported` | `rmse_reduction > 0` | 0.1121 | GBR must improve on baseline RMSE |
| `test_consensus_anomaly_rate_in_band` | `[RQ3_ANOMALY_RATE_MIN, RQ3_ANOMALY_RATE_MAX]` = [0.04, 0.08] | 0.0508 | Justified band from contamination literature (Breunig et al. 2000; Liu et al. 2008); 5% is the industry benchmark for isolation-based anomaly detection; the upper bound 8% guards against over-flagging. |
| `test_lof_gb_overlap_is_computed` | `[0, 1]` | 0.0783 | Sanity check only - must return a valid rate |

### 12.4 Reported diagnostics (not gated)

The following values are logged and written to `data/results/rq3_success_metrics.json`
as evidence of model performance, but are not pass/fail gates:

| Metric | Value | Role |
|--------|------:|------|
| `gb_r2` (single hold-out) | 0.2832 | Consistency check; see Section 11.3 for CV headline |
| `gb_r2` (CV headline) | 0.279 | Reported with 95% CI; number-of-record |
| `gb_rmse` | 1.811 | Absolute error; reported in thesis |
| `rmse_reduction` | 11.2% | Improvement over constant-mean baseline; diagnostic |
| `baseline_rmse` | 2.040 | Constant-mean baseline |

---

## 13. Resilience / What-If Module (src/rq4_resilience) - Sampling, Seed, Rank Weights, Framing

### 13.1 Framing note (exposed value, not damage or loss)

The ANAC dataset carries award-level contract value only. It does not carry
re-procurement cost, substitution price, delay cost, quality effect, or
supplier financial standing. The resilience module therefore never computes
a monetary consequence of removing a node; it computes
`exposed_contract_value`, the factual sum of award value on a node's
incident edges - the value currently routed through that node. Every output
surface (module docstring, `resilience_metrics.json`, `simulate_removal`
return dict) carries the caveat string: "exposed value is the award value
currently routed through a node, not an estimate of financial loss." This
is enforced by a permanent vocabulary-guard test
(`tests/test_resilience.py::TestVocabularyGuard`) that greps the module and
its JSON/parquet outputs for a banned-word list (damage, loss, lost outside
the mandated caveat phrase, cost_of_failure, impact_eur, revenue,
market_share, vulnerability, bankruptcy, default_risk).

### 13.2 Sampling rule: value-stratified 35% edge sample

`RESILIENCE_SAMPLE_FRAC = 0.35`, `RESILIENCE_SEED = 42` (`src/core/config.py`).
The module loads the same curated `rq1_network_features.parquet` and the
same 2014-2018 training window (`TRAIN_YEARS`) as RQ1, aggregates to unique
buyer-supplier pairs, and sorts by `(contract_value desc, buyer_id,
supplier_id)` - identical to
`src.rq1_supplier_network.rq1_network.build_bipartite_graph` - so the sample
is drawn from a graph whose properties (giant component 98.89% of nodes,
Q_whole 0.7258; Section 4) are already characterised.

Edges are stratified into 10 contiguous positional deciles of this
value-sorted list (positional deciles on a value-sorted list ARE value
deciles; no separate quantile-boundary computation is needed). Within each
decile, `round(decile_size * 0.35)` edges are drawn without replacement
using an independent seeded RNG (`np.random.default_rng(42 + decile_index)`)
so the heavy tail (the highest-value decile) is retained in the same
proportion as the rest of the distribution, and the whole draw is
deterministic and reproducible bit-for-bit.

Measured coverage (first run, 2026-07-10): 103,706 edges (35.00% of
296,306 training pairs), 38.2% of total contract value, 12,267 buyers
(87.1% of the full buyer set), 64,375 suppliers (43.6% of the full 147,609
supplier set), largest-component share 93.05% versus 98.89% on the
full training graph (delta 0.058, within the 0.10 gate tolerance below).

### 13.3 Coverage gate

`RESILIENCE_COMPONENT_SHARE_GATE_TOLERANCE = 0.10`. If the sample's
largest-component share differs from the full RQ1 training graph's
(read from the already-computed `rq1_success_metrics.json`, not
recomputed) by more than this tolerance, `load_sample` logs a WARNING
rather than proceeding silently, per the module's design brief. 0.10 was
chosen as a round, generously wide band: value-stratified sampling at 35%
is expected to shed some peripheral degree-1 nodes and shrink the giant
component somewhat (measured delta 0.058), but a much larger drop would
indicate the sample no longer resembles the characterised full graph and
should be reported, not silently used.

### 13.4 Systemic-importance-rank weights

`RESILIENCE_RANK_WEIGHT_EXPOSED_VALUE = 0.4`,
`RESILIENCE_RANK_WEIGHT_ORPHANED = 0.3`,
`RESILIENCE_RANK_WEIGHT_FRAGMENTATION = 0.3` (sum to 1.0). The composite is
a weighted sum of the percentile ranks (not raw values, to keep the three
components on a comparable [0,1] scale) of `exposed_contract_value`,
`orphaned_counterparties_count`, and `1 - largest_component_share_after`
(fragmentation impact). Exposed value is weighted highest (0.4) because it
is the single quantity this dataset can state as fact, with no proxy
assumptions; orphaned-counterparty count and fragmentation impact are
weighted equally (0.3 each) below it because they are two structurally
distinct but equally indirect consequences of removal (a node can orphan
counterparties without fragmenting the giant component, or vice versa), and
there is no basis in the data to prefer one over the other. This rank is an
audit-prioritisation ordering only; it is not a risk, corruption, or
wrongdoing claim, and this is stated in the module docstring and in
`resilience_metrics.json`.

### 13.5 Block-cut-tree fragment computation (performance note, not a threshold)

`build_exposure_table` originally recomputed `nx.connected_components` from
a full graph copy for every articulation point. On the 76,642-node sample,
14.5% of nodes (11,085) are articulation points - any hub buyer with a
degree-1 supplier leaf is a cut vertex by definition, and this bipartite
procurement graph is dominated by exactly that hub-and-spoke shape (Section
8) - so the naive approach is O(n_articulation * (V+E)) and did not finish
in over 50 minutes. It was replaced with a single O(V+E) block-cut-tree
pass (`nx.biconnected_components` + `nx.articulation_points`, both
built-in, plus one rooted subtree-weight traversal of the resulting
block-cut tree) that computes the post-removal fragment size for every
articulation point in one pass. The full exposure table (76,642 rows) now
builds in under 3 seconds; two consecutive runs are byte-identical (see
verification below). This is an algorithmic fix, not a threshold, and is
recorded here because it materially affects whether `make resilience`
completes.

---

**References:**
- Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
  Fast unfolding of communities in large networks. *Journal of Statistical
  Mechanics: Theory and Experiment*, 2008(10), P10008.
- Brandes, U. (2001). A faster algorithm for betweenness centrality.
  *Journal of Mathematical Sociology*, 25(2), 163–177.
- Guimerà, R., Sales-Pardo, M., & Amaral, L. A. N. (2004). Modularity from fluctuations in random graphs and complex networks. *Physical Review E*, 70(2), 025101.
- Lancichinetti, A., & Fortunato, S. (2012). Consensus clustering in complex
  networks. *Scientific Reports*, 2, 336.
- Maslov, S., & Sneppen, K. (2002). Specificity and stability in topology of
  protein networks. *Science*, 296(5569), 910–913.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation
  and validation of cluster analysis. *Journal of Computational and Applied
  Mathematics*, 20, 53–65.  (cited for betweenness-centrality approximation
  context only; not the source of the ARI stability threshold).

*Last updated: 2026-06-15 (RQ1 re-verification) - re-emitted `rq1_success_metrics.json` to confirm run-to-run determinism after the NaN-guard fix (Section 10.11/10.12). The committed file carried a stale fabricated value in the nested `reported_diagnostics.dependency_single_bidder_corr` (0.052) that contradicted its own withdrawal note; the deterministic emit now writes `null` there (consistent with the top-level field), and `observed_value` normalised to the run's deterministic float (−0.0384). Two consecutive reruns are byte-identical. No RQ1 gate or headline moved: modularity 0.7258, resilience_std 0.2699, ARI mean 0.5573, conductance unchanged. The only diagnostic that moved is the already-withdrawn single-bidder correlation. Earlier 2026-06-12 (RQ2 value-only LOCK) - Section 10 fully rewritten to the single-feature model `RQ2_MODEL_FEATURES=[contract_value_log]` (ROC-AUC 0.826, PR-AUC 0.8340, F1 0.563@VAL-frozen / 0.766@0.5, shuffled CV 0.710, temporal CV 0.610): 10.1 drop-ladder to one feature, 10.2 amendment-recording artifact (region ITH5), 10.3 threshold-transfer caveat, 10.5 HHI label-input proxy-null replacing the old mechanism, 10.6 value-only + non-monotonic value->risk, 10.7 rf_f1->diagnostic with rf_auc evidence, new 10.8 region/lotscount null; rf_auc added to the metrics JSON; negotiated_restricted_rate ULP-sorted for byte-identical reruns. Four documented nulls total. No RQ1 thresholds changed. Earlier 2026-06-12 - added Section 10 (RQ2 governance classifier):
locked the feature set to [contract_value_log, buyer_concentration_hhi] after
leakage-forensics + integration-probe ablations; recorded the drop rationale for
supplier_centrality / cpv_enc / cpv_risk_score / award_year; documented the honest
headline (2020 TEST ROC-AUC 0.8885, PR-AUC 0.8864, F1 0.742 at a VAL-2019-selected
threshold 0.384), the shuffled (0.837) vs forward-chaining temporal (0.719) CV split
and removal of the non-shuffled artifact, the reported RQ1->RQ2 integration null, the
HHI mechanism, and the value-dominance note. No RQ1 thresholds changed. Earlier - 2026-06-10 (clean recompute & lock) - order-dependent caches
rebuilt under the deterministic edge order; final headline values Q_whole 0.7258 /
Q_giant 0.7137, ARI mean 0.5573 / giant 0.5462, conductance_giant mean 0.2808;
two consecutive runs byte-identical after adding the buyer_id sort before the
dependency-SBR correlation (Section 4); gates unchanged and passing; no
thresholds adjusted. Earlier same day - RQ1 freeze pass: converted dependency-SBR correlation
and ARI stability from pass/fail gates to reported diagnostics (Section 9);
corrected the dependency-SBR note to a coverage limitation (168/18,357 buyers);
pinned the ARI seed set and documented the deterministic edge-ordering fix
(Section 4); replaced the absolute centrality threshold with a distribution-derived
5x hub-concentration ratio (Section 8); marked the null model appendix-only with
the Guimerà et al. (2004) caveat. Earlier (2026-06-09): removed silhouette gate;
added null-model significance test, per-community conductance, extended ARI
distribution and top-k; corrected ARI threshold citation from Rousseeuw (1987) to
Lancichinetti & Fortunato (2012); added Section 7 limitations note; corrected
GRAPH_EDGE_LIMIT default; added betweenness approximation documentation; added
bid-count field correction (Section 6).*
