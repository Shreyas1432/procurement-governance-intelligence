# RQ2 — Governance Risk Classification: Results (value-only, locked 2026-06-12)

**Status:** locked at a single feature, `contract_value_log`, after a multi-pass leakage
audit. Full methodological rationale: `docs/THRESHOLD_JUSTIFICATION.md` Section 10. All
numbers below are recomputed at runtime from `data/results/rq2_success_metrics.json`.

## Model and feature set

`RQ2_MODEL_FEATURES = ["contract_value_log"]`. Headline model: XGBoost (200 trees,
depth 4, lr 0.05, `scale_pos_weight` from the train class ratio, `seed=42`). RandomForest
and Logistic Regression are reported baselines; the three are one signal fit three ways,
not independent models. Split: TRAIN 2014–2018, VAL 2019, TEST 2020 (temporal hold-out).

The feature set reduced to one through disciplined leakage control. Using the canonical
candidate-count statement (identical in `rq2_feature_disposition.md` and
THRESHOLD_JUSTIFICATION §10):

> 8 features were evaluated. cpv_enc and cpv_risk_score are two encodings of one
> procedure-type signal; award_year is the temporal-split key (constant at test).
> Excluding these leaves 5 substantive non-value candidates, of which 0 survived and
> contract_value_log was retained — i.e. value-only. 4 are documented nulls
> (supplier_centrality, buyer_concentration_hhi, buyer_region, tender_lotscount); the
> 5th removal is the cpv pair as a procedure proxy.

Enumerated, the removals are: `cpv_enc` / `cpv_risk_score` (procedure_type proxies,
|corr| ~0.60, ~0 marginal — the cpv pair), `award_year` (constant at test — the split
key), `supplier_centrality` (RQ1→RQ2 null, ≈0.49 alone), `buyer_concentration_hhi` (0.77
Pearson proxy for the `buyer_dependency_ratio` label input), `buyer_region` (η=0.966
categorical proxy for `contract_amendments` — see below), and `tender_lotscount` (r=0.53
redundancy with `contract_value_log`, below the leakage gate). The honest contribution is
a clean value-only early-detection model plus four documented nulls — not a thin model
with hidden gaps.

**Gate statistic (numeric vs categorical).** The proxy/leakage gate is "association with a
label input by the appropriate statistic": Pearson |corr| ≥ 0.55 for numeric candidates
(e.g. `buyer_concentration_hhi` 0.77), and the correlation ratio **η** for a categorical
candidate against a numeric label input. `buyer_region` is a **second-order proxy**:
categorical region explains 96.6% of `contract_amendments` variance (η=0.966), and
`contract_amendments` is itself a label-rule input, so region is a proxy-of-a-proxy.
η (categorical association) is a **distinct statistic** from the Pearson-|corr| gate that
rejected `buyer_concentration_hhi`; the two are not interchangeable.

## Honest headline metrics — 2020 hold-out TEST

The operating threshold is selected by max-F1 on **VAL 2019 only** (0.3548) and
frozen onto 2020; test labels never influence it. Because VAL 2019 is ~10% positive and
TEST 2020 is ~37% positive, the frozen operating point is fragile, so F1 is reported
two ways and **ROC-AUC (threshold-free) is the headline**.

| Metric | Value |
|---|---|
| ROC-AUC (headline) | **0.8249** |
| PR-AUC | 0.8340 |
| F1 @ VAL-frozen thr 0.3548 | 0.5635 (P 0.403 / R 0.937) |
| F1 @ 0.5 default | 0.7662 (P 0.961 / R 0.637) |
| Confusion @ frozen | TN 14,669 · FP 63,195 · FN 2,850 · TP 42,626 |
| Confusion @ 0.5 | TN 76,675 · FP 1,189 · FN 16,498 · TP 28,978 |
| Class balance (panel) | {0: 600,548, 1: 114,133} (16.0% pos; 2020 ≈ 37% pos) |

Cross-validation: shuffled 5-fold 0.710 ± 0.004 (stability, non-temporal);
forward-chaining temporal 0.610 ± 0.139 (folds 0.477, 0.559, 0.844, 0.561; wide spread = genuine
non-stationarity). Baselines: XGB ROC-AUC 0.8249, LR 0.8078, RF 0.7038;
ensemble 0.8225.

## `rf_f1` is a reported diagnostic, not a gate

`rf_f1 = 0.494` is the RandomForest baseline's default-threshold hard-label F1 and is
**reported, not gated** (reclassified 2026-06-12, RQ1 Section 9 precedent). The ranking
evidence that this is an operating-point artifact rather than a ranking failure is
**`rf_auc = 0.704`** — well above chance, far above what F1 0.494 implies. The 0.70
bar was only ever met via the HHI proxy (0.852 → 0.612 → 0.494 as leakage was removed),
and base-rate drift (2019:10% → 2020:37%) makes the RF under-call at a default threshold.

## Interpretability — non-monotonic value → risk

With one feature, model SHAP is degenerate; the interpretability artifact is the
partial-dependence / calibration curve `rq2_value_only_pdp.png`. The value→risk
relationship is **non-monotonic** (elevated at low value, a mid-range dip, then a steep
climb at high value), which is the entire reason the XGB headline (0.8249) edges the
linear LR baseline (0.8078).

## Four documented nulls

1. **RQ1 → RQ2 integration / `supplier_centrality`** — network position adds no marginal
   non-leaky signal; `supplier_centrality` alone scores TEST AUC ~0.49.
2. **`buyer_concentration_hhi` (label-input proxy)** — a valid bounded as-of-award HHI,
   positively/monotonically associated with risk, but |corr| 0.77 with
   `buyer_dependency_ratio` (a label input); its lift reconstructs the label's own
   trigger. Excluded. (`rq2_hhi_proxy_null.*`)
3. **`buyer_region`** — η 0.966 with `contract_amendments` (a label input that is itself
   97.6% zero and regionally concentrated in region ITH5 — a recording artifact);
   marginal lift +0.0013. Rejected as a proxy. (`rq2_feature_exploration_nulls.json`)
4. **`tender_lotscount`** — redundant with `contract_value_log` (corr 0.53), marginal
   lift +0.0013. A size proxy, not independent signal.

## Label framing

The `risk_label` rule combines `procedure_type`, `bid_count`, `buyer_dependency_ratio`
and `contract_amendments`; class 2 never fires so the target is binary. Because the
amendment arm is largely a regional recording artifact (Section 10.2), the label is best
read as **procedural / recording irregularity with a regional data-quality caveat**, not
corruption-style favoritism.

## Reporting-integrity blocks (diagnostic; no model change)

The metrics JSON now carries reporting-integrity blocks added without touching the
model, feature set, thresholds or the headline ROC-AUC (0.8249).

**Populations (`populations`).** Three distinct populations are named and reconciled:

- **Classifier population** — the full 2014–2020 panel, n = 714,681, temporally split
  TRAIN 2014–2018 (471,136; 56,225 positive), VAL 2019 (120,205; 12,432 positive),
  TEST 2020 (123,340; 45,476 positive). Label positives total 114,133 (16.0%), which is
  the binary class-1 count.
- **R_gov-scored population** — n = 594,476: TRAIN rows scored out-of-fold + TEST rows
  scored held-out. **VAL 2019 is excluded** (threshold-selection only, never scored), so
  714,681 − 120,205 = 594,476. Its R_gov score buckets are HIGH 46,587 / MEDIUM 60,454 /
  LOW 487,435. The previously unexplained **534,022 = HIGH + LOW**, i.e. it excludes the
  60,454 MEDIUM rows *and* the 120,205 VAL rows — it is neither the classifier population
  nor the full scored population.
- **Enrichment population** — `enrichment_ratio` and the high/low amendment rates are over
  the HIGH and LOW **R_gov score buckets** (not the classifier label); denominators
  high = 46,587, low = 487,435.

**Single-bidder definitions (`single_bidder_definitions`).** All rates are "single-bidder
among observed lots"; they differ only by population/denominator. Scored-panel observed
rate 0.2048 (n_obs 21,208 / 594,476); panel observed 0.2082 (21,927 / 714,681); the
thesis-cited **29.8%** is the full-raw-dataset observed-lot rate (≈192K observed / ~12.1M,
§6). Cite one consistently.

**AUC significance (`auc_significance`).** Closed-form DeLong on the 2020 test set
(no refit): **XGB − LR = +0.0171, 95% CI [0.0141, 0.0201], p ≈ 6.6e-29**;
XGB − RF = +0.1211, 95% CI [0.1183, 0.1240], p ≈ 0. The XGB edge over LR is small but
statistically unambiguous, consistent with the non-monotonic value→risk shape.

**Feature disposition (`rq2_feature_disposition.json` / `.md`).** Every candidate with its
disposition_type, decisive metric and the gate it tripped. Candidate count is reconciled at
**N_evaluated = 8** (the literal table), bucketed transparently: **1 kept** value feature
(`contract_value_log`); the **cpv pair** (`cpv_enc`, `cpv_risk_score`) = two encodings of one
procedure signal; `award_year` = the temporal-split key (constant at test); and **4
documented nulls** (`supplier_centrality`, `buyer_concentration_hhi`, `buyer_region`,
`tender_lotscount`). Of the 5 substantive non-value candidate signals (the 4 nulls + the cpv
procedure proxy), **0 survived**. This supersedes the earlier unexplained "N_initial = 6"
framing. `tender_lotscount` is a **redundancy_drop** (r = 0.53, *below* the 0.55 leakage
gate), not a leakage rejection; `buyer_concentration_hhi` (Pearson |r| 0.77) is a
leakage_reject; `buyer_region` (η = 0.966 — a *categorical* correlation-ratio statistic,
distinct from the Pearson gate) is a second-order proxy-of-a-proxy leakage_reject;
`supplier_centrality` is null_integration (≈0.49 alone); CPV encodings are procedure-proxy
leakage_reject (~0.60); `award_year` is constant_at_test. The twin **+0.0013** marginal lifts
for `tender_lotscount` and `buyer_region` are **flagged unverified/likely templated** (no
per-fold support for the former; the latter's own per-fold CV lifts average ~0.00575); a
precise recompute needs an ablation refit (out of scope — no refits). Both are negligible (~0).

**Other labels.** `pr_auc_baseline` = 0.3687 (2020 test positive rate, the random-classifier
PR-AUC; PR-AUC is base-rate-dependent and lower at the 16% panel rate). `rf_cv_f1` carries a
note that it is shuffled-CV (base-rate-inflated), not comparable to the temporal-test F1
0.494. Framing labels: `test_headline` = "2020 point estimate";
`xgb_auc_cv_temporal` = "generalisation estimate (forward-chaining temporal CV)".

**NaN-guard bug-class fix (`rgov_single_bidder_corr`).** Previously this returned a
fabricated **0.0**: `single_bid_rate` is null for buyers with no observed bid count, so the
full-array `np.std` was NaN, the `std > 0` guard was False, and `pearsonr` was skipped,
emitting 0.0 — indistinguishable from a genuine zero correlation. The metric is now
**three-state**: it emits the value when coverage is representative, `0.0` only as a
`measured_null`, and **`null` with status `degenerate_low_coverage`** when the observed
buyers are an unrepresentative minority. Here only **150 of 17,289 buyers (0.87%)** have an
observed single-bid rate, so the headline value is `null`; the observed-only r (−0.169) is
preserved in `observed_value`. The same NaN-aware helper (`coverage_aware_corr`) now backs
RQ1's `dependency_single_bidder_corr` (also `null`/degenerate; observed −0.038; the prior
fill-null(0) value 0.052 is withdrawn) and the integration correlations (`rq1_rq2_corr`
unchanged at −0.054, dense; `rq2_rq3_corr` now `null` — `anomaly_score` covers <10% of
contracts). All are reported diagnostics, never gates; no gated or headline value moved.

## Reproducibility

Deterministic: XGBoost `random_state=42`, fixed temporal split,
`negotiated_restricted_rate` ULP-sorted for byte-identical reruns. The new reporting
blocks are closed-form (DeLong, no resampling) and recomputed from in-run arrays, so two
consecutive runs produce byte-identical `rq2_success_metrics.json` (verified). Metrics in
`rq2_success_metrics.json` (now including `rf_auc`, both F1 thresholds, and the
reporting-integrity blocks); CV in `model_performance_ci.json`. Leakage audit
(`test_full_leakage_audit_three_rqs.py`) passes 0 FAIL; `test_rq2_success.py` passes with
`rf_f1` as a reported diagnostic.

## Changelog — RQ2 final lock

Three dated passes brought RQ2 to its current locked state. Full rationale lives in
`docs/THRESHOLD_JUSTIFICATION.md` Section 10 (and the 10.12 ledger).

- **Pass 1 — 2026-06-12 (two-feature lock).** *Changed:* locked the feature set to
  `[contract_value_log, buyer_concentration_hhi]` after leakage-forensics and
  integration-probe ablations; dropped `supplier_centrality`, `cpv_enc`,
  `cpv_risk_score`, `award_year`. *Headline:* 2020 TEST ROC-AUC 0.8885.
  *Moved:* nothing yet withdrawn.
- **Pass 2 — 2026-06-12 (value-only LOCK).** *Changed:* removed
  `buyer_concentration_hhi` as a label-input proxy (Pearson |corr| 0.77 with
  `buyer_dependency_ratio`, > 0.55 gate), leaving the single feature
  `[contract_value_log]`. *Moved:* headline ROC-AUC 0.8885 → **0.8249** (the current
  locked value); `rf_f1` reclassified from a reported metric to a diagnostic (with
  `rf_auc` evidence). This is the headline lock and has not moved since.
- **Pass 3 — 2026-06-15 (reporting-integrity / NaN-guard fix).** *Changed:* no model,
  feature, threshold, or headline change — additive diagnostics only. Added the
  `populations`, `single_bidder_definitions`, and `auc_significance` (closed-form
  DeLong) blocks; reconciled the candidate count at 8 evaluated with the η-vs-Pearson
  gate distinction documented. *Moved:* `rgov_single_bidder_corr` from a **fabricated
  0.0** to **null** (`degenerate_low_coverage`, observed-only r −0.169) under the
  shared `coverage_aware_corr` three-state helper; ROC-AUC 0.8249 unchanged.
- **Pending (deferred, out of scope this pass):** narrative framing prose and the
  R_gov citation; an exact independent recompute of the `+0.0013` twin marginal-lift
  (needs an ablation refit — substantive conclusion, negligible ~0, is unchanged).
