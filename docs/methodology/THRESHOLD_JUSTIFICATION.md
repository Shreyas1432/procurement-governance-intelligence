# Threshold & Weight Justification (Literature Grounding)

**Purpose.** Grounds every fixed threshold and weight in the Procurement
Governance Intelligence (PGI) pipeline in published literature or established
statistical convention. Distinguishes (a) **domain/risk thresholds**, groundable
in procurement-corruption research, from (b) **model-performance thresholds**,
groundable in ML/diagnostic convention. Per project decision (2026-05-29), current
values are **retained**; this document records their basis and flags deviations.
**No threshold was changed to make a result pass.**

> Scope note: The orchestration brief referenced "HHI 0.50", "single-bidder 30%",
> and "risk weights 25/25/20/15/15". **None of these values exist in this codebase.**
> The thresholds documented below are the ones actually present in `src/config.py`,
> the `risk_label` rule (`src/feature_engineering.py`), and
> `sql/12_feature_competition_risk.sql`.

---

## A. Domain / risk thresholds

### A.1 Single-bidding as the primary competition-risk signal
- **In code:** `sql/12_feature_competition_risk.sql`: `bid_count <= 1 -> competition_risk = 1.0`; `= 2 -> 0.65`; `3-5 -> 0.30`; else `0.05`. The `risk_label` rule also treats `bid_count <= 1` as a core flag.
- **Literature basis:** Single bidding in otherwise competitive markets is the **single most validated objective proxy for procurement corruption** (Fazekas & Kocsis, 2020, *British Journal of Political Science*; DIGIWHIST / Government Transparency Institute). It directly operationalises "unjustified restriction of access to favour a selected bidder."
- **Assessment:** Well grounded. The monotone-decreasing risk-by-bid-count schedule is consistent with the literature's treatment of competition intensity.

### A.2 Buyer concentration (HHI)
- **In code:** `buyer_concentration_hhi` is used as a continuous model **feature** (not a hard cutoff) in `RQ2_MODEL_FEATURES`. It is a top RF/XGB predictor.
- **Convention basis:** US DOJ/FTC Horizontal Merger Guidelines HHI bands: **< 0.15 unconcentrated, 0.15-0.25 moderately concentrated, > 0.25 highly concentrated** (decimal scale; equivalently 1,500 / 2,500 on the 0-10,000 scale).
- **Assessment:** Using HHI as a continuous feature is defensible (no arbitrary cut imposed). **If** a categorical "high concentration" cut is ever reported, anchor it at **0.25** (DOJ "highly concentrated"), not 0.50.

### A.3 Buyer-dependency flags in `risk_label`
- **In code:** `risk_label` (`src/feature_engineering.py`) uses `buyer_dependency_ratio > 0.5` (medium flag) and `> 0.7` (high flag).
- **Basis:** Supplier/buyer dependency above ~50% is a common red-flag heuristic in procurement-integrity toolkits (EU/OECD procurement risk indicators). Heuristic, not a single canonical cut.
- **Assessment:** Defensible as heuristic; label these explicitly as expert-rule thresholds in the methodology, not empirically optimised values. **Note:** these columns are deliberately **excluded** from `RQ2_MODEL_FEATURES` to avoid target leakage (they define the label).

### A.4 Procedure-type risk weights
- **In code:** `sql/12`: negotiated-without-publication / outright-award -> 1.0; negotiated -> 0.75; restricted / competitive-dialogue -> 0.35; open -> 0.05.
- **Literature basis:** Restricted and (especially) negotiated-without-publication procedures are standard red flags in the Fazekas & Kocsis composite (alongside no-call-for-tender, short advertisement period, hard-to-quantify criteria). Open procedure = lowest risk.
- **Assessment:** Ordinal direction matches the literature. The specific magnitudes are author-assigned; document as such.

### A.5 Unified-score component weights
- **In code:** `src/integration.py`: `unified = 0.35·RQ1(dependency) + 0.40·RQ2(governance) + 0.25·RQ3(price-anomaly)`.
- **Literature basis:** Fazekas & Kocsis construct their corruption-risk composite with **equally weighted** red flags. The 0.35/0.40/0.25 split is **author-assigned, not literature-derived**.
- **Assessment:** **Deviation from literature convention (equal weighting).** Retained per project decision. Recommended thesis action: either (i) justify the unequal weighting explicitly (e.g. RQ2 governance signal weighted highest because it is supervised and validated), **or** (ii) report an equal-weight (0.33/0.33/0.33) variant as a robustness check. A sensitivity analysis over these weights is the natural follow-up.

---

## B. Model-performance thresholds (`src/config.py`)

Success criteria, not domain constants; grounded in ML/diagnostic convention.

| Threshold | Value | Basis | Assessment |
|---|---|---|---|
| `RQ2_RF_F1_THRESHOLD` | 0.70 | Common "acceptable" F1 floor for imbalanced classification | Reasonable |
| `RQ2_XGB_AUC_THRESHOLD` | 0.75 | Between Hosmer–Lemeshow "acceptable" (0.70) and "excellent" (0.80) discrimination | Reasonable |
| `RQ2_LR_AUC_THRESHOLD` | 0.65 | Lower bar for a linear baseline | Reasonable (baseline) |
| `RQ3_GB_R2_THRESHOLD` | 0.30 | Modest explanatory-power floor for price modelling | Conservative |
| `RQ3_RMSE_REDUCTION_THRESHOLD` | 0.15 | ≥15% improvement over naive baseline | Reasonable |
| `RQ1_MODULARITY_THRESHOLD` | 0.30 | Newman: modularity > 0.30 indicates meaningful community structure | Standard |
| `RQ3_ANOMALY_RATE` band | 4-8% | Fazekas finds corruption-risk flags in ~15-25% of *high-value* contracts; a 4-8% consensus-anomaly rate across *all* contracts is a conservative, plausible band | See note |

### Honest-validation caveats (post-remediation, on the 2014-2020 stratified panel)
- **`RQ1_DEPENDENCY_CORRELATION_THRESHOLD = 0.40`**: **not met** (actual −0.036). Genuine null result: in this data, buyer dependency and single-bidding are essentially uncorrelated. Report as a finding, not a feature-engineering failure.
- **`RQ2_ENRICHMENT_MULTIPLIER = 3.0`**: **not met** (ratio 0.0). The enrichment metric is partly **circular** (`contract_amendments` helps define `risk_label`), so it is a weak outcome proxy by construction; documented inline in `src/rq2_governance.py`. Low-risk amendment rate is 0.0, forcing the ratio to 0.
- **`RQ3_LOF_GB_OVERLAP_THRESHOLD = 0.50`**: **not met** (0.036). Low overlap between density-based (LOF) and residual-based (GB) detectors is expected, they capture different anomaly types; treat as complementary rather than requiring agreement. Consider dropping this criterion or reframing as "diversity is desirable."
- **`RQ3_ANOMALY_RATE` 4-8%**: **not met** (0.22%). The consensus rule (≥2 of 3 detectors) is strict; the consensus rate is far below the band. Either relax consensus to "≥1 of 3" (raises rate) or re-justify the band for a strict-consensus definition.

### RQ2 honest tuning note
RF/XGB hyperparameters were tuned on an honest protocol (train 2014-2017, validate 2018, **test 2019-2020 untouched during tuning**). XGB uses a runtime `scale_pos_weight` set from the train class ratio (~87/13). Final temporal-holdout metrics: RF F1 0.725, XGB AUC 0.753, LR AUC 0.693, all clear their floors. The modest XGB AUC reflects a **real temporal distribution shift**: the positive (risk) rate drops to 0.70 in 2019 vs ~0.90 in the training years. This is an honest ceiling, not an implementation weakness.

---

## References
- Fazekas, M. & Kocsis, G. (2020). *Uncovering High-Level Corruption: Cross-National Objective Corruption Risk Indicators Using Public Procurement Data.* British Journal of Political Science, 50(1), 155–164.
- Fazekas, M. & Kocsis, G. (2015). GTI Working Paper WP2015:2, Government Transparency Institute.
- U.S. DOJ/FTC Horizontal Merger Guidelines: Herfindahl–Hirschman Index thresholds (justice.gov/atr/herfindahl-hirschman-index).
- Quality of Government Institute: Corruption Risk Indicators dataset (datafinder.qog.gu.se/dataset/cri).
- Hosmer, D.W. & Lemeshow, S. (2000). *Applied Logistic Regression*: AUC discrimination bands.
- Newman, M.E.J. (2006). *Modularity and community structure in networks.* PNAS.

*Generated 2026-05-29. All in-code values verified against `src/config.py`, `src/feature_engineering.py`, `src/integration.py`, `sql/12_feature_competition_risk.sql`.*
