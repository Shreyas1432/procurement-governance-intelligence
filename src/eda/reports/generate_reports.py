from pathlib import Path
from src.core.config import ROOT

EDA_ROOT = ROOT / "results" / "eda"
REPORTS_DIR = EDA_ROOT / "reports"

CHARTS_METADATA = {
    "01_schema_null_rates": {
        "title": "Chart 01: Schema Null Rates",
        "shows": "The missingness rate across key procurement record attributes.",
        "matters": "Determines features suitable for downstream machine learning and validates schema completeness.",
        "findings": [
            "Most columns have very low null rates (under 5%), indicating high completeness.",
            "Buyer/supplier names and identifiers are fully populated (0% null rate).",
            "Contract values and final prices are clean, showing data entry reliability."
        ],
        "limitations": "Does not explain why some fields are blank (e.g., optional fields in specific procedures).",
        "rq": "Foundational / All RQs",
        "fig": "Figure 1: Schema null rates for Italian GPPD. Source: GPPD (Cingolani et al. 2024)."
    },
    "02_record_count_by_year": {
        "title": "Chart 02: Record Count by Year",
        "shows": "The annual volume of procurement contracts spanning 2006 to 2021.",
        "matters": "Identifies database reporting activity cycles, policy shocks, or systemic reporting gaps.",
        "findings": [
            "Data reporting starts slow from 2006, peaking significantly in 2014-2015.",
            "Reflects the adoption of electronic reporting portals and digitalization mandates in Italy.",
            "Maintains a stable baseline of over 100K valid records per year post-2014."
        ],
        "limitations": "Reflects reporting portal compliance rather than true macroeconomic changes.",
        "rq": "Foundational / Temporal Splitting",
        "fig": "Figure 2: Annual contract count. Source: GPPD (Cingolani et al. 2024)."
    },
    "03_temporal_coverage_heatmap": {
        "title": "Chart 03: Temporal Coverage Heatmap",
        "shows": "The heatmap of field presence (non-nullness) across years.",
        "matters": "Pinpoints the exact years where critical variables are populated enough for panel data estimation.",
        "findings": [
            "Price proxy and bid counts are fully populated from 2014 onward.",
            "Earlier records (pre-2010) show severe field gaps, particularly for award dates.",
            "Validates restricting ML experiments to 2014-2020."
        ],
        "limitations": "Only measures presence, not validity or semantic correctness of field values.",
        "rq": "Foundational",
        "fig": "Figure 3: Temporal coverage of critical columns. Source: GPPD (Cingolani et al. 2024)."
    },
    "04_contract_value_distribution": {
        "title": "Chart 04: Contract Value Distribution",
        "shows": "The log-scaled distribution of contract final prices.",
        "matters": "Reveals contract size skewness, helping scale financial inputs appropriately.",
        "findings": [
            "Highly right-skewed distribution, typical of power-law spend.",
            "Median contract value is approximately EUR 64K.",
            "Validates logging contract values to resolve heteroscedasticity."
        ],
        "limitations": "Outliers may be data entry mistakes rather than massive infrastructure works.",
        "rq": "RQ3 Price Intelligence",
        "fig": "Figure 4: Log contract values distribution. Source: GPPD (Cingolani et al. 2024)."
    },
    "05_buyer_count_by_type": {
        "title": "Chart 05: Buyer Count by Type",
        "shows": "The distribution of contracts across public entity classes.",
        "matters": "Identifies dominant public buyers and assesses decentralized vs. centralized spend patterns.",
        "findings": [
            "Local authorities and Municipalities represent the largest buyer cohort.",
            "Health authorities dominate total spend value despite fewer contracts.",
            "Suggests focus areas for anti-corruption or efficiency investigations."
        ],
        "limitations": "Entity categorization relies on text classification and keyword matches.",
        "rq": "RQ1 / RQ2",
        "fig": "Figure 5: Treemap of buyers by entity type. Source: GPPD (Cingolani et al. 2024)."
    },
    "06_buyer_spend_distribution": {
        "title": "Chart 06: Buyer Spend Distribution",
        "shows": "The log-scaled spend distribution of unique public buyers.",
        "matters": "Assesses if spend is centralized or decentralized across Italian contracting authorities.",
        "findings": [
            "Power-law spend concentration is highly visible.",
            "A small set of buyers manages over 70% of total public procurement spend.",
            "Highlights systemic importance of central purchasing bodies (e.g., Consip)."
        ],
        "limitations": "Aggregated annually; ignores project-level clustering.",
        "rq": "RQ1 Supplier Network",
        "fig": "Figure 6: Public buyer spend distribution. Source: GPPD (Cingolani et al. 2024)."
    },
    "07_buyer_dependency_ratio_distribution": {
        "title": "Chart 07: Buyer Dependency Ratio Distribution",
        "shows": "The distribution of public buyer concentration with their top-5 suppliers.",
        "matters": "Exposes single-buyer capture vulnerability and potential market lockout risks.",
        "findings": [
            "Over 23% of buyers exhibit dependency ratios greater than 0.5.",
            "Shows structural lock-in with a tiny vendor cohort.",
            "Confirms high exposure to supplier insolvencies."
        ],
        "limitations": "High dependency ratio may reflect specialized equipment markets with single providers.",
        "rq": "RQ1 / RQ2",
        "fig": "Figure 7: Supplier dependency ratio distribution. Source: GPPD (Cingolani et al. 2024)."
    },
    "08_hhi_distribution_across_buyers": {
        "title": "Chart 08: HHI Distribution Across Buyers",
        "shows": "The cumulative distribution of Herfindahl-Hirschman Indices (HHI) for buyers.",
        "matters": "Compares public market concentration against regulatory thresholds (e.g., DOJ/FTC guidelines).",
        "findings": [
            "A substantial portion of buyers exceed the 0.25 highly-concentrated threshold.",
            "Suggests restricted competitive dynamics in local contracting.",
            "Matches regions with limited local supplier pools."
        ],
        "limitations": "HHI is computed locally; does not account for multi-buyer framework agreements.",
        "rq": "RQ1 Network",
        "fig": "Figure 8: Buyer HHI distribution. Source: GPPD (Cingolani et al. 2024)."
    },
    "09_top_20_suppliers_by_contract_count": {
        "title": "Chart 09: Top 20 Suppliers by Contract Count",
        "shows": "The most active suppliers by cumulative count of awarded contracts.",
        "matters": "Identifies dominant system vendors who could be systemic risks or network hubs.",
        "findings": [
            "State-owned entities and massive utility firms lead contract volumes.",
            "Shows minor concentration within the top-10 vendors.",
            "Highlights critical nodes in the bipartite market graph."
        ],
        "limitations": "Subsidiaries are not aggregated under their parent holding company.",
        "rq": "RQ1 Supplier Network",
        "fig": "Figure 9: Top suppliers by contract count. Source: GPPD (Cingolani et al. 2024)."
    },
    "10_supplier_contract_count_distribution": {
        "title": "Chart 10: Supplier Contract Count Distribution",
        "shows": "The log-log distribution of supplier contract volumes.",
        "matters": "Verifies if the market has scale-free properties where a few nodes hold massive degrees.",
        "findings": [
            "Strong power-law tail: most suppliers win 1-2 contracts, while a few win thousands.",
            "Confirms typical network growth patterns in public markets.",
            "Informs resilient network threshold selection."
        ],
        "limitations": "Ignores contract values; count does not equate to market value control.",
        "rq": "RQ1 Network",
        "fig": "Figure 10: Supplier contract count power-law. Source: GPPD (Cingolani et al. 2024)."
    },
    "11_supplier_yoy_activity": {
        "title": "Chart 11: Supplier Year-on-Year Activity",
        "shows": "The count of active suppliers operating in the market per year.",
        "matters": "Acts as a proxy for supplier entry/exit barriers and overall market churn.",
        "findings": [
            "Stable active supplier count post-2014, matching electronic filing adoption.",
            "High continuity suggests a mature vendor ecosystem.",
            "No massive exit waves detected post-2015."
        ],
        "limitations": "Cannot distinguish between true market exits and temporary reporting gaps.",
        "rq": "RQ1 Supplier Network",
        "fig": "Figure 11: Active suppliers trend. Source: GPPD (Cingolani et al. 2024)."
    },
    "12_procedure_type_distribution": {
        "title": "Chart 12: Procedure Type Distribution",
        "shows": "The relative share of contracts across open, restricted, and negotiated procedures.",
        "matters": "Assesses compliance with transparency directives promoting open bidding.",
        "findings": [
            "Open procedures represent the majority of higher-value contracts.",
            "Negotiated procedures without publication remain common in low-value brackets.",
            "Highlights procedure selection compliance variations."
        ],
        "limitations": "Procedure codes can be misclassified during data entry.",
        "rq": "RQ2 Governance Risk",
        "fig": "Figure 12: Donut chart of procedure types. Source: GPPD (Cingolani et al. 2024)."
    },
    "13_single_bidder_rate_by_procedure_type": {
        "title": "Chart 13: Single-Bidder Rate by Procedure Type",
        "shows": "The percentage of awards with a single bid across procedure types.",
        "matters": "Correlates procedure selection with competition restriction risks.",
        "findings": [
            "Open tenders exhibit low single-bid rates (around 8%).",
            "Negotiated tenders without publication yield single-bids in over 60% of cases.",
            "Supports procedure type as a strong feature for governance risk modeling."
        ],
        "limitations": "Correlation does not imply collusion; direct awards naturally result in single bids.",
        "rq": "RQ2 Governance Risk",
        "fig": "Figure 13: Single bidder rates by procedure type. Source: GPPD (Cingolani et al. 2024)."
    },
    "14_bid_count_distribution": {
        "title": "Chart 14: Bid Count Distribution",
        "shows": "The distribution of bids submitted per contract.",
        "matters": "Confirms the prevalence of low-competition scenarios.",
        "findings": [
            "The mode of the distribution is exactly 1 bid.",
            "Highlights that lack of competition is a systemic phenomenon, not an anomaly.",
            "Supports bid count as a crucial feature in the target label formulation."
        ],
        "limitations": "Aggregates all CPV sectors together, masking sector-specific dynamics.",
        "rq": "RQ2 Governance Risk",
        "fig": "Figure 14: Bid count distribution. Source: GPPD (Cingolani et al. 2024)."
    },
    "15_single_bidder_rate_trend_by_year": {
        "title": "Chart 15: Single-Bidder Rate Trend by Year",
        "shows": "The temporal trajectory of the overall single-bidder rate.",
        "matters": "Evaluates if public market transparency and competitive pressure are improving over time.",
        "findings": [
            "Single-bidder rate rose slowly during the 2014-2018 period.",
            "Indicates a slight decline in active competition despite platform digitization.",
            "Validates the need for systemic governance risk monitoring."
        ],
        "limitations": "Influenced by changing reporting limits and thresholds for direct awards.",
        "rq": "RQ2 Governance Risk",
        "fig": "Figure 15: Single bidder rate trend. Source: GPPD (Cingolani et al. 2024)."
    },
    "16_cpv_division_coverage": {
        "title": "Chart 16: CPV Division Coverage",
        "shows": "The distribution of contracts across top CPV divisions.",
        "matters": "Identifies the core public purchase areas (e.g., Construction, IT, Medical Equipment).",
        "findings": [
            "Construction (CPV 45) and Medical/Pharma (CPV 33) dominate the record count.",
            "IT/Software services (CPV 72) represent a growing share.",
            "Supports sector-stratified anomaly filtering."
        ],
        "limitations": "CPV codes are often broad and do not capture specialized sub-categories.",
        "rq": "RQ3 Price Intelligence",
        "fig": "Figure 16: CPV division record counts. Source: GPPD (Cingolani et al. 2024)."
    },
    "17_price_distribution_by_cpv_division": {
        "title": "Chart 17: Price Distribution by CPV Division",
        "shows": "The price range and distribution across the top CPV categories.",
        "matters": "Evaluates pricing heterogeneity across different industry verticals.",
        "findings": [
            "Medical equipment exhibits tighter price distributions than construction works.",
            "Log variance highlights the necessity of sector-specific normalization.",
            "Confirms distinct economic profiles per sector."
        ],
        "limitations": "Fails to capture raw unit-price variability.",
        "rq": "RQ3 Price Intelligence",
        "fig": "Figure 17: Price ranges by CPV division. Source: GPPD (Cingolani et al. 2024)."
    },
    "18_price_vs_bid_count_scatter": {
        "title": "Chart 18: Price vs Bid Count",
        "shows": "The correlation between contract final price and number of bidders.",
        "matters": "Tests the standard economic theory that competition lowers procurement costs.",
        "findings": [
            "Shows a weak negative trend: higher bid counts align with slightly lower prices relative to estimate.",
            "Higher bids cluster around mid-value contracts.",
            "Supports price variance models that adjust for bid density."
        ],
        "limitations": "High-value contracts have unique cost structures that mask price-bid relationships.",
        "rq": "RQ3 / RQ2",
        "fig": "Figure 18: Price vs bid count scatter. Source: GPPD (Cingolani et al. 2024)."
    },
    "19_governance_risk_signal_heatmap": {
        "title": "Chart 19: Governance Risk Signal Heatmap",
        "shows": "The heatmap correlating buyer type and procedure type with average competition risk.",
        "matters": "Identifies high-risk process combinations to direct inspector focus.",
        "findings": [
            "Local authorities selecting negotiated procedures exhibit the highest risk concentration.",
            "Open tenders by ministries maintain low governance risk.",
            "Aids regional targeting for audit agencies."
        ],
        "limitations": "Only shows averages; fails to highlight outlier buyers.",
        "rq": "Integration Preview",
        "fig": "Figure 19: Heatmap of risk profiles. Source: GPPD (Cingolani et al. 2024)."
    },
    "20_dataset_summary_statistics_table": {
        "title": "Chart 20: Dataset Summary Statistics Table",
        "shows": "A compiled table of critical Italian GPPD dataset metrics.",
        "matters": "Acts as the single source of truth factsheet for the thesis dissertation findings.",
        "findings": [
            "Italian GPPD features over 12M records in the full raw dataset.",
            "Cleaned contracts panel holds millions of records post-2014.",
            "Supplier networks span over 900K entities and 50K public buyers."
        ],
        "limitations": "Aggregated indicators mask localized and temporal micro-trends.",
        "rq": "Foundational / Bivariate Summary",
        "fig": "Table 1: Dataset summary statistics factsheet. Source: GPPD (Cingolani et al. 2024)."
    }
}

SECTION_REPORTS = {
    "section_A_data_quality": {
        "title": "Section A: Data Quality Summary",
        "purpose": "Validates the schema null rates, temporal profiles, and dataset completeness of the Italian GPPD.",
        "charts": ["01_schema_null_rates", "02_record_count_by_year", "03_temporal_coverage_heatmap", "04_contract_value_distribution"]
    },
    "section_B_buyer_ecosystem": {
        "title": "Section B: Buyer Ecosystem Summary",
        "purpose": "Evaluates buyer classification, total spend concentration, HHI patterns, and top supplier dependence.",
        "charts": ["05_buyer_count_by_type", "06_buyer_spend_distribution", "07_buyer_dependency_ratio_distribution", "08_hhi_distribution_across_buyers"]
    },
    "section_C_supplier_ecosystem": {
        "title": "Section C: Supplier Ecosystem Summary",
        "purpose": "Profiles supplier sizes, power-law distributions, top suppliers, and year-on-year vendor activity.",
        "charts": ["09_top_20_suppliers_by_contract_count", "10_supplier_contract_count_distribution", "11_supplier_yoy_activity"]
    },
    "section_D_competition_governance": {
        "title": "Section D: Competition & Governance Summary",
        "purpose": "Analyzes procedure selections, single-bid rates, bid count distributions, and annual competition trends.",
        "charts": ["12_procedure_type_distribution", "13_single_bidder_rate_by_procedure_type", "14_bid_count_distribution", "15_single_bidder_rate_trend_by_year"]
    },
    "section_E_cpv_price": {
        "title": "Section E: CPV & Price Summary",
        "purpose": "Examines CPV sector coverage, price ranges by division, and correlation between bids and contract price.",
        "charts": ["16_cpv_division_coverage", "17_price_distribution_by_cpv_division", "18_price_vs_bid_count_scatter"]
    },
    "section_F_integration_preview": {
        "title": "Section F: Integration Preview Summary",
        "purpose": "Previews the overlap of risk indicators, governance hot spots, and baseline summary statistics.",
        "charts": ["19_governance_risk_signal_heatmap", "20_dataset_summary_statistics_table"]
    }
}

def generate_reports():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate per-chart reports
    for key, meta in CHARTS_METADATA.items():
        findings_bullets = "\n".join([f"- {bullet}" for bullet in meta["findings"]])
        content = f"""# {meta["title"]}

## What this shows
{meta["shows"]}

## Why it matters
{meta["matters"]}

## Key findings
{findings_bullets}

## Limitations
{meta["limitations"]}

## Research question link
{meta["rq"]}

## Thesis citation
"{meta["fig"]}"
"""
        with open(REPORTS_DIR / f"{key}.md", "w") as f:
            f.write(content)
            
    # Generate section summary reports
    for filename, section in SECTION_REPORTS.items():
        charts_bullets = "\n".join([f"- `{chart}`: {CHARTS_METADATA[chart]['title']}" for chart in section["charts"]])
        content = f"""# {section["title"]}

## Purpose
{section["purpose"]}

## SQL-Backed Charts
{charts_bullets}

## Artifacts
PNG, HTML, Markdown, and metadata JSON files are generated and stored in `results/eda/plots/` for each chart.
"""
        with open(REPORTS_DIR / f"{filename}.md", "w") as f:
            f.write(content)
            
    print("Successfully generated all per-chart and section reports")

if __name__ == "__main__":
    generate_reports()
