from pathlib import Path
from src.core.config import ROOT

EDA_ROOT = ROOT / "results" / "eda"
PLOTS_DIR = EDA_ROOT / "plots"
REPORTS_DIR = EDA_ROOT / "reports"

CHART_NAMES = [
    "01_schema_null_rates",
    "02_record_count_by_year",
    "03_temporal_coverage_heatmap",
    "04_contract_value_distribution",
    "05_buyer_count_by_type",
    "06_buyer_spend_distribution",
    "07_buyer_dependency_ratio_distribution",
    "08_hhi_distribution_across_buyers",
    "09_top_20_suppliers_by_contract_count",
    "10_supplier_contract_count_distribution",
    "11_supplier_yoy_activity",
    "12_procedure_type_distribution",
    "13_single_bidder_rate_by_procedure_type",
    "14_bid_count_distribution",
    "15_single_bidder_rate_trend_by_year",
    "16_cpv_division_coverage",
    "17_price_distribution_by_cpv_division",
    "18_price_vs_bid_count_scatter",
    "19_governance_risk_signal_heatmap",
    "20_dataset_summary_statistics_table"
]

SECTION_REPORTS = [
    "section_A_data_quality",
    "section_B_buyer_ecosystem",
    "section_C_supplier_ecosystem",
    "section_D_competition_governance",
    "section_E_cpv_price",
    "section_F_integration_preview"
]

def test_chart_artifacts_exist():
    for name in CHART_NAMES:
        assert (PLOTS_DIR / f"{name}.html").exists(), f"HTML missing for {name}"
        assert (PLOTS_DIR / f"{name}.png").exists(), f"PNG missing for {name}"
        assert (PLOTS_DIR / f"{name}.json").exists(), f"JSON missing for {name}"

def test_report_artifacts_exist():
    for name in CHART_NAMES:
        assert (REPORTS_DIR / f"{name}.md").exists(), f"Markdown report missing for {name}"
    for section in SECTION_REPORTS:
        assert (REPORTS_DIR / f"{section}.md").exists(), f"Section report missing for {section}"
