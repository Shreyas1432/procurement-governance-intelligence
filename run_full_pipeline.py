"""Full pipeline runner: executes all phases in sequence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Verify all imports succeed before running anything (catches import crashes early)
import importlib
for mod, fn in [
    ("src.rq1_supplier_network.rq1_network", "run_rq1"),
    ("src.rq2_governance_risk.rq2_governance", "run_rq2"),
    ("src.rq3_price_intelligence.rq3_pricing", "run_rq3"),
    ("src.integration.integration", "run_integration"),
]:
    assert hasattr(importlib.import_module(mod), fn), f"Missing: {mod}.{fn}"

from src.core.feature_engineering import run_feature_engineering
from src.rq1_supplier_network.rq1_network import run_rq1
from src.rq2_governance_risk.rq2_governance import run_rq2
from src.rq3_price_intelligence.rq3_pricing import run_rq3
from src.integration.integration import run_integration
from src.common.utils import get_logger

log = get_logger(__name__)

PHASES = [
    ("Feature Engineering", run_feature_engineering),
    ("RQ1: Supplier Network",    run_rq1),
    ("RQ2: Governance Risk",     run_rq2),
    ("RQ3: Price Intelligence",  run_rq3),
    ("Integration",              run_integration),
]

if __name__ == "__main__":
    for name, fn in PHASES:
        try:
            log.info(f"Starting: {name}")
            fn()
            log.info(f"Complete: {name}")
        except Exception as e:
            log.error(f"FAILED: {name}: {type(e).__name__}: {e}")
            log.error("Continuing with next phase...")

    # Data provenance metadata for the dashboard (CHANGE 8)
    import json
    from datetime import datetime
    from src.core.config import DATA_RESULTS, TRAIN_YEARS, TEST_YEARS

    def _read(name):
        try:
            return json.loads((DATA_RESULTS / name).read_text())
        except Exception:
            return {}

    rq2m, rq3m = _read("rq2_success_metrics.json"), _read("rq3_success_metrics.json")
    total_records = (rq2m.get("train_n", 0) + rq2m.get("test_n", 0)) or rq3m.get("total_records")
    metadata = {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "total_records": total_records,
        "train_cutoff": max(TRAIN_YEARS),
        "test_start": min(TEST_YEARS),
        "rq2_auc_xgb": rq2m.get("xgb_auc"),
        "rq3_r2_gbr": rq3m.get("gb_r2"),
        "anomaly_consensus_rate": rq3m.get("consensus_anomaly_rate"),
    }
    with open(DATA_RESULTS / "pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Wrote pipeline_metadata.json: {metadata}")
    log.info("Pipeline finished. Run 'make dashboard' to launch.")
