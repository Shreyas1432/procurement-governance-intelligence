from pathlib import Path
import os
import sys
import inspect
import dataclasses

# Monkey-patch for CPython 3.14.1 dataclass slots bug: slotting a class with
# init=False accesses __annotate__ on object's wrapper_descriptor and crashes.
# Only required on Python 3.14.x where the bug exists (and where networkx,
# which uses slotted dataclasses, fails to import without it).
if sys.version_info >= (3, 14):
    try:
        _source = inspect.getsource(dataclasses._add_slots)
        if "init_annotate = newcls.__init__.__annotate__" in _source:
            _source = _source.replace(
                "init_annotate = newcls.__init__.__annotate__",
                "init_annotate = getattr(newcls.__init__, '__annotate__', None)"
            )
            exec(_source, dataclasses.__dict__)
    except Exception:
        pass  # Patch failed silently — acceptable on 3.14


# Columns that DEFINE or are derived from the risk label. These must NEVER be
# used as model inputs — any model that sees them predicts the label from the
# label (target leakage). (leakage remediation 2026-06-03)
LABEL_DEFINING_COLS = [
    "risk_label", "R_gov", "R_proc", "R_comp", "R_dep",
    "R_corr", "R_sup", "single_bidder_flag", "is_single_bidder",
    "governance_score", "ensemble_risk_prob",
]

# Columns that are INPUTS to the deterministic label rule.  risk_label is
# computed as f(procedure_type, bid_count, buyer_dependency_ratio,
# contract_amendments) — feeding any of these to the classifier makes the
# target a trivial function of its inputs (AUC → 1.0).  Must also be excluded
# from the model feature matrix.  (leakage remediation 2026-06-04)
LABEL_RULE_INPUTS = [
    "bid_count", "bid_count_clipped",
    "buyer_dependency_ratio",
    "procedure_type_encoded", "proc_enc",
    "procedure_type",
    "contract_amendments",
]

ROOT = Path(__file__).parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CURATED = ROOT / "data" / "curated"
DATA_FEATURES = ROOT / "data" / "features"
DATA_MODELS = ROOT / "data" / "models"
DATA_RESULTS = ROOT / "data" / "results"
SQL_DIR = ROOT / "src" / "core" / "sql"
DOCS_DIR = ROOT / "docs"

PARQUET_FILE = Path(os.environ.get("PGI_PARQUET_PATH", DATA_RAW / "IT_DIB_2023.parquet"))
ROW_LIMIT = int(os.environ.get("PGI_ROW_LIMIT", "200000"))
# GRAPH_EDGE_LIMIT: maximum bipartite edges to load into NetworkX.
# 0 = unlimited (recommended — the 60,000 default was an EDA-era testing shortcut
# that silently discarded ~80% of training-period buyer-supplier pairs, inflating
# modularity and reducing ARI stability).  Set via PGI_GRAPH_EDGE_LIMIT env var
# if runtime is a concern on extremely large datasets.
GRAPH_EDGE_LIMIT = int(os.environ.get("PGI_GRAPH_EDGE_LIMIT", "0"))

# --- Year-stratified sampling -------------------------------------------------
STRATIFY_BY_YEAR = os.environ.get("PGI_STRATIFY_BY_YEAR", "1") == "1"
STRATA_YEAR_MIN = int(os.environ.get("PGI_STRATA_YEAR_MIN", "2014"))
STRATA_YEAR_MAX = int(os.environ.get("PGI_STRATA_YEAR_MAX", "2020"))
STRATA_PER_YEAR_CAP = int(os.environ.get("PGI_STRATA_PER_YEAR_CAP", "100000"))

# Temporal split windows, aligned to the stratified panel (2014-2020).
TRAIN_YEARS = range(2014, 2019)   # 2014-2018
VAL_YEARS = range(2019, 2020)     # 2019 only (for early stopping / validation)
TEST_YEARS = range(2020, 2021)    # 2020 only (final holdout — no overlap with VAL)

RQ1_CENTRALITY_TOP10_THRESHOLD = 0.3
RQ1_MODULARITY_THRESHOLD = 0.3
RQ1_RESILIENCE_STD_THRESHOLD = 0.15
RQ1_DEPENDENCY_CORRELATION_THRESHOLD = 0.4
# ARI stability threshold (Lancichinetti & Fortunato 2012 consensus clustering guidance;
# self-imposed project target: mean pairwise ARI > 0.70 across seeds indicates
# reproducible partitions).
RQ1_ARI_STABILITY_THRESHOLD = 0.70

# Random seed for reproducibility across ARI stability runs and null-model rewiring
RQ1_RANDOM_SEED = 42
# Max worker threads/processes for any parallel step in RQ1; keeps thermal load bounded.
# Applies to any sklearn or graph step that accepts n_jobs/workers.
PGI_MAX_WORKERS = int(os.environ.get("PGI_MAX_WORKERS", "2"))

# Modularity null-model significance test: number of degree-preserving rewired graphs.
# Set via PGI_NULL_MODEL_REWIRES env var.  Results are cached to parquet; delete
# data/results/rq1_null_modularity.parquet to force recompute.
RQ1_NULL_MODEL_REWIRES = int(os.environ.get("PGI_NULL_MODEL_REWIRES", "100"))
# p-value threshold for declaring modularity statistically significant vs null.
RQ1_NULL_MODEL_SIGNIFICANCE_ALPHA = 0.01

# ARI stability seed set: pinned literal values (originally derived once from
# numpy default_rng(42); recorded here so they cannot drift). The 5 seeds yield
# C(5,2)=10 pairwise ARI scores. Do not change without updating
# docs/THRESHOLD_JUSTIFICATION.md Section 4.
RQ1_ARI_SEEDS = (42, 892, 7739, 6545, 4388)

# Top-k community ARI: restrict stability check to the k largest communities.
# Instability concentrates in small peripheral communities; core should be higher.
# Set via PGI_ARI_TOPK env var.
RQ1_ARI_TOPK_COMMUNITIES = int(os.environ.get("PGI_ARI_TOPK", "20"))

# Conductance cutoff: communities below this value are considered well-separated.
# Set via PGI_CONDUCTANCE_CUTOFF env var.
RQ1_CONDUCTANCE_CUTOFF = float(os.environ.get("PGI_CONDUCTANCE_CUTOFF", "0.3"))

# Success criteria thresholds
RQ2_RF_F1_THRESHOLD = 0.70
RQ2_XGB_AUC_THRESHOLD = 0.75
RQ2_LR_AUC_THRESHOLD = 0.65
RQ2_ENRICHMENT_MULTIPLIER = 3.0
RQ2_ENRICHMENT_PVALUE = 0.001

RQ3_ANOMALY_SD_THRESHOLD = 2.0
RQ3_GB_R2_THRESHOLD = 0.30
RQ3_RMSE_REDUCTION_THRESHOLD = 0.15
RQ3_LOF_GB_OVERLAP_THRESHOLD = 0.5
RQ3_ANOMALY_RATE_MIN = 0.04
RQ3_ANOMALY_RATE_MAX = 0.08
RQ3_CONSENSUS_ENRICHMENT_THRESHOLD = 0.4

# Model parameters
RF_PARAMS = dict(n_estimators=300, max_depth=20, min_samples_leaf=5, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
LR_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced")
GB_REG_PARAMS = dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
LOF_PARAMS = dict(n_neighbors=20, contamination=0.05)
ISO_PARAMS = dict(n_estimators=100, contamination=0.05, random_state=42)

# Feature set fed to ML training.  ONLY pre-award observables that are
# genuinely independent of the risk_label rule.  Excludes:
#   - LABEL_DEFINING_COLS (outputs / derivatives of the label)
#   - LABEL_RULE_INPUTS  (inputs to the deterministic label rule)
# (leakage remediation 2026-06-04)
FEATURES = [
    "buyer_concentration_hhi",
    "supplier_centrality",
    "contract_value_log",
    "cpv_enc",
    "cpv_risk_score",
    "award_year",
]

# All columns excluded from the model — both label outputs and label-rule inputs.
RQ2_LABEL_INPUTS = list(set(LABEL_DEFINING_COLS + LABEL_RULE_INPUTS))

RQ2_MODEL_FEATURES = list(FEATURES)  # already clean; no filtering needed

def ensure_project_dirs() -> None:
    for path in [DATA_RAW, DATA_CURATED, DATA_FEATURES, DATA_MODELS, DATA_RESULTS, SQL_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
