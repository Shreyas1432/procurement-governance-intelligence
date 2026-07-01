"""RQ3 success gates.

R² and RMSE-reduction are DIAGNOSTIC ONLY (see THRESHOLD_JUSTIFICATION.md Section 12).
The contribution of RQ3 is anomaly detection, not price prediction. Gating on R² would
incentivise re-introducing the log_estimated_price leak that was removed in the
leak-forensics pass.

Behaviour-preservation gate: consensus_anomaly_rate must remain within the justified
band [RQ3_ANOMALY_RATE_MIN, RQ3_ANOMALY_RATE_MAX] (0.04 – 0.08).
"""
from src.core.config import RQ3_ANOMALY_RATE_MIN, RQ3_ANOMALY_RATE_MAX


def test_gb_r2_is_reported(rq3_results):
    """R² is a diagnostic; pass/fail gate removed post leak-fix (THRESHOLD_JUSTIFICATION.md §12).
    Assert it is computed and that GBR beats the constant-mean baseline (R² > 0)."""
    r2 = rq3_results["gb_r2"]
    assert isinstance(r2, float), "gb_r2 must be a float"
    assert r2 > 0.0, f"GBR must beat constant-mean baseline; got gb_r2={r2:.4f}"


def test_rmse_reduction_is_reported(rq3_results):
    """RMSE reduction is a diagnostic; pass/fail gate removed post leak-fix (§12).
    Assert GBR improves on the baseline mean predictor (reduction > 0)."""
    reduction = rq3_results["rmse_reduction"]
    assert reduction > 0.0, (
        f"GBR RMSE reduction must be positive; got {reduction:.4f}"
    )


def test_consensus_anomaly_rate_in_band(rq3_results):
    """3-detector consensus anomaly rate must lie within the justified band.
    Band [RQ3_ANOMALY_RATE_MIN, RQ3_ANOMALY_RATE_MAX] = [0.04, 0.08].
    This is the primary behaviour-preservation gate for RQ3 (§12)."""
    rate = rq3_results["consensus_anomaly_rate"]
    assert RQ3_ANOMALY_RATE_MIN <= rate <= RQ3_ANOMALY_RATE_MAX, (
        f"consensus_anomaly_rate={rate:.4f} outside justified band "
        f"[{RQ3_ANOMALY_RATE_MIN}, {RQ3_ANOMALY_RATE_MAX}]"
    )


def test_lof_gb_overlap_is_computed(rq3_results):
    assert 0 <= rq3_results["lof_gb_overlap"] <= 1
