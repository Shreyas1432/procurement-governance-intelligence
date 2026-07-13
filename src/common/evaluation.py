import numpy as np
import polars as pl
from scipy.stats import pearsonr

# Minimum paired observations required for any correlation computation.
# Below this, not even an observed-only r is reported (count-trigger path).
MIN_CORR_PAIRS = 3


def _to_float_array(x) -> np.ndarray:
    """Convert Polars Series, pandas Series, list, or ndarray to float ndarray."""
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(allow_copy=True), dtype=float)
    return np.asarray(x, dtype=float)


def coverage_aware_corr(left, right) -> dict:
    """Three-state NaN-aware Pearson correlation helper.

    Drops null/NaN pairs, then classifies the result as one of:

    - ``computed``: paired observations are a representative majority
                                  (n_paired * 2 >= n_total) and n_paired >= MIN_CORR_PAIRS.
                                  ``value`` is the Pearson r; ``observed_value`` mirrors it.
    - ``measured_null``: enough data but one input is constant (zero variance).
                                  ``value`` = 0.0; ``observed_value`` = 0.0.
                                  This state takes precedence over the minority rule.
    - ``degenerate_low_coverage``: either n_paired < MIN_CORR_PAIRS (count trigger,
                                  ``observed_value`` = None) or n_paired * 2 < n_total
                                  (minority rule, ``observed_value`` = Pearson r over
                                  paired subset). ``value`` = None in both cases.

    Parameters
    ----------
    left, right : array-like
        Equal-length sequences.  None / NaN elements are treated as missing.

    Returns
    -------
    dict with keys: value, status, n_paired, n_total, coverage_fraction, observed_value
    """
    a = _to_float_array(left)
    b = _to_float_array(right)
    n_total = len(a)

    mask = ~(np.isnan(a) | np.isnan(b))
    a_p = a[mask]
    b_p = b[mask]
    n_paired = int(mask.sum())
    coverage_fraction = float(n_paired / n_total) if n_total > 0 else 0.0

    # Count trigger: too few paired observations for any meaningful estimate.
    if n_paired < MIN_CORR_PAIRS:
        return {
            "value": None,
            "status": "degenerate_low_coverage",
            "n_paired": n_paired,
            "n_total": n_total,
            "coverage_fraction": coverage_fraction,
            "observed_value": None,
        }

    # Zero-variance check: genuine measured null (precedes the minority rule).
    if np.std(a_p) == 0 or np.std(b_p) == 0:
        return {
            "value": 0.0,
            "status": "measured_null",
            "n_paired": n_paired,
            "n_total": n_total,
            "coverage_fraction": coverage_fraction,
            "observed_value": 0.0,
        }

    observed_r = float(pearsonr(a_p, b_p).statistic)

    # Minority coverage trigger: paired subset is less than half the population.
    if n_paired * 2 < n_total:
        return {
            "value": None,
            "status": "degenerate_low_coverage",
            "n_paired": n_paired,
            "n_total": n_total,
            "coverage_fraction": coverage_fraction,
            "observed_value": observed_r,
        }

    # Representative coverage: return the computed correlation.
    return {
        "value": observed_r,
        "status": "computed",
        "n_paired": n_paired,
        "n_total": n_total,
        "coverage_fraction": coverage_fraction,
        "observed_value": observed_r,
    }


def safe_corr(left: pl.Series, right: pl.Series) -> float:
    a = np.asarray(left.fill_null(0), dtype=float)
    b = np.asarray(right.fill_null(0), dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(pearsonr(a, b).statistic)

def risk_class(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"
