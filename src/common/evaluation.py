import numpy as np
import polars as pl
from scipy.stats import pearsonr

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
