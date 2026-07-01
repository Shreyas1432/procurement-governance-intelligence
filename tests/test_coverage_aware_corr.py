"""Unit tests for the NaN-aware, three-state correlation helper.

Locks the NaN-guard bug-class fix (2026-06-15): a degenerate / low-coverage input
must NOT be silently coerced to a benign 0.0; it must be distinguishable from a
genuinely uncorrelated (measured-null) input.

Coverage targets all three states (computed / measured_null / degenerate_low_coverage),
both degenerate triggers (n_paired < MIN_CORR_PAIRS and the minority-coverage rule
n_paired*2 < n_total), and the trigger ORDER (a constant input is measured_null even
when it is also a coverage minority). observed_value is asserted in every state.
"""

import numpy as np

from src.common.evaluation import coverage_aware_corr, MIN_CORR_PAIRS


def test_computed_state_full_coverage():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = a * 0.8 + rng.normal(size=500) * 0.2
    r = coverage_aware_corr(a, b)
    assert r["status"] == "computed"
    assert r["value"] is not None and 0.5 < r["value"] <= 1.0
    assert r["coverage_fraction"] == 1.0
    # In the computed state observed_value mirrors the headline value (both = r).
    assert r["observed_value"] == r["value"]
    assert r["n_paired"] == 500 and r["n_total"] == 500


def test_computed_state_exact_r():
    # Deterministic, perfectly collinear, full majority coverage -> r == 1.0.
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = a * 2.0
    r = coverage_aware_corr(a, b)
    assert r["status"] == "computed"
    assert r["value"] is not None and abs(r["value"] - 1.0) < 1e-9
    assert r["observed_value"] == r["value"]
    assert r["coverage_fraction"] == 1.0
    assert r["n_paired"] == 6 and r["n_total"] == 6


def test_degenerate_low_coverage_is_not_zero():
    # Only a tiny minority of pairs are observed -> must be degenerate, value None,
    # and must NOT report a fabricated 0.0.
    n = 1000
    a = np.linspace(0, 1, n)
    b = np.full(n, np.nan)
    obs = np.arange(5)
    b[obs] = a[obs] * -1.0 + 0.01  # observed pairs ARE correlated (slope -1)
    r = coverage_aware_corr(a, b)
    assert r["status"] == "degenerate_low_coverage"
    assert r["value"] is None
    assert r["n_paired"] == 5 and r["n_total"] == n
    assert r["observed_value"] is not None  # observed-only r preserved
    # The preserved observed-only r is the real (negative) correlation, not 0.0.
    assert abs(r["observed_value"] - (-1.0)) < 1e-9
    assert r["coverage_fraction"] == 5 / n


def test_degenerate_minority_at_min_pairs_boundary():
    # n_paired == MIN_CORR_PAIRS (count trigger does NOT fire) but still a minority
    # (n_paired*2 < n_total) -> the minority rule alone must force degenerate, with
    # observed_value populated. Proves the two degenerate triggers are independent.
    n = 20
    a = np.linspace(0, 1, n)
    b = np.full(n, np.nan)
    b[0], b[1], b[2] = 0.0, 1.0, 2.0  # 3 collinear ascending points on the rising line
    r = coverage_aware_corr(a, b)
    assert r["n_paired"] == MIN_CORR_PAIRS and r["n_total"] == n
    assert r["status"] == "degenerate_low_coverage"
    assert r["value"] is None
    assert r["observed_value"] is not None
    assert abs(r["observed_value"] - 1.0) < 1e-9
    assert r["coverage_fraction"] == MIN_CORR_PAIRS / n


def test_measured_null_constant_input():
    # Plenty of data, but one variable is constant -> genuine measured-null = 0.0,
    # explicitly distinguished from the degenerate case.
    a = np.linspace(0, 1, 100)
    b = np.ones(100)
    r = coverage_aware_corr(a, b)
    assert r["status"] == "measured_null"
    assert r["value"] == 0.0
    assert r["observed_value"] == 0.0  # measured-null reports 0.0, not None
    assert r["n_paired"] == 100 and r["n_total"] == 100
    assert r["coverage_fraction"] == 1.0


def test_measured_null_precedes_minority_coverage():
    # A constant paired input that is ALSO a coverage minority must be measured_null,
    # not degenerate: the zero-variance check precedes the minority-coverage check.
    n = 100
    a = np.linspace(0, 1, n)
    b = np.full(n, np.nan)
    b[0:5] = 7.0  # 5 paired points, all constant -> std == 0
    r = coverage_aware_corr(a, b)
    assert r["status"] == "measured_null"  # std==0 wins over the minority rule
    assert r["value"] == 0.0
    assert r["observed_value"] == 0.0


def test_too_few_pairs():
    r = coverage_aware_corr([1.0, np.nan, np.nan], [np.nan, 2.0, np.nan])
    assert r["status"] == "degenerate_low_coverage"
    assert r["value"] is None
    # On the count-trigger path (n_paired < MIN_CORR_PAIRS) observed_value stays None.
    assert r["observed_value"] is None
    assert r["n_paired"] == 0 and r["n_total"] == 3
    assert r["coverage_fraction"] == 0.0
