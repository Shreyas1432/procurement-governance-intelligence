import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
    return logging.getLogger(name)

@contextmanager
def timed(logger: logging.Logger, label: str):
    start = time.time()
    logger.info("Starting %s", label)
    yield
    logger.info("Finished %s in %.2fs", label, time.time() - start)

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)

def merge_json(path: Path, updates: dict) -> None:
    """Read an existing JSON object (if any), shallow-merge ``updates`` into it,
    and write it back. Used to let independent pipeline phases contribute to a
    shared metrics file (e.g. model_performance_ci.json)."""
    existing: dict = {}
    if path.exists():
        try:
            with open(path) as handle:
                existing = json.load(handle)
        except Exception:
            existing = {}
    existing.update(updates)
    write_json(path, existing)

def ci95(scores) -> dict:
    """Mean, std, 95% normal-approx CI and per-fold scores for a CV score array."""
    arr = np.asarray(scores, dtype=float)
    mean = float(np.mean(arr)) if arr.size else 0.0
    std = float(np.std(arr)) if arr.size else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - 1.96 * std, mean + 1.96 * std],
        "folds": [float(s) for s in arr],
    }

def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)

def minmax(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0, 1)
