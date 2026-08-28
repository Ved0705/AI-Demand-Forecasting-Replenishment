"""Evaluation metrics for Phase 4 backtesting.

METRIC SELECTION (D-024)
------------------------
Primary metrics:
  MAE  — mean absolute error.  Always defined; directly interpretable in
          units of demand.  The right headline for intermittent series.
  WAPE — weighted absolute percentage error = sum(|y-ŷ|)/sum(|y|).
          Volume-weighted, so large-volume series dominate appropriately.
          Returns NaN when sum(|y|)=0 (all-zero test window).

Secondary metrics:
  RMSE — penalises large errors more; useful for smooth/high-volume series.
  Bias — mean(ŷ - y).  Positive = systematic over-forecast.

Informational only (use with care):
  sMAPE — symmetric MAPE.  The (|y|+|ŷ|)/2 denominator bounds the measure
          to [0, 2], but produces 0/0 when both y=0 and ŷ=0.  We treat
          those cases as 0 error.
  MAPE  — ONLY reported where y > 0.  Undefined and misleading for
          intermittent/zero-inflated series.  Phase 3 concluded (and D-024
          records) that MAPE is not a primary metric here.

All functions:
  - Accept array-like inputs (list, ndarray, Series).
  - Drop NaN pairs before computing.
  - Return np.nan rather than raising on empty or degenerate input.
  - Are tested against hand-calculated examples in tests/test_metrics.py.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(
    actual: Sequence[float],
    forecast: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float arrays, drop pairs where either value is NaN."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(
            f"actual and forecast must have the same shape; "
            f"got {a.shape} vs {f.shape}"
        )
    mask = ~(np.isnan(a) | np.isnan(f))
    return a[mask], f[mask]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def mae(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Mean Absolute Error: mean(|y - ŷ|).

    Primary metric.  Always finite for non-empty, non-NaN input.
    Returns np.nan for empty arrays.
    """
    a, f = _clean(actual, forecast)
    if a.size == 0:
        return np.nan
    return float(np.mean(np.abs(a - f)))


def wape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Weighted Absolute Percentage Error: sum(|y - ŷ|) / sum(|y|).

    Primary metric.  Volume-weighted: a series with total demand of 1000
    units contributes 1000x more than a series with total demand of 1 unit.
    This is appropriate for retail where a blended metric should reflect
    commercial importance.

    Returns np.nan when sum(|y|) == 0 (all-zero actual window).  The caller
    should record and report the count of NaN wape values rather than
    silently ignoring them.
    """
    a, f = _clean(actual, forecast)
    if a.size == 0:
        return np.nan
    denom = np.sum(np.abs(a))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(a - f)) / denom)


def rmse(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Root Mean Squared Error: sqrt(mean((y - ŷ)^2)).

    Secondary metric.  Penalises large errors more than MAE; useful for
    smooth/high-volume series where large errors are commercially significant.
    Not preferred for intermittent series where many zeros inflate RMSE.
    """
    a, f = _clean(actual, forecast)
    if a.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((a - f) ** 2)))


def bias(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Mean forecast bias: mean(ŷ - y).

    Positive = systematic over-forecast.
    Negative = systematic under-forecast.
    Zero bias is necessary but not sufficient for a good forecast.
    """
    a, f = _clean(actual, forecast)
    if a.size == 0:
        return np.nan
    return float(np.mean(f - a))


def smape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Symmetric MAPE: mean(2|y - ŷ| / (|y| + |ŷ|)).

    Bounded to [0, 2].  When both y=0 and ŷ=0 the term is 0/0; we treat
    this as 0 (no error, not undefined) because the forecast is correct.
    When y=0, ŷ>0 the term equals 2 (maximum error) — appropriate.

    Use with care: still behaves oddly near zero.  Report alongside MAE.
    """
    a, f = _clean(actual, forecast)
    if a.size == 0:
        return np.nan
    denom = np.abs(a) + np.abs(f)
    numer = 2.0 * np.abs(a - f)
    with np.errstate(invalid="ignore"):   # suppress 0/0 warning; handled by np.where
        terms = np.where(denom == 0, 0.0, numer / denom)
    return float(np.mean(terms))


def mape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Mean Absolute Percentage Error.

    **WARNING — USE WITH CAUTION.**
    - Undefined when y = 0, which occurs on the majority of rows for
      intermittent/lumpy series (Phase 3: mean zero-share = 59.9%).
    - Only computed over rows where y > 0.  The denominator-exclusion count
      is NOT returned by this function; callers should track it separately.
    - Do NOT report MAPE as a primary metric.  See D-024.

    Returns np.nan if no row has y > 0.
    """
    a, f = _clean(actual, forecast)
    mask = a > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(a[mask] - f[mask]) / a[mask]))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def summarise_metrics(
    actual: Sequence[float],
    forecast: Sequence[float],
    prefix: str = "",
) -> dict[str, float]:
    """Compute all metrics for one (actual, forecast) pair.

    Parameters
    ----------
    actual, forecast : Aligned sequences of actual and predicted values.
    prefix           : Optional prefix to add to all keys (e.g. "naive_").

    Returns
    -------
    dict with keys: mae, wape, rmse, bias, smape, mape.
    MAPE is always included but should be treated as informational only.
    """
    a, f = _clean(actual, forecast)
    n_nonzero_actual = int((np.asarray(actual, dtype=float) > 0).sum())
    result = {
        "n_obs": int(a.size),
        "n_nonzero_actual": n_nonzero_actual,
        "mae": mae(a, f),
        "wape": wape(a, f),
        "rmse": rmse(a, f),
        "bias": bias(a, f),
        "smape": smape(a, f),
        "mape": mape(a, f),        # informational; NaN for zero-only actuals
    }
    if prefix:
        result = {f"{prefix}{k}": v for k, v in result.items()}
    return result


def evaluate_predictions(
    df: pd.DataFrame,
    actual_col: str = "actual",
    method_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute metric summary for one or more forecast methods, optionally grouped.

    Parameters
    ----------
    df          : DataFrame with ``actual_col`` and one or more method columns.
    actual_col  : Column containing ground-truth values.
    method_cols : Columns containing forecasts.  If None, uses all numeric
                  columns that are not ``actual_col`` or group columns.
    group_cols  : Columns to group by before computing metrics
                  (e.g. ["segment", "store_id"]).

    Returns
    -------
    DataFrame with one row per (group × method) and metric columns.
    """
    if method_cols is None:
        exclude = set([actual_col] + (group_cols or []))
        method_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude
        ]

    records = []
    groups = df.groupby(group_cols) if group_cols else [(None, df)]

    for group_key, gdf in groups:
        for method in method_cols:
            row = summarise_metrics(gdf[actual_col], gdf[method])
            row["method"] = method
            if group_cols and group_key is not None:
                if isinstance(group_key, tuple):
                    for col, val in zip(group_cols, group_key):
                        row[col] = val
                else:
                    row[group_cols[0]] = group_key
            records.append(row)

    return pd.DataFrame(records)
