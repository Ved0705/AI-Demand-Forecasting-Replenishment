"""Phase 4 — Rolling-origin backtesting framework.

This module orchestrates:
  1. Fold generation (expanding window, config-driven — D-007/D-020).
  2. Baseline forecasting (naive, seasonal naive, MA7, MA28, zero — D-023).
  3. Metric evaluation by segment, store, category, and horizon (D-024/D-025).
  4. Output file writing.

HARD STOP: No forecasting models are implemented here.
This module establishes the infrastructure that Phase 5 models will plug into.

LEAKAGE POLICY (D-022)
-----------------------
  - Training data is always restricted to date <= train_end.
  - Baseline computations are performed on the training tail only.
  - Forecast date features are derived from known-future calendar columns.
  - No information from the test window is used to compute any forecast.

FOLD DESIGN (D-020)
-------------------
  Expanding window: train_start is always the global first date.
  train_end advances by step_days each fold.
  Folds are anchored at the end of the data and stepped backwards so that:
    - The most recent periods are always evaluated.
    - Every fold's test window falls within available data.
    - Folds with insufficient training history are skipped (not dropped silently).

See config.yaml backtest: section for all numeric parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.metrics import evaluate_predictions, summarise_metrics

logger = logging.getLogger(__name__)

BASELINES = ["naive", "seasonal_naive", "ma7", "ma28", "zero"]


# ---------------------------------------------------------------------------
# Fold specification
# ---------------------------------------------------------------------------

@dataclass
class FoldSpec:
    """Complete specification for one backtest fold.

    All date boundaries are inclusive.  The fold records both the intended
    boundary (from config) and the realised counts (from the data).
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp       # training cutoff (inclusive)
    forecast_start: pd.Timestamp  # first forecast date = train_end + 1
    forecast_end: pd.Timestamp    # last forecast date = train_end + horizon

    # Filled in after the fold is evaluated
    n_train_rows: int = 0
    n_test_rows: int = 0
    n_series: int = 0
    skipped_series: int = 0       # series with insufficient history

    def as_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "forecast_start": str(self.forecast_start.date()),
            "forecast_end": str(self.forecast_end.date()),
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "n_series": self.n_series,
            "skipped_series": self.skipped_series,
        }


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

def generate_folds(
    long_df: pd.DataFrame,
    cfg: Config,
    date_col: str = "date",
) -> list[FoldSpec]:
    """Generate the expanding-window fold specifications from config.

    Fold anchoring strategy (D-020):
      - Folds are positioned at the END of the data (most recent periods).
      - The last fold's test window ends at the last date in the data.
      - Earlier folds step backwards by step_days.
      - Folds where train_end - train_start < min_train_days are skipped.

    Parameters
    ----------
    long_df : Long-format active-row DataFrame.
    cfg     : Loaded config (reads backtest: section).

    Returns
    -------
    List of FoldSpec, sorted by fold_id (chronological order).
    The list may have fewer folds than n_folds if the data is too short.
    """
    bt = cfg["backtest"]
    horizon = int(bt["horizon_days"])
    n_folds = int(bt["n_folds"])
    step = int(bt["step_days"])
    min_train = int(bt["min_train_days"])

    dates = pd.to_datetime(long_df[date_col])
    last_date = dates.max()
    first_date = dates.min()

    # The last fold's train_end is (last_date - horizon_days).
    # Stepping backwards gives earlier fold train_ends.
    last_train_end = last_date - pd.Timedelta(days=horizon)

    folds: list[FoldSpec] = []
    for i in range(n_folds - 1, -1, -1):
        train_end = last_train_end - pd.Timedelta(days=i * step)
        forecast_start = train_end + pd.Timedelta(days=1)
        forecast_end = train_end + pd.Timedelta(days=horizon)

        # Sanity: forecast window must stay within the data
        if forecast_end > last_date:
            logger.warning(
                "Fold %d: forecast_end %s > last date %s; skipping.",
                len(folds) + 1, forecast_end.date(), last_date.date(),
            )
            continue

        train_days = (train_end - first_date).days
        if train_days < min_train:
            logger.warning(
                "Fold %d: only %d training days (< min_train_days=%d); skipping.",
                len(folds) + 1, train_days, min_train,
            )
            continue

        folds.append(FoldSpec(
            fold_id=len(folds) + 1,
            train_start=first_date,
            train_end=train_end,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
        ))

    if not folds:
        logger.error(
            "No valid folds generated.  "
            "Data spans %d days; need min_train_days=%d + n_folds=%d * step_days=%d = %d days.",
            (last_date - first_date).days, min_train, n_folds, step,
            min_train + n_folds * step,
        )
    else:
        logger.info("Generated %d fold(s) of %d requested.", len(folds), n_folds)

    return folds


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _series_tail(
    train_df: pd.DataFrame,
    n: int,
    key_cols: list[str] = ("store_id", "item_id"),
    target_col: str = "sales",
    date_col: str = "date",
) -> pd.DataFrame:
    """Return the last n rows per series from the training window, sorted."""
    key_cols = list(key_cols)
    return (
        train_df.sort_values(key_cols + [date_col])
        .groupby(key_cols, observed=True)
        .tail(n)
    )


def apply_naive(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    key_cols: list[str] = ("store_id", "item_id"),
    target_col: str = "sales",
    date_col: str = "date",
) -> pd.Series:
    """Naive forecast: last observed value per series.

    ŷ[t] = y[train_end] for all forecast dates t.
    If the series has no training observations, the forecast is NaN.
    """
    key_cols = list(key_cols)
    last_val = (
        train_df.sort_values(key_cols + [date_col])
        .groupby(key_cols, observed=True)[target_col]
        .last()
        .rename("naive")
        .reset_index()
    )
    result = forecast_df[key_cols + [date_col]].merge(last_val, on=key_cols, how="left")
    return result["naive"].values


def apply_seasonal_naive(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    period: int = 7,
    key_cols: list[str] = ("store_id", "item_id"),
    target_col: str = "sales",
    date_col: str = "date",
) -> pd.Series:
    """Seasonal naive: ŷ[t] = y[t - period].

    For each forecast date t, looks up y[t - period] in the training window.
    Returns NaN if that date falls outside the training window.

    The 7-day period captures the weekly demand cycle identified in Phase 3.
    """
    key_cols = list(key_cols)
    train_sorted = train_df.sort_values(key_cols + [date_col])

    # Build a lookup: (series_key, date) -> sales
    lookup = train_sorted.set_index(key_cols + [date_col])[target_col]

    results = []
    for _, row in forecast_df.iterrows():
        lookback = row[date_col] - pd.Timedelta(days=period)
        key = tuple(row[k] for k in key_cols) + (lookback,)
        results.append(float(lookup.get(key, np.nan)))

    return np.array(results, dtype=float)


def apply_moving_average(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    window: int,
    key_cols: list[str] = ("store_id", "item_id"),
    target_col: str = "sales",
    date_col: str = "date",
) -> np.ndarray:
    """Moving average: mean of the last ``window`` training observations.

    ŷ[t] = mean(y[train_end - window + 1 .. train_end]).
    Same value broadcast across all horizon days for a series.
    Uses min_periods=1 so a series with fewer than ``window`` observations
    still gets a forecast (from all available history).
    """
    key_cols = list(key_cols)
    tail = _series_tail(train_df, window, key_cols, target_col, date_col)
    ma_val = (
        tail.groupby(key_cols, observed=True)[target_col]
        .mean()
        .rename(f"ma{window}")
        .reset_index()
    )
    result = forecast_df[key_cols + [date_col]].merge(ma_val, on=key_cols, how="left")
    return result[f"ma{window}"].values


def apply_zero_baseline(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    key_cols: list[str] = ("store_id", "item_id"),
) -> np.ndarray:
    """Zero forecast: ŷ[t] = 0 for all series and all dates.

    This is a surprisingly strong baseline for intermittent/lumpy series
    (Phase 3: 8,350/9,147 = 91.3% of series are intermittent or lumpy).
    Any sophisticated model must beat this explicitly for those segments.

    See D-023.
    """
    return np.zeros(len(forecast_df), dtype=float)


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def _run_single_fold(
    long_df: pd.DataFrame,
    fold: FoldSpec,
    seg_df: pd.DataFrame | None,
    key_cols: list[str],
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Apply all baselines to one fold and return a prediction DataFrame.

    Returned columns:
        fold_id, store_id, item_id, date, horizon_day,
        actual, cat_id, segment,
        naive, seasonal_naive, ma7, ma28, zero
    """
    # Training data (leakage guard: strict <= train_end)
    train_df = long_df.loc[long_df[date_col] <= fold.train_end].copy()

    # Test data (actuals for evaluation — never used to build forecasts)
    test_df = long_df.loc[
        (long_df[date_col] >= fold.forecast_start) &
        (long_df[date_col] <= fold.forecast_end)
    ].copy()

    if test_df.empty:
        logger.warning("Fold %d: empty test window; skipping.", fold.fold_id)
        return pd.DataFrame()

    # Update fold metadata
    fold.n_train_rows = len(train_df)
    fold.n_test_rows = len(test_df)
    fold.n_series = train_df[key_cols].drop_duplicates().shape[0]

    # Series that appear in the test window but have no training history
    train_series = set(map(tuple, train_df[key_cols].drop_duplicates().values))
    test_series = set(map(tuple, test_df[key_cols].drop_duplicates().values))
    cold_start = test_series - train_series
    fold.skipped_series = len(cold_start)
    if cold_start:
        logger.warning(
            "Fold %d: %d series in test window have no training history.",
            fold.fold_id, len(cold_start),
        )

    # Forecast skeleton (one row per series × forecast date in test window)
    forecast_df = test_df[key_cols + [date_col]].copy().reset_index(drop=True)

    # Apply all baselines
    forecast_df["naive"] = apply_naive(train_df, forecast_df, key_cols, target_col, date_col)
    forecast_df["seasonal_naive"] = apply_seasonal_naive(
        train_df, forecast_df, period=7, key_cols=key_cols,
        target_col=target_col, date_col=date_col,
    )
    forecast_df["ma7"] = apply_moving_average(
        train_df, forecast_df, window=7, key_cols=key_cols,
        target_col=target_col, date_col=date_col,
    )
    forecast_df["ma28"] = apply_moving_average(
        train_df, forecast_df, window=28, key_cols=key_cols,
        target_col=target_col, date_col=date_col,
    )
    forecast_df["zero"] = apply_zero_baseline(train_df, forecast_df, key_cols)

    # Merge in actuals
    actual_cols = key_cols + [date_col, target_col]
    # Add category from long_df if available
    extra_cols = [c for c in ["cat_id", "dept_id", "state_id"] if c in long_df.columns]
    actual_cols += extra_cols
    actuals = test_df[actual_cols].copy()
    result = forecast_df.merge(actuals, on=key_cols + [date_col], how="left")
    result = result.rename(columns={target_col: "actual"})

    # Horizon day (1-indexed offset from forecast_start)
    result["horizon_day"] = (
        (result[date_col] - fold.forecast_start).dt.days + 1
    ).astype(int)
    result["fold_id"] = fold.fold_id

    # Merge segmentation
    if seg_df is not None:
        result = result.merge(seg_df[key_cols + ["segment"]], on=key_cols, how="left")
    else:
        result["segment"] = "unknown"

    return result


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _agg_metrics(
    pred_df: pd.DataFrame,
    group_cols: list[str],
    method_cols: list[str] = BASELINES,
) -> pd.DataFrame:
    """Aggregate predictions into a metric summary table."""
    records = []
    for keys, gdf in pred_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for method in method_cols:
            if method not in gdf.columns:
                continue
            row = dict(zip(group_cols, keys))
            row["method"] = method
            row.update(summarise_metrics(gdf["actual"], gdf[method]))
            records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main backtest orchestration
# ---------------------------------------------------------------------------

def run_backtest(
    long_df: pd.DataFrame,
    cfg: Config,
    seg_df: pd.DataFrame | None = None,
    date_col: str = "date",
    target_col: str = "sales",
    key_cols: tuple[str, ...] = ("store_id", "item_id"),
) -> dict[str, pd.DataFrame]:
    """Run the full Phase 4 baseline backtest.

    Parameters
    ----------
    long_df : Active-row long-format DataFrame (output of build_long_table).
    cfg     : Loaded Config (reads backtest: section).
    seg_df  : Optional DataFrame with (store_id, item_id, segment) columns.
              If None, all series are labelled 'unknown'.

    Returns
    -------
    dict with keys:
        'fold_meta'      : FoldSpec metadata, one row per fold.
        'all_predictions': Raw prediction DataFrame (one row per series × date × fold).
        'summary'        : Method × fold aggregation.
        'segment_results': Method × fold × segment aggregation.
        'horizon_results': Method × fold × horizon_day aggregation.
        'store_results'  : Method × fold × store aggregation.
        'category_results': Method × fold × category aggregation (if cat_id present).
    """
    key_cols = list(key_cols)
    long_df = long_df.copy()
    long_df[date_col] = pd.to_datetime(long_df[date_col])

    folds = generate_folds(long_df, cfg, date_col)
    if not folds:
        raise RuntimeError(
            "No valid folds could be generated.  "
            "Check config backtest: section and that the data covers enough days."
        )

    all_preds: list[pd.DataFrame] = []
    for fold in folds:
        logger.info(
            "Fold %d/%d  train=%s..%s  forecast=%s..%s",
            fold.fold_id, len(folds),
            fold.train_start.date(), fold.train_end.date(),
            fold.forecast_start.date(), fold.forecast_end.date(),
        )
        fold_preds = _run_single_fold(
            long_df, fold, seg_df, key_cols, date_col, target_col,
        )
        if not fold_preds.empty:
            all_preds.append(fold_preds)

    if not all_preds:
        raise RuntimeError("All folds produced empty results.")

    pred_df = pd.concat(all_preds, ignore_index=True)

    # --- Aggregations -------------------------------------------------------
    summary = _agg_metrics(pred_df, ["fold_id"])
    segment_results = _agg_metrics(pred_df, ["fold_id", "segment"])
    horizon_results = _agg_metrics(pred_df, ["fold_id", "horizon_day"])
    store_results = _agg_metrics(pred_df, ["fold_id", "store_id"])

    extra: dict[str, pd.DataFrame] = {}
    if "cat_id" in pred_df.columns:
        extra["category_results"] = _agg_metrics(pred_df, ["fold_id", "cat_id"])

    fold_meta = pd.DataFrame([f.as_dict() for f in folds])

    return {
        "fold_meta": fold_meta,
        "all_predictions": pred_df,
        "summary": summary,
        "segment_results": segment_results,
        "horizon_results": horizon_results,
        "store_results": store_results,
        **extra,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    results: dict[str, pd.DataFrame],
    out_dir: Path,
) -> dict[str, Path]:
    """Persist Phase 4 evaluation outputs to CSV files.

    Returns a dict of {name: path} for every file written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    file_map = {
        "fold_meta": "fold_meta.csv",
        "summary": "backtest_summary.csv",
        "segment_results": "segment_results.csv",
        "horizon_results": "horizon_results.csv",
        "store_results": "store_results.csv",
    }
    if "category_results" in results:
        file_map["category_results"] = "category_results.csv"

    for key, fname in file_map.items():
        if key in results:
            path = out_dir / fname
            results[key].to_csv(path, index=False, encoding="utf-8")
            written[key] = path
            logger.info("Wrote %s (%d rows)", fname, len(results[key]))

    # Fold-level prediction dump is gitignored by default (large file).
    # Written only for local inspection, not committed.
    pred_path = out_dir / "fold_results.csv"
    results["all_predictions"].to_csv(pred_path, index=False, encoding="utf-8")
    written["all_predictions"] = pred_path
    logger.info(
        "Wrote fold_results.csv (%d rows) — contains raw predictions; "
        "not committed to git.",
        len(results["all_predictions"]),
    )

    return written


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    results: dict[str, pd.DataFrame],
    cfg: Config,
    folds: list[FoldSpec],
    out_path: Path,
) -> Path:
    """Write the Phase 4 Markdown report to out_path."""
    bt = cfg["backtest"]
    L: list[str] = []
    A = L.append

    A("# Phase 4 — Backtesting Framework Results\n")
    A("Generated by `src/backtest.py`.  "
      "All figures come from the baseline forecasts against held-out data.\n")

    A("## Configuration\n")
    A(f"| Parameter | Value |")
    A(f"|-----------|-------|")
    A(f"| Scheme | {bt['scheme']} window |")
    A(f"| Horizon | {bt['horizon_days']} days |")
    A(f"| Folds | {len(folds)} (of {bt['n_folds']} requested) |")
    A(f"| Step | {bt['step_days']} days |")
    A(f"| Min training days | {bt['min_train_days']} |")
    A("")

    A("## Fold Summary\n")
    A(results["fold_meta"].to_markdown(index=False))
    A("")

    A("## Baselines\n")
    A("| Baseline | Description |")
    A("|----------|-------------|")
    A("| `naive` | Last observed value per series |")
    A("| `seasonal_naive` | Value from 7 days earlier in training |")
    A("| `ma7` | Mean of last 7 training observations |")
    A("| `ma28` | Mean of last 28 training observations |")
    A("| `zero` | Constant 0 (strong baseline for intermittent/lumpy series) |")
    A("")

    A("## Overall Metrics (all folds pooled)\n")
    A("> Primary metrics: **MAE** and **WAPE**.  "
      "MAPE is informational only; undefined for zero-demand rows "
      "(59.9% of series have mean zero-share > 50%). See D-024.\n")
    overall = results["summary"].groupby("method")[
        ["mae", "wape", "rmse", "bias"]
    ].mean().reset_index()
    A(overall.to_markdown(index=False))
    A("")

    A("## Metrics by Segment\n")
    A("> Segment-level evaluation is mandatory (D-025).  "
      "A model that performs well only because smooth/high-volume series dominate "
      "must not look artificially good.\n")
    if "segment_results" in results:
        seg = results["segment_results"].groupby(["method", "segment"])[
            ["mae", "wape"]
        ].mean().reset_index()
        A(seg.to_markdown(index=False))
    A("")

    A("## Metrics by Store\n")
    if "store_results" in results:
        store = results["store_results"].groupby(["method", "store_id"])[
            ["mae", "wape"]
        ].mean().reset_index()
        A(store.to_markdown(index=False))
    A("")

    if "category_results" in results:
        A("## Metrics by Category\n")
        cat = results["category_results"].groupby(["method", "cat_id"])[
            ["mae", "wape"]
        ].mean().reset_index()
        A(cat.to_markdown(index=False))
        A("")

    A("## Horizon Analysis\n")
    A("MAE by forecast horizon day (averaged across folds), best 3 methods:\n")
    if "horizon_results" in results:
        top_methods = (
            results["summary"].groupby("method")["mae"].mean().nsmallest(3).index.tolist()
        )
        hr = results["horizon_results"]
        hr_top = hr[hr["method"].isin(top_methods)]
        pivot = hr_top.pivot_table(
            index="horizon_day", columns="method", values="mae", aggfunc="mean"
        ).reset_index()
        A(pivot.to_markdown(index=False))
    A("")

    A("## Limitations\n")
    A("1. **Observed sales ≠ demand.**  Stockout censoring is invisible in M5.")
    A("2. **Three stores only.**  Results do not generalise to all 10 stores.")
    A("3. **Baselines only.**  Phase 5 will implement and compare forecasting models.")
    A("4. **SNAP is treated as known future.**  This is correct (published schedule) "
      "but requires that production systems have access to it at forecast time.")
    A("5. **Price is not used.**  It is endogenous and cannot be forecast reliably "
      "without an identification strategy (Phase 3 finding).")
    A("6. **Block bootstrap not applied to forecast errors.**  Temporal dependence "
      "means the variance of fold-to-fold metric variation is understated.")
    A("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    """Run the Phase 4 baseline backtest on the real M5 processed data."""
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(config_path)
    cfg.ensure_dirs()

    parquet = cfg.path("processed") / "sales_long.parquet"
    if not parquet.exists():
        raise SystemExit(
            f"No processed data at {parquet}.\n"
            "Run `python -m src.ingest` first."
        )

    long_df = pd.read_parquet(parquet)
    logger.info("Loaded %d rows from %s", len(long_df), parquet.name)

    # Load Phase 1 segmentation
    seg_path = cfg.path("reports") / "series_stats.csv"
    seg_df = None
    if seg_path.exists():
        seg_df = pd.read_csv(seg_path)[["store_id", "item_id", "segment"]]
        logger.info("Loaded segmentation for %d series", len(seg_df))
    else:
        logger.warning(
            "No segmentation file at %s.  "
            "Run `python -m src.profile` first for segment-level metrics.",
            seg_path,
        )

    results = run_backtest(long_df, cfg, seg_df)

    out_dir = cfg.path("reports") / "phase4"
    written = write_outputs(results, out_dir)

    folds = generate_folds(long_df, cfg)
    report_path = out_dir / "phase4_backtesting.md"
    write_report(results, cfg, folds, report_path)

    # Summary to stdout
    print("\n=== Phase 4 Backtest Complete ===")
    print(f"Folds:   {len(results['fold_meta'])}")
    print(f"Series:  {results['all_predictions'][['store_id','item_id']].drop_duplicates().shape[0]:,}")
    print(f"Pred rows: {len(results['all_predictions']):,}")
    print("\nOverall MAE by method (averaged across folds):")
    print(
        results["summary"]
        .groupby("method")["mae"]
        .mean()
        .sort_values()
        .to_string()
    )
    print("\nOverall WAPE by method:")
    print(
        results["summary"]
        .groupby("method")["wape"]
        .mean()
        .sort_values()
        .to_string()
    )
    print("\nWritten:")
    for k, p in written.items():
        print(f"  {p}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
