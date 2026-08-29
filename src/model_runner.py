"""Phase 5 — Model Runner for Forecasting Benchmark.

Orchestrates the evaluation of Croston, SBA, TSB, SES, and Global XGBoost
against the Phase 4 baseline framework.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.backtest import generate_folds, FoldSpec
from src.features import build_fold_features, FeatureSet
from src.metrics import summarise_metrics
from src.forecasting_models import (
    apply_croston,
    apply_sba,
    apply_tsb,
    apply_ses,
    GlobalXGBoostForecaster,
)

logger = logging.getLogger(__name__)

# Models to evaluate
MODELS = ["croston", "sba", "tsb", "ses", "xgboost"]
# Also carrying forward baselines for direct comparison
BASELINES = ["naive", "seasonal_naive", "ma7", "ma28", "zero"]
ALL_METHODS = BASELINES + MODELS

def _agg_metrics(
    pred_df: pd.DataFrame,
    group_cols: list[str],
    method_cols: list[str] = ALL_METHODS,
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

def run_phase5_backtest(
    long_df: pd.DataFrame,
    cfg: Config,
    seg_df: pd.DataFrame | None = None,
    date_col: str = "date",
    target_col: str = "sales",
    key_cols: tuple[str, ...] = ("store_id", "item_id"),
) -> dict[str, pd.DataFrame]:
    """Run the Phase 5 forecasting benchmark."""
    key_cols = list(key_cols)
    # Use shallow copy to prevent large memory allocations
    # (Avoids duplicating untouched columns like 'sales' and 'item_id')
    long_df = long_df.copy(deep=False)
    
    if not pd.api.types.is_datetime64_any_dtype(long_df[date_col]):
        long_df[date_col] = pd.to_datetime(long_df[date_col])

    # For XGBoost, convert categoricals efficiently
    cat_features = ["store_id", "cat_id", "dept_id", "state_id"]
    for c in cat_features:
        if c in long_df.columns and not isinstance(long_df[c].dtype, pd.CategoricalDtype):
            long_df[c] = long_df[c].astype("category")

    bt = cfg["backtest"]
    horizon = int(bt["horizon_days"])
    
    folds = generate_folds(long_df, cfg, date_col)
    if not folds:
        raise RuntimeError("No valid folds could be generated.")

    all_preds: list[pd.DataFrame] = []
    
    for fold in folds:
        logger.info(
            "Fold %d/%d  train=%s..%s  forecast=%s..%s",
            fold.fold_id, len(folds),
            fold.train_start.date(), fold.train_end.date(),
            fold.forecast_start.date(), fold.forecast_end.date(),
        )
        
        # 1. Build features (leakage-safe slice happens inside here)
        logger.info("Building features (chunked)...")
        keys_df = long_df[key_cols].drop_duplicates().reset_index(drop=True)
        chunk_size = 500
        chunks = [keys_df.iloc[i:i + chunk_size] for i in range(0, len(keys_df), chunk_size)]
        
        train_fs_list = []
        forecast_fs_list = []
        hist_cols, cal_cols = [], []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_df = long_df.merge(chunk, on=key_cols, how="inner")
            t_fs, f_fs = build_fold_features(
                chunk_df, fold.train_end, horizon,
                date_col=date_col, target_col=target_col, key_cols=key_cols
            )
            train_fs_list.append(t_fs.df)
            forecast_fs_list.append(f_fs.df)
            if not hist_cols and t_fs.historical_cols:
                hist_cols = t_fs.historical_cols
                cal_cols = t_fs.known_future_cols
                
        if not train_fs_list:
            continue
            
        train_df = pd.concat(train_fs_list, ignore_index=True)
        forecast_df = pd.concat(forecast_fs_list, ignore_index=True)
        
        train_fs = FeatureSet(train_df, hist_cols, cal_cols)
        forecast_fs = FeatureSet(forecast_df, hist_cols, cal_cols)
        
        # Test data (actuals)
        test_df = long_df.loc[
            (long_df[date_col] >= fold.forecast_start) &
            (long_df[date_col] <= fold.forecast_end)
        ].copy()
        
        if test_df.empty:
            continue
            
        fold.n_train_rows = len(train_fs.df)
        fold.n_test_rows = len(test_df)
        
        # Merge actuals with forecast framework
        forecast_df = forecast_fs.df.copy()
        actual_cols = key_cols + [date_col, target_col]
        extra_cols = [c for c in ["cat_id", "dept_id", "state_id"] if c in long_df.columns]
        actual_cols += extra_cols
        
        actuals = test_df[actual_cols].copy()
        result = forecast_df.merge(actuals, on=key_cols + [date_col], how="left")
        result = result.rename(columns={target_col: "actual"})
        
        # Add baselines from Phase 4 functions manually (we'll just use the fast np array apply or 
        # import them to keep it clean, but wait, Phase 4 baselines are evaluated by importing them)
        from src.backtest import (
            apply_naive, apply_seasonal_naive, apply_moving_average, apply_zero_baseline
        )
        
        train_raw = long_df.loc[long_df[date_col] <= fold.train_end]
        result["naive"] = apply_naive(train_raw, result, key_cols, target_col, date_col)
        result["seasonal_naive"] = apply_seasonal_naive(train_raw, result, period=7, key_cols=key_cols, target_col=target_col, date_col=date_col)
        result["ma7"] = apply_moving_average(train_raw, result, window=7, key_cols=key_cols, target_col=target_col, date_col=date_col)
        result["ma28"] = apply_moving_average(train_raw, result, window=28, key_cols=key_cols, target_col=target_col, date_col=date_col)
        result["zero"] = apply_zero_baseline(train_raw, result, key_cols)
        
        # 2. Intermittent & Statistical Models (Per-series)
        logger.info("Running per-series models (Croston, SBA, TSB, SES)...")
        # We need the grouped historical data. train_raw is already sliced <= cutoff.
        # Ensure it is sorted (it should already be sorted from ingest phase, but we just pass it)
        train_sorted = train_raw
        
        # Create a dictionary to hold the single-point forecasts for each series
        preds_croston = {}
        preds_sba = {}
        preds_tsb = {}
        preds_ses = {}
        
        # For performance, we can groupby
        for keys, sdf in train_sorted.groupby(key_cols, observed=True):
            y_arr = sdf[target_col].values
            if not isinstance(keys, tuple):
                keys = (keys,)
                
            preds_croston[keys] = apply_croston(y_arr, alpha=0.1)
            preds_sba[keys] = apply_sba(y_arr, alpha=0.1, beta=0.1)
            preds_tsb[keys] = apply_tsb(y_arr, alpha=0.1, beta=0.1)
            preds_ses[keys] = apply_ses(y_arr, alpha=0.1)
            
        # Map them back to result (flat forecast across horizon)
        def map_preds(pred_dict):
            series = pd.Series(pred_dict)
            series.index.names = key_cols
            series.name = "val"
            return result.merge(series, on=key_cols, how="left")["val"].fillna(0.0).values
            
        result["croston"] = map_preds(preds_croston)
        result["sba"] = map_preds(preds_sba)
        result["tsb"] = map_preds(preds_tsb)
        result["ses"] = map_preds(preds_ses)
        
        # 3. Global ML Model
        logger.info("Running Global XGBoost...")
        xgb_features = train_fs.all_feature_cols
        # Exclude 'year' as per user instruction if it's there
        xgb_features = [f for f in xgb_features if f != "year"]
        # Include categorical encoded
        for c in cat_features:
            if c in train_fs.df.columns and c not in xgb_features:
                xgb_features.append(c)
                
        # Fill NAs in features for XGBoost (XGBoost handles NA natively, but standard practice is 
        # to leave them or -999. We leave them for native handling)
        xgb_model = GlobalXGBoostForecaster(enable_categorical=True)
        xgb_model.fit(train_fs.df, xgb_features, target="y")
        
        result["xgboost"] = xgb_model.predict(forecast_fs.df, xgb_features)
        
        # Add metadata
        result["horizon_day"] = (
            (result[date_col] - fold.forecast_start).dt.days + 1
        ).astype(int)
        result["fold_id"] = fold.fold_id
        
        if seg_df is not None:
            result = result.merge(seg_df[key_cols + ["segment"]], on=key_cols, how="left")
        else:
            result["segment"] = "unknown"
            
        all_preds.append(result)

    if not all_preds:
        raise RuntimeError("All folds produced empty results.")

    pred_df = pd.concat(all_preds, ignore_index=True)

    # Aggregations
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

def write_outputs(
    results: dict[str, pd.DataFrame],
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    file_map = {
        "fold_meta": "fold_meta.csv",
        "summary": "model_comparison.csv",
        "segment_results": "segment_model_comparison.csv",
        "horizon_results": "horizon_model_comparison.csv",
        "store_results": "store_model_comparison.csv",
    }
    if "category_results" in results:
        file_map["category_results"] = "category_model_comparison.csv"

    for key, fname in file_map.items():
        if key in results:
            path = out_dir / fname
            results[key].to_csv(path, index=False, encoding="utf-8")
            written[key] = path

    # Local inspection
    pred_path = out_dir / "fold_results.csv"
    results["all_predictions"].to_csv(pred_path, index=False, encoding="utf-8")
    written["all_predictions"] = pred_path
    
    return written

if __name__ == "__main__":
    pass
