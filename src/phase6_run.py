"""Phase 6 — Production Forecasting & Replenishment Decision Engine.

Pipeline:
    DATA -> FEATURES -> FORECAST -> UNCERTAINTY/RISK -> INVENTORY STATE
         -> REPLENISHMENT DECISION -> AUDITABLE OUTPUT

This module owns the FORECAST step (production XGBoost training/loading,
production feature construction) and the CLI orchestration. The DECISION
step lives entirely in src/replenishment.py and is never mixed in here —
forecast generation has no knowledge of inventory, safety stock, or order
quantities.

XGBoost architecture, hyperparameters, and the leakage-safe feature pipeline
are reused verbatim from Phase 5 (src/features.py, src/forecasting_models.py)
per DECISION_LOG D-007/D-022/D-026. Nothing here retunes hyperparameters or
introduces a competing model.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import Config, load_config
from src.features import build_fold_features, KNOWN_FUTURE_BASE, make_known_future_features
from src.forecasting_models import GlobalXGBoostForecaster
from src.replenishment import (
    build_forecast_risk_summary,
    build_replenishment_recommendations,
    load_segment_risk_proxy,
    simulate_inventory_position,
    ReplenishmentPolicy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KEY_COLS = ["store_id", "item_id"]
CAT_FEATURES = ["store_id", "cat_id", "dept_id", "state_id"]


# ---------------------------------------------------------------------------
# Production model: train-or-load
# ---------------------------------------------------------------------------

def _expected_xgb_params() -> dict:
    """The exact Phase 5 architecture/configuration (D-007 preserved)."""
    return GlobalXGBoostForecaster(enable_categorical=True).params


def _model_paths(cfg: Config) -> tuple[Path, Path]:
    p6 = cfg.get("phase6", {})
    model_dir = cfg.root / p6.get("model_dir", "models")
    name = p6.get("model_name", "phase6_xgboost_production")
    return model_dir / f"{name}.json", model_dir / f"{name}_meta.json"


def train_or_load_production_model(
    cfg: Config,
    train_df: pd.DataFrame,
    features: list[str],
    training_cutoff: pd.Timestamp,
    force_retrain: bool = False,
) -> tuple[GlobalXGBoostForecaster, dict, bool]:
    """Reuse a persisted production model if its architecture matches Phase 5
    exactly; otherwise train ONE global model on the permitted historical
    data (date <= training_cutoff) and persist it.

    Returns (forecaster, metadata, reused_existing_model).
    """
    model_path, meta_path = _model_paths(cfg)
    expected_params = _expected_xgb_params()

    if not force_retrain and model_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("xgboost_params") == expected_params and meta.get("feature_cols") == features:
            logger.info(
                "Reusing persisted production model %s (trained %s, cutoff %s).",
                model_path.name, meta.get("trained_at"), meta.get("training_cutoff"),
            )
            regressor = xgb.XGBRegressor(**expected_params)
            regressor.load_model(str(model_path))
            forecaster = GlobalXGBoostForecaster(enable_categorical=True)
            forecaster.model = regressor
            return forecaster, meta, True
        logger.warning(
            "Persisted model at %s does not match the current Phase 5 "
            "architecture/features; retraining.", model_path,
        )

    logger.info("No usable persisted model found. Training production XGBoost "
                "on data up to cutoff=%s (%d rows).", training_cutoff.date(), len(train_df))
    forecaster = GlobalXGBoostForecaster(enable_categorical=True)
    forecaster.fit(train_df, features, target="y")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    forecaster.model.save_model(str(model_path))
    meta = {
        "xgboost_params": expected_params,
        "feature_cols": features,
        "training_cutoff": str(training_cutoff.date()),
        "n_train_rows": int(len(train_df)),
        "n_series": int(train_df[KEY_COLS].drop_duplicates().shape[0]) if set(KEY_COLS).issubset(train_df.columns) else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": "src.forecasting_models.GlobalXGBoostForecaster (Phase 5 architecture, unmodified)",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Persisted production model -> %s", model_path)
    return forecaster, meta, False


# ---------------------------------------------------------------------------
# Future known-future feature fallback
# ---------------------------------------------------------------------------

def _fill_future_calendar_features(
    forecast_df: pd.DataFrame,
    calendar_path: Path,
    store_state_map: dict[str, str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Fill known-future calendar columns for forecast dates that fall beyond
    the processed sales table's date range.

    sales_long.parquet only contains rows for dates with a sales observation,
    so genuinely future production-forecast dates (beyond the last observed
    sales day) have NaN calendar columns after build_fold_features. Those
    dates are still published in advance in M5's calendar.csv (it extends 28
    days beyond the last day of sales, exactly covering the standard 28-day
    horizon — D-021), so this is legitimate known-future information, not a
    fabrication. If calendar.csv is unavailable, the NaNs are left in place
    and logged: XGBoost handles NaN natively in tree splits (documented in
    src/forecasting_models.py), so this is a safe, explicit fallback rather
    than a crash.
    """
    cal_cols = list(KNOWN_FUTURE_BASE)
    forecast_df = forecast_df.copy()
    for c in cal_cols:
        if c not in forecast_df.columns:
            # Column absent entirely (forecast dates never appeared in
            # long_df at all) rather than merely NaN — normalize so the
            # missing-value fallback below covers both cases uniformly.
            forecast_df[c] = np.nan

    missing_mask = forecast_df[cal_cols].isna().any(axis=1)
    if not missing_mask.any():
        return forecast_df

    if not calendar_path.exists():
        logger.warning(
            "%d forecast rows have missing known-future calendar features and "
            "calendar.csv is unavailable at %s; leaving NaN (XGBoost handles "
            "NaN natively in tree splits).", int(missing_mask.sum()), calendar_path,
        )
        return forecast_df

    missing_dates = pd.to_datetime(forecast_df.loc[missing_mask, date_col].unique())
    cal = pd.read_csv(calendar_path, parse_dates=["date"])
    cal = cal.loc[cal["date"].isin(missing_dates)]

    pieces = []
    for store_id, state_id in store_state_map.items():
        feats = make_known_future_features(
            dates=cal["date"],
            store_id=store_id,
            state_id=state_id,
            snap_ca=cal.get("snap_CA"),
            snap_tx=cal.get("snap_TX"),
            snap_wi=cal.get("snap_WI"),
            event_name_1=cal.get("event_name_1"),
        )
        pieces.append(feats)
    future_cal = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    if future_cal.empty:
        return forecast_df

    merged = forecast_df.merge(
        future_cal[["store_id", "date"] + cal_cols],
        on=["store_id", "date"], how="left", suffixes=("", "_fut"),
    )
    for c in cal_cols:
        merged[c] = merged[c].combine_first(merged[f"{c}_fut"])
        merged.drop(columns=[f"{c}_fut"], inplace=True)

    remaining = merged[cal_cols].isna().any(axis=1).sum()
    if remaining:
        logger.warning(
            "%d forecast rows still have missing calendar features after the "
            "calendar.csv fallback (dates outside calendar.csv's range).",
            int(remaining),
        )
    return merged


# ---------------------------------------------------------------------------
# Production forecast generation (FORECAST layer)
# ---------------------------------------------------------------------------

def generate_production_forecast(
    long_df: pd.DataFrame,
    cfg: Config,
    calendar_path: Path | None = None,
    force_retrain: bool = False,
    training_cutoff: pd.Timestamp | None = None,
    date_col: str = "date",
    target_col: str = "sales",
    key_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict, pd.Timestamp]:
    """Produce store_id/item_id/forecast_date/forecast_units/horizon_day for
    the next ``backtest.horizon_days`` days beyond the training cutoff.

    Uses ONLY information available at forecast creation time:
      - historical features are computed with the same slice-first leakage
        guard as Phase 4/5 (src/features.py, cutoff = training_cutoff).
      - known-future features come from the published calendar (SNAP, DOW,
        events), never from future sales.

    training_cutoff defaults to the last date present in long_df, i.e. the
    most recent real M5 data available — made explicit in the returned
    metadata and the Phase 6 report.
    """
    key_cols = key_cols or KEY_COLS
    long_df = long_df.copy(deep=False)
    if not pd.api.types.is_datetime64_any_dtype(long_df[date_col]):
        long_df[date_col] = pd.to_datetime(long_df[date_col])

    for c in CAT_FEATURES:
        if c in long_df.columns and not isinstance(long_df[c].dtype, pd.CategoricalDtype):
            long_df[c] = long_df[c].astype("category")

    cutoff = training_cutoff or long_df[date_col].max()
    horizon = int(cfg["backtest"]["horizon_days"])

    logger.info("Production training cutoff: %s (explicit). Horizon: %d days.",
                cutoff.date(), horizon)

    train_fs, forecast_fs = build_fold_features(
        long_df, cutoff, horizon, date_col=date_col, target_col=target_col, key_cols=key_cols,
    )

    features = [f for f in train_fs.all_feature_cols if f != "year"]
    for c in CAT_FEATURES:
        if c in train_fs.df.columns and c not in features:
            features.append(c)

    forecaster, meta, reused = train_or_load_production_model(
        cfg, train_fs.df, features, cutoff, force_retrain=force_retrain,
    )
    meta["reused_existing_model"] = reused

    forecast_df = forecast_fs.df.copy()

    if calendar_path is None:
        calendar_path = cfg.raw_file(cfg["source"]["calendar_file"])
    if set(["store_id", "state_id"]).issubset(long_df.columns):
        store_state_map = (
            long_df[["store_id", "state_id"]].drop_duplicates()
            .astype(str).set_index("store_id")["state_id"].to_dict()
        )
        forecast_df = _fill_future_calendar_features(forecast_df, calendar_path, store_state_map, date_col)

    # The calendar-fallback merge upcasts categorical key columns to plain
    # object dtype; restore them so they match the categories the model was
    # trained on (enable_categorical=True requires true category dtype).
    for c in CAT_FEATURES:
        if c in forecast_df.columns and c in train_fs.df.columns:
            forecast_df[c] = forecast_df[c].astype(train_fs.df[c].dtype)

    preds = forecaster.predict(forecast_df, features)
    forecast_df["forecast_units"] = preds

    forecast_start = cutoff + pd.Timedelta(days=1)
    forecast_df["horizon_day"] = ((forecast_df[date_col] - forecast_start).dt.days + 1).astype(int)
    forecast_df["fold_id"] = None  # not fold-based; production forecast (auditability: distinguishes from Phase 5 backtest rows)

    out = forecast_df[key_cols + [date_col, "forecast_units", "horizon_day"]].rename(
        columns={date_col: "forecast_date"}
    )
    return out, meta, cutoff


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors src/phase5_run.py)
# ---------------------------------------------------------------------------

def _load_long_df(cfg: Config, fixture: bool, subset: int | None) -> tuple[pd.DataFrame, Path]:
    req_cols = [
        "store_id", "item_id", "date", "sales",
        "cat_id", "dept_id", "state_id",
        "wday", "month", "year",
        "event_name_1", "snap_CA", "snap_TX", "snap_WI",
    ]
    if fixture:
        parquet = Path("data/fixture/processed/sales_long.parquet")
    else:
        parquet = cfg.path("processed") / "sales_long.parquet"

    if not parquet.exists():
        raise SystemExit(f"No processed data at {parquet}. Run `python -m src.ingest` first.")

    long_df = pd.read_parquet(parquet, columns=req_cols)
    logger.info("Loaded %d rows from %s", len(long_df), parquet.name)

    if "is_event" not in long_df.columns:
        long_df["is_event"] = long_df["event_name_1"].notna().astype(np.int8)
    if "snap_active" not in long_df.columns:
        snap_ca_mask = (long_df["state_id"] == "CA") & (long_df["snap_CA"] == 1)
        snap_tx_mask = (long_df["state_id"] == "TX") & (long_df["snap_TX"] == 1)
        snap_wi_mask = (long_df["state_id"] == "WI") & (long_df["snap_WI"] == 1)
        long_df["snap_active"] = (snap_ca_mask | snap_tx_mask | snap_wi_mask).astype(np.int8)

    long_df["date"] = pd.to_datetime(long_df["date"])
    if "is_weekend" not in long_df.columns:
        long_df["is_weekend"] = long_df["date"].dt.dayofweek.isin([5, 6]).astype(np.int8)
    if "week_of_year" not in long_df.columns:
        long_df["week_of_year"] = long_df["date"].dt.isocalendar().week.astype(np.int8)

    if subset:
        series_keys = long_df[KEY_COLS].drop_duplicates().head(subset)
        long_df = long_df.merge(series_keys, on=KEY_COLS, how="inner")
        logger.info("Subset to %d series (%d rows)", subset, len(long_df))

    calendar_path = (
        Path("data/fixture/raw/calendar.csv") if fixture else cfg.raw_file(cfg["source"]["calendar_file"])
    )
    return long_df, calendar_path


def _load_segmentation(cfg: Config, fixture: bool) -> pd.DataFrame | None:
    seg_path = Path("reports/fixture/series_stats.csv") if fixture else cfg.path("reports") / "series_stats.csv"
    if seg_path.exists():
        return pd.read_csv(seg_path)[KEY_COLS + ["segment"]]
    logger.warning("No segmentation file at %s; risk map will use fallback_sigma for all series.", seg_path)
    return None


def _historical_mean_sales(long_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    train = long_df.loc[long_df["date"] <= cutoff]
    return (
        train.groupby(KEY_COLS, observed=True)["sales"].mean()
        .rename("historical_mean_sales").reset_index()
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_phase6_report(
    out_path: Path,
    cfg: Config,
    meta: dict,
    cutoff: pd.Timestamp,
    mode: str,
    forecast_df: pd.DataFrame,
    risk_summary: pd.DataFrame,
    recommendations: pd.DataFrame | None,
) -> None:
    L: list[str] = []
    A = L.append

    A("# Phase 6 — Production Forecasting & Replenishment Decision Engine\n")
    A(f"Mode: **{mode}**\n")

    A("## Architecture\n")
    A("```")
    A("DATA -> FEATURES -> FORECAST -> UNCERTAINTY/RISK -> INVENTORY STATE")
    A("     -> REPLENISHMENT DECISION -> AUDITABLE OUTPUT")
    A("```")
    A("Forecasting (src/phase6_run.py) and the replenishment decision engine "
      "(src/replenishment.py) are separate modules; the decision engine never "
      "fits or calls the forecasting model.\n")

    A("## Production Model\n")
    A(f"- Architecture: `GlobalXGBoostForecaster` (Phase 5, unmodified) — reg:tweedie global XGBoost.")
    A(f"- Reused existing persisted model: **{meta.get('reused_existing_model')}**")
    A(f"- Training cutoff (explicit): **{meta.get('training_cutoff')}**")
    A(f"- Training rows: {meta.get('n_train_rows')}")
    A(f"- Series trained on: {meta.get('n_series')}")
    A(f"- Feature columns: {len(meta.get('feature_cols', []))}")
    A("- XGBoost parameters:")
    A("```json")
    A(json.dumps(meta.get("xgboost_params", {}), indent=2))
    A("```\n")

    A("## Forecast Output\n")
    A(f"- Forecast rows: {len(forecast_df):,}")
    A(f"- Series forecast: {forecast_df[KEY_COLS].drop_duplicates().shape[0]:,}")
    if not forecast_df.empty:
        A(f"- Forecast window: {forecast_df['forecast_date'].min().date()} -> "
          f"{forecast_df['forecast_date'].max().date()}")
    A("")

    A("## Uncertainty / Risk Methodology\n")
    A("Per-segment RMSE from the Phase 5 XGBoost backtest "
      "(`reports/phase5/segment_model_comparison.csv`) is used as a demand-error "
      "sigma proxy for safety stock. **This is a heuristic risk proxy, not a "
      "calibrated prediction interval** — it is a segment-level historical average, "
      "never computed from the actuals of the decision being scored. See the "
      "module docstring in `src/replenishment.py` for the full methodology and "
      "its limitations.\n")
    if risk_summary is not None and not risk_summary.empty:
        risk_counts = risk_summary["risk_flag"].value_counts().to_dict()
        A(f"Risk flag distribution: {risk_counts}\n")

    A("## Replenishment Assumptions\n")
    rep = cfg["replenishment"]
    A("| Parameter | Value | Source |")
    A("|---|---|---|")
    A(f"| lead_time_days | {rep['lead_time_days']} | config.yaml `replenishment` (simulation assumption, D-004) |")
    A(f"| service_level | {rep['service_level']} | config.yaml `replenishment` (simulation assumption, D-004) |")
    A(f"| review_period_days | {rep['review_period_days']} | config.yaml `replenishment` (simulation assumption, D-004) |")
    A(f"| initial_inventory_days_of_cover | {rep['initial_inventory_days_of_cover']} | config.yaml `replenishment` (simulation assumption, D-004) |")
    A("\n**M5 contains no real on-hand inventory, open purchase order, or lead-time "
      "data.** All inventory positions in `--mode replenishment` are either "
      "user-supplied (`--inventory-csv`) or SIMULATED from `initial_inventory_days_of_cover`. "
      "Results in replenishment mode are scenario/simulation outputs, not measured "
      "historical business outcomes — no stockout reduction, service-level "
      "improvement, or cost saving is claimed.\n")

    if recommendations is not None:
        A("## Replenishment Recommendations Summary\n")
        A(f"- Series evaluated: {len(recommendations):,}")
        if not recommendations.empty:
            reason_counts = recommendations["decision_reason"].value_counts().to_dict()
            A(f"- Decision reasons: {reason_counts}")
            n_orders = int((recommendations["recommended_order_qty"].fillna(0) > 0).sum())
            A(f"- Series with a recommended order > 0: {n_orders:,}")
        A("")

    A("## Known Limitations\n")
    A("1. **Demand is censored.** Observed sales, not true demand (D-003).")
    A("2. **Uncertainty is a proxy, not calibrated.** Segment-level historical RMSE, not a per-series prediction interval.")
    A("3. **Inventory is simulated** unless real positions are supplied via `--inventory-csv`.")
    A("4. **No cost, stockout, or service-level outcome is measured** — only simulated/scenario recommendations.")
    A("5. **One global model.** Same XGBoost architecture as Phase 5, not retuned for production.")
    A("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 production forecasting + replenishment")
    parser.add_argument("--mode", choices=["forecast-only", "replenishment"], default="replenishment")
    parser.add_argument("--fixture", action="store_true", help="Run on fixture data")
    parser.add_argument("--subset", type=int, default=None, help="Run on N series")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore any persisted model and retrain")
    parser.add_argument("--inventory-csv", type=str, default=None,
                         help="Optional CSV with store_id,item_id,inventory_position (real data). "
                              "If omitted, inventory is SIMULATED from config.")
    args = parser.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()

    long_df, calendar_path = _load_long_df(cfg, args.fixture, args.subset)
    seg_df = _load_segmentation(cfg, args.fixture)

    forecast_df, meta, cutoff = generate_production_forecast(
        long_df, cfg, calendar_path=calendar_path, force_retrain=args.force_retrain,
    )

    if seg_df is not None:
        forecast_df = forecast_df.merge(seg_df, on=KEY_COLS, how="left")

    hist_mean_df = _historical_mean_sales(long_df, cutoff)
    risk_map = load_segment_risk_proxy(cfg)

    out_dir = cfg.path("reports") / ("phase6_fixture" if args.fixture else "phase6")
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_df.to_csv(out_dir / "forecast_summary.csv", index=False, encoding="utf-8")

    risk_summary = build_forecast_risk_summary(
        forecast_df, cfg, risk_map=risk_map, historical_mean_df=hist_mean_df,
    )
    risk_summary.to_csv(out_dir / "risk_summary.csv", index=False, encoding="utf-8")

    recommendations = None
    if args.mode == "replenishment":
        inventory_df = None
        if args.inventory_csv:
            inventory_df = pd.read_csv(args.inventory_csv)
            logger.info("Loaded user-supplied inventory positions for %d series.", len(inventory_df))
        else:
            logger.warning(
                "No --inventory-csv supplied: inventory positions are SIMULATED "
                "from replenishment.initial_inventory_days_of_cover. This is a "
                "scenario output, not a measured business outcome."
            )
        recommendations = build_replenishment_recommendations(
            forecast_df, cfg, inventory_df=inventory_df, risk_map=risk_map,
            historical_mean_df=hist_mean_df,
        )
        recommendations.to_csv(out_dir / "replenishment_recommendations.csv", index=False, encoding="utf-8")

    write_phase6_report(
        out_dir / "phase6_forecasting.md", cfg, meta, cutoff, args.mode,
        forecast_df, risk_summary, recommendations,
    )

    print("\n=== Phase 6 Complete ===")
    print(f"Mode: {args.mode}")
    print(f"Training cutoff: {meta.get('training_cutoff')}")
    print(f"Reused persisted model: {meta.get('reused_existing_model')}")
    print(f"Forecast rows: {len(forecast_df):,}  Series: {forecast_df[KEY_COLS].drop_duplicates().shape[0]:,}")
    if recommendations is not None:
        print(f"Recommendations: {len(recommendations):,}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
